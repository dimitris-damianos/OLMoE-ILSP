from torch import nn
import torch

from transformers.models.olmoe.modeling_olmoe import (
    OlmoeMLP, OlmoeDecoderLayer, OlmoeModel,OlmoeForCausalLM,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP, Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM,   
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import LossKwargs
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from typing import List, Optional, Tuple, Union

from config import OlmoeWithRIMConfig, Qwen3WithRIMConfig, Qwen2WithRIMConfig
from outputs import (
    MoeModelOutputWithPastAndExpertMask, MoeCausalLMOutputWithPastAndExpertMask,
    BaseModelOutputWithPastandMoe, CausalLMOutputWithPastandMoe,
)

from transformers.utils import logging
logger = logging.get_logger(__name__)

# Based on transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
# Changed to fit our test case
def load_balancing_loss_for_rim(logit_weights = None,
                             expert_mask = None,
                             attention_mask = None,
                             num_experts: int = 8,
                             ):
    num_layers = len(logit_weights)
    if logit_weights is None or not isinstance(logit_weights, tuple):
        return 0
    if isinstance(logit_weights, tuple) and isinstance(expert_mask, tuple):
        # concatenate logits from all layers
        compute_device = logit_weights[0].device
        concatenated_logits = torch.cat([layer_logits.to(compute_device) for layer_logits in logit_weights], dim=0)
        concatenated_expert_mask = torch.cat([layer_expert_mask.to(compute_device) for layer_expert_mask in expert_mask], dim=0)
    
    aux_loss = 0
    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(concatenated_expert_mask.float(), dim=0)
        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(concatenated_logits, dim=0)

    else: 
        # Flatten the attention mask to match the shape of the logits 
        attention_mask = attention_mask.reshape(-1, 1)
        # Expand to all layers end expert
        attention_mask = attention_mask.repeat(num_layers, 1)  # (num_layers, batch_size*seq_len)
        attention_mask = attention_mask.expand(-1, num_experts)  # (batch_size*seq_len*num_layers, num_experts)
        
        # Mask logits and expert mask
        masked_logits = concatenated_logits * attention_mask.float()
        masked_expert_mask = concatenated_expert_mask * attention_mask.float()
        
        tokens_per_expert = torch.mean(masked_expert_mask.float(), dim=0)   # Compute the percentage of tokens routed to each expert
        router_prob_per_expert = torch.mean(masked_logits, dim=0)           # Compute the average probability of routing to these experts
        
    aux_loss = torch.sum(tokens_per_expert * router_prob_per_expert)
    return aux_loss

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

class OlmoeMoeBlockWithRIM_(nn.Module):
    """
    DEPRECATED
    MoE block with RIM (Recurrent Inference Machine) attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])
        self.expert_states = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.num_experts)])
        
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.expert_querries = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.num_experts)])  # TODO check dimensions
        
        # TODO: add communication attention
        # print(f'Number of experts: {self.num_experts}, Top-k: {self.top_k}, Norm top-k prob: {self.norm_topk_prob}')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        null_values = torch.zeros(
            (batch_size, sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        
        # null values are located the second half of the sequence (on dim 1) 
        hidden_states = torch.cat([hidden_states, null_values], dim=1)  # (batch, 2*sequence_length, hidden_dim)
        print(hidden_states.shape)
        # keys, values shared between experts 
        keys = self.key(hidden_states)  # (batch, sequence_length, hidden_dim)
        values = self.value(hidden_states)  # (batch, sequence_length, hidden_dim)
        
        expert_mask = []
        expert_weights = []
        
        for expert_idx in range(self.num_experts):
            # Compute the expert's MLP state and query
            expert_state = self.expert_states[expert_idx](hidden_states)    # (batch, sequence_length, hidden_dim)
            expert_querry = self.expert_querries[expert_idx](expert_state)  #
            
            # Compute attention scores for each expert
            attention_scores = nn.functional.softmax(torch.bmm(expert_querry, keys.mT)/torch.sqrt(torch.tensor(hidden_dim)), dim=1)
            attention_weights = torch.bmm(attention_scores, values)     # (batch, sequence_length, hidden_dim)
            
            # Separate attention weights for real tokens and null tokens
            attention_to_real = attention_weights[:, :sequence_length, :].sum(dim=-1)  # (batch, sequence_length)
            attention_to_null = attention_weights[:, sequence_length:, :].sum(dim=-1)  # (batch, sequence_length)
            
            # Decide which tokens the expert prefers based on attention weights
            # If attention to real tokens is greater than to null tokens, the expert prefers the real tokens
            # Otherwise, it prefers the null tokens
            # Normalize the attention weights
            attn_diff = attention_to_real - attention_to_null
            print(attn_diff)
            token_perference = attn_diff > 0
            
            expert_mask.append(token_perference)
            expert_weights.append(attention_to_real)        # Normalize the attention weights      
            
        expert_mask = torch.stack(expert_mask, dim=0)     # (num_experts, batch, sequence_length
        expert_weights = torch.stack((expert_weights), dim=0)             # (num_experts, batch, sequence_length)
        expert_weights = torch.nn.functional.softmax(expert_weights, dim=0)  # Normalize the attention weights across experts
        
        # re-use null values to store the final hidden states
        for expert_idx in range(self.num_experts):
            # Get pairs of batch and token indices where the expert is selected
            batch_idx, token_idx = torch.where(expert_mask[expert_idx])
            selected_tokens = hidden_states[batch_idx, token_idx, :]  # (num_selected_tokens, hidden_dim)
            
            # the resulting hidden states are the weighted sum of the selected expert's output 
            null_values[batch_idx, token_idx, :] += self.experts[expert_idx](selected_tokens) * expert_weights[expert_idx, batch_idx, token_idx].unsqueeze(-1)
            
        return null_values, expert_weights.view(batch_size*sequence_length, self.num_experts), expert_mask.view(-1,expert_mask.shape[1]*expert_mask.shape[2]).T  

class OlmoeMoeBlockWithRIM(nn.Module):
    """
    MoE block with efficient RIM (Recurrent Inference Machine) attention mechanism.
    """
    def __init__(self, config: OlmoeWithRIMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.expert_attn_size = config.expert_attn_size
        self.top_p = config.experts_top_p
        
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])
        self.key = nn.Linear(config.hidden_size, 
                             self.num_experts*config.expert_attn_size, 
                             bias=False)    
        self.value = nn.Linear(config.hidden_size, 
                             self.num_experts*config.expert_attn_size, 
                             bias=False)
        self.expert_query = nn.Linear(self.num_experts*config.expert_attn_size, 
                                      self.num_experts*config.expert_attn_size, bias=False)  # TODO check dimensions
        self.expert_states_flat = nn.Linear(config.hidden_size, 
                                            self.num_experts*config.expert_attn_size, 
                                            bias=False)
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim) 
        null_hidden_states = torch.zeros(
            (batch_size*sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        nn.init.xavier_uniform_(null_hidden_states) 
        # Following original RIM, we concatenate the null hidden states to batch size 
        hidden_states = torch.cat([hidden_states, null_hidden_states], dim=0) 

        # keys, values shared between experts 
        keys = self.key(hidden_states) 
        values = self.value(hidden_states)  
        
        experts_flat_states = self.expert_states_flat(hidden_states)    
        experts_flat_query = self.expert_query(experts_flat_states)  
        experts_flat_query = experts_flat_query.view(2*batch_size*sequence_length,
                                                     self.num_experts,
                                                     self.expert_attn_size)
        q_x_k =  torch.bmm(experts_flat_query, keys.view(2*batch_size*sequence_length,
                                                         self.expert_attn_size,
                                                         self.num_experts))/torch.sqrt(torch.tensor(self.expert_attn_size))
        attention_scores_flat = nn.functional.softmax(
           q_x_k, dim=1
        )  
        values = values.view(2*batch_size*sequence_length, 
                             self.num_experts, 
                             self.expert_attn_size)  
        all_attn_weights = torch.matmul(attention_scores_flat, 
                                        values)

        # Separate attention weights for real tokens and null tokens to check which expert prefers which tokens  
        all_attn_weights = all_attn_weights.view(batch_size*sequence_length, 
                                                 self.num_experts, 
                                                 2*self.expert_attn_size)  # (batch*sequence_length, num_experts, 2*expert_attn_size)
        attention_to_real_flat = all_attn_weights[:, :, :self.expert_attn_size].sum(dim=-1)  
        attention_to_null_flat = all_attn_weights[:, :, self.expert_attn_size:].sum(dim=-1)  
       
            
        # Top-p selection using top-k on all experts
        # Attn diff is used to regulate each experts attn to real and null part
        attn_diff_normalized = nn.functional.softmax(attention_to_real_flat - attention_to_null_flat,dim=-1)
        sorted_weights, sorted_indices = torch.topk(attn_diff_normalized, k=self.num_experts, dim=-1,sorted=True)  # Sort the attention weights in descending order
        top_p_mask = torch.cumsum(sorted_weights, dim=-1) <= self.top_p
        top_p_mask[..., 0] = True   # always activate the expert with the highest attention weight

        experts_mask = torch.zeros_like(attn_diff_normalized, dtype=torch.bool)
        experts_mask.scatter_(dim=1, index=sorted_indices, src=top_p_mask) 
        
        # Normalize the attention weights
        hidden_states = hidden_states[:batch_size*sequence_length,:]
        for expert_idx in range(self.num_experts):
            token_idx = torch.where(experts_mask[:, expert_idx])[0]        # Get token indices (in tuple) where the expert is selected in the batch*sequence_length range
            selected_tokens = hidden_states[token_idx]
            # the resulting hidden states are the weighted sum of the selected expert's output 
            hidden_states[token_idx] += self.experts[expert_idx](selected_tokens) * attn_diff_normalized[token_idx, expert_idx].unsqueeze(-1)
            
        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)  # Reshape back to (batch, 2*sequence_length, hidden_dim)
        
        return hidden_states, attn_diff_normalized, experts_mask

class OlmoeDecoderLayerWithRIM(OlmoeDecoderLayer):
    def __init__(self, config,layer_idx=None):
        super().__init__(config,layer_idx=layer_idx)
        self.mlp = OlmoeMoeBlockWithRIM(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        output_expert_mask: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits, expert_mask = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)
            
        if output_expert_mask:
            outputs += (expert_mask,)

        return outputs 
        
class OlmoeModelWithRIM(OlmoeModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [OlmoeDecoderLayerWithRIM(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_expert_mask: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPastAndExpertMask]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_expert_mask = (
            output_expert_mask if output_expert_mask is not None else self.config.output_expert_mask
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_expert_masks = () if output_expert_mask else None
        next_decoder_cache = None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    output_expert_mask,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    output_expert_mask=output_expert_mask,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[2] is not None:
                all_router_logits += (layer_outputs[2],)
                
            if output_expert_mask and layer_outputs[3] is not None:
                all_expert_masks += (layer_outputs[3],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeModelOutputWithPastAndExpertMask(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            expert_mask=all_expert_masks,
        )
    
class OlmoeForCausalLMWithRIM(OlmoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = OlmoeModelWithRIM(config)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_expert_mask: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPastAndExpertMask]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_expert_mask = (
            self.config.output_expert_mask if output_expert_mask is None else output_expert_mask
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_expert_mask=output_expert_mask,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        aux_loss = None
        # TODO: Implement load balancing loss function
        if output_router_logits:
            aux_loss = load_balancing_loss_for_rim(
                outputs.router_logits if return_dict else outputs[2],
                outputs.expert_mask if return_dict else outputs[3],
                attention_mask = attention_mask,  # TODO: Implement attention mask for load balancing loss,
                num_experts=self.num_experts,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPastAndExpertMask(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            expert_mask=outputs.expert_mask
        )
             
class Qwen3MoeBlockWithRIM(nn.Module):
    """
    MoE block with efficient RIM (Recurrent Inference Machine) attention mechanism.
    """
    def __init__(self, config: Qwen3WithRIMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_attn_size = config.expert_attn_size
        self.top_p = config.experts_top_p
        
        self.experts = nn.ModuleList([Qwen3MLP(config) for _ in range(self.num_experts)])
        self.key = nn.Linear(config.hidden_size, 
                             self.num_experts*config.expert_attn_size, 
                             bias=False)    
        self.value = nn.Linear(config.hidden_size, 
                             self.num_experts*config.expert_attn_size, 
                             bias=False)
        self.expert_query = nn.Linear(self.num_experts*config.expert_attn_size, 
                                      self.num_experts*config.expert_attn_size, bias=False)  # TODO check dimensions
        self.expert_states_flat = nn.Linear(config.hidden_size, 
                                            self.num_experts*config.expert_attn_size, 
                                            bias=False)
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim) 
        null_hidden_states = torch.zeros(
            (batch_size*sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        nn.init.xavier_uniform_(null_hidden_states) 
        # Following original RIM, we concatenate the null hidden states to batch size 
        hidden_states = torch.cat([hidden_states, null_hidden_states], dim=0) 

        # keys, values shared between experts 
        keys = self.key(hidden_states) 
        values = self.value(hidden_states)  
        
        experts_flat_states = self.expert_states_flat(hidden_states)    
        experts_flat_query = self.expert_query(experts_flat_states)  
        experts_flat_query = experts_flat_query.view(2*batch_size*sequence_length,
                                                     self.num_experts,
                                                     self.expert_attn_size)
        q_x_k =  torch.bmm(experts_flat_query, keys.view(2*batch_size*sequence_length,
                                                         self.expert_attn_size,
                                                         self.num_experts))/torch.sqrt(torch.tensor(self.expert_attn_size))
        attention_scores_flat = nn.functional.softmax(
           q_x_k, dim=1
        )  
        values = values.view(2*batch_size*sequence_length, 
                             self.num_experts, 
                             self.expert_attn_size)  
        all_attn_weights = torch.matmul(attention_scores_flat, 
                                        values)

        # Separate attention weights for real tokens and null tokens to check which expert prefers which tokens  
        all_attn_weights = all_attn_weights.view(batch_size*sequence_length, 
                                                 self.num_experts, 
                                                 2*self.expert_attn_size)  # (batch*sequence_length, num_experts, 2*expert_attn_size)
        attention_to_real_flat = all_attn_weights[:, :, :self.expert_attn_size].sum(dim=-1)  
        attention_to_null_flat = all_attn_weights[:, :, self.expert_attn_size:].sum(dim=-1)  
       
            
        # Top-p selection using top-k on all experts
        # Attn diff is used to regulate each experts attn to real and null part
        attn_diff_normalized = nn.functional.softmax(attention_to_real_flat - attention_to_null_flat,dim=-1)
        sorted_weights, sorted_indices = torch.topk(attn_diff_normalized, k=self.num_experts, dim=-1,sorted=True)  # Sort the attention weights in descending order
        top_p_mask = torch.cumsum(sorted_weights, dim=-1) <= self.top_p
        top_p_mask[..., 0] = True   # always activate the expert with the highest attention weight

        experts_mask = torch.zeros_like(attn_diff_normalized, dtype=torch.bool)
        experts_mask.scatter_(dim=1, index=sorted_indices, src=top_p_mask) 
        
        # Normalize the attention weights
        hidden_states = hidden_states[:batch_size*sequence_length,:]
        
        # Add process states to residual
        for expert_idx in range(self.num_experts):
            token_idx = torch.where(experts_mask[:, expert_idx])[0]        # Get token indices (in tuple) where the expert is selected in the batch*sequence_length range
            selected_tokens = hidden_states[token_idx]
            # the resulting hidden states are the weighted sum of the selected expert's output 
            hidden_states[token_idx] += self.experts[expert_idx](selected_tokens) * attn_diff_normalized[token_idx, expert_idx].unsqueeze(-1)
            
        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)  # Reshape back to (batch, 2*sequence_length, hidden_dim)
        
        return hidden_states, attn_diff_normalized, experts_mask  

class Qwen3DecoderLayerWithRIM(Qwen3DecoderLayer):
    def __init__(self, config,layer_idx=None):
        super().__init__(config,layer_idx=layer_idx)
        self.mlp = Qwen3MoeBlockWithRIM(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        output_expert_mask: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Pass through the MoE block
        # MoE block returns residual + hidden_states
        hidden_states, router_logits, expert_mask = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        else: 
            outputs += (None,)
        if output_router_logits:
            outputs += (router_logits,)
        else:
            outputs += (None,)
        if output_expert_mask:
            outputs += (expert_mask,)
        else:
            outputs += (None,)
        return outputs
    
class Qwen3ModelWithRIM(Qwen3Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayerWithRIM(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_expert_mask: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPastandMoe:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_expert_mask = (
            output_expert_mask if output_expert_mask is not None else self.config.output_expert_mask
        )
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # FIXME: this works for Olmoe but not for Qwen
        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )

        # NOTE: alt approach
        # mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        # causal_mask = mask_function(
        #     config=self.config,
        #     input_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     cache_position=cache_position,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_expert_masks = () if output_expert_mask else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                # attention_mask=causal_mask,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                output_expert_mask=output_expert_mask,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_router_logits and layer_outputs[2] is not None:
                all_router_logits += (layer_outputs[2],)
            if output_expert_mask and layer_outputs[3] is not None:
                all_expert_masks += (layer_outputs[3],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPastandMoe(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            expert_mask=all_expert_masks,
        )
    
class Qwen3ForCausalLMWithRIM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts 
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.model = Qwen3ModelWithRIM(config)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_expert_mask: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPastandMoe:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_expert_mask = (
            self.config.output_expert_mask if output_expert_mask is None else output_expert_mask
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPastandMoe = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_expert_mask=output_expert_mask,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_for_rim(
                outputs.router_logits,     # logits
                outputs.expert_mask,     # expert mask
                attention_mask = attention_mask,  # TODO: Implement attention mask for load balancing loss,
                num_experts=self.num_experts,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return CausalLMOutputWithPastandMoe(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            expert_mask=outputs.expert_mask,
            aux_loss=aux_loss
        )


class Qwen2MoeBlockWithRIM(nn.Module):
    """
    MoE block with efficient RIM (Recurrent Inference Machine) attention mechanism.
    """
    def __init__(self, config: Qwen2WithRIMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_attn_size = config.expert_attn_size
        self.top_p = config.experts_top_p
        
        self.experts = nn.ModuleList([Qwen2MLP(config) for _ in range(self.num_experts)])
        self.key = nn.Linear(config.hidden_size, 
                             self.num_experts*config.expert_attn_size, 
                             bias=False)    
        self.value = nn.Linear(config.hidden_size, 
                             self.num_experts*config.expert_attn_size, 
                             bias=False)
        self.expert_query = nn.Linear(self.num_experts*config.expert_attn_size, 
                                      self.num_experts*config.expert_attn_size, bias=False)  # TODO check dimensions
        self.expert_states_flat = nn.Linear(config.hidden_size, 
                                            self.num_experts*config.expert_attn_size, 
                                            bias=False)
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim) 
        null_hidden_states = torch.zeros(
            (batch_size*sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        nn.init.xavier_uniform_(null_hidden_states) 
        # Following original RIM, we concatenate the null hidden states to batch size 
        hidden_states = torch.cat([hidden_states, null_hidden_states], dim=0) 

        # keys, values shared between experts 
        keys = self.key(hidden_states) 
        values = self.value(hidden_states)  
        
        experts_flat_states = self.expert_states_flat(hidden_states)    
        experts_flat_query = self.expert_query(experts_flat_states)  
        experts_flat_query = experts_flat_query.view(2*batch_size*sequence_length,
                                                     self.num_experts,
                                                     self.expert_attn_size)
        q_x_k =  torch.bmm(experts_flat_query, keys.view(2*batch_size*sequence_length,
                                                         self.expert_attn_size,
                                                         self.num_experts))/torch.sqrt(torch.tensor(self.expert_attn_size))
        attention_scores_flat = nn.functional.softmax(
           q_x_k, dim=1
        )  
        values = values.view(2*batch_size*sequence_length, 
                             self.num_experts, 
                             self.expert_attn_size)  
        all_attn_weights = torch.matmul(attention_scores_flat, 
                                        values)

        # Separate attention weights for real tokens and null tokens to check which expert prefers which tokens  
        all_attn_weights = all_attn_weights.view(batch_size*sequence_length, 
                                                 self.num_experts, 
                                                 2*self.expert_attn_size)  # (batch*sequence_length, num_experts, 2*expert_attn_size)
        attention_to_real_flat = all_attn_weights[:, :, :self.expert_attn_size].sum(dim=-1)  
        attention_to_null_flat = all_attn_weights[:, :, self.expert_attn_size:].sum(dim=-1)  
       
            
        # Top-p selection using top-k on all experts
        # Attn diff is used to regulate each experts attn to real and null part
        attn_diff_normalized = nn.functional.softmax(attention_to_real_flat - attention_to_null_flat,dim=-1)
        sorted_weights, sorted_indices = torch.topk(attn_diff_normalized, k=self.num_experts, dim=-1,sorted=True)  # Sort the attention weights in descending order
        top_p_mask = torch.cumsum(sorted_weights, dim=-1) <= self.top_p
        top_p_mask[..., 0] = True   # always activate the expert with the highest attention weight

        experts_mask = torch.zeros_like(attn_diff_normalized, dtype=torch.bool)
        experts_mask.scatter_(dim=1, index=sorted_indices, src=top_p_mask) 
        
        # Normalize the attention weights
        hidden_states = hidden_states[:batch_size*sequence_length,:]
        
        # Add process states to residual
        for expert_idx in range(self.num_experts):
            token_idx = torch.where(experts_mask[:, expert_idx])[0]        # Get token indices (in tuple) where the expert is selected in the batch*sequence_length range
            selected_tokens = hidden_states[token_idx]
            # the resulting hidden states are the weighted sum of the selected expert's output 
            hidden_states[token_idx] += self.experts[expert_idx](selected_tokens) * attn_diff_normalized[token_idx, expert_idx].unsqueeze(-1)
            
        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)  # Reshape back to (batch, 2*sequence_length, hidden_dim)
        
        return hidden_states, attn_diff_normalized, experts_mask  

class Qwen2DecoderLayerWithRIM(Qwen2DecoderLayer):
    def __init__(self, config,layer_idx=None):
        super().__init__(config,layer_idx=layer_idx)
        self.mlp = Qwen2MoeBlockWithRIM(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        output_expert_mask: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Pass through the MoE block
        # MoE block returns residual + hidden_states
        hidden_states, router_logits, expert_mask = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        else: 
            outputs += (None,)
        if output_router_logits:
            outputs += (router_logits,)
        else:
            outputs += (None,)
        if output_expert_mask:
            outputs += (expert_mask,)
        else:
            outputs += (None,)
        return outputs
    
class Qwen2ModelWithRIM(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerWithRIM(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_expert_mask: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPastandMoe:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_expert_mask = (
            output_expert_mask if output_expert_mask is not None else self.config.output_expert_mask
        )
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # FIXME: this works for Olmoe but not for Qwen
        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_expert_masks = () if output_expert_mask else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                # attention_mask=causal_mask,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                output_expert_mask=output_expert_mask,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_router_logits and layer_outputs[2] is not None:
                all_router_logits += (layer_outputs[2],)
            if output_expert_mask and layer_outputs[3] is not None:
                all_expert_masks += (layer_outputs[3],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPastandMoe(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            expert_mask=all_expert_masks,
        )
        
class Qwen2ForCausalLMWithRIM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts 
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.model = Qwen2ModelWithRIM(config)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_expert_mask: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPastandMoe:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_expert_mask = (
            self.config.output_expert_mask if output_expert_mask is None else output_expert_mask
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPastandMoe = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_expert_mask=output_expert_mask,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_for_rim(
                outputs.router_logits,     # logits
                outputs.expert_mask,     # expert mask
                attention_mask = attention_mask,  # TODO: Implement attention mask for load balancing loss,
                num_experts=self.num_experts,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return CausalLMOutputWithPastandMoe(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            expert_mask=outputs.expert_mask,
            aux_loss=aux_loss
        )
