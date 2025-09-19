from torch import nn
import torch

from transformers.models.olmoe.modeling_olmoe import (
    OlmoeMLP, OlmoeDecoderLayer, OlmoeModel,OlmoeForCausalLM,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP, Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM,
)

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeMLP, Qwen3MoeDecoderLayer, Qwen3MoeModel, Qwen3MoeForCausalLM
)
# TODO FIX MASKING
# from transformers.modeling_attn_mask_utils import _make_causal_mask, _expand_mask
# from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
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
import torch
torch.autograd.set_detect_anomaly(True)
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
 
class Qwen3MoeBlockWithRIM(nn.Module):
    """
    MoE block with efficient RIM (Recurrent Inference Machine) attention mechanism.
    """
    def __init__(self, config: Qwen3WithRIMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_attn_size = config.expert_attn_size
        self.top_p = config.experts_top_p
        self.top_k = config.experts_top_k
        self.use_dynamic_routing = config.use_dynamic_routing

        self.detach_null_states = config.detach_null_states
        self.use_latent_states = config.use_latent_states
        
        self.experts = nn.ModuleList([Qwen3MLP(config) for _ in range(self.num_experts)])
        self.key = nn.Linear(config.hidden_size, 
                             self.num_experts*config.expert_attn_size, 
                             bias=False)    
        self.value = nn.Linear(config.hidden_size, 
                             self.num_experts*config.expert_attn_size, 
                             bias=False)
        if not self.use_latent_states:
            self.expert_query = nn.Linear(config.hidden_size, 
                                          self.num_experts*config.expert_attn_size, bias=False)
        else:
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
        if self.use_latent_states:
            experts_flat_states = self.expert_states_flat(hidden_states)
            experts_flat_query = self.expert_query(experts_flat_states)
        else:
            experts_flat_query = self.expert_query(hidden_states) 
        
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
        

        attention_to_real_flat = all_attn_weights[:, :, :self.expert_attn_size].sum(dim=-1).to(hidden_states.dtype)
        attention_to_null_flat = all_attn_weights[:, :, self.expert_attn_size:].sum(dim=-1).to(hidden_states.dtype)
        
        # Detach null states to avoid gradients optimization based null states
        if self.detach_null_states:
            attention_to_null_flat = attention_to_null_flat.detach()         

        # Top-p selection using top-k on all experts
        # Attn diff is used to regulate each experts attn to real and null part
        attn_diff_normalized = nn.functional.softmax(attention_to_real_flat - attention_to_null_flat,dim=-1)
        sorted_weights, sorted_indices = torch.topk(attn_diff_normalized, k=self.num_experts, dim=-1,sorted=True)  # Sort the attention weights in descending order
        
        if self.use_dynamic_routing:
            top_p_mask = torch.cumsum(sorted_weights, dim=-1) <= self.top_p
            top_p_mask[..., 0] = True   # always activate the expert with the highest attention weight
            experts_mask = torch.zeros_like(attn_diff_normalized, dtype=torch.bool)
            experts_mask.scatter_(dim=1, index=sorted_indices, src=top_p_mask) 
        else:
            k = min(self.top_k, self.num_experts)  
            top_k_mask = torch.zeros_like(sorted_weights, dtype=torch.bool)
            top_k_mask[..., :k] = True  
            experts_mask = torch.zeros_like(attn_diff_normalized, dtype=torch.bool)
            experts_mask.scatter_(dim=1, index=sorted_indices, src=top_k_mask)
            
        hidden_states = hidden_states[:batch_size*sequence_length,:]
        
        # DO this to avoid in-place operations
        output = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        for expert_idx in range(self.num_experts):
            token_idx = torch.where(experts_mask[:, expert_idx])[0]
            selected_tokens = hidden_states[token_idx]
            update = self.experts[expert_idx](selected_tokens) * attn_diff_normalized[token_idx, expert_idx].unsqueeze(-1)
            update = update.to(output.dtype)  # Ensure the update is in the same dtype as output
            output[token_idx] = output[token_idx] + update
        return output.contiguous().view(batch_size, sequence_length, -1), attn_diff_normalized, experts_mask

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
        hidden_states, router_logits, expert_mask = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

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
    config_class = Qwen3WithRIMConfig
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
        output_expert_mask: Optional[bool] = True,
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

        
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

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

        # if not isinstance(causal_mask_mapping := attention_mask, dict):
        #     mask_kwargs = {
        #         "config": self.config,
        #         "input_embeds": inputs_embeds,
        #         "attention_mask": attention_mask,
        #         "cache_position": cache_position,
        #         "past_key_values": past_key_values,
        #         "position_ids": position_ids,
        #     }
        #     causal_mask_mapping = {
        #         "full_attention": create_causal_mask(**mask_kwargs),
        #     }
        #     if self.has_sliding_layers:
        #         causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

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
                attention_mask=causal_mask,
                # attention_mask=causal_mask_mapping[decoder_layer.attention_type],
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
    config_class = Qwen3WithRIMConfig
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
        if output_router_logits and labels is not None:
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

