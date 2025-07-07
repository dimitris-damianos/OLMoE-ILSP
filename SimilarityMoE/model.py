from torch import nn
import torch.nn.functional as F
import torch

from transformers.models.olmoe.modeling_olmoe import (
    OlmoeMLP,
    OlmoeDecoderLayer,
    OlmoeModel,
    OlmoeForCausalLM,
)

class OlmoeMoeBlockWithRIM_(nn.Module):
    """
    MoE block with RIM (Recurrent Inference Machine) attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
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
        hidden_states = torch.cat([hidden_states, null_values], dim=1)  # (batch, sequence_length, hidden_dim)
        
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
            attention_scores = nn.functional.softmax(torch.bmm(expert_querry, keys.mT)/torch.sqrt(torch.tensor(hidden_dim)), dim=-1)
            attention_weights = torch.bmm(attention_scores, values)     # (batch, sequence_length, hidden_dim)
            
            # Separate attention weights for real tokens and null tokens
            attention_to_real = attention_weights[:, :sequence_length, :].sum(dim=-1)  # (batch, sequence_length)
            attention_to_null = attention_weights[:, sequence_length:, :].sum(dim=-1)  # (batch, sequence_length)
            
            # Decide which tokens the expert prefers based on attention weights
            # If attention to real tokens is greater than to null tokens, the expert prefers the real tokens
            # Otherwise, it prefers the null tokens
            # Normalize the attention weights
            # weights_sum = attention_to_real + attention_to_null
            attn_diff = attention_to_real - attention_to_null
            token_perference = attn_diff > 0
            
            expert_mask.append(token_perference)
            expert_weights.append(attention_to_real/(attention_to_real + attention_to_null))        # Normalize the attention weights      
            
        expert_mask = torch.stack(expert_mask, dim=0)     # (num_experts, batch, sequence_length
        expert_weights = torch.stack(expert_weights, dim=0)             # (num_experts, batch, sequence_length)
        
        # re-use null values to store the final hidden states
        for expert_idx in range(self.num_experts):
            # Get pairs of batch and token indices where the expert is selected
            batch_idx, token_idx = torch.where(expert_mask[expert_idx])
            selected_tokens = hidden_states[batch_idx, token_idx, :]  # (num_selected_tokens, hidden_dim)
            
            # the resulting hidden states are the weighted sum of the selected expert's output 
            null_values[batch_idx, token_idx, :] += self.experts[expert_idx](selected_tokens) * expert_weights[expert_idx, batch_idx, token_idx].unsqueeze(-1)
            
        return null_values, expert_weights.view(batch_size*sequence_length, self.num_experts)  # Reshape to (batch*sequence_length, num_experts)

class OlmoeMoeBlockWithRIM(nn.Module):
    """
    MoE block with efficient RIM (Recurrent Inference Machine) attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])
        self.expert_states = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.num_experts)])
        
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.expert_querries = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.num_experts)])  # TODO check dimensions
        
        
        self.expert_query = nn.Linear(self.num_experts*config.hidden_size, 
                                      self.num_experts*config.hidden_size, bias=False)  # TODO check dimensions
        self.expert_states_flat = nn.Linear(config.hidden_size, 
                                            self.num_experts*config.hidden_size, bias=False)
        
        # TODO: add communication attention
        self.enable_comm = config.enable_comm 
        # if self.enable_comm:
        #     # Each expert has its own communication attention
        #     # keys, values, queries for communication attention
        #     self.comm_key = nn.Linear(self.num_experts*config.hidden_size, self.num_experts*config.hidden_size, bias=False)
        #     self.comm_value = nn.Linear(self.num_experts*config.hidden_size, self.num_experts*config.hidden_size, bias=False)
        #     self.comm_query = nn.Linear(self.num_experts*config.hidden_size, self.num_experts*config.hidden_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim) # (batch * sequence_length, hidden_dim)
        null_values = torch.zeros(
            (batch_size*sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        
        # Following original RIM, we concatenate the null values to the hidden states
        # null values are located the second half of the batch*sequence dim (dim 0) 
        hidden_states = torch.cat([hidden_states, null_values], dim=0)  # (batch*2*sequence_length, hidden_dim)
    
        # keys, values shared between experts 
        keys = self.key(hidden_states)  # (batch*2*sequence_length, hidden_dim)
        values = self.value(hidden_states)  # (batch*2*sequence_length, hidden_dim)
        
        # Compute the attention scores for each expert
        experts_flat_states = self.expert_states_flat(hidden_states)  # (batch *2* sequence_length, num_experts * hidden_dim)
        # if self.enable_comm:
        #     # Compute communication attention
        #     comm_keys = self.comm_key(experts_flat_states)
        #     comm_values = self.comm_value(experts_flat_states)
        #     comm_query = self.comm_query(experts_flat_states)
        #     comm_attn_scores = nn.functional.softmax(
        #         torch.matmul(comm_query, comm_keys.T)/torch.sqrt(torch.tensor(hidden_dim)), dim=-1
        #     )   # (batch*2*sequence_length, num_experts, batch*2*sequence_length)
        #     comm_attn_weights = torch.matmul(comm_attn_scores, comm_values)  # (batch * sequence_length, num_experts, hidden_dim)
        #     experts_flat_states += comm_attn_weights.view(batch_size*2*sequence_length, self.num_experts*hidden_dim)
        
        experts_flat_query = self.expert_query(experts_flat_states)  # (batch *2* sequence_length, num_experts * hidden_dim)
    
        attention_scores_flat = nn.functional.softmax(
            torch.matmul(experts_flat_query.view(batch_size*2*sequence_length,self.num_experts,hidden_dim), keys.T)/torch.sqrt(torch.tensor(hidden_dim)), dim=-1
        )   # (batch*2*sequence_length, num_experts, batch*2*sequence_length)
        
        attention_weights_flat = torch.matmul(attention_scores_flat, values)  # (batch * sequence_length, num_experts, hidden_dim)
        
        # Separate attention weights for real tokens and null tokens
        # reshape attention weights to (batch*sequence_length, num_experts, hidden_dim) to sum over hidden_dim
        attention_to_real_flat = attention_weights_flat[:batch_size*sequence_length, :].view(batch_size*sequence_length,
                                                                                             self.num_experts,
                                                                                             hidden_dim).sum(dim=-1)  # (batch*sequence_length,num_experts)
        
        attention_to_null_flat = attention_weights_flat[batch_size*sequence_length:, :].view(batch_size*sequence_length,
                                                                                             self.num_experts,
                                                                                             hidden_dim).sum(dim=-1)  # (batch*sequence_length,num_experts)
        
        # Decide which tokens the expert prefers based on attention weights
        # If attention to real tokens is greater than to null tokens, the expert prefers the real tokens
        # Otherwise, it prefers the null tokens
        # Normalize the attention weights
        experts_mask = ((attention_to_real_flat - attention_to_null_flat) > 0)    # (batch*sequence_length, num_experts)
        attention_to_real_flat = attention_to_real_flat / (attention_to_real_flat + attention_to_null_flat) # (batch*sequence_length, num_experts)
        
        for expert_idx in range(self.num_experts):
            token_idx = torch.where(experts_mask[:, expert_idx])[0]        # Get token indices (in tuple) where the expert is selected in the batch*sequence_length range
            selected_tokens = hidden_states[token_idx]
            # the resulting hidden states are the weighted sum of the selected expert's output 
            null_values[token_idx] += self.experts[expert_idx](selected_tokens) * attention_to_real_flat[token_idx, expert_idx].unsqueeze(-1)
            
        null_values = null_values.view(batch_size, sequence_length, hidden_dim)  # Reshape back to (batch, 2*sequence_length, hidden_dim)
        
        return null_values, attention_to_real_flat


class OlmoeDecoderLayerWithRIM(OlmoeDecoderLayer):
    def __init__(self, config,layer_idx=None):
        super().__init__(config,layer_idx=layer_idx)
        self.mlm = OlmoeMoeBlockWithRIM(config)
        
class OlmoeModelWithRIM(OlmoeModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [OlmoeDecoderLayerWithRIM(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
class OlmoeForCausalLMWithRIM(OlmoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = OlmoeModelWithRIM(config)
        
    