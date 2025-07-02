from torch import nn
import torch.nn.functional as F
import torch

from transformers.models.olmoe.modeling_olmoe import OlmoeMLP

class OlmoeMoeBlockWithRIM(nn.Module):
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
        
        print(f"Input hidden states shape: {hidden_states.shape}")
        
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
            
            # 
            selected_tokens = hidden_states[batch_idx, token_idx, :]  # (num_selected_tokens, hidden_dim)
            # print(f"Selected tokens shape: {selected_tokens.shape}, Batch indices: {batch_idx.shape}, Token indices: {token_idx.shape}")
            
            
            null_values[batch_idx, token_idx, :] = self.experts[expert_idx](selected_tokens) * expert_weights[expert_idx, batch_idx, token_idx].unsqueeze(-1)
            
        print(f"Final hidden states shape: {null_values.shape}")
        print(f"{expert_weights.shape}")
        
        return 