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

    # TODO return values as the base OlmoeSparseMoeBlock does
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

class OlmoeMoeBlockWithRIMFlat(nn.Module):
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
        # print(f'Number of experts: {self.num_experts}, Top-k: {self.top_k}, Norm top-k prob: {self.norm_topk_prob}')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim) # (batch * sequence_length, hidden_dim)
        null_values = torch.zeros(
            (batch_size*sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        
        # null values are located the second half of the sequence (on dim 1) 
        hidden_states = torch.cat([hidden_states, null_values], dim=0)  # (batch*2*sequence_length, hidden_dim)
        print(f"Input hidden states shape: {hidden_states.shape}")
        
        # keys, values shared between experts 
        keys = self.key(hidden_states)  # (batch*2*sequence_length, hidden_dim)
        print(f"Keys shape: {keys.shape}")
        values = self.value(hidden_states)  # (batch*2*sequence_length, hidden_dim)
        print(f"Values shape: {values.shape}")
        expert_mask = []
        expert_weights = []
        
        # Compute the attention scores for each expert
        experts_flat_states = self.expert_states_flat(hidden_states)  # (batch *2* sequence_length, num_experts * hidden_dim)
        
        print(f"Experts flat states shape: {experts_flat_states.shape}")
        experts_flat_query = self.expert_query(experts_flat_states)  # (batch *2* sequence_length, num_experts * hidden_dim)
        print(f"Experts flat query shape: {experts_flat_query.shape}")
        
        attention_scores_flat = nn.functional.softmax(
            torch.matmul(experts_flat_query, torch.concat([keys for _ in range(self.num_experts)],dim=-1).T)/torch.sqrt(torch.tensor(hidden_dim)), dim=-1
        )
        attention_weights_flat = torch.matmul(attention_scores_flat, 
                                              torch.concat([values for _ in range(self.num_experts)],dim=-1))  # (batch * sequence_length, hidden_dim)
        print(f"Attention weights flat shape: {attention_weights_flat.shape}")
        
        # Separate attention weights for real tokens and null tokens
        attention_to_real_flat = attention_weights_flat[:batch_size*sequence_length, :].view(batch_size*sequence_length,
                                                                                             self.num_experts,
                                                                                             hidden_dim).sum(dim=-1)  # (batch*sequence_length,num_experts)
        print(f"Attention to real tokens shape: {attention_to_real_flat.shape}")    # (batch*sequence_length, num_experts)
        attention_to_null_flat = attention_weights_flat[batch_size*sequence_length:, :].view(batch_size*sequence_length,
                                                                                             self.num_experts,
                                                                                             hidden_dim).sum(dim=-1)  # (batch*sequence_length,num_experts)
        
        # Decide which tokens the expert prefers based on attention weights
        # If attention to real tokens is greater than to null tokens, the expert prefers the real tokens
        # Otherwise, it prefers the null tokens
        # Normalize the attention weights
        experts_mask = ((attention_to_real_flat - attention_to_null_flat) > 0)    
        attention_to_real_flat = attention_to_real_flat / (attention_to_real_flat + attention_to_null_flat) # Normalize the attention weights
        
        # print(f"Attention to real tokens shape: {attention_to_real_flat.shape}")    # (batch*sequence_length)
        # print(f"Attention to null tokens shape: {attention_to_null_flat.shape}")
        print(f"FLAT Token preference shape: {experts_mask.shape}")
        print(f"FLAT Attention to real tokens shape: {attention_to_real_flat.shape}")
        
        print(experts_mask.T)
        
        for expert_idx in range(self.num_experts):
            print(f"Processing expert {expert_idx}")
            print(f"Expert {expert_idx} mask shape: {experts_mask[:, expert_idx]}")
            token_idx = torch.where(experts_mask[:, expert_idx])        # Get token indices where the expert is selected in the batch*sequence_length range
            
            # print(token_idx)
            selected_tokens = hidden_states[token_idx]
            
            print(f"Selected tokens shape: {selected_tokens.shape}, Token indices: {token_idx}, {token_idx[0].shape}")
        
        # print(token_perference)
        return
        

        
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