import torch
import torch.nn as nn
import math
from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
from RIM import RIMCell

class RIMRouter(nn.Module):
    """
    Router using RIMCell to make routing decisions.
    Maintains a recurrent state for experts that evolves as it processes tokens.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.rim_hidden_size = getattr(config, "rim_hidden_size", 128)
        
        # Use RIMCell for routing - this handles all the key/query/value attention
        self.rim_cell = RIMCell(
            device='cpu',  # Will be overridden by the device of the inputs
            input_size=self.hidden_size,
            hidden_size=self.rim_hidden_size,
            num_units=self.num_experts,
            k=self.top_k,
            enable_comm=config.enable_comm,
            rnn_cell='LSTM'  # Using LSTM for state maintenance
        )
        
        # Initialize states for experts
        self.reset_states()
        
    def reset_states(self, batch_size=1): 
        """Reset the RIM states for new sequence processing"""
        device = next(self.parameters()).device
        self.hidden_states = torch.randn(batch_size, self.num_experts, self.rim_hidden_size, device=device) 
        self.cell_states = torch.randn(batch_size, self.num_experts, self.rim_hidden_size, device=device) 
    
    def forward(self, hidden_states):
        """
        Process input sequence through RIMCell for expert routing
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            
        Returns:
            routing_weights: (batch_size*seq_len, top_k)
            selected_experts: (batch_size*seq_len, top_k)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Reset states for new sequence
        self.reset_states(batch_size)
        
        # Store routing decisions for all tokens
        all_routing_weights = []
        all_selected_experts = []
        
        # Process token by token
        for t in range(seq_len):
            # print(f'Token {t}:')
            # Extract current token embedding and reshape for RIMCell
            token_emb = hidden_states[:, t:t+1, :]  # (batch_size, 1, hidden_size)
            
            # Get input attention mask from RIMCell without executing full forward pass
            # This gives us which experts are active and their attention scores
            inputs, mask_, attention_probs = self.rim_cell.input_attention_mask(token_emb, self.hidden_states,return_scores=True)
            
            # attention_scores has shape (batch_size, 1, num_experts)
            token_attention_scores = attention_probs[:,:,0]  # Now (batch_size, num_experts)
            # print(f"Not null sttention scores: {token_attention_scores}")
            # print(f'=======================================\n')
            # # Extract top-k attention scores for each batch item
            # mask has shape (batch_size, num_experts)
            expert_scores = []
            expert_indices = []
            
            for b in range(batch_size):
                # Get mask for active experts
                mask_b = mask_[b].bool()
                
                # Get indices of active experts (where mask is 1)
                print(f"Active experts for batch (mask) {b}: {mask_b}")
                active_indices = torch.nonzero(mask_b).squeeze(-1)
                print(f"Active experts for batch {b}: {active_indices}")
                
                # Get attention scores for these active experts
                print(f"Attention scores for batch {b}: {token_attention_scores[b]}")
                active_scores = token_attention_scores[b, active_indices]
                print(f"Active scores for batch {b}: {active_scores}\n")
                # # Normalize active scores to sum to 1
                active_scores = active_scores / active_scores.sum()
                
                # If we have fewer active experts than top_k, pad with minimal scores
                if len(active_indices) < self.top_k:
                    padding_size = self.top_k - len(active_indices)
                    
                    # Create random indices for padding
                    padding_indices = torch.randperm(self.num_experts)[:padding_size]
                    padding_indices = padding_indices.to(device)
                    
                    # Remove any overlap with active indices
                    for idx in active_indices:
                        padding_indices = padding_indices[padding_indices != idx]
                    # If we still need more padding
                    if len(padding_indices) < padding_size:
                        extra = torch.randperm(self.num_experts)
                        for idx in torch.cat([active_indices, padding_indices]):
                            extra = extra[extra != idx]
                        padding_indices = torch.cat([padding_indices, extra[:padding_size - len(padding_indices)]])
                    
                    # Minimal scores for padding
                    padding_scores = torch.ones(padding_size, device=device) * 1e-8
                    
                    # Combine
                    top_indices = torch.cat([active_indices, padding_indices[:padding_size]])
                    top_scores = torch.cat([active_scores, padding_scores])
                else:
                    # If we have more active experts than top_k, get the top_k by score
                    if len(active_indices) > self.top_k:
                        topk_vals, topk_idx = torch.topk(active_scores, self.top_k)
                        top_indices = active_indices[topk_idx]
                        top_scores = topk_vals
                    else:
                        top_indices = active_indices
                        top_scores = active_scores
                
                expert_scores.append(top_scores)
                expert_indices.append(top_indices)
            
            # Stack batch results
            token_weights = torch.stack(expert_scores)  # (batch_size, top_k)
            token_indices = torch.stack(expert_indices)  # (batch_size, top_k)
            
            # Now do the actual RIMCell update to maintain expert states
            new_h, new_c = self.rim_cell(token_emb, self.hidden_states, self.cell_states)
            self.hidden_states = new_h
            self.cell_states = new_c
            
            # Store results for this token (flatten batch and top_k dimensions)
            all_routing_weights.append(token_weights.view(-1))
            all_selected_experts.append(token_indices.view(-1))
        
        # Concatenate all tokens' results
        routing_weights = torch.cat(all_routing_weights)  # (batch_size*seq_len*top_k)
        selected_experts = torch.cat(all_selected_experts)  # (batch_size*seq_len*top_k)
        
        return routing_weights, selected_experts
    
if __name__ == "__main__":
    # Example usage
    class Config:
        num_experts = 4
        hidden_size = 1024
        num_experts_per_tok = 1
        rim_hidden_size = 128
        norm_topk_prob = True
        enable_comm = True
    
    config = Config()
    model = RIMRouter(config)
    
    # Create dummy input
    batch_size, seq_len, hidden_size = 2, 10, config.hidden_size
    inputs = torch.randn(batch_size, seq_len, hidden_size)
    
    weights, indices = model(inputs)
    weights = weights.view(batch_size, seq_len, config.num_experts_per_tok)
    indices = indices.view(batch_size, seq_len, config.num_experts_per_tok)
    print(f"weights: {weights}, indices: {indices}")
    print(f"weights shape: {weights.shape}, indices shape: {indices.shape}")