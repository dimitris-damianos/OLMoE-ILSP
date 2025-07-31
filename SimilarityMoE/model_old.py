from torch import nn
import torch.nn.functional as F
import torch

from transformers.models.olmoe.modeling_olmoe import OlmoeMLP

from RIM import RIMCell, RIM

class OlmoeSimilarityMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.latent_size = config.latent_size
        self.gate = nn.Linear(config.hidden_size, 
                              self.num_experts*self.latent_size, 
                              bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim) # hidden_states: (batch * sequence_length, hidden_dim)
        
        # Step 1: Compute latent features for each expert
        latent_feats = self.gate(hidden_states).contiguous() # latent_feats: (batch * sequence_length, n_experts*latent_size)
        latent_feats = latent_feats.view(batch_size*sequence_length, # latent_feats: (batch * sequence_length, n_experts, latent_size)
                                           self.num_experts, 
                                           self.latent_size)
        
        # Step 2: Compute cosine similarity between experts (ignore diagonal)
        latent_feats = F.normalize(latent_feats, dim=-1)
        latent_feats = torch.bmm(latent_feats, latent_feats.contiguous().transpose(1, 2)) # FIXME change transpose to view
        print(f'Sim matrix shape: {latent_feats.shape}')
        mask = torch.eye(self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype).bool()
        latent_feats = latent_feats.masked_fill(mask, float('-inf'))
        
        # Step 3: Find the top-k pairs of experts with the highest similarity (k=2)
        expert_weights, max_idx = latent_feats.view(latent_feats.shape[0], -1).max(dim=1)  # NOTE: flattned ids, needs to be reshaped
        expert_i = max_idx // self.num_experts  # row expert_i: (batch_size*seq_len, top_k)
        expert_j = max_idx % self.num_experts  # col expert_j: (batch_size*seq_len, top_k)
    
        # Stack indices for easier downstream use (e.g., gating/routing)
        selected_experts = torch.stack([expert_i, expert_j], dim=1)  # shape: (batch_size*seq_len, top_k, 2)
        
        return expert_weights, selected_experts
    
class OlmoeRimMoeBlock_traditional(nn.Module):
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            
        Returns:
            final_hidden_states: (batch_size, seq_len, hidden_size)
            router_logits: Router logits for auxiliary loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Get routing weights and selected experts using RIMCell
        routing_weights, selected_experts = self.router(hidden_states)
        
        # Normalize routing weights if needed
        if self.norm_topk_prob:
            # Reshape for normalization across experts dimension
            routing_weights = routing_weights.view(-1, self.top_k)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.view(-1)
        
        # Ensure routing weights match hidden states dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Create output tensor
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # One-hot encode selected experts - reshape for processing
        token_indices = torch.arange(batch_size * seq_len, device=hidden_states.device)
        token_indices = token_indices.repeat_interleave(self.top_k)
        
        # Process in groups by expert
        for expert_idx in range(self.num_experts):
            # Find where this expert was selected
            expert_mask = (selected_experts == expert_idx)
            if not expert_mask.any():
                continue
                
            # Get the token indices where this expert is active
            token_idx = token_indices[expert_mask]
            
            # Get corresponding weights
            expert_weights = routing_weights[expert_mask].unsqueeze(-1)
            
            # Process through the expert
            expert_output = self.experts[expert_idx](hidden_states_flat[token_idx])
            
            # Add weighted output to the result tensor
            final_hidden_states.index_add_(0, token_idx, expert_output * expert_weights)
        
        # Reshape back to sequence format
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)
        
        # For compatibility, create a tensor for router logits
        # This is just a placeholder since RIM doesn't use traditional router logits
        router_logits = torch.zeros(
            batch_size * seq_len, self.num_experts,
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        return final_hidden_states, router_logits
    