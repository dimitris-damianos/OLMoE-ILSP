from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)
        args.init_method(self.layer.weight)

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1,keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1)
        expert_weights, expert_indices = self._top_k(scores)
        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights, p=self.args.moe_normalize_expert_weights,dim=-1, keepdim=True)

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, expert_weights, expert_indices
    
class SimilarityRouter(LearnedRouter):
    def __init__(self, args: Arguments):
        super().__init__(args)
        self.args = args
        self.moe_num_experts = args.moe_num_experts
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size
        
        self.expert_matrices = torch.nn.Linear(
            self.hidden_size,
            self.latent_size * self.moe_num_experts,    # one matrix multiplication for all
            bias=False,
            dtype=common.dtype(args),
            device=args.device
        )
        args.init_method(self.expert_matrices.weight)   # borrowed from the original code
        
    def forward(self, x):
        # Step 1: Compute latent features for each expert
        latent_feats = self.expert_matrices(x).contiguous() # shape: (batch_size, seq_len, num_experts*latent_size)
        latent_feats = latent_feats.view(
            x.shape[0]*x.shape[1], 
            self.moe_num_experts, 
            self.latent_size  # shape: (batch_size*seq_len, num_experts, latent_size)
        )
        
        # Step 2: Compute cosine similarity between experts (ignore diagonal)
        latent_feats = torch.nn.functional.normalize(latent_feats, dim=-1)  # Normalize along the latent size dimension
        latent_feats = torch.bmm(latent_feats, latent_feats.contiguous().transpose(1, 2)) # FIXME change transpose to view
        mask = torch.eye(self.moe_num_experts, device=x.device, dtype=x.dtype).bool()
        latent_feats = latent_feats.masked_fill(mask, float('-inf'))
        _, max_idx = latent_feats.view(latent_feats.shape[0], -1).max(dim=1)
        
        max_pair = (max_idx // self.moe_num_experts, max_idx % self.moe_num_experts)
        
        return max_pair