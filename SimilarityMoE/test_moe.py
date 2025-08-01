import torch
from torch import nn
import matplotlib.pyplot as plt
from config import Qwen3WithRIMConfig
from model import Qwen3MoeBlockWithRIM, load_balancing_loss_for_rim
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Create output directory
os.makedirs("moe_test_results", exist_ok=True)

def create_dummy_data(batch_size=4, seq_len=128, hidden_size=768):
    """Create dummy input data for testing."""
    return torch.randn(batch_size, seq_len, hidden_size)

def test_model():
    # Configuration
    config = Qwen3WithRIMConfig(
        num_experts=4,
        key_size=64,
        output_expert_mask=True,
        output_router_logits=True,
        router_aux_loss_coef=0.1,
        experts_top_p=0.8,
        hidden_size=768,
        intermediate_size=3072,
        detach_null_states=False,
        use_latent_states=True
    )
    batch_size, seq_len, hidden_size = 4, 128, 768
    x = create_dummy_data(batch_size, seq_len, hidden_size)
    # Create target data with same shape as output
    target = create_dummy_data(batch_size, seq_len, hidden_size)
        
    # Initialize model
    model = Qwen3MoeBlockWithRIM(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Set model to train mode
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    losses = []
    grad_norms = []
    
    print("Starting training...")
    for step in range(1000):
        optimizer.zero_grad()
        outputs = model(x)
        
        # Ensure outputs has expected format
        if isinstance(outputs, tuple) and len(outputs) == 3:
            output, router_logits, expert_mask = outputs
        else:
            raise ValueError(f"Expected model to return a tuple of 3 elements, got {outputs}")
        
        if output.shape != x.shape:
            output = output.view(batch_size, seq_len, hidden_size)
        
        mse_loss = nn.MSELoss()(output, target)
    
        # Add auxiliary loss if available
        if router_logits is not None and expert_mask is not None:
            aux_loss = load_balancing_loss_for_rim(
                (router_logits,), 
                (expert_mask,),
                attention_mask=None,
                num_experts=model.num_experts
            )
            total_loss = mse_loss + 0.1 * aux_loss
        else:
            total_loss = mse_loss
        
        total_loss.backward()
        
        has_nan_or_inf = False
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f"WARNING: NaN or Inf gradients in {name}")
                has_nan_or_inf = True
        
        if has_nan_or_inf:
            print(f"Step {step}: NaN or Inf gradients detected")
            break
        
        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Store metrics
        losses.append(total_loss.item())
        grad_norms.append(grad_norm.item())
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}: Loss = {total_loss.item():.4f}, Grad norm = {grad_norm.item():.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(grad_norms)
    plt.title("Gradient Norm")
    plt.xlabel("Step")
    plt.ylabel("Norm")
    
    plt.tight_layout()
    plt.savefig("moe_test_results/training_plot.png")
    
    print(f"Testing completed. Results saved to moe_test_results/")

if __name__ == "__main__":
    # Enable anomaly detection for better error messages
    torch.autograd.set_detect_anomaly(True)
    test_model()