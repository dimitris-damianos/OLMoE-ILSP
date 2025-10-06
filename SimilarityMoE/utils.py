import os 
import re

import wandb

from typing import List, Dict, Optional, Type, Union
from transformers import (
    AutoModel, Qwen2ForCausalLM, 
    TrainerCallback, TrainerControl, 
    TrainerState, TrainingArguments,
    Qwen3MoeConfig, Qwen3MoeForCausalLM
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from config import Qwen2WithRIMConfig, Qwen3WithRIMConfig
from model import Qwen2ForCausalLMWithRIM, Qwen3ForCausalLMWithRIM

from torch.utils.tensorboard import SummaryWriter
import torch
from transformers.trainer_utils import is_main_process
import subprocess
import datetime

import logging
logger = logging.getLogger(__name__)

def create_qwen3_moe_from_specialists(
    base_model_path: str,
    specialists: List[str],
    moe_config: Qwen3MoeConfig
):
    """Create a Qwen3Moe model with experts from specialists with detailed statistics tracking."""
    # Load base model (standard Qwen3)
    print(f"Loading base model from {base_model_path}")
    base_model = Qwen3ForCausalLM.from_pretrained(base_model_path)
    base_state_dict = base_model.state_dict()
    
    with open('./ddam_log/moe_config.txt','w') as f:
        f.write(str(moe_config))

    with open('./ddam_log/base_config.txt','w') as f:
        f.write(str(base_model.config))
    
    moe_config.moe_intermediate_size = base_model.config.intermediate_size
    
    
    # Create empty MoE model with the config
    print(f"Creating empty MoE model with config: num_experts={moe_config.num_experts}, "
                f"num_experts_per_tok={moe_config.num_experts_per_tok}, "
                f"hidden_size={moe_config.hidden_size}, "
                f"intermediate_size={moe_config.moe_intermediate_size}")
    moe_model = Qwen3MoeForCausalLM(config=moe_config)
    moe_state_dict = moe_model.state_dict()
    
    # Track statistics
    stats = {
        'total_params': 0,
        'copied_from_base': 0,
        'copied_from_specialists': 0,
        'random_router_params': 0,
        'other_random_params': 0,
        'layer_stats': {},
        'param_types': {
            'attention': 0,
            'mlp': 0,
            'router': 0,
            'other': 0,
        }
    }
    
    # Track weights by layer and component
    for key in moe_state_dict:
        param_size = moe_state_dict[key].numel()
        stats['total_params'] += param_size
        
        # Track by layer
        layer_match = re.search(r'model\.layers\.(\d+)\.', key)
        if layer_match:
            layer_idx = int(layer_match.group(1))
            if layer_idx not in stats['layer_stats']:
                stats['layer_stats'][layer_idx] = {
                    'total': 0,
                    'base': 0,
                    'specialists': 0,
                    'random': 0,
                    'attention': 0,
                    'mlp': 0,
                    'router': 0,
                }
            stats['layer_stats'][layer_idx]['total'] += param_size
            
            # Track by component type
            if 'attention' in key:
                stats['param_types']['attention'] += param_size
                stats['layer_stats'][layer_idx]['attention'] += param_size
            elif 'mlp' in key or 'block_sparse_moe' in key:
                stats['param_types']['mlp'] += param_size
                stats['layer_stats'][layer_idx]['mlp'] += param_size
            elif 'gate' in key and 'gate_proj' not in key:  # This is the router
                stats['param_types']['router'] += param_size
                stats['layer_stats'][layer_idx]['router'] += param_size
            else:
                stats['param_types']['other'] += param_size
    
    # 1. Copy all non-MoE weights from base model
    print("Copying non-MoE weights from base model...")
    for key in list(moe_state_dict.keys()):
        # Skip router weights - we want these randomly initialized
        if "gate" in key and "gate_proj" not in key:
            stats['random_router_params'] += moe_state_dict[key].numel()
            
            # Track by layer if applicable
            layer_match = re.search(r'model\.layers\.(\d+)\.', key)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                stats['layer_stats'][layer_idx]['random'] += moe_state_dict[key].numel()
            continue
        
        # Skip expert weights - we'll copy these from specialists
        if "experts" in key:
            continue
            
        # Copy matching parameters from base model
        if key in base_state_dict:
            moe_state_dict[key] = base_state_dict[key]
            param_size = moe_state_dict[key].numel()
            stats['copied_from_base'] += param_size
            
            # Track by layer if applicable
            layer_match = re.search(r'model\.layers\.(\d+)\.', key)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                stats['layer_stats'][layer_idx]['base'] += param_size
        else:
            # This parameter stays random
            stats['other_random_params'] += moe_state_dict[key].numel()
    
    # 2. Copy MLP weights from specialists to experts
    print(f"Copying MLP parameters from {len(specialists)} specialists...")
    specialist_stats = []
    for i, specialist_path in enumerate(specialists):
        specialist_copied = 0
        print(f"Loading specialist {i+1}/{len(specialists)}: {specialist_path}")
        specialist = Qwen3ForCausalLM.from_pretrained(specialist_path)
        specialist_dict = specialist.state_dict()
        
        # Copy MLP weights to each expert
        for layer_idx in range(moe_config.num_hidden_layers):
            # IMPORTANT: Corrected parameter mapping
            # Standard MoE parameter mapping - note the different path structure
            base_mlp_prefix = f"model.layers.{layer_idx}.mlp."
            # Correct path for MoE model experts
            moe_expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{i}."
            
            # Map each MLP parameter to the corresponding expert parameter
            param_mapping = {
                f"{base_mlp_prefix}gate_proj.weight": f"{moe_expert_prefix}gate_proj.weight",
                f"{base_mlp_prefix}down_proj.weight": f"{moe_expert_prefix}down_proj.weight",
                f"{base_mlp_prefix}up_proj.weight": f"{moe_expert_prefix}up_proj.weight",
            }
            
            for base_key, moe_key in param_mapping.items():
                if base_key in specialist_dict:
                    # Check if the key exists in the moe_state_dict
                    if moe_key not in moe_state_dict:
                        # Try alternative key format (handle differences in MoE architecture)
                        alt_moe_key = moe_key.replace("mlp.experts", "block_sparse_moe.experts")
                        if alt_moe_key in moe_state_dict:
                            moe_key = alt_moe_key
                        else:
                            print(f"Warning: Could not find matching key for {base_key} -> {moe_key} or {alt_moe_key}")
                            continue
                    
                    # Check for dimension mismatch
                    if specialist_dict[base_key].shape != moe_state_dict[moe_key].shape:
                        print(f"Shape mismatch for {base_key} -> {moe_key}: "
                              f"{specialist_dict[base_key].shape} vs {moe_state_dict[moe_key].shape}")
                        
                        # Handle specific dimension mismatches - scale down the larger model
                        if len(specialist_dict[base_key].shape) == 2 and len(moe_state_dict[moe_key].shape) == 2:
                            # Need to reshape/resize the tensor to fit
                            src_shape = specialist_dict[base_key].shape
                            dst_shape = moe_state_dict[moe_key].shape
                            
                            # Create new tensor with zeros of the target shape
                            new_tensor = torch.zeros(dst_shape, 
                                                    dtype=moe_state_dict[moe_key].dtype,
                                                    device=moe_state_dict[moe_key].device)
                            
                            # Copy values from specialist up to the minimum dimensions
                            min_dim0 = min(src_shape[0], dst_shape[0])
                            min_dim1 = min(src_shape[1], dst_shape[1])
                            new_tensor[:min_dim0, :min_dim1] = specialist_dict[base_key][:min_dim0, :min_dim1]
                            
                            # Assign the resized tensor
                            moe_state_dict[moe_key] = new_tensor
                            print(f"  Resized parameter from {src_shape} to {dst_shape}")
                            param_size = new_tensor.numel()
                        else:
                            print(f"  Cannot handle mismatch for non-2D tensors, skipping")
                            continue
                    else:
                        # Shapes match, direct copy
                        moe_state_dict[moe_key] = specialist_dict[base_key]
                        param_size = moe_state_dict[moe_key].numel()
                    
                    # Update statistics
                    specialist_copied += param_size
                    stats['copied_from_specialists'] += param_size
                    
                    # Update layer stats if applicable
                    layer_match = re.search(r'model\.layers\.(\d+)\.', moe_key)
                    if layer_match:
                        layer_idx = int(layer_match.group(1))
                        stats['layer_stats'][layer_idx]['specialists'] += param_size
        
        specialist_stats.append({
            'path': specialist_path,
            'params_copied': specialist_copied,
            'percentage': specialist_copied / stats['total_params'] * 100
        })
    
    # Load state dict into model
    print("Loading state dictionary into model...")
    incompatible_keys = moe_model.load_state_dict(moe_state_dict, strict=False)
    
    # Print statistics
    print("=" * 50)
    print("MODEL WEIGHT STATISTICS")
    print("=" * 50)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Copied from base model: {stats['copied_from_base']:,} "
          f"({stats['copied_from_base']/stats['total_params']*100:.2f}%)")
    print(f"Copied from specialists: {stats['copied_from_specialists']:,} "
          f"({stats['copied_from_specialists']/stats['total_params']*100:.2f}%)")
    print(f"Random router parameters: {stats['random_router_params']:,} "
          f"({stats['random_router_params']/stats['total_params']*100:.2f}%)")
    print(f"Other random parameters: {stats['other_random_params']:,} "
          f"({stats['other_random_params']/stats['total_params']*100:.2f}%)")

    # Parameter breakdown by type
    print("\nParameter breakdown by type:")
    for param_type, count in stats['param_types'].items():
        print(f"  - {param_type}: {count:,} ({count/stats['total_params']*100:.2f}%)")
    
    # Specialist breakdown
    print("\nSpecialist contributions:")
    for i, spec_stat in enumerate(specialist_stats):
        print(f"  - Specialist {i+1}: {spec_stat['path']}")
        print(f"    * Params copied: {spec_stat['params_copied']:,} ({spec_stat['percentage']:.2f}%)")

    # Check for any incompatible keys
    if incompatible_keys.missing_keys:
        print(f"Missing keys: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
    
    print("=" * 50)
    
    # Show model structure for verification
    print("\nFinal model structure:")
    for name, module in moe_model.named_children():
        print(f"- {name}: {type(module).__name__}")
        if hasattr(module, "layers") and len(module.layers) > 0:
            # Show first layer structure
            first_layer = module.layers[0]
            print(f"  First layer components:")
            for comp_name, comp in first_layer.named_children():
                print(f"    - {comp_name}: {type(comp).__name__}")
                if comp_name == "mlp" and hasattr(comp, "experts"):
                    print(f"      * Number of experts: {len(comp.experts)}")
                    print(f"      * Expert type: {type(comp.experts[0]).__name__}")
    
    # Verify router weights are actually random
    router_weights = []
    router_names = []
    for name, param in moe_model.named_parameters():
        if "gate" in name and "gate_proj" not in name:  # Router weights
            mean = param.data.mean().item()
            std = param.data.std().item()
            print(f"Router parameter: {name}, shape: {param.shape}, mean: {mean:.6f}, std: {std:.6f}")
            router_weights.append(param.data.flatten()[:20].tolist())
            router_names.append(name)
    
    # Compare router weights to ensure they're different (random)
    if len(router_weights) >= 2:
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(router_weights[0]), 
            torch.tensor(router_weights[1]), 
            dim=0
        ).item()
        print(f"Cosine similarity between {router_names[0]} and {router_names[1]}: {similarity:.6f} (should be near 0 for random)")
    
    return moe_model

# def create_qwen3_moe_from_specialists(
#     base_model_path: str,
#     specialists: List[str],
#     moe_config: Qwen3MoeConfig
# ):
#     """Create a Qwen3Moe model with experts from specialists."""
#     # Load base model (standard Qwen3)
#     base_model = Qwen3ForCausalLM.from_pretrained(base_model_path)
#     base_state_dict = base_model.state_dict()
    
#     # Create empty MoE model with the config
#     moe_model = Qwen3MoeForCausalLM(config=moe_config)
#     moe_state_dict = moe_model.state_dict()
    
#     # 1. Copy all non-MoE weights from base model
#     for key in moe_state_dict:
#         # Skip router weights - we want these randomly initialized
#         if "router" in key:
#             continue
            
#         # Copy matching parameters from base model
#         if key in base_state_dict:
#             moe_state_dict[key] = base_state_dict[key]
    
#     # 2. Copy MLP weights from specialists to experts
#     for i, specialist_path in enumerate(specialists):
#         specialist = Qwen3ForCausalLM.from_pretrained(specialist_path)
#         specialist_dict = specialist.state_dict()
        
#         # Copy MLP weights to each expert
#         for layer_idx in range(moe_config.num_hidden_layers):
#             # Standard MoE parameter mapping
#             base_mlp_prefix = f"model.layers.{layer_idx}.mlp."
#             moe_expert_prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{i}."
            
#             # Map each MLP parameter to the corresponding expert parameter
#             param_mapping = {
#                 f"{base_mlp_prefix}gate_proj.weight": f"{moe_expert_prefix}gate_proj.weight",
#                 f"{base_mlp_prefix}down_proj.weight": f"{moe_expert_prefix}down_proj.weight",
#                 f"{base_mlp_prefix}up_proj.weight": f"{moe_expert_prefix}up_proj.weight",
#             }
            
#             for base_key, moe_key in param_mapping.items():
#                 if base_key in specialist_dict and moe_key in moe_state_dict:
#                     moe_state_dict[moe_key] = specialist_dict[base_key]
    
#     # Load state dict into model
#     moe_model.load_state_dict(moe_state_dict)
#     return moe_model 

def create_moe_from_specialists(
    base_model: str,
    specialists: List[str],
    moe_config: Optional[Qwen2WithRIMConfig] = None,
    base_class: Union[Qwen2ForCausalLM, Qwen3ForCausalLM] = Qwen2ForCausalLM,
    moe_class: Union[Qwen2ForCausalLMWithRIM, Qwen3ForCausalLMWithRIM] = Qwen2ForCausalLMWithRIM
    ):
    print("Loading base model:", base_model)
    base_model = base_class.from_pretrained(base_model)
    moe_model = moe_class(config=moe_config)
    
    assert len(specialists) == moe_config.num_experts, \
        f"Number of specialists ({len(specialists)}) does not match the number of experts ({moe_config.num_experts})"
    
    moe_state_dict = moe_model.state_dict()
    base_state_dict = base_model.state_dict()
    
    replaced_params = {  # number of parameter tensors, not scalar params
        'experts': 0,
        'non_experts': 0,
        'rim_specific': 0,
    }
    
    print("Copying non-MLP parameters from base model...")
    for key in moe_state_dict:
        if key in base_state_dict and 'experts' not in key:
            moe_state_dict[key] = base_state_dict[key]
            # replaced_params['non_experts'] += 1
            replaced_params['non_experts'] += moe_state_dict[key].numel()  # scalar params
    
    print('Copying MLP parameters from specialists...')
    for i, specialist in enumerate(specialists):
        print(f"Loading specialist {i+1}/{len(specialists)}: {specialist}")
        specialist_model = base_class.from_pretrained(specialist)
        specialist_dict = specialist_model.state_dict()
        
        for layer_idx in range(moe_config.num_hidden_layers):
            base_mlp_prefix = f"model.layers.{layer_idx}.mlp."
            moe_mlp_prefix = f"model.layers.{layer_idx}.mlp.experts.{i}."
            
            param_mapping = {
                f"{base_mlp_prefix}gate_proj.weight": f"{moe_mlp_prefix}gate_proj.weight",
                f"{base_mlp_prefix}down_proj.weight": f"{moe_mlp_prefix}down_proj.weight",
                f"{base_mlp_prefix}up_proj.weight": f"{moe_mlp_prefix}up_proj.weight",
            }
            
            for base_key, moe_key in param_mapping.items():
                if base_key in specialist_dict and moe_key in moe_state_dict:
                    moe_state_dict[moe_key] = specialist_dict[base_key]
                    # replaced_params['experts'] += 1
                    replaced_params['experts'] += moe_state_dict[moe_key].numel()  # scalar params
    
    # Check for RIM-specific parameters
    for key in moe_state_dict:
        if any(name in key for name in ['key', 'value', 'expert_query', 'expert_states_flat']):
            # replaced_params['rim_specific'] += 1
            replaced_params['rim_specific'] += moe_state_dict[key].numel()  # scalar params
            
    moe_model.load_state_dict(moe_state_dict)
    print(f"Model merging complete. Stats:")
    print(f"  - Replaced {replaced_params['experts']} expert parameters")
    print(f"  - Copied {replaced_params['non_experts']} non-expert parameters")
    print(f"  - Kept {replaced_params['rim_specific']} RIM-specific parameters")
    
    return moe_model

class GradientNormLoggingCallbackWandClipped(TrainerCallback):

    def on_pre_optimizer_step(
        self,args: TrainingArguments,state: TrainerState,control: TrainerControl, **kwargs
    ) -> TrainerControl:

        model = kwargs["model"]
        per_param_norms = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            per_param_norms[name] = param.grad.detach().norm(2).item()

        grouped = {}
        pattern = re.compile(r"layers\.\d+\.(.+)")
        for name, norm in per_param_norms.items():
            m = pattern.search(name)
            group_key = m.group(1) if m else name
            grouped.setdefault(group_key, []).append(norm)

        mean_norms = {k: sum(v) / len(v) for k, v in grouped.items()}

        wandb.log(
            {
                f"grad_norms/{param_name}": norm
                for param_name, norm in per_param_norms.items()
            },
            step=state.global_step,
        )

        wandb.log(
            {
                f"mean_grad_norms/{group_name}": mean
                for group_name, mean in mean_norms.items()
            },
            step=state.global_step,
        )

        return control

class GradientNormLoggingCallbackGenericClipped(TrainerCallback):
    def on_pre_optimizer_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        model = kwargs["model"]
        
        per_param_norms = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            per_param_norms[name] = param.grad.detach().norm(2).item()

        grouped = {}
        pattern = re.compile(r"layers\.\d+\.(.+)")
        for name, norm in per_param_norms.items():
            m = pattern.search(name)
            key = m.group(1) if m else name
            grouped.setdefault(key, []).append(norm)
        
        mean_norms = {k: sum(v) / len(v) for k, v in grouped.items()}

        self._to_log = {
            **{f"grad_norms/{n}": v for n, v in per_param_norms.items()},
            **{f"mean_norms/{k}": v for k, v in mean_norms.items()},
        }
        return control

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and hasattr(self, "_to_log"):
            logs.update(self._to_log)
            del self._to_log
        return control
    
class GradNormHookCallbackWandPreClip(TrainerCallback):
    def __init__(self):
        self._per_param = {}
        self._grouped = {}
        self._hooks = []
        self._pattern = re.compile(r"layers\.\d+\.(.+)")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            def make_hook(pname):
                
                def hook_fn(grad):
                    norm = grad.norm(2).item()
                    self._per_param[pname] = norm
                    m = self._pattern.search(pname)
                    key = m.group(1) if m else pname
                    self._grouped.setdefault(key, []).append(norm)
                    return grad
                return hook_fn

            h = param.register_hook(make_hook(name))
            self._hooks.append(h)
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        mean_norms = {k: sum(v)/len(v) for k, v in self._grouped.items()}

        wandb.log(
            {f"grad_norms/{n}": v for n, v in self._per_param.items()},
            step=state.global_step,
        )
        wandb.log(
            {f"mean_grad_norms/{k}": v for k, v in mean_norms.items()},
            step=state.global_step,
        )

        self._per_param.clear()
        self._grouped.clear()

        return control
    
class GradNormHookCallbackGenericPreClip(TrainerCallback):

    def __init__(self):
        self._per_param = {}
        self._grouped = {}
        self._pattern = re.compile(r"layers\.\d+\.(.+)")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            def make_hook(pname):
                def hook_fn(grad):
                    norm = grad.norm(2).item()
                    self._per_param[pname] = norm
                    key = self._pattern.search(pname)
                    self._grouped.setdefault(key.group(1) if key else pname, []).append(norm)
                    return grad
                return hook_fn

            param.register_hook(make_hook(name))
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._mean_norms = {
            k: sum(v) / len(v) for k, v in self._grouped.items()
        }

        self._to_log = {
            **{f"grad_norms/{n}": v for n, v in self._per_param.items()},
            **{f"mean_grad_norms/{k}": v for k, v in self._mean_norms.items()},
        }

        self._per_param.clear()
        self._grouped.clear()
        return control

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and hasattr(self, "_to_log"):
            logs.update(self._to_log)
            del self._to_log
        return control
        
class SaveExpertMaskCallback(TrainerCallback):
    def __init__(self, save_dir, save_every_n_steps=100):
        self.save_dir = save_dir
        self.save_every_n_steps = save_every_n_steps
        os.makedirs(self.save_dir, exist_ok=True)
        self.step_counter = 0
        logger.info(f"SaveExpertMaskCallback initialized, saving masks to {self.save_dir}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Save expert mask at each step."""
        # The model stores the last outputs temporarily
        if not hasattr(model, "_last_forward_outputs"):
            logger.warning("Model does not have _last_forward_outputs attribute")
            return

        # Get outputs from the last forward pass
        outputs = model._last_forward_outputs
        
        # Check if outputs has expert_mask attribute
        if hasattr(outputs, "expert_mask") and outputs.expert_mask is not None:
            if self.step_counter % self.save_every_n_steps == 0:
                expert_mask = outputs.expert_mask
                # Save the expert mask
                save_path = os.path.join(self.save_dir, f"expert_mask_step_{state.global_step}.pt")
                torch.save(expert_mask, save_path)
                logger.info(f"Saved expert mask for step {state.global_step} to {save_path}")
            self.step_counter += 1
        else:
            logger.warning(f"No expert_mask found in model outputs at step {state.global_step}")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Add hook to capture outputs during forward pass."""
        # Store the original forward method
        original_forward = model.forward
        
        def forward_with_save(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            # Store the outputs for access in on_step_end
            model._last_forward_outputs = outputs
            return outputs
        
        # Replace the forward method with our version
        model.forward = forward_with_save
        logger.info("Added forward hook to capture expert masks")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Restore original forward method."""
        if hasattr(model, "forward") and hasattr(model, "_original_forward"):
            model.forward = model._original_forward
            logger.info(f"Training completed. Saved {self.step_counter} expert masks.")


class GradientLoggingCallbackTensorboard(TrainerCallback):
    def __init__(self, log_dir: str = "./runs/gradients", log_every: int = 100):
        super().__init__()
        self.log_every = log_every
        self.log_dir = log_dir
        # Initialize writer in on_train_begin to ensure proper distributed setup
        self.writer = None
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Gradient logging will be saved to {log_dir} every {log_every} steps")

    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize writer here after distributed training is set up
        if args.local_rank == 0 or args.local_rank == -1:  # Main process
            self.writer = SummaryWriter(self.log_dir)
            logger.info(f"Initialized TensorBoard writer for gradient logging on main process")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # Skip if not main process or writer not initialized
        if self.writer is None:
            return control
            
        # Skip if not at log interval
        if state.global_step % self.log_every != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control
            
        global_step = state.global_step

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().data.norm(2).item()
                self.writer.add_scalar(f"grad_norm/{name}", grad_norm, global_step)
                
        # Flush to ensure data is written
        self.writer.flush()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.close()
        return control
    
class MemoryUsageCallback(TrainerCallback):
    def __init__(self, step_interval=1):
        self.step_interval = step_interval
    def get_nvidia_smi_memory(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return int(result.stdout.strip().split('\n')[0])  # MB
        except Exception as e:
            print(f"[MemoryCallback] Failed to read nvidia-smi: {e}")
            return -1

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.step_interval == 0:
            torch_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            nvidia_mem = self.get_nvidia_smi_memory()

            print(f"[{datetime.datetime.now()}][Step {state.global_step}] Torch max mem: {torch_mem:.2f} MB | Nvidia-smi: {nvidia_mem} MB")

            torch.cuda.reset_peak_memory_stats()
        