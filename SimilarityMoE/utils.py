import os 
import re

import wandb

from typing import List, Dict, Optional, Type, Union
from transformers import AutoModel, Qwen2ForCausalLM, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from config import Qwen2WithRIMConfig, Qwen3WithRIMConfig
from model import Qwen2ForCausalLMWithRIM, Qwen3ForCausalLMWithRIM

from torch.utils.tensorboard import SummaryWriter
import torch

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
    
class GradientLoggingCallbackTensorboard(TrainerCallback):
    def __init__(self, log_dir="runs/gradient_logs"):
        self.writer = SummaryWriter(log_dir)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        global_step = state.global_step

        # Log gradients every `n` steps to avoid overhead
        if global_step % self.log_every_n_steps != 0:
            return

        if model.training:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach().cpu()
                    # Log gradient norm
                    grad_norm = torch.norm(grad)
                    self.writer.add_scalar(f"gradients_norm/{name}", grad_norm, global_step)
                    # Log full histogram
                    self.writer.add_histogram(f"gradients_hist/{name}", grad, global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()