import os 

from typing import List, Dict, Optional
from transformers import AutoModel, Qwen2ForCausalLM
from config import Qwen2WithRIMConfig, Qwen3WithRIMConfig
from model import Qwen2ForCausalLMWithRIM, Qwen3ForCausalLMWithRIM

def create_moe_from_specialists(
    base_model: str,
    specialists: List[str],
    moe_config: Optional[Qwen2WithRIMConfig] = None,
    ):
    print("Loading base model:", base_model)
    base_model = Qwen2ForCausalLM.from_pretrained(base_model)
    moe_model = Qwen2ForCausalLMWithRIM(config=moe_config)
    
    assert len(specialists) == moe_config.num_experts, \
        f"Number of specialists ({len(specialists)}) does not match the number of experts ({moe_config.num_experts})"
    
    moe_state_dict = moe_model.state_dict()
    base_state_dict = base_model.state_dict()
    
    replaced_params = {
        'experts': 0,
        'non_experts': 0,
        'rim_specific': 0,
    }
    
    print("Copying non-MLP parameters from base model...")
    for key in moe_state_dict:
        if key in base_state_dict and 'experts' not in key:
            moe_state_dict[key] = base_state_dict[key]
            replaced_params['non_experts'] += 1
    
    print('Copying MLP parameters from specialists...')
    for i, specialist in enumerate(specialists):
        print(f"Loading specialist {i+1}/{len(specialists)}: {specialist}")
        specialist_model = Qwen2ForCausalLM.from_pretrained(specialist)
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
                    replaced_params['experts'] += 1
    
    # Check for RIM-specific parameters
    for key in moe_state_dict:
        if any(name in key for name in ['key', 'value', 'expert_query', 'expert_states_flat']):
            replaced_params['rim_specific'] += 1
            
    moe_model.load_state_dict(moe_state_dict)
    print(f"Model merging complete. Stats:")
    print(f"  - Replaced {replaced_params['experts']} expert parameters")
    print(f"  - Copied {replaced_params['non_experts']} non-expert parameters")
    print(f"  - Kept {replaced_params['rim_specific']} RIM-specific parameters")
    
    return moe_model
