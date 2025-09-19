import os
import argparse
from typing import List, Optional, Union
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from config import Qwen2WithRIMConfig, Qwen3WithRIMConfig
from model import Qwen3ForCausalLMWithRIM
from utils import create_moe_from_specialists, create_qwen3_moe_from_specialists

from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig, Qwen3Config

def main():
    parser = argparse.ArgumentParser(description="Merge Qwen specialists into a RIM-based MoE model.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base pretrained model.")
    parser.add_argument("--specialists", type=str, nargs="+", required=True, help="Paths to fine-tuned specialist models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the merged MoE model.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type (qwen2 or qwen3).")
    parser.add_argument("--key_size", type=int, default=512, help="Attention key size for RIMs.")
    parser.add_argument("--experts_top_p", type=float, default=0.5, help="Top-p threshold for expert routing.")
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.1, help="Router auxiliary loss coefficient.")
    parser.add_argument("--output_expert_mask", action="store_true")
    parser.add_argument("--output_router_logits", action="store_true")
    parser.add_argument("--detach_null_states", action="store_true", help="Detach null states in RIMs.")
    parser.add_argument("--use_latent_states", action="store_true", help="Use latent states for experts.")
    parser.add_argument("--use_dynamic_routing", action="store_true", help="Set for top-p routing, otherwise top-k.")
    parser.add_argument("--experts_top_k", type=int, default=2, help="Top-k experts to select.")

    args = parser.parse_args()

    num_experts = len(args.specialists)

    
    if args.model_type == "qwen3":
        config = Qwen3WithRIMConfig.from_pretrained(
            args.base_model,
            num_experts=num_experts,
            expert_attn_size=args.key_size,
            output_expert_mask=args.output_expert_mask,
            output_router_logits=args.output_router_logits,
            router_aux_loss_coef=args.router_aux_loss_coef,
            experts_top_p=args.experts_top_p,
            experts_top_k=args.experts_top_k,
            detach_null_states=args.detach_null_states,
            use_latent_states=args.use_latent_states,
            use_dynamic_routing=args.use_dynamic_routing,
        )
        base_class = Qwen3ForCausalLM
        moe_class = Qwen3ForCausalLMWithRIM
    elif args.model_type == "qwen3_moe":
        config = Qwen3MoeConfig.from_pretrained(
            args.base_model,
            num_experts=num_experts,
            num_experts_per_tok = args.experts_top_k,
            output_router_logits = args.output_router_logits,
            router_aux_loss_coef=args.router_aux_loss_coef,
        )
        print(f'Qwen3MoeConfig: {config}')
        base_class = Qwen3MoeForCausalLM
        moe_class = Qwen3MoeForCausalLM
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    base_config = Qwen3Config.from_pretrained(args.base_model)
    print(f'Base config: {base_config}')

    print(f"Merging {num_experts} expert models into a RIM-based MoE model...\n")
    print("Base model:", args.base_model)
    print("Specialists:", args.specialists)
    print("Saving to:", args.output_dir)
    print("Model config:", config)

    if args.model_type == "qwen3_moe":
        moe_model = create_qwen3_moe_from_specialists(
            base_model_path=args.base_model,
            specialists=args.specialists,
            moe_config=config
        )
    else: 
        moe_model = create_moe_from_specialists(
            base_model=args.base_model,
            specialists=args.specialists,
            moe_config=config,
            base_class=base_class,
            moe_class=moe_class,
        )
    
    with open('merged_model.txt','w') as f: 
        f.write(str(moe_model))

    print(f"Saving merged MoE model to {args.output_dir}...")
    moe_model.save_pretrained(args.output_dir, safe_serialization=True)
    config.save_pretrained(args.output_dir)
    # TODO: save tokenizer
    
    total_params = sum(p.numel() for p in moe_model.parameters())
    print(f"Total parameters in merged MoE model: {total_params:,}")

if __name__ == "__main__":
    main()
