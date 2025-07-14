import os
import argparse
from typing import List, Optional, Union
from transformers import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from config import Qwen2WithRIMConfig, Qwen3WithRIMConfig
from model import Qwen2ForCausalLMWithRIM, Qwen3ForCausalLMWithRIM
from utils import create_moe_from_specialists

def main():
    parser = argparse.ArgumentParser(description="Merge Qwen specialists into a RIM-based MoE model.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base pretrained model.")
    parser.add_argument("--specialists", type=str, nargs="+", required=True, help="Paths to fine-tuned specialist models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the merged MoE model.")
    parser.add_argument("--model_type", type=str, choices=["qwen2", "qwen3"], required=True, help="Model type (qwen2 or qwen3).")
    parser.add_argument("--key_size", type=int, default=512, help="Attention key size for RIMs.")
    parser.add_argument("--experts_top_p", type=float, default=0.5, help="Top-p threshold for expert routing.")
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.1, help="Router auxiliary loss coefficient.")

    args = parser.parse_args()

    num_experts = len(args.specialists)

    if args.model_type == "qwen2":
        config = Qwen2WithRIMConfig.from_pretrained(
            args.base_model,
            num_experts=num_experts,
            expert_attn_size=args.key_size,
            experts_top_p=args.experts_top_p,
            router_aux_loss_coef=args.router_aux_loss_coef,
            output_expert_mask=True,
        )
        base_class = Qwen2ForCausalLM
        moe_class = Qwen2ForCausalLMWithRIM
    elif args.model_type == "qwen3":
        config = Qwen3WithRIMConfig.from_pretrained(
            args.base_model,
            num_experts=num_experts,
            expert_attn_size=args.key_size,
            experts_top_p=args.experts_top_p,
            router_aux_loss_coef=args.router_aux_loss_coef,
            output_expert_mask=True,
        )
        base_class = Qwen3ForCausalLM
        moe_class = Qwen3ForCausalLMWithRIM
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    print(f"Merging {num_experts} expert models into a RIM-based MoE model...")
    print("Base model:", args.base_model)
    print("Specialists:", args.specialists)
    print("Saving to:", args.output_dir)
    print("Model config:", config)

    moe_model = create_moe_from_specialists(
        base_model=args.base_model,
        specialists=args.specialists,
        moe_config=config,
        base_class=base_class,
        moe_class=moe_class,
    )

    print(f"Saving merged MoE model to {args.output_dir}...")
    moe_model.save_pretrained(args.output_dir, safe_serialization=True)
    config.save_pretrained(args.output_dir)
    
    total_params = sum(p.numel() for p in moe_model.parameters())
    print(f"Merge completed. Total parameters in merged MoE model: {total_params:,}")

if __name__ == "__main__":
    main()
