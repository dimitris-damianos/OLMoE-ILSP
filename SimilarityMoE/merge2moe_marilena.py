#!/usr/bin/env python3
"""
Merge Qwen specialists into a single RIM-based MoE model (Qwen2 or Qwen3).

Key fixes vs. your versions:
- NO AutoConfig.register() (avoids collisions with built-in "qwen2"/"qwen3").
- Forces a UNIQUE model_type from your custom config classes (e.g., "qwen2_with_rim", "qwen3_with_rim").
- Writes `architectures` and `auto_map` so Auto* loaders can resolve your custom classes in a fresh process.
- Saves tokenizer next to the model.
- Prints total params and final config fields for sanity checks.

Reload later with:
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    tok   = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True)
"""

import argparse
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from config import Qwen2WithRIMConfig, Qwen3WithRIMConfig
from model import Qwen3ForCausalLMWithRIM
# Qwen2 RIM class may be optional in your repo
try:
    from model import Qwen2ForCausalLMWithRIM
except Exception:
    Qwen2ForCausalLMWithRIM = None

from utils import create_moe_from_specialists


def _qualname(obj) -> str:
    """Return fully-qualified import path for an object (module + class)."""
    return f"{obj.__module__}.{obj.__name__}"


def _apply_reload_metadata(config, config_cls, model_cls):
    """
    Make the saved config self-describing so Auto* can load it later (with trust_remote_code=True),
    without any runtime registration.
    """
    # Ensure we keep the class-defined unique model_type (e.g., "qwen3_with_rim")
    config.model_type = type(config).model_type

    # Tell HF which model class to instantiate
    config.architectures = [model_cls.__name__]

    # Provide mapping so AutoConfig / AutoModelForCausalLM can import your classes
    config.auto_map = {
        "AutoConfig": _qualname(config_cls),
        "AutoModelForCausalLM": _qualname(model_cls),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge Qwen specialists into a RIM-based MoE model.")
    p.add_argument("--base_model", type=str, required=True, help="Path or HF id of the base pretrained model.")
    p.add_argument("--specialists", type=str, nargs="+", required=True, help="Paths to fine-tuned specialist models.")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save the merged MoE model.")
    p.add_argument("--model_type", type=str, choices=["qwen2", "qwen3"], required=True, help="Model family.")
    p.add_argument("--key_size", type=int, default=512, help="Attention key size for RIMs.")
    p.add_argument("--experts_top_p", type=float, default=0.5, help="Top-p threshold for expert routing.")
    p.add_argument("--router_aux_loss_coef", type=float, default=0.1, help="Router auxiliary loss coefficient.")
    p.add_argument("--output_expert_mask", action="store_true", help="Return a binary mask of selected experts.")
    p.add_argument("--output_router_logits", action="store_true", help="Return router logits for analysis.")
    p.add_argument("--detach_null_states", action="store_true", help="Detach null states in RIMs.")
    p.add_argument("--use_latent_states", action="store_true", help="Use latent states for experts.")
    p.add_argument("--use_dynamic_routing", action="store_true", help="Enable top-p routing; otherwise top-k.")
    p.add_argument("--experts_top_k", type=int, default=2, help="Top-k experts to select when not using top-p.")
    return p.parse_args()


def main():
    args = parse_args()

    num_experts: int = len(args.specialists)
    if num_experts < 1:
        raise ValueError("You must pass at least one specialist checkpoint.")

    # Build the custom config with RIM/MoE knobs
    if args.model_type == "qwen2":
        if Qwen2ForCausalLMWithRIM is None:
            raise RuntimeError("Qwen2ForCausalLMWithRIM not importable; install/define it or use --model_type qwen3.")
        config = Qwen2WithRIMConfig.from_pretrained(
            args.base_model,
            num_experts=num_experts,
            expert_attn_size=args.key_size,
            output_expert_mask=args.output_expert_mask,
            output_router_logits=args.output_router_logits,
            router_aux_loss_coef=args.router_aux_loss_coef,
            experts_top_p=args.experts_top_p,
            # (enable if your Qwen2 RIM config supports these)
            # detach_null_states=args.detach_null_states,
            # use_latent_states=args.use_latent_states,
        )
        base_class = Qwen2ForCausalLM
        moe_class = Qwen2ForCausalLMWithRIM
        config_cls = Qwen2WithRIMConfig

    elif args.model_type == "qwen3":
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
        config_cls = Qwen3WithRIMConfig

    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    # Make sure the saved config points to your custom classes
    _apply_reload_metadata(config, config_cls, moe_class)

    print(f"Merging {num_experts} expert models into a RIM-based MoE model...\n")
    print("Base model:        ", args.base_model)
    print("Specialists:       ", args.specialists)
    print("Saving to:         ", args.output_dir)
    print("Config model_type: ", config.model_type)
    print("Architectures:     ", getattr(config, "architectures", None))
    print("Auto-map:          ", getattr(config, "auto_map", None))

    # Build the MoE by combining specialists under the RIM router
    moe_model = create_moe_from_specialists(
        base_model=args.base_model,
        specialists=args.specialists,
        moe_config=config,
        base_class=base_class,
        moe_class=moe_class,
    )

    # Ensure the custom config rides along with the model
    moe_model.config = config

    print(f"\nSaving merged MoE model to {args.output_dir}...")
    moe_model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save tokenizer next to the model (best-effort)
    try:
        tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        tok.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"Warning: tokenizer save failed: {e}")

    # Useful diagnostics
    try:
        total_params = sum(p.numel() for p in moe_model.parameters())
        print(f"Total parameters in merged MoE model: {total_params:,}")
    except Exception as e:
        print(f"Warning: could not compute parameter count: {e}")

    print("\nReload test hint (in a fresh process):")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}', trust_remote_code=True)")
    print(f"  tok   = AutoTokenizer.from_pretrained('{args.output_dir}', use_fast=True)")


if __name__ == "__main__":
    main()

