# merge_peft_adapters.py

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.model_dir

    print(f"Loading LoRA model from {args.model_dir}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    print("Merging LoRA adapters...")
    merged_model = model.merge_and_unload()

    config = merged_model.config
    config.save_pretrained(output_dir)

    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
