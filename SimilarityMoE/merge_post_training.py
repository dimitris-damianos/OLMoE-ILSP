import sys
sys.path.append("../sft_experts")

import argparse
import os
import json
import shutil
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from model import Qwen3ForCausalLMWithRIM
from config import Qwen3WithRIMConfig

def verify_model_loading(model_dir):
    """Try loading the model to verify it works correctly."""
    print(f"\nVerifying model loading from {model_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = Qwen3ForCausalLMWithRIM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        print("✓ Success! Model loads correctly with custom Qwen3ForCausalLMWithRIM class")
        
        # Optional: Try a quick inference to verify functionality
        try:
            input_text = "What is 2+2?"
            inputs = tokenizer(input_text, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, max_new_tokens=10)
            output_text = tokenizer.decode(outputs[0])
            print(f"Model inference test: '{input_text}' -> '{output_text}'")
        except Exception as e:
            print(f"Warning: Inference test failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model with custom class: {e}")
        print("Trying with AutoModelForCausalLM as fallback...")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
            print("⚠️ Model loads with AutoModelForCausalLM but not with custom class")
        except Exception as e:
            print(f"❌ Model fails to load with any class: {e}")
        
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with LoRA adapters")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for merged model")
    parser.add_argument("--base_model_dir", type=str, default=None, 
                      help="Optional: path to base model if different from adapter's base_model.safetensors")
    parser.add_argument("--skip_verification", action="store_true", help="Skip model verification step")
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.model_dir}_merged"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model config from {args.model_dir}...")
    # First load the config to determine the model type
    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}, checking adapter_config.json")
        # Try to get base model path from adapter config
        adapter_config_path = os.path.join(args.model_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path")
                if base_model_path and not args.base_model_dir:
                    args.base_model_dir = base_model_path
                    print(f"Using base model from adapter config: {args.base_model_dir}")
    
    # Determine the base model path
    base_model_dir = args.base_model_dir or args.model_dir
    
    print(f"Loading base model from {base_model_dir}...")
    try:
        # Try loading the model with custom class first
        base_model = Qwen3ForCausalLMWithRIM.from_pretrained(
            base_model_dir,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("Successfully loaded model with custom Qwen3ForCausalLMWithRIM class")
    except Exception as e:
        print(f"Error loading with custom class: {e}")
        print("Falling back to AutoModelForCausalLM...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    # Check if we need to convert the model to PEFT format
    if os.path.exists(os.path.join(args.model_dir, "adapter_config.json")):
        print(f"Loading adapters from {args.model_dir}...")
        try:
            # Load the model with adapters
            model = PeftModel.from_pretrained(
                base_model,
                args.model_dir,
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
            print("Successfully loaded PEFT adapters")
        except Exception as e:
            print(f"Error loading adapters with PeftModel: {e}")
            print("Trying AutoPeftModelForCausalLM instead...")
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model_dir,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
    else:
        print("No adapter config found, assuming this is already a full model")
        model = base_model

    # Merge adapters if this is a PEFT model
    if hasattr(model, "merge_and_unload"):
        print("Merging adapters...")
        merged_model = model.merge_and_unload()
    else:
        print("No adapters to merge, using the model as-is")
        merged_model = model

    # Ensure we retain the custom config properties
    if hasattr(base_model, "config"):
        print("Copying config attributes from base model...")
        for attr in ["num_experts", "top_k", "top_p", "detach_null_states", 
                    "use_latent_states", "use_dynamic_routing", "expert_attn_size"]:
            if hasattr(base_model.config, attr):
                setattr(merged_model.config, attr, getattr(base_model.config, attr))
                print(f"  Set {attr} = {getattr(base_model.config, attr)}")

    # Set proper config for generation
    merged_model.config.use_cache = True
    
    # Fix the auto_map to point to the correct modules
    merged_model.config.auto_map = {
        "AutoConfig": "config.Qwen3WithRIMConfig",
        "AutoModelForCausalLM": "model.Qwen3ForCausalLMWithRIM"
    }
    merged_model.config.model_type = "qwen3moe_with_rim"
    print("Updated auto_map in config to use local model implementation")
    
    # Save the merged model
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    
    # Copy the custom model implementation files to the output directory
    print("Copying custom model implementation files...")
    model_files = ["model.py", "config.py"]
    for file in model_files:
        src_file = os.path.join(os.path.dirname(__file__), file)
        dst_file = os.path.join(output_dir, file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied {file} to output directory")
        else:
            print(f"Warning: Could not find source file {src_file}")
    
    # Save the tokenizer
    try:
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
        print("Trying to load tokenizer from base model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"Error loading base tokenizer: {e}")
    
    print("Model merging complete!")
    
    # Verify the model can be loaded correctly
    if not args.skip_verification:
        verification_result = verify_model_loading(output_dir)
        if verification_result:
            print("\n✅ Model was successfully merged and can be loaded correctly!")
        else:
            print("\n⚠️ Warning: The merged model may have issues when loading!")

if __name__ == "__main__":
    main()