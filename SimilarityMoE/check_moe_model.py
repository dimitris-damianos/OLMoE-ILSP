import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

ckpt_path = "/leonardo_work/EUHPC_A06_067/moe_models/qwen3_0.6B-moe-merged-11_experts-SFT_trainable_router_stage1/checkpoint-200"

config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
print(f"\nLoaded config: {type(config)}")
print(f"   model_type: {config.model_type}")
print(f"   expert_attn_size: {config.expert_attn_size}")
print(f"   architectures: {config.architectures}")
print(f"   num_experts: {getattr(config, 'num_experts', 'N/A')}")

model = AutoModelForCausalLM.from_pretrained(ckpt_path, config=config, trust_remote_code=True)
print(f"\nLoaded model: {type(model)}")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
print(f"\nLoaded tokenizer: {type(tokenizer)}")

# dummy input
inputs = tokenizer("The meaning of life is", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
    print("\nForward pass OK — output shape:", outputs.logits.shape)
topk = torch.topk(outputs.logits[0, -1], k=5)
tokens = [tokenizer.decode([i]) for i in topk.indices]
print(f"\nTop-5 predictions: {tokens}")
