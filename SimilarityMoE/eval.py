import lm_eval
from model import Qwen2ForCausalLMWithRIM, Qwen3ForCausalLMWithRIM

PATH= '/leonardo_work/EUHPC_A06_067/moe_models/qwen3_0.6B-moe-merged-11_experts-SFT_trainable_router_stage1/checkpoint-100'

model = Qwen3ForCausalLMWithRIM.from_pretrained(
            PATH,
            local_files_only=True,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
            # config=config,
        )

results = lm_eval.simple_evaluate(
    model="hf",
    model_args={"pretrained": model,},
    tasks="gsm8k",
    log_samples=True,
    batch_size=16,
)