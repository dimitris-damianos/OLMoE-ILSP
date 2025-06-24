from transformers import OlmoeForCausalLM, AutoTokenizer, BitsAndBytesConfig
from megablocks.megablocks.layers.router import SimilarityRouter
from megablocks.megablocks.layers.arguments import Arguments
import torch
from megablocks.layers import common


def test_olmoe():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924",
                                             quantization_config=quantization_config,
                                             device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    inputs = tokenizer("Explain the concept of MoE and its benefits over dense LLMs.", return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = model.generate(**inputs, max_length=256)
    print(tokenizer.decode(out[0]))
    
    
def test_router():
    args = Arguments(
        hidden_size=1024,
        latent_size=128,
        moe_num_experts=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    router = SimilarityRouter(args)
    x = torch.randn(2, 10, args.hidden_size, device=args.device).to(common.dtype(args))
    y = router(x)
    print(y)
    
if __name__ == "__main__":
    # test_olmoe()
    test_router()