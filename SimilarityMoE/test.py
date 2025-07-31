from transformers import OlmoeForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from megablocks.megablocks.layers.router import SimilarityRouter, LearnedRouter
# from megablocks.megablocks.layers.arguments import Arguments
import torch
from torch import nn
# from megablocks.layers import common


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
    # print(tokenizer.decode(out[0]))
    
    
# def test_router():
#     args = Arguments(
#         hidden_size=1024,
#         latent_size=128,
#         moe_num_experts=4,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         moe_top_k=2,
#     )
#     router = SimilarityRouter(args)
#     x = torch.randn(2, 10, args.hidden_size, device=args.device).to(common.dtype(args))
#     # y = router(x)
#     # print(y)
    
#     l_router = LearnedRouter(args)
#     scores, weights, indices = l_router(x)
#     print(f'Scores: shape {scores.shape}, {scores}')
#     print(f'Weights: shape {weights.shape}, {weights}')
#     print(f'Indices: shape {indices.shape}, {indices}')


def test_hf_moe_block():
    from model import OlmoeSimilarityMoeBlock, OlmoeSparseMoeBlock
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained("allenai/OLMoE-1B-7B-0924")
    # original_mlp = OlmoeSparseMoeBlock(config)
    # x = torch.randn(2, 10, config.hidden_size)
    # hm, logits = original_mlp(x)
    # print(f'Original MLP output shape: {hm.shape}, {hm}')
    # print(f'Original MLP logits shape: {logits.shape}, {logits}')
    
    
    config.num_experts = 4  # Set the number of experts for the MoE block
    config.latent_size = 128  # Set the latent size for the MoE block
    config.num_experts_per_tok = 2  # Set the number of experts per token
    moe_block = OlmoeSimilarityMoeBlock(config)
    x = torch.randn(2, 10, config.hidden_size)
    print((f'Num of experts: {moe_block.num_experts}, '))
    print(f'Input shape: {x.shape},')
    vals, idx = moe_block(x)
    print(f'Output values shape: {vals.shape}, {vals}')
    print(f'Output indices shape: {idx.shape}, {idx}')

def const_weight_init(model,val=1.0):
    for name, param in model.named_parameters():
        nn.init.constant_(param, val)

def test_custom_moe():
    from model import (
        OlmoeMoeBlockWithRIM, OlmoeDecoderLayerWithRIM, OlmoeForCausalLMWithRIM,
        Qwen3MoeBlockWithRIM, Qwen3ForCausalLMWithRIM, Qwen2ForCausalLMWithRIM,  
    )
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    from transformers import AutoConfig
    from config import OlmoeWithRIMConfig, Qwen3WithRIMConfig, Qwen2WithRIMConfig
    
    config = Qwen2WithRIMConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    # config = Qwen3WithRIMConfig.from_pretrained("Qwen/Qwen3-0.6B")
    # config = OlmoeWithRIMConfig.from_pretrained("allenai/OLMoE-1B-7B-0924")
    config.num_experts = 8  # Set the number of experts for the MoE block
    config.num_experts_per_tok = 2 
    config.enable_comm = False  # Enable communication attention
    config.expert_attn_size = 64
    config.experts_top_p = 0.5  # Set the top_p for experts selection
    
    
    torch.manual_seed(42)  # For reproducibility
    # x = torch.randn(2, 10, config.hidden_size) 
    # print('testing MoE block with RIM...')
    # model = OlmoeMoeBlockWithRIM(config)
    # # model = Qwen3ForCausalLMWithRIM(config)
    # # const_weight_init(model,0.1)
    # h, l, mask = model(x,output_attentions=True,output_router_logits=True, output_expert_mask=True)
    # print('Hidden shape:', h.shape)
    # print('Logits shape:', l.shape)
    # print('Experts mask shape:', mask.shape)
    # print(mask)
    
    # print('testing custom OlmoeForCausalLM with RIM...')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # model = OlmoeForCausalLMWithRIM(config).to(DEVICE)
    # model = Qwen3ForCausalLMWithRIM(config).to(DEVICE)
    model = Qwen2ForCausalLMWithRIM(config).to(DEVICE)
    with open('model.txt', 'w') as f:
        f.write(str(model))
    inputs = {
        "input_ids": torch.randint(0, 1000, (2, 10)).to(DEVICE),  # Example input IDs
        "attention_mask": torch.ones(2, 10).to(DEVICE),  # Example attention mask
        'output_router_logits': True,
        'output_expert_mask': True,
    }
    
    outputs = model(**inputs)
    print('Model outputs:', outputs.expert_mask)
    print(outputs.aux_loss)

def test_merging():
    from utils import create_moe_from_specialists
    from config import Qwen2WithRIMConfig, Qwen3WithRIMConfig
    from transformers import Qwen2ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from model import Qwen2ForCausalLMWithRIM, Qwen3ForCausalLMWithRIM
    
    base_model = "Qwen/Qwen3-0.6B"
    specialists = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B"
    ]
    # get pretrained config
    moe_config = Qwen3WithRIMConfig.from_pretrained(base_model)
    
    # additional config parameters
    moe_config.num_experts = len(specialists)
    moe_config.expert_attn_size = 64
    # moe_config.output_hidden_states = True
    # moe_config.output_attentions = True
    moe_config.output_expert_mask = True
    moe_config.output_router_logits = True
    moe_config.router_aux_loss_coef = 0.1 
    moe_config.experts_top_p = 0.5 
    
    moe_model = create_moe_from_specialists(base_model, 
                                            specialists, 
                                            moe_config,
                                            base_class=Qwen3ForCausalLM,
                                            moe_class=Qwen3ForCausalLMWithRIM)
    print("MoE model created with specialists merged successfully.")
    # print(moe_model)

if __name__ == "__main__":
    # test_olmoe()
    # test_router()
    # test_hf_moe_block()
    # test_custom_moe()
    test_merging()