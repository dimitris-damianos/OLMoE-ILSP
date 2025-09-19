from model import Qwen3ForCausalLMWithRIM, Qwen3ModelWithRIM
from config import Qwen3WithRIMConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoModel

print('Registering config')
AutoConfig.register("qwen3moe_with_rim", Qwen3WithRIMConfig)
print('Registering Model')
AutoModel.register(Qwen3WithRIMConfig, Qwen3ModelWithRIM)
print('Registering Causal Model')
AutoModelForCausalLM.register(Qwen3WithRIMConfig, Qwen3ForCausalLMWithRIM)