from transformers import OlmoeConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

class OlmoeWithRIMConfig(OlmoeConfig):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.expert_attn_size = kwargs.get("key_size", 512)
        self.enable_comm = kwargs.get("enable_comm", False)  # Enable communication attention
        self.expert_comm_size = kwargs.get("enable_comm", False)  # Enable communication attention
        self.output_expert_mask = kwargs.get("output_expert_mask", False)  # Output expert mask
        
        
        
class Qwen3WithRIMConfig(Qwen3Config):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.expert_attn_size = kwargs.get("key_size", 512)
        self.output_expert_mask = kwargs.get("output_expert_mask", False)  # Output expert mask
        self.output_router_logits = kwargs.get("output_router_logits", False)
        self.output_expert_mask = kwargs.get("output_expert_mask", False)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 0.1)  # Coefficient for router auxiliary loss
        self.experts_top_p = kwargs.get("experts_top_p", 0.5)