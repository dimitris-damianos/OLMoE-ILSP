from transformers import OlmoeConfig

class OlmoeWithRIMConfig(OlmoeConfig):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.expert_attn_size = kwargs.get("key_size", 512)
        self.enable_comm = kwargs.get("enable_comm", False)  # Enable communication attention
        self.expert_comm_size = kwargs.get("enable_comm", False)  # Enable communication attention
        
        self.output_expert_mask = kwargs.get("output_expert_mask", False)  # Output expert mask