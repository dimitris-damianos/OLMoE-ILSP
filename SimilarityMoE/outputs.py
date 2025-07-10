from transformers.modeling_outputs import (
    MoeModelOutputWithPast, MoeCausalLMOutputWithPast,
    BaseModelOutputWithPast, CausalLMOutputWithPast,
)
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch

@dataclass
class MoeModelOutputWithPastAndExpertMask(MoeModelOutputWithPast):
    expert_mask: Optional[Tuple[bool]] = None
    
@dataclass
class MoeCausalLMOutputWithPastAndExpertMask(MoeCausalLMOutputWithPast):
    expert_mask: Optional[Tuple[bool]] = None

@dataclass
class BaseModelOutputWithPastandMoe(BaseModelOutputWithPast):
    expert_mask: Optional[Tuple[bool]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    aux_loss: Optional[torch.FloatTensor] = None
    
@dataclass
class CausalLMOutputWithPastandMoe(CausalLMOutputWithPast):
    expert_mask: Optional[Tuple[bool]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    aux_loss: Optional[torch.FloatTensor] = None