import liger_kernel.transformers.monkey_patch as mp
from transformers import PreTrainedModel
from types import MethodType

def apply_liger_kernel_to_qwen2_rim(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2 models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward

    if rope:
        modeling_qwen2.apply_rotary_pos_emb = mp.liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen2.Qwen2RMSNorm = mp.LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = mp.liger_cross_entropy

    if fused_linear_cross_entropy:       
        if model is not None:
            model.forward = MethodType(qwen2_lce_forward, model)
        else:
            modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward

    if swiglu:
        modeling_qwen2.Qwen2MLP = mp.LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen2ModelWithRIM = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            mp._patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                for mlp_expert in decoder_layer.mlp.experts:
                    mp._patch_swiglu_module(mlp_expert, mp.LigerSwiGLUMLP)
            if rms_norm:
                mp._patch_rms_norm_module(decoder_layer.input_layernorm)
                mp._patch_rms_norm_module(decoder_layer.post_attention_layernorm)

    print("Applied Liger kernels to Qwen2ModelWithRim.")


def apply_liger_kernel_to_qwen3_rim(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    from liger_kernel.transformers.model.qwen3 import lce_forward as qwen3_lce_forward

    if rope:
        modeling_qwen3.apply_rotary_pos_emb = mp.liger_rotary_pos_emb

    if rms_norm:
        modeling_qwen3.Qwen3RMSNorm = mp.LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = mp.liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(qwen3_lce_forward, model)
        else:
            modeling_qwen3.Qwen3ForCausalLM.forward = qwen3_lce_forward

    if swiglu:
        modeling_qwen3.Qwen3MLP = mp.LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen3ModelWithRIM = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            mp._patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                for mlp_expert in decoder_layer.mlp.experts:
                    mp._patch_swiglu_module(mlp_expert, mp.LigerSwiGLUMLP)
            if rms_norm:
                mp._patch_rms_norm_module(decoder_layer.input_layernorm)
                mp._patch_rms_norm_module(decoder_layer.post_attention_layernorm)

    print("Applied Liger kernels to Qwen3ModelWithRim.")


def apply_liger_to_model(model, qwen_type):
    if isinstance(model, PreTrainedModel):
        # Patch the model with liger kernels. Use the default kernel configurations.
        if qwen_type == 'qwen3':
            apply_liger_kernel_to_qwen3_rim(model=model)
        if qwen_type == 'qwen2':
            apply_liger_kernel_to_qwen2_rim(model=model)
    
    elif hasattr(model, "get_base_model") and isinstance(model.get_base_model(), PreTrainedModel):
        # Patch the base model with liger kernels where model is a PeftModel. Use the default kernel configurations.
        if qwen_type == 'qwen3':
            apply_liger_kernel_to_qwen3_rim(model=model.get_base_model())
        if qwen_type == 'qwen2':
            apply_liger_kernel_to_qwen2_rim(model=model.get_base_model())
    
    else:
        print(
            "The model is not an instance of PreTrainedModel. No liger kernels will be applied."
        )