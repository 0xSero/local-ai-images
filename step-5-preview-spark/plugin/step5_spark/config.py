# SPDX-License-Identifier: Apache-2.0
"""HF config classes for Step-5 (registered with transformers + vLLM by the plugin; no trust-remote-code).

model_type "step5" (vision-language wrapper) holds:
  text_config   model_type "step5_text"  (Step-3.5 family text model + SSMax + fp32 residual, see step5.py)
  vision_config model_type "step5_vision" (perception encoder, same layout as Step-3.7)
"""
from transformers.configuration_utils import PretrainedConfig

from vllm.transformers_utils.configs.step3p5 import Step3p5Config


class Step5TextConfig(Step3p5Config):
    model_type = "step5_text"

    def __init__(self, fp32_residual_connection: bool = True, sparse_config: dict | None = None,
                 partial_rotary_factors: list[float] | None = None, moe_layers_enum: str | None = None,
                 moe_router_activation: str = "sigmoid", **kwargs):
        # Step-5 routes with sigmoid scores + router_bias for selection, renormalised, x moe_router_scaling_factor
        # (exllamav3 "dots" router); Step3p5Config would default to softmax when the key is absent.
        layer_types = kwargs.get("layer_types")
        super().__init__(moe_router_activation=moe_router_activation, **kwargs)
        # Step3p5Config truncates layer_types to num_hidden_layers; keep the entries of the MTP layers (92-94)
        if layer_types is not None:
            self.layer_types = list(layer_types)
        self.fp32_residual_connection = fp32_residual_connection
        self.sparse_config = sparse_config
        self.partial_rotary_factors = partial_rotary_factors
        self.moe_layers_enum = moe_layers_enum


class Step5VisionConfig(PretrainedConfig):
    model_type = "step5_vision"

    def __init__(self, image_size: int = 728, patch_size: int = 14, width: int = 1536, layers: int = 47,
                 heads: int = 16, hidden_act: str = "quick_gelu", ls_init_value: float = 0.1,
                 mlp_ratio: float = 8960 / 1536, output_dim: int | None = None, use_cls_token: bool = False,
                 use_abs_posemb: bool = True, use_rope2d: bool = True, use_ln_pre: bool = True,
                 use_ln_post: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.image_size, self.patch_size, self.width, self.layers, self.heads = image_size, patch_size, width, layers, heads
        self.hidden_act, self.ls_init_value, self.mlp_ratio, self.output_dim = hidden_act, ls_init_value, mlp_ratio, output_dim
        self.use_cls_token, self.use_abs_posemb, self.use_rope2d = use_cls_token, use_abs_posemb, use_rope2d
        self.use_ln_pre, self.use_ln_post = use_ln_pre, use_ln_post
        # Step3VL processing reads these names
        self.hidden_size = width
        self.num_hidden_layers = layers
        self.num_attention_heads = heads


class Step5Config(PretrainedConfig):
    model_type = "step5"
    sub_configs = {"text_config": Step5TextConfig, "vision_config": Step5VisionConfig}

    def __init__(self, text_config=None, vision_config=None, image_token_id: int = 128001,
                 understand_projector_stride: int = 2, projector_bias: bool = False, image_token_len: int = 169,
                 im_start_token: str = "<im_start>", im_end_token: str = "<im_end>", im_patch_token: str = "<im_patch>",
                 patch_token_len: int = 81, use_im_start_end: bool = True, vision_select_layer: int = -1, **kwargs):
        if isinstance(text_config, dict):
            text_config = Step5TextConfig(**{k: v for k, v in text_config.items() if k != "model_type"})
        if isinstance(vision_config, dict):
            vision_config = Step5VisionConfig(**{k: v for k, v in vision_config.items() if k != "model_type"})
        self.text_config = text_config or Step5TextConfig()
        self.vision_config = vision_config or Step5VisionConfig()
        self.image_token_id = image_token_id
        self.image_token_index = image_token_id   # read by the V1 runner's multimodal/MTP path
        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.image_token_len = image_token_len
        self.im_start_token, self.im_end_token, self.im_patch_token = im_start_token, im_end_token, im_patch_token
        self.patch_token_len = patch_token_len
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer
        super().__init__(**kwargs)

    def get_text_config(self, decoder=False, **kwargs):
        return self.text_config
