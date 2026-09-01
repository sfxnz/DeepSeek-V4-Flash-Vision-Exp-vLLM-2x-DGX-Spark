"""Install convertor + encoding hooks so vLLM treats Vision-Exp as multimodal.

The HF checkpoint keeps `architectures: [DeepseekV4ForCausalLM]`. Stock B12X
vLLM has no `deepseek_v4` convertor, so it loads the text class and the OpenAI
server returns 400 "is not a multimodal model". The B12X tokenizer also leaves
`image_url` blocks as Python lists, so even a VL class would never see a
`<｜deepseek_image｜>` placeholder.

Both hooks mutate in-process modules. `register()` runs from
`EngineArgs.__post_init__` before `ModelConfig` is built.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def flatten_content_blocks(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") in ("image", "image_url"):
            parts.append(IMAGE_PLACEHOLDER)
        elif block.get("type") == "text":
            parts.append(block.get("text", "") or "")
        else:
            raise ValueError(f"Unsupported content block type: {block.get('type')!r}")
    return "".join(parts)


def install_encoding_hooks() -> None:
    from vllm.tokenizers import deepseek_v4_encoding as enc

    if getattr(enc, "_dsv4_vision_flatten", False):
        return
    enc.IMAGE_PLACEHOLDER = IMAGE_PLACEHOLDER
    enc.flatten_content_blocks = flatten_content_blocks

    orig_merge = enc.merge_tool_messages

    def merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        for msg in messages:
            msg = copy.deepcopy(msg)
            if msg.get("role") == "user":
                msg["content"] = flatten_content_blocks(msg.get("content", ""))
            flattened.append(msg)
        return orig_merge(flattened)

    enc.merge_tool_messages = merge_tool_messages
    enc._dsv4_vision_flatten = True


def install_arch_convertor() -> None:
    from transformers import PretrainedConfig

    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )

    class DeepseekV4ModelArchConfigConvertor(ModelArchConfigConvertorBase):
        def __init__(
            self,
            hf_config: PretrainedConfig,
            hf_text_config: PretrainedConfig,
        ):
            if (
                getattr(hf_config, "vision_n_layers", 0) > 0
                and getattr(hf_config, "architectures", None)
                == ["DeepseekV4ForCausalLM"]
                and not getattr(hf_config, "_dsv4_vl_inner", False)
            ):
                hf_config.architectures = ["DeepseekV4ForConditionalGeneration"]
            super().__init__(hf_config, hf_text_config)

        def is_mm_prefix_lm(self, supports_multimodal: bool = True) -> bool:
            if not supports_multimodal:
                return False
            return getattr(self.hf_config, "vision_n_layers", 0) > 0

    MODEL_ARCH_CONFIG_CONVERTORS["deepseek_v4"] = DeepseekV4ModelArchConfigConvertor
