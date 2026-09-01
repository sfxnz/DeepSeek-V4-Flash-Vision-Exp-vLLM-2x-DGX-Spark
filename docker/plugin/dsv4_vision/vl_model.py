"""Vision-Exp as a multimodal subclass of B12X DeepseekV4ForCausalLM.

Nested `language_model` wrapping (upstream vLLM #54566) double-maps
`ffn.gate.bias` on this B12X tree and KeyErrors. Inheritance keeps the
working text load path, DSpark `lm_head` / `get_mtp_target_hidden_states`,
and adds the processor + embedding merge the OpenAI server requires.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY

from .mm_preprocess import (
    IMAGE_PLACEHOLDER,
    IMAGE_SENTINEL_BASE_ID,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
    image_sentinel_mask,
)
from .vision import DeepseekV4Aligner, DeepseekV4ViT


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class DeepseekV4ForConditionalGeneration(DeepseekV4ForCausalLM, SupportsMultiModal):
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return IMAGE_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality!r}")

    def get_language_model(self):
        # SupportsMultiModal.get_language_model walks children and can
        # miss CausalLM.model. DSpark Eagle3 needs that attribute.
        return self

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.multimodal_config = model_config.multimodal_config
        image_enabled = (
            int(getattr(config, "vision_n_layers", 0) or 0) > 0
            and self.multimodal_config is not None
            and self.multimodal_config.get_limit_per_prompt("image") > 0
        )
        self.vision: DeepseekV4ViT | None = None
        self.aligner: DeepseekV4Aligner | None = None
        self.image_start: nn.Parameter | None = None
        self.image_end: nn.Parameter | None = None
        self.image_newline: nn.Parameter | None = None
        self.image_pad: nn.Parameter | None = None
        if not image_enabled:
            return
        self.vision = DeepseekV4ViT(config)
        self.aligner = DeepseekV4Aligner(config)
        for name in ("image_start", "image_end", "image_newline", "image_pad"):
            setattr(
                self,
                name,
                nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float32)),
            )
        self.vision.to(dtype=model_config.dtype)
        self.aligner.to(dtype=model_config.dtype)

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        patches = kwargs.pop("patches", None)
        if patches is None:
            return None
        vit_grid = kwargs.pop("vit_grid", None)
        llm_grid = kwargs.pop("llm_grid", None)
        perm = kwargs.pop("perm", None)
        assert vit_grid is not None and llm_grid is not None and perm is not None
        return {
            "patches": patches,
            "vit_grid": vit_grid,
            "llm_grid": llm_grid,
            "perm": perm,
        }

    def _process_image_input(
        self,
        patches: torch.Tensor,
        vit_grid: torch.Tensor,
        llm_grid: torch.Tensor,
        perm: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        assert self.vision is not None and self.aligner is not None
        patches = patches.to(self.aligner.w1.weight.dtype)
        embeds: list[torch.Tensor] = []
        vit_offset = 0
        llm_offset = 0
        for (n_vit_h, n_vit_w), (n_llm_h, n_llm_w) in zip(
            vit_grid.tolist(), llm_grid.tolist(), strict=True
        ):
            n_vit = n_vit_h * n_vit_w
            n_llm = n_llm_h * n_llm_w
            image_embeds = self.aligner(
                self.vision(patches[vit_offset : vit_offset + n_vit], n_vit_h, n_vit_w),
                n_vit_h,
                n_vit_w,
            )
            item_perm = perm[llm_offset : llm_offset + n_llm].to(image_embeds.device)
            embeds.append(image_embeds[item_perm])
            vit_offset += n_vit
            llm_offset += n_llm
        return tuple(embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None or self.vision is None:
            return []
        return self._process_image_input(
            image_input["patches"],
            image_input["vit_grid"],
            image_input["llm_grid"],
            image_input["perm"],
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        inputs_embeds = super().embed_input_ids(input_ids)
        if self.image_start is not None:
            sentinel_mask = image_sentinel_mask(input_ids)
            if is_multimodal is not None:
                sentinel_mask = sentinel_mask & ~is_multimodal.to(input_ids.device)
            table = torch.stack(
                [
                    self.image_start,
                    self.image_pad,
                    self.image_pad,
                    self.image_newline,
                    self.image_end,
                ]
            ).to(inputs_embeds.dtype)
            idx = (input_ids - IMAGE_SENTINEL_BASE_ID).clamp(0, 4)
            inputs_embeds = torch.where(
                sentinel_mask.unsqueeze(-1), table[idx], inputs_embeds
            )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.vision is None:
            return super().load_weights(weights)
        leftover = []
        stacked = {"vision.": self.vision, "aligner.": self.aligner}
        params = {
            "image_start": self.image_start,
            "image_end": self.image_end,
            "image_newline": self.image_newline,
            "image_pad": self.image_pad,
        }
        for name, tensor in weights:
            if name in params:
                params[name].data.copy_(tensor)
                continue
            routed = False
            for prefix, module in stacked.items():
                if name.startswith(prefix):
                    AutoWeightsLoader(module).load_weights(
                        [(name[len(prefix) :], tensor)]
                    )
                    routed = True
                    break
            if not routed:
                leftover.append((name, tensor))
        leftover = [
            pair
            for pair in leftover
            if not _skip_unmapped_gate(pair[0], self.config)
        ]
        return super().load_weights(leftover)


def _skip_unmapped_gate(name: str, config) -> bool:
    # Hash MoE layers register tid2eid and leave e_score_correction_bias as
    # None. The B12X mapper still rewrites ffn.gate.bias onto that missing
    # name. bias_vl has no mapper at all.
    if name.endswith("ffn.gate.bias_vl"):
        return True
    if name.endswith("ffn.gate.bias"):
        parts = name.split(".")
        num_hash = int(getattr(config, "num_hash_layers", 0) or 0)
        if (
            len(parts) >= 5
            and parts[0] == "layers"
            and parts[1].isdigit()
            and int(parts[1]) < num_hash
        ):
            return True
    return False
