"""Vision-Exp wrapper around vLLM DeepseekV4ForCausalLM.

Loads official `vision.*` / `aligner.*` / image sentinel tensors and merges
ViT embeddings on prefill. Language decode stays on the B12X DeepSeek V4 path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM

from .vision import Aligner, ViT


def _vision_args(hf: Any) -> SimpleNamespace:
    return SimpleNamespace(
        vision_n_layers=int(getattr(hf, "vision_n_layers", 0) or 0),
        vision_dim=int(getattr(hf, "vision_dim", 1024)),
        vision_n_heads=int(getattr(hf, "vision_n_heads", 16)),
        vision_inter_dim=int(getattr(hf, "vision_inter_dim", 2816)),
        vision_patch_size=int(getattr(hf, "vision_patch_size", 14)),
        vision_rope_theta=float(getattr(hf, "vision_rope_theta", 10000.0)),
        vision_downsample_ratio=int(getattr(hf, "vision_downsample_ratio", 3)),
        vision_max_n_token=int(getattr(hf, "vision_max_n_token", 384)),
        vision_min_pixels=int(getattr(hf, "vision_min_pixels", 147456)),
        vision_max_wh_ratio=int(getattr(hf, "vision_max_wh_ratio", 8)),
        dim=int(getattr(hf, "hidden_size", 4096)),
        num_hash_layers=int(getattr(hf, "num_hash_layers", 0) or 0),
    )


class DeepseekV4VisionForCausalLM(DeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hf = vllm_config.model_config.hf_config
        args = _vision_args(hf)
        self._vision_args = args
        if args.vision_n_layers <= 0:
            self.vision = None
            self.aligner = None
            return
        self.vision = ViT(args)
        self.aligner = Aligner(args)
        self.image_start = nn.Parameter(torch.empty(args.dim))
        self.image_end = nn.Parameter(torch.empty(args.dim))
        self.image_newline = nn.Parameter(torch.empty(args.dim))
        self.image_pad = nn.Parameter(torch.empty(args.dim))

    def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Spec-decode extra kwargs must not TypeError. lm_head and
        # get_mtp_target_hidden_states stay on the parent class.
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def load_weights(self, weights):
        if self.vision is None:
            return super().load_weights(weights)
        leftover = []
        stacked = {
            "vision.": self.vision,
            "aligner.": self.aligner,
        }
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
            if not _skip_unmapped_gate(pair[0], self._vision_args)
        ]
        return super().load_weights(leftover)


def _skip_unmapped_gate(name: str, args: SimpleNamespace) -> bool:
    # Hash MoE layers register tid2eid and leave e_score_correction_bias as
    # None. The B12X mapper still rewrites ffn.gate.bias onto that missing
    # name. bias_vl has no mapper at all.
    if name.endswith("ffn.gate.bias_vl"):
        return True
    if name.endswith("ffn.gate.bias"):
        parts = name.split(".")
        if (
            len(parts) >= 5
            and parts[0] == "layers"
            and parts[1].isdigit()
            and int(parts[1]) < args.num_hash_layers
        ):
            return True
    return False
