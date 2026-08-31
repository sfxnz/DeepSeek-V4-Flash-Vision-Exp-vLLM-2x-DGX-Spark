#!/usr/bin/env python3
"""Skip Vision-Exp gate tensors the B12X DSpark draft loader cannot bind."""
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py")
text = p.read_text()
old = '''                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
'''
new = '''                if name.endswith(".ffn.gate.bias_vl"):
                    continue
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                if name not in params_dict:
                    continue
                param = params_dict[name]
'''
if old not in text:
    raise SystemExit(f"patch target missing in {p}")
p.write_text(text.replace(old, new, 1))
