#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker" / "plugin"))

import dsv4_vision  # noqa: E402


class PluginRegisterTests(unittest.TestCase):
    def test_register_is_callable(self) -> None:
        self.assertTrue(callable(dsv4_vision.register))

    def test_register_source_installs_vl_class_not_causal_overwrite(self) -> None:
        src = (ROOT / "docker" / "plugin" / "dsv4_vision" / "__init__.py").read_text()
        tree = ast.parse(src)
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "register_model"
        ]
        self.assertGreaterEqual(len(calls), 1)
        self.assertIn("DeepseekV4ForConditionalGeneration", src)
        self.assertIn("dsv4_vision.vl_model:DeepseekV4ForConditionalGeneration", src)
        self.assertIn("install_arch_convertor", src)
        self.assertIn("install_encoding_hooks", src)
        self.assertNotIn("DeepseekV4ForCausalLM", src)

    def test_module_import_does_not_need_vllm(self) -> None:
        self.assertIn("register", dsv4_vision.__all__ if hasattr(dsv4_vision, "__all__") else ("register",))

    def test_vl_wrapper_keeps_dspark_and_drops_bias_vl(self) -> None:
        src = (ROOT / "docker" / "plugin" / "dsv4_vision" / "vl_model.py").read_text()
        self.assertIn("class DeepseekV4ForConditionalGeneration", src)
        self.assertIn("SupportsMultiModal", src)
        self.assertIn("DeepseekV4ForCausalLM", src)
        self.assertIn("bias_vl", src)
        self.assertIn("def _skip_unmapped_gate", src)
        self.assertIn("num_hash_layers", src)
        self.assertIn("MULTIMODAL_REGISTRY.register_processor", src)
        self.assertIn("embed_multimodal", src)

    def test_hooks_rewrite_deepseek_v4_architecture(self) -> None:
        src = (ROOT / "docker" / "plugin" / "dsv4_vision" / "hooks.py").read_text()
        self.assertIn("DeepseekV4ForConditionalGeneration", src)
        self.assertIn("MODEL_ARCH_CONFIG_CONVERTORS", src)
        self.assertIn("flatten_content_blocks", src)
        self.assertIn("IMAGE_PLACEHOLDER", src)

    def test_processor_forces_mm_only_path(self) -> None:
        src = (ROOT / "docker" / "plugin" / "dsv4_vision" / "mm_preprocess.py").read_text()
        self.assertIn("def _apply_hf_processor_main", src)
        self.assertIn("enable_hf_prompt_update=False", src)
        self.assertIn("def _apply_hf_processor_text_only", src)
        self.assertIn("add_special_tokens=False", src)
        self.assertIn("def _apply_hf_processor_mm_only", src)
        self.assertNotIn("call_hf_processor_mm_only", src)
        self.assertIn("MULTIMODAL_REGISTRY", (ROOT / "docker" / "plugin" / "dsv4_vision" / "vl_model.py").read_text())


if __name__ == "__main__":
    unittest.main()
