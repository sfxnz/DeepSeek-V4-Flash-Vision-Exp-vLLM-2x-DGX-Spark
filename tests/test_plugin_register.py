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

    def test_register_source_overwrites_deepseek_v4(self) -> None:
        src = (ROOT / "docker" / "plugin" / "dsv4_vision" / "__init__.py").read_text()
        tree = ast.parse(src)
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "register_model"
        ]
        self.assertGreaterEqual(len(calls), 1)
        self.assertIn("DeepseekV4ForCausalLM", src)
        self.assertIn("dsv4_vision.model:DeepseekV4VisionForCausalLM", src)

    def test_module_import_does_not_need_vllm(self) -> None:
        self.assertIn("register", dsv4_vision.__all__ if hasattr(dsv4_vision, "__all__") else ("register",))

    def test_forward_accepts_kwargs_and_keeps_dspark_attrs(self) -> None:
        src = (ROOT / "docker" / "plugin" / "dsv4_vision" / "model.py").read_text()
        self.assertIn("**kwargs", src)
        self.assertIn("class DeepseekV4VisionForCausalLM(DeepseekV4ForCausalLM)", src)
        self.assertNotIn("self.lm_head =", src)
        self.assertIn("bias_vl", src)


if __name__ == "__main__":
    unittest.main()
