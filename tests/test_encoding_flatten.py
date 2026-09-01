#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "encoding"))
sys.path.insert(0, str(ROOT / "docker" / "plugin"))

from encoding_dsv4 import IMAGE_PLACEHOLDER, flatten_content_blocks, merge_tool_messages
from dsv4_vision.hooks import flatten_content_blocks as plugin_flatten


class EncodingFlattenTests(unittest.TestCase):
    def test_image_url_becomes_placeholder(self) -> None:
        content = [
            {"type": "text", "text": "What color?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
        ]
        flat = flatten_content_blocks(content)
        self.assertEqual(flat, "What color?" + IMAGE_PLACEHOLDER)
        self.assertEqual(plugin_flatten(content), flat)

    def test_plain_string_passes_through(self) -> None:
        self.assertEqual(flatten_content_blocks("hello"), "hello")

    def test_merge_flattens_user_image_blocks(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ],
            }
        ]
        merged = merge_tool_messages(messages)
        self.assertEqual(merged[0]["content"], "Describe this." + IMAGE_PLACEHOLDER)
        texts = [
            b.get("text", "")
            for b in merged[0]["content_blocks"]
            if b.get("type") == "text"
        ]
        self.assertTrue(any(IMAGE_PLACEHOLDER in t for t in texts))


if __name__ == "__main__":
    unittest.main()
