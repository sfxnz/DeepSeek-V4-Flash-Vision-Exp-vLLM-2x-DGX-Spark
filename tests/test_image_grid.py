#!/usr/bin/env python3
"""Invariants on the shipped Vision-Exp image grid, not a reimplementation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker" / "plugin"))

from dsv4_vision.image_processor import (  # noqa: E402
    grid_tokens,
    safe_resize,
    solve_resize_ratio,
)


class ImageGridTests(unittest.TestCase):
    def test_grid_tokens_positive(self) -> None:
        n_h, n_w, n = grid_tokens(384, 384, 14, 3)
        self.assertGreater(n_h, 0)
        self.assertGreater(n_w, 0)
        self.assertGreater(n, 2)

    def test_solve_resize_returns_token_count(self) -> None:
        n_h, n_w, bh, bw, n = solve_resize_ratio(800, 800, 14, 3, 384)
        self.assertGreater(bh, 0)
        self.assertGreater(bw, 0)
        again = grid_tokens(bh, bw, 14, 3)[2]
        self.assertEqual(n, again)

    def test_large_image_stays_within_max_tokens(self) -> None:
        n_h, n_w, bh, bw = safe_resize(5000, 5000, 5000, 5000, 14, 3, 384)
        n = grid_tokens(bh, bw, 14, 3)[2]
        self.assertLessEqual(n, 384)
        self.assertGreater(n, 2)
        self.assertGreater(n_h, 0)
        self.assertGreater(n_w, 0)


if __name__ == "__main__":
    unittest.main()
