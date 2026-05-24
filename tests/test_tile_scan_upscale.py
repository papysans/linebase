"""Iter 6.5 — `_maybe_upscale_for_verify` math + file-output contract.

Three cases:

  1. 100x100 crop → must upscale 4x to 400x400 and return a DIFFERENT path
     (so the caller can rm it without clobbering the source crop).
  2. 500x400 crop → already big enough; helper must return the SAME path
     (no temp file, no work).
  3. 50x50 crop → tiny case; scale factor must be >= 4 and the returned
     image must be at least 200x200 (the 4x of 50). This pins the floor;
     larger scales (e.g. 9x → 450x450) are fine.

All tests run synchronously and never touch the network — `_maybe_upscale_for_verify`
is a pure image-IO helper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from linebase import pipeline_runner as pr


def _write_solid_crop(path: Path, w: int, h: int) -> None:
    """Write a wxh solid-grey PNG. Pixel content is irrelevant — the helper
    only inspects size + writes the upscale; the resize itself doesn't care
    what colour the source pixels are."""
    arr = np.full((h, w, 3), 128, dtype="uint8")
    Image.fromarray(arr, "RGB").save(path)


def test_maybe_upscale_for_verify_upscales_small_crop(tmp_path: Path) -> None:
    """A 100x100 crop must come back as a NEW 400x400 file (4x LANCZOS)."""
    src = tmp_path / "small.png"
    _write_solid_crop(src, 100, 100)

    out = pr._maybe_upscale_for_verify(src)

    assert out != src, "helper must return a different path for upscaled crops"
    assert out.exists(), "upscaled file must be written to disk"
    with Image.open(out) as img:
        assert img.size == (400, 400), (
            f"expected 4x upscale to 400x400, got {img.size}"
        )


def test_maybe_upscale_for_verify_passthrough_large_crop(tmp_path: Path) -> None:
    """A 500x400 crop is already past the 400-px threshold → no upscale."""
    src = tmp_path / "big.png"
    _write_solid_crop(src, 500, 400)

    out = pr._maybe_upscale_for_verify(src)

    assert out == src, (
        f"helper must return the input path unchanged when crop is already "
        f"big enough, got {out!r} != {src!r}"
    )
    # And the original file must be untouched.
    with Image.open(src) as img:
        assert img.size == (500, 400)


def test_maybe_upscale_for_verify_tiny_crop_at_least_4x(tmp_path: Path) -> None:
    """A 50x50 crop must come back at >= 200x200 (the 4x floor for <100 px)."""
    src = tmp_path / "tiny.png"
    _write_solid_crop(src, 50, 50)

    out = pr._maybe_upscale_for_verify(src)

    assert out != src, "helper must return a different path for upscaled crops"
    with Image.open(out) as img:
        w, h = img.size
    assert w >= 200 and h >= 200, (
        f"50x50 must upscale to at least 200x200 (4x), got {w}x{h}"
    )
