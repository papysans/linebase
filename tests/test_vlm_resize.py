"""Iter 6.1 — `_resize_for_vlm` helper unit tests.

Verifies the pre-resize logic that maps VLM-input images into a known
longest-side coordinate frame so the returned bboxes can be reliably
scaled back into the original photo's pixel space.

No network. No actual LLM call. Pure PIL.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from linebase.llm import _MAX_VLM_PIXELS, _resize_for_vlm


def _write_blank(path: Path, width: int, height: int) -> None:
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    img.save(path)


def test_resize_downscales_when_above_budget(tmp_path: Path) -> None:
    """A 2000x1500 image must be downscaled to longest-side == _MAX_VLM_PIXELS."""
    src = tmp_path / "big.png"
    _write_blank(src, 2000, 1500)

    out_path, scale = _resize_for_vlm(src)
    assert out_path != src, "should have written a downscaled tempfile"
    try:
        with Image.open(out_path) as img:
            w, h = img.size
        expected_scale = _MAX_VLM_PIXELS / 2000  # longest side
        assert abs(scale - expected_scale) < 1e-9
        # 2000 -> 1280, 1500 -> 960 (scale 0.64).
        assert w == 1280
        assert h == 960
    finally:
        out_path.unlink(missing_ok=True)


def test_resize_passthrough_when_under_budget(tmp_path: Path) -> None:
    """An 800x600 image fits under the budget — return the original path, scale=1.0."""
    src = tmp_path / "small.png"
    _write_blank(src, 800, 600)

    out_path, scale = _resize_for_vlm(src)
    assert out_path == src
    assert scale == 1.0
    # The source must still exist and be unchanged after the call.
    with Image.open(src) as img:
        assert img.size == (800, 600)
