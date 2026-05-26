"""Iter 6.1 — `_resize_for_vlm` helper unit tests.

Verifies the pre-resize logic that maps VLM-input images into a known
longest-side coordinate frame so the returned bboxes can be reliably
scaled back into the original photo's pixel space.

No network. No actual LLM call. Pure PIL.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from linebase.llm import (
    _MAX_VLM_PIXELS,
    MatchResult,
    _map_bbox_coords_to_source,
    _resize_for_vlm,
    match_logo_in_zoomed,
)


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


def test_qwen_large_image_uses_1000_coordinate_frame_from_iter13_doc() -> None:
    """2016x1512 case from docs/iter13_dim_hint_report.md.

    Qwen returned x1≈188 in a 0-1000 frame; interpreting it as a 1280-resized
    pixel gives x≈296, the observed left shift. The correct mapping is
    188/1000*2016≈379, matching the expected bbox's x1=380.
    """
    mapped, mode = _map_bbox_coords_to_source(
        [188.5, 465.6, 269.8, 556.9],
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        source_w=2016,
        source_h=1512,
        sent_w=1280,
        sent_h=960,
        sent_scale=1280 / 2016,
    )

    assert mode == "qwen_normalized_1000"
    assert [round(v) for v in mapped] == [380, 704, 544, 842]


def test_non_qwen_large_image_keeps_sent_pixel_scale_back() -> None:
    """Pixel-frame providers keep the original resize scale-back behavior."""
    mapped, mode = _map_bbox_coords_to_source(
        [241, 447, 345, 535],
        model="doubao-seed-2-0-pro-260215",
        source_w=2016,
        source_h=1512,
        sent_w=1280,
        sent_h=960,
        sent_scale=1280 / 2016,
    )

    assert mode == "sent_pixels_scaled"
    assert [round(v) for v in mapped] == [380, 704, 543, 843]


def test_zoomed_refine_always_unscales_explicit_2x(
    tmp_path: Path, monkeypatch
) -> None:
    """A bbox in the upper-left half of a 2x zoom crop still needs /2.

    This is the failure mode the old `if bx2 > crop_w` heuristic missed.
    """
    logo = tmp_path / "logo.png"
    photo = tmp_path / "photo.png"
    _write_blank(logo, 64, 64)
    _write_blank(photo, 500, 500)

    captured: dict[str, tuple[int, int]] = {}

    def fake_run_match_call(**kwargs):
        with Image.open(kwargs["photo_path"]) as img:
            captured["sent_size"] = img.size
        return MatchResult(
            found=True,
            bbox=(20, 20, 80, 80),
            confidence=0.9,
            reason="upper-left refined bbox in upscaled crop",
            raw_response="{}",
            prompt_version="4_refine",
            model="test-model",
        )

    monkeypatch.setattr("linebase.llm._run_match_call", fake_run_match_call)

    result, origin = match_logo_in_zoomed(
        logo,
        photo,
        region_bbox=(100, 100, 200, 200),
        model="test-model",
    )

    assert origin == (70, 70)
    assert captured["sent_size"] == (320, 320)
    assert result.bbox == (10, 10, 40, 40)
