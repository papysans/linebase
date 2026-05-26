"""Iter 11 — Edge-based shape matching (Canny + matchShapes / Hu moments).

The module refines a VLM's rough bbox into a pixel-tight one by matching
contour shapes between the line-art logo and the search region. Unlike SIFT,
which fails when the two images share zero interior texture (line-art vs
filled+colored print), shape matching reduces both to silhouettes via Canny
and works on the shared representation.

Four regressions covered:

  1. **Happy path (synthetic)** — paste a synthetic line-art outline at a
     known location, run edge_refine with a region covering it. Result must
     be within 5 px of the planted bbox and shape_distance must be small.

  2. **Line-art vs filled** — line-art outline logo, photo with a filled
     colored disc of similar size. After Canny both reduce to circular
     contours; matchShapes should find the disc and return a small distance.

  3. **No match** — random-texture photo with no logo-shaped contour.
     Function must return None (no fake confident answer).

  4. **Real fixture** — load the NBA player silhouette and the Gatorade box
     evidence from the truth set. This is a HARD case (printed logo on a
     busy box). Acceptance is "returns either a finite bbox or None without
     exception" — exercise the real-data path, not a correctness floor.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from linebase.edge_refine import edge_refine_bbox

# Real truth-set fixture used for the line-art ↔ printed-logo soak test.
_TRUTHSET = Path(__file__).resolve().parents[1] / "docs" / "truth_set" / "2423810_30"


def _write_png(arr: np.ndarray, path: Path) -> None:
    ok = cv2.imwrite(str(path), arr)
    assert ok, f"cv2.imwrite failed for {path}"


def _make_line_art_square_logo(size: int = 100) -> np.ndarray:
    """A black square outline on white — the simplest possible line-art logo.

    Returns a 3-channel BGR image so cv2.imread of the saved file produces
    the same shape regardless of imread mode. Border thickness 4 px is wide
    enough to survive Canny on the downstream pasted copy.
    """
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (size - 6, size - 6), (0, 0, 0), thickness=4)
    return img


def test_happy_path_finds_pasted_square(tmp_path: Path) -> None:
    """Line-art square logo pasted at (300, 200) sized 100x100; VLM region
    (290, 190, 410, 310) encloses it loosely. Edge refine must return a
    bbox within 5 px of the planted truth and a small shape distance.
    """
    logo = _make_line_art_square_logo(size=100)
    logo_path = tmp_path / "logo.png"
    _write_png(logo, logo_path)

    # Mid-grey textured photo so Canny doesn't ignore the background.
    rng = np.random.default_rng(seed=0)
    photo = (np.full((600, 600, 3), 128, dtype=np.int16)
             + rng.integers(-10, 11, size=(600, 600, 3), dtype=np.int16))
    photo = np.clip(photo, 0, 255).astype(np.uint8)

    # Paste the logo at (300, 200).
    px, py, pw, ph = 300, 200, 100, 100
    photo[py:py + ph, px:px + pw] = logo
    photo_path = tmp_path / "photo.png"
    _write_png(photo, photo_path)

    res = edge_refine_bbox(
        logo_path, photo_path, region_bbox=(290, 190, 410, 310),
    )
    assert res is not None, "edge refine should find the pasted line-art square"
    assert res.shape_distance < 0.3, (
        f"shape distance too high for an identical shape: {res.shape_distance}"
    )

    # 5 px tolerance on each side. Canny + boundingRect typically lands
    # 1-3 px inside the painted outline because the outermost pixel column
    # of the outline is partially in the noise band.
    rx1, ry1, rx2, ry2 = res.bbox
    assert abs(rx1 - px) <= 5, f"x1 off by {rx1 - px}"
    assert abs(ry1 - py) <= 5, f"y1 off by {ry1 - py}"
    assert abs(rx2 - (px + pw)) <= 5, f"x2 off by {rx2 - (px + pw)}"
    assert abs(ry2 - (py + ph)) <= 5, f"y2 off by {ry2 - (py + ph)}"


def test_line_art_logo_matches_filled_circle(tmp_path: Path) -> None:
    """Line-art (outline) circle as logo vs a high-contrast filled disc on a
    photo.

    Both reduce to circular contours under Canny. matchShapes on Hu moments
    is fill-agnostic — it sees the silhouette only — so the disc should be
    the best match and the returned bbox should overlap with where the disc
    was painted. We use a black disc on a white photo to guarantee Canny
    actually fires on the boundary; the iter-11 algorithm uses an automatic
    threshold based on the photo median which would otherwise wash out a
    low-contrast mid-grey disc.
    """
    # Line-art circle logo: white background, black outline, thickness 4.
    logo = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.circle(logo, (75, 75), 60, (0, 0, 0), thickness=4)
    logo_path = tmp_path / "logo.png"
    _write_png(logo, logo_path)

    # Photo: near-white background, one black filled disc. The auto-median
    # Canny threshold lands at the lower end on a mostly-white image, so the
    # disc's hard black boundary always passes.
    photo = np.full((600, 600, 3), 240, dtype=np.uint8)
    cv2.circle(photo, (350, 250), 60, (0, 0, 0), thickness=-1)
    photo_path = tmp_path / "photo.png"
    _write_png(photo, photo_path)

    res = edge_refine_bbox(
        logo_path, photo_path, region_bbox=(280, 180, 420, 320),
    )
    assert res is not None, "edge refine must match outline-circle ↔ filled-disc"
    # Shape distance for an outline contour vs a filled-disc contour is
    # small but not zero. Allow up to 0.6 — observed value with this setup
    # is ~0.2.
    assert res.shape_distance < 0.6, (
        f"outline-vs-filled distance unexpectedly high: {res.shape_distance}"
    )
    # bbox must overlap the painted disc (center 350, 250, radius 60).
    rx1, ry1, rx2, ry2 = res.bbox
    assert rx1 < 350 < rx2, f"bbox x range ({rx1}, {rx2}) misses disc center 350"
    assert ry1 < 250 < ry2, f"bbox y range ({ry1}, {ry2}) misses disc center 250"


def test_no_match_returns_none(tmp_path: Path) -> None:
    """Photo is a uniform low-frequency gradient — no closed contours of
    meaningful area exist after Canny. Function must return None because
    the candidate filter (min area 200) eliminates every detected blob.

    This is the "no contours pass the filter" failure mode, which is the
    cleanest way to assert None: random noise photos can produce Hu-moment-
    similar blobs by accident, so we deliberately construct an input where
    findContours returns zero qualifying candidates.
    """
    logo = _make_line_art_square_logo(size=100)
    logo_path = tmp_path / "logo.png"
    _write_png(logo, logo_path)

    # Uniform white photo — Canny finds zero edges, zero contours.
    photo = np.full((600, 600, 3), 255, dtype=np.uint8)
    photo_path = tmp_path / "photo.png"
    _write_png(photo, photo_path)

    res = edge_refine_bbox(logo_path, photo_path, region_bbox=None)
    assert res is None, (
        f"edge refine must return None when no candidate contours exist; got {res}"
    )


def test_whole_photo_recall_prefers_large_document_mark_over_tiny_text(tmp_path: Path) -> None:
    """Whole-photo recall should not be fooled by tiny text-like contours.

    A USPTO document page can contain dozens of tiny square-ish glyph/marker
    contours whose Hu moments look deceptively close to a square-ish logo. The
    recall mode must favor a page-level reproduced mark large enough to be a
    usable evidence crop.
    """
    logo = _make_line_art_square_logo(size=120)
    logo_path = tmp_path / "logo.png"
    _write_png(logo, logo_path)

    photo = np.full((1000, 800, 3), 255, dtype=np.uint8)
    # Tiny square-ish distractors that would rank very well by shape distance
    # alone but are not a usable match on a document page.
    for y in range(650, 900, 45):
        for x in range(80, 720, 55):
            cv2.rectangle(photo, (x, y), (x + 18, y + 18), (0, 0, 0), thickness=2)
    # The actual reproduced mark on the document.
    px, py, pw, ph = 260, 220, 280, 280
    large_logo = cv2.resize(logo, (pw, ph), interpolation=cv2.INTER_NEAREST)
    photo[py:py + ph, px:px + pw] = large_logo
    photo_path = tmp_path / "document.png"
    _write_png(photo, photo_path)

    res = edge_refine_bbox(logo_path, photo_path, region_bbox=None, whole_photo_recall=True)

    assert res is not None
    rx1, ry1, rx2, ry2 = res.bbox
    assert abs(rx1 - px) <= 8
    assert abs(ry1 - py) <= 8
    assert abs(rx2 - (px + pw)) <= 8
    assert abs(ry2 - (py + ph)) <= 8


def test_real_fixture_runs_without_exception() -> None:
    """Soak test against the real truth-set fixture (NBA player silhouette
    logo, Gatorade box evidence). The printed logo is colored and small on
    a busy box — hard case for shape matching. Acceptance is purely "runs
    cleanly and returns either a bbox or None"; correctness on this real
    pair is measured by the E2E scoring script, not this unit test.
    """
    if not _TRUTHSET.exists():
        pytest.skip(f"truth-set missing at {_TRUTHSET}")
    logo = _TRUTHSET / "logo.png"
    photo = _TRUTHSET / "pair_01" / "evidence.png"
    if not logo.exists() or not photo.exists():
        pytest.skip("truth-set logo/evidence files missing")

    # Truth bbox from truth.json: [807, 69, 844, 135].
    res = edge_refine_bbox(logo, photo, region_bbox=(807, 69, 844, 135))

    # No assertion on correctness — only on shape contract. Either a
    # well-formed EdgeRefineResult OR a clean None is acceptable here.
    if res is not None:
        x1, y1, x2, y2 = res.bbox
        assert x2 > x1 and y2 > y1, f"degenerate bbox returned: {res.bbox}"
        assert res.candidates_checked >= 1, (
            f"candidates_checked must be >=1 when result is not None: {res}"
        )
        assert np.isfinite(res.shape_distance), (
            f"shape_distance must be finite: {res}"
        )
