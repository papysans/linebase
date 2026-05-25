"""Iter 10 — SIFT + FLANN + RANSAC homography refine.

The module turns a VLM's rough region into a pixel-tight bbox by running
SIFT inside the (3x-expanded) region. The same function doubles as a
whole-photo recall lifter when called with `region_bbox=None`.

Four regressions covered:

  1. **Happy path** — logo pasted into a known location, VLM region given
     as a slightly-loose box around it; SIFT must return a bbox within
     8 px of the planted ground truth.

  2. **Region mismatch** — same photo, but VLM region points at an empty
     corner that doesn't contain the logo. SIFT must return None (no
     fake-confident answer on a wrong region).

  3. **Logo not present** — photo is featureless / random texture with
     no logo content at all. SIFT must return None.

  4. **Whole-photo recall** — `region_bbox=None` and the logo is somewhere
     in the photo; SIFT must still find it within 8 px (this is the
     recall-lifter path used when VLM returned found=false).

We use the existing NBA logo at `docs/truth_set/2423810_30/logo.png` —
260x297 px with 87 SIFT keypoints, more than enough for a reliable
homography. Synthetic photos are constructed with cv2/numpy directly
(no PIL roundtrip) so the test is fast and deterministic.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from linebase.sift_refine import sift_refine_bbox

# Locate the NBA logo once at module load. Skip the whole module gracefully
# if the truth-set was not extracted in this checkout (CI fixture asymmetry).
_LOGO_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "truth_set"
    / "2423810_30"
    / "logo.png"
)


@pytest.fixture(scope="module")
def logo_path() -> Path:
    if not _LOGO_PATH.exists():
        pytest.skip(f"truth-set logo missing at {_LOGO_PATH}")
    return _LOGO_PATH


def _make_photo_with_logo(
    logo_path: Path,
    *,
    paste_xy: tuple[int, int],
    paste_size: tuple[int, int],
    photo_size: tuple[int, int] = (600, 600),
    seed: int = 0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Build a photo (BGR) with the logo pasted at `paste_xy` resized to
    `paste_size`. Background is mid-grey + low-amplitude noise so SIFT
    doesn't pick up phantom features from a perfectly uniform canvas.

    Returns (photo_array, ground_truth_bbox_xyxy).
    """
    W, H = photo_size
    rng = np.random.default_rng(seed=seed)
    # Mid-grey base + small random jitter. Amplitude 30 keeps the background
    # blandly textured — enough to avoid degenerate-image warnings, not
    # enough to drown out the logo's own features.
    photo = (np.full((H, W, 3), 128, dtype=np.int16)
             + rng.integers(-15, 16, size=(H, W, 3), dtype=np.int16))
    photo = np.clip(photo, 0, 255).astype(np.uint8)

    logo = cv2.imread(str(logo_path))
    assert logo is not None, f"could not load logo at {logo_path}"
    pw, ph = paste_size
    logo_resized = cv2.resize(logo, (pw, ph), interpolation=cv2.INTER_AREA)

    px, py = paste_xy
    photo[py : py + ph, px : px + pw] = logo_resized
    bbox = (px, py, px + pw, py + ph)
    return photo, bbox


def _write_png(arr: np.ndarray, path: Path) -> None:
    ok = cv2.imwrite(str(path), arr)
    assert ok, f"cv2.imwrite failed for {path}"


def test_happy_path_refines_to_within_8px(
    tmp_path: Path, logo_path: Path,
) -> None:
    """Logo pasted at (300, 200) sized 200x200; VLM region (290, 190, 510, 410)
    encloses it loosely. SIFT must come back within 8 px of the planted truth.
    """
    # 200x200 paste — at 100x100 the resized NBA logo only generates ~10
    # SIFT keypoints, which after Lowe's ratio test rarely reaches the
    # 8-good-match floor. 200x200 yields ~33 keypoints, giving a stable
    # 20+ inlier match. Larger paste sizes don't change behaviour.
    photo, gt_bbox = _make_photo_with_logo(
        logo_path,
        paste_xy=(300, 200),
        paste_size=(200, 200),
    )
    photo_path = tmp_path / "photo.png"
    _write_png(photo, photo_path)

    # VLM region: slightly larger than the truth bbox to simulate a
    # plausible rough VLM prediction (off by 10 px each side). After 3x
    # expansion the search region still doesn't cover the whole photo,
    # which is the realistic VLM-bbox scenario.
    region = (290, 190, 510, 410)
    res = sift_refine_bbox(logo_path, photo_path, region_bbox=region)

    assert res is not None, "SIFT should find the pasted logo inside the region"
    assert res.inliers >= 6, f"inliers below floor: {res.inliers}"

    # Bbox within 8 px of ground truth on each side.
    gx1, gy1, gx2, gy2 = gt_bbox
    rx1, ry1, rx2, ry2 = res.bbox
    assert abs(rx1 - gx1) <= 8, f"x1 off by {rx1 - gx1}"
    assert abs(ry1 - gy1) <= 8, f"y1 off by {ry1 - gy1}"
    assert abs(rx2 - gx2) <= 8, f"x2 off by {rx2 - gx2}"
    assert abs(ry2 - gy2) <= 8, f"y2 off by {ry2 - gy2}"


def test_region_mismatch_returns_none(
    tmp_path: Path, logo_path: Path,
) -> None:
    """Same photo, but VLM region (0, 0, 50, 50) points at the empty
    top-left corner — logo is at (300, 200) sized 200x200.

    Important: 3x expansion centers around the original bbox's center
    (25, 25) with extents (150, 150), clamped to the photo — so the
    expanded search region is (0, 0, 100, 100), well outside the logo.
    """
    photo, _gt = _make_photo_with_logo(
        logo_path,
        paste_xy=(300, 200),
        paste_size=(200, 200),
    )
    photo_path = tmp_path / "photo.png"
    _write_png(photo, photo_path)

    res = sift_refine_bbox(
        logo_path,
        photo_path,
        region_bbox=(0, 0, 50, 50),
    )
    assert res is None, (
        f"SIFT must not invent matches in a logo-free region; got {res}"
    )


def test_logo_absent_returns_none(
    tmp_path: Path, logo_path: Path,
) -> None:
    """Photo is pure random texture, no logo present. Whole-photo SIFT must
    return None — no logo content means no inlier-rich homography exists.
    """
    rng = np.random.default_rng(seed=42)
    # Random uniform-noise photo. SIFT will detect plenty of keypoints in
    # noise but the homography to the LOGO's structured points will not
    # have 6+ inliers.
    photo = rng.integers(0, 256, size=(600, 600, 3), dtype=np.uint8)
    photo_path = tmp_path / "photo.png"
    _write_png(photo, photo_path)

    res = sift_refine_bbox(logo_path, photo_path, region_bbox=None)
    assert res is None, (
        f"SIFT must not fabricate a match on logo-free noise; got {res}"
    )


def test_whole_photo_recall_finds_logo(
    tmp_path: Path, logo_path: Path,
) -> None:
    """Recall path (region_bbox=None) on a photo where the logo is at
    (300, 200) sized 200x200. SIFT must find it within 8 px without any
    region hint — this is the path triggered when VLM returned found=false.
    """
    # 200x200 paste — at 100x100 the resized NBA logo only generates ~10
    # SIFT keypoints, which after Lowe's ratio test rarely reaches the
    # 8-good-match floor. 200x200 yields ~33 keypoints, giving a stable
    # 20+ inlier match. Larger paste sizes don't change behaviour.
    photo, gt_bbox = _make_photo_with_logo(
        logo_path,
        paste_xy=(300, 200),
        paste_size=(200, 200),
    )
    photo_path = tmp_path / "photo.png"
    _write_png(photo, photo_path)

    res = sift_refine_bbox(logo_path, photo_path, region_bbox=None)

    assert res is not None, "SIFT recall must find the pasted logo"
    # Recall path doesn't have the strict 10-inlier gate that
    # pipeline_runner applies — the module itself enforces the 6-inlier
    # floor, and the caller decides whether to demand more.
    assert res.inliers >= 6, f"inliers below floor: {res.inliers}"
    assert res.search_origin == (0, 0), (
        f"search_origin should be (0,0) for whole-photo recall, got "
        f"{res.search_origin}"
    )

    gx1, gy1, gx2, gy2 = gt_bbox
    rx1, ry1, rx2, ry2 = res.bbox
    assert abs(rx1 - gx1) <= 8, f"x1 off by {rx1 - gx1}"
    assert abs(ry1 - gy1) <= 8, f"y1 off by {ry1 - gy1}"
    assert abs(rx2 - gx2) <= 8, f"x2 off by {rx2 - gx2}"
    assert abs(ry2 - gy2) <= 8, f"y2 off by {ry2 - gy2}"
