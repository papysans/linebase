"""SIFT + FLANN + RANSAC homography refine — turn a VLM's rough bbox into
a pixel-tight one. Falls back to None when there aren't enough inliers,
which is also the correct signal that the VLM's region was simply wrong.

Algorithm:
1. Crop the photo to `region_bbox` expanded 3x (clamped); call this the
   "search region". If region_bbox is None, use the whole photo.
2. SIFT-detect-and-compute features on the logo (grayscale) and the
   search region (grayscale).
3. FLANN k=2 match; keep matches whose first/second distance ratio < 0.75
   (Lowe's ratio test).
4. If <8 good matches, return None.
5. cv2.findHomography(... cv2.RANSAC, 5.0) on the good matches.
6. Count RANSAC inliers; if <6 inliers, return None.
7. Project the LOGO's four corners through the homography → 4 points in
   search-region coords → axis-aligned bbox.
8. Translate back from search-region coords to original-photo coords.
9. Clamp to photo dims; return (x1, y1, x2, y2, inlier_count).

Why this matters
================
Web research (truth-set analysis 2026-05-25) confirmed generic VLMs cap out
near ~40% any-overlap on hard logo-grounding tasks; the proven industrial
pipeline for trademark detection is classic CV — SIFT/ORB + FLANN + RANSAC
homography (USPTO patents 10438089 / 9536171 / 10769496). VLM stays as the
"what region" coarse locator; SIFT inside the expanded region finds the
pixel-tight match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_SIFT_MIN_GOOD_MATCHES = 8
_SIFT_MIN_INLIERS = 6
_LOWE_RATIO = 0.75
_SEARCH_EXPANSION = 3.0  # crop region_bbox expanded 3x per side
_RANSAC_REPROJ_THRESHOLD = 5.0


@dataclass
class SiftRefineResult:
    bbox: tuple[int, int, int, int]
    inliers: int
    total_matches: int
    search_origin: tuple[int, int]   # (x0, y0) of search region in photo coords


def _expand_bbox(
    b: tuple[int, int, int, int],
    factor: float,
    W: int,
    H: int,
) -> tuple[int, int, int, int]:
    """Expand a bbox by `factor` around its center, then clamp to (0,0,W,H)."""
    x1, y1, x2, y2 = b
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    nx1 = max(0, int(cx - w / 2))
    ny1 = max(0, int(cy - h / 2))
    nx2 = min(W, int(cx + w / 2))
    ny2 = min(H, int(cy + h / 2))
    return nx1, ny1, nx2, ny2


def sift_refine_bbox(
    logo_path: Path,
    photo_path: Path,
    *,
    region_bbox: tuple[int, int, int, int] | None = None,
) -> SiftRefineResult | None:
    """Run SIFT/FLANN/RANSAC inside the search region. Returns None on failure.

    Failure modes deliberately collapsed to None:
      - Either image fails to load (corrupt / wrong extension).
      - Search region is too small (<32 px on a side) — SIFT needs scale.
      - Either descriptor set is empty / <4 keypoints — degenerate input.
      - Fewer than 8 ratio-test "good" matches — not enough signal.
      - findHomography returns None (collinear / degenerate point set).
      - RANSAC inliers <6 — geometric agreement too weak to trust.
      - Projected corners are non-finite or bbox collapses below 8 px.
      - Final bbox extent inverts/clamps to zero area.

    A None return is the correct signal that the VLM's region was either
    wrong, or the logo simply isn't present at a SIFT-detectable scale.
    """
    logo = cv2.imread(str(logo_path), cv2.IMREAD_GRAYSCALE)
    photo = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
    if logo is None or photo is None:
        return None
    H, W = photo.shape[:2]

    if region_bbox is not None:
        sx1, sy1, sx2, sy2 = _expand_bbox(region_bbox, _SEARCH_EXPANSION, W, H)
        search = photo[sy1:sy2, sx1:sx2]
    else:
        sx1, sy1 = 0, 0
        search = photo

    if search.size == 0 or min(search.shape[:2]) < 32:
        return None

    # mypy: cv2 stubs are notoriously incomplete on the contrib API; the
    # SIFT_create / FLANN paths are well-documented and stable across the
    # 4.x line, so we silence the per-call type errors instead of polluting
    # the call sites with cast() wrappers.
    sift = cv2.SIFT_create()  # type: ignore[attr-defined]
    kp1, des1 = sift.detectAndCompute(logo, None)
    kp2, des2 = sift.detectAndCompute(search, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    # FLANN matcher (KD-tree index for SIFT's float descriptors)
    index_params: dict[str, bool | int | float | str] = {
        "algorithm": 1,  # FLANN_INDEX_KDTREE
        "trees": 5,
    }
    search_params: dict[str, bool | int | float | str] = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m_pair in matches:
        # knnMatch can return shorter lists when there are very few
        # descriptors on the train side — skip the unusable pairs.
        if len(m_pair) < 2:
            continue
        m, n = m_pair
        if m.distance < _LOWE_RATIO * n.distance:
            good.append(m)

    if len(good) < _SIFT_MIN_GOOD_MATCHES:
        return None

    # np.float32 accepts iterable-of-iterables at runtime even though the
    # stub declares it as scalar-only; cast away the spurious type errors.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # type: ignore[arg-type]
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # type: ignore[arg-type]

    homography, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, _RANSAC_REPROJ_THRESHOLD
    )
    if homography is None:
        return None
    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < _SIFT_MIN_INLIERS:
        return None

    # Project logo corners through the homography to get the matched
    # quadrilateral in search-region coordinates, then take its axis-aligned
    # bounding rectangle.
    lh, lw = logo.shape[:2]
    corners = np.float32(
        [[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]]  # type: ignore[arg-type]
    ).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, homography)
    except cv2.error:
        return None
    if warped is None:
        return None
    xs = warped[:, 0, 0]
    ys = warped[:, 0, 1]
    # Reject degenerate / inverted projections (NaN/Inf from a near-singular H).
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return None
    bx1, bx2 = float(xs.min()), float(xs.max())
    by1, by2 = float(ys.min()), float(ys.max())
    if bx2 - bx1 < 8 or by2 - by1 < 8:
        return None

    # Translate back from search-region coords to original-photo coords.
    gx1 = max(0, int(round(sx1 + bx1)))
    gy1 = max(0, int(round(sy1 + by1)))
    gx2 = min(W, int(round(sx1 + bx2)))
    gy2 = min(H, int(round(sy1 + by2)))
    if gx2 <= gx1 or gy2 <= gy1:
        return None

    return SiftRefineResult(
        bbox=(gx1, gy1, gx2, gy2),
        inliers=inliers,
        total_matches=len(good),
        search_origin=(sx1, sy1),
    )
