"""Edge-based shape matching — match a USPTO line-art logo against printed
logo instances in real-world photos. Designed for the case where SIFT fails
because the feature spaces are too different (line-art vs filled+colored
print).

Algorithm:
1. Crop the photo to ``region_bbox`` expanded 3x (clamped). If ``region_bbox``
   is None, use the whole photo.
2. Canny edges on the LOGO with an auto threshold based on the median pixel
   value (lower = 0.66 * med, upper = 1.33 * med). Find external contours.
3. Pick the single longest logo contour as the reference shape. The line-art
   logo IS already an outline so its longest contour is the silhouette we
   want to match against.
4. Canny edges on the search region with the same auto-threshold scheme.
5. Find candidate contours in the search region (external only).
6. Filter candidates: drop contours with area < ``_MIN_AREA`` or
   > ``_MAX_AREA_RATIO`` of the search-region area, and contours with fewer
   than 4 points (degenerate, ``matchShapes`` would either crash or return
   meaningless values).
7. For each surviving candidate, compute
   ``cv2.matchShapes(logo_contour, candidate, METHOD_I1, 0.0)``. Smaller is
   more similar; 0 means identical Hu-moments.
8. Pick the candidate with the smallest distance.
9. If the best distance exceeds ``_DIST_REJECT_THRESHOLD`` (1.0) the match is
   not really similar — return None.
10. ``boundingRect`` of the best candidate, translated from search-region
    coords back to original-photo coords.
11. Return :class:`EdgeRefineResult` with the bbox, the shape distance, the
    candidate count, and the search origin.

Why this matters
================
Web research (USPTO patent 9536171 "Logo detection by edge matching", iter-10
SIFT failure analysis 2026-05-25) confirmed that line-art ↔ printed-logo
matching needs contour shape matching, not keypoint matching. SIFT cannot
bridge the feature gap: the line-art drawing has no interior texture, the
printed logo has fill but no fine edges. Both, however, reduce to the same
silhouette under Canny — matchShapes on Hu moments is scale- and
rotation-invariant and works on that shared representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_SEARCH_EXPANSION = 3.0
_MIN_AREA = 200
_MAX_AREA_RATIO = 0.6   # contour can't be >60% of search area
_DIST_REJECT_THRESHOLD = 1.0  # cv2.matchShapes I1 distance


@dataclass
class EdgeRefineResult:
    bbox: tuple[int, int, int, int]
    shape_distance: float
    candidates_checked: int
    search_origin: tuple[int, int]   # (x0, y0) of search region in photo coords


def _canny_auto(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with a median-based auto threshold. Sigma defaults to 0.33, the
    value Adrian Rosebrock benchmarked across natural-image edge tasks; in
    practice it sits in the [0.66*med, 1.33*med] band that covers both the
    line-art "almost black-and-white" case and the printed-logo "mostly mid
    grey" case without per-image tuning.
    """
    v = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lo, hi)


def _largest_contour(edges: np.ndarray) -> np.ndarray | None:
    """Return the contour with the largest combined (area + 0.1 * arclength).
    Pure area picks tiny-but-fat blobs over long thin outlines, which is the
    wrong choice for line-art logos (their silhouette is the long edge).
    Pure arclength picks wiggly background contours over the logo's clean
    outline. The 0.1-weighted hybrid biases towards "long and substantial".
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=lambda c: cv2.contourArea(c) + 0.1 * cv2.arcLength(c, False))


def _expand_bbox(
    b: tuple[int, int, int, int],
    factor: float,
    W: int,
    H: int,
) -> tuple[int, int, int, int]:
    """Expand a bbox by ``factor`` around its center, then clamp to (0,0,W,H)."""
    x1, y1, x2, y2 = b
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    return (
        max(0, int(cx - w / 2)),
        max(0, int(cy - h / 2)),
        min(W, int(cx + w / 2)),
        min(H, int(cy + h / 2)),
    )


def edge_refine_bbox(
    logo_path: Path,
    photo_path: Path,
    *,
    region_bbox: tuple[int, int, int, int] | None = None,
) -> EdgeRefineResult | None:
    """Run Canny + matchShapes refine inside the search region. Returns None
    on failure.

    Failure modes deliberately collapsed to None:
      - Either image fails to load.
      - Search region too small (<32 px on a side) — not enough pixels for
        meaningful contour topology.
      - Logo has no contour at all, or the longest contour has <4 points.
      - No candidate contours in the search region pass the area filter.
      - All ``matchShapes`` calls raised cv2.error or returned non-finite.
      - Best shape distance exceeds the reject threshold (1.0).
      - boundingRect of the winner is degenerate (<8 px on a side) or
        clamps to zero area in original-photo coords.
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

    # Logo edges → reference contour.
    logo_edges = _canny_auto(logo)
    logo_contour = _largest_contour(logo_edges)
    if logo_contour is None or len(logo_contour) < 4:
        return None

    # Search edges → candidate contours.
    search_edges = _canny_auto(search)
    contours, _ = cv2.findContours(
        search_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    sh, sw = search.shape[:2]
    search_area = float(sh * sw)
    candidates: list[tuple[float, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < _MIN_AREA or area > _MAX_AREA_RATIO * search_area:
            continue
        if len(c) < 4:
            continue
        try:
            dist = float(
                cv2.matchShapes(logo_contour, c, cv2.CONTOURS_MATCH_I1, 0.0)
            )
        except cv2.error:
            continue
        if not np.isfinite(dist):
            continue
        candidates.append((dist, c))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    best_dist, best_c = candidates[0]
    if best_dist > _DIST_REJECT_THRESHOLD:
        return None

    x, y, w, h = cv2.boundingRect(best_c)
    if w < 8 or h < 8:
        return None

    gx1 = max(0, int(sx1 + x))
    gy1 = max(0, int(sy1 + y))
    gx2 = min(W, int(sx1 + x + w))
    gy2 = min(H, int(sy1 + y + h))
    if gx2 <= gx1 or gy2 <= gy1:
        return None

    return EdgeRefineResult(
        bbox=(gx1, gy1, gx2, gy2),
        shape_distance=best_dist,
        candidates_checked=len(candidates),
        search_origin=(sx1, sy1),
    )
