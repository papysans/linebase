"""Self-verify / re-crop loop layered on top of `match_logo_in_photo`.

Two-call orchestration:
  1) Pass 1 — first-pass matcher returns a bbox candidate.
  2) Verify — crop with +20% padding, ask a verify prompt whether the crop actually
     contains the registered trademark as a whole, and whether the crop is sized well.

Outcomes:
  - verify confirms tight/loose       → verified=True (optionally shrink via suggested_bbox)
  - verify rejects (wrong / no logo)  → verified=False, force found=False so the
                                        pipeline downgrades the row to needs_review
  - verify says too_tight             → expand bbox by 20% per side, verified=True
                                        (we trust the rejection signal more than a
                                         refinement instruction; no second LLM call)

At most ONE verify call per evidence — never loops forever.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

from linebase.config import Settings
from linebase.crop import crop_to_bbox
from linebase.llm import (
    MatchResult,
    VerifyAnswer,
    match_logo_in_photo,
    match_logo_with_feedback,
    verify_crop,
)

# Substrings (case-insensitive) that mark a verify rejection as "blank-crop"
# rather than "wrong-logo". When any of these appears in `verify_reason`,
# match_with_verify fires the Pass-3 retry-with-feedback path. Curated from
# the ~35 historical rejects in .data/linebase.db (the "blank-crop verify-
# rejects" cohort that motivated this iter).
BLANK_VERIFY_PATTERNS: tuple[str, ...] = (
    "blank",
    "no visible",
    "no trace",
    "background",
    "no part of",
)

# Pre-gate thresholds. A crop whose pixel std-dev is below 15.0 AND whose
# (>240,>240,>240) "near-white" pixel ratio is above 0.7 is considered
# blank — short-circuit verify, skip the API call, fire the retry directly.
_PRE_GATE_STD = 15.0
_PRE_GATE_WHITE_RATIO = 0.7


@dataclass
class VerifyResult:
    primary: MatchResult                                  # untouched Pass-1 MatchResult
    verified: bool                                        # did the verify step accept?
    final_bbox: tuple[int, int, int, int] | None          # bbox to actually crop with (in ORIGINAL photo coords)
    fit_label: str | None                                 # one of {tight, loose, too_tight, wrong} or None when verify skipped
    verify_confidence: float | None
    verify_reason: str | None
    iters: int                                            # 1 (skipped verify) or 2 (verify ran)
    verify_usage: dict[str, int] | None = None
    # Diagnostic extras — handy for debugging / eval but not part of the spec.
    skipped_reason: str | None = field(default=None)      # why verify was skipped, if iters==1
    # Iter 5 retry plumbing. When a blank-crop verify reject (or the variance
    # pre-gate) fires the Pass-3 retry-with-feedback path, `retried` flips to
    # True and `retry_reason` captures *what* triggered the retry — useful for
    # post-mortem diagnostics and the frontend pill.
    retried: bool = False
    retry_reason: str | None = field(default=None)
    # Retry's own bbox (in ORIGINAL coords) when it produced one and it
    # was actually used as the new primary. None when retry returned
    # found=false or the IoU gate rejected the retry's bbox.
    retry_bbox: tuple[int, int, int, int] | None = field(default=None)


def _clamp_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return x1, y1, x2, y2


def _padded_bbox(
    bbox: tuple[int, int, int, int], pad_ratio: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)
    return _clamp_bbox((x1 - pw, y1 - ph, x2 + pw, y2 + ph), width, height)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Standard axis-aligned IoU. Defensive against zero-area boxes (returns 0)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _is_blank_verify_reason(reason: str | None) -> bool:
    """True when the verify rejection signals "crop was blank, not a wrong logo"."""
    if not reason:
        return False
    low = reason.lower()
    return any(p in low for p in BLANK_VERIFY_PATTERNS)


def _stats_from_rgb_array(arr: "np.ndarray[Any, Any]") -> tuple[float, float] | None:
    """Core numpy calc shared by `_crop_blank_stats` and `_bbox_blank_stats`.

    Returns `(std_dev, near_white_ratio)` over the array's pixels, or None when
    the array is empty. `near_white_ratio` counts pixels with all three RGB
    channels > 240.
    """
    if arr.size == 0:
        return None
    std = float(arr.std())
    white_mask = (arr > 240).all(axis=-1)
    white_ratio = float(white_mask.mean())
    return std, white_ratio


def _crop_blank_stats(crop_path: Path) -> tuple[float, float] | None:
    """Return (std_dev, near_white_ratio) for the crop, or None on failure.

    `near_white_ratio` is the fraction of pixels where all three RGB channels
    exceed 240 — a USPTO blank margin is essentially pure white at >250 on all
    channels, so 240 catches mild compression artifacts too.
    """
    try:
        with Image.open(crop_path) as img:
            arr = np.asarray(img.convert("RGB"))
    except Exception:
        return None
    return _stats_from_rgb_array(arr)


def _bbox_blank_stats(
    photo_path: Path,
    bbox: tuple[int, int, int, int] | list[int],
) -> tuple[float, float] | None:
    """Return (std_dev, near_white_ratio) for `photo_path[bbox]`, or None on failure.

    Unlike `_crop_blank_stats`, this works directly on the source photo + a
    bbox in photo coords — no temp file. Used by the iter-7 Pass-1 variance
    pre-gate in `pipeline_runner` to filter blank-region hallucinations before
    they reach the tile-scan / crop / verify stages.

    The bbox is clamped to the image bounds; if clamping collapses it to zero
    area we return None (caller treats that as "can't decide, skip the gate").
    """
    try:
        with Image.open(photo_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            x1, y1, x2, y2 = (int(v) for v in bbox)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            crop = rgb.crop((x1, y1, x2, y2))
            arr = np.asarray(crop)
    except Exception:
        return None
    return _stats_from_rgb_array(arr)


def match_with_verify(
    logo_path: Path,
    photo_path: Path,
    settings: Settings | None = None,
    client: OpenAI | None = None,
    max_iters: int = 2,
    verify_threshold: float = 0.6,
    model: str | None = None,
) -> VerifyResult:
    """Pass-1 match + at most one verify call. See module docstring for the rules.

    Note on `client`: when `client is None` (the common case), we let the lower
    layer build the correct OpenAI-compatible client via `Settings.resolve_provider`
    so SiliconFlow / Ark / OpenAI all route correctly. Passing a pre-built
    `client` is a legacy escape hatch — only use it when you know it matches the
    provider for `settings.model`.

    `model`: optional per-call override. When set, BOTH Pass-1 and the verify
    call use this model id (so per-job/per-row overrides actually reach both
    passes). When None, Pass-1 falls back to `settings.model` and verify falls
    back to `settings.review_model`, matching the legacy behavior.
    """
    settings = settings or Settings.from_env()

    primary = match_logo_in_photo(
        logo_path, photo_path, settings=settings, client=client, model=model
    )

    # Skip verify when Pass-1 already says "no" or is too unsure.
    if not primary.found or primary.bbox is None or primary.confidence < 0.4:
        return VerifyResult(
            primary=primary,
            verified=False,
            final_bbox=None,
            fit_label=None,
            verify_confidence=None,
            verify_reason=None,
            iters=1,
            verify_usage=None,
            skipped_reason=("not_found" if not primary.found else "low_confidence"),
        )

    # Effective ceiling on the number of LLM calls in this loop.
    if max_iters < 2:
        return VerifyResult(
            primary=primary,
            verified=True,
            final_bbox=primary.bbox,
            fit_label=None,
            verify_confidence=None,
            verify_reason=None,
            iters=1,
            verify_usage=None,
            skipped_reason="max_iters_lt_2",
        )

    # Load the photo to learn its dimensions for padding / clamping.
    with Image.open(photo_path) as img:
        orig_w, orig_h = img.size

    verify_model = model or settings.review_model

    # ------- Round 0: pre-gate + verify ----------------------------------
    verify, crop_bbox_used, pre_gated, _crop_stats = _verify_round(
        primary.bbox, photo_path, logo_path, orig_w, orig_h,
        settings=settings, client=client, verify_model=verify_model,
    )
    cx1, cy1, cx2, cy2 = crop_bbox_used

    if pre_gated:
        verified = False
        final_bbox: tuple[int, int, int, int] | None = None
        fit_label_used: str | None = "blank_pre_gate"
        verify_confidence_used: float | None = 0.0
        std_v, white_v = _crop_stats or (0.0, 0.0)
        verify_reason_used: str | None = (
            f"pre-gate: crop is uniform/blank (std={std_v:.1f}, white={white_v:.2f})"
        )
        verify_usage_used: dict[str, int] | None = None
    else:
        assert verify is not None
        ok, fbb = _accept_verify(
            verify, primary.bbox, crop_bbox_used, orig_w, orig_h, verify_threshold,
        )
        verified = ok
        final_bbox = fbb
        fit_label_used = verify.fit
        verify_confidence_used = verify.confidence
        verify_reason_used = verify.reason
        verify_usage_used = verify.usage

    # ------- Round 1: Pass-3 retry-with-feedback (blank rejects only) ----
    retry_used = False
    retry_bbox_used: tuple[int, int, int, int] | None = None
    retry_reason: str | None = None
    if not verified and (pre_gated or _is_blank_verify_reason(verify_reason_used)):
        retry_used = True
        retry_reason = "blank_pre_gate" if pre_gated else "blank_verify_reject"
        try:
            rr = match_logo_with_feedback(
                logo_path, photo_path,
                prior_bbox=primary.bbox, prior_reason=primary.reason,
                verify_reason=(verify_reason_used or ""),
                settings=settings, client=client, model=model,
            )
        except Exception:
            rr = None
        if rr is not None and rr.found and rr.bbox is not None:
            new_bbox = _clamp_bbox(rr.bbox, orig_w, orig_h)
            if _bbox_iou(new_bbox, primary.bbox) < 0.5:
                v2, cbox2, pre2, st2 = _verify_round(
                    new_bbox, photo_path, logo_path, orig_w, orig_h,
                    settings=settings, client=client, verify_model=verify_model,
                )
                if pre2:
                    std_v, white_v = st2 or (0.0, 0.0)
                    verify_reason_used = (
                        f"retry pre-gate: still blank (std={std_v:.1f}, white={white_v:.2f})"
                    )
                    fit_label_used = "blank_pre_gate"
                    verify_confidence_used = 0.0
                else:
                    assert v2 is not None
                    ok2, fbb2 = _accept_verify(
                        v2, new_bbox, cbox2, orig_w, orig_h, verify_threshold,
                    )
                    if v2.usage:
                        verify_usage_used = _sum_int_dicts(verify_usage_used, v2.usage)
                    fit_label_used = v2.fit
                    verify_confidence_used = v2.confidence
                    verify_reason_used = v2.reason
                    if ok2:
                        verified, final_bbox, retry_bbox_used = True, fbb2, new_bbox
                    else:
                        verified, final_bbox = False, None

    return VerifyResult(
        primary=primary,
        verified=verified,
        final_bbox=final_bbox,
        fit_label=fit_label_used,
        verify_confidence=verify_confidence_used,
        verify_reason=verify_reason_used,
        iters=2,
        verify_usage=verify_usage_used,
        retried=retry_used,
        retry_reason=retry_reason,
        retry_bbox=retry_bbox_used,
    )


def _verify_round(
    candidate_bbox: tuple[int, int, int, int],
    photo_path: Path,
    logo_path: Path,
    orig_w: int,
    orig_h: int,
    *,
    settings: Settings,
    client: OpenAI | None,
    verify_model: str,
) -> tuple[
    VerifyAnswer | None,
    tuple[int, int, int, int],
    bool,
    tuple[float, float] | None,
]:
    """Crop the candidate bbox (+20% pad), pre-gate, optionally call verify_crop.

    Returns (verify_answer_or_None, padded_crop_bbox, pre_gated, blank_stats).
    pre_gated=True short-circuits without an API call.
    """
    cbox = _padded_bbox(candidate_bbox, 0.20, orig_w, orig_h)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        crop_to_bbox(photo_path, cbox, tmp_path, pad_ratio=0.0)
        stats = _crop_blank_stats(tmp_path)
        if stats is not None and stats[0] < _PRE_GATE_STD and stats[1] > _PRE_GATE_WHITE_RATIO:
            return None, cbox, True, stats
        answer = verify_crop(
            logo_path, tmp_path, settings=settings, client=client, model=verify_model
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return answer, cbox, False, stats


def _accept_verify(
    ans: VerifyAnswer,
    ref_bbox: tuple[int, int, int, int],
    cbox: tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
    verify_threshold: float,
) -> tuple[bool, tuple[int, int, int, int] | None]:
    """Apply legacy accept rules (tight/loose accept, too_tight expand, else reject)."""
    cx1, cy1, cx2, cy2 = cbox
    if (
        ans.contains_full_logo
        and ans.fit in ("tight", "loose")
        and ans.confidence >= verify_threshold
    ):
        if ans.fit == "loose" and ans.suggested_bbox is not None:
            sx1, sy1, sx2, sy2 = ans.suggested_bbox
            cand = _clamp_bbox((cx1 + sx1, cy1 + sy1, cx1 + sx2, cy1 + sy2), orig_w, orig_h)
            if cand[0] >= cx1 and cand[1] >= cy1 and cand[2] <= cx2 and cand[3] <= cy2:
                return True, cand
        return True, ref_bbox
    if ans.fit == "too_tight":
        return True, _padded_bbox(ref_bbox, 0.20, orig_w, orig_h)
    return False, None


def _sum_int_dicts(
    a: dict[str, int] | None, b: dict[str, int] | None
) -> dict[str, int] | None:
    """Element-wise add two usage dicts. Tolerates None on either side."""
    if not a:
        return dict(b) if b else None
    if not b:
        return dict(a)
    return {k: int(a.get(k, 0) or 0) + int(b.get(k, 0) or 0) for k in set(a) | set(b)}


__all__ = [
    "VerifyResult",
    "VerifyAnswer",
    "match_with_verify",
    "BLANK_VERIFY_PATTERNS",
    "_bbox_blank_stats",
    "_crop_blank_stats",
]
