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

import contextlib
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
    match_logo_in_zoomed,
    match_logo_with_feedback,
    verify_crop,
)


def _refine_env_enabled() -> bool:
    """Default-on; `LINEBASE_REFINE=0/false/off/no` disables. Case-insensitive."""
    import os

    raw = os.environ.get("LINEBASE_REFINE", "").strip().lower()
    return raw not in {"0", "false", "off", "no"}

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
SOFT_VERIFY_REJECT_PATTERNS: tuple[str, ...] = (
    "faint",
    "blur",
    "blurry",
    "degraded",
    "low-contrast",
    "low contrast",
    "dirty",
    "reflective",
    "incomplete",
    "distorted",
)
DESIGN_SURFACE_TERMS: tuple[str, ...] = (
    "cannage",
    "diamond",
    "geometric",
    "grid",
    "lattice",
    "motif",
    "octagonal",
    "ornament",
    "ornamental",
    "pattern",
    "quilt",
    "quilted",
    "repeat",
    "repeating",
    "stitch",
    "stitched",
    "studded",
    "surface",
    "texture",
)
DESIGN_SHAPE_REJECT_TERMS: tuple[str, ...] = (
    "different design",
    "different product",
    "different structure",
    "fundamentally different",
    "no matching shape",
    "no matching structure",
    "no visual shape correspondence",
    "shape correspondence",
    "unrelated product",
    "visually unrelated",
)
DESIGN_SURFACE_REJECT_TERMS: tuple[str, ...] = (
    "no matching design content",
    "not a match",
)
DESIGN_SURFACE_HARD_REJECT_TERMS: tuple[str, ...] = (
    "different pattern",
    "floral",
    "without the specified surface design",
)
DESIGN_SURFACE_SOFT_MIN_CONFIDENCE = 0.90
DESIGN_EDGE_RECALL_SOFT_MAX_DISTANCE = 0.10

# Pre-gate thresholds. A crop whose pixel std-dev is below 15.0 AND whose
# (>240,>240,>240) "near-white" pixel ratio is above 0.7 is considered
# blank — short-circuit verify, skip the API call, fire the retry directly.
_PRE_GATE_STD = 15.0
_PRE_GATE_WHITE_RATIO = 0.7
_VERIFY_UPSCALE_THRESHOLD = 400
_LOGO_FOREGROUND_THRESHOLD = 245
_LOOSE_BBOX_MAX_ASPECT_MISMATCH = 2.5


@dataclass
class VerifyResult:
    primary: MatchResult                                  # untouched Pass-1 MatchResult
    verified: bool                                        # did the verify step accept?
    final_bbox: tuple[int, int, int, int] | None
    fit_label: str | None
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
    # Iter 9 — refine pass. Flips to True when the refine branch fired and
    # its bbox SURVIVED a second verify (i.e. final_bbox was sourced from
    # `match_logo_in_zoomed`, not Pass-1 or the verify-suggested shrink).
    refined: bool = False
    # Refine's bbox in ORIGINAL photo coords. Only set when `refined=True`.
    refine_bbox: tuple[int, int, int, int] | None = field(default=None)
    # Origin (top-left x, y) of the zoom crop in ORIGINAL coords. Surfaced for
    # provenance + frontend visualisation; None when refine never ran.
    refine_origin: tuple[int, int] | None = field(default=None)
    # Set when Pass-1 is strong but Qwen's verifier rejects a low-quality crop.
    soft_verified: bool = False
    soft_verify_reason: str | None = field(default=None)


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


def _logo_foreground_aspect(logo_path: Path) -> float | None:
    """Estimate the logo shape aspect ratio from non-white line-art pixels."""
    try:
        with Image.open(logo_path) as img:
            arr = np.asarray(img.convert("L"))
    except Exception:
        return None
    mask = arr < _LOGO_FOREGROUND_THRESHOLD
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    if width <= 0 or height <= 0:
        return None
    return width / height


def _loose_final_bbox_reject_reason(
    ans: VerifyAnswer,
    ref_bbox: tuple[int, int, int, int],
    logo_path: Path,
) -> str | None:
    """Reject loose accepted crops whose final bbox shape is implausible.

    A `fit=loose` verifier answer only says the mark is somewhere in the crop;
    it is not enough to ship a final bbox that still has the wrong shape. Use
    the registered line-art's foreground aspect as a cheap guardrail against
    boxes that include large unrelated vertical or horizontal regions, including
    bad `suggested_bbox` values from the verifier.
    """
    if ans.fit != "loose":
        return None
    x1, y1, x2, y2 = ref_bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    bbox_aspect = bw / bh
    logo_aspect = _logo_foreground_aspect(logo_path)
    if logo_aspect is None or logo_aspect <= 0:
        return None
    mismatch = max(bbox_aspect / logo_aspect, logo_aspect / bbox_aspect)
    if mismatch <= _LOOSE_BBOX_MAX_ASPECT_MISMATCH:
        return None
    return (
        "loose bbox rejected: final bbox aspect "
        f"{bbox_aspect:.2f} differs from logo aspect {logo_aspect:.2f}"
    )


def _is_blank_verify_reason(reason: str | None) -> bool:
    """True when the verify rejection signals "crop was blank, not a wrong logo"."""
    if not reason:
        return False
    low = reason.lower()
    return any(p in low for p in BLANK_VERIFY_PATTERNS)


def _is_soft_verify_reject_reason(reason: str | None) -> bool:
    if not reason:
        return False
    low = reason.lower()
    return any(p in low for p in SOFT_VERIFY_REJECT_PATTERNS)


def _contains_any(text: str | None, patterns: tuple[str, ...]) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(p in low for p in patterns)


def _edge_recall_distance(primary: MatchResult) -> float | None:
    low = (primary.reason or "").lower()
    if "edge_recall:" not in low or "distance=" not in low:
        return None
    raw = low.split("distance=", 1)[1].split(maxsplit=1)[0].strip(" ,;")
    try:
        return float(raw)
    except ValueError:
        return None


def _is_design_surface_verify_reject(
    primary: MatchResult,
    reason: str | None,
) -> bool:
    """True for design-patent surface-pattern hits rejected as product-shape mismatches."""
    if not primary.prompt_version.startswith("design"):
        return False
    text = f"{primary.reason or ''} {reason or ''}"
    if not _contains_any(text, DESIGN_SURFACE_TERMS):
        return False
    if _contains_any(reason, DESIGN_SURFACE_HARD_REJECT_TERMS):
        return False
    if not (
        _contains_any(reason, DESIGN_SHAPE_REJECT_TERMS)
        or _contains_any(reason, DESIGN_SURFACE_REJECT_TERMS)
    ):
        return False
    if (
        primary.confidence >= DESIGN_SURFACE_SOFT_MIN_CONFIDENCE
    ):
        return True
    edge_distance = _edge_recall_distance(primary)
    return (
        edge_distance is not None
        and edge_distance <= DESIGN_EDGE_RECALL_SOFT_MAX_DISTANCE
    )


def _should_soft_accept_verify_reject(
    primary: MatchResult,
    reason: str | None,
    photo_path: Path,
) -> bool:
    """Protect strong bbox hits from verifier false rejects.

    Two narrowly-scoped cases are rescued:
    - design-patent surface/ornamentation matches where the verifier rejects
      only because the evidence is a different product carrier;
    - historical Qwen low-quality false rejects on non-blank crops.
    """
    if not primary.found or primary.bbox is None:
        return False
    design_surface_reject = _is_design_surface_verify_reject(primary, reason)
    qwen_low_quality_reject = (
        primary.confidence >= 0.8
        and primary.bbox_coord_mode == "qwen_normalized_1000"
        and _is_soft_verify_reject_reason(reason)
    )
    if not (design_surface_reject or qwen_low_quality_reject):
        return False
    stats = _bbox_blank_stats(photo_path, primary.bbox)
    if stats is None:
        return False
    std, white = stats
    return white <= _PRE_GATE_WHITE_RATIO and std >= 8.0


def _stats_from_rgb_array(arr: np.ndarray[Any, Any]) -> tuple[float, float] | None:
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


def _verify_upscale_factor_for(longest: int) -> int:
    """Scale factor for verifier crops that are too small to inspect reliably."""
    if longest <= 0 or longest >= _VERIFY_UPSCALE_THRESHOLD:
        return 1
    if longest < 100:
        return 4
    return (_VERIFY_UPSCALE_THRESHOLD + longest - 1) // longest


def _maybe_upscale_for_verify(crop_path: Path) -> tuple[Path, int]:
    """Return a verifier input path plus the coordinate scale factor.

    The original crop remains the coordinate frame for `_accept_verify`, so any
    verifier-suggested bbox must be divided by this factor before use.
    """
    with Image.open(crop_path) as img:
        w, h = img.size
        scale = _verify_upscale_factor_for(max(w, h))
        if scale <= 1:
            return crop_path, 1
        new_size = (w * scale, h * scale)
        upscaled = img.convert("RGB").resize(new_size, Image.LANCZOS)

    with tempfile.NamedTemporaryFile(suffix=crop_path.suffix or ".png", delete=False) as tmp:
        out = Path(tmp.name)
    save_fmt = "JPEG" if out.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
    if save_fmt == "JPEG":
        upscaled.save(out, format=save_fmt, quality=92)
    else:
        upscaled.save(out, format=save_fmt)
    return out, scale


def _downscale_suggested_bbox(
    answer: VerifyAnswer,
    scale: int,
) -> VerifyAnswer:
    if scale <= 1 or answer.suggested_bbox is None:
        return answer
    sx1, sy1, sx2, sy2 = answer.suggested_bbox
    suggested = (
        int(round(sx1 / scale)),
        int(round(sy1 / scale)),
        int(round(sx2 / scale)),
        int(round(sy2 / scale)),
    )
    return VerifyAnswer(
        contains_full_logo=answer.contains_full_logo,
        fit=answer.fit,
        confidence=answer.confidence,
        reason=answer.reason,
        suggested_bbox=suggested,
        raw_response=answer.raw_response,
        prompt_version=answer.prompt_version,
        model=answer.model,
        usage=answer.usage,
    )


def match_with_verify(
    logo_path: Path,
    photo_path: Path,
    settings: Settings | None = None,
    client: OpenAI | None = None,
    max_iters: int = 2,
    verify_threshold: float = 0.6,
    model: str | None = None,
    refine: bool | None = None,
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

    `refine`: iter-9 refine pass. When None we read the `LINEBASE_REFINE` env
    var (default ON). When True/False, override the env. Refine fires only
    when the verify call accepts a `fit=loose` bbox — it re-asks the SAME
    primary model on a +30%-padded zoom-crop, then re-verifies the refined
    bbox; if both succeed, the final_bbox shifts to the refined one. Costs
    +1 primary call + +1 verify call when it fires.
    """
    settings = settings or Settings.from_env()
    refine_enabled = _refine_env_enabled() if refine is None else bool(refine)

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
    soft_verified = False
    soft_verify_reason: str | None = None
    verify, crop_bbox_used, pre_gated, _crop_stats = _verify_round(
        primary.bbox, photo_path, logo_path, orig_w, orig_h,
        settings=settings, client=client, verify_model=verify_model,
    )
    cx1, cy1, cx2, cy2 = crop_bbox_used

    # Refine bookkeeping (populated inside the refine branch below).
    refined_used = False
    refine_bbox_used: tuple[int, int, int, int] | None = None
    refine_origin_used: tuple[int, int] | None = None

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
        loose_reject_reason = (
            _loose_final_bbox_reject_reason(verify, fbb, logo_path)
            if ok and fbb is not None
            else None
        )
        if loose_reject_reason:
            ok = False
            fbb = None
        verified = ok
        final_bbox = fbb
        fit_label_used = verify.fit
        verify_confidence_used = verify.confidence
        verify_reason_used = loose_reject_reason or verify.reason
        verify_usage_used = verify.usage
        if (
            not ok
            and loose_reject_reason is None
            and _should_soft_accept_verify_reject(primary, verify.reason, photo_path)
        ):
            verified = True
            final_bbox = primary.bbox
            fit_label_used = "loose"
            soft_verified = True
            soft_verify_reason = verify.reason

        # ---- Iter 9 refine: tighten loose bboxes via a zoomed re-ask -----
        if (
            ok
            and refine_enabled
            and verify.fit == "loose"
            and primary.bbox is not None
        ):
            ref_outcome = _refine_round(
                logo_path, photo_path, primary.bbox,
                orig_w=orig_w, orig_h=orig_h,
                settings=settings, client=client, model=model,
                verify_model=verify_model, verify_threshold=verify_threshold,
            )
            if ref_outcome is not None:
                refined_bbox, refine_origin, refine_usage_total = ref_outcome
                refined_used = True
                refine_bbox_used = refined_bbox
                refine_origin_used = refine_origin
                final_bbox = refined_bbox
                if refine_usage_total:
                    verify_usage_used = _sum_int_dicts(
                        verify_usage_used, refine_usage_total
                    )

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
                    loose_reject_reason2 = (
                        _loose_final_bbox_reject_reason(v2, fbb2, logo_path)
                        if ok2 and fbb2 is not None
                        else None
                    )
                    if loose_reject_reason2:
                        ok2 = False
                        fbb2 = None
                    if v2.usage:
                        verify_usage_used = _sum_int_dicts(verify_usage_used, v2.usage)
                    fit_label_used = v2.fit
                    verify_confidence_used = v2.confidence
                    verify_reason_used = loose_reject_reason2 or v2.reason
                    if ok2:
                        verified, final_bbox, retry_bbox_used = True, fbb2, new_bbox
                        # Refine can chain with retry: when the retry's verify
                        # says fit=loose, try tightening the retry's bbox.
                        if refine_enabled and v2.fit == "loose":
                            ref_outcome = _refine_round(
                                logo_path, photo_path, new_bbox,
                                orig_w=orig_w, orig_h=orig_h,
                                settings=settings, client=client, model=model,
                                verify_model=verify_model,
                                verify_threshold=verify_threshold,
                            )
                            if ref_outcome is not None:
                                rb, ro, ru = ref_outcome
                                refined_used = True
                                refine_bbox_used = rb
                                refine_origin_used = ro
                                final_bbox = rb
                                if ru:
                                    verify_usage_used = _sum_int_dicts(
                                        verify_usage_used, ru
                                    )
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
        refined=refined_used,
        refine_bbox=refine_bbox_used,
        refine_origin=refine_origin_used,
        soft_verified=soft_verified,
        soft_verify_reason=soft_verify_reason,
    )


def _refine_round(
    logo_path: Path,
    photo_path: Path,
    region_bbox: tuple[int, int, int, int],
    *,
    orig_w: int,
    orig_h: int,
    settings: Settings,
    client: OpenAI | None,
    model: str | None,
    verify_model: str,
    verify_threshold: float,
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int],
    dict[str, int] | None,
] | None:
    """Crop the photo around `region_bbox` (with +30% pad), re-ask the SAME
    primary model for a tight bbox, then verify that refined bbox.

    Returns `(refined_bbox_in_orig_coords, zoom_origin, summed_usage)` when
    refine succeeds AND the re-verify accepts it. Returns None on any of:
      - refine call raises
      - refine returns `found=False` or no bbox
      - refined bbox area is zero / degenerate
      - re-verify pre-gate trips or verify says `wrong` / low confidence

    """
    try:
        refine_res, zoom_origin = match_logo_in_zoomed(
            logo_path, photo_path,
            region_bbox=region_bbox,
            settings=settings,
            client=client,
            model=model,
        )
    except Exception:
        return None
    if not refine_res.found or refine_res.bbox is None:
        return None
    rx1, ry1, rx2, ry2 = refine_res.bbox
    zx0, zy0 = zoom_origin
    cand = (zx0 + rx1, zy0 + ry1, zx0 + rx2, zy0 + ry2)
    cand = _clamp_bbox(cand, orig_w, orig_h)
    if cand[2] <= cand[0] or cand[3] <= cand[1]:
        return None

    # Re-verify the refined bbox; if pre-gate trips or verify says wrong,
    # fall back to the pre-refine bbox (handled by caller via None return).
    v2, _cbox2, pre2, _st2 = _verify_round(
        cand, photo_path, logo_path, orig_w, orig_h,
        settings=settings, client=client, verify_model=verify_model,
    )
    usage_total: dict[str, int] | None = refine_res.usage
    if pre2:
        return None
    assert v2 is not None
    if v2.usage:
        usage_total = _sum_int_dicts(usage_total, v2.usage)
    ok2, fbb2 = _accept_verify(
        v2, cand, _cbox2, orig_w, orig_h, verify_threshold,
    )
    if not ok2 or fbb2 is None:
        return None
    if _loose_final_bbox_reject_reason(v2, fbb2, logo_path):
        return None
    # _accept_verify may have applied a tighter shrink via suggested_bbox or
    # an expansion for `too_tight`; surface that as the refined final.
    return fbb2, zoom_origin, usage_total


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
        verify_input, verify_scale = _maybe_upscale_for_verify(tmp_path)
        try:
            answer = verify_crop(
                logo_path, verify_input, settings=settings, client=client, model=verify_model
            )
        finally:
            if verify_input != tmp_path:
                with contextlib.suppress(OSError):
                    verify_input.unlink(missing_ok=True)
        answer = _downscale_suggested_bbox(answer, verify_scale)
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
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
    "_refine_env_enabled",
]
