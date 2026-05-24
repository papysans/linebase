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

from openai import OpenAI
from PIL import Image

from linebase.config import Settings
from linebase.crop import crop_to_bbox
from linebase.llm import MatchResult, VerifyAnswer, match_logo_in_photo, verify_crop


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

    pad_ratio = 0.20
    crop_bbox = _padded_bbox(primary.bbox, pad_ratio, orig_w, orig_h)
    cx1, cy1, cx2, cy2 = crop_bbox

    # Write the padded crop to a tempfile (PNG keeps things lossless).
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        crop_to_bbox(photo_path, crop_bbox, tmp_path, pad_ratio=0.0)
        # When the caller pinned a `model`, route the verify call to the SAME
        # model so per-job overrides apply to both passes (previously verify
        # silently used settings.review_model regardless). With no override,
        # keep the legacy behavior of using settings.review_model.
        verify_model = model or settings.review_model
        verify = verify_crop(
            logo_path, tmp_path, settings=settings, client=client, model=verify_model
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    fit = verify.fit
    final_bbox: tuple[int, int, int, int] | None = primary.bbox

    # --- Acceptance rules ---------------------------------------------------
    if (
        verify.contains_full_logo
        and fit in ("tight", "loose")
        and verify.confidence >= verify_threshold
    ):
        verified = True
        # Optionally tighten the bbox when verify reports a `loose` fit AND a
        # suggested_bbox in CROP coords — translate back to ORIGINAL coords.
        if fit == "loose" and verify.suggested_bbox is not None:
            sx1, sy1, sx2, sy2 = verify.suggested_bbox
            cand = (cx1 + sx1, cy1 + sy1, cx1 + sx2, cy1 + sy2)
            cand = _clamp_bbox(cand, orig_w, orig_h)
            # Only accept the shrink if it lies inside the original (un-padded) bbox
            # area-wise (we never want verify to grow the box on "loose"; growth is
            # only allowed via the `too_tight` branch).
            ox1, oy1, ox2, oy2 = primary.bbox
            if cand[0] >= cx1 and cand[1] >= cy1 and cand[2] <= cx2 and cand[3] <= cy2:
                final_bbox = cand
            else:
                final_bbox = primary.bbox
        else:
            final_bbox = primary.bbox

    elif fit == "too_tight":
        # Expand by 20% per side (clamped). Don't re-call the LLM.
        final_bbox = _padded_bbox(primary.bbox, 0.20, orig_w, orig_h)
        verified = True

    else:
        # Reject — either contains_full_logo=False, fit==wrong, or confidence too low.
        verified = False
        final_bbox = None

    return VerifyResult(
        primary=primary,
        verified=verified,
        final_bbox=final_bbox,
        fit_label=fit,
        verify_confidence=verify.confidence,
        verify_reason=verify.reason,
        iters=2,
        verify_usage=verify.usage,
    )


__all__ = ["VerifyResult", "VerifyAnswer", "match_with_verify"]
