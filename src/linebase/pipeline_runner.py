"""Background job orchestration + SSE event bus.

A job runs as one asyncio task that:
  1. resolves the row list from the uploaded xlsx,
  2. for each row: download evidences, call LLM per evidence, crop best, persist to DB,
  3. publishes events to a per-job asyncio.Queue that the SSE endpoint consumes.

Cost uses a per-model USD/1M-token table (see research/llm-pricing.md). Reasoning
tokens are billed as output tokens by every provider in MODEL_PRICING, so the
usage.completion_tokens count already includes them — no separate column needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from linebase import store
from linebase.config import Settings
from linebase.crop import crop_to_bbox
from linebase.edge_refine import edge_refine_bbox
from linebase.fetch import fetch
from linebase.llm import MatchResult, match_logo_in_photo, verify_crop
from linebase.sift_refine import sift_refine_bbox
from linebase.verify_loop import VerifyResult, _bbox_blank_stats, match_with_verify

_log = logging.getLogger(__name__)


def _verify_enabled() -> bool:
    """Opt-in via env: LINEBASE_VERIFY=1 (also accepts true/yes/on, case-insensitive)."""
    return os.environ.get("LINEBASE_VERIFY", "").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str, *, default: bool) -> bool:
    """Env-flag reader with explicit default. Accepts 1/true/yes/on (case-insensitive)
    for true and 0/false/no/off for false; empty / unset returns `default`.

    Used by the SIFT refine + recall toggles below, both default ON.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _sift_refine_enabled() -> bool:
    """Iter 10 — SIFT refine after Pass-1. DEFAULT OFF as of iter-10 review:
    line-art USPTO logos vs colored printed logos in real-world photos share
    almost no SIFT keypoints, so refine fires <5% of the time AND when it
    does, the homography between mis-matched feature spaces produces wild
    bboxes (truth-set IoU regressed 0.45→0.23 on 4334451_pair_03). Enable
    via LINEBASE_SIFT_REFINE=1 when you have textured (filled) logos rather
    than line-art outlines — the iter-10 commit message has the full
    explanation. The next iter should pursue edge-based shape matching
    (Canny + matchShapes / Hu moments) per USPTO patent 9536171, which is
    the correct technique for line-art ↔ printed-logo matching.
    """
    return _env_flag("LINEBASE_SIFT_REFINE", default=False)


def _sift_recall_enabled() -> bool:
    """Iter 10 — SIFT whole-photo recall lifter for Pass-1 found=False rows.
    DEFAULT OFF — same line-art/SIFT mismatch as the refine path; enabling
    didn't recover any of the 8 NONE cases on the truth set. Toggle via
    LINEBASE_SIFT_RECALL=1.
    """
    return _env_flag("LINEBASE_SIFT_RECALL", default=False)


# Iter 10 — stricter inlier floor for the whole-photo recall path. The
# refine path already has the VLM's region narrowing things down; recall
# searches the whole photo and so demands more geometric agreement before
# overruling the VLM's "not found" verdict.
_SIFT_RECALL_MIN_INLIERS = 10


def _edge_refine_enabled() -> bool:
    """Iter 11 — Edge-based shape matching after Pass-1. DEFAULT ON.

    Replaces SIFT for the line-art ↔ printed-logo case. Pure CV (zero LLM
    cost): Canny → findContours → matchShapes with Hu moments inside the
    3x-expanded VLM region. Per USPTO patent 9536171 "Logo detection by edge
    matching", this is the correct technique when the logo silhouette is
    shared between the two images but their interior textures are not.
    Toggle via ``LINEBASE_EDGE_REFINE=0``.
    """
    return _env_flag("LINEBASE_EDGE_REFINE", default=True)


def _edge_recall_enabled() -> bool:
    """Iter 11 — Whole-photo edge match for Pass-1 found=False rows. DEFAULT
    ON. Same shape-matching algorithm as the refine path but applied to the
    whole photo. Uses a stricter shape-distance threshold (``< 0.5``) than
    the refine path (``< 1.0``) because the whole-photo search space is much
    larger and the prior on a real logo being present is weaker. Toggle via
    ``LINEBASE_EDGE_RECALL=0``.
    """
    return _env_flag("LINEBASE_EDGE_RECALL", default=True)


def _is_design_prompt_result(result: MatchResult) -> bool:
    return result.prompt_version.startswith("design")


# Iter 11 — stricter shape-distance ceiling for the whole-photo recall path.
# matchShapes I1 distances cluster near 0 for true matches and balloon past
# 1.0 for unrelated contours; the refine path is forgiven up to 1.0 because
# the VLM already narrowed the region, but recall has no such narrowing and
# so demands a tighter match before overruling the VLM's "not found".
_EDGE_RECALL_MAX_DISTANCE = 0.5


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    """Read a positive-integer env var with a default + floor.

    Used for the per-evidence concurrency knobs below. A bad value (negative,
    zero, non-numeric) falls back to `default` silently — the runtime guarantee
    we care about is "always a workable cap", not "yell at the operator".
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(min_value, v)


# Per-evidence LLM concurrency caps. Read at module import — change via env
# before starting uvicorn. Two knobs:
#
#   LINEBASE_LLM_CONCURRENCY         (default 6)  → applied to Ark + SiliconFlow
#                                                   (Doubao explicitly allows
#                                                   multi-concurrent requests).
#   LINEBASE_LLM_CONCURRENCY_OPENAI  (default 2)  → applied to the OpenAI / 1m1ng
#                                                   relay, which has a tighter
#                                                   per-key cap than Ark.
#
# When LINEBASE_LLM_CONCURRENCY=1 the gather path degenerates to single-flight
# per row — observable behavior matches the pre-parallel implementation.
LLM_CONCURRENCY = _env_int("LINEBASE_LLM_CONCURRENCY", 6)
LLM_CONCURRENCY_OPENAI = _env_int("LINEBASE_LLM_CONCURRENCY_OPENAI", 2)


def _concurrency_for_model(settings: Settings, model: str) -> int:
    """Return the per-evidence concurrency cap for a model id.

    Strategy: resolve the provider via `Settings.resolve_provider` and use the
    OpenAI-specific cap when that provider is `openai` (i.e. the 1m1ng relay or
    direct OpenAI). Anything else (Ark / SiliconFlow) gets the broader cap.

    Defensive fallback: if provider resolution raises (e.g. an unconfigured
    provider for a typo'd model id), fall back to `LLM_CONCURRENCY_OPENAI` since
    that's the conservative choice.
    """
    try:
        pc = settings.resolve_provider(model)
    except Exception:
        return LLM_CONCURRENCY_OPENAI
    if pc.name == "openai":
        return LLM_CONCURRENCY_OPENAI
    return LLM_CONCURRENCY


DATA_DIR = store.DATA_DIR


# --- Event bus --------------------------------------------------------------

_buses: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)
_bus_lock = asyncio.Lock()


async def subscribe(job_id: str) -> asyncio.Queue[dict[str, Any]]:
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
    async with _bus_lock:
        _buses[job_id].append(q)
    return q


async def unsubscribe(job_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
    async with _bus_lock:
        if q in _buses[job_id]:
            _buses[job_id].remove(q)


async def publish(job_id: str, event: dict[str, Any]) -> None:
    async with _bus_lock:
        for q in list(_buses[job_id]):
            with suppress(asyncio.QueueFull):
                q.put_nowait(event)


def _row_to_dict(row: store.JobRow) -> dict[str, Any]:
    d = asdict(row)
    d["evidence_urls"] = json.loads(d.pop("evidence_urls_json"))
    d["all_crops"] = json.loads(d.pop("all_crops_json"))
    meta_raw = d.pop("match_meta_json", None)

    # Surface the chosen-best evidence's LLM scalars at the top level so the
    # frontend can render metric chips without re-parsing the meta dict.
    # Rule: pick the entry with the highest confidence among entries that
    # survived verify + sanity (when present). This mirrors `_process_row`'s
    # post-verify re-rank so the REST shape's `best_*` fields agree with the
    # `best_crop_path` chosen by the pipeline.
    best_url: str | None = None
    best: dict[str, Any] | None = None
    try:
        meta = json.loads(meta_raw) if meta_raw else {}
    except Exception:
        meta = {}
    crops_map = d.get("all_crops") or {}
    if isinstance(meta, dict):
        # Two-pass selection: prefer entries with a real crop on disk +
        # verified=True/None (i.e. not False) + no sanity_rejected. If none
        # qualify (e.g. older rows from before this column existed), fall
        # back to the original "max confidence among found=True" rule.
        def _qualifies(m: dict[str, Any], url: str) -> bool:
            if not m.get("found"):
                return False
            if m.get("sanity_rejected"):
                return False
            if m.get("verified") is False:
                return False
            return bool(crops_map.get(url))

        for url, m in meta.items():
            if not isinstance(m, dict) or not _qualifies(m, url):
                continue
            conf = float(m.get("confidence") or 0.0)
            if best is None or conf > float(best.get("confidence") or 0.0):
                best = m
                best_url = url
        if best is None:
            # Legacy fallback for rows pre-dating the post-verify columns.
            for url, m in meta.items():
                if not isinstance(m, dict) or not m.get("found"):
                    continue
                conf = float(m.get("confidence") or 0.0)
                if best is None or conf > float(best.get("confidence") or 0.0):
                    best = m
                    best_url = url

    def _f(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    d["best_evidence_url"] = best_url
    d["best_confidence"] = _f(best.get("confidence")) if best else None
    d["best_clarity"] = _f(best.get("clarity")) if best else None
    d["best_completeness"] = _f(best.get("completeness")) if best else None
    d["best_isolation"] = _f(best.get("isolation")) if best else None
    d["best_reason"] = (best.get("reason") if best else None) or None
    # Per-evidence bbox: list of [x1, y1, x2, y2] in pixel coords of the chosen
    # evidence image. Exposed so the review-detail modal can overlay the LLM's
    # bbox on top of the full-size evidence image. None when no best was found
    # or the model didn't return a bbox (rare — `found: true` without bbox is
    # treated as a malformed result by the matcher, but stay defensive).
    best_bbox: list[float] | None = None
    if best:
        raw_bbox = best.get("bbox")
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            try:
                best_bbox = [float(x) for x in raw_bbox]
            except (TypeError, ValueError):
                best_bbox = None
    d["best_bbox"] = best_bbox
    # When the primary vision model rejected the evidence (e.g. Qwen3-VL's
    # <28 px tile error) and pipeline_runner retried with gpt-5.5, surface
    # that on the row so the reviewer sees a "回落 gpt-5.5" pill.
    d["best_fallback_model"] = (best.get("fallback_model") if best else None) or None

    # Per-evidence meta projection for the row-detail modal: surface the verify
    # outcome + sanity rejection reason so the reviewer can understand WHY a
    # sibling evidence was skipped over (e.g. "verified=False fit=wrong" or
    # "sanity_rejected=crop_mostly_blank"). Schema deliberately small — only
    # the fields the modal renders. Empty dict when no meta yet (pending row).
    #
    # bbox + reason + clarity/completeness/isolation added 2026-05-24 so the
    # modal can overlay each evidence's own bbox + render that evidence's
    # metrics when the user clicks a non-best thumbnail (was previously stuck
    # showing only the chosen-best's bbox + metrics).
    match_meta: dict[str, dict[str, Any]] = {}
    if isinstance(meta, dict):
        for url, m in meta.items():
            if not isinstance(m, dict):
                continue
            raw_bbox = m.get("bbox")
            ev_bbox: list[float] | None = None
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                try:
                    ev_bbox = [float(x) for x in raw_bbox]
                except (TypeError, ValueError):
                    ev_bbox = None
            # Iter 9 bug fix: legacy rows from before the source-side null-out
            # may still carry the original (rejected) bbox alongside the
            # `pass1_blank_reject` flag. Force bbox=None during projection so
            # the API surface and the frontend overlay don't show a phantom
            # box on a rejected blank region.
            if m.get("pass1_blank_reject"):
                ev_bbox = None
            # Iter 5 retry surface: only emit `retried` / `retry_bbox` when the
            # backing pipeline actually fired the Pass-3 retry, so older rows
            # don't acquire stray null keys in their API shape.
            retry_bbox_raw = m.get("retry_bbox")
            retry_bbox_proj: list[float] | None = None
            if isinstance(retry_bbox_raw, (list, tuple)) and len(retry_bbox_raw) == 4:
                try:
                    retry_bbox_proj = [float(x) for x in retry_bbox_raw]
                except (TypeError, ValueError):
                    retry_bbox_proj = None
            entry: dict[str, Any] = {
                "found": m.get("found"),
                "confidence": _f(m.get("confidence")),
                "clarity": _f(m.get("clarity")),
                "completeness": _f(m.get("completeness")),
                "isolation": _f(m.get("isolation")),
                "reason": m.get("reason"),
                "bbox": ev_bbox,
                "verified": m.get("verified"),
                "fit": m.get("fit"),
                "verify_reason": m.get("verify_reason"),
                "sanity_rejected": m.get("sanity_rejected"),
                "fallback_model": m.get("fallback_model"),
                "error": m.get("error"),
            }
            if m.get("retried"):
                entry["retried"] = True
                if m.get("retry_reason"):
                    entry["retry_reason"] = m.get("retry_reason")
                if retry_bbox_proj is not None:
                    entry["retry_bbox"] = retry_bbox_proj
            # Iter 9 — refine pass provenance. Only emit keys when refine
            # actually fired so older rows stay null-key-clean.
            if m.get("refined"):
                entry["refined"] = True
                raw_refine_bbox = m.get("refine_bbox")
                if isinstance(raw_refine_bbox, (list, tuple)) and len(raw_refine_bbox) == 4:
                    with suppress(TypeError, ValueError):
                        entry["refine_bbox"] = [float(x) for x in raw_refine_bbox]
                raw_refine_origin = m.get("refine_origin")
                if isinstance(raw_refine_origin, (list, tuple)) and len(raw_refine_origin) == 2:
                    with suppress(TypeError, ValueError):
                        entry["refine_origin"] = [
                            int(raw_refine_origin[0]),
                            int(raw_refine_origin[1]),
                        ]
            # Iter 10 — SIFT refine / recall provenance. Two independent
            # paths emit through this branch:
            #   - refine: VLM gave a bbox, SIFT tightened it → `sift_refined`
            #   - recall: VLM said found=false, SIFT recovered → `sift_recall_hit`
            # Both surface inlier count for the modal's debug strip. Original
            # bbox is recorded only for the refine path (recall has no
            # original to compare against).
            if m.get("sift_refined"):
                entry["sift_refined"] = True
                si_raw = m.get("sift_inliers")
                if si_raw is not None:
                    with suppress(TypeError, ValueError):
                        entry["sift_inliers"] = int(si_raw)
                raw_orig_sift = m.get("sift_original_bbox")
                if isinstance(raw_orig_sift, (list, tuple)) and len(raw_orig_sift) == 4:
                    with suppress(TypeError, ValueError):
                        entry["sift_original_bbox"] = [float(x) for x in raw_orig_sift]
            if m.get("sift_recall_hit"):
                entry["sift_recall_hit"] = True
                # Recall path may share `sift_inliers` with refine — only
                # emit when refine didn't already set it (avoid duplicate
                # int-cast on the same value).
                if "sift_inliers" not in entry:
                    si_raw = m.get("sift_inliers")
                    if si_raw is not None:
                        with suppress(TypeError, ValueError):
                            entry["sift_inliers"] = int(si_raw)
            # Iter 11 — Edge refine / recall provenance. Mirrors the SIFT
            # branches above (refine = tighten VLM bbox via contour shape
            # match; recall = recover when VLM said found=false). Both
            # surface the shape distance + candidate count for the modal's
            # debug strip; original bbox is recorded only for the refine
            # path (recall has no original to compare against).
            if m.get("edge_refined"):
                entry["edge_refined"] = True
                sd_raw = m.get("edge_shape_distance")
                if sd_raw is not None:
                    with suppress(TypeError, ValueError):
                        entry["edge_shape_distance"] = float(sd_raw)
                cc_raw = m.get("edge_candidates_checked")
                if cc_raw is not None:
                    with suppress(TypeError, ValueError):
                        entry["edge_candidates_checked"] = int(cc_raw)
                raw_orig_edge = m.get("edge_original_bbox")
                if isinstance(raw_orig_edge, (list, tuple)) and len(raw_orig_edge) == 4:
                    with suppress(TypeError, ValueError):
                        entry["edge_original_bbox"] = [float(x) for x in raw_orig_edge]
            if m.get("edge_recall_hit"):
                entry["edge_recall_hit"] = True
                if "edge_shape_distance" not in entry:
                    sd_raw = m.get("edge_shape_distance")
                    if sd_raw is not None:
                        with suppress(TypeError, ValueError):
                            entry["edge_shape_distance"] = float(sd_raw)
                if "edge_candidates_checked" not in entry:
                    cc_raw = m.get("edge_candidates_checked")
                    if cc_raw is not None:
                        with suppress(TypeError, ValueError):
                            entry["edge_candidates_checked"] = int(cc_raw)
            # Iter 7 — Pass-1 variance pre-gate provenance. Only emit when
            # the gate actually tripped so older rows don't grow null noise.
            if m.get("pass1_blank_reject"):
                entry["pass1_blank_reject"] = True
                entry["pass1_blank_std"] = _f(m.get("pass1_blank_std"))
                entry["pass1_blank_white"] = _f(m.get("pass1_blank_white"))
                raw_orig = m.get("pass1_original_bbox")
                if isinstance(raw_orig, (list, tuple)) and len(raw_orig) == 4:
                    with suppress(TypeError, ValueError):
                        entry["pass1_original_bbox"] = [float(x) for x in raw_orig]
                if m.get("pass1_original_reason"):
                    entry["pass1_original_reason"] = m.get("pass1_original_reason")
            # Iter 6.3 — surface tile-scan provenance when the fallback fired.
            # Only emit these keys when actually set so older rows don't grow
            # null noise in the API shape.
            # Iter 6.4 — also surface `tile_attempts` (how many tiles produced
            # a non-degenerate found-candidate) and `tile_verified_idx` (which
            # tile won the verify pass, when verify ran inside tile-scan).
            if m.get("tile_scanned"):
                entry["tile_scanned"] = True
                tile_origin = m.get("tile_origin")
                if isinstance(tile_origin, (list, tuple)) and len(tile_origin) == 2:
                    with suppress(TypeError, ValueError):
                        entry["tile_origin"] = [int(tile_origin[0]), int(tile_origin[1])]
                if m.get("tile_index"):
                    entry["tile_index"] = m.get("tile_index")
                ta_raw = m.get("tile_attempts")
                if ta_raw is not None:
                    with suppress(TypeError, ValueError):
                        entry["tile_attempts"] = int(ta_raw)
                if m.get("tile_verified_idx"):
                    entry["tile_verified_idx"] = m.get("tile_verified_idx")
                # Iter 6.5 — surface verify_upscale only when non-default
                # (>1), so older rows that pre-date the field don't grow a
                # noisy `verify_upscale: 1` key in their API shape.
                vu_raw = m.get("verify_upscale")
                if vu_raw is not None:
                    try:
                        vu = int(vu_raw)
                        if vu > 1:
                            entry["verify_upscale"] = vu
                    except (TypeError, ValueError):
                        pass
            for diag_key in ("raw_bbox", "bbox_coord_mode", "source_size", "sent_size"):
                if diag_key in m:
                    entry[diag_key] = m.get(diag_key)
            if m.get("soft_verified"):
                entry["soft_verified"] = True
                if m.get("soft_verify_reason"):
                    entry["soft_verify_reason"] = m.get("soft_verify_reason")
            match_meta[url] = entry
    d["match_meta"] = match_meta
    return d


def _job_to_dict(job: store.Job) -> dict[str, Any]:
    d = asdict(job)
    d["sample_params"] = json.loads(d.pop("sample_params_json"))
    # `model` is already part of the dataclass, but if the DB column was added
    # by a migration and the row predates it, asdict() may still report None —
    # surface it as `model` so the frontend can always render a value.
    d["model"] = getattr(job, "model", None)
    # verify_loop is INTEGER in SQLite; project to a bool for the JSON wire so
    # frontend can use it directly in conditionals without 0/1 vs true/false
    # ambiguity.
    d["verify_loop"] = bool(getattr(job, "verify_loop", 0))
    # tile_scan: same int → bool projection as verify_loop. Surfaces the
    # iter-6.3 fallback toggle on the job summary so the frontend can render
    # a "Tile scan: ON" pill and show tile-scanned matches with provenance.
    d["tile_scan"] = bool(getattr(job, "tile_scan", 0))
    return d


# --- Pipeline ---------------------------------------------------------------

_active_tasks: dict[str, asyncio.Task[None]] = {}


# Per-model real-USD pricing (per 1,000,000 tokens) at standard / on-line tier.
# Tuple shape: (input_usd_per_1m, output_usd_per_1m, reasoning_usd_per_1m_or_None).
# When the third value is None, reasoning tokens are billed AS output tokens by
# the provider, and `usage.completion_tokens` already includes them — so a
# single output rate is correct (no double-counting). See
# .trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline/research/llm-pricing.md
# for sources, dates, and per-model notes. Last refreshed: 2026-05-24.
MODEL_PRICING: dict[str, tuple[float, float, float | None]] = {
    # OpenAI (via 1m1ng relay; markup unknown — treat these as a floor)
    "gpt-5.5":                            (5.00,  30.00, None),
    "gpt-5.4":                            (2.50,  15.00, None),
    # Volcengine Ark (Doubao); ≤32 K input bracket — see notes for segmented rates
    "doubao-seed-2-0-pro-260215":         (0.47,   2.35, None),
    "doubao-seed-2-0-mini-260428":        (0.029,  0.29, None),
    "doubao-1.5-vision-pro-250328":       (0.44,   1.32, None),
    # SiliconFlow (USD-denominated; Pro/ prefix = paid queue, same per-token cost)
    "Qwen/Qwen3-VL-30B-A3B-Instruct":     (0.29,   1.00, None),
    "Qwen/Qwen3-VL-32B-Instruct":         (0.20,   0.60, None),
    "zai-org/GLM-4.5V":                   (0.14,   0.86, None),
    "Pro/moonshotai/Kimi-K2.5":           (0.45,   2.25, None),
    "Pro/moonshotai/Kimi-K2.6":           (0.90,   4.00, None),
    "moonshotai/Kimi-K2.5":               (0.45,   2.25, None),  # non-Pro alias
    "moonshotai/Kimi-K2.6":               (0.90,   4.00, None),  # non-Pro alias
}

# Fallback for unknown model ids: use the most expensive entry in the table
# (gpt-5.5) as a defensible upper bound. Surface a one-time warning per model
# so we notice when a new id has slipped past the catalog.
_unknown_model_warned: set[str] = set()


def cost_estimate(usage: dict[str, int] | None, model: str | None = None) -> float:
    """USD spend for one LLM call, using MODEL_PRICING.

    Reasoning tokens for thinking-models (Doubao Seed 2.0 *, Kimi K2.x) are
    billed as output tokens by the provider and `usage.completion_tokens`
    already includes them on the OpenAI / Ark / SiliconFlow OpenAI-compatible
    surfaces we use — so there is no separate reasoning column to add here.

    Unknown model → fall back to gpt-5.5 rates and log once.
    """
    if not usage:
        return 0.0
    key = (model or "").strip()
    pricing = MODEL_PRICING.get(key)
    if pricing is None:
        if key and key not in _unknown_model_warned:
            _unknown_model_warned.add(key)
            _log.warning(
                "cost_estimate: unknown model %r, falling back to gpt-5.5 rates",
                key,
            )
        pricing = MODEL_PRICING["gpt-5.5"]
    in_rate, out_rate, _reasoning_rate = pricing
    return (
        usage.get("prompt_tokens", 0) * in_rate / 1_000_000
        + usage.get("completion_tokens", 0) * out_rate / 1_000_000
    )


# Qwen3-VL's image preprocessor rejects tiles < 28 px on either side. The
# upstream HTTP 400 carries phrases like
# "height(X) or width(Y) must be larger than 28" — we sniff for the stable
# substring "must be larger than 28" (and the older docs phrasing
# "must be > 28") so a wording bump doesn't bypass the fallback.
_QWEN_SMALL_IMAGE_MARKERS: tuple[str, ...] = (
    "must be larger than 28",
    "must be > 28",
)

# Model id the small-image branch falls back to. gpt-5.5 handles small tiles
# fine and is already configured in the OPENAI provider block.
_SMALL_IMAGE_FALLBACK_MODEL = "gpt-5.5"


def _is_small_image_rejection(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _QWEN_SMALL_IMAGE_MARKERS)


# Volcengine Ark surfaces "out of funds" as an HTTP 402 with a JSON body
# containing `AccountOverdueError` and/or the Chinese string "账户已欠费".
# When we see this we silently retry the SAME evidence against gpt-5.5 and
# log + SSE-warn once per job so the operator notices but the run continues.
#
# Markers are checked case-insensitively against `str(exc)` — the OpenAI SDK
# stringifies APIStatusError as something like
#   "Error code: 402 - {'error': {'message': 'AccountOverdueError: ...'}}"
# so the substring search catches both shapes (HTTP code + provider error name).
_ARK_OVERDUE_MARKERS: tuple[str, ...] = (
    "accountoverdueerror",
    "账户已欠费",
)

# Model id the Ark-overdue branch falls back to. gpt-5.5 is also the
# small-image fallback above; the choice is deliberate — one extra provider
# routing path means one place to maintain.
_ARK_OVERDUE_FALLBACK_MODEL = "gpt-5.5"


def _is_ark_overdue(exc: BaseException) -> bool:
    """True when the LLM call failed because the Ark account ran out of funds.

    We sniff three independent markers (case-insensitive against the full
    stringified exception): the provider-specific error name, the HTTP 402
    status code, and the Chinese billing message. Any one match triggers
    fallback — a provider rename of one marker won't silently bypass us.
    """
    msg = str(exc).lower()
    if any(marker in msg for marker in _ARK_OVERDUE_MARKERS):
        return True
    # The OpenAI SDK formats `APIStatusError` as "Error code: 402 - ...".
    # Match the literal "402" inside that prefix to catch billing errors from
    # providers that don't include the Ark-specific markers above.
    return "error code: 402" in msg


# Job-level "we've already warned about this provider being broke" set.
# Keyed by job_id so re-running the same job after a top-up doesn't suppress
# the warning forever. Touched only inside _one_evidence under no lock since
# asyncio is single-threaded per event loop.
_ark_overdue_warned: set[str] = set()


# Post-crop sanity-check thresholds. Added 2026-05-24 after the 74677567 row
# in job 2a2e801827dc457b yielded a `found=true conf=0.98` bbox that landed in
# a white margin of the evidence — the resulting crop file was visibly blank
# but the row still got marked `ok`. We now reject crops that are either
# absurdly tiny (< 0.5% of evidence area) or mostly featureless white space
# (mean brightness > 240 on 0-255 AND max per-channel stddev < 25).
#
# Bias is towards needs_review, not silent acceptance — see _process_row's
# downgrade path.
_SANITY_MIN_AREA_RATIO = 0.005
_SANITY_BLANK_BRIGHTNESS = 240.0
_SANITY_BLANK_MAX_STDDEV = 25.0

# Iter 7 — Pass-1 variance pre-gate thresholds. Applied as a POST-filter on
# Pass-1's returned bbox region (unpadded, in original-photo coords), BEFORE
# tile-scan / verify-verdict / crop fire. Empirical justification (truth set,
# Qwen3-VL-32B-Instruct, verify OFF, 19 pairs):
#   - All 6 hits had crop std ≥ 13 AND white_ratio ≤ 0.74.
#   - The pair (std < 10 OR white_ratio > 0.85) correctly killed 6 of 12 wrong
#     predictions (pure background / UI element / blank slider) without losing
#     any hit. Halves the wrong-region rate while keeping recall intact.
#
# These are STRICTER than verify_loop's iter-5 pre-gate constants
# (`_PRE_GATE_STD = 15.0`, `_PRE_GATE_WHITE_RATIO = 0.7`) BY DESIGN: the inner
# gate operates on the +20% padded crop the verifier would see, which has more
# margin and thus more white ratio. The outer gate looks at the raw Pass-1
# bbox region — empty white space inside the model's prediction is a much
# louder signal than a slightly white-bordered padded crop.
_PASS1_BLANK_STD_THRESHOLD = 10.0
_PASS1_BLANK_WHITE_THRESHOLD = 0.85


def _crop_sanity_check(crop_path: Path, evidence_path: Path) -> str | None:
    """Return a short rejection reason string when the crop looks bad, else None.

    Rejection rules (in order):
      1. crop_too_small  — area / evidence_area < 0.5% — bbox is microscopic.
      2. crop_mostly_blank — high brightness + low contrast across all channels
         means we cropped white/cream/featureless background, not a logo.

    Both rules are intentionally loose; a true-positive logo crop will have
    some darker strokes (low brightness) and structure (non-trivial stddev),
    while a blank-margin crop has neither.
    """
    try:
        with Image.open(crop_path) as img:
            crop = img.convert("RGB")
            cw, ch = crop.size
            stat = ImageStat.Stat(crop)
        with Image.open(evidence_path) as ev:
            ew, eh = ev.size
    except Exception as e:
        # If we can't even open the crop, that's a different kind of failure;
        # bubble it up as a sanity rejection so the row drops to needs_review.
        return f"crop_open_failed: {e}"

    if ew <= 0 or eh <= 0:
        return None  # can't compute ratio; skip the area check rather than misfire
    area_ratio = (cw * ch) / float(ew * eh)
    if area_ratio < _SANITY_MIN_AREA_RATIO:
        return f"crop_too_small (area_ratio={area_ratio:.4f})"

    mean_r, mean_g, mean_b = stat.mean
    std_r, std_g, std_b = stat.stddev
    mean_brightness = (mean_r + mean_g + mean_b) / 3.0
    max_std = max(std_r, std_g, std_b)
    if (
        mean_brightness > _SANITY_BLANK_BRIGHTNESS
        and max_std < _SANITY_BLANK_MAX_STDDEV
    ):
        return f"crop_mostly_blank (brightness={mean_brightness:.0f}, std={max_std:.0f})"
    return None


# --- Tile-scan fallback ---------------------------------------------------
# Iter 6.3 — when a primary match call fails (or its verify rejected the bbox)
# on a large evidence photo, we crop the photo into a 3x3 grid, run the
# matcher on each tile, and pick the highest-confidence tile that found the
# logo. Bboxes returned in tile-coords are translated back to original-photo
# coords. Costs up to 9 extra LLM calls per evidence — strictly opt-in.
#
# Iter 6.4 — Qwen3-VL was observed returning degenerate bboxes (width=1 or
# height=1) at conf=0.95 on some tiles, beating a non-degenerate conf=0.90
# real match on a sibling tile. Two changes:
#   1. Filter degenerate candidates (w<28 or h<28) BEFORE they reach selection
#      — Qwen3-VL's image preprocessor rejects tiles below this threshold on
#      the downstream verify call anyway, so they can never produce a usable
#      crop. Below-28 candidates are silently dropped.
#   2. When `verify_enabled=True`, verify the top-3 candidates by confidence
#      and return the first one that the verifier accepts. If all three fail
#      verify, return None (tile-scan can't recover this evidence; the row
#      stays in needs_review). This avoids the previous trap where the
#      highest-confidence-but-degenerate candidate would silently be chosen.

_TILE_SCAN_MIN_LONGEST_SIDE = 1500  # below this the global pass is fine
_TILE_SCAN_MIN_CONFIDENCE = 0.6     # per-tile floor before we accept a candidate
_TILE_SCAN_MIN_BBOX_DIM = 28        # Qwen3-VL preprocessor rejects sides < 28 px
# Iter 6.5 — bumped from 3 to 5: with 9 tiles, top-5 isn't much more expensive
# (5 verify calls vs 3) and gives genuine recall headroom when 3 false-positive
# tiles push the real answer down to rank 4.
_TILE_SCAN_VERIFY_TOP_N = 5         # how many top candidates to send through verify
# Iter 6.5 — lowered from 0.6 to 0.4. Tile-scan crops are inherently smaller and
# noisier than full-photo crops; the verifier returns lower confidence on them
# even when the match is correct. We compensate by accepting lower verify
# confidence to retain real-but-borderline matches.
_TILE_SCAN_VERIFY_THRESHOLD = 0.4   # verify-confidence floor when verify is on
# Iter 6.5 — small-crop upscale before verify. Below this longest-side (px) the
# crop gets a LANCZOS upscale so the verifier model sees enough detail to
# accept real patches. Threshold + math match `_maybe_upscale_for_verify`.
_TILE_SCAN_UPSCALE_THRESHOLD = 400


def _upscale_factor_for(longest: int) -> int:
    """Pure helper: integer scale factor for a crop with `longest`-px longest side.

    Contract: returns 1 when no upscale is needed; otherwise the smallest
    integer ``k`` such that ``k * longest >= 400`` (i.e. just enough to push
    the longest side up to the threshold), floored at 4 for tiny crops
    (< 100 px) so the verifier always sees at least 4x detail on a small patch.
    """
    if longest >= _TILE_SCAN_UPSCALE_THRESHOLD:
        return 1
    if longest < 100:
        return 4
    # ceil-divide: math.ceil(400/longest) without importing math.
    return (_TILE_SCAN_UPSCALE_THRESHOLD + longest - 1) // longest


def _maybe_upscale_for_verify(crop_path: Path) -> Path:
    """If crop is < 400 px on longest side, write a LANCZOS upscale to a new
    tempfile and return that path. Otherwise return ``crop_path`` unchanged.

    Caller is responsible for cleaning up the returned tempfile when the path
    differs from the input. Scale factor comes from `_upscale_factor_for`.
    """
    img = Image.open(crop_path)
    w, h = img.size
    longest = max(w, h)
    scale = _upscale_factor_for(longest)
    if scale <= 1:
        return crop_path
    new_size = (w * scale, h * scale)
    upscaled = img.convert("RGB").resize(new_size, Image.LANCZOS)
    fd, p = tempfile.mkstemp(suffix=crop_path.suffix or ".png")
    os.close(fd)
    tmp = Path(p)
    save_fmt = "JPEG" if crop_path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
    if save_fmt == "JPEG":
        upscaled.save(tmp, format=save_fmt, quality=92)
    else:
        upscaled.save(tmp, format=save_fmt)
    return tmp


def _translate_tile_bbox(
    tile_bbox: tuple[int, int, int, int],
    ox: int,
    oy: int,
    photo_w: int,
    photo_h: int,
) -> tuple[int, int, int, int]:
    """Translate a tile-local bbox to ORIGINAL photo coords, clamped to bounds."""
    bx1, by1, bx2, by2 = tile_bbox
    gx1 = max(0, min(photo_w - 1, ox + int(bx1)))
    gy1 = max(0, min(photo_h - 1, oy + int(by1)))
    gx2 = max(gx1 + 1, min(photo_w, ox + int(bx2)))
    gy2 = max(gy1 + 1, min(photo_h, oy + int(by2)))
    return gx1, gy1, gx2, gy2


def _clamp_photo_bbox(
    bbox: tuple[int, int, int, int],
    photo_w: int,
    photo_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    gx1 = max(0, min(photo_w - 1, int(x1)))
    gy1 = max(0, min(photo_h - 1, int(y1)))
    gx2 = max(gx1 + 1, min(photo_w, int(x2)))
    gy2 = max(gy1 + 1, min(photo_h, int(y2)))
    return gx1, gy1, gx2, gy2


def _pad_photo_bbox(
    bbox: tuple[int, int, int, int],
    pad_ratio: float,
    photo_w: int,
    photo_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)
    return _clamp_photo_bbox((x1 - pw, y1 - ph, x2 + pw, y2 + ph), photo_w, photo_h)


def _verify_recalled_bbox(
    logo_path: Path,
    photo_path: Path,
    primary: MatchResult,
    *,
    settings: Settings,
    model: str,
) -> VerifyResult:
    """Verify a pure-CV whole-photo recall bbox before it can produce a crop.

    `match_with_verify` cannot be reused here because it would run Pass-1 again
    and discard the bbox recovered by edge recall. This mirrors the tile-scan
    verifier: crop the recalled region with padding, ask the verify prompt, and
    only accept tight/loose/too_tight answers.
    """
    if primary.bbox is None:
        return VerifyResult(
            primary=primary,
            verified=False,
            final_bbox=None,
            fit_label=None,
            verify_confidence=None,
            verify_reason="edge_recall had no bbox",
            iters=1,
        )
    with Image.open(photo_path) as img:
        photo_w, photo_h = img.size
    bbox = _clamp_photo_bbox(primary.bbox, photo_w, photo_h)
    padded = _pad_photo_bbox(bbox, 0.20, photo_w, photo_h)

    fd, cname = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    crop_path = Path(cname)
    upscaled_path: Path | None = None
    try:
        crop_to_bbox(photo_path, padded, crop_path, pad_ratio=0.0)
        verify_input = _maybe_upscale_for_verify(crop_path)
        if verify_input != crop_path:
            upscaled_path = verify_input
        ans = verify_crop(logo_path, verify_input, settings=settings, model=model)
    finally:
        with suppress(OSError):
            crop_path.unlink(missing_ok=True)
        if upscaled_path is not None:
            with suppress(OSError):
                upscaled_path.unlink(missing_ok=True)

    verified = bool(
        ans.contains_full_logo
        and ans.fit in ("tight", "loose")
        and ans.confidence >= _TILE_SCAN_VERIFY_THRESHOLD
    )
    final_bbox: tuple[int, int, int, int] | None = bbox if verified else None
    if ans.fit == "loose" and ans.suggested_bbox is not None and verified:
        sx1, sy1, sx2, sy2 = ans.suggested_bbox
        suggested = _clamp_photo_bbox(
            (padded[0] + sx1, padded[1] + sy1, padded[0] + sx2, padded[1] + sy2),
            photo_w,
            photo_h,
        )
        if (
            suggested[0] >= padded[0]
            and suggested[1] >= padded[1]
            and suggested[2] <= padded[2]
            and suggested[3] <= padded[3]
        ):
            final_bbox = suggested
    elif ans.fit == "too_tight" and ans.confidence >= _TILE_SCAN_VERIFY_THRESHOLD:
        verified = True
        final_bbox = padded

    return VerifyResult(
        primary=primary,
        verified=verified,
        final_bbox=final_bbox,
        fit_label=ans.fit,
        verify_confidence=ans.confidence,
        verify_reason=ans.reason,
        iters=2,
        verify_usage=ans.usage,
    )


def _build_tile_result(
    tile_res: MatchResult,
    global_bbox: tuple[int, int, int, int],
    ox: int,
    oy: int,
    ty: int,
    tx: int,
) -> MatchResult:
    """Wrap the winning tile's MatchResult in a new one with bbox in photo coords."""
    return MatchResult(
        found=True,
        bbox=global_bbox,
        confidence=tile_res.confidence,
        reason=f"[tile-scan r{ty}c{tx} @ {ox},{oy}] {tile_res.reason}",
        raw_response=tile_res.raw_response,
        prompt_version=tile_res.prompt_version,
        model=tile_res.model,
        usage=tile_res.usage,
        clarity=tile_res.clarity,
        completeness=tile_res.completeness,
        isolation=tile_res.isolation,
        json_retry_count=tile_res.json_retry_count,
        raw_bbox=tile_res.raw_bbox,
        bbox_coord_mode=tile_res.bbox_coord_mode,
        source_size=tile_res.source_size,
        sent_size=tile_res.sent_size,
    )


def _tile_scan(
    logo_path: Path,
    photo_path: Path,
    *,
    settings: Settings,
    model: str,
    verify_enabled: bool = False,
    verify_model: str | None = None,
) -> tuple[MatchResult | None, dict[str, Any], float]:
    """Crop `photo_path` into a 3x3 grid, match each tile, return the best.

    Returns `(match_result_or_None, provenance_dict, llm_cost_delta)`.

    Provenance always carries `tile_scanned: True` when at least one tile was
    actually processed, plus `tile_attempts: <n>` (how many tiles returned a
    `found=True` candidate that survived the degenerate-bbox + confidence
    filters). When `verify_enabled=True` and a tile won verify, provenance
    also carries `tile_origin` / `tile_index` / `tile_verified_idx` of the
    winner. When `verify_enabled=False`, the highest-confidence non-degenerate
    candidate is returned (no verify call), and `tile_origin` / `tile_index`
    reflect its tile.

    `llm_cost_delta` is the sum of per-tile + per-verify call costs.

    The returned MatchResult's bbox is already translated back to ORIGINAL
    photo coordinates and clamped to the photo's bounds.
    """
    with Image.open(photo_path) as img:
        W, H = img.size
    if max(W, H) <= _TILE_SCAN_MIN_LONGEST_SIDE:
        return None, {}, 0.0

    tw = max(1, W // 3)
    th = max(1, H // 3)
    suffix = photo_path.suffix or ".png"

    # Each entry: (confidence, tile_match_result, tile_origin_x, tile_origin_y, tx, ty)
    candidates: list[tuple[float, MatchResult, int, int, int, int]] = []
    cost_delta = 0.0
    with Image.open(photo_path) as src:
        for ty in range(3):
            for tx in range(3):
                x0 = tx * tw
                y0 = ty * th
                x1 = W if tx == 2 else min(W, x0 + tw)
                y1 = H if ty == 2 else min(H, y0 + th)
                tile_img = src.crop((x0, y0, x1, y1))
                fd, tmp_name = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                tile_path = Path(tmp_name)
                try:
                    tile_img.save(tile_path)
                    try:
                        res = match_logo_in_photo(
                            logo_path, tile_path,
                            settings=settings, model=model,
                        )
                    except Exception as e:
                        _log.debug("tile-scan tile r%dc%d failed: %s", ty, tx, e)
                        continue
                    cost_delta += cost_estimate(res.usage, model=model)
                    # Filter: must be a real found result with a valid bbox
                    # AND non-degenerate sides (>= 28 px on both axes) AND
                    # above the per-tile confidence floor. Qwen3-VL has been
                    # observed returning width=1 / height=1 bboxes at high
                    # confidence — those crops are unusable downstream
                    # (verify itself rejects them with the same <28 error).
                    if not (res.found and res.bbox is not None
                            and res.confidence >= _TILE_SCAN_MIN_CONFIDENCE):
                        continue
                    bx1, by1, bx2, by2 = res.bbox
                    bw, bh = bx2 - bx1, by2 - by1
                    if bw < _TILE_SCAN_MIN_BBOX_DIM or bh < _TILE_SCAN_MIN_BBOX_DIM:
                        _log.debug(
                            "tile-scan tile r%dc%d skipped: degenerate bbox %dx%d at conf=%.2f",
                            ty, tx, bw, bh, res.confidence,
                        )
                        continue
                    candidates.append((res.confidence, res, x0, y0, tx, ty))
                finally:
                    with suppress(OSError):
                        tile_path.unlink(missing_ok=True)

    if not candidates:
        return None, {"tile_scanned": True, "tile_attempts": 0}, cost_delta

    # Highest confidence wins; on a tie Python's sort is stable so we
    # implicitly prefer earlier-iterated tiles (row-major, top-left first).
    candidates.sort(key=lambda c: c[0], reverse=True)

    # Verify-disabled path: pick the highest-confidence non-degenerate
    # candidate directly. This preserves the iter-6.3 behavior for runs
    # without verify_loop enabled.
    if not verify_enabled:
        _, best, ox, oy, tx, ty = candidates[0]
        global_bbox = _translate_tile_bbox(best.bbox, ox, oy, W, H)  # type: ignore[arg-type]
        provenance = {
            "tile_scanned": True,
            "tile_attempts": len(candidates),
            "tile_origin": [int(ox), int(oy)],
            "tile_index": f"r{ty}c{tx}",
        }
        return _build_tile_result(best, global_bbox, ox, oy, ty, tx), provenance, cost_delta

    # Verify-enabled path: try the top-N candidates in confidence order and
    # return the first that the verifier accepts. The bbox we verify is the
    # global (original-photo) bbox + 20% pad, so the verifier sees the
    # candidate in the same coordinate frame as the rest of the pipeline.
    #
    # Iter 6.5 — before sending the crop to verify, upscale it with LANCZOS
    # when it's too small (< 400 px longest side). Real patches in busy
    # full-resolution photos are routinely 200-300 px wide; without upscale
    # the verifier rejects them as too noisy/blurry. The upscale runs on the
    # PHYSICAL CROP FILE (not the bbox math), so the rest of the pipeline
    # still sees the original-coord bbox for downstream cropping/display.
    eff_verify_model = verify_model or settings.review_model
    for _conf, res, ox, oy, tx, ty in candidates[:_TILE_SCAN_VERIFY_TOP_N]:
        global_bbox = _translate_tile_bbox(res.bbox, ox, oy, W, H)  # type: ignore[arg-type]
        fd, cname = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        crop_path = Path(cname)
        upscaled_path: Path | None = None
        try:
            try:
                crop_to_bbox(photo_path, global_bbox, crop_path, pad_ratio=0.20)
            except Exception as ce:
                _log.debug(
                    "tile-scan verify-crop failed for r%dc%d: %s", ty, tx, ce,
                )
                continue
            # Compute upscale factor for provenance + verify input. Always read
            # the SOURCE crop's dimensions first so the recorded factor matches
            # what the upscale helper actually did.
            with Image.open(crop_path) as _cimg:
                _cw, _ch = _cimg.size
            upscale_factor = _upscale_factor_for(max(_cw, _ch))
            verify_input_path = _maybe_upscale_for_verify(crop_path)
            if verify_input_path != crop_path:
                upscaled_path = verify_input_path
            try:
                ans = verify_crop(
                    logo_path, verify_input_path,
                    settings=settings, model=eff_verify_model,
                )
            except Exception as ve:
                _log.debug(
                    "tile-scan verify_crop call failed for r%dc%d: %s", ty, tx, ve,
                )
                continue
            cost_delta += cost_estimate(ans.usage, model=eff_verify_model)
            accepted = bool(
                ans.contains_full_logo
                and ans.fit in ("tight", "loose")
                and ans.confidence >= _TILE_SCAN_VERIFY_THRESHOLD
            )
            if accepted:
                provenance = {
                    "tile_scanned": True,
                    "tile_attempts": len(candidates),
                    "tile_origin": [int(ox), int(oy)],
                    "tile_index": f"r{ty}c{tx}",
                    "tile_verified_idx": f"r{ty}c{tx}",
                }
                if upscale_factor > 1:
                    provenance["verify_upscale"] = upscale_factor
                return (
                    _build_tile_result(res, global_bbox, ox, oy, ty, tx),
                    provenance,
                    cost_delta,
                )
        finally:
            with suppress(OSError):
                crop_path.unlink(missing_ok=True)
            if upscaled_path is not None:
                with suppress(OSError):
                    upscaled_path.unlink(missing_ok=True)

    # All top-N candidates failed verify — return None so the caller marks
    # the row needs_review rather than producing a tile-scan false positive.
    return None, {"tile_scanned": True, "tile_attempts": len(candidates)}, cost_delta


async def _process_row(
    job: store.Job,
    row: store.JobRow,
    settings: Settings,
    run_dir: Path,
) -> tuple[float, str]:
    """Process one job_row. LLM + IO are blocking → off-loaded via run_in_executor.

    Per-evidence parallelism (added 2026-05-24):
      Evidence URLs are now matched in parallel via `asyncio.gather` bounded by
      an `asyncio.Semaphore` whose limit comes from `_concurrency_for_model`
      (Ark gets the wider cap, OpenAI/1m1ng gets the tighter cap). A 27-evidence
      row on doubao-pro drops from ~25 min to ~5 min at concurrency=6.

      Per-evidence failures (download error, LLM error, sanity-rejected crop)
      are still surfaced row-by-row in the `metas` / `crops` dicts — gather()
      with `return_exceptions=True` means one bad evidence never poisons the
      whole row.

      `LINEBASE_LLM_CONCURRENCY=1` collapses the semaphore to single-flight, so
      observable behavior matches the pre-parallel implementation byte-for-byte
      (modulo the iteration order being the original evidence order, since we
      zip evidences with results after gather).
    """
    evidences = json.loads(row.evidence_urls_json)
    if not row.logo_url or not evidences:
        store.update_job_row(row.id, status="failed", notes="missing logo_url or evidence_urls")
        return 0.0, "failed"

    loop = asyncio.get_event_loop()

    try:
        logo_path = await loop.run_in_executor(None, fetch, row.logo_url)
    except Exception as e:
        store.update_job_row(row.id, status="failed", notes=f"logo download failed: {e}")
        return 0.0, "failed"

    # Verify-loop is opt-in at TWO levels:
    #   - env LINEBASE_VERIFY=1 (process-wide default, set by deployments)
    #   - per-job job.verify_loop=1 (set by ConfigurePage or the per-row rerun
    #     dialog)
    # Either triggers it. We OR them so the per-row rerun can enable verify
    # for one job without flipping the env var globally.
    use_verify = _verify_enabled() or bool(getattr(job, "verify_loop", 0))
    # Iter 6.3 — opt-in 3x3 tile-scan fallback. When the primary match path
    # fails on a large photo (longest side > 1500 px), we re-try by tiling
    # the photo and matching each tile. Costs 9 extra LLM calls per affected
    # evidence — always strictly opt-in via the job-level flag.
    tile_scan_enabled = bool(getattr(job, "tile_scan", 0))
    # Per-job model override (added 2026-05-24). When None the lower layers
    # fall back to settings.model. Verify-loop keeps using settings.review_model
    # by default; pipeline jobs that pin a vision model only override the
    # primary match call, since the verify prompt is more sensitive to model
    # behavior.
    job_model = (job.model or "").strip() or None
    eff_model = job_model or settings.model

    # Per-row semaphore — fresh instance so it lives only as long as this row
    # processes. Provider-aware cap from `_concurrency_for_model`.
    concurrency = _concurrency_for_model(settings, eff_model)
    sem = asyncio.Semaphore(concurrency)

    # --- Inner per-evidence helper ----------------------------------------
    # Returns a tuple of (meta_dict, crop_path_or_None, match_result_or_None,
    # evidence_path_or_None). The match_result is the primary MatchResult and
    # is needed by the best-selection pass after gather; the evidence path is
    # needed to wire up `best_evidence_path` (kept around for API parity even
    # though it's currently unused downstream).
    async def _one_evidence(
        url: str, ev_idx: int,
    ) -> tuple[dict[str, Any], str | None, MatchResult | None, Path | None, float]:
        local_cost = 0.0
        async with sem:
            try:
                ev_path = await loop.run_in_executor(None, fetch, url)
            except Exception as e:
                return {"error": f"download: {e}"}, None, None, None, 0.0

            fallback_used = False
            fallback_reason: str | None = None
            # Tracks which provider-fallback class fired (small_image | ark_overdue).
            # Surfaced on the meta so the row-detail modal can render a
            # provider-specific pill instead of the generic "回落" one.
            fallback_kind: str | None = None
            vr: VerifyResult | None = None
            try:
                if use_verify:
                    # Thread `eff_model` through so per-job/per-row overrides
                    # apply to BOTH Pass-1 and the verify call.
                    try:
                        vr = await loop.run_in_executor(
                            None,
                            lambda lp=logo_path, ep=ev_path, m=eff_model: match_with_verify(
                                lp, ep, settings=settings, model=m
                            ),
                        )
                    except Exception as inner_exc:
                        # Ark "AccountOverdueError" / HTTP 402 / 账户已欠费 →
                        # retry with gpt-5.5. We retry the ENTIRE verify path
                        # against gpt-5.5 (both primary and verify call) so the
                        # row doesn't get stuck halfway through a billing
                        # outage. One warning per job.
                        if (
                            _is_ark_overdue(inner_exc)
                            and eff_model != _ARK_OVERDUE_FALLBACK_MODEL
                        ):
                            fallback_used = True
                            fallback_kind = "ark_overdue"
                            fallback_reason = "ark_overdue"
                            if job.id not in _ark_overdue_warned:
                                _ark_overdue_warned.add(job.id)
                                _log.warning(
                                    "Ark account overdue — falling back to %s "
                                    "for remaining LLM calls in job %s",
                                    _ARK_OVERDUE_FALLBACK_MODEL,
                                    job.id,
                                )
                                await publish(
                                    job.id,
                                    {
                                        "type": "warning",
                                        "message": (
                                            f"Ark 账户欠费 (HTTP 402)，本任务剩余调用回落到 "
                                            f"{_ARK_OVERDUE_FALLBACK_MODEL}"
                                        ),
                                    },
                                )
                            vr = await loop.run_in_executor(
                                None,
                                lambda lp=logo_path, ep=ev_path: match_with_verify(
                                    lp, ep, settings=settings,
                                    model=_ARK_OVERDUE_FALLBACK_MODEL,
                                ),
                            )
                        else:
                            raise
                    result: MatchResult = vr.primary
                else:
                    try:
                        result = await loop.run_in_executor(
                            None,
                            lambda lp=logo_path, ep=ev_path, m=eff_model: match_logo_in_photo(
                                lp, ep, settings=settings, model=m
                            ),
                        )
                    except Exception as inner_exc:
                        # Small-image fallback: Qwen3-VL rejects tiles < 28 px.
                        if (
                            _is_small_image_rejection(inner_exc)
                            and eff_model != _SMALL_IMAGE_FALLBACK_MODEL
                        ):
                            fallback_used = True
                            fallback_kind = "small_image"
                            fallback_reason = (
                                f"{eff_model} rejected <28 px image; "
                                f"retrying with {_SMALL_IMAGE_FALLBACK_MODEL}"
                            )
                            result = await loop.run_in_executor(
                                None,
                                lambda lp=logo_path, ep=ev_path: match_logo_in_photo(
                                    lp, ep, settings=settings, model=_SMALL_IMAGE_FALLBACK_MODEL
                                ),
                            )
                        # Ark "AccountOverdueError" / HTTP 402 / 账户已欠费 →
                        # retry this evidence against gpt-5.5. One log + one
                        # SSE warning per job; subsequent overdue hits on the
                        # same job retry silently to avoid event-log spam.
                        elif (
                            _is_ark_overdue(inner_exc)
                            and eff_model != _ARK_OVERDUE_FALLBACK_MODEL
                        ):
                            fallback_used = True
                            fallback_kind = "ark_overdue"
                            fallback_reason = "ark_overdue"
                            if job.id not in _ark_overdue_warned:
                                _ark_overdue_warned.add(job.id)
                                _log.warning(
                                    "Ark account overdue — falling back to %s "
                                    "for remaining LLM calls in job %s",
                                    _ARK_OVERDUE_FALLBACK_MODEL,
                                    job.id,
                                )
                                await publish(
                                    job.id,
                                    {
                                        "type": "warning",
                                        "message": (
                                            f"Ark 账户欠费 (HTTP 402)，本任务剩余调用回落到 "
                                            f"{_ARK_OVERDUE_FALLBACK_MODEL}"
                                        ),
                                    },
                                )
                            result = await loop.run_in_executor(
                                None,
                                lambda lp=logo_path, ep=ev_path: match_logo_in_photo(
                                    lp, ep, settings=settings,
                                    model=_ARK_OVERDUE_FALLBACK_MODEL,
                                ),
                            )
                        else:
                            raise
            except Exception as e:
                return {"error": f"llm: {e}"}, None, None, ev_path, 0.0

            meta: dict[str, Any] = {
                "found": result.found,
                "bbox": list(result.bbox) if result.bbox else None,
                "confidence": result.confidence,
                "reason": result.reason,
                "clarity": result.clarity,
                "completeness": result.completeness,
                "isolation": result.isolation,
                "usage": result.usage,
                "prompt_version": result.prompt_version,
                "model": result.model,
            }
            if result.raw_bbox is not None:
                meta["raw_bbox"] = list(result.raw_bbox)
            if result.bbox_coord_mode:
                meta["bbox_coord_mode"] = result.bbox_coord_mode
            if result.source_size is not None:
                meta["source_size"] = list(result.source_size)
            if result.sent_size is not None:
                meta["sent_size"] = list(result.sent_size)
            if fallback_used:
                # Both fallback branches currently retry against the same model
                # id (gpt-5.5), but record the model used by what actually
                # answered (`result.model`) so a future fallback target rename
                # doesn't silently misattribute cost.
                meta["fallback_model"] = result.model or _ARK_OVERDUE_FALLBACK_MODEL
                meta["fallback_reason"] = fallback_reason
                if fallback_kind:
                    meta["fallback_kind"] = fallback_kind
            # Bill the call against the model that actually answered.
            billed_model = (
                result.model
                if fallback_used and result.model
                else eff_model
            )
            local_cost += cost_estimate(result.usage, model=billed_model)

            if use_verify and vr is not None:
                meta["verified"] = bool(vr.verified)
                meta["fit"] = vr.fit_label
                meta["verify_reason"] = vr.verify_reason
                meta["verify_confidence"] = vr.verify_confidence
                meta["verify_iters"] = vr.iters
                meta["verify_final_bbox"] = (
                    list(vr.final_bbox) if vr.final_bbox else None
                )
                # Iter 5 — Pass-3 retry-with-feedback bookkeeping. Only surfaced
                # when non-default so older rows don't grow noise columns.
                if vr.retried:
                    meta["retried"] = True
                    if vr.retry_reason:
                        meta["retry_reason"] = vr.retry_reason
                    if vr.retry_bbox is not None:
                        meta["retry_bbox"] = list(vr.retry_bbox)
                # Iter 9 — refine pass bookkeeping. Only emitted when refine
                # actually fired AND its bbox SURVIVED the re-verify, so older
                # rows don't grow noise keys.
                if vr.refined:
                    meta["refined"] = True
                    if vr.refine_bbox is not None:
                        meta["refine_bbox"] = list(vr.refine_bbox)
                    if vr.refine_origin is not None:
                        meta["refine_origin"] = list(vr.refine_origin)
                if vr.verify_usage:
                    # When the ark-overdue fallback fired, the entire verify
                    # call ran against gpt-5.5 too — bill against that model
                    # rather than the original review_model so the cost
                    # estimate matches the actual provider invoice.
                    if fallback_used and fallback_kind == "ark_overdue":
                        verify_billed_model = _ARK_OVERDUE_FALLBACK_MODEL
                    else:
                        verify_billed_model = job_model or settings.review_model
                    local_cost += cost_estimate(
                        vr.verify_usage, model=verify_billed_model
                    )
                    meta["verify_usage"] = vr.verify_usage
                if vr.soft_verified:
                    meta["soft_verified"] = True
                    if vr.soft_verify_reason:
                        meta["soft_verify_reason"] = vr.soft_verify_reason

            # ---- Iter 7 Pass-1 variance pre-gate -------------------------
            # Drop Pass-1 predictions whose bbox lands on a blank/background
            # region (truth-set analysis: this filters ~50% of wrong-region
            # hallucinations without losing any real hit — see
            # `_PASS1_BLANK_STD_THRESHOLD` comment above). The gate fires
            # whether verify_loop is ON or OFF; when verify already ran inside
            # match_with_verify, we still invalidate its verdict because the
            # bbox itself is unusable. The tile-scan fallback below uses
            # `not result.found` as its trigger, so a gate-rejected Pass-1
            # still gets a tile-scan recovery attempt — the logo might be in
            # a real region elsewhere in the photo.
            if result.found and result.bbox:
                _stats = await loop.run_in_executor(
                    None,
                    _bbox_blank_stats,
                    ev_path,
                    result.bbox,
                )
                if _stats is not None:
                    _std, _white = _stats
                    if (
                        _std < _PASS1_BLANK_STD_THRESHOLD
                        or _white > _PASS1_BLANK_WHITE_THRESHOLD
                    ):
                        meta["pass1_blank_reject"] = True
                        meta["pass1_blank_std"] = _std
                        meta["pass1_blank_white"] = _white
                        meta["pass1_original_bbox"] = list(result.bbox)
                        meta["pass1_original_reason"] = result.reason
                        meta["found"] = False
                        # Iter 9 bug fix: when the variance gate fires the
                        # bbox itself is unusable — null it out so downstream
                        # consumers (frontend overlay, eval) don't render the
                        # blank-region rectangle. The provenance lives on
                        # `pass1_original_bbox` only.
                        meta["bbox"] = None
                        # Invalidate verify's verdict — its `verified=True`
                        # output on a blank bbox is the exact failure mode
                        # this gate exists to suppress. Setting vr=None lets
                        # the acceptance gate below take the no-verify branch
                        # which checks `result.found` (now False).
                        vr = None
                        result = MatchResult(
                            found=False,
                            bbox=None,
                            confidence=0.0,
                            reason=(
                                f"pass-1 bbox is blank region "
                                f"(std={_std:.1f}, white={_white:.2f})"
                            ),
                            raw_response=result.raw_response,
                            prompt_version=result.prompt_version,
                            model=result.model,
                            usage=result.usage,
                            clarity=result.clarity,
                            completeness=result.completeness,
                            isolation=result.isolation,
                            json_retry_count=result.json_retry_count,
                            raw_bbox=result.raw_bbox,
                            bbox_coord_mode=result.bbox_coord_mode,
                            source_size=result.source_size,
                            sent_size=result.sent_size,
                        )

            # ---- Iter 11 Edge-based shape match refine ------------------
            # Pure-CV refine path that runs BEFORE the iter-10 SIFT chain.
            # Canny → findContours → matchShapes(Hu moments) inside the
            # 3x-expanded VLM region. Designed for the line-art ↔ printed-
            # logo case that SIFT cannot bridge (zero shared keypoints).
            # Default ON; toggle via LINEBASE_EDGE_REFINE. When the iter-10
            # SIFT refine is ALSO enabled it runs after this block and would
            # operate on the edge-refined bbox — in practice SIFT is off so
            # this composition is theoretical, but the chain order is the
            # right one (cheaper shape match first, geometric refine after).
            edge_refine_on = _edge_refine_enabled()
            edge_recall_on = _edge_recall_enabled()

            if (
                edge_refine_on
                and not _is_design_prompt_result(result)
                and result.found
                and result.bbox is not None
            ):
                try:
                    edge_res = await loop.run_in_executor(
                        None,
                        lambda lp=logo_path, ep=ev_path, rb=result.bbox: edge_refine_bbox(
                            lp, ep, region_bbox=rb,
                        ),
                    )
                except Exception as e:
                    _log.warning(
                        "edge_refine_bbox raised %r — keeping VLM bbox", e,
                    )
                    edge_res = None

                if edge_res is not None:
                    meta["edge_refined"] = True
                    meta["edge_shape_distance"] = edge_res.shape_distance
                    meta["edge_candidates_checked"] = edge_res.candidates_checked
                    meta["edge_original_bbox"] = list(result.bbox)
                    result = MatchResult(
                        found=result.found,
                        bbox=edge_res.bbox,
                        confidence=result.confidence,
                        reason=result.reason,
                        raw_response=result.raw_response,
                        prompt_version=result.prompt_version,
                        model=result.model,
                        usage=result.usage,
                        clarity=result.clarity,
                        completeness=result.completeness,
                        isolation=result.isolation,
                        json_retry_count=result.json_retry_count,
                        raw_bbox=result.raw_bbox,
                        bbox_coord_mode=result.bbox_coord_mode,
                        source_size=result.source_size,
                        sent_size=result.sent_size,
                    )
                    meta["bbox"] = list(edge_res.bbox)
                    # When verify already passed, semantic verdict still
                    # holds (logo IS in that region); we only tighten the
                    # pixels. Update final_bbox so the downstream crop uses
                    # the refined coords.
                    if vr is not None and vr.verified:
                        vr = VerifyResult(
                            primary=result,
                            verified=vr.verified,
                            final_bbox=edge_res.bbox,
                            fit_label=vr.fit_label,
                            verify_confidence=vr.verify_confidence,
                            verify_reason=vr.verify_reason,
                            iters=vr.iters,
                            verify_usage=vr.verify_usage,
                            retried=vr.retried,
                            retry_reason=vr.retry_reason,
                            retry_bbox=vr.retry_bbox,
                        )

            # Iter 11 — Edge recall: whole-photo shape match when Pass-1
            # returned found=False (or the variance gate nulled it). Tighter
            # shape-distance threshold than refine because the search space
            # is much larger and the prior on a real logo is weaker.
            if edge_recall_on and not result.found:
                try:
                    edge_recall_res = await loop.run_in_executor(
                        None,
                        lambda lp=logo_path, ep=ev_path: edge_refine_bbox(
                            lp, ep, region_bbox=None, whole_photo_recall=True,
                        ),
                    )
                except Exception as e:
                    _log.warning(
                        "edge_refine_bbox(recall) raised %r — skipping", e,
                    )
                    edge_recall_res = None

                if (
                    edge_recall_res is not None
                    and edge_recall_res.shape_distance <= _EDGE_RECALL_MAX_DISTANCE
                ):
                    meta["edge_recall_hit"] = True
                    meta["edge_shape_distance"] = edge_recall_res.shape_distance
                    meta["edge_candidates_checked"] = edge_recall_res.candidates_checked
                    # Synthesize a found=True MatchResult mirroring the SIFT
                    # recall path's confidence (0.6) — high enough to clear
                    # the usual 0.5 acceptance threshold without dominating
                    # real VLM hits in best-crop ranking.
                    result = MatchResult(
                        found=True,
                        bbox=edge_recall_res.bbox,
                        confidence=0.6,
                        reason=(
                            f"edge_recall: distance={edge_recall_res.shape_distance:.3f} "
                            f"over {edge_recall_res.candidates_checked} candidates"
                        ),
                        raw_response=result.raw_response,
                        prompt_version=result.prompt_version,
                        model=result.model,
                        usage=result.usage,
                        clarity=result.clarity,
                        completeness=result.completeness,
                        isolation=result.isolation,
                        json_retry_count=result.json_retry_count,
                        raw_bbox=result.raw_bbox,
                        bbox_coord_mode=result.bbox_coord_mode,
                        source_size=result.source_size,
                        sent_size=result.sent_size,
                    )
                    meta["found"] = True
                    meta["bbox"] = list(edge_recall_res.bbox)
                    meta["confidence"] = 0.6
                    meta["reason"] = result.reason
                    if use_verify:
                        try:
                            recall_vr = await loop.run_in_executor(
                                None,
                                lambda lp=logo_path, ep=ev_path, mr=result: _verify_recalled_bbox(
                                    lp,
                                    ep,
                                    mr,
                                    settings=settings,
                                    model=eff_model,
                                ),
                            )
                        except Exception as e:
                            _log.warning(
                                "verify recalled edge bbox raised %r — rejecting recall",
                                e,
                            )
                            recall_vr = VerifyResult(
                                primary=result,
                                verified=False,
                                final_bbox=None,
                                fit_label="wrong",
                                verify_confidence=0.0,
                                verify_reason=f"edge_recall verify failed: {e}",
                                iters=1,
                            )
                        vr = recall_vr
                        meta["verified"] = bool(vr.verified)
                        meta["fit"] = vr.fit_label
                        meta["verify_reason"] = vr.verify_reason
                        meta["verify_confidence"] = vr.verify_confidence
                        meta["verify_iters"] = vr.iters
                        meta["verify_final_bbox"] = (
                            list(vr.final_bbox) if vr.final_bbox else None
                        )
                        if vr.verify_usage:
                            local_cost += cost_estimate(
                                vr.verify_usage,
                                model=eff_model,
                            )
                            meta["verify_usage"] = vr.verify_usage

            # ---- Iter 10 SIFT + RANSAC homography refine -----------------
            # Two paths, both pure CV (zero LLM cost):
            #
            #   (a) refine — when Pass-1 (or post-tile-scan, see below) gave
            #       a bbox B0, SIFT/FLANN/RANSAC inside the 3x-expanded B0
            #       to find pixel-tight corners. On success we replace
            #       `result.bbox` (and `vr.final_bbox` when verify ran) and
            #       record the original as provenance. On failure we keep B0
            #       unchanged — SIFT failing inside a verified VLM region is
            #       common (engraved / low-texture logos) and not a reason to
            #       reject the row.
            #
            #   (b) recall — when Pass-1 returned found=False, SIFT the WHOLE
            #       photo (no region hint). A high-inlier match (>= 10) means
            #       the VLM missed something the geometry-based matcher can
            #       still recover; treat it as a fresh bbox and proceed
            #       through verify / acceptance like any Pass-1 hit.
            #
            # Why here (between variance gate and tile-scan):
            #   - Variance gate may have zero'd out a hallucinated bbox; we
            #     must respect that and treat the row as "not found" for
            #     refine purposes. The recall path THEN gets the chance to
            #     find the real logo if it exists.
            #   - Tile-scan is opt-in + LLM-cost-heavy; SIFT recall is free
            #     and frequently catches the same cases without 9 extra
            #     calls, so running it first is the right cost order.
            sift_refine_on = _sift_refine_enabled()
            sift_recall_on = _sift_recall_enabled()

            # Refine path: only when Pass-1 produced a bbox AND the variance
            # gate didn't null it. `result.bbox` is the authoritative source
            # post-variance-gate (the gate nulls it to None on rejection).
            if sift_refine_on and result.found and result.bbox is not None:
                try:
                    sift_res = await loop.run_in_executor(
                        None,
                        lambda lp=logo_path, ep=ev_path, rb=result.bbox: sift_refine_bbox(
                            lp, ep, region_bbox=rb,
                        ),
                    )
                except Exception as e:
                    # SIFT itself shouldn't throw on valid inputs, but if cv2
                    # ever surprises us we don't want the whole evidence to
                    # tank — fall back to the unrefined bbox + a log line.
                    _log.warning("sift_refine_bbox raised %r — keeping VLM bbox", e)
                    sift_res = None

                if sift_res is not None:
                    # Record provenance BEFORE overwriting so the modal can
                    # render both the original VLM bbox and the refined one.
                    meta["sift_refined"] = True
                    meta["sift_inliers"] = sift_res.inliers
                    meta["sift_original_bbox"] = list(result.bbox)
                    # Mutate `result` via a fresh MatchResult — dataclass
                    # fields are not frozen, but a fresh instance keeps the
                    # rest of `_one_evidence` reading from a consistent
                    # snapshot (and matches how the variance gate also
                    # rewrites the result).
                    result = MatchResult(
                        found=result.found,
                        bbox=sift_res.bbox,
                        confidence=result.confidence,
                        reason=result.reason,
                        raw_response=result.raw_response,
                        prompt_version=result.prompt_version,
                        model=result.model,
                        usage=result.usage,
                        clarity=result.clarity,
                        completeness=result.completeness,
                        isolation=result.isolation,
                        json_retry_count=result.json_retry_count,
                        raw_bbox=result.raw_bbox,
                        bbox_coord_mode=result.bbox_coord_mode,
                        source_size=result.source_size,
                        sent_size=result.sent_size,
                    )
                    meta["bbox"] = list(sift_res.bbox)
                    # When verify already ran, its semantic verdict still
                    # applies (it confirmed the logo IS in that region); we
                    # just tighten the pixels. Update final_bbox so the
                    # downstream crop uses the refined coordinates.
                    if vr is not None and vr.verified:
                        vr = VerifyResult(
                            primary=result,
                            verified=vr.verified,
                            final_bbox=sift_res.bbox,
                            fit_label=vr.fit_label,
                            verify_confidence=vr.verify_confidence,
                            verify_reason=vr.verify_reason,
                            iters=vr.iters,
                            verify_usage=vr.verify_usage,
                            retried=vr.retried,
                            retry_reason=vr.retry_reason,
                            retry_bbox=vr.retry_bbox,
                        )

            # Recall path: SIFT the whole photo when Pass-1 came back empty
            # (or the variance gate nulled it). Stricter inlier floor than
            # refine because the search space is much larger — false positives
            # are cheaper to allow on a verified region than on a blind sweep.
            if sift_recall_on and not result.found:
                try:
                    recall_res = await loop.run_in_executor(
                        None,
                        lambda lp=logo_path, ep=ev_path: sift_refine_bbox(
                            lp, ep, region_bbox=None,
                        ),
                    )
                except Exception as e:
                    _log.warning("sift_refine_bbox(recall) raised %r — skipping", e)
                    recall_res = None

                if (
                    recall_res is not None
                    and recall_res.inliers >= _SIFT_RECALL_MIN_INLIERS
                ):
                    meta["sift_recall_hit"] = True
                    meta["sift_inliers"] = recall_res.inliers
                    # Synthesize a found=True MatchResult so the rest of the
                    # evidence path treats it like a successful Pass-1.
                    # Confidence is set to a neutral 0.6 — high enough to
                    # clear the typical 0.5 acceptance threshold but not so
                    # high that it dominates real VLM hits during best-crop
                    # ranking. Reason carries the SIFT provenance.
                    result = MatchResult(
                        found=True,
                        bbox=recall_res.bbox,
                        confidence=0.6,
                        reason=(
                            f"sift_recall: {recall_res.inliers} inliers / "
                            f"{recall_res.total_matches} good matches"
                        ),
                        raw_response=result.raw_response,
                        prompt_version=result.prompt_version,
                        model=result.model,
                        usage=result.usage,
                        clarity=result.clarity,
                        completeness=result.completeness,
                        isolation=result.isolation,
                        json_retry_count=result.json_retry_count,
                        raw_bbox=result.raw_bbox,
                        bbox_coord_mode=result.bbox_coord_mode,
                        source_size=result.source_size,
                        sent_size=result.sent_size,
                    )
                    meta["found"] = True
                    meta["bbox"] = list(recall_res.bbox)
                    meta["confidence"] = 0.6
                    meta["reason"] = result.reason

            # ---- Iter 6.3 tile-scan fallback -----------------------------
            # Trigger conditions (job.tile_scan must be ON):
            #   - Pass-1 returned found=False                                → tile-scan
            #   - verify enabled, found=true but verified=False AND retried  → tile-scan
            #   - verify enabled, found=true, not retried, verified=False    → tile-scan
            #     (iter-5 retry already covered the blank-reject path; any
            #      other verify reject on a busy photo is the exact case
            #      tile-scan is designed to recover)
            # Skip when Pass-1 errored (we never reach here in that case),
            # or when the photo is small enough not to need it (the helper
            # itself early-returns based on _TILE_SCAN_MIN_LONGEST_SIDE).
            if tile_scan_enabled:
                # Trigger: Pass-1 found nothing, OR (verify on AND Pass-1 found
                # but verify rejected). Skip when verify already passed — that
                # result is already good. Skip when Pass-1 errored (we never
                # reach here in that case; the early return above bails first).
                wants_tile_scan = (
                    not result.found
                    or (use_verify and vr is not None and not vr.verified)
                )
                if wants_tile_scan:
                    # Iter 6.4 — verify-in-the-loop is now done INSIDE _tile_scan
                    # over the top-3 candidates, so a degenerate high-conf tile
                    # can no longer beat a real lower-conf match. When verify
                    # is enabled, _tile_scan returns ONLY a verify-confirmed
                    # result (or None when all top-3 fail).
                    ts_verify_model = job_model or settings.review_model
                    ts_result, ts_prov, ts_cost = await loop.run_in_executor(
                        None,
                        lambda lp=logo_path, ep=ev_path, m=eff_model,
                               vm=ts_verify_model, ve=use_verify: _tile_scan(
                            lp, ep, settings=settings, model=m,
                            verify_enabled=ve, verify_model=vm,
                        ),
                    )
                    local_cost += ts_cost

                    # Always surface tile-scan provenance — even on a miss the
                    # reviewer should see "we tried tile-scan, N candidates
                    # survived the degenerate filter".
                    if ts_prov:
                        meta.update(ts_prov)

                    if ts_result is not None and ts_result.bbox is not None:
                        # Overwrite the now-superseded primary fields with the
                        # tile-scan winner so downstream consumers (best-crop
                        # ranking, frontend bbox overlay) see the right bbox.
                        meta["found"] = True
                        meta["bbox"] = list(ts_result.bbox)
                        meta["confidence"] = ts_result.confidence
                        meta["reason"] = ts_result.reason
                        # Replace `result` so the acceptance gate below uses
                        # the tile-scan bbox + confidence.
                        result = ts_result
                        if use_verify:
                            # _tile_scan only returns a result when verify
                            # passed, so we can mark verified=True here.
                            meta["verified"] = True
                            # Patch `vr` to keep the acceptance-gate branch
                            # below in sync with the post-tile-scan state.
                            if vr is not None:
                                vr = VerifyResult(
                                    primary=ts_result,
                                    verified=True,
                                    final_bbox=ts_result.bbox,
                                    fit_label=vr.fit_label,
                                    verify_confidence=vr.verify_confidence,
                                    verify_reason=vr.verify_reason,
                                    iters=vr.iters + 1,
                                    verify_usage=vr.verify_usage,
                                    retried=vr.retried,
                                    retry_reason=vr.retry_reason,
                                    retry_bbox=vr.retry_bbox,
                                )
                    # When ts_result is None, leave `result` / `vr` untouched.
                    # If Pass-1 was found=False, the row will fall through to
                    # needs_review naturally; if Pass-1 was found-but-rejected,
                    # the verify-reject state already drives needs_review.

            # Acceptance gate. When tile-scan fired, `result` and (if verify
            # enabled) `vr` have already been overwritten with the tile-scan
            # outcome, so the same gate logic applies uniformly.
            if use_verify and vr is not None:
                accept = (
                    bool(vr.verified)
                    and vr.final_bbox is not None
                    and result.confidence >= job.threshold
                )
                crop_bbox = vr.final_bbox if accept else None
            else:
                accept = bool(
                    result.found and result.bbox and result.confidence >= job.threshold
                )
                crop_bbox = result.bbox if accept else None

            crop_path: str | None = None
            if accept and crop_bbox is not None:
                crop_out = run_dir / "images" / f"{row.appno or row.id}_{ev_idx + 1}.png"
                await loop.run_in_executor(
                    None, crop_to_bbox, ev_path, crop_bbox, crop_out, 0.05
                )
                sanity = await loop.run_in_executor(
                    None, _crop_sanity_check, crop_out, ev_path
                )
                if sanity is not None:
                    meta["sanity_rejected"] = sanity
                    crop_path = None
                else:
                    crop_path = str(crop_out)
            return meta, crop_path, result, ev_path, local_cost

    # --- Fan-out ---------------------------------------------------------
    # `return_exceptions=True` so a coroutine raising at the asyncio layer
    # (rare — every inner-helper exception path already returns a tuple) does
    # not cancel the rest of the row. We map those leaked exceptions into
    # synthetic `{"error": ...}` metas to preserve the per-evidence contract.
    results = await asyncio.gather(
        *[_one_evidence(u, i) for i, u in enumerate(evidences)],
        return_exceptions=True,
    )

    cost_delta = 0.0
    crops: dict[str, str | None] = {}
    metas: dict[str, dict[str, Any]] = {}
    # Preserve the per-evidence MatchResult alongside the meta dict so the
    # post-verify, post-sanity re-ranking pass below can recover the original
    # confidence + bbox without having to JSON-roundtrip through `meta`.
    results_by_url: dict[str, MatchResult] = {}

    for url, outcome in zip(evidences, results, strict=False):
        if isinstance(outcome, BaseException):
            metas[url] = {"error": f"async: {outcome!r}"}
            crops[url] = None
            continue
        meta, crop_path, result, ev_path, local_cost = outcome
        metas[url] = meta
        crops[url] = crop_path
        cost_delta += local_cost
        if result is not None:
            results_by_url[url] = result

    # --- Best-crop selection (post-verify, post-sanity re-rank) -----------
    # Old behavior: pick max(confidence) over evidences whose CROP survived
    # sanity+verify. That collapsed in the row-85094272 (Polo) case: a sibling
    # evidence had its crop sanity-rejected mid-loop but was still ranked
    # ahead of a clean-but-lower-confidence sibling because we accumulated the
    # "best" inside the gather-walk loop and never re-evaluated.
    #
    # New behavior: AFTER the gather is complete, rebuild the candidate list
    # from rows that genuinely survived every downstream filter (verify, sanity,
    # crop-file-exists). Sort that list by confidence desc and pick the top.
    # If everything failed, status=needs_review with best_crop=None.
    use_verify_now = use_verify  # local alias for readability in the filter
    viable: list[tuple[str, dict[str, Any]]] = []
    for url, meta in metas.items():
        if not isinstance(meta, dict):
            continue
        if not meta.get("found"):
            continue
        if meta.get("sanity_rejected"):
            continue
        if use_verify_now and meta.get("verified") is not True:
            continue
        # Crop file must actually exist — the inner helper sets crops[url]=None
        # when verify rejected the bbox or sanity flagged a blank/tiny crop.
        if not crops.get(url):
            continue
        viable.append((url, meta))

    if not viable:
        status = "needs_review"
        best_crop = None
    else:
        # Sort by confidence desc; ties broken by URL order which is stable
        # (Python's sort is stable, and we iterate metas in insertion order
        # which matches evidence-URL order via the gather-results zip).
        viable.sort(key=lambda kv: float(kv[1].get("confidence") or 0.0), reverse=True)
        best_url = viable[0][0]
        status = "ok"
        best_crop = crops.get(best_url)

    store.update_job_row(
        row.id,
        status=status,
        best_crop_path=best_crop,
        all_crops_json=json.dumps(crops),
        match_meta_json=json.dumps(metas, ensure_ascii=False),
    )
    return cost_delta, status


async def _run_job(job_id: str, source_path: Path, settings: Settings) -> None:
    job = store.get_job(job_id)
    assert job is not None
    run_dir = DATA_DIR / "runs" / job_id
    (run_dir / "images").mkdir(parents=True, exist_ok=True)

    store.update_job(job_id, status="running", started_at=time.time())
    await publish(job_id, {"type": "progress", "job": _job_to_dict(store.get_job(job_id))})  # type: ignore[arg-type]

    rows = store.list_job_rows(job_id)
    cost_total = 0.0
    # Resume / single-row rerun: only process rows that are still pending or
    # were mid-flight when we last stopped. Rows that already reached a
    # terminal state (ok / bad / needs_review / failed) are skipped — the
    # /rerun endpoints reset row.status to "pending" first, which is what
    # makes them re-eligible here.
    _RERUNNABLE = {"pending", "running"}
    for row in rows:
        if row.status not in _RERUNNABLE:
            continue
        store.update_job_row(row.id, status="running")
        await publish(job_id, {"type": "progress", "row": _row_to_dict(store.get_job_row(row.id))})  # type: ignore[arg-type]
        try:
            cost_delta, terminal = await _process_row(job, row, settings, run_dir)
        except Exception as e:
            store.update_job_row(row.id, status="failed", notes=f"unhandled: {e}")
            terminal = "failed"
            cost_delta = 0.0

        cost_total += cost_delta
        done = sum(
            1
            for r in store.list_job_rows(job_id)
            if r.status in ("ok", "bad", "needs_review", "failed")
        )
        store.update_job(job_id, done_rows=done, cost_usd=cost_total)

        ev_type = "row_failed" if terminal == "failed" else "row_done"
        await publish(job_id, {"type": ev_type, "row": _row_to_dict(store.get_job_row(row.id))})  # type: ignore[arg-type]

    store.update_job(job_id, status="finished", finished_at=time.time())
    await publish(job_id, {"type": "finished", "job": _job_to_dict(store.get_job(job_id))})  # type: ignore[arg-type]


def start_job(job_id: str, source_path: Path, settings: Settings) -> None:
    if job_id in _active_tasks and not _active_tasks[job_id].done():
        return
    # Must be called from inside the FastAPI event loop. Prefer
    # `get_running_loop()` which is the only correct API on 3.10+; falling back
    # to `get_event_loop()` only as a defensive measure on older paths.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    task = loop.create_task(_run_job(job_id, source_path, settings))
    _active_tasks[job_id] = task


async def stream_events(job_id: str) -> AsyncIterator[dict[str, Any]]:
    q = await subscribe(job_id)
    try:
        # send a one-time snapshot first so clients have current state
        job = store.get_job(job_id)
        if job:
            yield {"type": "progress", "job": _job_to_dict(job)}
        while True:
            ev = await q.get()
            yield ev
            if ev.get("type") == "finished":
                return
    finally:
        await unsubscribe(job_id, q)
