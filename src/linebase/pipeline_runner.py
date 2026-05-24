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
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator

import os

from PIL import Image, ImageStat

from linebase import store
from linebase.config import Settings
from linebase.crop import crop_to_bbox
from linebase.fetch import fetch
from linebase.io_excel import iter_rows
from linebase.llm import MatchResult, match_logo_in_photo
from linebase.verify_loop import VerifyResult, match_with_verify

_log = logging.getLogger(__name__)


def _verify_enabled() -> bool:
    """Opt-in via env: LINEBASE_VERIFY=1 (also accepts true/yes/on, case-insensitive)."""
    return os.environ.get("LINEBASE_VERIFY", "").strip().lower() in {"1", "true", "yes", "on"}

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
    # Rule: pick the entry with the highest confidence among found=True ones
    # (same as the pipeline's best selection).
    best_url: str | None = None
    best: dict[str, Any] | None = None
    try:
        meta = json.loads(meta_raw) if meta_raw else {}
    except Exception:
        meta = {}
    if isinstance(meta, dict):
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


async def _process_row(
    job: store.Job,
    row: store.JobRow,
    settings: Settings,
    run_dir: Path,
) -> tuple[float, str]:
    """Process one job_row in a thread (LLM + IO are blocking). Returns (cost_delta, terminal_status)."""
    evidences = json.loads(row.evidence_urls_json)
    if not row.logo_url or not evidences:
        store.update_job_row(row.id, status="failed", notes="missing logo_url or evidence_urls")
        return 0.0, "failed"

    loop = asyncio.get_event_loop()
    cost_delta = 0.0
    best: MatchResult | None = None
    best_evidence_path: Path | None = None
    best_url: str | None = None
    crops: dict[str, str | None] = {}
    metas: dict[str, dict[str, Any]] = {}

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
    # Per-job model override (added 2026-05-24). When None the lower layers
    # fall back to settings.model. Verify-loop keeps using settings.review_model
    # by default; pipeline jobs that pin a vision model only override the
    # primary match call, since the verify prompt is more sensitive to model
    # behavior.
    job_model = (job.model or "").strip() or None
    eff_model = job_model or settings.model

    for url in evidences:
        try:
            ev_path = await loop.run_in_executor(None, fetch, url)
        except Exception as e:
            metas[url] = {"error": f"download: {e}"}
            crops[url] = None
            continue

        fallback_used = False
        fallback_reason: str | None = None
        try:
            if use_verify:
                vr: VerifyResult = await loop.run_in_executor(
                    None, match_with_verify, logo_path, ev_path, settings
                )
                result: MatchResult = vr.primary
            else:
                vr = None  # type: ignore[assignment]
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda lp=logo_path, ep=ev_path, m=eff_model: match_logo_in_photo(
                            lp, ep, settings=settings, model=m
                        ),
                    )
                except Exception as inner_exc:
                    # Small-image fallback: Qwen3-VL rejects tiles < 28 px. If
                    # the chosen model already IS gpt-5.5, skip the retry —
                    # it would just hit the same OpenAI provider with the same
                    # image and there's no smaller-image-tolerant fallback.
                    if (
                        _is_small_image_rejection(inner_exc)
                        and eff_model != _SMALL_IMAGE_FALLBACK_MODEL
                    ):
                        fallback_used = True
                        fallback_reason = f"{eff_model} rejected <28 px image; retrying with {_SMALL_IMAGE_FALLBACK_MODEL}"
                        result = await loop.run_in_executor(
                            None,
                            lambda lp=logo_path, ep=ev_path: match_logo_in_photo(
                                lp, ep, settings=settings, model=_SMALL_IMAGE_FALLBACK_MODEL
                            ),
                        )
                    else:
                        raise
        except Exception as e:
            metas[url] = {"error": f"llm: {e}"}
            crops[url] = None
            continue

        meta: dict[str, Any] = {
            "found": result.found, "bbox": list(result.bbox) if result.bbox else None,
            "confidence": result.confidence, "reason": result.reason,
            "clarity": result.clarity, "completeness": result.completeness,
            "isolation": result.isolation,
            "usage": result.usage, "prompt_version": result.prompt_version,
            "model": result.model,
        }
        if fallback_used:
            meta["fallback_model"] = _SMALL_IMAGE_FALLBACK_MODEL
            meta["fallback_reason"] = fallback_reason
        # Bill the call against the model that actually answered — that's the
        # fallback model when fallback fired, otherwise the configured one.
        billed_model = _SMALL_IMAGE_FALLBACK_MODEL if fallback_used else eff_model
        cost_delta += cost_estimate(result.usage, model=billed_model)

        # When verify is on, attach its outcome to the meta so the review page can see it.
        if use_verify and vr is not None:
            meta["verified"] = bool(vr.verified)
            meta["fit"] = vr.fit_label
            meta["verify_reason"] = vr.verify_reason
            meta["verify_confidence"] = vr.verify_confidence
            meta["verify_iters"] = vr.iters
            meta["verify_final_bbox"] = list(vr.final_bbox) if vr.final_bbox else None
            if vr.verify_usage:
                cost_delta += cost_estimate(vr.verify_usage, model=settings.review_model)
                meta["verify_usage"] = vr.verify_usage
        metas[url] = meta

        # Crop only when the row passes our acceptance gate.
        if use_verify and vr is not None:
            accept = bool(vr.verified) and vr.final_bbox is not None and result.confidence >= job.threshold
            crop_bbox = vr.final_bbox if accept else None
        else:
            accept = bool(result.found and result.bbox and result.confidence >= job.threshold)
            crop_bbox = result.bbox if accept else None

        if accept and crop_bbox is not None:
            crop_out = run_dir / "images" / f"{row.appno or row.id}_{evidences.index(url) + 1}.png"
            await loop.run_in_executor(None, crop_to_bbox, ev_path, crop_bbox, crop_out, 0.05)
            # Post-crop sanity check: reject hallucinated bboxes that landed
            # on a blank margin OR are absurdly tiny. Downgrades to crops[url]
            # = None so this evidence is excluded from best-selection, and
            # if EVERY evidence gets rejected the row ends up needs_review
            # (the `best is None` branch below).
            sanity = await loop.run_in_executor(
                None, _crop_sanity_check, crop_out, ev_path
            )
            if sanity is not None:
                metas[url]["sanity_rejected"] = sanity
                crops[url] = None
                # Don't delete the file — keep it around so a human can audit
                # what the LLM thought was a logo. It just doesn't count as a
                # candidate for best-selection.
            else:
                crops[url] = str(crop_out)
                if best is None or result.confidence > best.confidence:
                    best = result
                    best_evidence_path = ev_path
                    best_url = url
        else:
            crops[url] = None

    if best is None or best_evidence_path is None or best_url is None:
        status = "needs_review"
        best_crop = None
    else:
        status = "ok"
        # Direct lookup by URL — the old version walked confidence equality
        # through metas which broke when two evidences happened to share the
        # same float and could pick the wrong (or even sanity-rejected) URL.
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
        done = sum(1 for r in store.list_job_rows(job_id) if r.status in ("ok", "bad", "needs_review", "failed"))
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
