"""Background job orchestration + SSE event bus.

A job runs as one asyncio task that:
  1. resolves the row list from the uploaded xlsx,
  2. for each row: download evidences, call LLM per evidence, crop best, persist to DB,
  3. publishes events to a per-job asyncio.Queue that the SSE endpoint consumes.

Cost is approximated (model pricing is unknown for the 1m1ng relay; we estimate
gpt-4o-class rates as a sane upper bound).
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator

import os

from linebase import store
from linebase.config import Settings
from linebase.crop import crop_to_bbox
from linebase.fetch import fetch
from linebase.io_excel import iter_rows
from linebase.llm import MatchResult, match_logo_in_photo
from linebase.verify_loop import VerifyResult, match_with_verify


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
    return d


# --- Pipeline ---------------------------------------------------------------

_active_tasks: dict[str, asyncio.Task[None]] = {}


# Per-provider adjustment factor applied to the gpt-5.x-rate base estimate.
# The base formula uses OpenAI gpt-5 rates as a single scalar; Ark (Doubao) and
# SiliconFlow (Qwen / GLM / Kimi) bill ~30-100× lower in practice. Multiplying
# by these factors brings UI / SQLite `cost_usd` totals within ~2× of real
# spend instead of the previous 30-100× over-count. See
# research/lite-benchmark-4way.md and research/user-model-picker.md.
_PROVIDER_COST_FACTOR: dict[str, float] = {
    "openai": 1.0,
    "ark": 0.02,
    "siliconflow": 0.02,
}


def cost_estimate(usage: dict[str, int] | None, model: str | None = None) -> float:
    """Approximate USD spend for one LLM call.

    Base = gpt-5-rate scalar; multiplied by `_PROVIDER_COST_FACTOR[provider]`
    when the model resolves to a non-OpenAI provider. `model=None` keeps the
    old behaviour for backwards compatibility with callers that don't yet pass
    one (in practice every call site does, after 2026-05-24).
    """
    if not usage:
        return 0.0
    base = (
        usage.get("prompt_tokens", 0) * 2.5e-6
        + usage.get("completion_tokens", 0) * 10e-6
    )
    if not model:
        return base
    try:
        # Lazy: Settings is heavy to construct, so cache the provider lookup
        # per process. Module-scope `_settings_cache` is fine — config doesn't
        # change at runtime.
        global _settings_cache
        if _settings_cache is None:
            _settings_cache = Settings.from_env()
        provider_name = _settings_cache.resolve_provider(model).name
    except Exception:
        return base
    factor = _PROVIDER_COST_FACTOR.get(provider_name, 1.0)
    return base * factor


_settings_cache: Settings | None = None


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
    crops: dict[str, str | None] = {}
    metas: dict[str, dict[str, Any]] = {}

    try:
        logo_path = await loop.run_in_executor(None, fetch, row.logo_url)
    except Exception as e:
        store.update_job_row(row.id, status="failed", notes=f"logo download failed: {e}")
        return 0.0, "failed"

    use_verify = _verify_enabled()
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
            crops[url] = str(crop_out)
            if best is None or result.confidence > best.confidence:
                best = result
                best_evidence_path = ev_path
        else:
            crops[url] = None

    if best is None or best_evidence_path is None:
        status = "needs_review"
        best_crop = None
    else:
        status = "ok"
        best_crop = crops.get(evidences[[i for i, u in enumerate(evidences) if metas.get(u, {}).get("confidence") == best.confidence][0]]) if evidences else None

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
    for row in rows:
        if row.status == "ok":
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
