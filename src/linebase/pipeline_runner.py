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
            match_meta[url] = {
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
    if "error code: 402" in msg:
        return True
    return False


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
    ) -> tuple[dict[str, Any], str | None, "MatchResult | None", "Path | None", float]:
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

            # Acceptance gate
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

    for url, outcome in zip(evidences, results):
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
