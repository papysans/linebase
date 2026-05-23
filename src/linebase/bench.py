"""Shared per-fixture eval loop, factored out of `scripts/eval_runner.py`.

This module owns the "for each fixture, run the matcher, score selection +
bbox-IoU, write crops, accumulate metrics" loop. Both `eval_runner.py` (single
model + optional verify) and `scripts/benchmark_models.py` (sweep multiple
models, no verify) call into here.

Design notes:
  * Single-pass over fixtures, no parallelism — model latency stats stay clean.
  * Verify path is preserved for parity with the existing runner. The benchmark
    leaves `verify=False` (one variable at a time).
  * Returns a `RunResult` dataclass with the augmented metrics dict + raw log,
    so the caller decides where/how to persist. The legacy runner just dumps
    `metrics.json` + `raw_log.json` + `report.html` itself.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from linebase.config import ProviderConfig, Settings
from linebase.crop import crop_to_bbox
from linebase.eval import PairScore, RunSummary, score_pair, write_html_report, write_metrics_json
from linebase.llm import MatchResult, match_logo_in_photo

# Lazy import — verify_loop drags in the verify prompt requirement, only needed
# when use_verify=True.

REPO = Path(__file__).resolve().parents[2]
FIXTURES_DEFAULT = REPO / ".trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline/fixtures"
GOLD_PATH = REPO / "eval/redbox_gold.json"
REDBOX_BBOX_PATH = REPO / "eval/redbox_bboxes.json"

# Minimum border_red_ratio to consider that a sample actually has a hand-drawn red box.
GOLD_BORDER_THRESHOLD = 0.005


# ---------------------------------------------------------------------------
# red-box detection (verbatim from eval_runner.py — single source of truth here)
# ---------------------------------------------------------------------------

def detect_redbox_bbox(image_path: Path) -> tuple[int, int, int, int] | None:
    img = np.asarray(Image.open(image_path).convert("RGB"))
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    red = (r > 200) & (g < 80) & (b < 80)
    if red.sum() < 20:
        return None
    ys, xs = np.where(red)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return None
    h, w = img.shape[:2]
    if (x2 - x1) > 0.98 * w and (y2 - y1) > 0.98 * h:
        return None
    return (x1, y1, x2, y2)


def load_or_build_redbox_cache(samples: list[Path], gold: dict) -> dict:
    cache: dict = {}
    if REDBOX_BBOX_PATH.exists():
        try:
            cache = json.loads(REDBOX_BBOX_PATH.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    dirty = False
    for s_dir in samples:
        sample = s_dir.name.removeprefix("sample_")
        entry = gold.get(sample, {})
        if entry.get("best_guess_border_ratio", 0) < GOLD_BORDER_THRESHOLD:
            continue
        gold_ev = entry.get("best_guess")
        if not gold_ev:
            continue
        if sample in cache and cache[sample].get("evidence") == gold_ev:
            continue
        ev_path = s_dir / gold_ev
        if not ev_path.exists():
            continue
        bbox = detect_redbox_bbox(ev_path)
        cache[sample] = {"evidence": gold_ev, "bbox": list(bbox) if bbox else None}
        dirty = True
    if dirty:
        REDBOX_BBOX_PATH.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return cache


# ---------------------------------------------------------------------------

def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def composite_score(r: MatchResult) -> float:
    aux = 0.4 * r.completeness + 0.4 * r.isolation + 0.2 * r.clarity
    if aux <= 0:
        return float(r.confidence)
    return float(r.confidence) * aux


# Per-provider adjustment factor applied to the gpt-5-rate base estimate.
# OpenAI = 1.0 (real billed rates), Ark (Doubao) and SiliconFlow (Qwen / GLM /
# Kimi) bill ~30-100× lower than gpt-5 — so we scale by 0.02 to bring eval
# totals within ~2× of real spend instead of the previous 30-100× over-count.
# Same constants as pipeline_runner._PROVIDER_COST_FACTOR — kept duplicated
# rather than imported because bench.py is also used standalone by the
# benchmark script and we want it to stand on its own.
_PROVIDER_COST_FACTOR: dict[str, float] = {
    "openai": 1.0,
    "ark": 0.02,
    "siliconflow": 0.02,
}

_settings_cache_bench: Settings | None = None


def cost_estimate(usage: dict[str, int] | None, model: str | None = None) -> float:
    """Approximate USD spend for one LLM call.

    Base = gpt-5-rate scalar (matches the original benchmark formula).
    When `model` is supplied, we multiply by `_PROVIDER_COST_FACTOR[provider]`
    so cross-provider runs in `eval_runner.py` and `benchmark_models.py`
    produce honest USD totals instead of over-counting non-OpenAI providers.
    `model=None` keeps the legacy behaviour for any caller that still hasn't
    been updated.
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
        global _settings_cache_bench
        if _settings_cache_bench is None:
            _settings_cache_bench = Settings.from_env()
        provider_name = _settings_cache_bench.resolve_provider(model).name
    except Exception:
        return base
    factor = _PROVIDER_COST_FACTOR.get(provider_name, 1.0)
    return base * factor


# ---------------------------------------------------------------------------
# main eval loop
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Everything one model produced on the fixture set.

    The benchmark + the legacy single-model runner both consume this; the
    benchmark dumps it under `eval/bench_NNN/<slug>/` and the runner under
    `eval/run_<n>_v<prompt>/`.
    """

    run_dir: Path
    summary: RunSummary
    metrics: dict[str, Any]
    raw_log: list[dict] = field(default_factory=list)
    pairs: list[PairScore] = field(default_factory=list)
    json_retry_count: int = 0
    completion_tokens_total: int = 0
    reasoning_tokens_total: int = 0
    latency_total_s: float = 0.0
    call_count: int = 0  # number of LLM calls that succeeded (per-evidence)
    failed_count: int = 0  # number of LLM calls that raised / timed out
    cost_total: float = 0.0
    aborted: bool = False  # True when we cut a model short (latency / budget)
    abort_reason: str | None = None


def run_eval(
    *,
    model: str,
    settings: Settings,
    run_dir: Path,
    provider: ProviderConfig | None = None,
    verify: bool = False,
    timeout_s: float = 60.0,
    abort_if_mean_latency_over: float | None = None,
    abort_after_n_fixtures: int = 3,
    fixtures_dir: Path | None = None,
    on_progress=None,  # optional callable(sample_name, summary_dict_so_far)
) -> RunResult:
    """Run the matcher on every fixture under `fixtures_dir` for one model.

    Args:
      model: model id, e.g. "doubao-seed-2-0-pro-260215".
      settings: loaded Settings instance.
      run_dir: directory to write crops/ into. Caller pre-creates it.
      provider: optional explicit provider override; otherwise inferred from `model`.
      verify: enable the verify-loop (matches eval_runner's behaviour).
      timeout_s: per-call OpenAI SDK timeout.
      abort_if_mean_latency_over: if set, after `abort_after_n_fixtures`
        completed *fixtures* (not evidences), check mean per-evidence latency.
        If it exceeds this threshold, mark `aborted=True` and stop.
      fixtures_dir: override the default fixtures path.
    """
    fixtures_dir = fixtures_dir or FIXTURES_DEFAULT
    (run_dir / "crops").mkdir(parents=True, exist_ok=True)

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8")) if GOLD_PATH.exists() else {}
    samples = sorted(
        [p for p in fixtures_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")]
    )
    redbox_cache = load_or_build_redbox_cache(samples, gold)

    pairs: list[PairScore] = []
    raw_log: list[dict] = []
    cost_total = 0.0
    correct_selection = 0
    selection_evaluated = 0
    matched = 0
    iou_values: list[float] = []
    retries_total = 0
    completion_tokens_total = 0
    reasoning_tokens_total = 0
    latency_total = 0.0
    call_count = 0
    failed_count = 0
    aborted = False
    abort_reason: str | None = None
    # Verify-loop accounting (only meaningful when verify=True).
    verify_calls = 0
    verified_true = 0
    verified_false = 0
    verify_rejected_to_nr = 0
    verify_cost_total = 0.0

    # Lazy verify import
    match_with_verify = None
    if verify:
        from linebase.verify_loop import match_with_verify as _mwv  # noqa: WPS433

        match_with_verify = _mwv

    fixtures_done = 0

    for s_dir in samples:
        sample = s_dir.name.removeprefix("sample_")
        pngs = sorted(s_dir.glob("*.png"))
        logo = pngs[0]
        expected = pngs[-1]
        evidences = pngs[1:-1]

        gold_entry = gold.get(sample, {})
        has_real_gold = gold_entry.get("best_guess_border_ratio", 0) >= GOLD_BORDER_THRESHOLD
        gold_evidence = gold_entry.get("best_guess") if has_real_gold else None
        redbox_bbox = None
        if has_real_gold and sample in redbox_cache and redbox_cache[sample].get("bbox"):
            redbox_bbox = tuple(redbox_cache[sample]["bbox"])  # type: ignore[assignment]

        print(
            f"\n[{model}] sample {sample}: logo={logo.name} "
            f"evidence_count={len(evidences)} gold={gold_evidence}"
        )

        all_results: list[tuple[Path, MatchResult]] = []
        verify_outcomes: dict[Path, dict | None] = {}
        for ev in evidences:
            t0 = time.time()
            try:
                if verify and match_with_verify is not None:
                    vr = match_with_verify(logo, ev, settings=settings)
                    result = vr.primary
                else:
                    vr = None
                    result = match_logo_in_photo(
                        logo, ev,
                        settings=settings, model=model, provider=provider,
                        timeout=timeout_s,
                    )
            except Exception as e:
                dt = time.time() - t0
                err_msg = f"{type(e).__name__}: {e}"
                print(f"  ! {ev.name}: ERROR {err_msg} (after {dt:.1f}s)")
                failed_count += 1
                raw_log.append({
                    "sample": sample, "evidence": ev.name,
                    "error": err_msg, "latency_s": round(dt, 2),
                })
                continue

            dt = time.time() - t0
            c = cost_estimate(result.usage, model=model)
            cost_total += c
            call_count += 1
            latency_total += dt
            retries_total += result.json_retry_count
            if result.usage:
                completion_tokens_total += result.usage.get("completion_tokens", 0)
                reasoning_tokens_total += result.usage.get("reasoning_tokens", 0)

            v_cost = 0.0
            verify_outcomes[ev] = None
            if verify and vr is not None:
                if vr.iters >= 2:
                    verify_calls += 1
                    # Verify call hits settings.review_model; bill against
                    # the same model so the per-provider factor matches what
                    # actually ran. Cheap to read settings here — already in
                    # process scope via _settings_cache_bench.
                    v_model = settings.review_model
                    v_cost = cost_estimate(vr.verify_usage, model=v_model)
                    verify_cost_total += v_cost
                    cost_total += v_cost
                    if vr.verified:
                        verified_true += 1
                    else:
                        verified_false += 1
                        if result.found:
                            verify_rejected_to_nr += 1
                verify_outcomes[ev] = {
                    "verified": bool(vr.verified),
                    "fit": vr.fit_label,
                    "iters": vr.iters,
                    "final_bbox": list(vr.final_bbox) if vr.final_bbox else None,
                    "verify_confidence": vr.verify_confidence,
                    "verify_reason": vr.verify_reason,
                    "skipped_reason": vr.skipped_reason,
                }
            cs = composite_score(result)
            entry = {
                "sample": sample, "evidence": ev.name,
                "found": result.found, "bbox": result.bbox,
                "confidence": result.confidence,
                "clarity": result.clarity, "completeness": result.completeness,
                "isolation": result.isolation,
                "composite_score": cs,
                "reason": result.reason, "usage": result.usage,
                "cost": c, "latency_s": round(dt, 2),
                "json_retry_count": result.json_retry_count,
            }
            if verify:
                entry["verify"] = verify_outcomes[ev]
                entry["verify_cost"] = v_cost
            raw_log.append(entry)
            mark = "[+]" if result.found else "[ ]"
            extra = ""
            if verify and verify_outcomes[ev] is not None:
                vo = verify_outcomes[ev]
                extra = f" verify={'OK' if vo['verified'] else 'NO'}/{vo['fit']}/i{vo['iters']}"
            print(
                f"  {mark} {ev.name} found={result.found} "
                f"conf={result.confidence:.2f} composite={cs:.3f} dt={dt:.1f}s "
                f"retries={result.json_retry_count}{extra}"
            )
            # In verify mode, only keep evidences that the verify step accepted.
            if verify:
                if vr is not None and vr.verified and result.found:
                    if vr.final_bbox is not None:
                        result = MatchResult(
                            **{**result.__dict__, "bbox": vr.final_bbox}
                        )
                    all_results.append((ev, result))
            else:
                all_results.append((ev, result))

        # rank among found=True
        positives = [(ev, r) for ev, r in all_results if r.found]
        if not positives:
            print("  -> no LLM match on any evidence")
            sel_correct = (False if gold_evidence else None)
            if gold_evidence:
                selection_evaluated += 1
            pairs.append(PairScore(
                sample=sample, candidate="(none)", expected=expected.name,
                phash_distance=64, ssim=0.0, notes="no LLM match",
                iou_vs_redbox=None,
                selection_correct=sel_correct,
            ))
            fixtures_done += 1
            if abort_if_mean_latency_over is not None and fixtures_done >= abort_after_n_fixtures:
                if call_count > 0 and (latency_total / call_count) > abort_if_mean_latency_over:
                    aborted = True
                    abort_reason = (
                        f"mean_latency_s={latency_total/call_count:.1f} > "
                        f"{abort_if_mean_latency_over:.0f} after {fixtures_done} fixtures"
                    )
                    print(f"  !! ABORT: {abort_reason}")
                    break
            continue

        positives.sort(key=lambda t: composite_score(t[1]), reverse=True)
        best_evidence, best_result = positives[0]
        matched += 1

        if gold_evidence:
            selection_evaluated += 1
            is_correct = (best_evidence.name == gold_evidence)
            if is_correct:
                correct_selection += 1
                sel_note = "SEL_OK"
            else:
                sel_note = f"SEL_WRONG gold={gold_evidence}"
            sel_correct = is_correct
        else:
            sel_note = "no_gold"
            sel_correct = None

        mine = run_dir / "crops" / f"{sample}__mine.png"
        expected_copy = run_dir / "crops" / f"{sample}__expected.png"
        if best_result.bbox is not None:
            try:
                crop_to_bbox(best_evidence, best_result.bbox, mine, pad_ratio=0.05)
            except Exception:
                shutil.copy(best_evidence, mine)
        else:
            shutil.copy(best_evidence, mine)
        shutil.copy(expected, expected_copy)

        iou_val: float | None = None
        iou_note = ""
        if gold_evidence and best_evidence.name == gold_evidence:
            if redbox_bbox is None:
                iou_note = " IOU_UNAVAILABLE(red-box not detected)"
            elif best_result.bbox is None:
                iou_note = " IOU_UNAVAILABLE(no bbox)"
            else:
                iou_val = bbox_iou(best_result.bbox, redbox_bbox)
                iou_values.append(iou_val)
                iou_note = f" IoU={iou_val:.2f}"
        elif gold_evidence:
            iou_note = " IoU=n/a (wrong evidence)"

        try:
            ph, sim = score_pair(mine, expected)
        except Exception as e:
            ph, sim = 64, 0.0
            print(f"  ! score error: {e}")

        pairs.append(PairScore(
            sample=sample, candidate=best_evidence.name, expected=expected.name,
            phash_distance=ph, ssim=sim,
            notes=(
                f"conf={best_result.confidence:.2f} "
                f"comp={best_result.completeness:.2f} "
                f"iso={best_result.isolation:.2f} "
                f"clar={best_result.clarity:.2f} "
                f"bbox={best_result.bbox}  {sel_note}{iou_note}"
            ),
            iou_vs_redbox=iou_val,
            selection_correct=sel_correct,
        ))
        print(
            f"  -> best={best_evidence.name} composite={composite_score(best_result):.3f} "
            f"bbox={best_result.bbox} pHash={ph} SSIM={sim:.3f}  {sel_note}{iou_note}"
        )

        fixtures_done += 1
        if abort_if_mean_latency_over is not None and fixtures_done >= abort_after_n_fixtures:
            if call_count > 0 and (latency_total / call_count) > abort_if_mean_latency_over:
                aborted = True
                abort_reason = (
                    f"mean_latency_s={latency_total/call_count:.1f} > "
                    f"{abort_if_mean_latency_over:.0f} after {fixtures_done} fixtures"
                )
                print(f"  !! ABORT: {abort_reason}")
                break

        if on_progress is not None:
            try:
                on_progress(sample, {
                    "fixtures_done": fixtures_done,
                    "correct_selection": correct_selection,
                    "selection_evaluated": selection_evaluated,
                    "cost_total": cost_total,
                    "latency_total": latency_total,
                    "call_count": call_count,
                })
            except Exception:
                pass

    sel_acc = (correct_selection / selection_evaluated) if selection_evaluated else None
    bbox_iou_mean = (sum(iou_values) / len(iou_values)) if iou_values else None
    bbox_iou_pass_50 = (
        sum(1 for v in iou_values if v >= 0.5) / len(iou_values)
    ) if iou_values else None

    pv = ""
    # Find the prompt version used (env-overridable in eval_runner). Cheap to peek.
    import os as _os
    pv = _os.environ.get("LINEBASE_PROMPT_VERSION", "") or "?"

    summary = RunSummary(
        run_id=run_dir.name, prompt_version=pv, model=model,
        samples=len(samples), matched=matched,
        mean_ssim=sum(p.ssim for p in pairs) / max(1, len(pairs)),
        mean_phash=sum(p.phash_distance for p in pairs) / max(1, len(pairs)),
        pass_rate_ssim_50=sum(1 for p in pairs if p.ssim >= 0.5) / max(1, len(pairs)),
        cost_usd_estimate=cost_total, pairs=pairs,
    )
    metrics_dict = summary.to_dict()
    metrics_dict.update({
        "selection_accuracy": sel_acc,
        "selection_evaluated": selection_evaluated,
        "correct_selection": correct_selection,
        "bbox_iou_mean": bbox_iou_mean,
        "bbox_iou_pass_50": bbox_iou_pass_50,
        "bbox_iou_scored": len(iou_values),
        "json_retry_count": retries_total,
        "completion_tokens_total": completion_tokens_total,
        "reasoning_tokens_total": reasoning_tokens_total,
        "mean_completion_tokens": (
            completion_tokens_total / call_count if call_count else 0.0
        ),
        "mean_reasoning_tokens": (
            reasoning_tokens_total / call_count if call_count else 0.0
        ),
        "mean_latency_s": (latency_total / call_count) if call_count else 0.0,
        "call_count": call_count,
        "failed_count": failed_count,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "verify_enabled": verify,
    })
    if verify:
        metrics_dict.update({
            "verify_calls": verify_calls,
            "verified_true": verified_true,
            "verified_false": verified_false,
            "verify_rejected_to_NR": verify_rejected_to_nr,
            "verify_cost_usd_estimate": verify_cost_total,
        })

    return RunResult(
        run_dir=run_dir,
        summary=summary,
        metrics=metrics_dict,
        raw_log=raw_log,
        pairs=pairs,
        json_retry_count=retries_total,
        completion_tokens_total=completion_tokens_total,
        reasoning_tokens_total=reasoning_tokens_total,
        latency_total_s=latency_total,
        call_count=call_count,
        failed_count=failed_count,
        cost_total=cost_total,
        aborted=aborted,
        abort_reason=abort_reason,
    )


def persist_run(result: RunResult) -> None:
    """Dump metrics.json, raw_log.json, report.html into result.run_dir."""
    write_html_report(result.summary, result.run_dir)
    write_metrics_json(result.summary, result.run_dir)
    # Overwrite metrics.json with the augmented dict (includes selection_accuracy etc).
    (result.run_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result.run_dir / "raw_log.json").write_text(
        json.dumps(result.raw_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
