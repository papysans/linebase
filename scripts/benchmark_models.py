"""Multi-model benchmark — sweep N multimodal models on the 10-fixture set.

Sequential, no verify-loop, one prompt (v_2). Writes per-model artifacts under
`eval/bench_NNN/<slug>/` and a summary at `eval/bench_NNN/summary.{md,json}`.

Guardrails:
  * HARD STOP at cumulative cost > $4 (PRD).
  * Kimi-K2.5: abort that model's run if mean_latency_s > 60s after the first 3 fixtures.
  * Per-call timeout: 300s for K2.5, 60s for everyone else (override-able per row).
  * Models run **sequentially** — parallel runs tangle latency stats and trip rate limits.

Usage:
  python scripts/benchmark_models.py
  python scripts/benchmark_models.py --prompt 2 --limit-to gpt-5.5,doubao-seed-2-0-pro-260215
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from linebase import store
from linebase.bench import RunResult, persist_run, run_eval
from linebase.config import Settings


EVAL_ROOT = REPO / "eval"
HARD_BUDGET_USD = 4.0


@dataclass(frozen=True)
class Candidate:
    model: str                   # full model id passed to the provider
    provider_name: str           # "openai" | "ark" | "siliconflow"
    timeout_s: float
    abort_if_mean_latency_over: float | None = None
    abort_after_n_fixtures: int = 3
    notes: str = ""


# The 7 confirmed-working multimodal models (probed by the main session).
CANDIDATES: list[Candidate] = [
    Candidate("gpt-5.5", "openai", timeout_s=60.0),
    Candidate("doubao-seed-2-0-pro-260215", "ark", timeout_s=60.0,
              notes="thinking model, ~500 reasoning tokens"),
    Candidate("doubao-seed-2-0-mini-260428", "ark", timeout_s=60.0,
              notes="thinking model, ~540 reasoning tokens"),
    Candidate("zai-org/GLM-4.5V", "siliconflow", timeout_s=60.0,
              notes="wraps output in <|begin_of_box|>...<|end_of_box|>"),
    Candidate("Qwen/Qwen3-VL-32B-Instruct", "siliconflow", timeout_s=60.0),
    Candidate("Qwen/Qwen3-VL-30B-A3B-Instruct", "siliconflow", timeout_s=60.0,
              notes="fastest probed"),
    Candidate("Pro/moonshotai/Kimi-K2.5", "siliconflow", timeout_s=300.0,
              abort_if_mean_latency_over=60.0, abort_after_n_fixtures=3,
              notes="heavy thinking, ~2400 reasoning tokens"),
]


def slugify(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")


def find_bench_dir() -> Path:
    EVAL_ROOT.mkdir(exist_ok=True)
    existing = sorted(EVAL_ROOT.glob("bench_*"))
    next_n = 1 + max(
        [int(p.name.split("_")[1]) for p in existing if p.name.split("_")[1].isdigit()] + [0]
    )
    d = EVAL_ROOT / f"bench_{next_n:03d}"
    d.mkdir(parents=True)
    return d


def summarize_row(c: Candidate, r: RunResult) -> dict:
    m = r.metrics
    return {
        "model": c.model,
        "provider": c.provider_name,
        "selection_acc": m.get("selection_accuracy"),
        "selection_correct": m.get("correct_selection"),
        "selection_evaluated": m.get("selection_evaluated"),
        "mean_iou": m.get("bbox_iou_mean"),
        "iou_pass_50": m.get("bbox_iou_pass_50"),
        "iou_scored": m.get("bbox_iou_scored"),
        "mean_ssim": m.get("mean_ssim"),
        "n_correct": m.get("correct_selection"),
        "total_cost_usd": r.cost_total,
        "mean_latency_s": m.get("mean_latency_s"),
        "mean_completion_tokens": m.get("mean_completion_tokens"),
        "mean_reasoning_tokens": m.get("mean_reasoning_tokens"),
        "json_retry_count": m.get("json_retry_count"),
        "call_count": m.get("call_count"),
        "failed_count": m.get("failed_count"),
        "aborted": m.get("aborted"),
        "abort_reason": m.get("abort_reason"),
    }


def _fmt(v, fmt=".3f"):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:{fmt}}"
    return str(v)


def write_summary_md(bench_dir: Path, rows: list[dict], prompt_version: str) -> Path:
    lines = []
    lines.append(f"# Multi-model benchmark — {bench_dir.name}\n")
    lines.append(f"- Prompt: v_{prompt_version}")
    lines.append(f"- Fixtures: 10 (sample_* under .trellis/tasks/.../fixtures)")
    lines.append(f"- Verify-loop: OFF")
    lines.append(f"- Hard budget cap: ${HARD_BUDGET_USD:.2f}")
    total_cost = sum(r["total_cost_usd"] for r in rows)
    lines.append(f"- Total spend: ${total_cost:.3f}")
    lines.append("")
    lines.append(
        "| model | provider | sel_acc | n_correct | mean_iou | iou_pass_50 | "
        "mean_ssim | cost_usd | mean_latency_s | mean_completion_tokens | "
        "mean_reasoning_tokens | json_retries | aborted |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for r in rows:
        lines.append(
            f"| `{r['model']}` | {r['provider']} | "
            f"{_fmt(r['selection_acc'], '.0%') if r['selection_acc'] is not None else 'n/a'} | "
            f"{r['n_correct']}/{r['selection_evaluated']} | "
            f"{_fmt(r['mean_iou'])} | "
            f"{_fmt(r['iou_pass_50'], '.0%') if r['iou_pass_50'] is not None else 'n/a'} | "
            f"{_fmt(r['mean_ssim'])} | "
            f"${r['total_cost_usd']:.4f} | "
            f"{_fmt(r['mean_latency_s'], '.1f')} | "
            f"{_fmt(r['mean_completion_tokens'], '.0f')} | "
            f"{_fmt(r['mean_reasoning_tokens'], '.0f')} | "
            f"{r['json_retry_count']} | "
            f"{_fmt(r['aborted'])} |"
        )
    md_path = bench_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="2", help="Prompt version to evaluate (default: 2)")
    p.add_argument(
        "--limit-to",
        default="",
        help="Comma-separated subset of model ids to run (default: all 7)",
    )
    p.add_argument(
        "--budget", type=float, default=HARD_BUDGET_USD,
        help=f"Hard $ cap, kills the sweep on overflow (default: {HARD_BUDGET_USD})",
    )
    p.add_argument(
        "--no-verify-budget",
        action="store_true",
        help="Disable the hard $ cap (debug only).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    os.environ["LINEBASE_PROMPT_VERSION"] = args.prompt

    subset = {m.strip() for m in args.limit_to.split(",") if m.strip()}
    if subset:
        cands = [c for c in CANDIDATES if c.model in subset]
        if not cands:
            print(f"No candidates match --limit-to={args.limit_to}", file=sys.stderr)
            return 1
    else:
        cands = list(CANDIDATES)

    # Pre-flight: every requested provider must be configured.
    for c in cands:
        if c.provider_name not in settings.providers:
            print(
                f"FAIL: model {c.model!r} needs provider {c.provider_name!r}, "
                f"but only {sorted(settings.providers)} are configured.",
                file=sys.stderr,
            )
            return 2

    bench_dir = find_bench_dir()
    print(f"[bench] dir={bench_dir} prompt=v_{args.prompt} "
          f"models={len(cands)} budget=${args.budget:.2f}")

    rows: list[dict] = []
    cumulative_cost = 0.0
    t_start = time.time()

    for i, c in enumerate(cands, 1):
        slug = slugify(c.model)
        run_dir = bench_dir / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        provider = settings.providers[c.provider_name]
        print(
            f"\n=== [{i}/{len(cands)}] {c.model} via {c.provider_name} "
            f"(timeout={c.timeout_s:.0f}s) ==="
        )
        if c.notes:
            print(f"  notes: {c.notes}")
        t_model = time.time()
        try:
            result = run_eval(
                model=c.model,
                settings=settings,
                run_dir=run_dir,
                provider=provider,
                verify=False,
                timeout_s=c.timeout_s,
                abort_if_mean_latency_over=c.abort_if_mean_latency_over,
                abort_after_n_fixtures=c.abort_after_n_fixtures,
            )
        except Exception as e:
            elapsed = time.time() - t_model
            print(f"  !! FATAL on {c.model}: {type(e).__name__}: {e} (after {elapsed:.0f}s)")
            # Persist a stub metrics.json so summary still shows the row.
            stub = {
                "model": c.model, "provider": c.provider_name,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": elapsed,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(stub, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            rows.append({
                "model": c.model, "provider": c.provider_name,
                "selection_acc": None, "selection_correct": 0, "selection_evaluated": 0,
                "mean_iou": None, "iou_pass_50": None, "iou_scored": 0,
                "mean_ssim": None, "n_correct": 0, "total_cost_usd": 0.0,
                "mean_latency_s": None, "mean_completion_tokens": None,
                "mean_reasoning_tokens": None, "json_retry_count": 0,
                "call_count": 0, "failed_count": 0, "aborted": True,
                "abort_reason": f"fatal: {type(e).__name__}: {e}",
            })
            continue

        persist_run(result)
        store.insert_eval_run(
            prompt_version=f"v{args.prompt}", model=c.model, metrics=result.metrics
        )
        row = summarize_row(c, result)
        rows.append(row)
        cumulative_cost += result.cost_total
        elapsed = time.time() - t_model
        sel = row["selection_acc"]
        sel_s = f"{sel:.0%}" if sel is not None else "n/a"
        iou = row["mean_iou"]
        iou_s = f"{iou:.3f}" if iou is not None else "n/a"
        print(
            f"  -> {c.model}: sel_acc={sel_s}  mean_iou={iou_s}  "
            f"cost=${result.cost_total:.4f}  cum=${cumulative_cost:.4f}  "
            f"mean_latency={row['mean_latency_s']:.1f}s  "
            f"retries={row['json_retry_count']}  elapsed={elapsed:.0f}s"
        )

        # Incremental summary write — survives mid-sweep crashes.
        write_summary_md(bench_dir, rows, args.prompt)
        (bench_dir / "summary.json").write_text(
            json.dumps({
                "prompt_version": args.prompt,
                "cumulative_cost_usd": cumulative_cost,
                "completed": i,
                "total": len(cands),
                "rows": rows,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if not args.no_verify_budget and cumulative_cost > args.budget:
            print(
                f"\n!! HARD STOP: cumulative cost ${cumulative_cost:.3f} "
                f"> budget ${args.budget:.2f}. Aborting sweep."
            )
            break

    total_elapsed = time.time() - t_start
    print(f"\n=== bench done in {total_elapsed/60:.1f} min, "
          f"spend=${cumulative_cost:.4f} ===")
    md = write_summary_md(bench_dir, rows, args.prompt)
    print(f"summary: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
