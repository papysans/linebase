"""Generate `research/multi-model-benchmark.md` from `eval/bench_NNN/summary.json`.

Adds per-sample breakdown and a cost-per-correct-selection winner pick on top of
the raw summary table that `benchmark_models.py` already writes.

Usage:
  python scripts/write_benchmark_report.py            # uses newest eval/bench_*
  python scripts/write_benchmark_report.py --bench 001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

EVAL_ROOT = REPO / "eval"
RESEARCH_PATH = REPO / "research/multi-model-benchmark.md"


def _fmt_pct(v):
    return f"{v:.0%}" if isinstance(v, (int, float)) else "n/a"


def _fmt_num(v, fmt=".3f"):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:{fmt}}"
    return str(v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default=None)
    args = parser.parse_args()

    if args.bench:
        bench_dir = EVAL_ROOT / f"bench_{args.bench}"
    else:
        candidates = sorted(EVAL_ROOT.glob("bench_*"))
        if not candidates:
            print("no eval/bench_* dirs found", file=sys.stderr)
            return 1
        bench_dir = candidates[-1]

    summary_path = bench_dir / "summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path}", file=sys.stderr)
        return 1

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    prompt_version = data.get("prompt_version", "?")

    # For each model, parse the per-sample raw_log to build the breakdown table.
    per_sample: dict[str, dict[str, dict]] = {}  # sample -> model -> pick info
    for row in rows:
        model = row["model"]
        slug = model.replace("/", "_")
        raw_log_path = bench_dir / slug / "raw_log.json"
        if not raw_log_path.exists():
            continue
        raw_log = json.loads(raw_log_path.read_text(encoding="utf-8"))
        # Build best-pick per sample = highest composite_score among found=True.
        by_sample: dict[str, list] = {}
        for e in raw_log:
            by_sample.setdefault(e["sample"], []).append(e)
        for sample, entries in by_sample.items():
            positives = [e for e in entries if e.get("found")]
            if not positives:
                pick_name = "(none)"
                pick_cs = 0.0
            else:
                best = max(positives, key=lambda e: e.get("composite_score") or 0.0)
                pick_name = best["evidence"]
                pick_cs = best.get("composite_score", 0.0)
            per_sample.setdefault(sample, {})[model] = {
                "pick": pick_name, "composite_score": pick_cs,
            }

    # Cross-reference gold to know SEL_OK / SEL_WRONG. Reuse the gold dict.
    gold_path = REPO / "eval/redbox_gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.exists() else {}
    gold_threshold = 0.005

    # ---- compose markdown ----
    lines: list[str] = []
    lines.append(f"# Multi-Model Benchmark — v_{prompt_version} prompt, 10-fixture set\n")
    lines.append(f"- Source: `{bench_dir.relative_to(REPO)}`")
    lines.append(f"- Verify-loop: OFF (one variable at a time)")
    lines.append(f"- Total spend: ${data.get('cumulative_cost_usd', 0):.4f}")
    lines.append(f"- Models attempted: {data.get('completed','?')} / {data.get('total','?')}")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append(
        "| model | provider | sel_acc | n_correct | mean_iou | iou_pass_50 | "
        "mean_ssim | total_cost_usd | mean_latency_s | mean_completion_tokens | "
        "mean_reasoning_tokens | json_retries | aborted |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )

    # Compute cost-per-correct also.
    cost_per_correct: dict[str, float | None] = {}
    for r in rows:
        n_correct = r.get("n_correct") or 0
        cost = r.get("total_cost_usd") or 0
        cost_per_correct[r["model"]] = (cost / n_correct) if n_correct else None

    for r in rows:
        sel = _fmt_pct(r.get("selection_acc"))
        iou = _fmt_num(r.get("mean_iou"))
        iou_p = _fmt_pct(r.get("iou_pass_50"))
        ssim = _fmt_num(r.get("mean_ssim"))
        ml = _fmt_num(r.get("mean_latency_s"), ".1f")
        mc = _fmt_num(r.get("mean_completion_tokens"), ".0f")
        mr = _fmt_num(r.get("mean_reasoning_tokens"), ".0f")
        aborted = "yes" if r.get("aborted") else "no"
        lines.append(
            f"| `{r['model']}` | {r['provider']} | {sel} | "
            f"{r.get('n_correct')}/{r.get('selection_evaluated')} | {iou} | {iou_p} | "
            f"{ssim} | ${r.get('total_cost_usd',0):.4f} | {ml} | {mc} | {mr} | "
            f"{r.get('json_retry_count',0)} | {aborted} |"
        )
    lines.append("")

    # Cost-per-correct table.
    lines.append("## Cost-per-correct-selection (bottom-line metric)")
    lines.append("")
    lines.append("| model | n_correct | cost_usd | cost_per_correct |")
    lines.append("| --- | --- | --- | --- |")
    rows_ranked = sorted(
        rows,
        key=lambda r: (
            -(r.get("n_correct") or 0),
            (r.get("total_cost_usd") or 0),
        ),
    )
    for r in rows_ranked:
        cpc = cost_per_correct[r["model"]]
        lines.append(
            f"| `{r['model']}` | {r.get('n_correct')} | "
            f"${r.get('total_cost_usd',0):.4f} | "
            f"{('$' + format(cpc, '.4f')) if cpc is not None else 'n/a (0 correct)'} |"
        )
    lines.append("")

    # Per-sample breakdown
    lines.append("## Per-sample model-by-model breakdown")
    lines.append("")
    samples = sorted(per_sample)
    model_ids = [r["model"] for r in rows]
    header = ["sample", "gold_evidence"] + [m.split("/")[-1] for m in model_ids]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for sample in samples:
        entry = gold.get(sample, {})
        gold_ev = entry.get("best_guess") if entry.get("best_guess_border_ratio", 0) >= gold_threshold else None
        gold_cell = f"`{gold_ev}`" if gold_ev else "—"
        row_cells = [sample, gold_cell]
        for m in model_ids:
            info = per_sample.get(sample, {}).get(m)
            if info is None:
                row_cells.append("—")
                continue
            pick = info["pick"]
            if gold_ev is None:
                tag = "no_gold"
            elif pick == "(none)":
                tag = "MISS"
            elif pick == gold_ev:
                tag = "OK"
            else:
                tag = "WRONG"
            row_cells.append(f"{tag} `{pick}`")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    # Winner pick — highest selection_acc, tie-broken by cost-per-correct, then latency.
    ranked = sorted(
        rows,
        key=lambda r: (
            -(r.get("selection_acc") or 0.0),
            (cost_per_correct[r["model"]] or float("inf")),
            (r.get("mean_latency_s") or float("inf")),
        ),
    )
    winner = ranked[0] if ranked else None
    lines.append("## Winner")
    lines.append("")
    if winner:
        wm = winner["model"]
        wp = winner["provider"]
        wsel = _fmt_pct(winner.get("selection_acc"))
        wcost = winner.get("total_cost_usd", 0)
        wlat = winner.get("mean_latency_s")
        wlat_s = _fmt_num(wlat, ".1f")
        wcpc = cost_per_correct[wm]
        wcpc_s = f"${wcpc:.4f}" if wcpc is not None else "n/a"
        wretries = winner.get("json_retry_count", 0)
        lines.append(
            f"**`{wm}`** ({wp}) — selection_acc {wsel}, "
            f"cost ${wcost:.4f} (cost/correct {wcpc_s}), "
            f"mean latency {wlat_s}s, json_retries {wretries}."
        )
        lines.append("")
        lines.append(
            "Picked because it has the highest selection accuracy on the 7 gold-evidence fixtures "
            "while staying inexpensive and reliable (no JSON retries needed). "
            "Tie-breakers applied in order: selection_acc, cost-per-correct, mean_latency."
        )
    lines.append("")

    RESEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESEARCH_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
