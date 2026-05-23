"""Compare two or more eval runs: print a delta table across prompt versions.

Usage:
  python scripts/compare_runs.py                    # diff latest two runs
  python scripts/compare_runs.py run_001 run_002    # explicit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
EVAL = REPO / "eval"


def load_run(name: str) -> dict:
    candidates = sorted(EVAL.glob(f"{name}*"))
    if not candidates:
        raise FileNotFoundError(f"No run matching {name}")
    metrics = candidates[0] / "metrics.json"
    return {"run_id": candidates[0].name, **json.loads(metrics.read_text(encoding="utf-8"))}


def _fmt_sel(p: dict) -> str:
    sc = p.get("selection_correct")
    if sc is True:
        return "OK"
    if sc is False:
        return "X"
    # legacy fallback: try to derive from notes for old runs that didn't store the field
    notes = p.get("notes", "")
    if "SEL_OK" in notes:
        return "OK"
    if "SEL_WRONG" in notes:
        return "X"
    return "-"


def _fmt_iou(p: dict) -> str:
    iou = p.get("iou_vs_redbox")
    if iou is None:
        return "n/a"
    return f"{float(iou):.2f}"


def main() -> int:
    if len(sys.argv) >= 3:
        names = sys.argv[1:]
    else:
        runs = sorted(EVAL.glob("run_*"))
        if len(runs) < 2:
            print("Need at least two runs to compare")
            return 1
        names = [r.name for r in runs[-2:]]
    runs_data = [load_run(n) for n in names]

    col_w = 32
    print(f"{'sample':<10} " + " ".join(f"{r['run_id']:<{col_w}}" for r in runs_data))
    print("-" * (10 + (col_w + 1) * len(runs_data)))
    by_sample: dict[str, list[dict]] = {}
    for r in runs_data:
        for p in r["pairs"]:
            by_sample.setdefault(p["sample"], []).append({"run": r["run_id"], **p})
    for sample, entries in sorted(by_sample.items()):
        cells = []
        for e in entries:
            ssim = float(e.get("ssim", 0))
            cell = f"sel={_fmt_sel(e)} iou={_fmt_iou(e)} ssim={ssim:.2f}"
            cells.append(cell)
        print(f"{sample:<10} " + " ".join(f"{c:<{col_w}}" for c in cells))

    print()
    mw = 20
    print(f"{'metric':<25} " + " ".join(f"{r['run_id']:<{mw}}" for r in runs_data))
    print("-" * (25 + (mw + 1) * len(runs_data)))
    metric_keys = (
        "samples",
        "matched",
        "selection_evaluated",
        "correct_selection",
        "selection_accuracy",
        "bbox_iou_scored",
        "bbox_iou_mean",
        "bbox_iou_pass_50",
        "mean_ssim",
        "pass_rate_ssim_50",
        "cost_usd_estimate",
    )
    for key in metric_keys:
        vals = []
        for r in runs_data:
            v = r.get(key)
            if v is None:
                vals.append("n/a")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        print(f"{key:<25} " + " ".join(f"{v:<{mw}}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
