"""Generic eval runner — pick a prompt version, score against:
  (1) evidence-selection accuracy vs red-box ground truth,
  (2) bbox IoU vs the human-drawn red box on the gold evidence image,
  (3) (secondary) bbox-crop SSIM/pHash vs expected-crop image — kept as a sanity check.

Usage:
  python scripts/eval_runner.py            # uses latest prompts/v_*.md
  LINEBASE_PROMPT_VERSION=2 python scripts/eval_runner.py
  LINEBASE_VERIFY=1        python scripts/eval_runner.py

This is a thin wrapper around `linebase.bench.run_eval` — the heavy lifting
moved into that shared module so the benchmark script can reuse it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from linebase import store
from linebase.bench import persist_run, run_eval
from linebase.config import Settings

EVAL_ROOT = REPO / "eval"


def _verify_enabled() -> bool:
    return os.environ.get("LINEBASE_VERIFY", "").strip().lower() in {"1", "true", "yes", "on"}


def find_run_dir(prompt_version: str) -> Path:
    EVAL_ROOT.mkdir(exist_ok=True)
    existing = sorted(EVAL_ROOT.glob("run_*"))
    next_n = 1 + max(
        [int(p.name.split("_")[1]) for p in existing if p.name.split("_")[1].isdigit()] + [0]
    )
    d = EVAL_ROOT / f"run_{next_n:03d}_v{prompt_version}"
    d.mkdir(parents=True)
    (d / "crops").mkdir()
    return d


def main() -> int:
    settings = Settings.from_env()
    pv = os.environ.get("LINEBASE_PROMPT_VERSION")
    if pv is None:
        files = sorted((REPO / "prompts").glob("v_*.md"))
        if not files:
            print("No prompts/v_*.md found", file=sys.stderr)
            return 1
        pv = files[-1].stem.removeprefix("v_")
    os.environ["LINEBASE_PROMPT_VERSION"] = pv
    use_verify = _verify_enabled()
    run_dir = find_run_dir(pv)
    print(
        f"[eval] run_dir={run_dir}  model={settings.model}  prompt=v_{pv}"
        f"  verify={'ON' if use_verify else 'off'}"
    )

    result = run_eval(
        model=settings.model,
        settings=settings,
        run_dir=run_dir,
        verify=use_verify,
        timeout_s=120.0,
    )
    persist_run(result)
    store.insert_eval_run(
        prompt_version=f"v{pv}", model=settings.model, metrics=result.metrics
    )

    m = result.metrics
    summary = result.summary
    print(f"\n=== {run_dir.name} done ===")
    print(f"samples={summary.samples}  matched={summary.matched}")
    sel_acc = m.get("selection_accuracy")
    if sel_acc is not None:
        print(
            f"selection_acc={m['correct_selection']}/{m['selection_evaluated']} "
            f"({sel_acc:.0%})"
        )
    else:
        print("selection_acc=n/a")
    bbox_iou_mean = m.get("bbox_iou_mean")
    if bbox_iou_mean is not None:
        print(
            f"bbox_iou_mean={bbox_iou_mean:.3f} (n={m['bbox_iou_scored']})  "
            f"bbox_iou_pass_50={m['bbox_iou_pass_50']:.0%}"
        )
    else:
        print("bbox_iou: no scored samples")
    print(
        f"[secondary] mean_ssim={summary.mean_ssim:.3f} "
        f"pass@SSIM>=0.5={summary.pass_rate_ssim_50:.0%}"
    )
    print(f"cost=${summary.cost_usd_estimate:.3f}")
    if use_verify:
        print(
            f"verify: calls={m.get('verify_calls',0)} "
            f"verified_true={m.get('verified_true',0)} "
            f"verified_false={m.get('verified_false',0)} "
            f"rejected_to_NR={m.get('verify_rejected_to_NR',0)} "
            f"extra_cost=${m.get('verify_cost_usd_estimate',0.0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
