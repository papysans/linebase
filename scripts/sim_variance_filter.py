"""Simulate applying the variance pre-gate to Pass-1 predictions (no LLM calls).
For each predicted bbox, crop the evidence and compute std+white_ratio.
Reject as 'blank' if std<15 OR white_ratio>0.7.
Report: how many of the 12 WRONG predictions get filtered out vs how many HITs are accidentally lost."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
runs = sorted(Path(".data/score_runs").glob("*Qwen*.json"), key=lambda p: p.stat().st_mtime)
run = next(r for r in reversed(runs) if json.loads(r.read_text(encoding="utf-8")).get("verify") is False)
data = json.loads(run.read_text(encoding="utf-8"))


def iou(a, b):
    if not a or not b: return 0.0
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    A=max(0,ax2-ax1)*max(0,ay2-ay1); B=max(0,bx2-bx1)*max(0,by2-by1)
    return inter / (A+B-inter) if (A+B-inter) > 0 else 0.0


def crop_stats(evidence_path: Path, bbox: list) -> tuple[float, float]:
    img = Image.open(evidence_path).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1=max(0,x1); y1=max(0,y1); x2=min(img.width,x2); y2=min(img.height,y2)
    if x2<=x1 or y2<=y1: return 0.0, 1.0
    arr = np.asarray(img.crop((x1,y1,x2,y2)))
    std = float(arr.std())
    white = float(((arr > 240).all(axis=-1)).mean())
    return std, white


# Test variance + white-ratio thresholds
thresholds = [
    (15.0, 0.7),  # current iter-5 default
    (10.0, 0.85),
    (8.0, 0.85),
    (5.0, 0.85),
    (5.0, 0.80),
    (3.0, 0.90),
]

for std_thr, white_thr in thresholds:
    keep_hit = lose_hit = filter_wrong = keep_wrong = filter_none = 0
    for r in data["results"]:
        pred = r["pred_bbox"]; truth = r["truth_bbox"]
        if pred is None:
            filter_none += 1
            continue
        i = iou(truth, pred)
        std, white = crop_stats(Path(r["evidence"]), pred)
        is_blank = std < std_thr or white > white_thr
        if i > 0:
            if is_blank: lose_hit += 1
            else: keep_hit += 1
        else:
            if is_blank: filter_wrong += 1
            else: keep_wrong += 1
    total = len(data["results"])
    print(f"thr(std<{std_thr},white>{white_thr}): kept_hit={keep_hit}/6  lost_hit={lose_hit}/6  filtered_wrong={filter_wrong}/12  kept_wrong={keep_wrong}/12  →  hit_rate={keep_hit}/{total}={100*keep_hit/total:.0f}%, wrong_rate={keep_wrong}/{total}={100*keep_wrong/total:.0f}%")

# Also: WHAT std/white do hit cases vs wrong cases actually have?
print("\nPer-case std/white_ratio:")
print(f"{'tm_pair':<20} {'label':<6} {'std':>6} {'white':>6}")
for r in data["results"]:
    pred = r["pred_bbox"]; truth = r["truth_bbox"]
    if pred is None: continue
    std, white = crop_stats(Path(r["evidence"]), pred)
    label = "HIT" if iou(truth, pred) > 0 else "WRONG"
    tag = f"{r['tm']}_p{r['pair_idx']:02d}"
    print(f"  {tag:<20} {label:<6} {std:>6.1f} {white:>6.2f}")
