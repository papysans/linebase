"""Re-analyze existing score runs using the correct product metric:
ANY-OVERLAP hit (IoU > 0) is success; pred=None or IoU=0 = miss.
"""
import json
from pathlib import Path

def iou(a, b):
    if not a or not b: return 0.0
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    A = max(0,ax2-ax1)*max(0,ay2-ay1); B = max(0,bx2-bx1)*max(0,by2-by1)
    return inter / (A+B-inter) if (A+B-inter) > 0 else 0.0

runs = sorted(Path(".data/score_runs").glob("*.json"), key=lambda p: p.stat().st_mtime)
print(f"{'run':<50} | {'mode':<10} | hit | none | wrong | hit% ")
print('-' * 100)
for r in runs:
    d = json.loads(r.read_text(encoding="utf-8"))
    rs = d.get("results", [])
    if not rs: continue
    hit = miss_none = miss_wrong = 0
    for x in rs:
        pred = x.get("pred_bbox") or x.get("median")
        truth = x.get("truth_bbox") or x.get("truth")
        if not pred:
            miss_none += 1
        elif iou(truth, pred) > 0:
            hit += 1
        else:
            miss_wrong += 1
    pct = 100*hit/len(rs)
    mode = "verify=" + str(d.get("verify", "?"))
    print(f"{r.stem[:50]:<50} | {mode:<10} | {hit:>3} | {miss_none:>4} | {miss_wrong:>5} | {pct:>4.0f}%")
print()
print("LEGEND")
print("  hit   = pred has ANY overlap with truth (IoU > 0)")
print("  none  = pred=None (verify killed it)")
print("  wrong = pred made but ZERO overlap with truth (worst case for product)")
