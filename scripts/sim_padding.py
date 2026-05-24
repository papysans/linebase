"""Take an existing score-run JSON and re-compute IoU with various bbox-pad
amounts applied to predictions. Pure analysis — no LLM calls.

Padding model: expand bbox by pct/2 on each side around its center, clamped
to image dims (use evidence image's size loaded from disk).
"""
import json
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def iou(a, b):
    if not a or not b: return 0.0
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    A = max(0,ax2-ax1)*max(0,ay2-ay1)
    B = max(0,bx2-bx1)*max(0,by2-by1)
    return inter / (A+B-inter) if (A+B-inter) > 0 else 0.0


def pad_bbox(b, pct, W, H):
    if not b: return None
    x1,y1,x2,y2 = b
    w, h = x2-x1, y2-y1
    px, py = w*pct/2, h*pct/2
    return [max(0,int(x1-px)), max(0,int(y1-py)), min(W,int(x2+px)), min(H,int(y2+py))]


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else None
    if not run:
        runs = sorted(Path(".data/score_runs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        run = str(runs[0])
    data = json.loads(Path(run).read_text(encoding="utf-8"))
    rs = data["results"]
    print(f"Loaded {len(rs)} results from {run}")
    print(f"Model: {data['model']}  verify: {data['verify']}\n")

    # Get evidence dims for each row (from disk)
    for r in rs:
        ev = Path(r["evidence"])
        with Image.open(ev) as img:
            r["W"], r["H"] = img.size

    pcts = [0.0, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20, 1.80, 2.50]
    thresholds = [0.1, 0.3, 0.5, 0.7]

    print(f"{'pad%':>6} | {'IoU≥0.1':>8} | {'IoU≥0.3':>8} | {'IoU≥0.5':>8} | {'IoU≥0.7':>8} | avg_iou")
    print('-'*72)
    for pct in pcts:
        ious = []
        for r in rs:
            pred = pad_bbox(r["pred_bbox"], pct, r["W"], r["H"])
            ious.append(iou(r["truth_bbox"], pred))
        row = [f"{100*pct:>4.0f}%"]
        for t in thresholds:
            n = sum(1 for v in ious if v >= t)
            row.append(f"{n:>2}/{len(ious)}    ")
        avg = sum(ious)/len(ious)
        row.append(f"{avg:.3f}")
        print(" | ".join(row))
    print()


if __name__ == "__main__":
    main()
