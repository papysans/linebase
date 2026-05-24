"""For each of the 19 truth pairs, render a side-by-side image:
  [evidence with TRUTH bbox green] + [evidence with PRED bbox red] + [expected crop]
Save into docs/baseline_report_qwen_noverify/. Also write a markdown index.

Uses the latest qwen no-verify run.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "baseline_report_qwen_noverify"
OUT.mkdir(parents=True, exist_ok=True)

# pick the qwen no-verify run
runs = sorted(Path(".data/score_runs").glob("*Qwen*.json"), key=lambda p: p.stat().st_mtime)
run = None
for r in reversed(runs):
    d = json.loads(r.read_text(encoding="utf-8"))
    if d.get("verify") is False:
        run = r
        break
print(f"Using: {run.name}")
data = json.loads(run.read_text(encoding="utf-8"))


def iou(a, b):
    if not a or not b: return 0.0
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    A = max(0,ax2-ax1)*max(0,ay2-ay1); B = max(0,bx2-bx1)*max(0,by2-by1)
    return inter / (A+B-inter) if (A+B-inter) > 0 else 0.0


md = ["# Baseline Report — Qwen3-VL-32B-Instruct, no verify\n"]
md.append("19 truth pairs, success metric = any-overlap (IoU > 0)\n")

hit = none = wrong = 0
for i, r in enumerate(data["results"], 1):
    tm = r["tm"]; pair_i = r["pair_idx"]
    ev_path = Path(r["evidence"])
    truth_bbox = r["truth_bbox"]
    pred_bbox = r["pred_bbox"]
    s = iou(truth_bbox, pred_bbox)
    if pred_bbox is None:
        label = "NONE"; none += 1
    elif s > 0:
        label = "HIT"; hit += 1
    else:
        label = "WRONG"; wrong += 1

    img = Image.open(ev_path).convert("RGB")
    W, H = img.size
    canvas = img.copy()
    d = ImageDraw.Draw(canvas)
    lw = max(3, min(W, H) // 200)
    # truth = green
    if truth_bbox:
        d.rectangle(truth_bbox, outline=(0, 220, 0), width=lw)
    # pred = red (if any)
    if pred_bbox:
        d.rectangle([int(x) for x in pred_bbox], outline=(220, 0, 0), width=lw)
    out_name = f"{tm}_p{pair_i:02d}_{label}.png"
    canvas.save(OUT / out_name)
    md.append(f"## {i}. tm={tm} pair={pair_i}  →  **{label}** (IoU={s:.2f})\n")
    md.append(f"green=truth, red=pred")
    md.append(f"- truth: `{truth_bbox}`")
    md.append(f"- pred:  `{pred_bbox}`")
    md.append(f"- reason: {r.get('reason','')[:160]}")
    md.append(f"\n![{out_name}]({out_name})\n")
    md.append("---\n")
md.append(f"\n## Summary\n- HIT: {hit}/19\n- WRONG: {wrong}/19\n- NONE: {none}/19\n")
(OUT / "README.md").write_text("\n".join(md), encoding="utf-8")
print(f"hit={hit} wrong={wrong} none={none}")
print(f"Wrote {OUT/'README.md'}")
