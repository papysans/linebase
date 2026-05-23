"""Detect the 'red-box-annotated' evidence image for each docx sample.

The user hand-marks the best-match evidence with a red rectangle border.
We scan each evidence image for high-saturation red pixels arranged in a roughly
rectangular border pattern, and emit a JSON file with the inferred "gold" evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

FIXTURES = Path("D:/Project/linebase/.trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline/fixtures")


def red_score(path: Path) -> tuple[float, float]:
    """Return (red_pixel_ratio, border_red_ratio).
    Pixels qualify as red when R > 200 and G < 80 and B < 80 (bright pure red, like hand-drawn boxes).
    border_red_ratio = ratio in the outer 10% frame ring (where annotation rectangles tend to sit).
    """
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    red = (r > 200) & (g < 80) & (b < 80)
    full = red.mean()
    # outer 10% ring
    margin_h = max(1, h // 10)
    margin_w = max(1, w // 10)
    ring = np.zeros_like(red)
    ring[:margin_h, :] = True
    ring[-margin_h:, :] = True
    ring[:, :margin_w] = True
    ring[:, -margin_w:] = True
    border = red[ring].mean() if ring.any() else 0.0
    return float(full), float(border)


def main() -> int:
    out = {}
    for sample_dir in sorted(FIXTURES.glob("sample_*")):
        sample = sample_dir.name.removeprefix("sample_")
        pngs = sorted(sample_dir.glob("*.png"))
        # exclude logo (first) and expected (last) — they shouldn't have red boxes
        evidences = pngs[1:-1]
        scored = []
        for p in evidences:
            full, border = red_score(p)
            scored.append({"file": p.name, "red_ratio": round(full, 5), "border_red_ratio": round(border, 5)})
        # pick best by border_red_ratio (most red on outer ring)
        scored.sort(key=lambda x: x["border_red_ratio"], reverse=True)
        out[sample] = {
            "all": scored,
            "best_guess": scored[0]["file"] if scored else None,
            "best_guess_border_ratio": scored[0]["border_red_ratio"] if scored else 0.0,
        }
        print(f"sample {sample}: best={scored[0]['file']} border_red={scored[0]['border_red_ratio']:.4f}  (top3: {[(s['file'], s['border_red_ratio']) for s in scored[:3]]})")
    dest = Path("D:/Project/linebase/eval/redbox_gold.json")
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
