"""Bbox crop with coordinate clamping. PIL is enough — no OpenCV needed for v1."""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def crop_to_bbox(
    photo_path: Path,
    bbox: tuple[int, int, int, int],
    out_path: Path,
    pad_ratio: float = 0.0,
) -> Path:
    """Crop photo to bbox, clamping to image bounds. Saves to out_path (parent must exist)."""
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = bbox
    if pad_ratio > 0:
        pw = int((x2 - x1) * pad_ratio)
        ph = int((y2 - y1) * pad_ratio)
        x1 -= pw
        y1 -= ph
        x2 += pw
        y2 += ph
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    cropped = img.crop((x1, y1, x2, y2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out_path)
    return out_path
