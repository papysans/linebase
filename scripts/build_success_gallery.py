"""Build a Markdown gallery of all successful bbox matches across past runs.

Output structure:
  docs/successful_crops/
    README.md                  index, one entry per successful row
    assets/<appno>_logo.<ext>  the line-art logo
    assets/<appno>_orig.<ext>  the original photo
    assets/<appno>_bbox.png    original with bbox overlay
    assets/<appno>_crop.png    the resulting crop (copied)

A row is "successful" when:
  job_row.status == 'ok'
  best_crop_path is not None and file exists
  meta[chosen_url].found is True
  meta[chosen_url].verified is not False (None means verify wasn't run)
  not meta[chosen_url].sanity_rejected
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / ".data" / "linebase.db"
CACHE = ROOT / ".cache" / "images"
OUT = ROOT / "docs" / "successful_crops"
ASSETS = OUT / "assets"


def url_to_cache(url: str) -> Path | None:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    hits = list(CACHE.glob(f"{h}.*"))
    return hits[0] if hits else None


def render_bbox(src: Path, bbox: list[int], dst: Path) -> bool:
    try:
        img = Image.open(src).convert("RGB")
    except Exception as e:
        print(f"  [skip bbox] cannot open {src}: {e}")
        return False
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = img.size
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
    line_w = max(3, min(w, h) // 200)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=line_w)
    label = f"bbox {x1},{y1},{x2},{y2}"
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, min(w, h) // 60))
    except Exception:
        font = ImageFont.load_default()
    tx, ty = x1 + 4, max(0, y1 - 22)
    draw.rectangle([tx - 2, ty - 2, tx + len(label) * 8, ty + 18], fill=(255, 60, 60))
    draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    img.save(dst, format="PNG")
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """
        SELECT jr.id, jr.job_id, jr.appno, jr.logo_url, jr.best_crop_path,
               jr.all_crops_json, jr.match_meta_json, j.model, j.prompt_version, j.verify_loop
          FROM job_row jr
          JOIN job j ON j.id = jr.job_id
         WHERE jr.status = 'ok' AND jr.best_crop_path IS NOT NULL
         ORDER BY jr.appno, jr.job_id
        """
    ).fetchall()

    # Pick best row per appno: prefer verify-loop=1, then highest confidence
    by_appno: dict[str, list[tuple]] = {}
    for r in rows:
        meta = json.loads(r[6] or "{}")
        crops = json.loads(r[5] or "{}")
        chosen = next((u for u, p in crops.items() if p == r[4]), None)
        if not chosen:
            continue
        info = meta.get(chosen, {})
        if not info.get("found"):
            continue
        if info.get("verified") is False:
            continue
        if info.get("sanity_rejected"):
            continue
        bbox = info.get("bbox")
        conf = info.get("confidence", 0.0)
        if not bbox or len(bbox) != 4:
            continue
        by_appno.setdefault(r[2], []).append((r, chosen, info, bbox, conf))

    picked: list[tuple] = []
    for appno, candidates in by_appno.items():
        candidates.sort(key=lambda x: (1 if x[0][9] else 0, x[4]), reverse=True)
        picked.append(candidates[0])

    print(f"Total successful unique appnos: {len(picked)}")

    md = ["# 已成功识别的 Bbox 样本画廊\n"]
    md.append(f"统计自 `.data/linebase.db`，共 **{len(picked)}** 行被 pipeline 标记为 `ok` 且通过 `verify` + `sanity` 双检。\n")
    md.append("每行三联图：**线稿 Logo** / **原图 + bbox 红框** / **裁剪结果**。\n")
    md.append("---\n")

    for r, chosen, info, bbox, conf in picked:
        row_id, job_id, appno, logo_url, best_crop, _, _, model, prompt_v, verify = r
        print(f"[{appno}] job={job_id[:8]} model={model} verify={'on' if verify else 'off'} conf={conf}")

        logo_cache = url_to_cache(logo_url) if logo_url else None
        orig_cache = url_to_cache(chosen)

        logo_dst = ASSETS / f"{appno}_logo{logo_cache.suffix if logo_cache else '.png'}"
        orig_dst = ASSETS / f"{appno}_orig{orig_cache.suffix if orig_cache else '.png'}"
        bbox_dst = ASSETS / f"{appno}_bbox.png"
        crop_dst = ASSETS / f"{appno}_crop{Path(best_crop).suffix}"

        if logo_cache and logo_cache.exists():
            shutil.copy(logo_cache, logo_dst)
        else:
            logo_dst = None
        if orig_cache and orig_cache.exists():
            shutil.copy(orig_cache, orig_dst)
            render_bbox(orig_cache, bbox, bbox_dst)
        else:
            orig_dst = None; bbox_dst = None
        crop_src = Path(best_crop)
        if crop_src.exists():
            shutil.copy(crop_src, crop_dst)
        else:
            crop_dst = None

        md.append(f"## 申请号 `{appno}`\n")
        md.append(f"- **Job**: `{job_id}`")
        md.append(f"- **Model**: `{model}`  (verify-loop: {'ON' if verify else 'off'}, prompt v_{prompt_v or '?'})")
        md.append(f"- **Confidence**: `{conf}`")
        md.append(f"- **bbox**: `{bbox}`")
        md.append(f"- **Reason**: {info.get('reason', '').strip()}")
        clarity = info.get("clarity"); completeness = info.get("completeness"); isolation = info.get("isolation")
        if clarity is not None:
            md.append(f"- **Quality**: clarity={clarity}, completeness={completeness}, isolation={isolation}")
        verify_reason = info.get("verify_reason")
        if verify_reason:
            md.append(f"- **Verify-pass reason**: {verify_reason}")
        md.append("")

        cells = []
        if logo_dst: cells.append(f"![logo](assets/{logo_dst.name})")
        else:        cells.append("_(logo 未缓存)_")
        if bbox_dst: cells.append(f"![bbox](assets/{bbox_dst.name})")
        elif orig_dst: cells.append(f"![orig](assets/{orig_dst.name})")
        else:        cells.append("_(原图未缓存)_")
        if crop_dst: cells.append(f"![crop](assets/{crop_dst.name})")
        else:        cells.append("_(裁剪丢失)_")
        md.append("| 线稿 Logo | 原图 + bbox | 裁剪结果 |")
        md.append("| --- | --- | --- |")
        md.append("| " + " | ".join(cells) + " |\n")
        md.append("---\n")

    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT / 'README.md'}")
    print(f"Assets in {ASSETS}")


if __name__ == "__main__":
    main()
