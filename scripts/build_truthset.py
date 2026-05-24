"""Rebuild docs/truth_set/ from the docx by:
1. Walking doc rows in order; "header rows" with text start a new TM block
   carrying logo + 1st (evidence, expected) pair; continuation rows extend pairs
2. For each (evidence, expected) pair, derive ground-truth bbox by template-
   matching the expected crop inside the evidence image (multi-scale because
   the expected crop may be saved at a different size)

Output structure:
  docs/truth_set/
    INDEX.json
    {TM}_{cat}/
      logo.png
      pair_{i}/
        evidence.png
        expected.png
        truth.json  (bbox + scale + match score)
"""
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "docs" / "_truthset_imgs" / "document.xml"
RELS = ROOT / "docs" / "_truthset_imgs" / "rels.json"
IMG_DIR = ROOT / "docs" / "_truthset_imgs"
OUT = ROOT / "docs" / "truth_set"


def walk_rows(xml: str) -> list[tuple[list[str], str]]:
    rows = re.findall(r"<w:tr[^>]*>(.*?)</w:tr>", xml, re.DOTALL)
    out = []
    for r in rows:
        embeds = re.findall(r'r:embed="(rId\d+)"', r)
        text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", r))
        out.append((embeds, text))
    return out


def parse_tm_blocks(rows: list[tuple[list[str], str]]) -> list[dict]:
    """Each TM block: header row has text 'TM_NUMBER_CAT' + 3 embeds (logo, ev1, exp1).
    Continuation rows have empty text + 2 embeds (ev_i, exp_i). 3-embed continuation
    rows (e.g. row[16]) are anomalies — treat the first as ev, second as exp,
    drop the third (overflow).
    """
    blocks: list[dict] = []
    cur: dict | None = None
    for embeds, text in rows:
        text_clean = text.strip()
        # Header row: text is digits like "242381030" → split TM (7-8 digits) + cat (1-3 digits)
        if text_clean and text_clean.isdigit() and len(embeds) >= 3:
            # TM=7 digits, cat=2 digits (per the docx data)
            m = re.match(r"^(\d{7})(\d{2})$", text_clean)
            if m:
                tm, cat = m.group(1), m.group(2)
            else:
                tm, cat = text_clean[:-2], text_clean[-2:]
            cur = {"tm": tm, "cat": cat, "logo_rid": embeds[0], "pairs": [(embeds[1], embeds[2])]}
            blocks.append(cur)
            # if header has >3 embeds, treat the extras as another pair
            if len(embeds) >= 5:
                cur["pairs"].append((embeds[3], embeds[4]))
        elif cur is not None and not text_clean:
            # Continuation
            if len(embeds) == 2:
                cur["pairs"].append((embeds[0], embeds[1]))
            elif len(embeds) >= 3:
                # 3-embed continuation: take first 2 as (ev, exp), drop the rest
                cur["pairs"].append((embeds[0], embeds[1]))
    return blocks


def template_match_bbox(evidence_path: Path, expected_path: Path) -> dict | None:
    """Multi-scale template match: try 0.25..1.0 scale on the expected (template) image
    inside the evidence image. Returns dict with bbox + score or None.
    Uses pure NumPy normalized cross-correlation."""
    ev = np.asarray(Image.open(evidence_path).convert("L"), dtype=np.float32)
    tpl_full = Image.open(expected_path).convert("L")

    best: dict | None = None
    EV_H, EV_W = ev.shape
    for scale in (1.0, 0.75, 0.6, 0.5, 0.4, 0.33, 0.25, 0.18, 0.14, 0.1):
        nw = int(tpl_full.width * scale)
        nh = int(tpl_full.height * scale)
        if nw < 16 or nh < 16:
            continue
        if nw >= EV_W or nh >= EV_H:
            continue
        tpl = np.asarray(tpl_full.resize((nw, nh), Image.LANCZOS), dtype=np.float32)
        # normalize template
        tpl_mean = tpl.mean()
        tpl_dev = tpl - tpl_mean
        tpl_norm = float(np.sqrt((tpl_dev ** 2).sum()))
        if tpl_norm < 1e-6:
            continue
        # Sliding window — slow but only on small evidence images. We sample with
        # stride to keep this fast.
        stride = max(1, min(nw, nh) // 8)
        score_best = -2.0
        loc_best = (0, 0)
        for y in range(0, EV_H - nh, stride):
            for x in range(0, EV_W - nw, stride):
                win = ev[y:y + nh, x:x + nw]
                w_dev = win - win.mean()
                w_norm = float(np.sqrt((w_dev ** 2).sum()))
                if w_norm < 1e-6:
                    continue
                s = float((tpl_dev * w_dev).sum() / (tpl_norm * w_norm))
                if s > score_best:
                    score_best = s
                    loc_best = (x, y)
        # Refine around loc_best at stride=1 in a small window
        sx, sy = loc_best
        x_lo, x_hi = max(0, sx - stride), min(EV_W - nw, sx + stride)
        y_lo, y_hi = max(0, sy - stride), min(EV_H - nh, sy + stride)
        for y in range(y_lo, y_hi + 1):
            for x in range(x_lo, x_hi + 1):
                win = ev[y:y + nh, x:x + nw]
                w_dev = win - win.mean()
                w_norm = float(np.sqrt((w_dev ** 2).sum()))
                if w_norm < 1e-6:
                    continue
                s = float((tpl_dev * w_dev).sum() / (tpl_norm * w_norm))
                if s > score_best:
                    score_best = s
                    loc_best = (x, y)
        if best is None or score_best > best["score"]:
            best = {
                "scale": scale,
                "bbox": [loc_best[0], loc_best[1], loc_best[0] + nw, loc_best[1] + nh],
                "score": score_best,
                "tpl_size": [nw, nh],
                "evidence_size": [EV_W, EV_H],
            }
    return best


def rid_to_img_path(rels: dict[str, str], rid: str) -> Path | None:
    target = rels.get(rid)
    if not target:
        return None
    name = target.split("/")[-1]
    p = IMG_DIR / name
    return p if p.exists() else None


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    xml = XML.read_text(encoding="utf-8")
    rels = json.loads(RELS.read_text(encoding="utf-8"))

    rows = walk_rows(xml)
    blocks = parse_tm_blocks(rows)
    print(f"parsed {len(blocks)} TM blocks")

    index: list[dict] = []
    for block in blocks:
        tm = block["tm"]; cat = block["cat"]
        target = OUT / f"{tm}_{cat}"
        target.mkdir(exist_ok=True)
        logo_src = rid_to_img_path(rels, block["logo_rid"])
        if not logo_src:
            print(f"  [{tm}] skip: no logo file")
            continue
        logo_dst = target / f"logo{logo_src.suffix}"
        logo_dst.write_bytes(logo_src.read_bytes())

        block_index = {"tm": tm, "cat": cat, "logo": str(logo_dst.relative_to(ROOT)), "pairs": []}
        print(f"  [{tm} cat={cat}] {len(block['pairs'])} (ev,exp) pairs")
        for i, (ev_rid, exp_rid) in enumerate(block["pairs"], 1):
            ev_src = rid_to_img_path(rels, ev_rid)
            exp_src = rid_to_img_path(rels, exp_rid)
            if not (ev_src and exp_src):
                continue
            pair_dir = target / f"pair_{i:02d}"
            pair_dir.mkdir(exist_ok=True)
            ev_dst = pair_dir / f"evidence{ev_src.suffix}"
            exp_dst = pair_dir / f"expected{exp_src.suffix}"
            ev_dst.write_bytes(ev_src.read_bytes())
            exp_dst.write_bytes(exp_src.read_bytes())

            try:
                tm_match = template_match_bbox(ev_dst, exp_dst)
            except Exception as e:
                print(f"    pair {i}: template match error: {e}")
                tm_match = None

            truth = {
                "tm": tm, "category": cat, "pair_index": i,
                "evidence": ev_dst.name, "expected": exp_dst.name,
                "truth_bbox": tm_match["bbox"] if tm_match else None,
                "template_match_score": tm_match["score"] if tm_match else None,
                "best_scale": tm_match["scale"] if tm_match else None,
                "evidence_size": tm_match["evidence_size"] if tm_match else None,
            }
            (pair_dir / "truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
            block_index["pairs"].append({
                "i": i,
                "evidence": str(ev_dst.relative_to(ROOT)),
                "expected": str(exp_dst.relative_to(ROOT)),
                "truth_bbox": truth["truth_bbox"],
                "score": truth["template_match_score"],
            })
            score_str = f"{truth['template_match_score']:.3f}" if truth['template_match_score'] is not None else "N/A"
            print(f"    pair {i:02d}: bbox={truth['truth_bbox']} score={score_str}")

        index.append(block_index)

    (OUT / "INDEX.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote INDEX.json: {len(index)} TM blocks → {OUT}")


if __name__ == "__main__":
    main()
