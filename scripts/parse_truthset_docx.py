"""Parse the truth-set docx → emit docs/truth_set/{tm_number}/ with logo, evidence (incl red box),
expected_crop, plus a truth_bbox.json derived from the red-box pixels in the evidence image.

Document structure (header row dropped):
| 商标号 | 类别 | LOGO图 | 使用证据图（红框中图是与LOGO最匹配图） | 期望匹配效果图 |
- col 3 (LOGO): exactly 1 image
- col 4 (evidence with red box): 1+ images — pick the FIRST that contains a red rectangle (the marked one)
- col 5 (expected crop): exactly 1 image

10 data rows for trademarks: 2423810, 1969989, 4338293, 4827580, 6433801, 4334451, 4601531, 3089225, 1494172, 1044246
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOC_XML = ROOT / "docs" / "_truthset_imgs" / "document.xml"
RELS_JSON = ROOT / "docs" / "_truthset_imgs" / "rels.json"
IMG_DIR = ROOT / "docs" / "_truthset_imgs"
OUT = ROOT / "docs" / "truth_set"
OUT.mkdir(parents=True, exist_ok=True)


def parse_tables_from_xml(xml: str) -> list[list[list[list[str]]]]:
    """Return tables → rows → cells → list of rId strings (in doc order inside cell)."""
    # crude but works for the docx structure here
    # find each <w:tbl>...</w:tbl>
    tables_raw = re.findall(r"<w:tbl>(.*?)</w:tbl>", xml, re.DOTALL)
    tables: list[list[list[list[str]]]] = []
    for tb in tables_raw:
        rows_raw = re.findall(r"<w:tr[^>]*>(.*?)</w:tr>", tb, re.DOTALL)
        rows: list[list[list[str]]] = []
        for rw in rows_raw:
            cells_raw = re.findall(r"<w:tc[^>]*>(.*?)</w:tc>", rw, re.DOTALL)
            cells: list[list[str]] = []
            for c in cells_raw:
                ids = re.findall(r'r:embed="(rId\d+)"', c)
                # also collect plain text — strip <w:t...>X</w:t>
                txt = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", c))
                cells.append(ids if ids else [txt])
            rows.append(cells)
        tables.append(rows)
    return tables


def detect_red_box_bbox(img_path: Path) -> tuple[int, int, int, int] | None:
    """Find the bounding box of red-stroke pixels in the image. Returns (x1,y1,x2,y2) or None."""
    arr = np.asarray(Image.open(img_path).convert("RGB"))
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    # red-ish: high R, low G & B
    red_mask = (R > 180) & (G < 90) & (B < 90)
    if red_mask.sum() < 80:
        return None
    ys, xs = np.where(red_mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main() -> None:
    xml = DOC_XML.read_text(encoding="utf-8")
    rels = json.loads(RELS_JSON.read_text(encoding="utf-8"))

    def rid_to_img(rid: str) -> Path | None:
        target = rels.get(rid)
        if not target:
            return None
        name = target.split("/")[-1]
        p = IMG_DIR / name
        return p if p.exists() else None

    tables = parse_tables_from_xml(xml)
    print(f"tables found: {len(tables)}")
    # the truth table is the big one; pick the table with the most rows that has image cells
    table = max(tables, key=lambda t: sum(1 for r in t if any(c and isinstance(c[0], str) and c[0].startswith("rId") for c in r)))
    print(f"chose table with {len(table)} rows")

    out_idx: list[dict] = []
    for ri, row in enumerate(table):
        if not row or len(row) < 5:
            continue
        # column 0: 商标号 text, column 1: 类别 text, column 2: logo, column 3: evidence (1+), column 4: expected
        c_tm = "".join(row[0]) if row[0] and not row[0][0].startswith("rId") else ""
        c_cat = "".join(row[1]) if row[1] and not row[1][0].startswith("rId") else ""
        tm = c_tm.strip()
        if not tm.isdigit():
            print(f"  row {ri}: skip (header/non-data: tm={tm!r})")
            continue

        logo_ids = [x for x in row[2] if x.startswith("rId")]
        evidence_ids = [x for x in row[3] if x.startswith("rId")]
        expected_ids = [x for x in row[4] if x.startswith("rId")]
        print(f"  row {ri}: tm={tm} cat={c_cat.strip()!r} logo={len(logo_ids)} ev={len(evidence_ids)} expected={len(expected_ids)}")

        if not logo_ids or not evidence_ids or not expected_ids:
            print(f"     skip: missing one of (logo, evidence, expected)")
            continue

        target = OUT / tm
        target.mkdir(parents=True, exist_ok=True)

        # 1 logo
        logo_path_src = rid_to_img(logo_ids[0])
        if not logo_path_src:
            print(f"     skip: logo file missing")
            continue
        logo_dst = target / f"logo{logo_path_src.suffix}"
        logo_dst.write_bytes(logo_path_src.read_bytes())

        # All evidence images: save them; identify which one has the red box
        truth_bbox = None
        evidence_used = None
        for i, rid in enumerate(evidence_ids):
            src = rid_to_img(rid)
            if not src:
                continue
            dst = target / f"evidence_{i+1}{src.suffix}"
            dst.write_bytes(src.read_bytes())
            bbox = detect_red_box_bbox(dst)
            if bbox and truth_bbox is None:
                truth_bbox = bbox
                evidence_used = dst.name

        # expected crop
        exp_src = rid_to_img(expected_ids[0])
        if exp_src:
            exp_dst = target / f"expected_crop{exp_src.suffix}"
            exp_dst.write_bytes(exp_src.read_bytes())

        truth_json = {
            "trademark_number": tm,
            "category": c_cat.strip(),
            "logo": logo_dst.name,
            "evidence_with_red_box": evidence_used,
            "expected_crop": exp_dst.name if exp_src else None,
            "truth_bbox_in_evidence": list(truth_bbox) if truth_bbox else None,
        }
        (target / "truth.json").write_text(json.dumps(truth_json, indent=2, ensure_ascii=False), encoding="utf-8")
        out_idx.append(truth_json)
        bb = truth_bbox or ("?",)
        print(f"     → wrote {target.name}/  truth_bbox={bb}")

    (OUT / "INDEX.json").write_text(json.dumps(out_idx, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(out_idx)} truth rows → {OUT}")


if __name__ == "__main__":
    main()
