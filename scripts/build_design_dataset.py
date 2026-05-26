"""Extract design-patent image pairs from the WPS XLSX into train/test manifests.

The source workbook stores product photos and patent line drawings as WPS
``DISPIMG("ID_...", 1)`` cell images rather than normal hyperlinks. This script
reads the XLSX package directly, resolves ``xl/cellimages.xml`` relationships,
and writes local image pairs under ``.data/design_dataset``.

Output:
    .data/design_dataset/manifest.jsonl
    .data/design_dataset/train.jsonl
    .data/design_dataset/test.jsonl
    .data/design_dataset/skipped.jsonl
    .data/design_dataset/summary.json
"""
from __future__ import annotations

import argparse
import json
import posixpath
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from PIL import Image, UnidentifiedImageError

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / ".data" / "design_dataset"
DEFAULT_SHEETS = ("美专2357", "tro332")
SHEET_SLUGS = {"美专2357": "meizhuan2357", "tro332": "tro332"}

NS = {
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}

DISPIMG_ID_RE = re.compile(r"ID_[0-9A-Fa-f]+")
CELL_REF_RE = re.compile(r"([A-Z]+)([0-9]+)")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def _find_workbook(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = list(REPO.glob("美专实物图排查-*.xlsx"))
    if not candidates:
        candidates = list(REPO.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError("No .xlsx workbook found in repo root")
    return max(candidates, key=lambda p: p.stat().st_size)


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out: list[str] = []
    for si in root.findall("s:si", NS):
        out.append("".join(t.text or "" for t in si.findall(".//s:t", NS)))
    return out


def _resolve_target(source_path: str, target: str) -> str:
    target = target.replace("\\", "/")
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join(posixpath.dirname(source_path), target))


def _load_relationships(
    zf: zipfile.ZipFile,
    rels_path: str,
    *,
    source_path: str,
) -> dict[str, dict[str, str]]:
    root = ET.fromstring(zf.read(rels_path))
    out: dict[str, dict[str, str]] = {}
    for rel in root.findall("rel:Relationship", NS):
        rel_id = rel.attrib["Id"]
        target = rel.attrib.get("Target", "")
        out[rel_id] = {
            "type": rel.attrib.get("Type", ""),
            "target": _resolve_target(source_path, target),
        }
    return out


def _workbook_sheets(zf: zipfile.ZipFile) -> dict[str, str]:
    rels = _load_relationships(
        zf,
        "xl/_rels/workbook.xml.rels",
        source_path="xl/workbook.xml",
    )
    root = ET.fromstring(zf.read("xl/workbook.xml"))
    sheets: dict[str, str] = {}
    for sheet in root.findall(".//s:sheet", NS):
        name = sheet.attrib["name"]
        rid = sheet.attrib[f"{{{NS['r']}}}id"]
        sheets[name] = rels[rid]["target"]
    return sheets


def _cell_text(cell: ET.Element, shared: list[str]) -> str | None:
    cell_type = cell.attrib.get("t")
    value = cell.find("s:v", NS)
    formula = cell.find("s:f", NS)
    if cell_type == "s" and value is not None and value.text is not None:
        try:
            return shared[int(value.text)]
        except (IndexError, ValueError):
            return value.text.strip()
    if cell_type == "inlineStr":
        text = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
        return text.strip() if text else None
    if value is not None and value.text is not None:
        return value.text.strip()
    if formula is not None and formula.text:
        return formula.text.strip()
    return None


def _cell_col(cell_ref: str) -> str | None:
    match = CELL_REF_RE.match(cell_ref)
    return match.group(1) if match else None


def _dispimg_id(text: str | None) -> str | None:
    if not text:
        return None
    match = DISPIMG_ID_RE.search(text)
    return match.group(0).upper() if match else None


def _load_cell_image_map(zf: zipfile.ZipFile) -> dict[str, str]:
    wb_rels = _load_relationships(
        zf,
        "xl/_rels/workbook.xml.rels",
        source_path="xl/workbook.xml",
    )
    cellimages_path: str | None = None
    for rel in wb_rels.values():
        if "cellImage" in rel["type"] or rel["target"].endswith("cellimages.xml"):
            cellimages_path = rel["target"]
            break
    if not cellimages_path:
        raise RuntimeError("Workbook has no WPS cellimages.xml relationship")

    rels_path = posixpath.join(
        posixpath.dirname(cellimages_path),
        "_rels",
        posixpath.basename(cellimages_path) + ".rels",
    )
    image_rels = _load_relationships(zf, rels_path, source_path=cellimages_path)
    root = ET.fromstring(zf.read(cellimages_path))

    out: dict[str, str] = {}
    for pic in root.findall(".//xdr:pic", NS):
        props = pic.find(".//xdr:cNvPr", NS)
        blip = pic.find(".//a:blip", NS)
        if props is None or blip is None:
            continue
        image_id = props.attrib.get("name")
        rid = blip.attrib.get(f"{{{NS['r']}}}embed")
        if not image_id or not rid or rid not in image_rels:
            continue
        out[image_id.upper()] = image_rels[rid]["target"]
    return out


def _parse_sheet_rows(
    zf: zipfile.ZipFile,
    *,
    sheet_name: str,
    sheet_path: str,
    shared: list[str],
) -> list[dict[str, Any]]:
    root = ET.fromstring(zf.read(sheet_path))
    rows: list[dict[str, Any]] = []
    for xml_row in root.findall(".//s:sheetData/s:row", NS):
        row_index = int(xml_row.attrib.get("r", "0") or "0")
        if row_index <= 1:
            continue

        cells: dict[str, str] = {}
        for cell in xml_row.findall("s:c", NS):
            col = _cell_col(cell.attrib.get("r", ""))
            if not col:
                continue
            value = _cell_text(cell, shared)
            if value is not None:
                cells[col] = value.strip()

        appno = cells.get("A", "").strip()
        if not appno:
            continue
        rows.append(
            {
                "source_sheet": sheet_name,
                "source_row": row_index,
                "appno": appno,
                "owner": cells.get("B", "").strip() or None,
                "product_image_id": _dispimg_id(cells.get("C")),
                "product_url": cells.get("D", "").strip() or None,
                "line_image_id": _dispimg_id(cells.get("E")),
                "grant_date": cells.get("F", "").strip() or None,
                "expires_date": cells.get("G", "").strip() or None,
                "case_no": cells.get("H", "").strip() or None,
                "loc_class": cells.get("J", "").strip() or None,
                "upc_category": cells.get("K", "").strip() or None,
                "upc_class": cells.get("L", "").strip() or None,
            }
        )
    return rows


def _slug(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    return slug[:80] or fallback


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _extract_image(
    zf: zipfile.ZipFile,
    *,
    zip_path: str,
    target: Path,
) -> tuple[str | None, tuple[int, int] | None]:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_bytes(zf.read(zip_path))
        size = _image_size(target)
    except (KeyError, OSError, UnidentifiedImageError) as exc:
        return str(exc), None
    return None, size


def _materialize_records(
    zf: zipfile.ZipFile,
    *,
    rows: list[dict[str, Any]],
    cell_images: dict[str, str],
    out_dir: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ready: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in rows:
        product_id = row.get("product_image_id")
        line_id = row.get("line_image_id")
        if not product_id or not line_id:
            skipped.append({**row, "skip_reason": "missing_dispimg_id"})
            continue
        if product_id not in cell_images or line_id not in cell_images:
            skipped.append({**row, "skip_reason": "missing_cellimage_relationship"})
            continue

        sheet_slug = SHEET_SLUGS.get(
            row["source_sheet"],
            _slug(row["source_sheet"], fallback="sheet"),
        )
        row_slug = f"r{int(row['source_row']):04d}_{_slug(str(row['appno']), fallback='appno')}"
        image_dir = out_dir / "images" / sheet_slug / row_slug

        product_zip = cell_images[product_id]
        line_zip = cell_images[line_id]
        product_target = image_dir / f"product{Path(product_zip).suffix.lower() or '.bin'}"
        line_target = image_dir / f"line{Path(line_zip).suffix.lower() or '.bin'}"

        product_error, product_size = _extract_image(
            zf,
            zip_path=product_zip,
            target=product_target,
        )
        line_error, line_size = _extract_image(zf, zip_path=line_zip, target=line_target)
        if product_error or line_error:
            skipped.append(
                {
                    **row,
                    "skip_reason": "image_extract_or_decode_failed",
                    "product_error": product_error,
                    "line_error": line_error,
                }
            )
            continue

        record_id = f"{sheet_slug}:{row['source_row']}:{row['appno']}"
        ready.append(
            {
                **row,
                "id": record_id,
                "product_path": _repo_rel(product_target),
                "line_path": _repo_rel(line_target),
                "product_size": list(product_size or (0, 0)),
                "line_size": list(line_size or (0, 0)),
                "source_product_media": product_zip,
                "source_line_media": line_zip,
            }
        )
        if limit and len(ready) >= limit:
            break

    return ready, skipped


def _assign_splits(
    records: list[dict[str, Any]],
    *,
    seed: int,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], [], []
    ids = [row["id"] for row in records]
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    test_count = max(1, round(len(ids) * test_ratio))
    test_ids = set(ids[:test_count])

    manifest: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for row in records:
        split = "test" if row["id"] in test_ids else "train"
        item = {**row, "split": split}
        manifest.append(item)
        if split == "test":
            test.append(item)
        else:
            train.append(item)
    return manifest, train, test


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    workbook = _find_workbook(args.workbook)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(workbook) as zf:
        shared = _load_shared_strings(zf)
        sheets = _workbook_sheets(zf)
        cell_images = _load_cell_image_map(zf)

        parsed_rows: list[dict[str, Any]] = []
        for sheet_name in args.sheets:
            if sheet_name not in sheets:
                raise KeyError(f"Sheet not found: {sheet_name!r}")
            parsed_rows.extend(
                _parse_sheet_rows(
                    zf,
                    sheet_name=sheet_name,
                    sheet_path=sheets[sheet_name],
                    shared=shared,
                )
            )

        ready, skipped = _materialize_records(
            zf,
            rows=parsed_rows,
            cell_images=cell_images,
            out_dir=out_dir,
            limit=args.limit,
        )

    manifest, train, test = _assign_splits(ready, seed=args.seed, test_ratio=args.test_ratio)

    _write_jsonl(out_dir / "manifest.jsonl", manifest)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "test.jsonl", test)
    _write_jsonl(out_dir / "skipped.jsonl", skipped)

    summary = {
        "workbook": _repo_rel(workbook),
        "out_dir": _repo_rel(out_dir),
        "sheets": list(args.sheets),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "parsed_rows": len(parsed_rows),
        "ready_rows": len(ready),
        "train_rows": len(train),
        "test_rows": len(test),
        "skipped_rows": len(skipped),
        "skipped_reasons": dict(Counter(row["skip_reason"] for row in skipped)),
        "label_note": "Positive line-drawing/product pairs only; no ground-truth bboxes.",
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=None, help="Source workbook path.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output dataset directory.")
    parser.add_argument("--sheets", nargs="+", default=list(DEFAULT_SHEETS), help="Sheet names.")
    parser.add_argument("--seed", type=int, default=20260206, help="Deterministic split seed.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Held-out test ratio.")
    parser.add_argument("--limit", type=int, default=None, help="Optional ready-row cap for debug.")
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    args = _parse_args()
    summary = build_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
