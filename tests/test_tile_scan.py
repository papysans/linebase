"""Iter 6.3 — 3x3 tile-scan fallback for small logos in busy photos.

We stub `match_logo_in_photo` so the helper never touches the network. The
fake matcher inspects each tile's pixel content to decide whether the red
square is in this tile — that's the contract `_tile_scan` relies on:
"the model returns found=True only when the tile actually contains the logo".

Two tests:
  1. Large 2400x1800 photo → tile-scan must fire, return a global bbox that
     overlaps the red square's true position in the original image.
  2. Small 1000x800 photo (longest side < 1500) → tile-scan must NOT fire;
     `_tile_scan` returns None and incurs zero LLM cost.

Plus a runner-level test that confirms `match_meta[url]["tile_scanned"]`
surfaces in the projected row dict via `_row_to_dict`.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from linebase import pipeline_runner as pr
from linebase import store
from linebase.llm import MatchResult


# Where in the 2400x1800 fixture we'll plant the red square.
_RED_X = 1200
_RED_Y = 1500
_RED_W = 50
_RED_H = 50


def _write_logo(path: Path) -> None:
    """A 50x50 solid-red 'logo' image."""
    arr = np.zeros((50, 50, 3), dtype="uint8")
    arr[:, :] = (255, 0, 0)
    Image.fromarray(arr, "RGB").save(path)


def _write_busy_photo(path: Path, w: int, h: int, *, with_red: bool) -> None:
    """A wxh image with light grey fill + (optionally) a 50x50 red square."""
    arr = np.full((h, w, 3), 200, dtype="uint8")
    if with_red:
        arr[_RED_Y : _RED_Y + _RED_H, _RED_X : _RED_X + _RED_W] = (255, 0, 0)
    Image.fromarray(arr, "RGB").save(path)


def _fake_match_returns_red(logo_path, photo_path, **_kw):
    """Mimic a real matcher: open the photo and check for a red region.

    Returns `found=True` with a bbox around the red region when the tile (or
    full photo) contains red pixels, otherwise `found=False`.
    """
    del logo_path  # unused — we only look at the photo
    with Image.open(photo_path) as img:
        arr = np.asarray(img.convert("RGB"))
    red_mask = (arr[..., 0] > 200) & (arr[..., 1] < 80) & (arr[..., 2] < 80)
    ys, xs = np.where(red_mask)
    usage = {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "reasoning_tokens": 0,
    }
    if xs.size == 0:
        return MatchResult(
            found=False,
            bbox=None,
            confidence=0.0,
            reason="no red",
            raw_response="",
            prompt_version="fake",
            model="fake-model",
            usage=usage,
        )
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    return MatchResult(
        found=True,
        bbox=bbox,
        confidence=0.95,
        reason="red detected",
        raw_response="",
        prompt_version="fake",
        model="fake-model",
        usage=usage,
    )


def test_tile_scan_finds_red_square_in_large_photo(tmp_path: Path) -> None:
    """Tile-scan on a 2400x1800 photo must return a global bbox overlapping
    the planted red square's position in original-image coordinates.
    """
    logo = tmp_path / "logo.png"
    photo = tmp_path / "photo.png"
    _write_logo(logo)
    _write_busy_photo(photo, 2400, 1800, with_red=True)

    # Stub Settings so we don't need the env to be configured for the
    # provider-routing helper inside `match_logo_in_photo` (which is mocked
    # away entirely below).
    fake_settings = object()

    with patch.object(pr, "match_logo_in_photo", side_effect=_fake_match_returns_red):
        result, prov, cost = pr._tile_scan(
            logo, photo, settings=fake_settings, model="fake-model",
        )

    assert result is not None, "tile-scan must produce a match"
    assert prov.get("tile_scanned") is True
    assert isinstance(prov.get("tile_origin"), list)
    assert isinstance(prov.get("tile_index"), str)
    # cost should be a number — we don't assert exact value since cost depends
    # on the unknown-model fallback rate in MODEL_PRICING.
    assert cost >= 0.0

    # The returned bbox is in ORIGINAL photo coords. It must overlap the red
    # square's true position (_RED_X, _RED_Y, _RED_X+_RED_W, _RED_Y+_RED_H).
    bx1, by1, bx2, by2 = result.bbox  # type: ignore[misc]
    rx1, ry1 = _RED_X, _RED_Y
    rx2, ry2 = _RED_X + _RED_W, _RED_Y + _RED_H
    # IoU > 0 → bboxes overlap.
    ix1 = max(bx1, rx1)
    iy1 = max(by1, ry1)
    ix2 = min(bx2, rx2)
    iy2 = min(by2, ry2)
    assert ix2 > ix1 and iy2 > iy1, (
        f"tile-scan bbox {result.bbox} does not overlap red region "
        f"({rx1},{ry1},{rx2},{ry2})"
    )

    # tile_origin should fall on a 3x3 tile boundary of the 2400x1800 photo.
    # tw=800, th=600 → origins are multiples of (800, 600) less than (2400, 1800).
    ox, oy = prov["tile_origin"]
    assert ox in (0, 800, 1600)
    assert oy in (0, 600, 1200)


def test_tile_scan_skips_small_photo(tmp_path: Path) -> None:
    """Photos with longest side <= 1500 px must NOT trigger tile-scan."""
    logo = tmp_path / "logo.png"
    photo = tmp_path / "small.png"
    _write_logo(logo)
    _write_busy_photo(photo, 1000, 800, with_red=True)

    called = {"n": 0}

    def _should_not_run(*_a, **_kw):
        called["n"] += 1
        raise AssertionError("match_logo_in_photo must not run for sub-1500 photo")

    with patch.object(pr, "match_logo_in_photo", side_effect=_should_not_run):
        result, prov, cost = pr._tile_scan(
            logo, photo, settings=object(), model="fake-model",
        )

    assert result is None
    assert prov == {}
    assert cost == 0.0
    assert called["n"] == 0


def test_tile_scan_no_red_returns_none(tmp_path: Path) -> None:
    """When no tile contains the logo, `_tile_scan` returns None.

    This is the "tile-scan also fails" path — caller must handle gracefully.

    Iter 6.4 contract change: provenance now carries `tile_scanned=True` +
    `tile_attempts=0` even on a miss so the reviewer can see "we tried tile-
    scan; nothing survived the degenerate + confidence filters".
    """
    logo = tmp_path / "logo.png"
    photo = tmp_path / "no_red.png"
    _write_logo(logo)
    _write_busy_photo(photo, 2400, 1800, with_red=False)

    with patch.object(pr, "match_logo_in_photo", side_effect=_fake_match_returns_red):
        result, prov, cost = pr._tile_scan(
            logo, photo, settings=object(), model="fake-model",
        )

    assert result is None
    # Provenance must surface that tile-scan ran but yielded no candidates.
    assert prov.get("tile_scanned") is True
    assert prov.get("tile_attempts") == 0
    # cost is the sum of 9 per-tile fake calls; should be a non-negative number.
    assert cost >= 0.0


def test_row_dict_surfaces_tile_scanned(tmp_path: Path, monkeypatch) -> None:
    """`_row_to_dict` projects `tile_scanned` / `tile_origin` / `tile_index`
    into `match_meta[url]` when the runner recorded a tile-scan win.
    """
    # Point store.DATA_DIR at a temp dir so tests don't pollute the real db.
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "tile_scan.db")
    monkeypatch.setattr(store, "_singleton", None)
    store.init_schema()

    # Insert a minimal upload + job + row to exercise _row_to_dict.
    up = store.insert_upload("fake.xlsx", 0, str(tmp_path / "fake.xlsx"))
    job = store.insert_job(
        upload_id=up.id, sheet_name="s", logo_column="D", evidence_column="K",
        appno_column="B", threshold=0.5,
        sample_kind="first_n", sample_params={"n": 1},
        model=None, verify_loop=0, tile_scan=1,
    )
    assert job.tile_scan == 1, "tile_scan column must persist on insert"

    row = store.insert_job_row(
        job_id=job.id, row_index=3, appno="X1", logo_url="L", evidence_urls=["E"],
    )
    meta = {
        "E": {
            "found": True,
            "bbox": [826, 1834, 1112, 1974],
            "confidence": 0.82,
            "reason": "[tile-scan r2c1 @ 612,1584] patch",
            "tile_scanned": True,
            "tile_origin": [612, 1584],
            "tile_index": "r2c1",
        },
    }
    store.update_job_row(
        row.id,
        status="ok",
        match_meta_json=json.dumps(meta),
        all_crops_json=json.dumps({"E": str(tmp_path / "crop.png")}),
        best_crop_path=str(tmp_path / "crop.png"),
    )
    refreshed = store.get_job_row(row.id)
    assert refreshed is not None
    d = pr._row_to_dict(refreshed)
    mm = d.get("match_meta", {})
    assert "E" in mm
    entry = mm["E"]
    assert entry.get("tile_scanned") is True
    assert entry.get("tile_origin") == [612, 1584]
    assert entry.get("tile_index") == "r2c1"
