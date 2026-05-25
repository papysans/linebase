"""Iter 9 bug fix — when iter-7 Pass-1 blank-reject fires, the meta MUST NOT
expose the original (rejected) bbox via the API projection.

Background: iter-7's variance pre-gate (`_PASS1_BLANK_STD_THRESHOLD` /
`_PASS1_BLANK_WHITE_THRESHOLD`) flips `meta[url]["found"]` to False when a
high-confidence Pass-1 bbox lands on a blank/background region. Until iter-9 the
sibling `meta[url]["bbox"]` was left dangling at the original (now-rejected)
coordinates — the frontend overlay would render a phantom box on the blank
region, and downstream evaluators would treat the row as `WRONG` instead of
`NONE`.

The fix lives in two places:
  1. `pipeline_runner._one_evidence` clears `meta["bbox"] = None` when the gate
     trips, so newly-processed rows persist with bbox=None.
  2. `pipeline_runner._row_to_dict` defensively forces `entry["bbox"] = None`
     when `pass1_blank_reject=True`, covering legacy rows from before the
     source-side fix.

This test exercises path (2): we hand-craft a row dict whose JSON already
contains the legacy stale-bbox shape, push it through `_row_to_dict`, and assert
the API projection nulls out `bbox`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from linebase import pipeline_runner as pr
from linebase import store


def _setup_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "blank_reject.db")
    monkeypatch.setattr(store, "_singleton", None)
    store.init_schema(store.DB_PATH)


def test_pass1_blank_reject_nulls_bbox_in_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_db(tmp_path, monkeypatch)

    up = store.insert_upload("fake.xlsx", 0, str(tmp_path / "fake.xlsx"))
    job = store.insert_job(
        upload_id=up.id,
        sheet_name="s",
        logo_column="D",
        evidence_column="K",
        appno_column="B",
        threshold=0.5,
        sample_kind="first_n",
        sample_params={"n": 1},
        model=None,
        verify_loop=0,
        tile_scan=0,
    )
    row = store.insert_job_row(
        job_id=job.id, row_index=1, appno="X1", logo_url="L", evidence_urls=["E"],
    )

    # Legacy shape: pass1_blank_reject is set, but bbox still carries the
    # rejected coordinates (a row written before the source-side null-out).
    legacy_meta = {
        "E": {
            "found": False,
            "bbox": [100, 100, 200, 200],
            "confidence": 0.0,
            "reason": "pass-1 bbox is blank region",
            "pass1_blank_reject": True,
            "pass1_blank_std": 3.5,
            "pass1_blank_white": 0.97,
            "pass1_original_bbox": [100, 100, 200, 200],
            "pass1_original_reason": "I see the trademark in the top-left margin",
        },
    }
    store.update_job_row(
        row.id,
        status="needs_review",
        match_meta_json=json.dumps(legacy_meta),
    )

    refreshed = store.get_job_row(row.id)
    assert refreshed is not None
    d = pr._row_to_dict(refreshed)

    mm = d["match_meta"]
    assert "E" in mm
    entry = mm["E"]
    # The bbox MUST be nulled out by the projection, regardless of what the
    # legacy JSON carried.
    assert entry["bbox"] is None, (
        f"expected bbox=None when pass1_blank_reject=True, got {entry!r}"
    )
    # Provenance fields remain intact — only the live `bbox` is suppressed.
    assert entry["pass1_blank_reject"] is True
    assert entry["pass1_original_bbox"] == [100.0, 100.0, 200.0, 200.0]
