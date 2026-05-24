"""Iter 7 — Pass-1 variance pre-gate must drop blank-region predictions.

The gate sits in `_one_evidence` BEFORE the tile-scan trigger and BEFORE the
acceptance gate. It runs whether `verify_loop` is ON or OFF, since the empirical
motivation is "Qwen3-VL-32B occasionally hallucinates a high-confidence bbox
on the empty top-left margin of the photo". Truth-set analysis (19 pairs, see
the iter-7 spec) shows the threshold `std<10 OR white>0.85` correctly rejects
6 of 12 wrong predictions while keeping all 6 real hits.

Two regressions covered:

  1. Gate trips on a bbox over empty white margin → `meta[url]` is rewritten
     with `pass1_blank_reject=True`, `found=False`, no crop produced.

  2. Gate does NOT trip on a bbox over the planted red square → `found=True`
     survives, crop is produced, no `pass1_blank_reject` key.

We patch `pipeline_runner.match_logo_in_photo` and `pipeline_runner.fetch` so
the test never touches the network. The DB layer is monkeypatched to use a
temp dir so `_process_row` can persist row state through its normal code path.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from linebase import pipeline_runner as pr
from linebase import store
from linebase.llm import MatchResult


def _usage() -> dict[str, int]:
    return {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "reasoning_tokens": 0,
    }


# Where in the 1000x800 fixture we plant the red square. Position chosen
# so a bbox over the top-left corner (40,40,180,140) misses it entirely
# AND lands on white pixels (gate must trip), while a bbox over the red
# square itself (490,395,590,485) contains heavy red content (gate must not
# trip).
_RED_X = 500
_RED_Y = 400
_RED_W = 100
_RED_H = 80


def _write_logo(path: Path) -> None:
    """A 50x50 solid-red 'logo' image — content is irrelevant since
    `match_logo_in_photo` is mocked away."""
    arr = np.zeros((50, 50, 3), dtype="uint8")
    arr[:, :] = (255, 0, 0)
    Image.fromarray(arr, "RGB").save(path)


def _write_planted_photo(path: Path) -> None:
    """A 1000x800 near-white image with a 100x80 red square at (500, 400)."""
    arr = np.full((800, 1000, 3), 255, dtype="uint8")
    arr[_RED_Y : _RED_Y + _RED_H, _RED_X : _RED_X + _RED_W] = (255, 0, 0)
    Image.fromarray(arr, "RGB").save(path)


class _FakeSettings:
    """Minimal Settings stub — only `model` / `review_model` are read on the
    code paths we exercise. `resolve_provider` is consulted by the concurrency
    helper; we return a stub with `name='openai'` so the cap calc doesn't blow
    up on an unknown provider."""

    model = "fake-model"
    review_model = "fake-verify-model"

    def resolve_provider(self, model: str):  # noqa: ARG002
        class _PC:
            name = "openai"
        return _PC()


def _setup_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the store at a fresh per-test sqlite file."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "_singleton", None)
    store.init_schema(store.DB_PATH)
    # Also point pipeline_runner.DATA_DIR at the temp dir so the run-dir
    # crop output lands somewhere safe under the test sandbox.
    monkeypatch.setattr(pr, "DATA_DIR", tmp_path)


def _make_job_and_row(
    *,
    logo_url: str,
    evidence_urls: list[str],
) -> tuple[store.Job, store.JobRow]:
    up = store.insert_upload("fake.xlsx", 0, "fake-path")
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
        job_id=job.id,
        row_index=1,
        appno="APP1",
        logo_url=logo_url,
        evidence_urls=evidence_urls,
    )
    return job, row


def test_pass1_blank_gate_rejects_white_margin_bbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A high-confidence Pass-1 bbox covering the top-left white margin must
    trip the gate: `found=False`, `pass1_blank_reject=True`, no crop produced.
    """
    _setup_db(tmp_path, monkeypatch)

    logo = tmp_path / "logo.png"
    photo = tmp_path / "photo.png"
    _write_logo(logo)
    _write_planted_photo(photo)

    # fetch returns the local path verbatim (no network).
    monkeypatch.setattr(pr, "fetch", lambda url: logo if url == "L" else photo)

    # match_logo_in_photo returns a high-confidence bbox over EMPTY white space
    # at the top-left of the photo. The bbox (40,40,180,140) is entirely inside
    # the all-white margin (red square sits at 500,400-600,480).
    def fake_match(_logo_path, _photo_path, **_kw):
        return MatchResult(
            found=True,
            bbox=(40, 40, 180, 140),
            confidence=0.95,
            reason="hallucinated on margin",
            raw_response="",
            prompt_version="fake",
            model="fake-model",
            usage=_usage(),
        )

    job, row = _make_job_and_row(logo_url="L", evidence_urls=["E"])

    run_dir = tmp_path / "runs" / job.id
    (run_dir / "images").mkdir(parents=True, exist_ok=True)

    with patch.object(pr, "match_logo_in_photo", side_effect=fake_match):
        asyncio.run(pr._process_row(job, row, _FakeSettings(), run_dir))

    refreshed = store.get_job_row(row.id)
    assert refreshed is not None
    meta = json.loads(refreshed.match_meta_json)
    entry = meta["E"]
    assert entry["pass1_blank_reject"] is True, (
        f"expected pass1_blank_reject=True, got meta={entry}"
    )
    assert entry["found"] is False
    # Original bbox + reason preserved as provenance.
    assert entry["pass1_original_bbox"] == [40, 40, 180, 140]
    assert "hallucinated" in (entry.get("pass1_original_reason") or "")
    # Statistics surfaced for diagnostics.
    assert "pass1_blank_std" in entry
    assert "pass1_blank_white" in entry
    # Crop dict has no surviving file (gate set found=False → acceptance gate
    # short-circuits to crop=None).
    crops = json.loads(refreshed.all_crops_json)
    assert crops.get("E") is None
    # Row downgraded to needs_review (no viable candidate after the gate).
    assert refreshed.status == "needs_review"
    # And _row_to_dict surfaces the new fields in match_meta.
    d = pr._row_to_dict(refreshed)
    proj = d["match_meta"]["E"]
    assert proj["pass1_blank_reject"] is True
    assert proj["pass1_original_bbox"] == [40.0, 40.0, 180.0, 140.0]


def test_pass1_blank_gate_allows_real_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Pass-1 bbox covering the actual red square must survive the gate:
    no `pass1_blank_reject`, `found=True`, crop produced.
    """
    _setup_db(tmp_path, monkeypatch)

    logo = tmp_path / "logo.png"
    photo = tmp_path / "photo.png"
    _write_logo(logo)
    _write_planted_photo(photo)

    monkeypatch.setattr(pr, "fetch", lambda url: logo if url == "L" else photo)

    # bbox tightly covers the red square (500,400-600,480) with a little
    # padding (490,395-590,485). std on a solid red patch is 0; white_ratio
    # is also 0 — but the gate is `std < 10 OR white > 0.85` which means a
    # pure-red patch (std=0) would still trip the std side of the OR!
    #
    # That matches the spec's intent though: a uniform region — whether white
    # or red — is suspicious. The realistic red-square fixture is wrong here;
    # we need texture. Switch to a *textured* red-ish region by varying the
    # green channel slightly so std is well above 10.
    arr = np.full((800, 1000, 3), 255, dtype="uint8")
    rng = np.random.default_rng(seed=0)
    # Replace the red square area with a textured pattern (R=255, G in [0,80],
    # B=0). Std of this patch is ~23 on the green channel, white_ratio = 0
    # since R/G/B aren't all >240. Gate must NOT trip.
    g_noise = rng.integers(0, 80, size=(_RED_H, _RED_W), dtype="uint8")
    arr[_RED_Y : _RED_Y + _RED_H, _RED_X : _RED_X + _RED_W, 0] = 255
    arr[_RED_Y : _RED_Y + _RED_H, _RED_X : _RED_X + _RED_W, 1] = g_noise
    arr[_RED_Y : _RED_Y + _RED_H, _RED_X : _RED_X + _RED_W, 2] = 0
    Image.fromarray(arr, "RGB").save(photo)

    def fake_match(_logo_path, _photo_path, **_kw):
        return MatchResult(
            found=True,
            bbox=(490, 395, 590, 485),
            confidence=0.95,
            reason="real match",
            raw_response="",
            prompt_version="fake",
            model="fake-model",
            usage=_usage(),
        )

    job, row = _make_job_and_row(logo_url="L", evidence_urls=["E"])
    run_dir = tmp_path / "runs" / job.id
    (run_dir / "images").mkdir(parents=True, exist_ok=True)

    with patch.object(pr, "match_logo_in_photo", side_effect=fake_match):
        asyncio.run(pr._process_row(job, row, _FakeSettings(), run_dir))

    refreshed = store.get_job_row(row.id)
    assert refreshed is not None
    meta = json.loads(refreshed.match_meta_json)
    entry = meta["E"]
    assert "pass1_blank_reject" not in entry, (
        f"gate must NOT trip on textured non-white region, got meta={entry}"
    )
    assert entry["found"] is True
    crops = json.loads(refreshed.all_crops_json)
    # Crop must have been produced and accepted (no sanity rejection on a
    # textured red region — well above the brightness floor).
    assert crops.get("E"), f"expected a crop file, got crops={crops}"
    assert refreshed.status == "ok"
