"""Iter 6.4 — tile-scan must filter degenerate bboxes and verify top candidates.

Two regressions covered:

  1. Degenerate-bbox filter: when one tile returns conf=0.95 but w=5/h=5 and
     a sibling tile returns conf=0.85 with a non-degenerate bbox, the
     degenerate one must be filtered out BEFORE verify. Combined with the
     top-N verify pass, the chosen winner must be the verifier-confirmed
     candidate, NOT the higher-confidence degenerate one.

  2. All top-N candidates fail verify → tile-scan returns None. The caller
     must mark the row as needs_review rather than producing a tile-scan
     false-positive crop.

Both tests run synchronously against `pr._tile_scan` directly so we don't
need to spin up the FastAPI orchestration. `match_logo_in_photo` and
`verify_crop` are both patched on the `pipeline_runner` module so the helper
never touches the network.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from linebase import pipeline_runner as pr
from linebase.llm import MatchResult, VerifyAnswer


def _write_photo(path: Path, w: int, h: int) -> None:
    """Write a wxh light-grey photo (no logo, no markings — content does not
    matter since the matcher and verifier are both mocked away)."""
    arr = np.full((h, w, 3), 200, dtype="uint8")
    Image.fromarray(arr, "RGB").save(path)


def _write_logo(path: Path) -> None:
    arr = np.zeros((50, 50, 3), dtype="uint8")
    arr[:, :] = (255, 0, 0)
    Image.fromarray(arr, "RGB").save(path)


def _usage() -> dict[str, int]:
    return {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "reasoning_tokens": 0,
    }


class _FakeSettings:
    """Minimal stand-in for Settings — only `review_model` is read by _tile_scan."""

    review_model = "fake-verify-model"


def test_tile_scan_filters_degenerate_and_picks_verified(tmp_path: Path) -> None:
    """High-confidence but degenerate (w<28) candidate must lose to a lower-
    confidence verifier-confirmed real match on a sibling tile."""
    logo = tmp_path / "logo.png"
    photo = tmp_path / "photo.png"
    _write_logo(logo)
    # 2400x1800 forces tile-scan past the longest-side gate. tw=800, th=600.
    _write_photo(photo, 2400, 1800)

    # The matcher returns different results based on which tile it sees. Tiles
    # are 800x600. We identify tiles by their pixel dimensions (every tile in a
    # 2400x1800 photo is exactly 800x600 here).
    #
    # r0c0 → degenerate (w=5, h=5) at high confidence
    # r2c1 → real, larger bbox at slightly lower confidence
    # everything else → not found
    call_log: list[str] = []

    def fake_match(logo_path, photo_path, **_kw):
        # We can't directly tell which tile is which without inspecting pixel
        # content, but we can use the temp filename as a serialised counter:
        # _tile_scan iterates row-major (r0c0, r0c1, r0c2, r1c0, ...) so the
        # call ORDER tells us which tile we're processing.
        idx = len(call_log)
        call_log.append(str(photo_path))
        # r0c0 == idx 0; r2c1 == idx 7 (3 rows of 3 cols → r2 starts at idx 6).
        if idx == 0:
            return MatchResult(
                found=True,
                bbox=(0, 0, 5, 5),       # degenerate
                confidence=0.95,
                reason="degenerate",
                raw_response="",
                prompt_version="fake",
                model="fake-model",
                usage=_usage(),
            )
        if idx == 7:  # r2c1
            return MatchResult(
                found=True,
                bbox=(50, 50, 250, 200),  # 200x150 — non-degenerate
                confidence=0.85,
                reason="real",
                raw_response="",
                prompt_version="fake",
                model="fake-model",
                usage=_usage(),
            )
        return MatchResult(
            found=False,
            bbox=None,
            confidence=0.0,
            reason="no",
            raw_response="",
            prompt_version="fake",
            model="fake-model",
            usage=_usage(),
        )

    # verify_crop is called per top-N candidate. Since the degenerate bbox is
    # filtered out BEFORE verify, the only top candidate is r2c1's bbox in
    # original-photo coords. The verifier accepts that one.
    verify_calls: list[Path] = []

    def fake_verify(logo_path, crop_path, **_kw):
        verify_calls.append(crop_path)
        return VerifyAnswer(
            contains_full_logo=True,
            fit="tight",
            confidence=0.8,
            reason="ok",
            suggested_bbox=None,
            raw_response="",
            prompt_version="verify-test",
            model="fake-verify-model",
            usage=_usage(),
        )

    with patch.object(pr, "match_logo_in_photo", side_effect=fake_match), \
         patch.object(pr, "verify_crop", side_effect=fake_verify):
        result, prov, _cost = pr._tile_scan(
            logo, photo,
            settings=_FakeSettings(),
            model="fake-model",
            verify_enabled=True,
            verify_model="fake-verify-model",
        )

    assert result is not None, "tile-scan must produce a verified result"
    # Origin must correspond to r2c1: tw=800, th=600 → ox=800, oy=1200.
    assert prov.get("tile_index") == "r2c1", (
        f"expected r2c1 to win, got {prov.get('tile_index')!r}"
    )
    assert prov.get("tile_origin") == [800, 1200]
    assert prov.get("tile_verified_idx") == "r2c1"
    # The degenerate r0c0 candidate must have been filtered before verify, so
    # the verifier should only have been called once (on r2c1).
    assert len(verify_calls) == 1, (
        f"verify should only run on the non-degenerate candidate, got {len(verify_calls)} calls"
    )
    # The bbox must be in ORIGINAL photo coords (tile-relative + origin).
    bx1, by1, bx2, by2 = result.bbox  # type: ignore[misc]
    assert bx1 == 800 + 50 and by1 == 1200 + 50
    assert bx2 == 800 + 250 and by2 == 1200 + 200


def test_tile_scan_returns_none_when_all_verify_fail(tmp_path: Path) -> None:
    """If every top-N candidate fails verify, tile-scan returns None.

    Provenance still carries `tile_scanned=True` and `tile_attempts=<n>` so
    the reviewer can see we tried and how many tiles survived the degenerate
    filter, but no MatchResult is produced.
    """
    logo = tmp_path / "logo.png"
    photo = tmp_path / "photo.png"
    _write_logo(logo)
    _write_photo(photo, 2400, 1800)

    def fake_match(logo_path, photo_path, **_kw):
        # Every tile returns a valid (non-degenerate) found result so the
        # filter passes a non-empty candidate list into verify. Verify will
        # then reject all of them.
        return MatchResult(
            found=True,
            bbox=(40, 40, 200, 200),  # 160x160 — non-degenerate
            confidence=0.9,
            reason="maybe",
            raw_response="",
            prompt_version="fake",
            model="fake-model",
            usage=_usage(),
        )

    def fake_verify_rejects(*_a, **_kw):
        return VerifyAnswer(
            contains_full_logo=False,
            fit="wrong",
            confidence=0.1,
            reason="not the logo",
            suggested_bbox=None,
            raw_response="",
            prompt_version="verify-test",
            model="fake-verify-model",
            usage=_usage(),
        )

    with patch.object(pr, "match_logo_in_photo", side_effect=fake_match), \
         patch.object(pr, "verify_crop", side_effect=fake_verify_rejects):
        result, prov, _cost = pr._tile_scan(
            logo, photo,
            settings=_FakeSettings(),
            model="fake-model",
            verify_enabled=True,
            verify_model="fake-verify-model",
        )

    assert result is None, "tile-scan must NOT return a result when all top-N verify fail"
    assert prov.get("tile_scanned") is True
    # All 9 tiles produced valid candidates → tile_attempts == 9.
    assert prov.get("tile_attempts") == 9
    # `tile_verified_idx` must be absent when no tile won verify.
    assert "tile_verified_idx" not in prov


def test_edge_recall_verify_soft_accepts_design_surface_shape_reject(
    tmp_path: Path,
) -> None:
    """Edge recall on design-surface rows can pass when verifier only objects to carrier shape."""
    logo = tmp_path / "logo.png"
    photo = tmp_path / "photo.png"
    _write_logo(logo)
    rng = np.random.default_rng(seed=74241482)
    arr = rng.integers(20, 200, size=(500, 500, 3), dtype="uint8")
    Image.fromarray(arr, "RGB").save(photo)

    primary = MatchResult(
        found=True,
        bbox=(90, 100, 360, 330),
        confidence=0.6,
        reason="edge_recall: distance=0.049 over 4 candidates",
        raw_response="",
        prompt_version="design_1",
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
    )

    def fake_verify_rejects(*_a, **_kw):
        return VerifyAnswer(
            contains_full_logo=False,
            fit="wrong",
            confidence=0.0,
            reason=(
                "Image 1 shows a geometric grid pattern, while Image 2 shows "
                "a quilted handbag with handles; no visual shape correspondence exists."
            ),
            suggested_bbox=None,
            raw_response="",
            prompt_version="verify-design_1",
            model="fake-verify-model",
            usage=_usage(),
        )

    with patch.object(pr, "verify_crop", side_effect=fake_verify_rejects):
        result = pr._verify_recalled_bbox(
            logo,
            photo,
            primary,
            settings=_FakeSettings(),
            model="fake-model",
        )

    assert result.verified is True
    assert result.soft_verified is True
    assert result.final_bbox == primary.bbox
    assert result.fit_label == "loose"
