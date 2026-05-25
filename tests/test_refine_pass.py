"""Iter 9 — refine pass.

When the verify call accepts a `fit=loose` bbox, `match_with_verify` re-asks
the SAME primary model on a +30%-padded zoom-crop. The refined bbox is
translated from crop coords to original coords and re-verified; only if the
re-verify accepts does `final_bbox` shift to the refined one.

Two coverage cases:
  1. Happy path: refine returns a tight bbox AND re-verify accepts → final_bbox
     moves to the refined coords AND `refined=True`.
  2. Negative path: refine returns `found=false` → fall back to the original B0
     and leave `refined=False`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from linebase import verify_loop as vl
from linebase.llm import MatchResult, VerifyAnswer


def _write_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype("uint8"), "RGB").save(path)


@pytest.fixture()
def noisy_photo(tmp_path: Path) -> Path:
    """A 1000x800 textured photo — std on every region is well above the
    pre-gate threshold, so the inner verify_round won't short-circuit."""
    rng = np.random.default_rng(seed=7)
    arr = rng.integers(0, 256, size=(800, 1000, 3), dtype="uint8")
    p = tmp_path / "photo.png"
    _write_png(arr, p)
    return p


@pytest.fixture()
def logo_img(tmp_path: Path) -> Path:
    arr = np.full((64, 64, 3), 255, dtype="uint8")
    arr[16:48, 16:48] = 0
    p = tmp_path / "logo.png"
    _write_png(arr, p)
    return p


def _usage() -> dict[str, int]:
    return {"prompt_tokens": 1, "completion_tokens": 1,
            "total_tokens": 2, "reasoning_tokens": 0}


def test_refine_pass_tightens_loose_bbox(
    monkeypatch: pytest.MonkeyPatch, noisy_photo: Path, logo_img: Path
) -> None:
    """Happy path: verify=loose → refine returns a tighter bbox in crop coords →
    re-verify accepts → final_bbox = translated refined coords."""
    # Pass-1 returns a loose bbox.
    fake_primary = MatchResult(
        found=True,
        bbox=(50, 50, 200, 200),
        confidence=0.85,
        reason="loose match",
        raw_response="",
        prompt_version="test",
        model="test",
        usage=_usage(),
    )
    monkeypatch.setattr(vl, "match_logo_in_photo", lambda *a, **kw: fake_primary)

    # verify_crop fires twice: first on the +20%-padded crop around (50,50,200,200)
    # → fit=loose, contains_full_logo=True; then on the +20%-padded crop around
    # the translated refined bbox → fit=tight, contains_full_logo=True.
    verify_calls: list[VerifyAnswer] = [
        VerifyAnswer(
            contains_full_logo=True,
            fit="loose",
            confidence=0.7,
            reason="loose around logo",
            suggested_bbox=None,
            raw_response="",
            prompt_version="verify-test",
            model="test",
            usage=_usage(),
        ),
        VerifyAnswer(
            contains_full_logo=True,
            fit="tight",
            confidence=0.9,
            reason="now tight",
            suggested_bbox=None,
            raw_response="",
            prompt_version="verify-test",
            model="test",
            usage=_usage(),
        ),
    ]
    call_idx = {"n": 0}

    def _fake_verify_crop(*_a, **_kw):
        i = call_idx["n"]
        call_idx["n"] += 1
        return verify_calls[i]

    monkeypatch.setattr(vl, "verify_crop", _fake_verify_crop)

    # match_logo_in_zoomed returns a refined bbox in CROP coords (40,40,80,80)
    # with zoom_origin (40,40) → global = (80,80,120,120). Note the helper
    # ordinarily computes the zoom origin from a +30% pad on the primary bbox;
    # here we hard-code both to make the assertion deterministic regardless
    # of the helper's internal cropping.
    refined_match = MatchResult(
        found=True,
        bbox=(40, 40, 80, 80),
        confidence=0.92,
        reason="refined tight",
        raw_response="",
        prompt_version="4_refine",
        model="test",
        usage=_usage(),
    )

    def _fake_refine(*_a, **kw):
        return refined_match, (40, 40)

    monkeypatch.setattr(vl, "match_logo_in_zoomed", _fake_refine)

    result = vl.match_with_verify(
        logo_img,
        noisy_photo,
        max_iters=2,
        verify_threshold=0.6,
        model="test-model",
        refine=True,
    )

    assert result.verified is True
    assert result.refined is True
    assert result.final_bbox == (80, 80, 120, 120), (
        f"expected final_bbox=(80,80,120,120), got {result.final_bbox!r}"
    )
    assert result.refine_bbox == (80, 80, 120, 120)
    assert result.refine_origin == (40, 40)
    # Both verify calls must have fired.
    assert call_idx["n"] == 2


def test_refine_pass_falls_back_when_refine_returns_not_found(
    monkeypatch: pytest.MonkeyPatch, noisy_photo: Path, logo_img: Path
) -> None:
    """Negative path: refine returns found=False → keep B0 as final_bbox, and
    `refined` stays False."""
    fake_primary = MatchResult(
        found=True,
        bbox=(50, 50, 200, 200),
        confidence=0.85,
        reason="loose match",
        raw_response="",
        prompt_version="test",
        model="test",
        usage=_usage(),
    )
    monkeypatch.setattr(vl, "match_logo_in_photo", lambda *a, **kw: fake_primary)

    monkeypatch.setattr(
        vl, "verify_crop",
        lambda *a, **kw: VerifyAnswer(
            contains_full_logo=True,
            fit="loose",
            confidence=0.7,
            reason="loose",
            suggested_bbox=None,
            raw_response="",
            prompt_version="verify-test",
            model="test",
            usage=_usage(),
        ),
    )

    def _fake_refine_miss(*_a, **kw):
        return (
            MatchResult(
                found=False,
                bbox=None,
                confidence=0.1,
                reason="couldn't refine",
                raw_response="",
                prompt_version="4_refine",
                model="test",
                usage=_usage(),
            ),
            (40, 40),
        )

    monkeypatch.setattr(vl, "match_logo_in_zoomed", _fake_refine_miss)

    result = vl.match_with_verify(
        logo_img,
        noisy_photo,
        max_iters=2,
        verify_threshold=0.6,
        model="test-model",
        refine=True,
    )

    # Verify accepted the original loose bbox; refine missed; fall back to B0.
    assert result.verified is True
    assert result.refined is False
    assert result.final_bbox == (50, 50, 200, 200), (
        f"expected final_bbox to fall back to B0 (50,50,200,200), "
        f"got {result.final_bbox!r}"
    )
    assert result.refine_bbox is None
