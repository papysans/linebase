"""Iter 5 — variance pre-gate on the verify-loop.

A crop whose RGB array has std-dev < 15.0 AND >70% near-white pixels is
considered blank; `match_with_verify` should short-circuit without calling
the verify LLM and label the result `blank_pre_gate`.

We stub out the network-facing pieces (`match_logo_in_photo`, `verify_crop`,
`match_logo_with_feedback`) so this test never touches a real LLM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from linebase import verify_loop as vl
from linebase.llm import MatchResult


def _write_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype("uint8"), "RGB").save(path)


@pytest.fixture()
def blank_photo(tmp_path: Path) -> Path:
    """A 256x256 near-white image — every pixel >240 on all channels."""
    arr = np.full((256, 256, 3), 250, dtype="uint8")
    p = tmp_path / "blank.png"
    _write_png(arr, p)
    return p


@pytest.fixture()
def logo_image(tmp_path: Path) -> Path:
    """A tiny black-on-white square as the LOGO arg."""
    arr = np.full((64, 64, 3), 255, dtype="uint8")
    arr[16:48, 16:48] = 0
    p = tmp_path / "logo.png"
    _write_png(arr, p)
    return p


def test_variance_pre_gate_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
    blank_photo: Path,
    logo_image: Path,
) -> None:
    """Pre-gate must fire on a uniform-white candidate, skipping verify_crop."""
    # Pass-1 reports a found bbox inside the all-white image. Bbox = whole image.
    fake_primary = MatchResult(
        found=True,
        bbox=(10, 10, 200, 200),
        confidence=0.95,
        reason="fake",
        raw_response="",
        prompt_version="test",
        model="test",
        usage={"prompt_tokens": 1, "completion_tokens": 1,
               "total_tokens": 2, "reasoning_tokens": 0},
    )

    monkeypatch.setattr(
        vl, "match_logo_in_photo", lambda *a, **kw: fake_primary
    )

    # verify_crop must NOT be called when the pre-gate fires. Make it raise so
    # the test fails loudly if the short-circuit regresses.
    def _verify_should_not_run(*_a, **_kw):  # noqa: ANN001
        raise AssertionError("verify_crop must not run when pre-gate fires")

    monkeypatch.setattr(vl, "verify_crop", _verify_should_not_run)

    # The Pass-3 retry fires on a pre-gate, but we want the test to focus on
    # the pre-gate itself, so stub the retry to return "still nothing" — that
    # leaves the terminal state as (verified=False, fit_label="blank_pre_gate").
    monkeypatch.setattr(
        vl,
        "match_logo_with_feedback",
        lambda *a, **kw: MatchResult(
            found=False,
            bbox=None,
            confidence=0.0,
            reason="",
            raw_response="",
            prompt_version="4_retry",
            model="test",
        ),
    )

    result = vl.match_with_verify(
        logo_image,
        blank_photo,
        max_iters=2,
        verify_threshold=0.6,
        model="test-model",
    )

    assert result.verified is False
    assert result.final_bbox is None
    assert result.fit_label == "blank_pre_gate"
    assert result.verify_reason is not None
    assert "pre-gate" in result.verify_reason
    # The pre-gate should have fired the retry-with-feedback path.
    assert result.retried is True
    assert result.retry_reason == "blank_pre_gate"


def test_variance_pre_gate_skips_when_crop_has_texture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    logo_image: Path,
) -> None:
    """A noisy crop must NOT trigger the pre-gate — verify_crop should run."""
    rng = np.random.default_rng(seed=42)
    noisy = rng.integers(0, 256, size=(256, 256, 3), dtype="uint8")
    noisy_path = tmp_path / "noisy.png"
    _write_png(noisy, noisy_path)

    fake_primary = MatchResult(
        found=True,
        bbox=(10, 10, 200, 200),
        confidence=0.95,
        reason="fake",
        raw_response="",
        prompt_version="test",
        model="test",
        usage={"prompt_tokens": 1, "completion_tokens": 1,
               "total_tokens": 2, "reasoning_tokens": 0},
    )

    monkeypatch.setattr(vl, "match_logo_in_photo", lambda *a, **kw: fake_primary)

    called = {"verify": 0}

    def _verify_runs(*_a, **_kw):  # noqa: ANN001
        called["verify"] += 1
        from linebase.llm import VerifyAnswer

        return VerifyAnswer(
            contains_full_logo=True,
            fit="tight",
            confidence=0.9,
            reason="ok",
            suggested_bbox=None,
            raw_response="",
            prompt_version="verify-test",
            model="test",
            usage={"prompt_tokens": 1, "completion_tokens": 1,
               "total_tokens": 2, "reasoning_tokens": 0},
        )

    monkeypatch.setattr(vl, "verify_crop", _verify_runs)

    result = vl.match_with_verify(
        logo_image,
        noisy_path,
        max_iters=2,
        verify_threshold=0.6,
        model="test-model",
    )

    assert called["verify"] == 1, "verify_crop must run when pre-gate does not fire"
    assert result.verified is True
    assert result.fit_label == "tight"
    assert result.retried is False
