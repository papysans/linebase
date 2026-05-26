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
from linebase.llm import MatchResult, VerifyAnswer


def _write_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype("uint8"), "RGB").save(path)


_REPO = Path(__file__).resolve().parents[1]
_TRUTH_SET = _REPO / "docs" / "truth_set"


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


def test_verify_round_upscales_small_crop_and_downscales_suggested_bbox(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    logo_image: Path,
) -> None:
    """Verifier sees an upscaled crop, but suggested_bbox returns to crop coords."""
    photo = tmp_path / "photo.png"
    arr = np.zeros((600, 600, 3), dtype="uint8")
    arr[:, :, 0] = np.arange(600, dtype="uint16")[:, None] % 256
    arr[:, :, 1] = np.arange(600, dtype="uint16")[None, :] % 256
    arr[:, :, 2] = 80
    _write_png(arr, photo)

    seen_size: dict[str, tuple[int, int]] = {}

    def _fake_verify_crop(_logo, crop_path, **_kw):  # noqa: ANN001
        with Image.open(crop_path) as img:
            seen_size["crop"] = img.size
        return VerifyAnswer(
            contains_full_logo=True,
            fit="loose",
            confidence=0.9,
            reason="ok",
            suggested_bbox=(40, 60, 240, 300),
            raw_response="",
            prompt_version="verify-test",
            model="test",
            usage=None,
        )

    monkeypatch.setattr(vl, "verify_crop", _fake_verify_crop)

    ans, cbox, pre_gated, _stats = vl._verify_round(
        (200, 200, 320, 320),
        photo,
        logo_image,
        600,
        600,
        settings=None,  # type: ignore[arg-type]
        client=None,
        verify_model="test-model",
    )

    assert pre_gated is False
    assert cbox == (176, 176, 344, 344)
    assert seen_size["crop"] == (504, 504)
    assert ans is not None
    assert ans.suggested_bbox == (13, 20, 80, 100)


def test_match_with_verify_soft_accepts_strong_qwen_low_quality_reject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    logo_image: Path,
) -> None:
    """A strong Qwen bbox should survive a verifier's low-quality false reject."""
    photo = tmp_path / "textured.png"
    rng = np.random.default_rng(seed=123)
    arr = rng.integers(20, 180, size=(500, 500, 3), dtype="uint8")
    _write_png(arr, photo)

    primary = MatchResult(
        found=True,
        bbox=(120, 140, 240, 260),
        confidence=0.85,
        reason="shape match",
        raw_response="",
        prompt_version="test",
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        raw_bbox=(240.0, 280.0, 480.0, 520.0),
        bbox_coord_mode="qwen_normalized_1000",
    )
    monkeypatch.setattr(vl, "match_logo_in_photo", lambda *a, **kw: primary)
    monkeypatch.setattr(
        vl,
        "verify_crop",
        lambda *a, **kw: VerifyAnswer(
            contains_full_logo=False,
            fit="wrong",
            confidence=0.45,
            reason="The crop is faint, blurry, and degraded.",
            suggested_bbox=None,
            raw_response="",
            prompt_version="verify-test",
            model="test",
            usage=None,
        ),
    )

    result = vl.match_with_verify(
        logo_image,
        photo,
        max_iters=2,
        verify_threshold=0.6,
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        refine=False,
    )

    assert result.verified is True
    assert result.soft_verified is True
    assert result.final_bbox == primary.bbox
    assert result.fit_label == "loose"


def test_soft_accept_rejects_near_white_blank_qwen_bbox(
    tmp_path: Path,
    logo_image: Path,
) -> None:
    """The soft path must not rescue a high-confidence bbox on blank white."""
    photo = tmp_path / "blankish.png"
    arr = np.full((500, 500, 3), 250, dtype="uint8")
    _write_png(arr, photo)
    primary = MatchResult(
        found=True,
        bbox=(120, 140, 240, 260),
        confidence=0.9,
        reason="shape match",
        raw_response="",
        prompt_version="test",
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        bbox_coord_mode="qwen_normalized_1000",
    )

    assert (
        vl._should_soft_accept_verify_reject(
            primary,
            "The crop is faint and blurry.",
            photo,
        )
        is False
    )


def test_loose_verify_rejects_aspect_mismatch_after_suggestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truth-set 4601531 pair_01: loose suggested bbox is still rejected."""
    logo = _TRUTH_SET / "4601531_14" / "logo.png"
    photo = _TRUTH_SET / "4601531_14" / "pair_01" / "evidence.png"
    if not logo.exists() or not photo.exists():
        pytest.skip("truth-set 4601531_14 fixture missing")

    primary = MatchResult(
        found=True,
        # Observed E2E bad bbox: includes the true wide logo plus page chrome
        # and recommendation area below it, producing IoU≈0.216 vs truth.
        bbox=(348, 275, 618, 657),
        confidence=0.92,
        reason="wide emblem but bbox includes page chrome",
        raw_response="",
        prompt_version="test",
        model="test",
    )
    monkeypatch.setattr(vl, "match_logo_in_photo", lambda *a, **kw: primary)
    monkeypatch.setattr(
        vl,
        "verify_crop",
        lambda *a, **kw: VerifyAnswer(
            contains_full_logo=True,
            fit="loose",
            confidence=0.9,
            reason="logo is present but crop has large background padding",
            # E2E observed the verifier shrink the crop, but still to a tall
            # downward-shifted bbox that remains far from the truth box.
            suggested_bbox=(100, 130, 300, 330),
            raw_response="",
            prompt_version="verify-test",
            model="test",
            usage=None,
        ),
    )
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
        logo,
        photo,
        max_iters=2,
        verify_threshold=0.6,
        model="test-model",
        refine=False,
    )

    assert result.verified is False
    assert result.final_bbox is None
    assert result.fit_label == "loose"
    assert result.verify_reason is not None
    assert "loose bbox rejected" in result.verify_reason


def test_loose_verify_accepts_matching_aspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truth-set 6433801 pair_01: plausible low-quality loose bbox is accepted."""
    logo = _TRUTH_SET / "6433801_12" / "logo.png"
    photo = _TRUTH_SET / "6433801_12" / "pair_01" / "evidence.png"
    if not logo.exists() or not photo.exists():
        pytest.skip("truth-set 6433801_12 fixture missing")

    primary = MatchResult(
        found=True,
        bbox=(379, 711, 534, 829),
        confidence=0.9,
        reason="plausible Corvette windshield mark bbox",
        raw_response="",
        prompt_version="test",
        model="test",
    )
    monkeypatch.setattr(vl, "match_logo_in_photo", lambda *a, **kw: primary)
    monkeypatch.setattr(
        vl,
        "verify_crop",
        lambda *a, **kw: VerifyAnswer(
            contains_full_logo=True,
            fit="loose",
            confidence=0.9,
            reason="logo is present with mild padding",
            suggested_bbox=None,
            raw_response="",
            prompt_version="verify-test",
            model="test",
            usage=None,
        ),
    )

    result = vl.match_with_verify(
        logo,
        photo,
        max_iters=2,
        verify_threshold=0.6,
        model="test-model",
        refine=False,
    )

    assert result.verified is True
    assert result.final_bbox == primary.bbox
    assert result.fit_label == "loose"
