"""Smoke probe for the verify-loop model-override plumbing.

Background: `pipeline_runner._process_row` passes `eff_model` to
`match_with_verify`, which now threads it through both Pass-1
(`match_logo_in_photo`) and the verify call (`verify_crop`). This probe
monkey-patches `OpenAI` in both `linebase.llm` and `linebase.verify_loop`
so we can assert that BOTH passes resolve to the Ark provider when the
caller pins a `doubao-*` model — without making any real HTTP calls.

Run: `.venv/Scripts/python.exe scripts/_probe_plumbing.py`
Expected output: a single PASS line and a list of base_urls (all volces.com).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


calls: list[str] = []


class _StubCompletion:
    """Minimal stand-in for a chat.completions response: one JSON-string choice + usage."""

    def __init__(self, content: str = '{"found": false, "confidence": 0}'):
        self.choices = [
            type(
                "C",
                (),
                {
                    "message": type("M", (), {"content": content})(),
                },
            )()
        ]
        self.usage = type(
            "U",
            (),
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "completion_tokens_details": None,
            },
        )()


class _StubCompletions:
    def create(self, *args, **kwargs):
        # Vary the response so the verify call still walks into a "no" branch
        # gracefully (contains_full_logo defaults to False on missing field).
        return _StubCompletion('{"found": true, "bbox": [10, 10, 50, 50], "confidence": 0.9}')


class _StubChat:
    def __init__(self):
        self.completions = _StubCompletions()


class StubClient:
    """Records every base_url passed to OpenAI(...) during the run."""

    def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None):
        calls.append(base_url or "")
        self.chat = _StubChat()


def main() -> int:
    from linebase.verify_loop import match_with_verify

    sample = Path(
        ".trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline"
        "/fixtures/sample_6433801"
    )
    logo = sample / "00_rId50_image47.png"
    photo = sample / "01_rId51_image48.png"
    if not logo.exists() or not photo.exists():
        print(f"FAIL: fixture missing under {sample}")
        return 1

    with (
        patch("linebase.llm.OpenAI", StubClient),
        patch("linebase.verify_loop.OpenAI", StubClient),
    ):
        match_with_verify(logo, photo, model="doubao-seed-2-0-pro-260215")

    print("OpenAI clients created with base_urls:")
    for u in calls:
        print(f"  - {u!r}")

    if not calls:
        print("FAIL: no OpenAI client constructions recorded")
        return 1

    bad = [u for u in calls if "volces.com" not in (u or "")]
    if bad:
        print(f"FAIL: expected all Ark base_urls (volces.com), got non-Ark: {bad}")
        return 1

    print(f"PASS: all {len(calls)} call(s) routed to Ark provider when model=doubao-*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
