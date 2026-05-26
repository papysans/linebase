from __future__ import annotations

from linebase import llm


def test_active_prompt_auto_selects_only_numeric_primary_prompt(monkeypatch):
    monkeypatch.delenv("LINEBASE_PROMPT_VERSION", raising=False)

    version, _prompt = llm._active_prompt()

    assert version == "4"


def test_active_verify_prompt_auto_ignores_task_specific_prompts(monkeypatch):
    monkeypatch.delenv("LINEBASE_VERIFY_PROMPT_VERSION", raising=False)

    version, _prompt = llm._active_verify_prompt()

    assert version == "verify-1"


def test_active_verify_prompt_can_explicitly_select_design_prompt(monkeypatch):
    monkeypatch.setenv("LINEBASE_VERIFY_PROMPT_VERSION", "design_1")

    version, prompt = llm._active_verify_prompt()

    assert version == "verify-design_1"
    assert "design-patent line drawing" in prompt
