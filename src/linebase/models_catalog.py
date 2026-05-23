"""Curated whitelist of vision-capable models for the model picker.

Each entry has been probed working against the LOGO + evidence vision task in
prior turns. The frontend renders this list as a dropdown and also accepts a
free-text "custom" model id; on submit, validation requires the id to either be
in the whitelist OR start with one of the routable provider prefixes from
`linebase.config._PROVIDER_PREFIXES`.

Notes column is informational — surfaces known per-model quirks (e.g. the
Qwen3-VL <28 px tile rejection that triggers the gpt-5.5 fallback in
pipeline_runner._process_row).
"""
from __future__ import annotations

from dataclasses import dataclass

from linebase.config import _PROVIDER_PREFIXES


@dataclass(frozen=True)
class ModelOption:
    id: str
    provider: str
    label: str
    notes: str


MODEL_WHITELIST: tuple[ModelOption, ...] = (
    ModelOption(
        id="doubao-seed-2-0-pro-260215",
        provider="ark",
        label="Doubao Seed 2.0 Pro (默认 · 最准)",
        notes="thinking model; 71% sel-acc bench winner",
    ),
    ModelOption(
        id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        provider="siliconflow",
        label="Qwen3-VL 30B A3B (快 · 小图回落)",
        notes="MoE; rejects images < 28 px (auto-falls back to gpt-5.5)",
    ),
    ModelOption(
        id="Qwen/Qwen3-VL-32B-Instruct",
        provider="siliconflow",
        label="Qwen3-VL 32B Instruct",
        notes="dense",
    ),
    ModelOption(
        id="doubao-seed-2-0-mini-260428",
        provider="ark",
        label="Doubao Seed 2.0 Mini (国产·快)",
        notes="thinking model; occasional 150 s timeouts",
    ),
    ModelOption(
        id="zai-org/GLM-4.5V",
        provider="siliconflow",
        label="GLM-4.5V",
        notes="wraps output in <|box|> tokens",
    ),
    ModelOption(
        id="gpt-5.5",
        provider="openai",
        label="GPT-5.5 (1m1ng 中转)",
        notes="fallback; bbox less precise",
    ),
)


def whitelist_ids() -> set[str]:
    return {opt.id for opt in MODEL_WHITELIST}


def is_model_routable(model_id: str) -> bool:
    """True iff `model_id` is either in the whitelist OR starts with a known
    provider prefix from the config's routing table.

    Used by `/api/jobs` to decide whether to accept a custom-typed model id.
    """
    if model_id in whitelist_ids():
        return True
    for prefix, _name in _PROVIDER_PREFIXES:
        if model_id.startswith(prefix):
            return True
    return False


def to_dict(opt: ModelOption) -> dict[str, str]:
    return {"id": opt.id, "provider": opt.provider, "label": opt.label, "notes": opt.notes}
