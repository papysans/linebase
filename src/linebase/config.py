"""Load .env config — pure module, no side effects beyond reading env at call time.

Multi-provider support (added 2026-05-23):
  * Provider blocks: OPENAI_*, ARK_*, SILICONFLOW_*. Any block whose API key is
    missing is simply not registered — non-fatal.
  * `Settings.resolve_provider(model)` picks the right provider for a model id
    based on a prefix table, with `LINEBASE_PROVIDER` env override.
  * The legacy single-provider fields (`api_key`, `base_url`) remain as
    convenience aliases for `Settings.primary.*` so older call sites still work.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class ProviderConfig:
    name: str         # "openai" | "ark" | "siliconflow"
    api_key: str
    base_url: str


# Routing rules — prefix → provider name. First match wins.
# Order matters: keep the more specific prefixes first.
_PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("doubao-", "ark"),
    ("Doubao-", "ark"),
    ("zai-org/", "siliconflow"),
    ("Pro/zai-org/", "siliconflow"),
    ("Pro/moonshotai/", "siliconflow"),
    ("moonshotai/", "siliconflow"),
    ("Qwen/", "siliconflow"),
    ("Pro/Qwen/", "siliconflow"),
    ("THUDM/", "siliconflow"),
    ("deepseek-ai/", "siliconflow"),
    ("Pro/deepseek-ai/", "siliconflow"),
    ("gpt-", "openai"),
    ("claude-", "openai"),
)


@dataclass(frozen=True)
class Settings:
    primary: ProviderConfig          # the default provider (OPENAI block)
    model: str                       # default model id (OPENAI_MODEL)
    review_model: str
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # --- Convenience aliases (so legacy code that still reads s.api_key works) -
    @property
    def api_key(self) -> str:
        return self.primary.api_key

    @property
    def base_url(self) -> str:
        return self.primary.base_url

    # --- Routing -----------------------------------------------------------
    def resolve_provider(self, model: str) -> ProviderConfig:
        """Pick the provider config for a given model id.

        Env override:
          LINEBASE_PROVIDER=siliconflow → forces that provider regardless of model.

        Otherwise: longest-prefix match on `_PROVIDER_PREFIXES`. Falls back to
        the primary provider (openai) for anything unrecognized.
        """
        override = os.environ.get("LINEBASE_PROVIDER", "").strip().lower()
        if override:
            if override not in self.providers:
                raise RuntimeError(
                    f"LINEBASE_PROVIDER={override!r} but provider not configured. "
                    f"Available: {sorted(self.providers)}"
                )
            return self.providers[override]
        for prefix, name in _PROVIDER_PREFIXES:
            if model.startswith(prefix):
                if name in self.providers:
                    return self.providers[name]
                raise RuntimeError(
                    f"Model {model!r} routes to provider {name!r} which is not configured. "
                    f"Set {name.upper()}_API_KEY (+ optional {name.upper()}_BASE_URL) in .env. "
                    f"Configured: {sorted(self.providers)}"
                )
        # Unknown prefix — default to primary.
        return self.primary

    @classmethod
    def from_env(cls, env_path: str | os.PathLike[str] | None = None) -> "Settings":
        if env_path is None:
            env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(env_path, override=False)

        providers: dict[str, ProviderConfig] = {}

        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError(f"OPENAI_API_KEY missing (looked in env and {env_path})")
        providers["openai"] = ProviderConfig(
            name="openai",
            api_key=openai_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

        ark_key = os.environ.get("ARK_API_KEY")
        if ark_key:
            providers["ark"] = ProviderConfig(
                name="ark",
                api_key=ark_key,
                base_url=os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            )

        sf_key = os.environ.get("SILICONFLOW_API_KEY")
        if sf_key:
            providers["siliconflow"] = ProviderConfig(
                name="siliconflow",
                api_key=sf_key,
                base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            )

        # Pick the default model — LINEBASE_DEFAULT_MODEL wins over OPENAI_MODEL
        # so we can switch to a non-OpenAI default without touching the legacy
        # OPENAI_MODEL value.
        default_model = (
            os.environ.get("LINEBASE_DEFAULT_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        review_model = (
            os.environ.get("LINEBASE_REVIEW_MODEL")
            or os.environ.get("OPENAI_REVIEW_MODEL")
            or default_model
        )

        return cls(
            primary=providers["openai"],
            model=default_model,
            review_model=review_model,
            providers=providers,
        )
