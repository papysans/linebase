# Multi-Provider LLM Routing

> How `Settings.resolve_provider(model)` routes a model id to the right
> OpenAI-compatible upstream, and how to add a new provider.

---

## Overview

`linebase.config.Settings` registers one provider block per `<NAME>_API_KEY`
present in the environment, then routes per-call to the right one based on the
model id prefix. This lets the pipeline mix OpenAI (gpt-5.5), Volcengine Ark
(doubao-seed-*), and SiliconFlow (Qwen / GLM / Kimi / DeepSeek) in the same
process without separate clients.

Three providers are currently wired:

| Provider | Block prefix | Default base URL | Models routed |
|---|---|---|---|
| `openai` | `OPENAI_*` | `https://api.openai.com/v1` (overridden to `https://api.1m1ng.net/v1` in our `.env`) | `gpt-*`, `claude-*` |
| `ark` | `ARK_*` | `https://ark.cn-beijing.volces.com/api/v3` | `doubao-*`, `Doubao-*` |
| `siliconflow` | `SILICONFLOW_*` | `https://api.siliconflow.cn/v1` | `Qwen/*`, `Pro/Qwen/*`, `zai-org/*`, `Pro/zai-org/*`, `moonshotai/*`, `Pro/moonshotai/*`, `THUDM/*`, `deepseek-ai/*`, `Pro/deepseek-ai/*` |

`openai` is the mandatory primary provider — startup fails if
`OPENAI_API_KEY` is missing. The other two are optional; a missing key just
means models that route to that provider raise a clear `RuntimeError` at call
time.

---

## Routing rules (verbatim from `_PROVIDER_PREFIXES`)

```python
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
```

First match wins — order matters when prefixes overlap (`Pro/Qwen/` must be
checked before `Qwen/` if we ever care which one fires; today both route
to siliconflow so the order is informational).

Fallback: any model id that matches none of the prefixes routes to the
primary provider (openai). This is by design — handing an unrecognized id to
the OpenAI relay surfaces a clear "model not found" upstream error instead of
silently picking something wrong.

---

## Env conventions

`.env` (at repo root, gitignored) carries one block per provider. Example:

```bash
# Mandatory — openai routes all gpt-* and claude-* prefixes.
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.1m1ng.net/v1     # 1m1ng OpenAI-compatible relay
OPENAI_MODEL=gpt-5.5                          # fallback for <28 px USPTO thumbnails
OPENAI_REVIEW_MODEL=gpt-5.5

# Default active model (overrides OPENAI_MODEL for the pipeline's primary call).
LINEBASE_DEFAULT_MODEL=doubao-seed-2-0-pro-260215
LINEBASE_REVIEW_MODEL=doubao-seed-2-0-pro-260215

# Optional — Ark for Doubao Seed 2.0 family.
ARK_API_KEY=ark-...
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# Optional — SiliconFlow for Qwen / GLM / Kimi / DeepSeek / THUDM.
SILICONFLOW_API_KEY=sk-...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

Two override knobs:

- `LINEBASE_DEFAULT_MODEL` — wins over `OPENAI_MODEL` as the active model id.
  Use this to switch the default without touching legacy OpenAI vars.
- `LINEBASE_PROVIDER` — forces every call to a specific provider regardless of
  model id. Useful for debugging or for routing custom-typed model ids to a
  known-good provider. Raises if the named provider isn't configured.

Per-job override: `POST /api/jobs { model: "..." }` pins one model for that
job's primary match call. The verify-loop keeps using `settings.review_model`.

---

## Adding a 4th provider

Walk-through for hypothetically adding `bailian` (Aliyun) which exposes
qwen-vl-max:

1. **Pick the prefix(es).** Probably `qwen-vl-` (raw Aliyun model ids).
2. **Add the routing entry** in `config._PROVIDER_PREFIXES`. Insert before
   any broader prefix that would otherwise swallow it:
   ```python
   ("qwen-vl-", "bailian"),
   ```
3. **Register the provider block** in `Settings.from_env()`:
   ```python
   bailian_key = os.environ.get("BAILIAN_API_KEY")
   if bailian_key:
       providers["bailian"] = ProviderConfig(
           name="bailian",
           api_key=bailian_key,
           base_url=os.environ.get(
               "BAILIAN_BASE_URL",
               "https://dashscope.aliyuncs.com/compatible-mode/v1",
           ),
       )
   ```
4. **Add the env block** to `.env.example` (and your real `.env`):
   ```bash
   BAILIAN_API_KEY=sk-...
   BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```
5. **(Optional) Whitelist a curated model** in
   `linebase.models_catalog.MODEL_WHITELIST` so the picker UI offers it.
   Custom-typed ids that match the new prefix already pass `is_model_routable`.
6. **(Optional) Add a cost-adjustment factor** in
   `pipeline_runner._PROVIDER_COST_FACTOR` and
   `bench._PROVIDER_COST_FACTOR`. Default of `1.0` (no adjustment) is
   correct only when the provider bills at OpenAI gpt-5 rates; cheaper
   providers should use the Ark/SiliconFlow precedent of `0.02`.

That's it — no client wiring changes. `linebase.llm._build_client(settings,
model)` already calls `settings.resolve_provider(model)` and constructs a
fresh `OpenAI(...)` against the right base URL + key.

---

## Why prefix routing (and not a model→provider dict)?

- New SiliconFlow models land continuously (`Pro/Qwen/...`, `Pro/zai-org/...`).
  A prefix list keeps the whitelist small and lets the user type a fresh
  model id in the picker without a code change.
- Provider boundaries are stable: SiliconFlow has owned the `Qwen/` and
  `zai-org/` namespaces since the project began.
- The whitelist (`models_catalog.MODEL_WHITELIST`) is for UX (the dropdown
  in ConfigurePage), not for routing. Treat the two concerns separately.

---

## Forbidden patterns

- Don't `os.environ["LINEBASE_PROVIDER"] = ...` from inside library code.
  That env var is a user-facing override; library code routes via
  `settings.resolve_provider(model)`.
- Don't hard-code `base_url=` in a new caller. Always go through
  `linebase.llm._build_client(settings, model)` which reads the right
  provider block.
- Don't add a provider block whose API key is required for startup.
  Only `OPENAI_API_KEY` is mandatory; the others must remain optional so a
  cloned repo with only one key still runs.
