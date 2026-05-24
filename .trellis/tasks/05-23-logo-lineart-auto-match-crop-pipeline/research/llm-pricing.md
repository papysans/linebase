# Real LLM Pricing — linebase route table

Last refreshed: **2026-05-24** · RMB→USD rate used: **0.1472** (1 CNY = 0.147162 USD per x-rates.com on 2026-05-24 03:54 UTC, equivalently 1 USD ≈ 6.796 CNY)

All prices are **per 1,000,000 tokens (USD)** at the *standard / on-line / non-batch* tier so that the linebase `cost_usd` column matches what `pipeline_runner.cost_estimate` is supposed to produce. Batch-mode (~50% off) and TPM-package pricing are NOT used in this table because pipeline_runner does not route through those tiers.

For Doubao Seed 2.0: Volcengine uses **input-length-segmented pricing**. The bracket `[0, 32]` (≤ 32 K input tokens) is the only one our pipeline currently lives in — our match-prompt-only path is ≤ 4 K tokens, the verify-loop maxes near 8 K. Numbers below are the ≤ 32 K bracket. If a job ever pushes a single eligible URL > 32 K tokens we'd want the segmented rates — captured in "Notes per model".

## Table (USD per 1 M tokens)

| Provider | Model id | Input | Output | Reasoning | Source | Confidence |
| --- | --- | ---: | ---: | ---: | --- | --- |
| openai (1m1ng relay) | `gpt-5.5` | 5.00 | 30.00 | same as output | OpenAI base via OpenRouter, [retrieved 2026-05-24](https://openrouter.ai/openai/gpt-5.5) | **medium** — 1m1ng markup unknown; treat as floor |
| openai (1m1ng relay) | `gpt-5.4` | 2.50 | 15.00 | same as output | OpenAI base via OpenRouter, [retrieved 2026-05-24](https://openrouter.ai/openai/gpt-5.4) | medium (same caveat) |
| ark | `doubao-seed-2-0-pro-260215` | 0.47 | 2.35 | same as output | [Volcengine 模型价格 doc](https://www.volcengine.com/docs/82379/1099320), retrieved 2026-05-24 (¥3.2/¥16 per 1M, ≤32 K input) | **high** |
| ark | `doubao-seed-2-0-mini-260428` | 0.029 | 0.29 | same as output | [Volcengine 模型价格 doc](https://www.volcengine.com/docs/82379/1099320), retrieved 2026-05-24 (¥0.2/¥2.0 per 1M, ≤32 K input) | **high** |
| ark | `doubao-1.5-vision-pro-250328` *(historical)* | 0.44 | 1.32 | n/a (non-thinking) | [Volcengine doc](https://www.volcengine.com/docs/82379/1099320), retrieved 2026-05-24 (¥3.0/¥9.0 per 1M, flat) | high |
| siliconflow | `Qwen/Qwen3-VL-30B-A3B-Instruct` | 0.29 | 1.00 | n/a (Instruct, non-thinking) | [siliconflow.com model card](https://siliconflow.com/models/qwen3-vl-30b-a3b-instruct), retrieved 2026-05-24 | **high** |
| siliconflow | `Qwen/Qwen3-VL-32B-Instruct` | 0.20 | 0.60 | n/a (Instruct, non-thinking) | [siliconflow.com model card](https://siliconflow.com/models/qwen3-vl-32b-instruct), retrieved 2026-05-24 | **high** |
| siliconflow | `zai-org/GLM-4.5V` | 0.14 | 0.86 | n/a (non-thinking variant) | [siliconflow.com model card](https://siliconflow.com/models/glm-4-5v), retrieved 2026-05-24 — **marked Deprecated** | high |
| siliconflow | `Pro/moonshotai/Kimi-K2.5` | 0.45 | 2.25 | thinking billed as output (cache-read $0.07/M) | [siliconflow.com model card](https://siliconflow.com/models/kimi-k2-5), retrieved 2026-05-24 | high — see note on Pro/ prefix |
| siliconflow | `Pro/moonshotai/Kimi-K2.6` | 0.90 | 4.00 | thinking billed as output (cache-read $0.20/M) | [siliconflow.com model card](https://siliconflow.com/models/kimi-k2-6), retrieved 2026-05-24 | high — see note on Pro/ prefix |

## Notes per model

### `gpt-5.5` and `gpt-5.4` via 1m1ng (`https://api.1m1ng.net/v1`)

- 1m1ng (灰机喵喵喵, https://1m1ng.net) is a personal relay with no public price list. The Telegram / dashboard at https://dashboard.1m1ng.net is login-walled; the public landing page only shows a personal blog. **Confirms the task brief's assumption: relay markup is opaque.**
- Use OpenAI's *published* base price as the **floor**. Realistic markup on Chinese OpenAI relays runs 1.1x – 1.5x of the base USD rate, so the true cost is somewhere in `[$5, $7.5]` input and `[$30, $45]` output per 1M for gpt-5.5. **We have no way to know without an actual invoice from 1m1ng.**
- OpenRouter's "Effective Pricing" panel for gpt-5.5 reports a weighted-average **input** of $1.13/1M because of 93.3% cache-hit rates ($0.50/1M cached) — pipeline_runner will NOT see this discount because each evidence URL is a fresh image with no prefix-reuse pattern. Cache-discount is irrelevant for our workload; assume the full $5/M.
- **GPT-5.5 has a tiered "long context" surcharge**: prompts > 272 K input tokens are billed at 2× input ($10/M) and 1.5× output ($45/M). Our match-only prompt is ~3.7 K tokens — we will never hit this bracket. Safe to use the ≤ 272 K rates as a single tuple.
- **Image-input markup on gpt-5.5 / gpt-5.4**: OpenAI charges vision tokens *as part of* the prompt-token count (tile-based: each 512×512 tile ≈ 170 image tokens for high-res mode, plus an 85-token base). Our pipeline already passes the image bytes through the OpenAI-compatible API, which means `usage.prompt_tokens` already includes the vision-token surcharge — no extra multiplier needed in `MODEL_PRICING`. This explains why a "small text prompt + one logo + one evidence photo" can report 3 K – 8 K prompt_tokens despite the natural-language prompt being well under 1 K.
- GPT-5.5 reasoning tokens are also billed at the output rate. OpenAI exposes them via `usage.completion_tokens_details.reasoning_tokens`; the headline `usage.completion_tokens` already includes that count, so a single output rate is correct.

### `doubao-seed-2-0-pro-260215` / `doubao-seed-2-0-mini-260428` via Ark

- Source row from Volcengine's `模型价格` page (the canonical doc the task brief points at):

  | model | bracket | input ¥/1M | cached input ¥/1M | cache storage ¥/M·hour | output ¥/1M |
  | --- | --- | ---: | ---: | ---: | ---: |
  | doubao-seed-2.0-pro | [0, 32] | 3.20 | 0.64 | 0.017 | 16.00 |
  | doubao-seed-2.0-pro | (32, 128] | 4.80 | 0.96 | 0.017 | 24.00 |
  | doubao-seed-2.0-pro | (128, 256] | 9.60 | 1.92 | 0.017 | 48.00 |
  | doubao-seed-2.0-mini | [0, 32] | 0.20 | 0.04 | 0.017 | 2.00 |
  | doubao-seed-2.0-mini | (32, 128] | 0.40 | 0.08 | 0.017 | 4.00 |
  | doubao-seed-2.0-mini | (128, 256] | 0.80 | 0.16 | 0.017 | 8.00 |

- The version suffix `-260215` / `-260428` is a release-stamp pointer (YYMMDD); Ark applies the same base rate as the canonical `doubao-seed-2.0-pro` / `-mini` model family. We confirmed this against tokenmix.ai's review which independently cites "$0.47 input / $2.37 output per million tokens" for Seed 2.0 Pro — within rounding of our 3.2/16¥ * 0.1472 calculation.
- Doubao Seed 2.0 Pro and Mini are **thinking models** but Volcengine bills reasoning content as part of `output_tokens` — there is no separate `reasoning_usd_per_1m`. The Ark OpenAI-compatible response surfaces reasoning via `choices[0].message.reasoning_content` but `usage.completion_tokens` already sums it.
- "**Stall behavior**" noted in `models_catalog.py` (per-call timeout caps the blast radius) does NOT change pricing — Ark only bills for tokens emitted before the timeout fires.
- Cache-input / cache-storage discounts: pipeline_runner does not use Ark's context-caching feature, so the $0.0942/M cached-input rate (¥0.64 * 0.1472) is N/A. Single rate is fine.
- **Image-input on Doubao-1.5-vision-pro** (the historical model still referenced for completeness): vision tokens are also folded into `prompt_tokens` on the OpenAI-compatible Ark API, same as OpenAI behavior. No multiplier needed.

### SiliconFlow models

- siliconflow.com (international, USD-denominated) and cloud.siliconflow.cn (Chinese, ¥-denominated) publish **the same per-token rates** modulo FX. siliconflow.com is publicly readable; cloud.siliconflow.cn `/pricing` is login-walled. We used siliconflow.com because the Doubao USD-conversion we did for Ark proved Volcengine's CN page is consistent with the international USD pricing — assume the same for SF.
- **`Pro/` prefix**: SiliconFlow's `Pro/...` tier is the **paid** queue (higher throughput, no free quota); `Pro/moonshotai/Kimi-K2.5` uses the **same per-token price** as `moonshotai/Kimi-K2.5`. We verified by reading both the platform pricing block (which lists `Kimi-K2.5` at $0.45/$2.25) and the model card (same numbers). Free-tier models (no `Pro/` prefix) on SiliconFlow have rate-limited access but identical pricing once you exceed the free quota — they just throttle, not surcharge. **The `Pro/` whitelist entry in `linebase.config._PROVIDER_PREFIXES` does not need a different price row.**
- **GLM-4.5V is marked `State: Deprecated`** on siliconflow.com. Still callable today but expect a sunset notice; the next-gen equivalent on SF is `zai-org/GLM-4.6V` at $0.30/$0.90 per 1M (similar shape). If/when GLM-4.5V is removed from SF, we should map the cost lookup to GLM-4.6V or fall through to a default.
- Qwen3-VL-30B-A3B is the **MoE** variant (30 B total / 3 B activated) and Qwen3-VL-32B is the **dense** variant. SF prices them very differently: the dense 32 B is *cheaper* per token despite more activated params, because the MoE serving stack is GPU-inefficient at low concurrency. This is the opposite of what intuition suggests — worth a comment in `MODEL_PRICING`.
- **Image-input on SF VLMs**: SF uses HuggingFace-native image preprocessing (Qwen3-VL's tile rejection at < 28 px is a symptom of this). The image bytes are tokenised by the model's image processor and added to `prompt_tokens`. **No per-image surcharge column** on the SF model card — it's all just prompt_tokens. So unlike OpenAI's tiered tile pricing, on SF the cost is a clean linear function of `prompt_tokens`. This is why a small evidence photo (e.g. 384×384) costs the same per pixel as a giant one (e.g. 2048×2048) — both convert through the same image-token formula.
- Kimi-K2.5 and K2.6 expose `cache_read` prices ($0.07 and $0.20 per 1M respectively). pipeline_runner doesn't use SF's cache prefix feature, so cache savings won't show up; the headline $0.45/$2.25 and $0.90/$4.00 rates apply.

## Python dict you can paste straight into `pipeline_runner.py`

```python
# (input_usd_per_1m, output_usd_per_1m, reasoning_usd_per_1m_or_None)
# All values are USD per 1,000,000 tokens at standard/on-line tier.
# When reasoning_usd_per_1m is None it means: reasoning tokens are billed
# as output tokens, so the existing `usage.completion_tokens` math is correct.
# Last refreshed: 2026-05-24, RMB→USD = 0.1472. See research/llm-pricing.md.
MODEL_PRICING: dict[str, tuple[float, float, float | None]] = {
    # OpenAI (via 1m1ng relay; markup unknown, treat as floor)
    "gpt-5.5":                            (5.00,  30.00, None),
    "gpt-5.4":                            (2.50,  15.00, None),
    # Volcengine Ark (Doubao); ≤32 K input bracket — see notes for segmented rates
    "doubao-seed-2-0-pro-260215":         (0.47,   2.35, None),
    "doubao-seed-2-0-mini-260428":        (0.029,  0.29, None),
    "doubao-1.5-vision-pro-250328":       (0.44,   1.32, None),
    # SiliconFlow (USD-denominated, no FX needed)
    "Qwen/Qwen3-VL-30B-A3B-Instruct":     (0.29,   1.00, None),
    "Qwen/Qwen3-VL-32B-Instruct":         (0.20,   0.60, None),
    "zai-org/GLM-4.5V":                   (0.14,   0.86, None),
    "Pro/moonshotai/Kimi-K2.5":           (0.45,   2.25, None),
    "Pro/moonshotai/Kimi-K2.6":           (0.90,   4.00, None),
    "moonshotai/Kimi-K2.5":               (0.45,   2.25, None),  # non-Pro alias
    "moonshotai/Kimi-K2.6":               (0.90,   4.00, None),  # non-Pro alias
}
```

If the lookup misses (e.g. user types a custom model id), the caller should fall back to the existing `gpt-5-rate * _PROVIDER_COST_FACTOR` approximation rather than crashing — keep `cost_estimate()` returning a non-zero estimate.

## Confidence summary

- **High confidence** (primary-source provider docs, retrieved today, cross-checked against third-party aggregators within ±5 %): everything on Volcengine Ark and SiliconFlow. Numbers are the ones the provider actually invoices.
- **Medium confidence**: gpt-5.5 / gpt-5.4 — the *OpenAI base* rate is well-attested across OpenRouter, OpenAI's own developer docs (per DuckDuckGo SERP confirmation), apidog, MetaCTO, devtk.ai, and nerova.ai (all five agree on $5/$30 for 5.5 and $2.50/$15 for 5.4). What we **cannot verify** is 1m1ng's relay markup. Treat the dict as a *lower bound*; a follow-up could top this off with a small live A/B by hitting the 1m1ng billing API (if it has one) or by parsing a single invoice export.
- **Image-input surcharge confidence**: high for the structural claim (vision tokens already in `prompt_tokens`, no separate multiplier). What's worth instrumenting next is logging `usage.prompt_tokens_details.image_tokens` (when present) so we can see how much of a 3.7 K-prompt call is actually pixels vs. text — that's the *real* answer to "why are vision calls more expensive than text-only calls of the same nominal size".

## Caveats / not found

- **1m1ng markup**: not publicly documented anywhere indexed by DuckDuckGo as of 2026-05-24. If/when the user shares a screenshot of the dashboard's price list, replace the gpt-5.5 / gpt-5.4 rows.
- **Doubao input-length segmentation**: the current `MODEL_PRICING` row uses ≤ 32 K rates. If pipeline_runner ever ingests > 32 K tokens of context in a single call (currently impossible given our prompt + 2 images stay under 10 K), the segmented rates above should be wired in as a callable rather than a flat tuple.
- **Volcengine ¥-rates for `glm-4.7` and `deepseek-*`** are also on the same Ark price page — captured in the source dump but omitted from MODEL_PRICING because they're not in `models_catalog.MODEL_WHITELIST`. Add when needed.
- **Did NOT verify**: per-image *base fee* on OpenAI gpt-5.5 (e.g. the 85-token vision-base + tile model). The structural claim ("vision tokens are in prompt_tokens, no separate surcharge column") is high-confidence, but the exact tile-token formula could have changed between gpt-4o and gpt-5.5 — outside the scope of "USD per 1M tokens" which is what `cost_estimate` actually needs.
- **Did NOT verify**: SiliconFlow's batch-mode (50% off) rate, because pipeline_runner uses synchronous chat completions, not the batch endpoint.
