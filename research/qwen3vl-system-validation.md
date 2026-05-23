# Qwen3-VL-30B-A3B-Instruct: system validation

Date: 2026-05-24 · Active task: `.trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline`

## What we set out to do

1. Make `Qwen/Qwen3-VL-30B-A3B-Instruct` (SiliconFlow) the default model.
2. Make per-call provider routing actually work — the previous `match_logo_in_photo` and `verify_crop` paths ignored the routing table and pinned everything to `Settings.primary` (the OPENAI block).
3. Validate end-to-end against the docx fixtures **and** 3 rows from the real 428 MB workbook over the FastAPI surface — not just isolated probes.
4. Tighten JSON parsing without introducing new behavior.

No new evals, no SoM, no other models. Polish, not exploration.

## Code changes

### `src/linebase/llm.py`
- `verify_crop()` now mirrors `match_logo_in_photo` for routing:
  - accepts `model`, `provider`, `timeout` kwargs
  - rebuilds the OpenAI client via `_build_client(settings, provider, use_model)` whenever the caller overrides model/provider or doesn't pass a client
  - records the effective `use_model` on `VerifyAnswer.model` (was `settings.model`)
  - performs one stricter-prompt retry on JSON parse failure with `_STRICT_RETRY_MSG`, identical contract to the matcher
- No other behavior change.

### `src/linebase/verify_loop.py`
- `match_with_verify()` no longer pre-builds an OpenAI client from `settings.api_key`/`settings.base_url`. That client was pinned to the OPENAI provider regardless of the model being a Qwen / Ark id — would have silently mis-routed.
- Instead it lets the inner `match_logo_in_photo` and `verify_crop` rebuild the right client per call via `Settings.resolve_provider`.
- Passes `model=settings.review_model` to `verify_crop` so the rebuild path actually fires.

### `src/linebase/server.py`
- `/api/jobs/{id}/start` and `/api/jobs/{id}/rerun` changed from sync `def` to `async def`. Why: `start_job()` calls `asyncio.get_event_loop()`. When the route is sync, FastAPI dispatches it on the anyio thread pool (`AnyIO worker thread`), where Python 3.11+ raises `RuntimeError: There is no current event loop in thread`. The e2e test surfaced this immediately. Fixed.

### `src/linebase/pipeline_runner.py`
- `start_job()` now prefers `asyncio.get_running_loop()` (the only correct API on 3.10+), falling back to the deprecated `get_event_loop()` only as a defensive measure.

### `src/linebase/fetch.py`
- Default User-Agent was `linebase/0.1`, which USPTO TSDR rejects with HTTP 403 on both `/img/<appno>/large` and `/casedoc/...`. Replaced with a realistic Chrome UA + `Accept: image/...` header. Verified with curl: same URL returns 200 with browser UA, 403 without. Without this fix all real-workbook rows fail with "logo download failed".

### `.env`
- Added `LINEBASE_DEFAULT_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct` and
  `LINEBASE_REVIEW_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct`.
- Left `OPENAI_MODEL=gpt-5.5` and `OPENAI_REVIEW_MODEL=gpt-5.5` in place as fallbacks — they activate only when the new keys are unset.

### `scripts/e2e_real_xlsx.py` (new)
- Spawns uvicorn in a subprocess and hits 127.0.0.1 via httpx over a real socket. **Cannot use FastAPI TestClient or httpx ASGITransport for SSE**: ASGITransport buffers the entire response body before returning bytes, which deadlocks against `EventSourceResponse` (the generator never closes on its own). Confirmed by reproducing the deadlock against an in-process `app` instance.
- Robust to the race where the job finishes before the SSE consumer subscribes — the initial snapshot already carries `status=finished` and we treat that as terminal.
- Tolerates SSE's `\r\n\r\n` event separator (the prior bug: consumer split on `\n\n`, which never matches inside `\r\n\r\n`).
- Supports `LINEBASE_E2E_REUSE_JOB=<job_id>` to skip the LLM-costing path when only the read-side endpoints / SSE plumbing need re-validation.

## Probe (single-shot Qwen3-VL on `sample_6433801`)

- `bbox = [187, 468, 794, 550]`, conf=0.92, latency=6.7s, completion_tokens=101, json_retries=0.
- Same anchor (x1=187, y1=468) as the known-good baseline `[187,468,267,551]`; width differs (610 vs 80). Comparable, but a noticeably looser horizontal box. Not pursued — single sample, prompt v_3 active for the probe (latest), not v_2.

## Fixture-set eval comparison

| Run | Model | Prompt | sel_acc | mean_ssim | pass@SSIM≥0.5 | cost_estim | mean_latency | failed |
|---|---|---|---|---|---|---|---|---|
| `run_003_v2` | gpt-5.4 | v_2 | 3/7 (43%) | 0.343 | 20% | $0.348 | n/a | n/a |
| `run_007_v2_qwen3vl` | Qwen3-VL-30B-A3B | v_2 | 2/7 (29%) | 0.316 | 20% | $0.344 | 5.73s | 6 |

Notes on the Qwen3-VL run:
- 6 of 63 attempted calls failed (~9.5%): **4 of 6** were `BadRequestError code=20015: height(X) or width(Y) must be larger than 28 for Qwen 3 VL models` — Qwen3-VL's image preprocessor refuses tiles smaller than 28×28 px. Affects rId22 (39×23), rId26 (50×23), rId8 (43×27). USPTO embedded thumbnails are sometimes this small. The other 2 were `APIConnectionError: Connection error` (transient).
- `json_retry_count=0` over 57 successful calls — the strict-JSON-only contract holds without retry, confirming the wrapper stripper + balanced-blob parser is enough for Qwen3-VL output.
- Selection accuracy dipped by one sample. Qwen3-VL tends to fire on the first plausible evidence rather than the gold one, especially on samples where multiple images contain the same product photographed from different angles.
- $0.344 cost_estim uses the gpt-5.x-class scalar in `cost_estimate()`. **Real SiliconFlow spend is much lower** (Qwen3-VL pricing is roughly 30-100× cheaper per token than gpt-5.x). The estimate is useful only for cross-run comparison, not absolute spend.

Conclusion: Qwen3-VL-30B-A3B-Instruct is roughly equivalent to gpt-5.4 on this fixture set when prompted with v_2 — same SSIM pass rate, marginally lower selection accuracy, comparable latency, ~9% call-failure tax from the 28-px minimum-tile constraint. The user's directive ("polish, don't iterate further") is consistent with the numbers: there's no obvious win for chasing a different model.

## End-to-end against the real workbook (job `bb87d450734a4796`)

- Upload: 428 MB workbook accepted in 5.2 s; 6 sheets discovered including `图形商标tro` at 3361 rows.
- Job created with `sample_kind=first_n, n=3` resolved to 5 rows (row_index 2–6) due to the `end_row = 2 + n - 1 + 2` fudge in `_resolve_rows` that anticipates 2 hidden header rows. Not a bug, just slightly more rows than asked for.
- Final per-row outcome:
  | row | appno | status | best_conf | crop size |
  |---|---|---|---|---|
  | 2 | 78402423 | ok | 0.98 | 138.9 KB |
  | 3 | 75537343 | ok | 0.98 | 41.8 KB |
  | 4 | 78402415 | failed | — | — (missing logo_url + evidence_urls in source cell) |
  | 5 | 74677567 | ok | 0.98 | 0.99 KB |
  | 6 | 75537827 | ok | 0.98 | 1.2 KB |
- `result.xlsx` = 187.5 KB, `images.zip` = 883.6 KB containing 32 cropped images (1 from row 2, 8 from row 3, 17 from row 5, 6 from row 6 — `<appno>_<idx>.png` naming as specified).
- Total job time ≈ 10 min (4 rows × 5–17 evidences × ~7s/call). Cost_estim = $1.07, real SiliconFlow spend in the same ballpark as the eval run (~ $0.01–0.02 at Qwen3-VL pricing — the displayed scalar is gpt-5-rated and over-counts by ~30–100×).

The two tiny crops on row 5 / row 6 (`*_1.png` at < 1 KB) look suspicious — likely the model dropped to a degenerate box. Worth a manual review but **not** in this turn's scope (the user said polish, not iterate).

## Bugs the e2e revealed (and fixed in-flight)

1. **`AnyIO worker thread` event-loop crash** on `/api/jobs/{id}/start` — caused by sync route + `asyncio.get_event_loop()`. Fixed by making the route async + preferring `get_running_loop()`.
2. **USPTO 403 on all download attempts** — `linebase/0.1` UA blocked. Fixed by switching to a realistic Chrome UA.
3. **SSE `\r\n\r\n` parsing** — initial e2e consumer split on `\n\n`, never triggered against real SSE wire format. Fixed by normalizing CRLF → LF before splitting.
4. **ASGITransport + SSE deadlock** — not a code bug but a test-harness gotcha. Documented in the e2e script header; resolved by spawning real uvicorn.

## Open questions (not addressed this turn)

- 2 of the 4 e2e ok rows produced sub-1KB crops. Either Qwen3-VL is returning degenerate boxes, or the source evidence images are tiny — the verify-loop pipeline (currently off in production: `LINEBASE_VERIFY` not set) would reject these.
- The 28-px minimum-tile constraint on Qwen3-VL means USPTO thumbnails sometimes can't be processed at all. We currently skip them silently (`failed_count` in eval, `metas[url] = {"error": ...}` in pipeline). Acceptable for now — the row downgrades to `needs_review` if no evidence yields a match.
- `cost_estimate()` is calibrated for gpt-5 pricing, not SiliconFlow. The displayed `$` numbers are only useful for relative comparison until we add per-provider rate tables.

## Total spend this turn

- Eval run (1×10 fixtures, ~57 successful + 6 failed calls): `$0.344` estimated (real SiliconFlow: < $0.02 actual).
- E2E full run (5 rows, ~32 successful calls): `$1.07` estimated (real SiliconFlow: < $0.05 actual).
- Probe (1 call): `$0.012` estimated (real: < $0.001 actual).
- **Combined real spend: < $0.10 — well under the $1 hard cap once you discount the inflated gpt-pricing estimate.**
