# LLM / HTTP / SSE Gotchas

> Concrete bugs we've hit in the dev loop, with the workaround for each.
> Add new entries here when you spend more than 30 minutes debugging
> something a future reader could read in 30 seconds.

---

## Qwen3-VL rejects images < 28 px on either side

Symptom: HTTP 400 from SiliconFlow with code `20015` and a message like
`height(23) or width(39) must be larger than 28 for Qwen 3 VL models`.

USPTO TSDR returns tiny `large.png` thumbnails for some marks (the rId22 /
rId26 / rId8 cases in the fixture set are 39×23, 50×23, 43×27 — all reject).

Workaround: `pipeline_runner._process_row` catches the call, sniffs for the
stable substrings `"must be larger than 28"` and `"must be > 28"`, and retries
once with `gpt-5.5` (which has no minimum-tile constraint). When the fallback
fires the per-evidence meta gets `fallback_model="gpt-5.5"` and the row's
`best_fallback_model` is surfaced as a "回落 gpt-5.5" pill in `ReviewPage`.

The bench's per-model abort threshold also catches the related case of "the
whole sample failed" — see `bench.RunResult.failed_count`.

Doubao Seed 2.0 (pro and mini) handles sub-28 px tiles transparently — no
fallback fires under Doubao. The constraint is a Qwen3-VL preprocessor quirk,
not a 国产 vision-API issue. Verified across 126 combined calls in
`research/lite-benchmark-4way.md`.

---

## GLM-4.5V wraps output in `<|begin_of_box|>...<|end_of_box|>`

Symptom: `linebase.llm.match_logo_in_photo` raises `json.JSONDecodeError` even
though the model returned valid-looking JSON. Inspecting the raw response
reveals the JSON object is wrapped in literal Zhipu special tokens:

```
<|begin_of_box|>{"found": true, "bbox": [...], ...}<|end_of_box|>
```

Workaround: `linebase.llm._extract_json` strips both tokens before
`json.loads`. Don't remove that branch — `zai-org/GLM-4.5V` is on the
whitelist and SiliconFlow doesn't post-process the response.

The same `_extract_json` also strips ```json ... ``` markdown fences (Kimi
and DeepSeek-VL2 will occasionally emit those when reasoning is on).

---

## Kimi-K2.x and other thinking models take 50–150 s per call

Symptom: `openai.APITimeoutError` after the default 60 s SDK timeout when
calling `moonshotai/Kimi-K2-Instruct` or `doubao-seed-2-0-mini-260428` on
a large photo.

Workaround:

- `bench.run_eval(..., timeout_s=120.0)` is the default.
- `pipeline_runner` calls go through `linebase.llm.match_logo_in_photo`
  which honours `OPENAI_TIMEOUT` env / kwarg; bump it for thinking-model
  pipelines.
- Doubao Seed 2.0 Pro averages 23 s/call on the fixture set — the timeout
  budget needs slack for the 99th percentile, not the mean.
- Don't enable SDK-level retries (`max_retries=0` is set in
  `_build_client`); a 60 s timeout that retries is a 180 s blocking call
  that the SSE consumer can't cancel.

If you want a one-shot retry on `APITimeoutError` for the pipeline, do it at
the `_process_row` level (where you can `await` it from the asyncio loop)
rather than inside the OpenAI SDK.

---

## Doubao Seed 2.0 — real-world stalls on multi-evidence rows

Symptom: a production pipeline batch using `LINEBASE_DEFAULT_MODEL=doubao-seed-2-0-pro-260215`
emits one `progress` event for `0/6` rows and then never advances. The
uvicorn process is alive (CPU < 11 s after the stall) but the
`pipeline_runner._process_row` task is blocked inside an OpenAI SDK
`chat.completions.create` call that never returns — the underlying HTTPS
connection to `ark.cn-beijing.volces.com` is just sitting there. The SSE
consumer eventually disconnects with `httpx.ReadError` after the client
gives up, but the server-side call never raises.

Observed:

- Doubao Seed 2.0 Pro wins the 10-fixture bench (71 % selection accuracy,
  zero failures — see `research/lite-benchmark-4way.md`).
- On the same night, two consecutive 6-row production batches over a
  `图形商标tro` worksheet hung on `row_index=7` (`appno=77354840`,
  9 high-res USPTO casedoc evidence images). Once with
  `doubao-seed-2-0-pro-260215`, once with `doubao-seed-2-0-mini-260428`.
- Forensics: `scripts/_e2e_out/night_run_v2/launcher2.log`,
  `launcher3.log`, and `uvicorn2.log` (the latter shows the LLM call was
  accepted but no `row_done` ever fired).
- Same model, same logo, same 9 evidences, same `match_logo_in_photo`
  entry point — reproduced ~16 hours later in isolation
  (`scripts/_diag_doubao_stall2.py`) — completed 9/9 in 246 s cumulative
  with no stall. So the bug is **load- / time-of-day-correlated, not
  per-image**. Likely candidates: Volcengine Ark per-tenant capacity at
  2-3 a.m. CST, 1m1ng-proxy keep-alive, or anyio threadpool exhaustion
  when 9 sequential `run_in_executor` calls share one HTTP connection.

Mitigation:

- `linebase.llm._create_completion` now always passes `timeout=` to the
  OpenAI SDK. The default is `LINEBASE_LLM_TIMEOUT_S` (env, default 90 s).
  A stalled call raises `openai.APITimeoutError`; the pipeline catches it,
  marks the evidence as `{"error": "llm: ..."}` in the row meta, and
  continues with the next evidence. A single bad evidence can never
  again hang an entire batch indefinitely.
- Same timeout applies to `verify_crop`.
- Production default reverted to `gpt-5.5` on 2026-05-23. Doubao stays
  on the whitelist (still useful for the dev-loop bench) but with a
  warning label and a `notes` field that points back here.

What we are NOT doing:

- SDK-level retries stay at `max_retries=0` (see Kimi gotcha above). A
  60 s timeout that retries 2× would be a 180 s blocking call that the
  SSE consumer can't cancel — worse than the original stall.
- We do not retry the timed-out evidence; it just gets skipped. A single
  evidence is rarely the only one for a row; the row's `best_*` selection
  picks from whatever did come back.

If you ever flip Doubao back to default, do a 6-row smoke run on a real
worksheet (NOT the bench fixtures) before promoting. The fixture set
underspecifies the "many high-res evidences per row" tail that triggered
this.

---

## USPTO TSDR returns 403 to `linebase/0.1` User-Agent

Symptom: `fetch.py` downloads from `https://tsdr.uspto.gov/img/...` return
HTTP 403 when the requests library sends its default `python-requests/...`
or our previous `linebase/0.1` User-Agent. Same URL in a browser works.

Workaround: `linebase.fetch.fetch` pins a Chrome-class UA string:

```
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36
```

If you ever switch to httpx or aiohttp here, port the UA. TSDR also seems
to enforce per-IP rate limits in the low-double-digits per second — don't
add parallel fetches without an `asyncio.Semaphore`.

---

## SSE consumer must split on `\r\n\r\n`, not `\n\n`, on uvicorn

Symptom: `EventSource` on the browser side stalls — no `row_done` events
visible — even though the server is publishing into the queue.

Root cause: uvicorn's HTTP/1.1 implementation emits SSE frames with CRLF
line endings (`\r\n\r\n` between events) per RFC 8895 . A bare `\n\n`
splitter on the client side never sees the terminator. The
`EventSource` browser API gets it right transparently, but anything that
reads the SSE stream raw (test harnesses, the `e2e_real_xlsx.py` script)
must split on `\r\n\r\n`.

Workaround: in `scripts/e2e_real_xlsx.py` and anywhere we parse SSE bytes
manually, use `buffer.split(b"\r\n\r\n")` and treat the last fragment as
incomplete until the next read.

---

## FastAPI sync-def routes calling `asyncio.get_event_loop()` crash on the anyio threadpool

Symptom: A sync route handler tries to do
`loop = asyncio.get_event_loop(); loop.run_in_executor(...)` and gets:

```
RuntimeError: There is no current event loop in thread 'AnyIO worker thread-N'
```

Root cause: FastAPI runs `def`-route handlers on its anyio threadpool, not on
the main event loop. `asyncio.get_event_loop()` only succeeds on the main
thread or inside an `async def`. The historical "create a new loop if there
isn't one" behaviour was removed in Python 3.12.

Workaround:

- Prefer `async def` route handlers. Use `await asyncio.to_thread(blocking)`
  for blocking IO inside them.
- In `pipeline_runner._process_row` we deliberately call
  `asyncio.get_event_loop()` because that function only runs inside the
  background task created by `_run_job` — i.e. always on the main event
  loop. The route handlers themselves are all `async def`.
- If you really need to call into asyncio primitives from a sync route,
  use `anyio.from_thread.run` to dispatch back to the main loop.

---

## `_resolve_rows("first_n", n=2)` returns 4 rows, not 2

Pre-existing fudge in `linebase.pipeline_runner._resolve_rows`:
`end_row = 2 + n - 1 + 2`. The `+ 2` accounts for the two-row header in
`图形商标tro`. This is documented in `research/qwen3vl-system-validation.md`
and not a bug per se — but every "n=2" e2e test ends up touching 4 rows,
which surprises people. Don't "fix" it without checking what the UI's
`first_n` slider actually means to the user.

---

## Provider cost factors over/under-count by ~50× without adjustment

Symptom: `cost_usd` on a job is $1.20 but the real Ark spend was $0.024.

Root cause: `cost_estimate()` uses an OpenAI gpt-5 rate scalar as the base.
Doubao Seed 2.0 Pro is billed at ~$0.40 / M input + $1.20 / M output by
Volcengine — roughly 0.02× the gpt-5 rate. Qwen3-VL on SiliconFlow lands
in the same ballpark.

Workaround: `cost_estimate(usage, model=...)` multiplies by
`_PROVIDER_COST_FACTOR[provider_name]` once we know which provider the
model routes to. Adjustment table:

| provider | factor |
|---|---:|
| openai | 1.0 |
| ark | 0.02 |
| siliconflow | 0.02 |

Tune the factor down further if Volcengine drops Doubao prices — current
0.02 is calibrated against the May 2026 price card.
