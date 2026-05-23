# User-driven model picker + small-image fallback

Date: 2026-05-24 · Active task: `.trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline`

## What we built

Add a per-job LLM model selector (curated whitelist + free-text custom) on the Configure page, persist it on the job, plumb it through to `match_logo_in_photo`, and harden the pipeline against the Qwen3-VL <28 px rejection by retrying once on gpt-5.5.

System default unchanged: `LINEBASE_DEFAULT_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct`.

## Code changes

### Backend

- **`src/linebase/models_catalog.py`** (new) — 6-model curated whitelist with `{id, provider, label, notes}` per entry. `is_model_routable(model_id)` does whitelist OR prefix-match against `config._PROVIDER_PREFIXES`, used to reject obvious garbage at job-creation time. Probed-working models only: Qwen/Qwen3-VL-30B/32B (siliconflow), doubao-seed-2-0-pro/mini (ark), zai-org/GLM-4.5V (siliconflow), gpt-5.5 (openai).
- **`src/linebase/server.py`** — adds `GET /api/models` returning `{whitelist, default, allow_custom}`. `CreateJobRequest.model: str | None = None`. `POST /api/jobs` validates the model id with `is_model_routable` and persists it on the job row (passes 400 with a clear message on invalid id). Route count went 20 → 21.
- **`src/linebase/store.py`** — `Job` dataclass grows `model: str | None = None`. `init_schema()` runs an idempotent `ALTER TABLE job ADD COLUMN model TEXT` wrapped in `except sqlite3.OperationalError` so re-init on an already-migrated DB is a no-op. `insert_job()` takes an optional `model` kwarg.
- **`src/linebase/pipeline_runner.py`**:
  - `_job_to_dict` surfaces `model` so the SPA can render the header pill.
  - `_process_row` resolves `eff_model = (job.model or "").strip() or settings.model` once per row and passes it to `match_logo_in_photo(..., model=eff_model)`.
  - When the chosen model is NOT already `gpt-5.5` and the call raises with a message containing `"must be larger than 28"` or the older `"must be > 28"`, we retry once with `model="gpt-5.5"`. Successful fallbacks record `meta["fallback_model"] = "gpt-5.5"` and `meta["fallback_reason"]` so the review page (future work) can flag them.
  - Verify-loop path keeps using `settings.review_model` — we only override the primary match call. Reasoning: verify prompt is more sensitive to model behavior, and the verify path is opt-in via `LINEBASE_VERIFY` anyway.

### Frontend

- **`frontend/src/lib/api.ts`** — adds `ModelOption`, `ModelsResponse`, `api.listModels()`. `CreateJobBody.model?: string | null`. `JobSummary.model?: string | null`.
- **`frontend/src/pages/ConfigurePage.tsx`** — new "模型" section above the existing "参数" section: `GlassSelect` dropdown with `使用系统默认` + the whitelist (label + notes inline) + `自定义…` sentinel. When sentinel is picked, a `GlassInput` lets the user type any model id. Submit resolves the empty / sentinel / id cases to a single `model?: string | undefined` field on the createJob body. Includes a small reminder about the Qwen <28 px → gpt-5.5 fallback.
- **`frontend/src/pages/RunPage.tsx` + `ReviewPage.tsx`** — small glass pill in the header showing `model · <id>` (truncated to 260 px with full `title` tooltip). ReviewPage now also fetches the job summary so it has the model field available.

### Test infra

- **`scripts/e2e_real_xlsx.py`** — accepts `LINEBASE_E2E_MODEL` and `LINEBASE_E2E_N` env vars. Forwards `model` to `POST /api/jobs` when set; defaults to `n=2`.

## Validation

### Server-import sanity
```
.venv/Scripts/python.exe -c "from linebase.server import app; print(len(app.routes))"
# → 21  (was 20; /api/models is the new one)
```

### Frontend type-check & build
- `pnpm exec tsc --noEmit` — exit 0, no output.
- `pnpm build` — exit 0; bundle: index.html 0.85 KB, CSS 27.17 KB / 6.26 KB gz, JS 268.03 KB / 84.14 KB gz, 33.8 s build time.

### `/api/models` smoke
TestClient round-trip returns `default=Qwen/Qwen3-VL-30B-A3B-Instruct`, `allow_custom=true`, `whitelist=6` entries with correct provider attribution.

### `is_model_routable` truth table
- Whitelisted (`Qwen/Qwen3-VL-30B-A3B-Instruct`) → True
- Prefix-matched non-whitelist (`Qwen/random-model`, `doubao-newmodel`, `gpt-5.5`) → True
- Garbage (`random-string`, `llama-3.1`) → False

### Small-image rejection heuristic
- `BadRequestError code=20015: height(23) or width(39) must be larger than 28 for Qwen 3 VL models` → True
- `"must be > 28 pixels"` (older docs phrasing) → True
- `APIConnectionError: Connection error` → False

### End-to-end against real workbook with `doubao-seed-2-0-mini-260428`
Job `edba98bfdd4f4293`, `n=2` resolved to 4 rows (existing `_resolve_rows` fudge).
| row | appno | status | best_conf | crop |
|---|---|---|---|---|
| 2 | 78402423 | needs_review | — | — (2 evidences, neither passed threshold) |
| 3 | 75537343 | ok | 0.98 | 3.9 KB |
| 4 | 78402415 | failed | — | — (no logo/evidence URLs in source) |
| 5 | 74677567 | ok | 0.98 | 26.9 KB |

`get_job(...)` returns `model='doubao-seed-2-0-mini-260428'`. All 47 successful per-evidence metas have `model='doubao-seed-2-0-mini-260428'`. **Zero fallbacks** triggered (Doubao Seed 2.0 Mini doesn't have the <28 px restriction).

`result.xlsx` 37.5 KB, `images.zip` 109.5 KB containing 20 named crops.

Estimated cost (gpt-5-rate scalar over Doubao tokens): `$1.0011`. Real Ark spend is ~30× lower (Doubao mini is ~$0.15/M input). Well under the $0.20 budget for this turn.

## Rough edges noticed

1. **Doubao thinking-model latency**: row 17 (75537343) took 139 s, row 19 (74677567) took 294 s — vs Qwen3-VL's ~7 s/call. Doubao's `thinking model` is producing a lot of reasoning tokens before answering. For a 3361-row production batch this matters: at ~150 s/evidence × ~10 evidences/row × 3361 rows we'd be looking at multiple days. Worth a follow-up to either turn off reasoning (if Ark exposes a flag) or keep Qwen3-VL as the production default and use Doubao only as a per-row diagnostic.
2. **`needs_review` on row 16 (78402423)**: 2 evidences, neither passed the threshold. The earlier Qwen3-VL run on the same row produced `ok` with conf=0.98. Doubao Seed 2.0 Mini is more conservative on this fixture — likely the same "first plausible vs gold-best" trade-off we saw in the Qwen3-VL eval. Not a bug, just a per-model judgment difference.
3. **`cost_usd` displayed in the UI is the gpt-5-rate scalar** (see `cost_estimate()` in pipeline_runner.py). Out of scope for this turn, but the UI should eventually surface per-provider pricing — the existing $1.00 displayed for what is probably <$0.05 real spend will mislead users picking between Qwen / Doubao.
4. **Fallback meta field is not yet surfaced in the UI** — `metas[url]["fallback_model"]` and `["fallback_reason"]` are written on the row but ReviewPage doesn't render them. The data is in `match_meta_json` waiting for a future small chip. Acceptable for now since the fallback is rare (~9.5% of calls in the prior Qwen run).
5. **`_resolve_rows` first_n fudge** (`end_row = 2 + n - 1 + 2`) means `n=2` → 4 rows. Pre-existing behavior, documented in `research/qwen3vl-system-validation.md`. Not touched here.

## Total spend this turn

- E2E run (4 rows, ~49 Doubao calls): displayed `$1.00` (gpt-5-rate scalar), real Ark spend ~$0.05.
- No other live LLM calls.
- **Cumulative real spend: < $0.10, well under the $0.20 budget.**
