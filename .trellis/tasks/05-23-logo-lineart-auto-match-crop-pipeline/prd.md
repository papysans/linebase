# Logo-Lineart Auto Match & Crop Pipeline

## Goal

Build a **local web application** (frontend + backend) that lets the user upload a USPTO trademark workbook, pick the relevant sheet/columns, run an LLM-driven logo-vs-real-photo matcher row-by-row, review the results in-browser, mark bad rows, re-run only those, and finally download a new XLSX + a folder of cropped images named by application number.

Under the hood the matching is a tuned **multimodal-LLM-gives-bbox + OpenCV-crop** pipeline, developed first as an autonomous self-iterating dev loop against a held-out ground-truth set (the 10 docx samples), so quality is measured, not assumed, before any real batch is run.

## What I already know

### Input data
- **Workbook**: `D:/Project/linebase/美专实物图排查-2026.2.6.xlsx` (428 MB).
- **Target sheet**: `图形商标tro` (sheet5, 3361 data rows; rows 1-2 are headers/hidden).
- Columns in the target sheet:
  - **B** `申请号` — USPTO application number (e.g. `75537343`) → output filename prefix.
  - **D** `图形商标logo` — a **single URL** to the line-art logo (e.g. `https://tsdr.uspto.gov/img/<appno>/large`).
  - **K** `使用证据` — **comma-separated URLs** of real-product photos (USPTO `casedoc` endpoints).
- The 5078 embedded images in `xl/cellimages.xml` belong to *other* sheets — D/K in `图形商标tro` are plain URL strings.

### Ground truth (evaluation set)
- `D:/Project/linebase/商标去噪音图期望检测效果.docx` contains **10 hand-curated samples**. Per sample: (appno, class, 1 LOGO image, several evidence images — one annotated with a red bounding box marking the human-judged best match, and 1 "expected crop" image).
- Extracted to `fixtures/sample_<appno>/` (10 directories) with `_manifest.json` recording the rId order. Image counts per sample range 3-21.
- Visual inspection of `sample_6433801` confirmed: image 0 = logo line-art (Corvette flags), image 1 = evidence (windshield with small logo visible bottom-left), image 2 = expected crop (the logo region cropped out, light grey background).

### LLM service
- OpenAI-compatible relay at `https://api.1m1ng.net` ("1m1ng"), model `gpt-5.4`, wire API `responses`. Credentials in `D:/Project/linebase/.env` (gitignored).
- Standalone testing-purpose endpoint provided by the user; will be probed before any batch run.

### Project state
- Python 3.11.9, `openpyxl 3.1.5` available system-wide. No `.venv`, no source code yet, no `pyproject.toml`. Empty repo aside from Trellis scaffolding.

## Decision (ADR-lite)

### Decision 1 — Algorithm route
- **Decision**: Pure multimodal LLM returns a bounding box; OpenCV crops it; an optional self-verify loop sends the crop back for confirmation.
- **Context**: Real photos have heavy background clutter (windshield reflections, fabric textures, packaging), arbitrary angles, and logo-as-decal scenarios. Pure classical CV (template / SIFT / contour) is brittle here. The 10-sample fixture set is small enough to iterate cheaply with an LLM.
- **Consequences**: Per-row API cost (≈ $0.01-0.03 with vision input). Stability depends on prompt + model behavior — must be measured, not assumed. Verify-loop adds latency and cost; only enable if baseline fails on edge cases.

### Decision 2 — LLM provider
- **Decision**: `gpt-5.4` via the 1m1ng OpenAI-compatible relay; SDK = `openai` Python package with `base_url` override.
- **Context**: User explicitly provided this endpoint for the project. Cost reportedly lower than Anthropic direct; network reachable from Windows host.
- **Consequences**: Must verify model id and image-input compatibility before kicking off. Selection among other gpt-N variants exposed by the relay is at my discretion if gpt-5.4 misbehaves.

### Decision 3 — Dev workflow = sampling-first autonomous loop, NOT one-shot batch
- **Decision**: Build an autonomous tuning loop (sample → eval → tune → re-run) against the docx fixtures. Production batch over the 3361-row Excel is gated by the user's explicit approval after the dev loop converges.
- **Context**: User's directive: "我们不用全部调用，我们只需要抽样，然后确保稳定就行 ... 把开发过程写成一个你可以长时间自迭代的 loop ... 我不用过多的去干涉开发"
- **Consequences**: First-deliverable is a measurable dev harness + tuned prompt/algo, not a finished batch processor. Significant cost-control wins (no 3000-row burns on a bad prompt). Forces an honest eval metric instead of vibes-based "looks good".

### Decision 4 — Delivery form = local web app (frontend + backend)
- **Decision**: Single-process FastAPI server that serves a built React SPA + REST API + SSE progress stream. User starts it with `linebase serve`, opens `http://localhost:8000`.
- **Context**: User requirement (2026-05-23): "最终我们的形式我需要做一个网页，有完整的前端和后端". CLI is no longer the primary surface.
- **Tech choices** (final, not asking):
  - Backend: FastAPI + uvicorn + Pydantic v2; SQLite for job/row state (single file `.data/linebase.db`).
  - Frontend: Vite + React 18 + TypeScript + Tailwind + shadcn/ui + TanStack Query.
  - Progress: Server-Sent Events (one-way stream, simpler than WebSocket, suffices for progress + log lines).
  - File storage: filesystem under `.data/uploads/<job_id>/` and `.data/runs/<job_id>/`. SQLite stores metadata + status + per-row result paths.
- **Output**: same artifact as before — a new XLSX + a folder of named images — but the user gets them via the browser (download buttons + ZIP export). The 428 MB source is never modified; uploads land in `.data/uploads/` and are read-only thereafter.

### Decision 5 — Web UX (defaults I picked without asking)
- **Upload page**: drop an XLSX, preview sheet list + first 5 rows, pick the sheet, pick the LOGO-URL column and the EVIDENCE-URL column (auto-detect URLs to suggest defaults, e.g. column D + column K).
- **Sample-run page**: choose N rows (start/end range or first-N) and a confidence threshold; "Start run" begins processing; SSE feeds a live table.
- **Review page**: paginated table per row with logo, all evidence images, the chosen crop, and OK/BAD/NEEDS_REVIEW toggles. Bulk re-run on selected rows.
- **Download page**: download the result XLSX or a ZIP of the images folder (filtered by status / by selected rows).
- **Dev loop dashboard** (separate `/dev` page): shows eval runs, prompt versions, baseline metrics. This is the autonomous-loop's reporting surface — present even before any real Excel job is created.

### Decision 6 — K-column multi-URL handling (defaults I picked without asking)
- Every URL is sent to the LLM. The LLM returns `{found, bbox, confidence}` for each. Crops are produced only when `found && confidence >= threshold` (initial threshold = 0.5, tunable).
- The crop with the highest confidence is embedded in column M as the "main" preview; the rest live in `out/images/` for the human to inspect.
- If 0 evidence images yield a match, the row is marked `NEEDS_REVIEW` (not `BAD` — distinguishes "model unsure" from "model wrong").

## Requirements

### Core pipeline (callable from FastAPI background tasks)
- Parse the uploaded XLSX → iterate selected rows → extract appno + logo URL + evidence URL list.
- Download every URL to a content-addressed local cache (`.data/cache/<sha256>.<ext>`) so re-runs are network-free.
- For each `(logo, evidence)` pair: call LLM → parse JSON bbox → PIL crop → save with naming convention.
- Write a new XLSX with status + crop columns. Don't touch the uploaded source.
- "Re-run on selected rows" mode for bad/needs-review rows.

### Backend API (FastAPI)
- `POST /api/uploads` — multipart upload of XLSX → returns `{job_id, sheets:[{name, rows, columns:[…]}]}`.
- `POST /api/jobs` — create a job with `{upload_id, sheet, logo_column, evidence_column, appno_column, sample: {kind:"first-n"|"range"|"row-ids", …}, threshold}`.
- `POST /api/jobs/{id}/start` — kick off background task.
- `GET /api/jobs/{id}/events` — SSE stream of `{type:"row_done"|"row_failed"|"progress"|"finished", …}`.
- `GET /api/jobs/{id}` — current state + per-row results.
- `POST /api/jobs/{id}/rows/{row_id}/status` — mark OK/BAD/NEEDS_REVIEW.
- `POST /api/jobs/{id}/rerun` — re-run on rows where `status != OK` (or a specific selection).
- `GET /api/jobs/{id}/xlsx` — download result XLSX.
- `GET /api/jobs/{id}/images.zip` — download images folder (optional `?status=OK&rows=…` filter).
- `GET /api/dev/eval-runs` + `GET /api/dev/eval-runs/{id}` — eval-harness results for the dev-loop dashboard.

### Autonomous dev loop (the primary deliverable for the research phase)
- Use the 10 docx samples as the ground-truth fixture set.
- Evaluation metric: per-sample image similarity (pHash + SSIM, plus a human-readable HTML side-by-side report) between my crop and the docx "expected crop".
- Loop: run → score → diff vs previous version → tune prompt / threshold / verify-loop → run again. Persist each iteration's prompt + scores under `prompts/v_<n>.md` and `eval/run_<n>/`.
- Stop condition: pass rate plateau for 2 consecutive rounds, OR cumulative API cost ≥ $5, OR I detect a problem class I can't fix without user input — then escalate with a single concrete question.
- Report format: numbers first (sample N, pass rate, mean similarity, $ cost, what changed in this round), then 1-line takeaway. The dev-loop runs as a CLI command initially and surfaces its data through `/api/dev/eval-runs` on the web side.

### Autonomous dev loop (the primary deliverable for the research phase)
- Use the 10 docx samples as the ground-truth fixture set.
- Evaluation metric: per-sample image similarity (pHash + SSIM, plus a human-readable HTML side-by-side report) between my crop and the docx "expected crop".
- Loop: run → score → diff vs previous version → tune prompt / threshold / verify-loop → run again. Persist each iteration's prompt + scores under `prompts/v_<n>.md` and `eval/run_<n>/`.
- Stop condition: pass rate plateau for 2 consecutive rounds, OR cumulative API cost ≥ $5, OR I detect a problem class I can't fix without user input — then escalate with a single concrete question.
- Report format: numbers first (sample N, pass rate, mean similarity, $ cost, what changed in this round), then 1-line takeaway.

## Acceptance Criteria

### Phase 1 — Dev loop deliverable (no user approval needed; I drive)
- [ ] `pyproject.toml` + `.venv` working on Windows; deps installed.
- [ ] LLM probe script runs successfully against gpt-5.4 + image input.
- [ ] Eval harness compares my crop vs docx expected crop and outputs an HTML report + JSON metrics.
- [ ] v1 of the matcher runs on all 10 fixtures and produces a baseline report.
- [ ] Documented tuning rounds (≥ 2) showing measurable change in pass rate.

### Phase 2 — Web app MVP (gated by phase-1 baseline reaching a workable pass rate)
- [ ] FastAPI server runs locally via `linebase serve`, exposes the API in `## Backend API`.
- [ ] Vite + React SPA: upload page, configure page, run page (SSE progress), review page, download page.
- [ ] Upload + parse XLSX, preview sheets/columns, auto-detect URL columns.
- [ ] Sampled run end-to-end through the browser → results visible in review page.
- [ ] OK/BAD/NEEDS_REVIEW marking + re-run-on-selected works.
- [ ] Download XLSX + image ZIP from the browser.

### Phase 3 — Full batch + Dev dashboard polish (gated by user approval)
- [ ] Stable on all 3361 rows with crash-resume and SSE checkpoints.
- [ ] `/dev` dashboard surfaces eval-run history + prompt versions.

## Definition of Done

- Unit tests on (a) URL/image cache, (b) bbox JSON parser, (c) crop coordinate clamping.
- Smoke test that calls a mocked LLM end-to-end so CI doesn't hit the network.
- Lint + typecheck green (`ruff`, `mypy`).
- README explaining: install, how to run the dev loop, how to run a sampled production batch, where the LLM key lives.
- Crash-resume notes (where cache lives, how to nuke partial runs).
- Cost log: every batch logs `# rows × # LLM calls × $ spent`.

## Out of Scope (explicit)

- Processing sheets other than `图形商标tro`.
- **Auth, multi-tenant, deployment to a public host.** App is a single-user local tool; no login, no HTTPS.
- Background-removing the crops (output is rectangular bbox crop, not masked).
- Modifying or re-writing the source workbook.
- Auto-fixing the 5078 embedded images in cellimages.xml (they belong to other sheets and are irrelevant here).
- Detecting/handling rows where D or K is missing or unreachable (will mark `NEEDS_REVIEW`, not retry forever).

## Technical Approach

### Module layout (proposed)

```
src/linebase/                  # python backend
  config.py                    # env loader
  fetch.py                     # URL → .data/cache/<sha>.<ext>
  llm.py                       # OpenAI-compatible vision client + JSON-bbox parser
  crop.py                      # PIL bbox crop with clamping
  io_excel.py                  # parse uploaded xlsx, write result xlsx with embedded images
  pipeline.py                  # row → matcher × N evidence → per-row outputs
  eval.py                      # pHash/SSIM scorer + HTML report (used by dev loop)
  store.py                     # SQLite layer for jobs/rows/eval runs
  pipeline_runner.py           # FastAPI BackgroundTasks orchestration + SSE event publisher
  server.py                    # FastAPI app + routes (mounts /api and static /assets)
  cli.py                       # `linebase serve` / `linebase eval` / `linebase tune`
frontend/                      # Vite + React TS SPA
  src/
    pages/{Upload,Configure,Run,Review,Download,Dev}.tsx
    components/, hooks/, lib/
  vite.config.ts
  tailwind.config.ts
scripts/probe_llm.py           # one-off probe (already exists)
fixtures/sample_*/             # docx ground-truth samples (already extracted)
prompts/v_<n>.md               # prompt revisions
eval/run_<n>/                  # per-iteration eval outputs
.data/{uploads,cache,runs,linebase.db}    # runtime state (gitignored)
.env                           # OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
```

### LLM prompt v0 (starting point — will iterate)

```
You are given two images. Image 1 is a line-art trademark logo (black on white).
Image 2 is a real product photograph that may or may not contain that logo
(printed, embossed, embroidered, or as a decal — possibly small, rotated, occluded,
or low-contrast).

Return strict JSON:
{
  "found": bool,
  "bbox": [x1, y1, x2, y2] | null,    // pixel coords in Image 2, 0-indexed
  "confidence": 0.0 - 1.0,
  "reason": "<one short sentence>"
}

If the logo appears multiple times, return the most prominent instance.
If unsure, set "found": false and confidence near 0.
Do NOT include any text outside the JSON.
```

(v1+ will refine: image-size hint, coordinate-system clarification, calibration sanity-check.)

### Self-verify loop (deferred to v2+, only if baseline fails)
1. Get bbox from v1.
2. Crop a 20%-padded version of the bbox.
3. Send (logo, cropped-region) → "does this crop contain the logo? if not, where in this crop is it?" → adjust bbox or reject.

## Research References

- (to populate during dev) `research/llm-probe.md`, `research/usp-tsdr-endpoints.md`, `research/eval-metrics.md`.

## Technical Notes

- USPTO TSDR endpoints return images directly via HTTPS without auth.
- WPS-flavored cellImages in the source workbook are irrelevant to this sheet — verified.
- The 10-fixture set is small; budget per dev-loop iteration ≈ 80 LLM calls × ~$0.02 ≈ $1.6 — affordable for many rounds.
- Re-using the existing `openpyxl` install is fine for read; for write-with-embedded-image we will install fresh into `.venv`.
- Crash-resume: a JSON sidecar per row in `out/_state/<appno>.json` records "downloaded / matched / written" so a re-run skips finished work.
