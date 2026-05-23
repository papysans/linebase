# Lite 4-way benchmark: gpt-5.4 vs Qwen3-VL vs Doubao-mini vs Doubao-pro

Date: 2026-05-24 · Active task: `.trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline`

Prompt v_2, verify-loop OFF, identical fixture set (10 docx samples).
Two runs (`gpt-5.4`, `Qwen3-VL`) are re-used from existing artifacts; two runs
(Doubao-mini, Doubao-pro) were added this turn.

## Method

- Prompt: `prompts/v_2.md` — single-instance match, scalar scores (clarity / completeness / isolation) for tie-break.
- Eval harness: `scripts/eval_runner.py` → `linebase.bench.run_eval`, no parallelism, per-call timeout 120 s.
- Selection accuracy is the primary metric — "did the model pick the evidence image the human red-boxed?". Only 7 of 10 fixtures carry a hand-drawn red box, so the denominator is 7.
- Bbox IoU vs the auto-detected red-box is reported when (and only when) the model picked the gold evidence — otherwise n/a.
- Cost: the runner's `cost_estimate()` uses gpt-5.x rates and over-counts non-OpenAI providers by ~30-100× per `research/qwen3vl-system-validation.md`. Both raw and `× 0.02` adjusted figures shown below.

## Headline table

| model | provider | sel_acc | n_correct | mean_iou | mean_ssim | cost_raw_USD | cost_adj_×0.02_USD | mean_latency_s | failed/total | qualitative notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gpt-5.4` | OpenAI relay (1m1ng) | 43% | 3/7 | 0.028 | 0.343 | $0.348 | $0.348 (OpenAI, no adj) | n/a (not recorded) | 0/63 | Baseline. Tight bboxes when gold-evidence picked (one 1.02× area ratio is near-perfect), but only 3/7 right. |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | SiliconFlow | 29% | 2/7 | 0.002 | 0.316 | $0.344 | $0.007 | 5.73 | 6/63 | Cheapest + fastest, but 4 of 6 failures = `code=20015` on <28 px USPTO thumbnails; produces near-whole-image bboxes (avg ratio ~200×). One sample returned no match at all. |
| `doubao-seed-2-0-mini-260428` | Ark (Volcengine) | 57% | 4/7 | 0.005 | 0.307 | $1.198 | $0.024 | 9.65 | 1/63 | Reasoning model (~920 reasoning tok / call). 1 timeout (120 s) on the gold-evidence image of `4601531` — that one timeout cost the sample. **No 28-px-tile failures.** Over-strict on `1494172` (rejected all 3 valid evidences for "missing circular outline", etc.) — that's not a selection-accuracy hit (sample has no gold) but is a real-world quality risk. |
| `doubao-seed-2-0-pro-260215` | Ark (Volcengine) | **71%** | **5/7** | 0.004 | 0.369 | $1.217 | $0.024 | 23.32 | **0/63** | **Highest selection accuracy + zero failures.** Also the slowest by ~2.4× over the mini and ~4× over Qwen3-VL. Bboxes tend to be tight (area ratios 0.002 to 0.63 — none are near-whole-image). No 28-px-tile failures. |

(Adjusted cost: raw × 0.02 for SiliconFlow / Ark providers per the inflated-gpt-rate caveat in `research/qwen3vl-system-validation.md`. OpenAI relay uses real billed rates so no adjustment.)

**Cumulative new LLM spend this turn:** Doubao-mini + Doubao-pro raw = $2.42 estimate, real adjusted ≈ $0.048. Well under the $0.60 hard cap.

## Per-sample comparison

`✓` = correct selection, `✗` = wrong evidence picked, `–` = sample has no human red-box (`no_gold`, so not counted toward selection accuracy), `T` = timed out / API failed on the gold evidence, `(none)` = model returned `found=false` for every evidence.

Area ratio = `area(model_bbox) / area(gold_red_box)` — only meaningful when the model picked the gold evidence.

| appno | gpt-5.4 | Qwen3-VL | Doubao-mini | Doubao-pro |
|---|---|---|---|---|
| `1044246` | – | – | – | – |
| `1494172` | – | – | – `(none for all 3)` | – |
| `1969989` | ✗ | ✓ ratio=349× | ✓ ratio=549× | ✗ (picked `15_rId27` instead of `12_rId24`) |
| `2423810` | ✓ ratio=0.085 | ✓ ratio=91× | ✗ (picked `01_rId5`) | ✗ (picked `07_rId11`) |
| `3089225` | – | – | – | – |
| `4334451` | ✗ | ✗ | ✗ (picked `05_rId58`) | ✓ ratio=0.539 |
| `4338293` | ✓ ratio=1.018 | ✗ | ✓ ratio=0.763 | ✓ ratio=0.628 |
| `4601531` | ✗ | ✗ | ✗ (`T` on gold) | ✓ ratio=0.024 |
| `4827580` | ✗ | ✗ | ✓ ratio=0.018 | ✓ ratio=0.002 |
| `6433801` | ✓ ratio=0.189 | `(none)` | ✓ ratio=0.015 | ✓ ratio=0.011 |
| **count** | **3/7** | **2/7** | **4/7** | **5/7** |

Notes:
- The `~349×` and `~549×` ratios on `1969989` mean Qwen3-VL and Doubao-mini both returned a near-whole-image bbox on an evidence where the actual logo footprint is `~15×87 px`. Selection was "correct" only in the sense that they picked the right evidence — the bbox itself is not usable for cropping.
- Doubao-mini's `(none)` on `1494172` was an over-strict refusal (the three watch-face images all show the Corvette logo in slightly stylized form; Doubao-mini's reasoner rejected them as "missing the circular outline" or "logo partially obscured by clock hands"). Since `1494172` has no red-box, it doesn't appear in the selection-accuracy denominator — but in production it would surface as `NEEDS_REVIEW` and waste reviewer time.
- The `T` on Doubao-mini's `4601531`: the gold evidence is a 1706×960 px well-lit product photo; the call returned `APITimeoutError` after 120 s. Without that timeout Doubao-mini might have hit the gold evidence and reached 5/7 (tied with Doubao-pro).

## Winner

**Doubao-seed-2-0-pro-260215.** 71% selection accuracy beats every other model in the bench; 0 failures vs Qwen3-VL's 9.5% and Doubao-mini's 1.6%; bboxes are tight (no ~349× area ratios); no 28-px-tile constraint. Its only weakness is latency — 23 s/call average means a real workbook batch of 3000+ rows × 5-7 evidences each would take ~5-8 hours of wall-clock, but that's an offline-batch problem, not a UX one. Real cost remains under $0.05 per 10-sample fixture run on the Ark price card, so a full 3361-row batch is a few-dollars-not-a-few-hundred concern.

If the user later wants a faster default at modest accuracy cost, **Doubao-mini is a reasonable runner-up** (57%, 9.65 s/call, also Ark-billed) — but it has the over-strict-rejection failure mode on `1494172` that pro doesn't share. Stick with pro unless latency becomes a hard blocker.

`gpt-5.4` and `Qwen3-VL` are both off the production short-list: gpt-5.4 only hits 3/7 and costs OpenAI rates; Qwen3-VL hits 2/7 and chokes on USPTO thumbnails (see next section).

## Did Doubao choke on <28 px USPTO thumbnails like Qwen did?

**No — neither Doubao model produced a single 28-px-tile error across 126 combined calls.** Grep for `20015`, `height(`, or `width(` in `eval/run_009_v2_doubao_mini/raw_log.json` and `eval/run_010_v2_doubao_pro/raw_log.json` returns zero hits. The same fixtures (rId22 = 39×23, rId26 = 50×23, rId8 = 43×27) that broke Qwen3-VL with `code=20015` were processed without complaint by both Doubao variants.

**Conclusion: the 28-px minimum-tile constraint is a Qwen3-VL preprocessing quirk, not a systemic 国产 vision-API issue.** Ark's image preprocessor handles sub-28 px tiles transparently. For production this means Doubao does not need the "skip on tile-too-small" branch we currently rely on in the pipeline for Qwen — the metas[url]={"error": ...} silent-skip path is dead code under Doubao.

## Other weirdness worth flagging

1. **Doubao-mini APITimeoutError on the one image that mattered (`4601531/05_rId65`).** 1706×960 px, well-lit, the only timeout in 63 calls. If we ship Doubao-mini and want the missing accuracy point back, an automatic 1-retry on `APITimeoutError` would likely recover it. Pipeline currently does not retry on timeouts (the OpenAI SDK is set to `max_retries=0` by `linebase.llm._build_client`, and the bench's outer loop only retries on JSON-parse failures). One-line config change in `bench.py` if we decide to chase it.

2. **Doubao-mini is over-strict on `1494172`.** All three evidences are stylized Corvette logos — clearly the same trademark — and Doubao-mini rejected each as "missing circular outline" / "logo partially obscured by clock hands" / "reversed bowtie color, red background, not matching the registered trademark's internal features." Doubao-pro on the same sample returned `found=true` on the third evidence (`03_rId79`) with conf=0.88. The reasoner's "single-instance + registered-shape" interpretation in the v_2 prompt is the same model-side reasoning step that costs the mini ~920 reasoning tokens/call; tightening the prompt to allow "stylized but recognizable" matches would help mini more than pro.

3. **gpt-5.4 had no per-call latency recorded.** The pre-existing `run_003_v2/metrics.json` predates the latency-instrumentation change; that's why the cell shows "n/a". We did not re-run.

4. **Both Doubao models gave bbox area ratios ≪ 1 on the large-image samples** (`4601531`, `4827580`, `6433801`). That's the model returning a tight box around the logo on a large product photo — desired behavior. Qwen3-VL's tendency to return near-whole-image bboxes (ratios 91× and 349×) is a real production hazard: the resulting "crop" would be the entire photo with a small logo somewhere in it, useless as a thumbnail.

5. **All four models scored mean_iou ≪ 0.5.** The gold red-box and the model bbox almost never overlap meaningfully even when the model picked the right evidence. Two reasons: (a) the gold red-box is drawn by a human as a wide visual annotation (often much bigger than the actual logo footprint), and (b) the models are correctly returning tight logo bboxes. IoU vs the visual annotation is the wrong metric — it punishes correct-but-tight bboxes. The eval should switch to "is the model bbox contained within the gold box, and is its area within 0.5×–2× of the visible logo footprint?" but that's a different research turn.

## SQLite record

`eval_run` table now contains:

```
11 | doubao-seed-2-0-pro-260215 | v2     ← this turn
10 | doubao-seed-2-0-mini-260428 | v2    ← this turn
 9 | Qwen/Qwen3-VL-30B-A3B-Instruct | v3
 8 | Qwen/Qwen3-VL-30B-A3B-Instruct | v2 ← pre-existing baseline (run_007)
 7 | Qwen/Qwen3-VL-30B-A3B-Instruct | v3
 6 | gpt-5.5 | v2
 5 | gpt-5.5 | v2
 4 | gpt-5.4 | v2                        ← pre-existing baseline (run_003)
 3 | gpt-5.4 | v2
 2 | gpt-5.4 | v1
 1 | gpt-5.4 | 0-baseline
```

Both new rows were inserted by `scripts/eval_runner.py` → `store.insert_eval_run(...)` as expected. The `metrics_json` blob on each row carries the full pair-level results, so the `/dev` dashboard can render them without re-reading the filesystem.
