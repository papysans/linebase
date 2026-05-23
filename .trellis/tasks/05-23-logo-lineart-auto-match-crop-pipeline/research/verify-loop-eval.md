# Verify-Loop Eval — v_2 prompt + verify-loop ON vs prompt-only baselines

- Date: 2026-05-23
- Run id: `run_005_v2verify` (renamed from `eval/run_005_v2` produced by the eval_runner)
- Prompt: `prompts/v_2.md` for primary call, `prompts/verify_v_1.md` for verify call
- Verify orchestration: `src/linebase/verify_loop.py::match_with_verify` (one padded-crop verify call per Pass-1 positive, threshold 0.6)
- Fixture set: 10 docx samples; 7 with red-box gold

## 1) 4-way numeric table

| metric                     | run_001 (v0)  | run_002_v1 | run_003_v2 | run_005_v2verify |
| ---                        | ---           | ---        | ---        | ---              |
| samples                    | 10            | 10         | 10         | 10               |
| matched                    | 10            | 10         | 10         | 10               |
| selection_evaluated        | n/a           | 7          | 7          | 7                |
| correct_selection          | n/a           | 2          | **3**      | **2**            |
| selection_accuracy         | n/a           | 28.6 %     | **42.9 %** | **28.6 %**       |
| bbox_iou_scored            | n/a           | n/a        | 3          | 2                |
| bbox_iou_mean              | n/a           | n/a        | 0.028      | 0.000            |
| bbox_iou_pass_50           | n/a           | n/a        | 0 %        | 0 %              |
| mean_ssim (secondary)      | 0.328         | 0.352      | 0.343      | 0.342            |
| pass@SSIM ≥ 0.5            | 20 %          | 20 %       | 20 %       | 20 %             |
| total cost (USD est.)      | $0.216        | $0.270     | $0.348     | $0.496           |
| verify_calls               | —             | —          | —          | 45               |
| verified_true              | —             | —          | —          | 22               |
| verified_false             | —             | —          | —          | 23               |
| verify_rejected_to_NR      | —             | —          | —          | 23               |
| verify_cost_usd_estimate   | —             | —          | —          | $0.149           |

Headline: **verify mode regressed selection accuracy 43 % → 29 %** at +43 % cost ($0.348 → $0.496). One v_2 correct (sample `2423810`) was wrongly downgraded by verify. None of the three v_2 wrong picks was flipped to correct by the verify-driven re-ranking. SSIM is flat (within noise).

## 2) Per-sample classification

Definitions (per spec):
- **STILL_CORRECT_AND_VERIFIED** — was correct in v_2, verify confirmed.
- **STILL_CORRECT_BUT_DOWNGRADED** — was correct in v_2, verify rejected (regression).
- **WAS_WRONG_NOW_DOWNGRADED** — was wrong in v_2, verify rejected the wrong pick AND the pipeline ended up at `(none)` / needs_review (no positives left after verify).
- **WAS_WRONG_NOW_FLIPPED_CORRECT** — was wrong in v_2, verify re-ranking picked the gold.
- **STILL_WRONG_VERIFIED_ANYWAY** — was wrong in v_2, verify confirmed the wrong pick.
- **STILL_WRONG_DIFFERENT_WRONG** — was wrong in v_2, verify shuffled to a different wrong pick (worse than just confirming, because it spent extra calls without product gain).
- **NEW_FAILURE_MODE** — anything else.
- **n/a (no gold)** — sample has no red-box gold; selection accuracy not measured.

| sample   | v_2 pick           | gold              | v2verify pick      | classification                                                                                                  |
| ---      | ---                | ---               | ---                | ---                                                                                                              |
| 1044246  | 01 (no_gold)       | —                 | 01 (verify=OK loose) | n/a (no gold). Verify confirmed v_2's pick. 02 and 03 rejected.                                                |
| 1494172  | 03 (no_gold)       | —                 | 01 (verify=OK loose) | n/a (no gold). Verify rejected v_2's 03 as `wrong`; pipeline fell back to 01 — a *different* evidence.        |
| 1969989  | 04 (wrong)         | 12                | 02 (wrong, expanded) | **STILL_WRONG_DIFFERENT_WRONG.** Verify rejected v_2's 04 and the gold 12 (both `wrong`); accepted 02 as `too_tight` and expanded its bbox; composite picked the expanded 02. Gold rejected as wrong. |
| 2423810  | 02 (correct)       | 02                | 07 (wrong, OK loose) | **STILL_CORRECT_BUT_DOWNGRADED.** Verify rejected gold 02 as `wrong` (verify_confidence 0.99). Pipeline fell to 07, which verify confirmed. *Direct regression caused by verify.* |
| 3089225  | 04 (no_gold)       | —                 | 02 (verify=OK tight) | n/a (no gold). Verify rejected v_2's 04 and the v_2 #2 (05). Fell to 02.                                       |
| 4334451  | 04 (wrong)         | 03                | 04 (verify=OK tight) | **STILL_WRONG_VERIFIED_ANYWAY.** Verify accepted 04 AND the gold 03 (`OK/loose`), 05 (`OK/loose`), 01 (`OK/tight`). Composite still ranks 04 highest. Verify did not help disambiguate at all here. |
| 4338293  | 06 (correct)       | 06                | 06 (verify=OK loose) | **STILL_CORRECT_AND_VERIFIED.** Clean case.                                                                    |
| 4601531  | 04 (wrong)         | 05                | 02 (wrong, OK tight) | **STILL_WRONG_DIFFERENT_WRONG.** Verify rejected v_2's 04 as `wrong`; accepted 02, 03 (→ too_tight, expanded), 05 (gold), 01. Composite picks 02. Gold was verified OK/tight but ranked below 02. |
| 4827580  | 02 (wrong)         | 07                | 04 (wrong, OK tight) | **STILL_WRONG_DIFFERENT_WRONG.** Verify rejected v_2's 02 AND the gold 07 as `wrong`. Accepted 01, 03, 04. Composite picks 04. |
| 6433801  | 01 (correct, single ev) | 01            | 01 (verify=OK loose) | **STILL_CORRECT_AND_VERIFIED.** Single-evidence sample.                                                       |

### Counts (over the 7 samples with gold)

- STILL_CORRECT_AND_VERIFIED: **2** (`4338293`, `6433801`)
- STILL_CORRECT_BUT_DOWNGRADED: **1** (`2423810`)  ← active regression
- WAS_WRONG_NOW_DOWNGRADED: 0
- WAS_WRONG_NOW_FLIPPED_CORRECT: 0
- STILL_WRONG_VERIFIED_ANYWAY: **1** (`4334451`)
- STILL_WRONG_DIFFERENT_WRONG: **3** (`1969989`, `4601531`, `4827580`)
- NEW_FAILURE_MODE: 0

### Special attention to the three v_2 misses

- **`1969989` NBA**: did verify catch it? **Verify *correctly rejected* v_2's wrong pick (04)** as `wrong`. But verify *also* rejected the gold image 12 as `wrong` — the model thinks the tiny strip-thumbnail does not contain the registered NBA logo. So verify did not produce a fix; it just shuffled the wrong-pick. Note: this matches the v2-eval observation that the gold here is a "sloppy thumbnail strip" and the prompt is doing what it was told to.
- **`4334451` earmuffs / badge**: did verify catch it? **No** — verify *confirmed* v_2's wrong pick (04, `OK/tight`, conf 0.99). Verify also accepted the gold 03 as `OK/loose`. So both are "valid" by verify; composite-score still ranks 04 above 03. Verify does not solve the "isolated TM vs TM on the product" ambiguity from v2-eval.
- **`4601531` silverware**: did verify catch it? **Partial** — verify rejected v_2's exact pick (04 as `wrong`), but the new pick (02) is still wrong. The gold (05) was verified as `OK/tight` but ranked below 02 by composite.

## 3) Verify-loop behaviour summary

- 45 verify calls fired (every Pass-1 positive with confidence ≥ 0.4 triggers one).
- 22 verifies returned OK, 23 returned NO (`wrong` / low-conf reject / `contains_full_logo=false`).
- All 23 NOs led to that evidence being dropped from the positives ranking; 0 evidences were rescued by the `too_tight` expansion path (well, actually `too_tight` did fire on 4 evidences across `1969989/4338293/4601531`, all of which expanded the bbox and kept the evidence — but in only one case did the expanded-bbox evidence end up being the final pick: `1969989` 02, which was still wrong).
- The verify model is **strict** — it routinely says `wrong` on legitimate matches (gold 12 in `1969989`, gold 02 in `2423810`, gold 07 in `4827580`). This is the dominant failure mode of the verify pass: false negatives that delete a correct evidence from contention.
- It's also **inconsistent across same-product duplicates** — for `4334451` it accepted 4 of 5 evidence images (all showing the same badge in different framings), validating its instinct correctly there but not helping picking.

## 4) Why verify doesn't fix the selection problem

The SOTA-prompt research file (`research/sota-prompt-techniques.md` Finding 4) was right: **per-image self-verification can't disambiguate among many high-confidence positives** because each verify call sees only one (logo, crop) pair and cannot compare candidates. When 4–5 evidences each contain a valid instance of the trademark (very common in our data — products are photographed multiple times), the verify pass either:

1. Confirms all/most of them (`4334451`) → composite ranking still decides → same outcome.
2. Rejects most of them on cosmetic grounds (`4827580`, `4601531`) → ranking among the survivors is essentially random.
3. Rejects the gold along with the wrong picks (`1969989`, `2423810`, `4827580` — 3 of 7 gold images failed verify!) → the gold is removed from contention and a worse evidence wins.

(3) is the fatal class — verify *removes* correct rows. That's why we went from 3/7 → 2/7 correct.

The bbox-tightening behaviour (`too_tight` → expand, `loose` → suggested_bbox shrink) did execute correctly in 7 cases. But the bbox-quality lift it produces is dwarfed by the selection-accuracy regression: SSIM mean shifted from 0.343 → 0.342 (noise).

## 5) Decision — concrete ship-recommendation

**Hold and build SoM grid selection (option B from the spec).**

Reasoning:

1. **Verify-only does not move the needle and actively regresses selection** (43 % → 29 % on the gold set). The one correct case it broke (`2423810`) is exactly the case v_2 paid for over v_1 — we'd be undoing our previous win.
2. **The dominant failure mode is now visible**: when multiple evidence images contain valid trademark instances, *no per-image verify can pick the right one*. The decision needs to be cross-image, which means a Set-of-Mark-style "all candidates in one prompt → pick one number" — exactly the lift SoM gives in the literature (Yang et al. 2023, Finding 1 of the SOTA research file).
3. **Bbox-quality is currently a secondary concern**: red-box IoU gold is itself loose (Finding from v2-eval) so it can't yet give us a real bbox-quality signal. Spending $0.15/run to "tighten" bboxes against a noisy gold is not a good trade.
4. **Cost ratio is bad**: +43 % cost, -33 % accuracy. No version of "tune the verify threshold" or "tweak the verify prompt" recovers the gold-rejection failure mode — that's a per-image-context limitation, not a prompt-tuning problem.

### Concrete next step

Skip "ship v_2 + verify" and "mixed". Build the SoM stage. v3 architecture should be:

- **Stage A (cheap CV / template-match / SAM)**: pre-extract N candidate regions across all evidence images for a row.
- **Stage B (one LLM call)**: render the candidates onto a single composite image with numeric overlays, ask GPT-5 *"Which numbered region is the best evidence for this registered trademark? Reply with just the number, or 'none'."*
- **Stage C (deterministic geometry)**: take the selected region, tighten via OpenCV contour/edge on the picked sub-image.
- **Stage D (optional verify, only on close calls)**: verify the *chosen* crop, with a much higher reject threshold than 0.6 — and only as a "should this row go to needs_review" gate, not as a re-ranking signal.

Until SoM is built, **ship v_2 prompt-only (run_003_v2)** as the production baseline. It's the best version we have (43 % selection vs 29 % for both v1 and v2verify).

## 6) Cost log (this turn)

- run_005_v2verify: $0.496 (primary $0.347 + verify $0.149). 65 primary + 45 verify = 110 LLM calls.
- Cumulative dev-loop spend (v0+v1+v2+v3+v2verify): ≈ $1.85. Still well under the $5 stop budget.
