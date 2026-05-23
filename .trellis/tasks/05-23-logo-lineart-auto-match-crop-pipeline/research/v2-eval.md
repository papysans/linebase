# v2 Eval — prompts/v_2.md with calibrated clarity/completeness/isolation scalars

Date: 2026-05-23
Run id: `run_003_v2`
Prompt: `prompts/v_2.md` (single-instance rule + four-scalar scoring + 6-band confidence calibration)
Model: `gpt-5.4` via 1m1ng relay
Fixture set: 10 docx samples; 7 of them have a hand-drawn red-box ground-truth evidence pick.

## Probe result (sample_6433801, sanity check)

```
{"found":true,
 "bbox":[347,652,435,740],
 "confidence":0.91,
 "clarity":0.72,
 "completeness":0.96,
 "isolation":0.71,
 "reason":"The Corvette cross-flags logo is clearly visible as branding on the windshield."}
```

All three new scalars came back as distinct, calibrated numbers — not all-0 (model-ignoring) and not all-0.99 (no spread). Good enough to drive composite scoring; proceeded with full eval.

## Headline numbers (v0 → v1 → v2)

| metric                | run_001 (v0)  | run_002_v1    | run_003_v2    |
| ---                   | ---           | ---           | ---           |
| samples               | 10            | 10            | 10            |
| matched (any found)   | 10            | 10            | 10            |
| selection_evaluated   | n/a           | 7             | 7             |
| correct_selection     | n/a           | 2             | **3**         |
| selection_accuracy    | n/a           | 28.6 %        | **42.9 %**    |
| bbox_iou_scored       | n/a           | n/a           | 3             |
| bbox_iou_mean         | n/a           | n/a           | 0.028         |
| bbox_iou_pass_50      | n/a           | n/a           | 0 %           |
| mean_ssim (secondary) | 0.328         | 0.352         | 0.343         |
| pass@SSIM ≥ 0.5       | 20 %          | 20 %          | 20 %          |
| cost                  | $0.22         | $0.27         | $0.35         |

Notes:
- v0 had no gold-vs-LLM selection logic, so its `selection_*` columns are n/a; the SSIM lift from v0 → v2 is small (+0.015) and noisy.
- Cost is up ~30 % from v1 because v2 emits 4 extra scalar fields per call and runs against the same 65 evidence images.
- IoU was added in v2 but only scores when LLM picks the gold evidence AND a red-box bbox was detectable (n = 3 here). The IoU numbers themselves are misleading and need their own analysis (see below).

## Per-sample delta (v1 → v2)

```
1044246    sel=- (no gold)                  ssim 0.13 → 0.15
1494172    sel=- (no gold)                  ssim 0.21 → 0.19
1969989    sel=X (12_…) → X (04_…)          ssim 0.19 → 0.14   regressed pick
2423810    sel=X (01_…) → OK (02_…)         ssim 0.29 → 0.26   ← FIXED by v2
3089225    sel=- (no gold)                  ssim 0.37 → 0.36
4334451    sel=X (05_…) → X (04_…)          ssim 0.38 → 0.43   different wrong pick
4338293    sel=OK                           ssim 0.18 → 0.15
4601531    sel=X (03_…) → X (04_…)          ssim 0.64 → 0.68   different wrong pick
4827580    sel=X (02_…) → X (02_…)          same wrong pick
6433801    sel=OK                           ssim 0.75 → 0.70
```

Net: **+1 sample fixed (2423810)**, **0 regressed from OK to X**, several "different but still wrong" picks because the composite-score reshuffled the order among many high-confidence positives.

## What v2 actually fixed and didn't

### 2423810 (FIXED, sample where v2 paid off)

In v1, `max(confidence)` picked `01_rId5_image2.png` (a cluttered shot) over the gold `02_rId6_image3.png` (a clean close-up of the trademark tag). Confidence was 0.99 vs 0.98 — a tie in any practical sense. In v2 the model marked the cluttered shot `found=false conf=0.06` (correctly: it's actually a labeled-bottle photo where the registered TM is not the dominant subject), and the gold close-up got `composite=0.966` (conf 0.98, clarity 0.99, completeness 1.00, isolation 0.97). The new single-instance rule + clarity/isolation scoring did exactly what we wanted.

### 1969989 (regressed pick within the X column — but interestingly)

`12_rId24_image21.png` is the gold (a tiny NBA logo strip at the edge of a screenshot). In v1 the model picked `01_rId13_image10.png` (an NBA-branded poster). In v2 it picked `04_rId16_image13.png` — a different NBA poster, with `iso=1.00 clar=0.98` because the registered TM is the sole element. **That's actually correct behavior from the prompt's wording** — the prompt rewards "clean subject" — but the human gold annotation was a sloppy thumbnail strip rather than the cleanest occurrence. This is a ground-truth quality issue, not a prompt issue. v2 picked an arguably better evidence image; the test set just disagrees.

### 4334451 (regressed within X column)

v1 picked `05_…` (a styled badge close-up). v2 picked `04_…` (a clean isolated badge). Gold is `03_…` (the badge on the actual product car body). Same pattern: the prompt prefers "clean isolated trademark" but the human marker wanted "trademark in product context". This is a **conflicting definition of what "best evidence" means** — the user's red-box convention seems to prefer "on the product" while the prompt currently rewards "isolated / clean". One of the two must change.

### 4601531 (regressed within X column)

v1 picked `03_…`, v2 picked `04_…` — both wrong vs gold `05_…`. The gold image is a wide product photo where the TM occupies the bumper area; the picks are tighter close-ups. Same isolation-vs-context dilemma as 4334451.

### 6433801 (still SEL_OK but SSIM dropped)

Only one evidence in this sample, so selection was forced. The bbox shifted: v1's was `(377, 655, 484, 785)` — a tight crop of the actual emblem area; v2's is `(379, 635, 972, 826)` — a wider strip including the rear logo area. The wider crop drives the IoU score to 0 against the red-box (which spans `(0, 1213, 2015, 1511)` — the whole windshield base). The red-box gold is *itself* very loose here, so all bbox IoU numbers in v2 are dominated by definition mismatch, not by detection accuracy.

### bbox IoU = 0.028 is not real failure

The red-box gold rectangles, as detected by `detect_redbox_bbox`, are the *human's hand-drawn rectangles around a wide product area*, not tight bounding boxes around the trademark itself. Example: sample 6433801 red-box covers nearly the whole bottom of the windshield (`(0, 1213, 2015, 1511)`), while the actual TM emblem is `(379, 635, 972, 826)` — completely outside that rectangle because the red box is on the *bottom* of the photo and the emblem the model found is in the *middle*. Conclusion: the IoU metric as constructed is measuring agreement between two different conventions (LLM = tight-around-TM, human = loose-region-of-interest), and is not useful as a quality signal until we either (a) regenerate gold with tight TM boxes, or (b) accept a "bbox center inside red box" looser metric.

## Confidence calibration check

Old prompts: nearly every found=true got 0.98 or 0.99.
v2 prompt: distribution is genuinely spread. Examples from the raw log:

- 0.99 — clean TM, no defects (e.g. 1969989 image 04 NBA poster, 4338293 image 06 the gold).
- 0.95-0.98 — clearly the TM, mild blemishes (1044246 imgs 02/03 with varying clarity).
- 0.91-0.93 — most of the trademark visible but partial issues (sample_6433801 windshield, 3089225 image 04 stylized).
- 0.78 — partial / heavily-stylized appearance (3089225 image 02 — a stitched-logo polo).
- 0.02-0.18 — true negatives, used consistently across the 1969989 long-tail of off-topic images.

The model is using the full range now. The single-instance rule also worked — it produced more `found=false` on cluttered images that v1 would have marked positive (e.g. 2423810 image 01, 4601531 image 05).

## Recommendation: iterate to v3, do NOT ship v2 to the web app yet

v2 demonstrably helps (selection accuracy 29 % → 43 %), but two structural issues remain that are bigger than another prompt tweak can fix:

1. **The "best evidence" convention is ambiguous.** Some red-boxes in the gold set are on "the trademark cleanly in isolation", others are on "the trademark in product context with surrounding scene". The prompt currently optimizes for the former. Until we decide which one we want (and re-mark gold accordingly), selection accuracy is bounded.

2. **The bbox IoU metric measures the wrong thing.** Right now it measures agreement with the human's loose region-of-interest box, not with a tight TM box. Either:
   - Re-annotate gold with tight TM bounding boxes (manual effort, but small — 7 samples), or
   - Add a softer metric like "LLM's bbox center lies inside the red box, AND area ratio between 5 %-50 % of red box" so we keep using the existing gold.

### Concrete v3 hypotheses worth testing

- **v3a — "context-aware isolation"**: rewrite the `isolation` definition so that "registered TM clearly as branding on a product surface" scores 0.9, not "registered TM on a white background isolated from product" — currently those probably score the same. This would push the model back toward picking 03_… / 05_… type "TM on the product" candidates and might fix 4334451 / 4601531.
- **v3b — "two-pass with shortlist"**: do one pass with v2 to filter found=true candidates, then send the model a montage of the top-3 candidates + the logo and ask "which ONE of these is the best evidence image for this trademark?" — uses the model itself to disambiguate the convention. ~3 extra LLM calls per sample, ~$0.10 extra per sample.
- **v3c (data fix, not prompt)**: re-mark the 7 gold samples with tight TM boxes and re-run v2 unchanged; if selection accuracy stays at 43 % but IoU now means something, the prompt may be production-ready.

My best guess: **v3a first** (cheapest, one prompt edit, one re-run, $0.35). If it doesn't lift to >= 60 % selection accuracy, then **v3c** to get a real bbox IoU signal, then revisit.

## Cost log

- run_003_v2: $0.348 / 10 samples (65 LLM calls). Avg per-call cost ≈ $0.005. Per-sample mean ≈ $0.035.
- Total cumulative dev-loop spend (v0+v1+v2): $0.83 — well under the $5 stop budget.
