# SOTA Prompt Techniques for Logo bbox Detection in Real Product Photos

- **Query**: What's publicly known about coaxing tight bboxes for a specific small logo / trademark from frontier multimodal LLMs (GPT-4o / GPT-5 / Claude / Gemini)?
- **Scope**: External (web research, papers, vendor docs, practitioner blogs)
- **Date**: 2026-05-23
- **Time spent**: ~15 min, mostly arxiv + vendor docs + Simon Willison + Roboflow

---

## TL;DR — what to try next, ranked by expected impact / cost

1. **Stop asking GPT for raw `[x1,y1,x2,y2]` pixels.** Vendor consensus (OpenAI Cookbook, Roboflow, Simon Willison) is that **GPT-4o/4.1/5 do not reliably output pixel-accurate bounding boxes** — this is a documented weakness, not a prompt bug we can fix. Either: (a) route the *localization* step to Gemini 2.x+ (which has a first-party `box_2d` 0–1000 schema), OR (b) switch GPT's role to *selection over pre-cropped candidates*, not localization.
2. **Set-of-Mark (SoM) is the highest-leverage prompt change.** Pre-segment candidate regions with SAM/SEEM, overlay numeric labels, and ask GPT-5 to pick a label (not draw a box). Original SoM (Yang et al., 2023) reports zero-shot GPT-4V *beats fully-finetuned RefCOCOg SOTA* on referring-expression grounding — i.e. selection-over-marks is provably stronger than asking for coords. (arxiv.org/abs/2310.11441)
3. **Two-stage spec-first decomposition**, exactly per OpenAI's own Nov-2026 cookbook: stage 1 = "ground semantics" (does this candidate region contain the logo? what part?), stage 2 = act/refine. Don't ask one prompt to "find AND tighten AND verify" — that is the failure mode VL-RewardBench documents (GPT-4o = 65.4% on visual perception sub-tasks). Cookbook explicitly recommends *deterministic geometry over model self-validation* for the tightening step.
4. **Don't trust the model to self-verify its own bbox.** Huang et al. 2024 ("LLMs Cannot Self-Correct Reasoning Yet") show intrinsic self-correction can *degrade* accuracy. If we want a second pass, give it *new evidence* (a crop, a flipped/zoomed view, a competitor candidate) — not the same image.
5. **For multi-candidate selection, prefer SoM-style "all candidates in one prompt" over per-image scoring.** This is what V\* (Wu & Xie 2023) and Visual CoT (Shao et al. 2024) effectively do. Per-image confidence scores are known to anchor near 0.9 with no calibration, so ranking by absolute score is unreliable; ranking by *forced choice among labeled options* sidesteps this.

If I had to pick one thing for v3: **build a SoM-style pipeline** — pre-extract N candidate crops from the product photo using cheap CV (template match, SAM mask, even sliding window), overlay them with bright numeric labels back onto the original image, and ask GPT-5 a single forced-choice question: *"Which numbered region contains the target logo? Reply with just the number."* Expected lift: SoM took GPT-4V from below-SOTA to above-fully-finetuned-SOTA on RefCOCOg in the original paper.

---

## Finding 1 — Set-of-Mark (SoM): selection beats coordinate-emission

**Source**: Yang, J. et al. "Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V." arxiv.org/abs/2310.11441 (v2, Oct 2023, code microsoft/SoM). Retrieved 2026-05-23.

SoM = use an off-the-shelf segmenter (SAM / SEEM) to partition the image into regions, overlay each region with a unique mark (number, letter, mask), and feed the *marked image* to the VLM. The model then refers to regions by mark ID instead of pixel coordinates. Reported result: **zero-shot GPT-4V + SoM beats the fully finetuned SOTA on RefCOCOg** referring-expression segmentation. The mechanism is that the model's weak coordinate emission is replaced by its strong symbol-grounding ability ("region 7 contains the logo" is a categorical answer, not a numeric regression). The HuggingFace paper page and follow-on derivatives (Flickr30k_Grounding_Som, COCO_OVSEG_Som datasets) confirm the technique generalizes.

**Apply to linebase**: Add a pre-segmentation step (SAM or even a cheap template-matched candidate generator), render the segments back onto the original image with bright numeric overlays of varying sizes, and rewrite the v3 prompt as *"Which numbered region contains the trademark logo? Reply with just the number, or 'none'."*

## Finding 2 — GPT-4o/4.1/5 do not emit pixel-accurate bboxes; Gemini does (0–1000 normalized)

**Sources**: (a) ai.google.dev/gemini-api/docs/vision — Gemini official: returns `box_2d` as `[ymin, xmin, ymax, xmax]` scaled to 0–1000, and you descale by original dims; (b) simonwillison.net/2024/Aug/26/gemini-bounding-box-visualization/ — Simon Willison explicitly says GPT-4o and Claude 3 / 3.5 "can't do this (yet)"; (c) blog.roboflow.com/multimodal-vision-models/ — "OpenAI's models struggle with object detection. You can fine-tune GPT for object detection, but you can get better performance with other models like Florence-2"; (d) blog.roboflow.com/gpt-4o-object-detection/ — the only Roboflow-recommended path to bboxes from GPT-4o is *fine-tuning* (~$50, 2M tokens). All retrieved 2026-05-23.

There is **no known prompt** that makes vanilla GPT-4o/4.1/5 reliably output tight pixel bboxes. The Qwen-VL / Gemini 0–1000 convention is **not honored by GPT-series** out of the box; GPT will produce numbers shaped like a bbox but the spatial fidelity is poor for small objects. This is the load-bearing reason our v0–v2 prompt-only plateau exists.

**Apply to linebase**: Drop "ask GPT for `[x1,y1,x2,y2]`" entirely from v3. If we must use a single VLM for end-to-end localization, swap to Gemini 2.x+ and use its `box_2d` schema verbatim. If we keep GPT-5, change its job to *selection* (Finding 1) or *verification of an externally proposed crop* (Finding 4).

## Finding 3 — V\* and Visual CoT: zoom-then-look beats one-shot

**Sources**: (a) Wu & Xie, "V\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs", arxiv.org/abs/2312.14135, NeurIPS 2024 highlight. (b) Shao et al., "Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning", arxiv.org/abs/2403.16999, 438k samples annotated with intermediate bboxes. Retrieved 2026-05-23.

Both papers attack the exact failure mode we hit: **small target inside high-res / cluttered image**. V\* introduces an LLM-guided *visual search* loop (look at low-res, propose region of interest, zoom in, re-look) and reports the V\*Bench benchmark specifically built around this. Visual CoT explicitly annotates *intermediate bboxes* that gate the final answer, and trains the model to "dynamically focus on visual inputs". The shared insight: **a single forward pass over the full image is the wrong primitive for small-object reasoning** — you need an iterative attend-then-decide loop, where each step works on a tighter crop.

**Apply to linebase**: v3 should be two-pass minimum. Pass 1: low-res (e.g. 768px) + SoM marks over coarse candidates → pick coarse region. Pass 2: high-res crop of the picked region + a fresh prompt asking "does this contain the logo? if yes, where exactly within this crop?" — let pass-2 emit a *small* relative bbox (which is the regime where GPT is least bad, because the target is now a large fraction of the image).

## Finding 4 — Self-verification on the same image does not help (and can hurt)

**Source**: Huang, J. et al. "Large Language Models Cannot Self-Correct Reasoning Yet", arxiv.org/abs/2310.01798, ICLR 2024. Retrieved 2026-05-23. Direct quote: "LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction."

This is a text-LLM paper, but the mechanism (the model anchors on its first answer and rationalizes) transfers to VLMs — and is consistent with the VL-RewardBench finding (next) that even GPT-4o only hits 65.4% on visual perception judgement. Reflexion / Self-Refine loops work when the second pass has *new information* (a tool result, a new view, a critic). They do not work when you re-ask the same model the same question on the same image.

**Apply to linebase**: Do NOT add a "verify your own bbox" pass over the same image. If we add a pass-2, it must change the *input* — e.g. (a) crop to the proposed region and re-ask, (b) flip / desaturate / mark the proposed region and ask "is this still the logo?", or (c) feed the *other* top-N candidates as a forced-choice ranking task. The cheapest variant: pass-2 receives only the crop, not the original.

## Finding 5 — VL-RewardBench: even GPT-4o is at 65% on basic visual perception

**Source**: Li et al., "VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models", arxiv.org/abs/2411.17451, v-2025-06. Retrieved 2026-05-23. Quote: "even GPT-4o achieves only 65.4% accuracy ... models predominantly fail at basic visual perception tasks rather than reasoning tasks."

This is the most important *expectation-calibration* data point we have. Our 43% on tricky cases is bad, but the ceiling on hard visual perception for current frontier VLMs is not 100% — it is closer to ~65% on a curated hard set. **Prompt-only improvements have a hard ceiling somewhere in the 60–70% range for the perception sub-task**; further gains require either (a) better candidates fed in (Finding 1), (b) iterative zoom (Finding 3), or (c) leaving the VLM and using a dedicated detector for the geometry step.

**Apply to linebase**: Set v3's target accuracy honestly. Going from 43% → 70% via SoM + zoom is plausible. Going to 90%+ via prompt-only is not supported by any public evidence and we should stop trying.

## Finding 6 — OpenAI's own cookbook says: spec-first, deterministic geometry

**Source**: developers.openai.com/cookbook/examples/multimodal/grounded_spatial_reasoning_layouts ("Evaluating Grounded Spatial Reasoning with GPT-5.5"). Retrieved 2026-05-23.

OpenAI's recommended workflow for any spatial-reasoning task: (1) split into "ground semantics from image" + "act on the grounded spec" as **two LLM calls**; (2) make the model emit a *structured spec* (JSON), not a rendered answer; (3) use **deterministic code, not the model, for geometric cleanup** ("bounds checks, object overlap, source-wall collisions, rotations, and repacking"); (4) eval at the spec level, not at the rendered-image level. Direct quote: *"Model reasoning is therefore not treated as self-validating; it is constrained, checked, and improved through the surrounding system."*

**Apply to linebase**: v3 architecture should be: (stage A) GPT-5 picks a labeled candidate region (semantic decision) → (stage B) Python/OpenCV refines the crop using contour / edge / template-match on the picked region (deterministic geometry) → (stage C) optional GPT-5 verify-by-crop pass for the close calls only. Do not ask GPT to do A+B+C in one prompt.

## Finding 7 — Multi-candidate selection format: numbered all-in-one > pairwise > per-image

**Sources**: SoM (Finding 1) implicitly endorses "all-in-one numbered prompt". No paper I found directly compares the three formats head-to-head for VLMs, so this is a weaker claim. The supporting circumstantial evidence: (a) RefCOCOg referring-expression task is structured as forced choice over labeled regions, and that's where SoM wins; (b) per-image scoring suffers from the calibration anchor problem (no paper specifically tested this on VLMs, but it's a well-documented LLM phenomenon — confidence clusters near majority-class probability).

**Apply to linebase**: For v3 selection, present all N candidates in one image (SoM-style numbered overlays) with one forced-choice prompt. Avoid asking N independent "is this the logo?" prompts and then comparing the model's self-reported confidences — they are not calibrated.

---

## Skipped / negative — techniques that don't apply or don't work for us

- **Ferret / Ferret-v2 / Kosmos-2 / PaliGemma** (arxiv 2310.07704, 2404.07973, 2306.14824, 2407.07726): these are **finetuned grounding models**, not prompt techniques. They work but require us to host & finetune a model. Out of scope unless we accept that as a new architecture. **Why noted anyway**: if v3 still plateaus, the *next* escape hatch is "use a dedicated grounding VLM as the geometry step and keep GPT for selection/verification only" — Ferret-v2 + GPT-5 selector is a known-good combo template.
- **Fine-tuning GPT-4o for object detection** (Roboflow): documented to work (~$50 / 2M tokens) but (a) we don't have ground-truth bboxes at scale, (b) it locks us into OpenAI hosting. Note for later, not for v3.
- **Cambrian-1's Spatial Vision Aggregator** (arxiv 2406.16860): architecture-level change to the vision encoder. Inapplicable to closed-API frontier models.
- **GLOV** (arxiv 2410.06154): prompt-optimization via an LLM-as-optimizer loop. Works for *classification* over a known label set; our task is localization, not classification, so it doesn't transfer cleanly. Also requires a labeled training set.
- **iPhone EXIF rotation bug** (Simon Willison 2024): cute footnote — Gemini's bbox API doesn't honor TIFF orientation metadata. **Apply to linebase: if any of our product photos are from iPhone JPEGs, normalize orientation before sending.** Cheap to add, prevents a class of weird-rotation failures.
- **Anthropic Claude bbox capability**: as of Aug 2024, "can't do this yet" per Simon Willison; no newer public claim I found contradicts that. So Claude 3.5+ is *worse* than Gemini for localization. Don't switch from GPT to Claude expecting a win.

---

## Notes on search quality

- Strongest sources (multi-source consensus): GPT-series weak at raw bbox; SoM works; Gemini uses 0–1000.
- Weaker / single-source: the exact "per-image vs all-in-one selection" claim — no head-to-head VLM benchmark found. Labeled as such in Finding 7.
- I could not find good 2025-2026 numbers specifically on **GPT-5** bbox quality — OpenAI hasn't published bbox benchmarks for GPT-5 series, and the OpenAI cookbook page (which uses "GPT-5.5") doesn't claim coordinate accuracy improved. Assume the GPT-4o weakness persists in GPT-5 until proven otherwise.
- No primary-source Hacker News / r/LocalLLaMA thread surfaced cleanly in time budget (Reddit JSON returned empty without auth, HN search blocked). The Roboflow + Simon Willison posts cover the same practitioner ground.
