---
prompt_version: "verify-1"
created: 2026-05-23
predecessor: null
changes:
  - First verify prompt. Used by the self-verify loop in src/linebase/verify_loop.py.
  - Input is Image 1 = the registered line-art logo; Image 2 = a candidate CROPPED REGION (with ~20% padding) that an earlier first-pass call thought contained that logo.
  - The model must say (a) whether the registered trademark is present as a WHOLE in the crop, (b) how the crop fits the logo, (c) optionally suggest a tighter bbox in the CROP's own pixel coordinates.
---
You are given two images.

**Image 1** is a line-art registered trademark logo as it appears in the USPTO registry.
**Image 2** is a CANDIDATE CROPPED REGION cut from a larger real-world product photograph. A first-pass model already thought this crop contains the registered trademark and added ~20% padding around its bounding box; your job is to **verify** whether that decision was correct, and to comment on how well the crop is sized.

You must reason as a US trademark examiner. Be strict about identity: a partial
element, a different but similar brand, or background patterning does NOT count
as the registered trademark being present.

But do NOT reject a candidate only because it is faint, low-contrast, blurry,
dirty, reflective, or printed behind glass. Real product photos often contain
weak or partially degraded marks. If the overall silhouette, proportions, and
identifying internal structure are visible as a whole and correspond to Image 1,
return `contains_full_logo=true` with lower confidence and `fit="loose"` or
`fit="tight"` as appropriate. Use `fit="wrong"` only when the shape identity is
actually different, missing, partial, or background-only.

Respond with strict JSON, no prose, no markdown fences:

{
  "contains_full_logo": true | false,
  "fit": "tight" | "loose" | "too_tight" | "wrong",
  "confidence": 0.0 - 1.0,
  "reason": "<one short sentence>",
  "suggested_bbox": [x1, y1, x2, y2] or null
}

**Field definitions:**

- `contains_full_logo` — is the registered trademark visible **as a whole** in Image 2? Its overall outline, proportions, and identifying internal features must all be present together, used as branding on the product. A single decorative element or a different brand is NOT the registered trademark.

- `fit` — how the crop is sized relative to the registered trademark:
  - `tight` — the trademark fills most of the crop with only a thin band of background. No part of the mark is cut off at the edges.
  - `loose` — the trademark is fully inside the crop but there is noticeable empty space (background, surrounding product, padding > ~20%) around it. When fit is `loose`, you SHOULD provide a tighter `suggested_bbox` in the crop's own pixel coordinates.
  - `too_tight` — the trademark is partially cut off at the crop's edges (clipped at top / bottom / left / right). Some identifying feature of the mark continues past the crop boundary.
  - `wrong` — Image 2 does NOT contain the registered trademark as a whole
    (maybe a different brand, only a partial element of the registered mark, or
    just background pattern). Do not use `wrong` solely for low image quality if
    the whole matching shape is still visible. When fit is `wrong`,
    `contains_full_logo` MUST be false.

- `confidence` — your confidence in the `contains_full_logo` decision, 0.0 to 1.0. Use the full range — 0.95+ is reserved for unmistakable, textbook cases.

- `reason` — one short sentence describing why.

- `suggested_bbox` — when `fit == "loose"` and you can give a tighter rectangle, return `[x1, y1, x2, y2]` in the CROPPED IMAGE's pixel coordinates (Image 2's own coordinate system), 0-indexed, (0,0) at the TOP-LEFT, x1 < x2, y1 < y2. Otherwise return null. Do NOT return a bbox larger than the crop itself.

**Consistency rules:**
- If `contains_full_logo` is false, `fit` MUST be `"wrong"` and `suggested_bbox` MUST be null.
- If `fit` is `"wrong"`, `contains_full_logo` MUST be false.
- If `fit` is `"too_tight"`, `suggested_bbox` should be null (you cannot suggest a tighter box in a crop where the mark is already clipped).
- If `fit` is `"tight"`, `suggested_bbox` should be null (no shrink needed).

When in doubt between `contains_full_logo=true` and `contains_full_logo=false`, lean false — false positives cost more than misses.
