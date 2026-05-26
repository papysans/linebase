---
prompt_version: "verify-design-1"
created: 2026-05-26
purpose: |
  Verify-loop prompt for design-patent table evaluation. Image 1 is a patent
  line drawing or design drawing; Image 2 is a candidate crop from a product
  photo. This prompt verifies product-shape correspondence, not trademark
  identity or branding use.
---
You are given two images.

**Image 1** is a design-patent line drawing or product design reference.
**Image 2** is a CANDIDATE CROPPED REGION cut from a real-world product photo.
A first-pass model already thought this crop contains the product/design shown
in Image 1 and added padding around its bounding box. Your job is to verify
whether the crop contains the corresponding real product or design feature, and
to comment on how well the crop is sized.

You must compare visual shape only: silhouette, proportions, visible structure,
and distinctive product features. Do NOT require Image 2 to contain a trademark
logo, brand text, or any kind of branding. Many valid matches are plain
products with no logo at all.

Return `contains_full_logo=true` when Image 2 contains the same product/design
as Image 1, or a clearly corresponding perspective of it. Differences in color,
material, lighting, photo angle, minor accessories, or surface texture do not
make it wrong when the core shape and diagnostic features match.

Use `fit="wrong"` only when Image 2 is a different product/design, only shows a
small non-diagnostic fragment, or the crop misses the object. If the object is
present but padded, use `fit="loose"` and provide a tighter `suggested_bbox` in
the cropped image's own coordinates when you can.

Respond with strict JSON, no prose, no markdown fences:

{
  "contains_full_logo": true | false,
  "fit": "tight" | "loose" | "too_tight" | "wrong",
  "confidence": 0.0 - 1.0,
  "reason": "<one short sentence>",
  "suggested_bbox": [x1, y1, x2, y2] or null
}

Field definitions:

- `contains_full_logo` means the candidate crop contains the corresponding
  product/design from Image 1 as a whole or as a clear equivalent view. Treat
  this field as "contains_full_design" for this task.
- `fit="tight"` means the product/design fills most of the crop with little
  surrounding background and no diagnostic part cut off.
- `fit="loose"` means the product/design is fully inside the crop but there is
  noticeable extra background or surrounding context.
- `fit="too_tight"` means part of the product/design is clipped by the crop edge.
- `fit="wrong"` means the crop does not contain the corresponding design.
- `suggested_bbox`, when used, is `[x1, y1, x2, y2]` in Image 2's cropped-image
  coordinates, 0-indexed, top-left origin, x1 < x2, y1 < y2.

Consistency rules:
- If `contains_full_logo` is false, `fit` MUST be `"wrong"` and
  `suggested_bbox` MUST be null.
- If `fit` is `"wrong"`, `contains_full_logo` MUST be false.
- If `fit` is `"tight"` or `"too_tight"`, `suggested_bbox` should be null.
