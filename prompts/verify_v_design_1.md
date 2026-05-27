---
prompt_version: "verify-design-1"
created: 2026-05-26
purpose: |
  Verify-loop prompt for design-patent table evaluation. Image 1 is a patent
  line drawing or design drawing; Image 2 is a candidate crop from a product
  photo. This prompt verifies design correspondence, including product-shape
  designs and surface/ornamental patterns, not trademark identity or branding
  use.
---
You are given two images.

**Image 1** is a design-patent line drawing or design reference.
**Image 2** is a CANDIDATE CROPPED REGION cut from a real-world product photo.
A first-pass model already thought this crop contains the product/design shown
in Image 1 and added padding around its bounding box. Your job is to verify
whether the crop contains the corresponding real product, design feature, or
surface/ornamental pattern, and to comment on how well the crop is sized.

Compare the design content that Image 1 actually depicts:
- for product-shape designs, compare silhouette, proportions, visible
  structure, and distinctive product features;
- for surface/ornamental designs, compare the repeated geometry, stitching
  layout, lattice, grid, diamond, octagonal, studded, quilted, or other
  distinctive pattern.

Do NOT require Image 2 to contain a trademark logo, brand text, or any kind of
branding. Many valid matches are plain products with no logo at all.

Return `contains_full_logo=true` when Image 2 contains the same product/design
as Image 1, a clearly corresponding perspective of it, or the same distinctive
surface/ornamental pattern applied to a visible product area. Differences in
color, material, lighting, photo angle, minor accessories, or product carrier do
not make it wrong when the diagnostic design content matches. In particular, do
not reject a surface-pattern match merely because it appears on a different
carrier product such as a wallet, handbag, glove, backpack, or garment.

Use `fit="wrong"` only when Image 2 does not contain the corresponding design
content, only shows a small non-diagnostic fragment, or the crop misses the
object/pattern. If the design is present but padded, use `fit="loose"` and
provide a tighter `suggested_bbox` in the cropped image's own coordinates when
you can.

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
  product/design or surface/ornamental design from Image 1 as a whole or as a
  clear equivalent view/use. Treat this field as "contains_full_design" for this
  task.
- `fit="tight"` means the corresponding design content fills most of the crop
  with little surrounding background and no diagnostic part cut off.
- `fit="loose"` means the corresponding design content is fully inside the crop
  but there is noticeable extra background or surrounding context.
- `fit="too_tight"` means part of the design content is clipped by the crop edge.
- `fit="wrong"` means the crop does not contain the corresponding design
  content.
- `suggested_bbox`, when used, is `[x1, y1, x2, y2]` in Image 2's cropped-image
  coordinates, 0-indexed, top-left origin, x1 < x2, y1 < y2.

Consistency rules:
- If `contains_full_logo` is false, `fit` MUST be `"wrong"` and
  `suggested_bbox` MUST be null.
- If `fit` is `"wrong"`, `contains_full_logo` MUST be false.
- If `fit` is `"tight"` or `"too_tight"`, `suggested_bbox` should be null.
