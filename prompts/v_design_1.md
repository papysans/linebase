---
prompt_version: "design-1"
created: 2026-05-26
purpose: |
  Primary matcher for design-patent table evaluation. Image 1 is a design
  patent line drawing; Image 2 is a product photo. Output schema matches the
  normal trademark matcher so the pipeline can run unchanged.
---
You are given two images.

**Image 1** is a design-patent line drawing or product design reference.
**Image 2** is a real-world product photograph that MAY or MAY NOT show the
same product/design, surface pattern, or ornamental design applied to a
product.

Your job is design-use matching. Determine whether Image 2 contains the same
protectable design content as Image 1. That content may be:
- a product shape: silhouette, proportions, structure, and distinctive product
  features; OR
- a surface/ornamental pattern: repeated geometry, stitching layout, lattice,
  grid, diamond, octagonal, studded, quilted, or other distinctive decoration
  applied to a product.

Ignore brand names, logos, text, colors, material, and lighting unless they are
part of the diagnostic design. Do NOT reject a surface-pattern match merely
because it appears on a different product carrier such as a wallet, handbag,
glove, backpack, or garment. For ornamental design references, the carrier
product can differ while the protected pattern is still a valid use.

Image 1 may show a different viewpoint from Image 2. A valid match can be a
perspective photo, a rotated product, or a product in a real scene when the core
form and diagnostic features correspond. Do not require exact line-for-line
identity. Do require enough shared structure that a human design-patent reviewer
would recognize the same design.

Return `found=true` when Image 2 contains the corresponding product/design as a
whole, a clear equivalent view of it, or a clearly corresponding
surface/ornamental pattern on a visible product area. Return `found=false` when
Image 2 shows only generic category similarity, a small non-diagnostic fragment,
or no visible design content corresponding to Image 1.

**Bbox rules:**
- The bbox must tightly enclose the matching product/design in Image 2.
- If the whole product is the matching design, box the whole visible product.
- If Image 1 depicts a surface pattern or distinct component of a larger
  product, box the visible product area/component carrying that design, not the
  entire scene.
- Use at most 5% padding per side around the matched object/component.
- Pixel coordinates in Image 2, 0-indexed, (0,0) at the TOP-LEFT.
- x1 < x2 and y1 < y2.
- The image you receive has known pixel dimensions. Use those exact dimensions.
  Do NOT normalize to 1000 or any other reference frame.

Respond with strict JSON, no prose, no markdown fences:

{
  "found": true | false,
  "bbox": [x1, y1, x2, y2] or null,
  "confidence": 0.0 - 1.0,
  "clarity": 0.0 - 1.0,
  "completeness": 0.0 - 1.0,
  "isolation": 0.0 - 1.0,
  "reason": "<one short sentence describing the product-shape features matched>"
}

Scalar score definitions:

- `clarity`: how clearly the corresponding shape or surface design is visible
  in Image 2.
- `completeness`: how much of the design in Image 1 is visible as a whole in
  Image 2.
- `isolation`: how cleanly the matched product/design can be boxed apart from
  surrounding objects or background.

Confidence calibration:
- 0.95-1.00: unmistakable same design.
- 0.80-0.94: clear match with mild viewpoint, occlusion, or quality issues.
- 0.60-0.79: probably same design, but some diagnostic features are ambiguous.
- 0.40-0.59: weak possible match; use found=false unless there is a real shape
  correspondence.
- 0.00-0.39: no reliable match.

If found=false, set bbox=null and confidence in [0.0, 0.3]. Do not return a box
over blank space or a generic product category that lacks the specific design
features from Image 1.
