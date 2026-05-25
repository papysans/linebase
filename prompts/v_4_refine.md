---
prompt_version: "4_refine"
created: 2026-05-25
predecessor: "4"
purpose: |
  Iter-9 refine pass. Sent when verify says the Pass-1 / tile-scan bbox contains
  the logo but is `loose` (the bbox is correct in spirit but not tight). The
  caller crops the photo to a +30% padded region around the loose bbox, optionally
  upscales it 2x when the longest side < 400 px, and asks the VLM to give a
  tight bbox INSIDE THAT ZOOMED CROP.

  The returned bbox is in the ZOOM-CROP's pixel coords. The caller translates
  back to original-photo coords by adding `(zoom_origin_x, zoom_origin_y)` —
  which it computed from the +30% pad. Output schema is identical to v_4 so the
  existing parser keeps working unchanged.
---
You are looking at a ZOOMED-IN view of a region that DEFINITELY contains the line-art trademark from Image 1.

Your job: give a TIGHT pixel-perfect bounding box around the trademark shape in this zoomed image.

The trademark IS visible here — do NOT return found=false. The previous attempt already confirmed the logo is in this region; you are now refining the bounding box.

Look at Image 1's shape (silhouette, proportions, internal structure). Find the matching shape in Image 2 (the zoom). Return its tightest bbox in Image 2's pixel coordinates (the zoom, NOT the original).

Output (strict JSON, no prose, no fences):
{
  "found": true,
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0 - 1.0,
  "clarity": 0.0 - 1.0,
  "completeness": 0.0 - 1.0,
  "isolation": 0.0 - 1.0,
  "reason": "<one short sentence describing the SHAPE you matched>"
}

If the bbox is genuinely smaller than 28 pixels in either dimension, expand it to at least 28x28 around the matched shape's center.

Coordinates are absolute pixel positions in the zoomed image as transmitted. Do NOT normalize to 1000 or any other reference frame.
