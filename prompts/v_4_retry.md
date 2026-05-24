---
prompt_version: "4_retry"
created: 2026-05-24
predecessor: "4"
purpose: |
  Pass-3 retry-with-feedback. Sent ONLY when a Pass-1 high-confidence bbox was
  verify-rejected with a "blank-crop" signal (verifier reports the crop is
  uniform background / has no trace of the logo). We tell the model exactly
  what its previous mistake looked like and ask it to re-scan with shape-only
  framing — the logo MAY still be in the same image, just somewhere else.

  Substitute %PRIOR_X1 / %PRIOR_Y1 / %PRIOR_X2 / %PRIOR_Y2 / %VERIFY_REASON
  at call time. Output schema MUST stay identical to v_4 so the existing
  match-result parser keeps working unchanged.
---
You are re-examining an image after a previous attempt failed.

PRIOR ATTEMPT
- Bounding box tried: (x1=%PRIOR_X1, y1=%PRIOR_Y1, x2=%PRIOR_X2, y2=%PRIOR_Y2)
- Confidence claimed: high
- Verifier said: "%VERIFY_REASON"

This means your previous bounding box pointed at background / blank space, not at the logo. The logo may STILL be present elsewhere in the same image — but at a different location and possibly smaller, partially occluded, or stylized.

Your job NOW:
1. Ignore your previous answer. Scan the ENTIRE image fresh.
2. Look for the SHAPE described by the line-art reference (silhouette, proportions, distinctive contours) — NOT brand text, NOT color.
3. If you find a region whose SHAPE matches the line-art, return new bbox in original-image coords. The image you receive has known pixel dimensions — use those exact dimensions. Do NOT normalize to 1000 or any other reference frame. Coordinates are absolute pixel positions in the image as transmitted.
4. If after a careful re-scan there is truly no shape match anywhere, return found=false.

Output schema is identical to before:
{"found": bool, "bbox": [x1,y1,x2,y2] or null, "confidence": float, "reason": str, "clarity": float, "completeness": float, "isolation": float}

The bbox MUST NOT overlap heavily with the prior bbox (>50% IoU is forbidden — if you re-find the same region, you have failed again).
