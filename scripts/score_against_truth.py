"""Run the current match pipeline on every (logo, evidence) pair in the truth set
and score IoU vs truth_bbox. Filters: only include pairs with template-match
score ≥ MIN_SCORE (default 0.85) — these are the "trusted" truth labels.

Reports per-pair table + aggregate hit-rate at IoU thresholds {0.0 (any overlap), 0.3, 0.5, 0.7}.

Usage:
    PYTHONPATH=src python scripts/score_against_truth.py [--model M] [--tile_scan] [--min-score 0.85]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linebase.config import Settings
from linebase.llm import match_logo_in_photo  # type: ignore
from linebase.verify_loop import _bbox_blank_stats, match_with_verify  # type: ignore
# Iter 7 — apply the same Pass-1 variance gate the pipeline runner uses, so
# the score script reflects what production rows actually see.
from linebase.pipeline_runner import (  # type: ignore
    _PASS1_BLANK_STD_THRESHOLD,
    _PASS1_BLANK_WHITE_THRESHOLD,
)


def iou(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b: return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--min-score", type=float, default=0.85, help="filter truth pairs by template-match score")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    settings = Settings.from_env()
    truth = json.loads((ROOT / "docs" / "truth_set" / "INDEX.json").read_text(encoding="utf-8"))
    pairs: list[dict] = []
    for block in truth:
        logo_path = ROOT / block["logo"]
        for p in block["pairs"]:
            if p.get("score") is None or p["score"] < args.min_score:
                continue
            if not p["truth_bbox"]:
                continue
            pairs.append({
                "tm": block["tm"], "cat": block["cat"], "pair_idx": p["i"],
                "logo": logo_path, "evidence": ROOT / p["evidence"],
                "truth_bbox": p["truth_bbox"], "score": p["score"],
            })

    if args.limit: pairs = pairs[:args.limit]
    print(f"Truth pairs to score: {len(pairs)}")
    print(f"Model: {args.model}  Verify: {not args.no_verify}\n")

    results: list[dict] = []
    t_total = 0.0
    blank_rejects = 0
    for i, p in enumerate(pairs, 1):
        t0 = time.time()
        blank_reject = False
        blank_stats: tuple[float, float] | None = None
        try:
            if args.no_verify:
                res = match_logo_in_photo(p["logo"], p["evidence"], settings=settings, model=args.model)
                pred_bbox = list(res.bbox) if res.bbox else None
                found = res.found
                verified = None
                reason = res.reason or ""
            else:
                vr = match_with_verify(p["logo"], p["evidence"], settings=settings, model=args.model)
                pred_bbox = list(vr.final_bbox) if vr.final_bbox else None
                found = vr.primary.found
                verified = vr.verified
                reason = vr.primary.reason or ""
            # Iter 7 — Pass-1 variance pre-gate. Mirrors the production gate in
            # `pipeline_runner._one_evidence` so the score reflects deployed
            # behavior. Applied to the FIRST returned bbox (Pass-1 in no-verify
            # mode; vr.primary.bbox in verify mode) — if the gate trips we drop
            # the prediction.
            gate_target_bbox = pred_bbox if args.no_verify else (
                list(vr.primary.bbox) if (vr.primary.found and vr.primary.bbox) else None
            )
            if gate_target_bbox is not None:
                stats = _bbox_blank_stats(p["evidence"], tuple(gate_target_bbox))
                if stats is not None:
                    std_v, white_v = stats
                    blank_stats = stats
                    if (
                        std_v < _PASS1_BLANK_STD_THRESHOLD
                        or white_v > _PASS1_BLANK_WHITE_THRESHOLD
                    ):
                        blank_reject = True
                        blank_rejects += 1
                        # Treat as not-found (matches production gate).
                        pred_bbox = None
                        found = False
                        reason = (
                            f"pass-1 bbox is blank region "
                            f"(std={std_v:.1f}, white={white_v:.2f})"
                        )
        except Exception as e:
            pred_bbox = None
            found = None
            verified = None
            reason = f"ERROR: {e}"
        dt = time.time() - t0
        t_total += dt

        score = iou(p["truth_bbox"], pred_bbox)
        results.append({
            **p, "pred_bbox": pred_bbox, "found": found, "verified": verified,
            "iou": score, "elapsed": dt, "reason": reason,
            "blank_reject": blank_reject,
            "blank_std": blank_stats[0] if blank_stats else None,
            "blank_white": blank_stats[1] if blank_stats else None,
        })

        if blank_reject:
            flag = "BLANK_REJECT"
        elif score >= 0.5:
            flag = "OK"
        elif score >= 0.1:
            flag = "partial"
        else:
            flag = "MISS"
        bs = f" std={blank_stats[0]:.1f} white={blank_stats[1]:.2f}" if blank_stats else ""
        print(f"[{i:>2}/{len(pairs)}] tm={p['tm']} pair={p['pair_idx']:>2} IoU={score:.3f} pred={pred_bbox} truth={p['truth_bbox']} {flag}{bs}  ({dt:.1f}s)")

    print(f"\nTotal LLM time: {t_total:.1f}s")
    if not results: return

    # Iter 7 — any-overlap recall (IoU > 0) is the user's success metric.
    # `hit`   = a non-blank-rejected pair with IoU > 0 vs truth.
    # `wrong` = a non-blank-rejected pair with IoU == 0 (model picked a bbox
    #           somewhere, but it doesn't overlap the truth).
    # `none`  = blank-rejected pair, OR a pair where the model returned no
    #           bbox at all (found=False from the LLM upstream).
    hit_n = sum(1 for r in results if not r["blank_reject"] and r["iou"] > 0)
    wrong_n = sum(1 for r in results
                  if not r["blank_reject"] and r["iou"] == 0 and r["pred_bbox"] is not None)
    none_n = sum(1 for r in results if r["pred_bbox"] is None)
    print("\nIter 7 success metric (any-overlap recall):")
    print(f"  hit   = {hit_n}/{len(results)} (model bbox overlaps truth)")
    print(f"  wrong = {wrong_n}/{len(results)} (model bbox lands in wrong region)")
    print(f"  none  = {none_n}/{len(results)} (no bbox: blank-reject or found=False)")
    print(f"  blank_rejects = {blank_rejects} (Pass-1 variance gate fired)\n")

    for thr in (0.0, 0.1, 0.3, 0.5, 0.7):
        hits = sum(1 for r in results if r["iou"] >= thr)
        print(f"  IoU ≥ {thr:.1f}: {hits}/{len(results)}  ({100*hits/len(results):.0f}%)")

    out = ROOT / f".data/score_runs/{time.strftime('%Y%m%d_%H%M%S')}_{args.model.replace('/','_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "verify": not args.no_verify,
        "results": [{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()} for r in results],
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWrote run report: {out}")


if __name__ == "__main__":
    main()
