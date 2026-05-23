"""Baseline eval: run the v0 matcher across all 10 docx fixtures.

For each sample:
  - LOGO   = first image (alphabetical by index prefix)
  - EXPECTED_CROP = last image
  - EVIDENCES = middle images
  - run match_logo_in_photo on each evidence, crop the best one (highest confidence),
    score it vs EXPECTED_CROP with pHash + SSIM.

Outputs eval/run_<n>/{report.html, metrics.json, crops/}.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from linebase.config import Settings
from linebase.crop import crop_to_bbox
from linebase.eval import PairScore, RunSummary, score_pair, write_html_report, write_metrics_json
from linebase.llm import match_logo_in_photo, MatchResult

FIXTURES = REPO / ".trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline/fixtures"
EVAL_ROOT = REPO / "eval"


def find_run_dir() -> Path:
    EVAL_ROOT.mkdir(exist_ok=True)
    existing = sorted(EVAL_ROOT.glob("run_*"))
    next_n = 1 + max([int(p.name.split("_")[1]) for p in existing if p.name.split("_")[1].isdigit()] + [0])
    d = EVAL_ROOT / f"run_{next_n:03d}"
    d.mkdir(parents=True)
    (d / "crops").mkdir()
    return d


def list_sample_images(sample_dir: Path) -> tuple[Path, list[Path], Path]:
    pngs = sorted(sample_dir.glob("*.png"))
    if len(pngs) < 3:
        raise ValueError(f"sample {sample_dir.name} has only {len(pngs)} images")
    return pngs[0], pngs[1:-1], pngs[-1]


def cost_estimate(usage: dict[str, int] | None) -> float:
    """Rough cost estimate; 1m1ng pricing is unknown, treat as gpt-4o-class ($2.5/M in, $10/M out)."""
    if not usage:
        return 0.0
    return usage["prompt_tokens"] * 2.5e-6 + usage["completion_tokens"] * 10e-6


def main() -> int:
    settings = Settings.from_env()
    run_dir = find_run_dir()
    print(f"[baseline] run_dir={run_dir}  model={settings.model}")

    sample_dirs = sorted([p for p in FIXTURES.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    pairs: list[PairScore] = []
    matched_count = 0
    total_cost = 0.0
    raw_log = []

    for s_dir in sample_dirs:
        sample = s_dir.name.removeprefix("sample_")
        logo, evidences, expected = list_sample_images(s_dir)
        print(f"\n[sample {sample}] logo={logo.name} expected={expected.name} evidence_count={len(evidences)}")

        best_result: MatchResult | None = None
        best_evidence: Path | None = None
        for ev in evidences:
            t0 = time.time()
            try:
                result = match_logo_in_photo(logo, ev, settings=settings)
            except Exception as e:
                print(f"  ! {ev.name}: ERROR {type(e).__name__}: {e}")
                continue
            dt = time.time() - t0
            cost = cost_estimate(result.usage)
            total_cost += cost
            raw_log.append({
                "sample": sample,
                "evidence": ev.name,
                "found": result.found,
                "bbox": result.bbox,
                "confidence": result.confidence,
                "reason": result.reason,
                "usage": result.usage,
                "cost_usd": cost,
                "latency_s": round(dt, 2),
            })
            mark = "[+]" if result.found else "[ ]"
            print(f"  {mark} {ev.name} found={result.found} conf={result.confidence:.2f} dt={dt:.1f}s")
            if result.found and (best_result is None or result.confidence > best_result.confidence):
                best_result = result
                best_evidence = ev

        if best_result is None or best_evidence is None or best_result.bbox is None:
            print(f"  -> no match on any evidence")
            pairs.append(PairScore(
                sample=sample, candidate="(none)", expected=expected.name,
                phash_distance=64, ssim=0.0, notes="no LLM match",
            ))
            continue

        # crop best evidence and copy expected for the report
        crops_dir = run_dir / "crops"
        mine = crops_dir / f"{sample}__mine.png"
        expected_copy = crops_dir / f"{sample}__expected.png"
        crop_to_bbox(best_evidence, best_result.bbox, mine)
        shutil.copy(expected, expected_copy)

        try:
            ph, sim = score_pair(mine, expected)
        except Exception as e:
            ph, sim = 64, 0.0
            print(f"  ! score error: {e}")
        pairs.append(PairScore(
            sample=sample, candidate=best_evidence.name, expected=expected.name,
            phash_distance=ph, ssim=sim,
            notes=f"conf={best_result.confidence:.2f} bbox={best_result.bbox}",
        ))
        matched_count += 1
        print(f"  -> best={best_evidence.name} bbox={best_result.bbox}  pHash={ph} SSIM={sim:.3f}")

    summary = RunSummary(
        run_id=run_dir.name,
        prompt_version=(pairs[0].notes.split()[0] if pairs else "v0"),
        model=settings.model,
        samples=len(sample_dirs),
        matched=matched_count,
        mean_ssim=sum(p.ssim for p in pairs) / max(1, len(pairs)),
        mean_phash=sum(p.phash_distance for p in pairs) / max(1, len(pairs)),
        pass_rate_ssim_50=sum(1 for p in pairs if p.ssim >= 0.5) / max(1, len(pairs)),
        cost_usd_estimate=total_cost,
        pairs=pairs,
    )

    html = write_html_report(summary, run_dir)
    metrics = write_metrics_json(summary, run_dir)
    (run_dir / "raw_log.json").write_text(json.dumps(raw_log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== baseline done ===")
    print(f"samples={summary.samples}  matched={summary.matched}  mean_ssim={summary.mean_ssim:.3f}")
    print(f"pass_rate@SSIM>=0.5={summary.pass_rate_ssim_50:.0%}  est_cost=${summary.cost_usd_estimate:.2f}")
    print(f"report: {html}")
    print(f"metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
