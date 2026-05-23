"""Eval harness: compare my crop vs the docx "expected crop" ground truth.

Metrics:
  - pHash distance (perceptual hash) — small = similar
  - SSIM after resizing both to a common canvas — 1.0 = identical

Reports HTML side-by-side under eval/run_<n>/report.html and JSON metrics under metrics.json.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


@dataclass
class PairScore:
    sample: str
    candidate: str  # filename of evidence image we tried
    expected: str   # filename of expected crop
    phash_distance: int
    ssim: float
    notes: str = ""
    # v2+: red-box bbox accuracy.
    # iou_vs_redbox: IoU between LLM bbox and the red-box annotation on the gold evidence.
    #   None when there is no red-box ground truth OR the LLM picked the wrong evidence.
    iou_vs_redbox: float | None = None
    # selection_correct: True iff LLM's best evidence == the human-marked gold evidence.
    #   None when no gold exists for this sample.
    selection_correct: bool | None = None


def _normalize(img: Image.Image, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    arr = np.array(img.convert("L").resize(size, Image.LANCZOS))
    return arr.astype(np.float32) / 255.0


def score_pair(my_crop_path: Path, expected_crop_path: Path) -> tuple[int, float]:
    a = Image.open(my_crop_path).convert("RGB")
    b = Image.open(expected_crop_path).convert("RGB")
    ph_a = imagehash.phash(a)
    ph_b = imagehash.phash(b)
    phash_dist = int(ph_a - ph_b)
    sim = float(ssim(_normalize(a), _normalize(b), data_range=1.0))
    return phash_dist, sim


@dataclass
class RunSummary:
    run_id: str
    prompt_version: str
    model: str
    samples: int
    matched: int  # how many samples produced a crop at all
    mean_ssim: float
    mean_phash: float
    pass_rate_ssim_50: float  # fraction with SSIM >= 0.5
    cost_usd_estimate: float
    pairs: list[PairScore]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pairs"] = [asdict(p) for p in self.pairs]
        return d


def write_html_report(summary: RunSummary, run_dir: Path) -> Path:
    rows = []
    for p in summary.pairs:
        ph = p.phash_distance
        s = p.ssim
        # row colour driven by selection_correct (primary signal); fall back to SSIM (legacy)
        if p.selection_correct is True:
            cls = "ok"
        elif p.selection_correct is False:
            cls = "bad"
        else:
            cls = "ok" if s >= 0.5 else "bad"
        if p.selection_correct is None:
            sel_cell = "<span class='dim'>n/a</span>"
        elif p.selection_correct:
            sel_cell = "<span class='ok-mark'>OK</span>"
        else:
            sel_cell = "<span class='bad-mark'>WRONG</span>"
        if p.iou_vs_redbox is None:
            iou_cell = "<span class='dim'>n/a</span>"
        else:
            iou_cls = "ok-mark" if p.iou_vs_redbox >= 0.5 else "bad-mark"
            iou_cell = f"<span class='{iou_cls}'>{p.iou_vs_redbox:.2f}</span>"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{p.sample}</td>"
            f"<td><img src='./crops/{p.sample}__mine.png' style='max-height:160px'></td>"
            f"<td><img src='./crops/{p.sample}__expected.png' style='max-height:160px'></td>"
            f"<td>{sel_cell}</td>"
            f"<td>{iou_cell}</td>"
            f"<td class='secondary'>{ph}</td>"
            f"<td class='secondary'>{s:.3f}</td>"
            f"<td>{p.notes}</td>"
            f"</tr>"
        )
    summary_top = (
        f"prompt={summary.prompt_version} model={summary.model} "
        f"samples={summary.samples} matched={summary.matched}"
    )
    summary_sec = (
        f"mean_ssim={summary.mean_ssim:.3f} mean_phash={summary.mean_phash:.1f} "
        f"pass_rate@SSIM>=0.5={summary.pass_rate_ssim_50:.0%}"
    )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>linebase eval {summary.run_id}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; padding: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px; text-align: center; }}
  tr.bad {{ background: #fee; }}
  tr.ok {{ background: #efe; }}
  .summary {{ background: #f5f5f5; padding: 12px; margin-bottom: 12px; }}
  .secondary {{ color: #888; font-size: 11px; }}
  .dim {{ color: #aaa; }}
  .ok-mark {{ color: #1a7f1a; font-weight: 600; }}
  .bad-mark {{ color: #b00020; font-weight: 600; }}
</style>
<h1>linebase eval — {summary.run_id}</h1>
<div class="summary">
  {summary_top}<br>
  <strong>selection / bbox-IoU are primary;</strong> SSIM/pHash are secondary sanity checks.<br>
  <span class="secondary">{summary_sec}</span><br>
  est_cost=${summary.cost_usd_estimate:.3f}
</div>
<table>
  <tr>
    <th>sample</th><th>mine</th><th>expected</th>
    <th>selection</th><th>IoU(red-box)</th>
    <th class="secondary">pHash</th><th class="secondary">SSIM</th>
    <th>notes</th>
  </tr>
  {''.join(rows)}
</table>
"""
    p = run_dir / "report.html"
    p.write_text(html, encoding="utf-8")
    return p


def write_metrics_json(summary: RunSummary, run_dir: Path) -> Path:
    p = run_dir / "metrics.json"
    p.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
