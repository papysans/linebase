"""Pull a finished job's rows and score each evidence URL's bbox against the
truth-set ground-truth bbox. Emits a hit/wrong/none table + per-row detail
including provenance (retried, tile_scanned, pass1_blank_reject).
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = sys.argv[1] if len(sys.argv) > 1 else None
if not JOB:
    sys.exit("usage: score_e2e_against_truth.py <job_id>")

# Truth-set: appno → {evidence_filename: truth_bbox}
truth_index = json.loads((ROOT / "docs" / "truth_set" / "INDEX.json").read_text(encoding="utf-8"))
truth_by_tm: dict[str, dict[str, list[int]]] = {}
for b in truth_index:
    truth_by_tm[b["tm"]] = {}
    for p in b["pairs"]:
        if p.get("score", 0) >= 0.85 and p.get("truth_bbox"):
            # key by the evidence filename (e.g. "evidence.png") plus the pair dir
            ev_name = Path(p["evidence"]).name
            pair_dir = Path(p["evidence"]).parent.name  # pair_01 etc
            key = f"{pair_dir}/{ev_name}"
            truth_by_tm[b["tm"]][key] = p["truth_bbox"]


def iou(a, b):
    if not a or not b: return 0.0
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    A = max(0,ax2-ax1)*max(0,ay2-ay1); B = max(0,bx2-bx1)*max(0,by2-by1)
    return inter / (A+B-inter) if (A+B-inter) > 0 else 0.0


def url_to_pair_key(url: str) -> str | None:
    """Extract 'pair_NN/evidence.png' from .../truth_set/{tm}_{cat}/pair_NN/evidence.png"""
    decoded = urllib.parse.unquote(url)
    parts = decoded.rstrip("/").split("/")
    if len(parts) < 2: return None
    return "/".join(parts[-2:])


rows = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8765/api/jobs/{JOB}/rows").read())
job_info = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8765/api/jobs/{JOB}").read())
print(f"=== Job {JOB} model={job_info['model']} verify={job_info['verify_loop']} tile_scan={job_info.get('tile_scan')} ===\n")

hits, wrongs, nones = 0, 0, 0
covered = 0
per_row = []
for r in rows:
    tm = r["appno"]
    meta = r.get("match_meta", {}) or {}
    truth_pairs = truth_by_tm.get(tm, {})
    print(f"\n--- tm={tm} status={r['status']} best_crop={'YES' if r.get('best_crop_path') else 'no'} ---")
    for url, info in meta.items():
        if not isinstance(info, dict): continue
        key = url_to_pair_key(url)
        truth = truth_pairs.get(key)
        if truth is None: continue  # this evidence isn't in our truth-set
        covered += 1
        pred = info.get("bbox")
        ov = iou(pred, truth)
        if pred is None:
            label = "NONE"; nones += 1
        elif ov > 0:
            label = "HIT"; hits += 1
        else:
            label = "WRONG"; wrongs += 1
        prov = []
        if info.get("retried"): prov.append("retried")
        if info.get("tile_scanned"): prov.append(f"tile@{info.get('tile_verified_idx') or info.get('tile_index')}")
        if info.get("pass1_blank_reject"): prov.append("blank-reject")
        if info.get("verified") is False and not info.get("tile_scanned"): prov.append("verify-rej")
        prov_str = " ".join(prov) if prov else "(no-special)"
        ov_str = f"iou={ov:.2f}" if pred else ""
        print(f"  [{label:<5}] {key:<35} pred={pred} truth={truth} {ov_str} {prov_str}")
        per_row.append({"tm": tm, "key": key, "pred": pred, "truth": truth, "iou": ov, "label": label, "info": info})

print(f"\n=== Totals (over {covered} truth-pair evidence URLs in this job) ===")
print(f"  HIT   = {hits}/{covered}  ({100*hits/covered:.0f}%)")
print(f"  WRONG = {wrongs}/{covered} ({100*wrongs/covered:.0f}%)")
print(f"  NONE  = {nones}/{covered}  ({100*nones/covered:.0f}%)")

# also write to .data
out = ROOT / f".data/score_runs/e2e_{JOB[:8]}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"job_id": JOB, "hits": hits, "wrong": wrongs, "none": nones, "covered": covered, "per_row": per_row}, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
print(f"\nWrote {out}")
