"""Direct-DB rerun of rows 79 (75537343) and 80 (74677567) using v_4 prompt.

Bypasses the running uvicorn (which is OLD code) by importing the pipeline
runner in-process. Required because:
  - Adding the verify_loop column would break the running server's Job
    deserializer; the live process is stuck on the pre-2026-05-24 schema.
  - The /api/jobs/{id}/rows/{rowId}/rerun endpoint doesn't exist on the
    running server, so a curl-based rerun isn't possible.

Budget-controlled strategy (hard cap $0.40 — verify-loop on both rows would
~5x that):
  - Row 79 (75537343 — brand-recognition bug): full LLM rerun with v_4
    prompt. NO verify-loop (would double cost). The v_4 prompt is the actual
    fix for "LLM matched the Heat fireball when the TM was a silhouette".
  - Row 80 (74677567 — blank-crop bug): NO LLM rerun. Instead, replay the
    existing (already-paid-for) bbox metadata through the new sanity check
    + crop-rebuild path to verify the blank crop gets rejected. The sanity
    check is the actual fix here — the LLM bbox itself was wrong (pointed
    into white margin) so a fresh LLM call wouldn't necessarily help; what
    we need is the post-crop reject-and-fall-back behavior.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force new prompt before importing the runner.
os.environ["LINEBASE_PROMPT_VERSION"] = "4"
# Verify-loop is DISABLED to stay within the $0.40 cost cap. The prompt
# change + sanity check are the headline fixes; verify-loop is a belt-and-
# braces add-on that the user can flip on via the new UI checkbox when
# they're willing to pay 2x.
os.environ["LINEBASE_VERIFY"] = "0"

from linebase import store  # noqa: E402
from linebase.config import Settings  # noqa: E402
from linebase.crop import crop_to_bbox  # noqa: E402
from linebase.pipeline_runner import (  # noqa: E402
    _crop_sanity_check,
    _process_row,
    DATA_DIR,
)
from linebase.fetch import fetch  # noqa: E402

JOB_ID = "2a2e801827dc457b"
ROW_LLM_RERUN = 79  # 75537343 — full LLM rerun with v_4 prompt
ROW_SANITY_REPLAY = 80  # 74677567 — replay existing bboxes through sanity check
OUT_DIAG = ROOT / "scripts" / "_e2e_out" / "morning_e2e" / "_diag_after_v4.txt"


async def run_llm_rerun(job: store.Job, row_id: int, settings: Settings, run_dir: Path) -> dict:
    """Full LLM rerun via _process_row. Resets the row first."""
    row = store.get_job_row(row_id)
    assert row is not None
    print(f"--- LLM rerun row {row_id} (appno {row.appno}) with v_4 prompt ---")
    store.update_job_row(
        row_id,
        status="pending",
        best_crop_path=None,
        all_crops_json="{}",
        match_meta_json="{}",
        human_status=None,
        notes=None,
    )
    t0 = time.time()
    try:
        cost_delta, terminal = await _process_row(job, store.get_job_row(row_id), settings, run_dir)
    except Exception as e:
        print(f"  EXC: {e}", file=sys.stderr)
        store.update_job_row(row_id, status="failed", notes=f"unhandled: {e}")
        terminal = "failed"
        cost_delta = 0.0
    dt = time.time() - t0
    fresh = store.get_job_row(row_id)
    return {
        "row_id": row_id, "appno": row.appno, "status": terminal,
        "cost": cost_delta, "dt": dt, "best_crop_path": fresh.best_crop_path,
        "match_meta": json.loads(fresh.match_meta_json or "{}"),
    }


async def replay_sanity_check(row_id: int, run_dir: Path) -> dict:
    """Replay the EXISTING bboxes through the new sanity-check path — zero LLM cost.

    Doesn't call the LLM. Re-crops each evidence using the bbox the LLM
    previously returned, runs the sanity check on each crop, picks the best
    surviving one, and writes the row back. This validates that the sanity
    check would have caught the blank-margin bug.
    """
    row = store.get_job_row(row_id)
    assert row is not None
    print(f"--- sanity-check replay row {row_id} (appno {row.appno}) ---")
    metas = json.loads(row.match_meta_json or "{}")
    evidences = json.loads(row.evidence_urls_json or "[]")
    new_crops: dict[str, str | None] = {}
    new_metas: dict[str, dict] = {}
    best_url: str | None = None
    best_conf = -1.0
    best_crop = None

    loop = asyncio.get_event_loop()
    for url in evidences:
        m = metas.get(url) or {}
        new_m = dict(m)
        bbox = m.get("bbox") if isinstance(m, dict) else None
        if not (m.get("found") and bbox):
            new_crops[url] = None
            new_metas[url] = new_m
            continue
        # Re-fetch evidence (uses cache so no network on hot rerun)
        try:
            ev_path = await loop.run_in_executor(None, fetch, url)
        except Exception as e:
            new_m["error"] = f"refetch: {e}"
            new_crops[url] = None
            new_metas[url] = new_m
            continue
        crop_out = run_dir / "images" / f"{row.appno or row.id}_{evidences.index(url) + 1}.png"
        try:
            await loop.run_in_executor(None, crop_to_bbox, ev_path, tuple(bbox), crop_out, 0.05)
            sanity = await loop.run_in_executor(None, _crop_sanity_check, crop_out, ev_path)
        except Exception as e:
            new_m["error"] = f"crop: {e}"
            new_crops[url] = None
            new_metas[url] = new_m
            continue
        if sanity is not None:
            new_m["sanity_rejected"] = sanity
            new_crops[url] = None
        else:
            new_crops[url] = str(crop_out)
            conf = float(m.get("confidence") or 0.0)
            if conf > best_conf:
                best_conf = conf
                best_url = url
                best_crop = str(crop_out)
        new_metas[url] = new_m

    status = "ok" if best_url else "needs_review"
    store.update_job_row(
        row_id,
        status=status,
        best_crop_path=best_crop,
        all_crops_json=json.dumps(new_crops),
        match_meta_json=json.dumps(new_metas, ensure_ascii=False),
        human_status=None,
        notes=None,
    )
    return {
        "row_id": row_id, "appno": row.appno, "status": status,
        "cost": 0.0, "dt": 0.0, "best_crop_path": best_crop,
        "match_meta": new_metas,
    }


async def main() -> int:
    # NB: we deliberately do NOT call store.init_schema() here. The migration
    # adds a verify_loop column that the still-running uvicorn (old code) can
    # no longer deserialize. We rely on env var LINEBASE_VERIFY=0 (no verify)
    # so the missing column doesn't matter — pipeline_runner's
    # `getattr(job, "verify_loop", 0)` returns 0 when the column is absent.
    settings = Settings.from_env()
    job = store.get_job(JOB_ID)
    if not job:
        print(f"ERR: job {JOB_ID} not found", file=sys.stderr)
        return 1

    run_dir = DATA_DIR / "runs" / JOB_ID
    (run_dir / "images").mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    results.append(await run_llm_rerun(job, ROW_LLM_RERUN, settings, run_dir))
    results.append(await replay_sanity_check(ROW_SANITY_REPLAY, run_dir))

    diag_lines: list[str] = []
    diag_lines.append(f"# v_4 rerun diag — job {JOB_ID}")
    diag_lines.append(f"# prompt=v_4, verify=disabled (cost cap), model={job.model or settings.model}")
    diag_lines.append(f"# generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    diag_lines.append("")

    total_cost = 0.0
    for r in results:
        appno = r["appno"]
        diag_lines.append(f"=== row {r['row_id']} (appno {appno}) ===")
        diag_lines.append(f"status={r['status']} cost=${r['cost']:.4f} dt={r['dt']:.1f}s")
        diag_lines.append(f"best_crop_path={r['best_crop_path']}")
        for url, m in r["match_meta"].items():
            if not isinstance(m, dict):
                diag_lines.append(f"  {url[-50:]}  error={m}")
                continue
            tag = url[-60:]
            ev_line = (
                f"  {tag}  found={m.get('found')} conf={m.get('confidence')} "
                f"bbox={m.get('bbox')} reason={(m.get('reason') or '')!r}"
            )
            if m.get("sanity_rejected"):
                ev_line += f"  SANITY_REJECTED={m['sanity_rejected']}"
            diag_lines.append(ev_line)
        diag_lines.append("")
        total_cost += r["cost"]

    diag_lines.append(f"# total LLM cost this rerun: ${total_cost:.4f}")
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIAG.write_text("\n".join(diag_lines), encoding="utf-8")
    print(f"\nDiag written to {OUT_DIAG}")

    print("\n=== summary ===")
    for r in results:
        print(
            f"  row {r['row_id']} ({r['appno']}): {r['status']}  "
            f"cost=${r['cost']:.4f}  crop={r['best_crop_path']}"
        )
    print(f"  total cost: ${total_cost:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
