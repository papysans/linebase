"""Drain in-flight job 2a2e801827dc457b and capture artifacts.

Polls the live server (do NOT restart) until the job is finished/failed,
then dumps job.json / rows.json / result.xlsx / images.zip to
scripts/_e2e_out/morning_e2e/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

JOB_ID = "2a2e801827dc457b"
BASE = "http://127.0.0.1:8765"
OUT = Path(__file__).resolve().parent / "_e2e_out" / "morning_e2e"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print(f"[drain] polling {JOB_ID} -> {OUT}", flush=True)
    while True:
        try:
            j = httpx.get(f"{BASE}/api/jobs/{JOB_ID}", timeout=10).json()
        except Exception as e:
            print(f"[drain] poll error: {e!r}", flush=True)
            time.sleep(10)
            continue
        print(
            f"[drain] {time.strftime('%H:%M:%S')} {j['status']} "
            f"{j['done_rows']}/{j['total_rows']} cost=${j['cost_usd']:.3f}",
            flush=True,
        )
        if j["status"] in ("finished", "failed"):
            break
        time.sleep(20)

    # capture artifacts
    print("[drain] capturing artifacts...", flush=True)
    job = httpx.get(f"{BASE}/api/jobs/{JOB_ID}", timeout=30).json()
    (OUT / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = httpx.get(f"{BASE}/api/jobs/{JOB_ID}/rows", timeout=60).json()
    (OUT / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    xlsx = httpx.get(f"{BASE}/api/jobs/{JOB_ID}/xlsx", timeout=120)
    (OUT / "result.xlsx").write_bytes(xlsx.content)
    print(f"[drain] xlsx {len(xlsx.content)} bytes", flush=True)

    zipr = httpx.get(f"{BASE}/api/jobs/{JOB_ID}/images.zip", timeout=120)
    (OUT / "images.zip").write_bytes(zipr.content)
    print(f"[drain] zip {len(zipr.content)} bytes", flush=True)

    # Summary
    ok = sum(1 for r in rows if (r.get("human_status") or r.get("status")) == "ok")
    nr = sum(1 for r in rows if (r.get("human_status") or r.get("status")) == "needs_review")
    failed = sum(1 for r in rows if (r.get("human_status") or r.get("status")) == "failed")
    bad = sum(1 for r in rows if (r.get("human_status") or r.get("status")) == "bad")
    print(
        f"[drain] summary ok={ok} nr={nr} bad={bad} failed={failed} "
        f"cost=${job['cost_usd']:.3f}",
        flush=True,
    )
    print("[drain] per-appno best_confidence:", flush=True)
    for r in rows:
        print(
            f"  row {r['row_index']:>3} appno={r.get('appno')!s:<14} "
            f"status={(r.get('human_status') or r.get('status'))!s:<14} "
            f"conf={r.get('best_confidence')!s:<6} "
            f"fb={r.get('best_fallback_model')!s}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
