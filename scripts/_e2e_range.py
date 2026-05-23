"""Variant of e2e_real_xlsx.py that uses range sampling to avoid the
slow 20+/27-evidence rows. Aims to get more OK rows in less wall clock.

Env:
    LINEBASE_RANGE_START (default 8)
    LINEBASE_RANGE_END   (default 13)
    LINEBASE_E2E_MODEL   (default doubao-seed-2-0-pro-260215)
    LINEBASE_PER_ROW_HARD_SEC (default 90; abort row if it exceeds)

Writes artifacts directly into scripts/_e2e_out/night_run_v2/.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import httpx  # noqa: E402

WORKBOOK = REPO / "美专实物图排查-2026.2.6.xlsx"
TARGET_SHEET = "图形商标tro"
OUT_DIR = REPO / "scripts" / "_e2e_out" / "night_run_v2"

RANGE_START = int(os.environ.get("LINEBASE_RANGE_START", "8"))
RANGE_END = int(os.environ.get("LINEBASE_RANGE_END", "13"))
# Specific row_indices to test — defaults skip rows 3 (20 ev), 5 (27 ev), 6 (19 ev),
# 9 (19 ev), 10 (15 ev), 11 (12 ev) which were taking >5 min each on doubao-pro tonight.
# Keeps the cheap ones (1, 2, 3, 6, 8, 9 evidence rows) for fast turnaround.
ROW_IDS = os.environ.get("LINEBASE_ROW_IDS", "7,8,12,13,14,15")
E2E_MODEL = os.environ.get("LINEBASE_E2E_MODEL", "doubao-seed-2-0-pro-260215").strip()
PER_ROW_HARD_SEC = float(os.environ.get("LINEBASE_PER_ROW_HARD_SEC", "180"))
COST_HARD_CAP = 0.10


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_health(port: int, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/api/dev/eval-runs", timeout=2.0)
            if r.status_code < 500:
                return
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
            time.sleep(1.0)
    raise RuntimeError(f"uvicorn at :{port} did not come up in {timeout_s}s")


async def _consume_sse(client: httpx.AsyncClient, job_id: str) -> dict:
    last_job: dict = {}
    last_row_start_t: dict[int, float] = {}
    t0 = time.time()
    async with client.stream(
        "GET", f"/api/jobs/{job_id}/events",
        timeout=httpx.Timeout(900.0, connect=10.0),
    ) as r:
        if r.status_code >= 300:
            body = (await r.aread()).decode("utf-8", errors="replace")[:1000]
            print(f"[FAIL] GET events HTTP {r.status_code}: {body}")
            sys.exit(2)
        buf = ""
        async for chunk in r.aiter_text():
            buf += chunk
            buf = buf.replace("\r\n", "\n")
            while "\n\n" in buf:
                raw_event, buf = buf.split("\n\n", 1)
                data_lines = [
                    line[len("data:"):].strip()
                    for line in raw_event.splitlines()
                    if line.startswith("data:")
                ]
                if not data_lines:
                    continue
                try:
                    ev = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    continue
                t_e = time.time() - t0
                et = ev.get("type")
                if et == "progress" and "job" in ev:
                    j = ev["job"]
                    last_job = j
                    print(f"  [{t_e:6.1f}s] progress  status={j['status']}  "
                          f"{j['done_rows']}/{j['total_rows']}  "
                          f"cost=${j['cost_usd']:.4f}")
                    if j["cost_usd"] > COST_HARD_CAP:
                        print(f"  [!] COST CAP TRIPPED ${j['cost_usd']:.4f} > ${COST_HARD_CAP}")
                        return j
                elif et == "progress" and "row" in ev:
                    rw = ev["row"]
                    last_row_start_t[rw["id"]] = time.time()
                    print(f"  [{t_e:6.1f}s] row.start row_id={rw['id']} "
                          f"appno={rw.get('appno')} status={rw['status']}")
                elif et in ("row_done", "row_failed"):
                    rw = ev["row"]
                    print(f"  [{t_e:6.1f}s] {et:10s} row_id={rw['id']} "
                          f"appno={rw.get('appno')} status={rw['status']} "
                          f"best_crop={rw.get('best_crop_path')!r} "
                          f"best_conf={rw.get('best_confidence')}")
                elif et == "finished":
                    j = ev["job"]
                    last_job = j
                    print(f"  [{t_e:6.1f}s] FINISHED status={j['status']} "
                          f"done={j['done_rows']}/{j['total_rows']} "
                          f"cost=${j['cost_usd']:.4f}")
                    return j
                else:
                    print(f"  [{t_e:6.1f}s] event {ev}")
    return last_job


async def _verify_downloads(client: httpx.AsyncClient, job_id: str) -> None:
    resp = await client.get(f"/api/jobs/{job_id}")
    j = resp.json()
    print(f"[e2e] job state status={j['status']} done={j['done_rows']}/{j['total_rows']} "
          f"cost=${j['cost_usd']:.4f}")
    (OUT_DIR / "job_id.txt").write_text(job_id)

    resp = await client.get(f"/api/jobs/{job_id}/rows")
    rows = resp.json()
    (OUT_DIR / "rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[e2e] rows.json written, n={len(rows)}")

    # summary
    from collections import Counter
    dist = Counter(r["status"] for r in rows)
    summary = {
        "job_id": job_id,
        "model": j["model"],
        "status": j["status"],
        "total_rows": j["total_rows"],
        "done_rows": j["done_rows"],
        "cost_usd_calibrated": j["cost_usd"],
        "distribution": dict(dist),
        "sample_kind": "row_ids",
        "row_ids": ROW_IDS,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        resp = await client.get(f"/api/jobs/{job_id}/xlsx", timeout=60.0)
        if resp.status_code < 300:
            (OUT_DIR / "result.xlsx").write_bytes(resp.content)
            print(f"[e2e] result.xlsx size={(OUT_DIR / 'result.xlsx').stat().st_size}")
    except Exception as e:
        print(f"[e2e] xlsx download failed: {e}")

    try:
        resp = await client.get(f"/api/jobs/{job_id}/images.zip", timeout=60.0)
        if resp.status_code < 300:
            (OUT_DIR / "images.zip").write_bytes(resp.content)
            print(f"[e2e] images.zip size={(OUT_DIR / 'images.zip').stat().st_size}")
    except Exception as e:
        print(f"[e2e] zip download failed: {e}")


async def _full_run(client: httpx.AsyncClient) -> int:
    if not WORKBOOK.exists():
        print(f"[FAIL] workbook missing: {WORKBOOK}")
        return 2
    size_mb = WORKBOOK.stat().st_size / (1024 * 1024)
    print(f"[e2e] workbook={WORKBOOK.name} size={size_mb:.1f}MB")

    # upload
    t0 = time.time()
    with WORKBOOK.open("rb") as fh:
        files = {"file": ("workbook.xlsx", fh,
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        resp = await client.post("/api/uploads", files=files, timeout=300.0)
    if resp.status_code >= 300:
        print(f"[FAIL] upload: {resp.status_code} {resp.text[:500]}")
        return 2
    upload = resp.json()
    print(f"[e2e] upload ok in {time.time()-t0:.1f}s  id={upload['id']}")

    # create job with row_ids sample (precise control over which rows we test)
    ids = [int(x) for x in ROW_IDS.split(",") if x.strip()]
    job_req: dict = {
        "upload_id": upload["id"],
        "sheet_name": TARGET_SHEET,
        "appno_column": "B",
        "logo_column": "D",
        "evidence_column": "K",
        "sample_kind": "row_ids",
        "sample_params": {"ids": ids},
        "threshold": 0.5,
        "model": E2E_MODEL,
    }
    resp = await client.post("/api/jobs", json=job_req)
    if resp.status_code >= 300:
        print(f"[FAIL] create job: {resp.status_code} {resp.text[:500]}")
        return 2
    job = resp.json()
    job_id = job["id"]
    print(f"[e2e] job created id={job_id} total_rows={job['total_rows']} model={job.get('model')!r}")

    # start
    resp = await client.post(f"/api/jobs/{job_id}/start")
    if resp.status_code >= 300:
        print(f"[FAIL] start job: {resp.status_code} {resp.text[:500]}")
        return 2
    print(f"[e2e] job started")

    # drain
    print("[e2e] streaming events...")
    final_job = await _consume_sse(client, job_id)
    print(f"[e2e] final state: {final_job.get('status')}")
    await _verify_downloads(client, job_id)
    print(f"\n[e2e] DONE   job_id={job_id}")
    return 0


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    port = _pick_free_port()
    print(f"[e2e] starting uvicorn on 127.0.0.1:{port}")
    log_path = OUT_DIR / "uvicorn2.log"
    log_fh = log_path.open("wb")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn", "linebase.server:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"],
        cwd=str(REPO),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_health(port)
        print(f"[e2e] uvicorn up — base http://127.0.0.1:{port}")
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=httpx.Timeout(600.0, connect=10.0),
        ) as client:
            return await _full_run(client)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            log_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
