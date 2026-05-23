"""End-to-end web test against the real 428 MB workbook.

Spawns a real uvicorn server in a subprocess (ASGITransport does NOT stream —
it buffers the entire response, which deadlocks against SSE endpoints that
never close their generator). Hits localhost via httpx over a real socket.

Hard-fails (with body printout) on the first non-2xx response or unexpected
state — no silent passes.

Usage:
    .venv/Scripts/python.exe -u scripts/e2e_real_xlsx.py

Optional env:
    LINEBASE_E2E_REUSE_JOB=<job_id>   # skip upload/create/start, only verify
                                       # the REST + SSE endpoints on the given
                                       # finished job. Avoids LLM cost on re-tries.
    LINEBASE_E2E_MODEL=<model_id>     # forwarded as the `model` field on
                                       # POST /api/jobs; defaults to unset
                                       # (server uses Settings.model).
    LINEBASE_E2E_N=<int>              # sample size for first_n (default 2).
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
OUT_DIR = REPO / "scripts" / "_e2e_out"

REUSE_JOB_ID = os.environ.get("LINEBASE_E2E_REUSE_JOB", "").strip()
E2E_MODEL = os.environ.get("LINEBASE_E2E_MODEL", "").strip()
E2E_N = int(os.environ.get("LINEBASE_E2E_N", "2"))


def _hard_fail(label: str, resp: httpx.Response) -> None:
    body = resp.text[:1000]
    print(f"[FAIL] {label}: HTTP {resp.status_code}")
    print(body)
    sys.exit(2)


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_health(port: int, timeout_s: float = 60.0) -> None:
    """Poll the server until it responds to any HTTP request, or give up."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/api/dev/eval-runs", timeout=2.0)
            if r.status_code < 500:
                return
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
            time.sleep(1.0)
    raise RuntimeError(f"uvicorn at :{port} did not come up in {timeout_s}s")


async def _consume_sse(client: httpx.AsyncClient, job_id: str, until_finished: bool) -> dict:
    """Drain the SSE stream. Returns the last seen progress/finished job dict."""
    last_job: dict = {}
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
                    if not until_finished and j["status"] == "finished":
                        return j
                elif et == "progress" and "row" in ev:
                    rw = ev["row"]
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
    """REST endpoints that don't depend on SSE: rows + xlsx + images.zip."""
    resp = await client.get(f"/api/jobs/{job_id}")
    if resp.status_code >= 300:
        _hard_fail("GET /api/jobs/{id}", resp)
    j = resp.json()
    print(f"[e2e] job state status={j['status']} done={j['done_rows']}/{j['total_rows']} "
          f"cost=${j['cost_usd']:.4f}")
    if j["done_rows"] < j["total_rows"]:
        print(f"[FAIL] done_rows={j['done_rows']} < total_rows={j['total_rows']}")
        sys.exit(2)

    resp = await client.get(f"/api/jobs/{job_id}/rows")
    if resp.status_code >= 300:
        _hard_fail("GET rows", resp)
    rows = resp.json()
    print(f"[e2e] rows returned: {len(rows)}")
    valid_statuses = {"ok", "needs_review", "failed", "bad"}
    for r_ in rows:
        st = r_.get("status")
        if st not in valid_statuses:
            print(f"[FAIL] row {r_['id']} has unexpected status: {st!r}")
            sys.exit(2)
        crop = r_.get("best_crop_path")
        crop_size = None
        if crop and Path(crop).exists():
            crop_size = Path(crop).stat().st_size
        print(f"  appno={r_.get('appno')!r}  status={st}  "
              f"best_conf={r_.get('best_confidence')}  "
              f"crop={crop!r}  crop_size={crop_size}")

    resp = await client.get(f"/api/jobs/{job_id}/xlsx", timeout=60.0)
    if resp.status_code >= 300:
        _hard_fail("GET /xlsx", resp)
    xlsx_out = OUT_DIR / "result.xlsx"
    xlsx_out.write_bytes(resp.content)
    if xlsx_out.stat().st_size <= 0:
        print("[FAIL] result.xlsx is empty")
        sys.exit(2)
    print(f"[e2e] xlsx downloaded: {xlsx_out} size={xlsx_out.stat().st_size}")

    resp = await client.get(f"/api/jobs/{job_id}/images.zip", timeout=60.0)
    if resp.status_code >= 300:
        _hard_fail("GET /images.zip", resp)
    zip_out = OUT_DIR / "images.zip"
    zip_out.write_bytes(resp.content)
    if zip_out.stat().st_size <= 0:
        print("[FAIL] images.zip is empty")
        sys.exit(2)
    print(f"[e2e] images.zip downloaded: {zip_out} size={zip_out.stat().st_size}")
    import zipfile
    with zipfile.ZipFile(zip_out) as zf:
        names = zf.namelist()
        print(f"[e2e] zip contains {len(names)} files:")
        for n in names:
            print(f"      {n}  ({zf.getinfo(n).file_size}B)")


async def _full_run(client: httpx.AsyncClient) -> int:
    if not WORKBOOK.exists():
        print(f"[FAIL] workbook missing: {WORKBOOK}")
        return 2
    size_mb = WORKBOOK.stat().st_size / (1024 * 1024)
    print(f"[e2e] workbook={WORKBOOK.name} size={size_mb:.1f}MB")

    # ---- Step 1: upload ----
    t0 = time.time()
    with WORKBOOK.open("rb") as fh:
        files = {"file": ("workbook.xlsx", fh,
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        resp = await client.post("/api/uploads", files=files, timeout=300.0)
    if resp.status_code >= 300:
        _hard_fail("POST /api/uploads", resp)
    upload = resp.json()
    print(f"[e2e] upload ok in {time.time()-t0:.1f}s  id={upload['id']}  "
          f"size={upload['size']}  sheets={len(upload['sheets'])}")

    # ---- Step 2: confirm sheets ----
    sheet_names = [s["name"] for s in upload["sheets"]]
    print(f"[e2e] sheets={sheet_names}")
    if TARGET_SHEET not in sheet_names:
        print(f"[FAIL] target sheet {TARGET_SHEET!r} not in upload sheets")
        return 2
    target = next(s for s in upload["sheets"] if s["name"] == TARGET_SHEET)
    print(f"[e2e] target sheet rows={target['rows']} cols={target['columns']}")

    # ---- Step 3: create job ----
    job_req: dict = {
        "upload_id": upload["id"],
        "sheet_name": TARGET_SHEET,
        "appno_column": "B",
        "logo_column": "D",
        "evidence_column": "K",
        "sample_kind": "first_n",
        "sample_params": {"n": E2E_N},
        "threshold": 0.5,
    }
    if E2E_MODEL:
        job_req["model"] = E2E_MODEL
    resp = await client.post("/api/jobs", json=job_req)
    if resp.status_code >= 300:
        _hard_fail("POST /api/jobs", resp)
    job = resp.json()
    job_id = job["id"]
    print(f"[e2e] job created id={job_id} total_rows={job['total_rows']} "
          f"model={job.get('model')!r}")

    # ---- Step 4: start job ----
    resp = await client.post(f"/api/jobs/{job_id}/start")
    if resp.status_code >= 300:
        _hard_fail("POST /api/jobs/.../start", resp)
    print(f"[e2e] job started status={resp.json().get('status')}")

    # ---- Step 5: drain SSE ----
    print("[e2e] streaming events...")
    final_job = await _consume_sse(client, job_id, until_finished=True)
    if final_job.get("status") != "finished":
        print(f"[FAIL] job did not reach 'finished' state: {final_job!r}")
        return 2

    # ---- Steps 6-9: downloads ----
    await _verify_downloads(client, job_id)
    print(f"\n[e2e] DONE   job_id={job_id}")
    return 0


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    port = _pick_free_port()
    print(f"[e2e] starting uvicorn on 127.0.0.1:{port}")
    log_path = OUT_DIR / "uvicorn.log"
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
            if REUSE_JOB_ID:
                print(f"[e2e] REUSE_JOB_ID={REUSE_JOB_ID} — skipping upload/create/start")
                # Verify the SSE consumer can still parse a snapshot from a
                # finished job. On a finished job stream_events emits one
                # snapshot then blocks on the empty queue — we bail out as
                # soon as that snapshot arrives.
                print("[e2e] streaming events (reuse mode, will bail after snapshot)...")
                try:
                    await asyncio.wait_for(
                        _consume_sse(client, REUSE_JOB_ID, until_finished=False),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    print("[FAIL] no SSE snapshot in 30s for reuse-mode job")
                    return 2
                await _verify_downloads(client, REUSE_JOB_ID)
                print(f"\n[e2e] reuse-mode DONE   job_id={REUSE_JOB_ID}")
                return 0
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
