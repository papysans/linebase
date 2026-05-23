"""FastAPI app: REST API + SSE + static SPA mount."""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from linebase import store
from linebase.config import Settings
from linebase.fetch import fetch
from linebase.io_excel import inspect_workbook, iter_rows, write_result_workbook
from linebase.models_catalog import MODEL_WHITELIST, is_model_routable, to_dict as model_to_dict
from linebase.pipeline_runner import (
    _job_to_dict,
    _row_to_dict,
    start_job,
    stream_events,
)

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = store.DATA_DIR
UPLOADS_DIR = DATA_DIR / "uploads"
RUNS_DIR = DATA_DIR / "runs"
STATIC_DIR = Path(__file__).resolve().parent / "static"


app = FastAPI(title="linebase", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    store.init_schema()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


# --- Models catalog --------------------------------------------------------


@app.get("/api/models")
def list_models() -> dict:
    """Return the curated model whitelist + the system default + a custom-allowed flag.

    The frontend renders this as a dropdown with a "custom" escape hatch. The
    server validates the chosen model on POST /api/jobs via `is_model_routable`.
    """
    settings = Settings.from_env()
    return {
        "whitelist": [model_to_dict(opt) for opt in MODEL_WHITELIST],
        "default": settings.model,
        "allow_custom": True,
    }


# --- Upload ----------------------------------------------------------------

class UploadResponse(BaseModel):
    id: str
    filename: str
    size: int
    sheets: list[dict]


@app.post("/api/uploads", response_model=UploadResponse)
async def post_upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(400, "expected .xlsx file")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    body = await file.read()
    uid = store.new_id()
    target = UPLOADS_DIR / f"{uid}_{file.filename}"
    target.write_bytes(body)
    upload = store.insert_upload(filename=file.filename, size=len(body), path=str(target))
    sheets = [s.__dict__ for s in inspect_workbook(target)]
    store.set_upload_sheets(upload.id, json.dumps(sheets, ensure_ascii=False))
    return UploadResponse(id=upload.id, filename=upload.filename, size=upload.size, sheets=sheets)


@app.get("/api/uploads/{upload_id}", response_model=UploadResponse)
def get_upload(upload_id: str) -> UploadResponse:
    u = store.get_upload(upload_id)
    if not u:
        raise HTTPException(404, "upload not found")
    sheets = json.loads(u.sheets_json or "[]")
    return UploadResponse(id=u.id, filename=u.filename, size=u.size, sheets=sheets)


# --- Jobs ------------------------------------------------------------------

class CreateJobRequest(BaseModel):
    upload_id: str
    sheet_name: str
    appno_column: str
    logo_column: str
    evidence_column: str
    sample_kind: str  # "first_n" | "range" | "row_ids"
    sample_params: dict
    threshold: float = 0.5
    # Per-job model override. None → fall back to Settings.from_env().model.
    # Validated in create_job(): must be in MODEL_WHITELIST or routable via
    # _PROVIDER_PREFIXES — otherwise 400.
    model: str | None = None


def _resolve_rows(upload: store.Upload, req: CreateJobRequest) -> list[dict]:
    path = Path(upload.path)
    if req.sample_kind == "first_n":
        n = int(req.sample_params.get("n", 10))
        # rows below header. Header detection: assume row 1 = header, data starts at row 2.
        # But some sheets (图形商标tro) have row 2 also as a Chinese header — we skip rows 2-3 if hidden.
        return iter_rows(path, req.sheet_name, req.appno_column, req.logo_column, req.evidence_column,
                         start_row=2, end_row=2 + n - 1 + 2)
    if req.sample_kind == "range":
        start = int(req.sample_params["start"])
        end = int(req.sample_params["end"])
        return iter_rows(path, req.sheet_name, req.appno_column, req.logo_column, req.evidence_column,
                         start_row=start, end_row=end)
    if req.sample_kind == "row_ids":
        ids: list[int] = req.sample_params["ids"]
        start = min(ids); end = max(ids)
        all_rows = iter_rows(path, req.sheet_name, req.appno_column, req.logo_column, req.evidence_column,
                             start_row=start, end_row=end)
        return [r for r in all_rows if r["row_index"] in ids]
    raise HTTPException(400, f"unknown sample_kind: {req.sample_kind}")


@app.post("/api/jobs")
def create_job(req: CreateJobRequest) -> dict:
    upload = store.get_upload(req.upload_id)
    if not upload:
        raise HTTPException(404, "upload not found")
    # Reject unroutable model ids early so users see the failure at job-creation
    # time, not as a per-row LLM error 5 minutes into the run.
    if req.model is not None and req.model.strip():
        candidate = req.model.strip()
        if not is_model_routable(candidate):
            raise HTTPException(
                400,
                f"model {candidate!r} is not in the whitelist and does not match any routable provider prefix",
            )
        model_value: str | None = candidate
    else:
        model_value = None
    job = store.insert_job(
        upload_id=req.upload_id, sheet_name=req.sheet_name,
        logo_column=req.logo_column, evidence_column=req.evidence_column,
        appno_column=req.appno_column, threshold=req.threshold,
        sample_kind=req.sample_kind, sample_params=req.sample_params,
        model=model_value,
    )
    rows_data = _resolve_rows(upload, req)
    # filter out blatantly empty rows (no logo URL AND no evidence)
    rows_data = [r for r in rows_data if r["logo_url"] or r["evidence_urls"]]
    for r in rows_data:
        store.insert_job_row(
            job_id=job.id, row_index=int(r["row_index"]),
            appno=r.get("appno"), logo_url=r.get("logo_url"),
            evidence_urls=r["evidence_urls"] or [],
        )
    store.update_job(job.id, total_rows=len(rows_data))
    return _job_to_dict(store.get_job(job.id))  # type: ignore[arg-type]


@app.post("/api/jobs/{job_id}/start")
async def start(job_id: str) -> dict:
    # NB: async def is required — start_job calls asyncio.get_event_loop().
    # When this route was sync, FastAPI dispatched it on the anyio thread pool
    # where get_event_loop() raises "no current event loop in thread 'AnyIO
    # worker thread'" on Python 3.11+. See e2e_real_xlsx.py for the repro.
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404)
    upload = store.get_upload(job.upload_id)
    assert upload is not None
    settings = Settings.from_env()
    start_job(job_id, Path(upload.path), settings)
    return _job_to_dict(store.get_job(job_id))  # type: ignore[arg-type]


@app.get("/api/jobs/{job_id}")
def get_job_view(job_id: str) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404)
    return _job_to_dict(job)


@app.get("/api/jobs/{job_id}/rows")
def list_rows(job_id: str, status: str | None = None) -> list[dict]:
    rows = store.list_job_rows(job_id, status=status)
    return [_row_to_dict(r) for r in rows]


class SetStatusRequest(BaseModel):
    human_status: str  # "ok" | "bad" | "needs_review"
    notes: str | None = None


@app.post("/api/jobs/{job_id}/rows/{row_id}/status")
def set_row_status(job_id: str, row_id: int, req: SetStatusRequest) -> dict:
    row = store.get_job_row(row_id)
    if not row or row.job_id != job_id:
        raise HTTPException(404)
    store.update_job_row(row_id, human_status=req.human_status, notes=req.notes)
    return _row_to_dict(store.get_job_row(row_id))  # type: ignore[arg-type]


class RerunRequest(BaseModel):
    row_ids: list[int] | None = None


@app.post("/api/jobs/{job_id}/rerun")
async def rerun(job_id: str, req: RerunRequest) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404)
    # reset selected rows to pending; if no list, reset all rows that aren't human-OK
    rows = store.list_job_rows(job_id)
    targets = [r for r in rows if (req.row_ids is None and r.human_status != "ok") or (req.row_ids and r.id in req.row_ids)]
    for r in targets:
        store.update_job_row(r.id, status="pending", best_crop_path=None, all_crops_json="{}", match_meta_json="{}")
    store.update_job(job_id, status="pending")
    upload = store.get_upload(job.upload_id)
    assert upload is not None
    settings = Settings.from_env()
    start_job(job_id, Path(upload.path), settings)
    return _job_to_dict(store.get_job(job_id))  # type: ignore[arg-type]


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str, request: Request) -> EventSourceResponse:
    async def gen():
        async for ev in stream_events(job_id):
            if await request.is_disconnected():
                return
            yield {"data": json.dumps(ev, ensure_ascii=False)}
    return EventSourceResponse(gen())


@app.get("/api/jobs/{job_id}/xlsx")
def download_xlsx(job_id: str) -> FileResponse:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404)
    upload = store.get_upload(job.upload_id)
    assert upload is not None
    rows = store.list_job_rows(job_id)
    rows_data = []
    for r in rows:
        meta = json.loads(r.match_meta_json or "{}")
        # find confidence for best_crop_path
        conf = 0.0
        for m in meta.values():
            if isinstance(m, dict) and m.get("confidence", 0) > conf:
                conf = m.get("confidence", 0)
        rows_data.append({
            "row_index": r.row_index, "appno": r.appno, "logo_url": r.logo_url,
            "evidence_urls": json.loads(r.evidence_urls_json),
            "status": r.human_status or r.status,
            "best_crop_path": r.best_crop_path, "confidence": conf, "notes": r.notes or "",
        })
    out = RUNS_DIR / job_id / f"result_{int(time.time())}.xlsx"
    write_result_workbook(Path(upload.path), out, rows_data)
    return FileResponse(out, filename=out.name, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/api/jobs/{job_id}/images.zip")
def download_images(job_id: str, status: str | None = None) -> StreamingResponse:
    rows = store.list_job_rows(job_id)
    if status:
        # Normalise: frontend may send "OK" / "BAD" / "NEEDS_REVIEW"; the DB
        # stores lowercase ("ok" / "bad" / "needs_review").
        wanted = status.strip().lower()
        rows = [r for r in rows if (r.human_status or r.status or "").lower() == wanted]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            crops = json.loads(r.all_crops_json or "{}")
            idx = 0
            for url, path in crops.items():
                if not path or not Path(path).exists():
                    continue
                idx += 1
                ext = Path(path).suffix or ".png"
                z.write(path, f"{r.appno or r.id}_{idx}{ext}")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=images_{job_id}.zip"})


@app.get("/api/jobs/{job_id}/file")
def get_job_file(job_id: str, p: str) -> Response:
    # safe-serve: only allow paths within RUNS_DIR/<job_id>/
    target = Path(p).resolve()
    base = (RUNS_DIR / job_id).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(403)
    if not target.exists():
        raise HTTPException(404)
    return FileResponse(target)


@app.get("/api/img")
def get_image(u: str) -> Response:
    """Proxy + cache: fetch the URL and return its bytes, used by frontend to display logos."""
    path = fetch(u)
    return FileResponse(path)


@app.get("/api/dev/eval-runs")
def eval_runs() -> list[dict]:
    runs = store.list_eval_runs()
    return [{"id": r["id"], "prompt_version": r["prompt_version"], "model": r["model"],
             "metrics": json.loads(r["metrics_json"]), "created_at": r["created_at"]} for r in runs]


# --- Static SPA ------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{rest:path}")
    def spa(rest: str) -> Response:
        # only serve SPA for non-api paths; otherwise FastAPI route would have matched
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(404)
        return FileResponse(index)
