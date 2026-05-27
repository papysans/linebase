"""SQLite store. Single file at .data/linebase.db. No ORM — plain sqlite3 + dataclasses."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator


def _build(cls, row_dict: dict[str, Any]):
    """Build a dataclass instance from a sqlite Row dict, ignoring unknown keys.

    Two scenarios this protects against:
      1. Forward-compat: the DB has columns added by a newer code path; an
         older interpreter in the same process should still be able to load
         rows without TypeError on unexpected kwargs.
      2. Backward-compat: a row predates a column the dataclass added with a
         default — without this helper we'd have to remember to backfill or
         re-create rows on every migration.
    Loses no information vs `cls(**row_dict)` when the schema and dataclass
    are in sync, which is the steady-state.
    """
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in row_dict.items() if k in valid})

DATA_DIR = Path(__file__).resolve().parents[2] / ".data"
DB_PATH = DATA_DIR / "linebase.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS upload (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    size INTEGER NOT NULL,
    path TEXT NOT NULL,
    sheets_json TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS job (
    id TEXT PRIMARY KEY,
    upload_id TEXT NOT NULL REFERENCES upload(id),
    sheet_name TEXT NOT NULL,
    logo_column TEXT NOT NULL,
    evidence_column TEXT NOT NULL,
    appno_column TEXT NOT NULL,
    threshold REAL NOT NULL DEFAULT 0.5,
    sample_kind TEXT NOT NULL,
    sample_params_json TEXT NOT NULL,
    prompt_version TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    total_rows INTEGER NOT NULL DEFAULT 0,
    done_rows INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);

CREATE TABLE IF NOT EXISTS job_row (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES job(id),
    row_index INTEGER NOT NULL,
    appno TEXT,
    logo_url TEXT,
    evidence_urls_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    best_crop_path TEXT,
    all_crops_json TEXT NOT NULL DEFAULT '{}',
    match_meta_json TEXT NOT NULL DEFAULT '{}',
    human_status TEXT,
    notes TEXT,
    updated_at REAL NOT NULL,
    UNIQUE(job_id, row_index)
);

CREATE TABLE IF NOT EXISTS eval_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_row_job ON job_row(job_id);
CREATE INDEX IF NOT EXISTS idx_job_row_status ON job_row(job_id, status);
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or DB_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def init_schema(db_path: Path | None = None) -> None:
    db_path = db_path or DB_PATH
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        # Idempotent migrations for columns added after the schema's first
        # release. Wrapped in try/except so a re-run on an already-migrated DB
        # is a no-op (sqlite3 raises OperationalError on "duplicate column").
        try:
            conn.execute("ALTER TABLE job ADD COLUMN model TEXT")
        except sqlite3.OperationalError:
            pass
        # Added 2026-05-24: opt-in "二次校验" flag. When 1, the pipeline runs
        # `match_with_verify` (extra LLM call per evidence to confirm the crop)
        # instead of the single-shot `match_logo_in_photo`. Persisted on the
        # job so the per-row /rerun endpoint can override it transiently.
        try:
            conn.execute("ALTER TABLE job ADD COLUMN verify_loop INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        # Added 2026-05-24 (iter 6.3): opt-in 3x3 tile-scan fallback for small
        # logos buried in busy photos. When 1, `_one_evidence` re-tries failed
        # / unverified matches by cropping the evidence into 9 tiles and
        # matching each tile, then translating the best tile's bbox back to
        # the original photo's coords. Costs 9 extra LLM calls per evidence —
        # always opt-in, never default.
        try:
            conn.execute("ALTER TABLE job ADD COLUMN tile_scan INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass


_singleton: sqlite3.Connection | None = None


def db() -> sqlite3.Connection:
    global _singleton
    if _singleton is None:
        init_schema()
        _singleton = _connect()
    return _singleton


def new_id() -> str:
    return uuid.uuid4().hex[:16]


# --- Upload -----------------------------------------------------------------

@dataclass
class Upload:
    id: str
    filename: str
    size: int
    path: str
    sheets_json: str | None
    created_at: float


def insert_upload(filename: str, size: int, path: str) -> Upload:
    uid = new_id()
    now = time.time()
    db().execute(
        "INSERT INTO upload(id, filename, size, path, created_at) VALUES (?,?,?,?,?)",
        (uid, filename, size, path, now),
    )
    return Upload(uid, filename, size, path, None, now)


def get_upload(upload_id: str) -> Upload | None:
    row = db().execute("SELECT * FROM upload WHERE id=?", (upload_id,)).fetchone()
    return _build(Upload, dict(row)) if row else None


def set_upload_sheets(upload_id: str, sheets_json: str) -> None:
    db().execute("UPDATE upload SET sheets_json=? WHERE id=?", (sheets_json, upload_id))


# --- Job --------------------------------------------------------------------

@dataclass
class Job:
    id: str
    upload_id: str
    sheet_name: str
    logo_column: str
    evidence_column: str
    appno_column: str
    threshold: float
    sample_kind: str
    sample_params_json: str
    prompt_version: str | None
    status: str
    total_rows: int
    done_rows: int
    cost_usd: float
    created_at: float
    started_at: float | None
    finished_at: float | None
    model: str | None = None  # added 2026-05-24: per-job LLM override; NULL → use settings.model
    verify_loop: int = 0  # added 2026-05-24: 1 → run the verify-loop on every evidence; 0 → single-shot match
    tile_scan: int = 0  # added 2026-05-24 (iter 6.3): 1 → fall back to 3x3 tile scan when primary fails on large photos


def insert_job(
    upload_id: str,
    sheet_name: str,
    logo_column: str,
    evidence_column: str,
    appno_column: str,
    threshold: float,
    sample_kind: str,
    sample_params: dict[str, Any],
    model: str | None = None,
    verify_loop: int = 0,
    tile_scan: int = 0,
) -> Job:
    jid = new_id()
    now = time.time()
    db().execute(
        """INSERT INTO job(id, upload_id, sheet_name, logo_column, evidence_column, appno_column,
           threshold, sample_kind, sample_params_json, status, created_at, model, verify_loop, tile_scan)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (jid, upload_id, sheet_name, logo_column, evidence_column, appno_column,
         threshold, sample_kind, json.dumps(sample_params), "pending", now, model,
         int(bool(verify_loop)), int(bool(tile_scan))),
    )
    return get_job(jid)  # type: ignore[return-value]


def get_job(job_id: str) -> Job | None:
    row = db().execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
    return _build(Job, dict(row)) if row else None


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    db().execute(f"UPDATE job SET {cols} WHERE id=?", (*fields.values(), job_id))


def list_jobs(limit: int = 50) -> list[Job]:
    """Most-recent-first listing of jobs. Backs the frontend empty-state
    "open a recent task" picker on /run, /review, /download.
    """
    rows = db().execute(
        "SELECT * FROM job ORDER BY created_at DESC LIMIT ?",
        (max(1, int(limit)),),
    ).fetchall()
    return [_build(Job, dict(r)) for r in rows]


# --- JobRow -----------------------------------------------------------------

@dataclass
class JobRow:
    id: int
    job_id: str
    row_index: int
    appno: str | None
    logo_url: str | None
    evidence_urls_json: str
    status: str
    best_crop_path: str | None
    all_crops_json: str
    match_meta_json: str
    human_status: str | None
    notes: str | None
    updated_at: float


def insert_job_row(job_id: str, row_index: int, appno: str | None, logo_url: str | None, evidence_urls: list[str]) -> JobRow:
    now = time.time()
    cur = db().execute(
        """INSERT INTO job_row(job_id, row_index, appno, logo_url, evidence_urls_json, updated_at)
           VALUES (?,?,?,?,?,?)""",
        (job_id, row_index, appno, logo_url, json.dumps(evidence_urls), now),
    )
    rid = cur.lastrowid
    return get_job_row(rid)  # type: ignore[return-value]


def get_job_row(row_id: int) -> JobRow | None:
    row = db().execute("SELECT * FROM job_row WHERE id=?", (row_id,)).fetchone()
    return _build(JobRow, dict(row)) if row else None


def list_job_rows(job_id: str, status: str | None = None) -> list[JobRow]:
    if status:
        rows = db().execute("SELECT * FROM job_row WHERE job_id=? AND status=? ORDER BY row_index", (job_id, status)).fetchall()
    else:
        rows = db().execute("SELECT * FROM job_row WHERE job_id=? ORDER BY row_index", (job_id,)).fetchall()
    return [_build(JobRow, dict(r)) for r in rows]


def update_job_row(row_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields = {**fields, "updated_at": time.time()}
    cols = ", ".join(f"{k}=?" for k in fields)
    db().execute(f"UPDATE job_row SET {cols} WHERE id=?", (*fields.values(), row_id))


# --- EvalRun ----------------------------------------------------------------

def insert_eval_run(prompt_version: str, model: str, metrics: dict[str, Any]) -> int:
    cur = db().execute(
        "INSERT INTO eval_run(prompt_version, model, metrics_json, created_at) VALUES (?,?,?,?)",
        (prompt_version, model, json.dumps(metrics), time.time()),
    )
    return cur.lastrowid  # type: ignore[return-value]


def list_eval_runs(limit: int = 50) -> list[dict[str, Any]]:
    rows = db().execute("SELECT * FROM eval_run ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
