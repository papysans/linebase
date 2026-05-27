from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from openpyxl import Workbook

from linebase import server, store


class _Req:
    upload_id = ""
    sheet_name = "Sheet1"
    appno_column = "B"
    logo_column = "D"
    evidence_column = "K"
    sample_kind = "first_n"
    sample_params = {"n": 10}
    threshold = 0.5
    model = None
    verify_loop = True
    tile_scan = False


def _setup_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "_singleton", None)
    store.init_schema(store.DB_PATH)
    monkeypatch.setattr(store, "_singleton", store._connect(store.DB_PATH))  # noqa: SLF001


def _write_workbook(path: Path, rows: list[list[object | None]]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    for row in rows:
        ws.append(row)
    wb.save(path)


def test_resolve_rows_accepts_headerless_single_data_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_db(tmp_path, monkeypatch)
    xlsx = tmp_path / "one-row.xlsx"
    row = [None] * 11
    row[1] = "74435340"
    row[3] = "https://tsdr.uspto.gov/img/74435340/large"
    row[10] = "https://example.test/e1,https://example.test/e2"
    _write_workbook(xlsx, [row])

    upload = store.insert_upload("one-row.xlsx", xlsx.stat().st_size, str(xlsx))
    req = _Req()

    rows = server._resolve_rows(upload, req)  # noqa: SLF001

    assert len(rows) == 1
    assert rows[0]["row_index"] == 1
    assert rows[0]["appno"] == "74435340"
    assert rows[0]["logo_url"] == "https://tsdr.uspto.gov/img/74435340/large"
    assert rows[0]["evidence_urls"] == [
        "https://example.test/e1",
        "https://example.test/e2",
    ]


def test_create_job_rejects_zero_usable_rows_without_inserting_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_db(tmp_path, monkeypatch)
    xlsx = tmp_path / "empty.xlsx"
    _write_workbook(xlsx, [["not", "a", "usable", "row"]])

    upload = store.insert_upload("empty.xlsx", xlsx.stat().st_size, str(xlsx))
    req = _Req()
    req.upload_id = upload.id

    with pytest.raises(HTTPException) as exc:
        server.create_job(req)  # type: ignore[arg-type]

    assert exc.value.status_code == 400
    assert store.list_jobs(limit=10) == []
