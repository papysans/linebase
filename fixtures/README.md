# fixtures/

Test fixtures for the logo-lineart match-and-crop pipeline. Everything here is
committable — small, deterministic, network-free where possible.

## What lives here

### `demo_workbook.xlsx` (~10 KB)

A slimmed canonical XLSX extracted from the 428 MB source
`美专实物图排查-2026.2.6.xlsx`. Contains **only** the `图形商标tro` sheet with
**1 header row + 10 valid data rows** (11 rows total, 11 columns A–K).

Each of the 10 data rows is guaranteed to have:

- column B (`申请号`) non-empty,
- column D (`图形商标logo`) starts with `http`,
- column K (`使用证据`) contains ≥ 1 comma-separated http URL.

This file is the **format contract** for upload / parse / job-create. Use it
in e2e tests, fixtures, and Playwright runs instead of touching the 428 MB
source.

Note: the PRD originally assumed the source had 2 header rows; inspection
showed it has only 1. `demo_workbook.xlsx` reflects the actual format, which
also matches what `linebase.io_excel.iter_rows` expects (`start_row=2`).

Regenerate with:

```powershell
.venv/Scripts/python.exe -u scripts/extract_demo_workbook.py
```

### `sample_<appno>/` (10 dirs)

Ground-truth samples extracted from `商标去噪音图期望检测效果.docx`. Each
directory contains the LOGO line-art image, several evidence photos (one
annotated with a red bounding box marking the human-judged best match), and
an "expected crop" image. `_manifest.json` records the rId order and per-sample
metadata (appno, class, image_count).

NOTE: the docx-extracted `sample_*` directories currently live under
`.trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline/fixtures/`, not
in this directory. The eval harness reads them from there. Symlinking or
moving them is out of scope for this slice.
