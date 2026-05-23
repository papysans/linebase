"""V2 scalar probe — verify the model emits clarity/completeness/isolation as numbers.

Run with:  $env:LINEBASE_PROMPT_VERSION="2"; D:/Project/linebase/.venv/Scripts/python.exe scripts/probe_v2_scalars.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from linebase.config import Settings
from linebase.llm import match_logo_in_photo


def main() -> int:
    os.environ.setdefault("LINEBASE_PROMPT_VERSION", "2")
    settings = Settings.from_env()
    print(f"[probe] base_url={settings.base_url}  model={settings.model}  prompt=v{os.environ['LINEBASE_PROMPT_VERSION']}")

    sample_dir = REPO / ".trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline/fixtures/sample_6433801"
    logo = sample_dir / "00_rId50_image47.png"
    photo = sample_dir / "01_rId51_image48.png"
    assert logo.exists() and photo.exists()
    print(f"[probe] logo={logo.name}  photo={photo.name}")

    t0 = time.time()
    try:
        result = match_logo_in_photo(logo, photo, settings=settings)
    except Exception as e:
        print(f"[probe] ERROR: {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0

    print(f"[probe] response in {dt:.1f}s — prompt_version={result.prompt_version}")
    print("[probe] RAW response from model:")
    print(result.raw_response)
    print()
    print("[probe] PARSED result fields:")
    print(json.dumps({
        "found": result.found,
        "bbox": result.bbox,
        "confidence": result.confidence,
        "clarity": result.clarity,
        "completeness": result.completeness,
        "isolation": result.isolation,
        "reason": result.reason,
        "usage": result.usage,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
