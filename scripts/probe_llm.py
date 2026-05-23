"""Smallest possible end-to-end probe: read one fixture, ask the model for a bbox.

Run:  D:/Project/linebase/.venv/Scripts/python.exe scripts/probe_llm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# allow running before pip install -e .
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from linebase.config import Settings
from linebase.llm import match_logo_in_photo


def main() -> int:
    settings = Settings.from_env()
    print(f"[probe] base_url={settings.base_url}  model={settings.model}")

    sample_dir = REPO / ".trellis/tasks/05-23-logo-lineart-auto-match-crop-pipeline/fixtures/sample_6433801"
    logo = sample_dir / "00_rId50_image47.png"
    photo = sample_dir / "01_rId51_image48.png"
    expected = sample_dir / "02_rId52_image49.png"
    assert logo.exists() and photo.exists() and expected.exists()
    print(f"[probe] logo={logo.name}  photo={photo.name}")

    t0 = time.time()
    try:
        result = match_logo_in_photo(logo, photo, settings=settings)
    except Exception as e:
        print(f"[probe] ERROR: {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0

    print(f"[probe] response in {dt:.1f}s — prompt_version={result.prompt_version}")
    print(json.dumps({
        "found": result.found,
        "bbox": result.bbox,
        "confidence": result.confidence,
        "reason": result.reason,
        "usage": result.usage,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
