"""Quick diagnosis: does doubao-seed-2-0-pro-260215 stall on a single evidence
from row 77354840?

Runs ONE LLM call (logo, first evidence URL of row 77354840) with a hard 60 s
timeout passed via the OpenAI SDK. If doubao stalls we'll see APITimeoutError;
if it returns we'll print latency. Then does the same with gpt-5.5 for control.
"""
from __future__ import annotations

import time
from pathlib import Path

from linebase.config import Settings
from linebase.fetch import fetch
from linebase.llm import match_logo_in_photo


def run_one(model: str, logo_path: Path, ev_path: Path, timeout_s: float) -> None:
    print(f"\n--- {model} (timeout={timeout_s}s) ---", flush=True)
    t0 = time.perf_counter()
    try:
        r = match_logo_in_photo(
            logo_path, ev_path, model=model, timeout=timeout_s
        )
        dt = time.perf_counter() - t0
        print(
            f"OK in {dt:.1f}s  found={r.found} conf={r.confidence:.2f} "
            f"bbox={r.bbox} usage={r.usage} reason={r.reason[:80]!r}",
            flush=True,
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"FAIL in {dt:.1f}s  {type(e).__name__}: {e}", flush=True)


def main() -> int:
    settings = Settings.from_env()
    print(f"providers configured: {sorted(settings.providers)}", flush=True)

    # Pull logo + first evidence URL of the 9-evidence row 77354840.
    logo_url = "https://tsdr.uspto.gov/img/77354840/large"
    ev_url = (
        "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/"
        "SPE20200404142330/1/webcontent?scale=1"
    )

    print(f"fetching logo: {logo_url}", flush=True)
    logo_path = fetch(logo_url)
    print(f"  -> {logo_path} ({logo_path.stat().st_size} bytes)", flush=True)

    print(f"fetching evidence: {ev_url}", flush=True)
    ev_path = fetch(ev_url)
    print(f"  -> {ev_path} ({ev_path.stat().st_size} bytes)", flush=True)

    # Try doubao first with a 60 s timeout (the stall last night exceeded ~60 s).
    run_one("doubao-seed-2-0-pro-260215", logo_path, ev_path, timeout_s=60.0)

    # Same evidence with gpt-5.5 for control.
    run_one("gpt-5.5", logo_path, ev_path, timeout_s=120.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
