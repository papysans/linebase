"""Walk all 9 evidences of row 77354840 with doubao and a 90 s per-call timeout.

If one evidence stalls, the timeout will surface it. We log latency per call so
we can pin the culprit. We stop on the first APITimeoutError to keep the
diagnostic budget bounded.
"""
from __future__ import annotations

import time
from pathlib import Path

from linebase.config import Settings
from linebase.fetch import fetch
from linebase.llm import match_logo_in_photo

EVIDENCES = [
    "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/SPE20200404142330/1/webcontent?scale=1",
    "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/SPE20200404142330/2/webcontent?scale=1",
    "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/SPE20200404142330/3/webcontent?scale=1",
    "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/SPE20200404142330/4/webcontent?scale=1",
    "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/SPE20200404142330/5/webcontent?scale=1",
    "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/SPE20160309160358/1/webcontent?scale=1",
    "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/SPE20100116172624/1/webcontent?scale=1",
    "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/SPE20090918135230/1/webcontent?scale=1",
    "https://tsdrsec.uspto.gov/ts/cd/casedoc/sn77354840/SPE20090918135230/2/webcontent?scale=1",
]


def main() -> int:
    Settings.from_env()
    logo_path = fetch("https://tsdr.uspto.gov/img/77354840/large")
    print(f"logo {logo_path} ({logo_path.stat().st_size} bytes)", flush=True)

    fails = 0
    total = 0.0
    for i, url in enumerate(EVIDENCES, 1):
        ev = fetch(url)
        size = ev.stat().st_size
        print(f"\n[{i}/9] {url}\n      bytes={size}", flush=True)
        t0 = time.perf_counter()
        try:
            r = match_logo_in_photo(
                logo_path, ev, model="doubao-seed-2-0-pro-260215", timeout=90.0
            )
            dt = time.perf_counter() - t0
            total += dt
            print(
                f"      OK {dt:.1f}s  found={r.found} conf={r.confidence:.2f} "
                f"prompt={r.usage.get('prompt_tokens') if r.usage else '?'} "
                f"completion={r.usage.get('completion_tokens') if r.usage else '?'}",
                flush=True,
            )
        except Exception as e:
            dt = time.perf_counter() - t0
            total += dt
            fails += 1
            print(f"      FAIL {dt:.1f}s  {type(e).__name__}: {str(e)[:200]}", flush=True)
            if fails >= 2:
                print("\n[stop] 2 fails — aborting walk", flush=True)
                break

    print(f"\nsummary: walked {i}/9  fails={fails}  cumulative={total:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
