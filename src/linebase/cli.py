"""CLI entry: `linebase serve` / `linebase eval` / `linebase tune`."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("linebase.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "baseline_eval.py"
    return subprocess.call([sys.executable, str(script)])


def main() -> int:
    parser = argparse.ArgumentParser(prog="linebase")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="run FastAPI server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_eval = sub.add_parser("eval", help="run the baseline eval against docx fixtures")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
