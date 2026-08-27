from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import ROOT, load_settings
from agent.orchestrator import Orchestrator


def _run_dir() -> Path:
    return ROOT / "run"


def cmd_run(args: argparse.Namespace) -> None:
    settings = load_settings()
    orch = Orchestrator(settings, _run_dir())
    best = orch.run(max_iters=args.max_iters, smoke=args.smoke)
    if best is None:
        print("no successful trial")
        return
    print(f"incumbent {best.node_id} primary={best.primary}")


def cmd_status(_args: argparse.Namespace) -> None:
    path = _run_dir() / "status.json"
    if not path.exists():
        print("no run yet; python -m agent run --smoke")
        return
    print(path.read_text(encoding="utf-8"))


def cmd_init(_args: argparse.Namespace) -> None:
    settings = load_settings()
    Orchestrator(settings, _run_dir())
    print(f"run dir ready: {_run_dir()}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--max-iters", type=int, default=None)
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)
    i = sub.add_parser("init")
    i.set_defaults(func=cmd_init)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
