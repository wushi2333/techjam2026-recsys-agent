from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import ROOT, load_settings
from agent.llm.client import build_llm
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


def cmd_ping(_args: argparse.Namespace) -> None:
    settings = load_settings()
    llm = build_llm(settings)
    print(f"provider={llm.provider} model={llm.model or '(none)'} base={llm.base_url or '(none)'}")
    if llm.provider == "dummy":
        print("no API key; copy .env.example to .env and set OPENAI_API_KEY or XAI_API_KEY")
        return
    from agent.llm.client import deepseek_extra
    from agent.llm.openai_compat import chat_completions

    text, tin, tout = chat_completions(
        base_url=llm.base_url,
        api_key=llm.api_key,
        model=llm.model,
        messages=[
            {"role": "system", "content": 'Reply with JSON {"ok": true}'},
            {"role": "user", "content": "ping"},
        ],
        temperature=0,
        timeout=180,
        extra_body=deepseek_extra(llm.base_url),
    )
    print(f"ok tokens_in={tin} tokens_out={tout}")
    print(text[:300])


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
    g = sub.add_parser("ping-llm")
    g.set_defaults(func=cmd_ping)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
