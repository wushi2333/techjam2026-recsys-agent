from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent.config import ROOT, load_settings
from agent.llm.client import build_llm
from agent.orchestrator import Orchestrator


def _run_dir(args=None) -> Path:
    if args is not None and getattr(args, "run_dir", None):
        return Path(args.run_dir).expanduser().resolve()
    return (ROOT / "run").resolve()


def _apply_launch_env(args: argparse.Namespace) -> None:
    if getattr(args, "config", None):
        os.environ["KUAI_CONFIG"] = str(Path(args.config).expanduser().resolve())
    if getattr(args, "data_scale", None):
        os.environ["KUAI_DATA_SCALE"] = str(args.data_scale)
    if getattr(args, "data_dir", None):
        os.environ["KUAI_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())
    if getattr(args, "kit_dir", None):
        os.environ["KUAI_KIT_DIR"] = str(Path(args.kit_dir).expanduser().resolve())


def _settings(args: argparse.Namespace | None = None):
    if args is not None:
        _apply_launch_env(args)
    path = None
    if args is not None and getattr(args, "config", None):
        path = Path(args.config).expanduser().resolve()
    return load_settings(path)


def cmd_run(args: argparse.Namespace) -> None:
    if getattr(args, "llm", False):
        os.environ["LLM_PROVIDER"] = "openai"
    if getattr(args, "dummy", False):
        os.environ["LLM_PROVIDER"] = "dummy"
    if getattr(args, "research", False):
        os.environ["RESEARCH_ENABLED"] = "1"
    if getattr(args, "no_research", False):
        os.environ["RESEARCH_ENABLED"] = "0"
    settings = _settings(args)
    orch = Orchestrator(settings, _run_dir(args))
    best = orch.run(max_iters=args.max_iters, smoke=args.smoke)
    if best is None:
        print("no successful trial")
        return
    print(f"incumbent {best.node_id} primary={best.primary}")
    print("next: python -m agent finalize --run-dir <dir>   # test submission, no test labels")


def cmd_status(args: argparse.Namespace) -> None:
    path = _run_dir(args) / "status.json"
    if not path.exists():
        print("no run yet; python -m agent run --smoke")
        return
    print(path.read_text(encoding="utf-8"))


def cmd_init(args: argparse.Namespace) -> None:
    settings = _settings(args)
    Orchestrator(settings, _run_dir(args))
    print(f"run dir ready: {_run_dir(args)}")


def cmd_env_check(args: argparse.Namespace) -> None:
    from agent.env.autodl import check_ready, render_ready

    settings = _settings(args)
    rec = check_ready(settings)
    print(render_ready(rec), end="")
    if not rec["ok"]:
        raise SystemExit(1)


def cmd_finalize(args: argparse.Namespace) -> None:
    from agent.finalize import run as run_finalize

    settings = _settings(args)
    report = run_finalize(settings, _run_dir(args), smoke=bool(args.smoke))
    print(f"source {report['source']} valid_primary={report['valid_primary']}")
    print(report["check"])
    print(f"submission {report['submission']}")
    if report.get("log_random"):
        print(f"log_random {report['log_random']}")


def cmd_intervene(args: argparse.Namespace) -> None:
    from agent.observe.interventions import append

    path = _run_dir(args) / "interventions.jsonl"
    append(path, args.kind, note=args.note, phase="runtime")
    print(f"recorded runtime intervention {args.kind} -> {path}")


def cmd_interventions(args: argparse.Namespace) -> None:
    if not getattr(args, "add", False):
        print("use: python -m agent interventions --add --kind <kind> --note <text>")
        return
    if not args.kind:
        raise SystemExit("--kind is required with --add")
    cmd_intervene(args)


def cmd_ping(args: argparse.Namespace) -> None:
    settings = _settings(args)
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


def _job_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="toml settings path")
    parser.add_argument("--data-scale", type=str, default=None, choices=["pure", "1k", "27k"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--kit-dir", type=str, default=None)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--llm", action="store_true")
    r.add_argument("--dummy", action="store_true")
    r.add_argument("--research", action="store_true", help="enable capped arXiv research")
    r.add_argument("--no-research", action="store_true")
    r.add_argument("--run-dir", type=str, default=None)
    r.add_argument("--max-iters", type=int, default=None)
    _job_flags(r)
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("status")
    s.add_argument("--run-dir", type=str, default=None)
    s.set_defaults(func=cmd_status)
    i = sub.add_parser("init")
    _job_flags(i)
    i.add_argument("--run-dir", type=str, default=None)
    i.set_defaults(func=cmd_init)
    f = sub.add_parser("finalize")
    f.add_argument("--run-dir", type=str, default=None)
    f.add_argument("--smoke", action="store_true")
    _job_flags(f)
    f.set_defaults(func=cmd_finalize)
    g = sub.add_parser("ping-llm")
    _job_flags(g)
    g.set_defaults(func=cmd_ping)
    e = sub.add_parser("env-check")
    _job_flags(e)
    e.set_defaults(func=cmd_env_check)
    iv = sub.add_parser("intervene")
    iv.add_argument("--kind", type=str, required=True)
    iv.add_argument("--note", type=str, default="")
    iv.add_argument("--run-dir", type=str, default=None)
    iv.set_defaults(func=cmd_intervene)
    ivs = sub.add_parser("interventions")
    ivs.add_argument("--add", action="store_true")
    ivs.add_argument("--kind", type=str, default="manual")
    ivs.add_argument("--note", type=str, default="")
    ivs.add_argument("--run-dir", type=str, default=None)
    ivs.set_defaults(func=cmd_interventions)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
