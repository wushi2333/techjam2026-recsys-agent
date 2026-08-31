"""python scripts/autodl_ready.py  —  fail closed if the 1K job cannot start."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cli import cmd_env_check  # noqa: E402
from argparse import Namespace  # noqa: E402


if __name__ == "__main__":
    cmd_env_check(Namespace(config=None, data_scale="1k", data_dir=None, kit_dir=None))
