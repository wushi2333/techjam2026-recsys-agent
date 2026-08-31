"""Hidden-test long_view access. Search / diagnose / EDA have no token."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

TEST_LO, TEST_HI = 20220429, 20220508
ENV_KEY = "KUAI_TEST_ACCESS"
# Keep in sync with templates/dataset.py. Not 0: 0 is an observed negative.
LABEL_MISSING = -1


class TestLabelError(PermissionError):
    """Raised when hidden-test long_view is read without a finalize token."""

    __test__ = False


@dataclass(frozen=True)
class TestAccessToken:
    id: str
    reason: str
    issued_at: str
    experiment_id: str

    def bind_env(self) -> None:
        os.environ[ENV_KEY] = self.id

    def write(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "test_access.json").write_text(
            json.dumps(asdict(self)), encoding="utf-8"
        )


def is_test_date(date: int) -> bool:
    return TEST_LO <= int(date) <= TEST_HI


def current_token_id() -> str:
    return str(os.environ.get(ENV_KEY) or "").strip()


def require_token(token: TestAccessToken | None = None) -> str:
    tid = token.id if token is not None else current_token_id()
    if not tid:
        raise TestLabelError(
            "test long_view requires a finalize token; "
            "search/diagnose/EDA must not read hidden-test labels"
        )
    return tid


def issue(*, reason: str, log_path: Path, experiment_id: str = "") -> TestAccessToken:
    token = TestAccessToken(
        id=secrets.token_hex(8),
        reason=str(reason),
        issued_at=datetime.now(timezone.utc).isoformat(),
        experiment_id=str(experiment_id or ""),
    )
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(token)) + "\n")
    token.write(path.parent)
    return token


def long_view(raw, date: int, token: TestAccessToken | None = None) -> int:
    """Parse a label. Test dates require a token and return LABEL_MISSING."""
    if is_test_date(date):
        require_token(token)
        return LABEL_MISSING
    s = str(raw if raw is not None else "").strip()
    return 0 if s in {"", "0", "0.0"} else 1


def load_token_file(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str((raw or {}).get("id") or "").strip()
