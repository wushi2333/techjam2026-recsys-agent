"""Durable tagged findings. Journal is per-run; this file survives the next run."""

from __future__ import annotations

import json
from pathlib import Path

from agent.eval.dedup import fingerprint
from agent.eval.incumbent import incumbent_identity
from agent.observe.events import dumps

PACK = Path(__file__).resolve().parents[2] / "benchmarks" / "kuairand"
JSONL = PACK / "findings.jsonl"
MARKDOWN = PACK / "findings.md"
ARM_STATE = PACK / "arm_state.json"


def arm_state_pack_path(scale: str | None) -> Path:
    s = str(scale or "pure").strip().lower() or "pure"
    if s not in {"pure", "1k", "27k"}:
        s = "pure"
    return PACK / f"arm_state_{s}.json"

_GRAVES: dict[str, set[str]] | None = None


def _fmt(v) -> str:
    if v is None:
        return "na"
    if isinstance(v, float):
        return f"{v:.5f}"
    return str(v)


def records_from_journal(journal, run_id: str = "") -> list[dict]:
    recs: list[dict] = []
    ident = incumbent_identity(journal)
    if ident.get("node_id") and ident.get("submit_primary") is not None:
        tag = "measured-3seed" if ident.get("is_bag") or ident.get("confirmed_mean") is not None else "measured-1seed"
        recs.append(
            {
                "key": f"incumbent:{ident['node_id']}:{_fmt(ident.get('submit_primary'))}",
                "tag": tag,
                "run": run_id,
                "text": (
                    f"incumbent {ident['node_id']} is_bag={ident['is_bag']} "
                    f"submit={_fmt(ident.get('submit_primary'))} "
                    f"seed0={_fmt(ident.get('seed0_primary'))} "
                    f"member_mean={_fmt(ident.get('member_mean'))} "
                    f"screen_bar={_fmt(ident.get('screen_bar'))}"
                ),
            }
        )
    for n in journal.confirmed():
        extra = n.extra or {}
        patch = extra.get("config_patch") or extra.get("full_config") or {}
        recs.append(
            {
                "key": "confirmed:" + fingerprint(patch or {"node": n.node_id}),
                "tag": "measured-3seed" if extra.get("confirmed_mean") is not None or extra.get("members") else "measured-1seed",
                "run": run_id,
                "text": (
                    f"{n.node_id} seed0={_fmt(n.primary)} mean={_fmt(extra.get('confirmed_mean'))} "
                    f"patch={patch or extra.get('members')}"
                ),
            }
        )
    seen: set[str] = set()
    for nid in journal.order:
        n = journal.nodes[nid]
        extra = n.extra or {}
        if n.arm == "ablate":
            continue
        if n.stage not in {"improve", "draft"}:
            continue
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        if n.stage == "draft" and extra.get("confirmed"):
            continue
        hi, dp = extra.get("ci95_hi"), extra.get("delta_primary")
        patch = extra.get("config_patch") or extra.get("full_config") or {}
        if hi is None or dp is None or not patch:
            continue
        if float(hi) >= 0:
            continue
        from agent.eval.dedup import canonical_patch

        canon = canonical_patch(patch) or dict(patch)
        fp = fingerprint(canon)
        if fp in seen:
            continue
        seen.add(fp)
        kind = "draft 1-seed" if n.stage == "draft" else "1-seed on this parent, not 3-seed"
        scale = str((extra.get("full_config") or {}).get("data_scale") or canon.get("data_scale") or "pure")
        recs.append(
            {
                "key": f"falsified:{scale}:{n.parent_id or 'root'}:{fp}",
                "tag": "measured-1seed",
                "run": run_id,
                "fingerprint": fp,
                "patch": canon,
                "scale": scale,
                "text": (
                    f"{n.node_id} parent={n.parent_id or '(root)'} scale={scale} {canon} "
                    f"dP={_fmt(float(dp))} CI_hi={_fmt(float(hi))}  "
                    f"# {kind}"
                ),
            }
        )
    from agent.memory.facts import calibration_block

    cal = calibration_block(journal)
    if cal and "none" not in cal[0]:
        recs.append(
            {
                "key": "calibration:" + cal[0],
                "tag": "diagnosis",
                "run": run_id,
                "text": cal[0],
            }
        )
    return recs


def render_markdown(recs: list[dict]) -> str:
    lines = [
        "# Auto findings (not a to-do list)",
        "",
        "Tagged measurements written by the harness from journals.",
        "[measured-3seed] is fact; [measured-1seed] needs ablate; [diagnosis] is a direction.",
        "Do not treat this file as a human trial agenda.",
        "",
    ]
    for rec in recs:
        lines.append(f"- [{rec.get('tag')}] {rec.get('text')}")
    return "\n".join(lines).rstrip() + "\n"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def merge_records(old: list[dict], new: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for rec in old + new:
        key = str(rec.get("key") or rec.get("text") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def write_run_findings(path: Path, journal, run_id: str = "") -> str:
    recs = records_from_journal(journal, run_id)
    text = render_markdown(recs)
    Path(path).write_text(text, encoding="utf-8")
    return text


def write_pack_findings(journal, run_id: str = "") -> str:
    PACK.mkdir(parents=True, exist_ok=True)
    merged = merge_records(load_jsonl(JSONL), records_from_journal(journal, run_id))
    JSONL.write_text("".join(dumps(r) + "\n" for r in merged), encoding="utf-8")
    text = render_markdown(merged)
    MARKDOWN.write_text(text, encoding="utf-8")
    clear_graveyard_cache()
    return text


def load_findings() -> str:
    if MARKDOWN.exists():
        return MARKDOWN.read_text(encoding="utf-8").strip()
    return ""


def _fps_for_patch(patch: dict | None) -> set[str]:
    from agent.eval.dedup import canonical_patch

    body = {k: v for k, v in (patch or {}).items() if k != "seed"}
    if not body:
        return set()
    out = {fingerprint(body)}
    canon = canonical_patch(body)
    if canon:
        out.add(fingerprint(canon))
    return out


def clear_graveyard_cache() -> None:
    global _GRAVES
    _GRAVES = None


def _parse_falsified_key(key: str) -> tuple[str, str]:
    """Return (scale, fingerprint). Empty scale = legacy unscoped (treated as pure)."""
    raw = str(key or "")
    if not raw.startswith("falsified:"):
        return "", ""
    rest = raw[len("falsified:") :]
    scale = ""
    for tag in ("pure:", "1k:", "27k:"):
        if rest.startswith(tag):
            scale = tag[:-1]
            rest = rest[len(tag) :]
            break
    if rest.startswith("["):
        return scale, rest
    idx = rest.find(":[")
    if idx >= 0:
        return scale, rest[idx + 1 :]
    return scale, ""


def _patch_from_text(text: str) -> dict | None:
    import ast
    import re

    m = re.search(r"\{[^{}]+\}", text or "")
    if not m:
        return None
    try:
        val = ast.literal_eval(m.group(0))
    except (SyntaxError, ValueError):
        return None
    return val if isinstance(val, dict) else None


def _graves_table(*, reload: bool = False) -> dict[str, set[str]]:
    global _GRAVES
    if _GRAVES is not None and not reload:
        return _GRAVES
    table: dict[str, set[str]] = {}
    for rec in load_jsonl(JSONL):
        if rec.get("tag") != "measured-1seed":
            continue
        scale, fp = _parse_falsified_key(str(rec.get("key") or ""))
        rec_scale = str(rec.get("scale") or scale or "").strip().lower()
        if rec_scale not in {"pure", "1k", "27k"}:
            rec_scale = "pure"
        bucket = table.setdefault(rec_scale, set())
        if fp:
            bucket.add(fp)
        stored = rec.get("fingerprint")
        if stored:
            bucket.add(str(stored))
        patch = rec.get("patch") if isinstance(rec.get("patch"), dict) else None
        if patch is None:
            patch = _patch_from_text(str(rec.get("text") or ""))
        bucket.update(_fps_for_patch(patch))
    _GRAVES = table
    return table


def graveyard_fingerprints(*, reload: bool = False, scale: str | None = None) -> set[str]:
    """Cross-run CI_hi<0 fingerprints for this data_scale (legacy keys count as pure)."""
    table = _graves_table(reload=reload)
    want = str(scale or "pure").strip().lower() or "pure"
    if want not in {"pure", "1k", "27k"}:
        want = "pure"
    return set(table.get(want) or ())


def is_graveyard_patch(patch: dict | None, scale: str | None = None) -> bool:
    if not patch:
        return False
    return bool(_fps_for_patch(patch) & graveyard_fingerprints(scale=scale))
