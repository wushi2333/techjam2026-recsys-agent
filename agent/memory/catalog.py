"""Progressive skill catalog. Index is always cheap; bodies load via read_paper."""

from __future__ import annotations

from pathlib import Path

PACK = Path(__file__).resolve().parents[2] / "benchmarks" / "kuairand"
SKILLS_DIR = PACK / "skills"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    raw = text or ""
    if not raw.startswith("---"):
        return {}, raw
    rest = raw[3:]
    end = rest.find("\n---")
    if end < 0:
        return {}, raw
    meta: dict[str, str] = {}
    for line in rest[:end].splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta, rest[end + 4 :].strip()


def list_skills(root: Path | None = None) -> list[dict]:
    base = Path(root) if root is not None else SKILLS_DIR
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.glob("*/SKILL.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        name = str(meta.get("name") or path.parent.name)
        out.append(
            {
                "name": name,
                "description": str(meta.get("description") or "")[:240],
                "arm": str(meta.get("arm") or "any"),
                "keys": str(meta.get("keys") or ""),
                "status": str(meta.get("status") or "wired"),
                "path": path,
                "rel": f"skills/{name}/SKILL.md",
                "body": body,
            }
        )
    return out


def index_block(root: Path | None = None) -> list[str]:
    skills = list_skills(root)
    if not skills:
        return ["legal_skills=(none)"]
    lines = [
        "legal_skills (index only; load a body with read_paper path=skills/<name>/SKILL.md):",
    ]
    for rec in skills:
        lines.append(
            f"- {rec['name']} arm={rec['arm']} status={rec['status']} "
            f"keys={rec['keys'] or '(none)'} | {rec['description']}"
        )
    lines.append(
        "Do not dump RecBole/Qlib/tsfresh into context. Blend weights are a valid-only grid, not ARIMA."
    )
    return lines


def is_catalog_path(path) -> bool:
    s = str(path or "").replace("\\", "/").lower()
    return (
        "/skills/" in s
        or "/github/" in s
        or s.endswith("/knowledge.md")
        or s.endswith("findings.md")
        or s.endswith("skill.md")
        or s.endswith("readme.md")
    )


def paper_file_key(path: str) -> str:
    parts = str(path or "").replace("\\", "/").rstrip("/").split("/")
    parts = [p for p in parts if p]
    if len(parts) >= 2 and parts[-1].lower() == "skill.md":
        return "/".join(parts[-2:])
    return parts[-1] if parts else ""


def distill_cards(journal) -> list[str]:
    """Claims-with-scope cards from this run. Agent-written; not a human agenda."""
    from agent.eval.dedup import fingerprint
    from agent.memory.facts import _fmt

    cards = []
    seen: set[str] = set()
    for nid in journal.order:
        n = journal.nodes[nid]
        extra = n.extra or {}
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        patch = extra.get("config_patch") or {}
        if not patch:
            continue
        fp = fingerprint(patch)
        parent = str(n.parent_id or "(root)")
        key = f"{parent}:{fp}"
        if key in seen:
            continue
        seen.add(key)
        hi, dp = extra.get("ci95_hi"), extra.get("delta_primary")
        if hi is not None and dp is not None and float(hi) < 0:
            cards.append(
                f"- claim={patch} status=falsified-1seed scope=parent={parent} "
                f"evidence={n.node_id} dP={_fmt(float(dp))}  "
                f"# not a family ban; a new incumbent identity may retry"
            )
        elif extra.get("confirmed") and extra.get("confirmed_mean") is not None:
            cards.append(
                f"- claim={patch} status=confirmed-3seed scope=parent={parent} "
                f"evidence={n.node_id} mean={_fmt(extra.get('confirmed_mean'))}"
            )
    if extra := _blend_cards(journal):
        cards.extend(extra)
    if not cards:
        return ["skill_cards=(none yet; fills as screens land)"]
    return ["skill_cards (harness-distilled, claims-with-scope):"] + cards[-12:]


def _blend_cards(journal) -> list[str]:
    from agent.memory.facts import _fmt

    out = []
    for n in journal.nodes.values():
        extra = n.extra or {}
        if n.stage != "ensemble" or extra.get("ensemble_kind") != "complementary":
            continue
        if extra.get("blend_alpha") is None:
            continue
        out.append(
            f"- claim=complementary-blend status=measured-1seed scope=run "
            f"evidence={n.node_id} alpha={_fmt(extra.get('blend_alpha'))} "
            f"gamma={_fmt(extra.get('blend_gamma'))} top1={_fmt(extra.get('blend_top1'))}"
        )
    return out


def distill_markdown(journal) -> str:
    lines = [
        "# Distilled skill cards (this run)",
        "",
        "Harness-written claims-with-scope. Not a to-do list.",
        "",
    ]
    lines.extend(distill_cards(journal))
    return "\n".join(lines).rstrip() + "\n"
