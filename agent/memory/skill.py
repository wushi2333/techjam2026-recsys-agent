from __future__ import annotations

from pathlib import Path

from agent.memory.journal import Journal


SEED = """# Experiment Skill

## Organizer priors
- Do not spend trials on extra static features or larger embedding k.
- User-side first-order terms cannot change within-user ranking.
- Prefer loss (BPR / listwise), then DIN-lite sequences (seq_len/seq_mode), then hour, then lr.

## Live
No successful edits yet.
"""


def render_skill(journal: Journal) -> str:
    lines = [SEED.rsplit("## Live", 1)[0].rstrip(), "", "## Live"]
    best = journal.best()
    if best is None:
        lines.append("No incumbent.")
        return "\n".join(lines) + "\n"
    lines.append(f"Incumbent {best.node_id} primary={best.primary} arm={best.arm}")
    lines.append("")
    lines.append("Recent good nodes:")
    for n in journal.good()[-8:]:
        lines.append(f"- {n.node_id} {n.arm} {n.primary}: {n.hypothesis}")
    buggy = [n for n in journal.nodes.values() if n.is_buggy][-5:]
    if buggy:
        lines.append("")
        lines.append("Avoid repeating:")
        for n in buggy:
            lines.append(f"- {n.error or 'bug'} ({n.arm})")
    return "\n".join(lines) + "\n"


def write_skill(path: Path, journal: Journal) -> None:
    path.write_text(render_skill(journal), encoding="utf-8")
