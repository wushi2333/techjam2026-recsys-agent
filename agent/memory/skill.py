from __future__ import annotations

from pathlib import Path

from agent.memory.journal import Journal


SEED = """# Experiment Skill

## Organizer priors
- Extra static ID fields and larger embedding k were already measured by the kit: no gain.
- User-side first-order terms cannot change within-user ranking.
- Measurements from this run are in run_facts (auto). Domain pack is a prior, not a to-do list.
- legal_skills is an index; load a body with read_paper path skills/<name>/SKILL.md.

## Live
No successful edits yet.
"""


def render_skill(journal: Journal) -> str:
    from agent.memory.facts import loop_brief

    head = SEED.rsplit("## Live", 1)[0].rstrip()
    return head + "\n\n## Live\n" + loop_brief(journal)


def write_skill(path: Path, journal: Journal) -> None:
    path.write_text(render_skill(journal), encoding="utf-8")
