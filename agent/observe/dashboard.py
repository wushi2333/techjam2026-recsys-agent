from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>recsys-agent</title>
<style>
body { font: 15px/1.45 ui-sans-serif, system-ui; background:#0f1218; color:#e8edf5; margin:24px; }
pre { background:#1a2030; padding:12px; overflow:auto; }
.ok { color:#6ee7b7; } .bad { color:#fca5a5; }
</style>
<h1>recsys-agent</h1>
<p>Watch-only. Reading this is not an intervention.</p>
<pre id="s">%STATUS%</pre>
<h2>journal tail</h2>
<pre>%JOURNAL%</pre>
"""


def render(status_path: Path, journal_path: Path, dest: Path) -> None:
    status = "{}"
    if status_path.exists():
        status = json.dumps(json.loads(status_path.read_text(encoding="utf-8")), indent=2)
    tail = ""
    if journal_path.exists():
        lines = journal_path.read_text(encoding="utf-8").splitlines()[-8:]
        tail = "\n".join(lines)
    dest.write_text(
        TEMPLATE.replace("%STATUS%", status).replace("%JOURNAL%", tail),
        encoding="utf-8",
    )
