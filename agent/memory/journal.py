from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.types import Metrics, Stage


@dataclass
class Node:
    node_id: str
    parent_id: str | None
    stage: Stage
    arm: str
    hypothesis: str
    diff: str
    metrics: Metrics | None
    is_buggy: bool
    recovery: str | None = None
    error: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    gpu_seconds: float = 0.0
    children: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def debug_depth(self) -> int:
        return 0

    @property
    def primary(self) -> float | None:
        if self.metrics is None:
            return None
        return self.metrics.primary


def submit_score(node: Node | None) -> float | None:
    """Search/submit number: bag/blend primary, else confirmed mean, else node primary."""
    if node is None:
        return None
    extra = node.extra or {}
    if extra.get("submit_bag_primary") is not None:
        return float(extra["submit_bag_primary"])
    if node.stage == "ensemble" and node.primary is not None:
        return float(node.primary)
    if extra.get("members") and node.primary is not None:
        return float(node.primary)
    if extra.get("confirmed_mean") is not None:
        return float(extra["confirmed_mean"])
    if node.primary is not None:
        return float(node.primary)
    return None


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.nodes: dict[str, Node] = {}
        self.order: list[str] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self._ingest(json.loads(line))

    def _ingest(self, raw: dict[str, Any]) -> None:
        metrics = None
        if raw.get("metrics"):
            m = raw["metrics"]
            known = {"GAUC", "nDCG@5", "primary"}
            extra_m = {
                k: float(v)
                for k, v in m.items()
                if k not in known and isinstance(v, (int, float))
            }
            metrics = Metrics(m.get("GAUC"), m.get("nDCG@5"), m.get("primary"), extra_m)
        node = Node(
            node_id=raw["node_id"],
            parent_id=raw.get("parent_id"),
            stage=raw["stage"],
            arm=raw["arm"],
            hypothesis=raw["hypothesis"],
            diff=raw.get("diff", ""),
            metrics=metrics,
            is_buggy=raw["is_buggy"],
            recovery=raw.get("recovery"),
            error=raw.get("error"),
            tokens_in=raw.get("tokens_in", 0),
            tokens_out=raw.get("tokens_out", 0),
            gpu_seconds=raw.get("gpu_seconds", 0.0),
            extra=raw.get("extra") or {},
        )
        self.nodes[node.node_id] = node
        self.order.append(node.node_id)
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node.node_id)

    def append(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.order.append(node.node_id)
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node.node_id)
        payload = asdict(node)
        if node.metrics is not None:
            payload["metrics"] = node.metrics.as_dict()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def drafts(self) -> list[Node]:
        return [
            self.nodes[i]
            for i in self.order
            if self.nodes[i].parent_id is None and self.nodes[i].stage == "draft"
        ]

    def good(self) -> list[Node]:
        return [n for n in self.nodes.values() if not n.is_buggy and n.primary is not None]

    def buggy_leaves(self) -> list[Node]:
        out = []
        for n in self.nodes.values():
            if n.is_buggy and not n.children:
                out.append(n)
        return out

    def best(self) -> Node | None:
        good = self.good()
        if not good:
            return None
        confirmed = [n for n in good if n.extra.get("confirmed")]
        pool = confirmed or good

        def _key(n: Node) -> tuple:
            extra = n.extra or {}
            val = submit_score(n)
            val = float("-inf") if val is None else val
            is_submit = 1 if n.stage == "ensemble" or extra.get("members") else 0
            weak = 0 if extra.get("weak_incumbent") else 1
            return val, is_submit, weak

        return max(pool, key=_key)

    def incumbent_primary(self) -> float | None:
        return submit_score(self.best())

    def screen_target(self) -> float | None:
        """Screen bar is the submit/bag number, not the 3-seed member mean."""
        return self.incumbent_primary()

    def confirmed(self) -> list[Node]:
        return [n for n in self.good() if n.extra.get("confirmed")]

    def skip_streak(self) -> int:
        n = 0
        for nid in reversed(self.order):
            node = self.nodes[nid]
            if node.stage in {"eda", "ablate", "research", "read_paper"}:
                continue
            if node.diff == "skip" or (node.extra or {}).get("action") == "skip":
                n += 1
                continue
            break
        return n

    def ablate_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.stage == "ablate")

    def is_ablate_child(self, node: Node) -> bool:
        return node.stage == "improve" and node.arm == "ablate"

    def is_billed(self, node: Node) -> bool:
        """Train/search steps that consume the 50-iter cap.

        Ensemble bag/blend does not train and is not billed. Ablate children,
        EDA, and harness research/README ingest are free.
        """
        if node.stage in {"eda", "ensemble"} or self.is_ablate_child(node):
            return False
        if node.stage in {"research", "read_paper"} and (node.extra or {}).get("harness"):
            return False
        return True

    def billed_count(self) -> int:
        return sum(1 for nid in self.order if self.is_billed(self.nodes[nid]))

    def read_paper_arms(self) -> set[str]:
        from agent.memory.catalog import is_catalog_path

        out = set()
        for n in self.nodes.values():
            if n.stage != "read_paper":
                continue
            extra = n.extra or {}
            if extra.get("catalog") or is_catalog_path(extra.get("path")):
                continue
            out.add(n.arm)
        return out

    def pending_screen(self) -> Node | None:
        ablated_parents = {
            n.parent_id for n in self.nodes.values() if n.stage == "ablate"
        }
        for nid in reversed(self.order):
            node = self.nodes[nid]
            if node.extra.get("screen_pass") and nid not in ablated_parents:
                if not node.extra.get("confirmed"):
                    return node
        return None

    def debug_depth(self, node: Node) -> int:
        depth = 0
        cur = node
        while cur.parent_id and cur.stage == "debug":
            depth += 1
            parent = self.nodes.get(cur.parent_id)
            if parent is None:
                break
            cur = parent
        return depth

    def summary(self) -> str:
        from agent.memory.facts import loop_brief

        return loop_brief(self).strip()

    def knowledge_notes(self) -> str:
        harness: list[str] = []
        llm: list[str] = []
        for nid in self.order:
            n = self.nodes[nid]
            extra = n.extra or {}
            if n.stage == "research":
                excerpt = str(extra.get("excerpt") or "")[:1800]
                bit = f"{n.node_id} research: {n.hypothesis}\n{excerpt}"
                (harness if extra.get("harness") else llm).append(bit)
            elif n.stage == "read_paper":
                path = extra.get("path")
                bit = f"{n.node_id} read_paper path={path}"
                catalog = extra.get("catalog") or extra.get("harness")
                if catalog:
                    excerpt = str(extra.get("excerpt") or "")[:600]
                    if excerpt:
                        bit += "\n" + excerpt
                    harness.append(bit)
                else:
                    llm.append(bit)
        chunks = harness[-2:] + llm[-4:]
        return "\n---\n".join(chunks)

    def research_count(self) -> int:
        return sum(
            1
            for n in self.nodes.values()
            if n.stage == "research" and not (n.extra or {}).get("harness")
        )

    def diagnose_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.stage == "diagnose")

    def diagnose_queries(self) -> set[str]:
        out = set()
        for n in self.nodes.values():
            if n.stage != "diagnose":
                continue
            q = str((n.extra or {}).get("query") or "")
            if q:
                out.add(q)
        return out

    def read_paper_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.stage == "read_paper")

    def read_paper_paths(self) -> set[str]:
        out = set()
        for n in self.nodes.values():
            if n.stage != "read_paper":
                continue
            path = str((n.extra or {}).get("path") or "")
            if path:
                out.add(path.replace("\\", "/").lower())
        return out

    def read_paper_keys(self) -> set[str]:
        out = set()
        for n in self.nodes.values():
            if n.stage != "read_paper":
                continue
            path = str((n.extra or {}).get("path") or "").replace("\\", "/").lower()
            excerpt = str((n.extra or {}).get("excerpt") or "")[:200]
            digest = hashlib.sha1(excerpt.encode("utf-8", errors="ignore")).hexdigest()[:12]
            if path:
                out.add(f"{path}#{digest}")
        return out

    def billed_no_improve_streak(self, epsilon: float) -> int:
        """Official-style: billed scored/skip steps since last >ε incumbent move."""
        best = None
        streak = 0
        for nid in self.order:
            n = self.nodes[nid]
            if n.stage in {"eda", "research", "read_paper", "diagnose", "ablate"} or self.is_ablate_child(n):
                continue
            extra = n.extra or {}
            skipped = extra.get("action") == "skip" or n.diff == "skip"
            mean = extra.get("confirmed_mean")
            if mean is None:
                mean = n.primary
            if skipped or mean is None:
                if best is not None:
                    streak += 1
                continue
            mean = float(mean)
            if best is None:
                best = mean
                streak = 0
                continue
            if mean > best + epsilon:
                best = mean
                streak = 0
            else:
                streak += 1
        return streak

    def no_improve_streak(self, epsilon: float) -> int:
        best = None
        streak = 0
        for nid in self.order:
            n = self.nodes[nid]
            if not n.extra.get("confirmed"):
                continue
            if n.stage == "eda" or self.is_ablate_child(n):
                continue
            mean = (n.extra or {}).get("confirmed_mean")
            if mean is None:
                mean = n.primary
            if mean is None:
                continue
            mean = float(mean)
            if best is None:
                best = mean
                streak = 0
                continue
            if mean > best + epsilon:
                best = mean
                streak = 0
            else:
                streak += 1
        return streak
