from __future__ import annotations

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

    @property
    def debug_depth(self) -> int:
        return 0

    @property
    def primary(self) -> float | None:
        if self.metrics is None:
            return None
        return self.metrics.primary


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
            metrics = Metrics(m.get("GAUC"), m.get("nDCG@5"), m.get("primary"))
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
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def drafts(self) -> list[Node]:
        return [self.nodes[i] for i in self.order if self.nodes[i].parent_id is None]

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
        return max(good, key=lambda n: n.primary or float("-inf"))

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
        parts = []
        for n in self.good():
            p = n.primary
            parts.append(f"{n.node_id} arm={n.arm} primary={p} :: {n.hypothesis}")
        return "\n".join(parts[-12:])

    def no_improve_streak(self, epsilon: float) -> int:
        best = None
        streak = 0
        for nid in self.order:
            n = self.nodes[nid]
            if n.is_buggy or n.primary is None:
                continue
            if best is None or n.primary > best + epsilon:
                best = n.primary
                streak = 0
            else:
                streak += 1
        return streak
