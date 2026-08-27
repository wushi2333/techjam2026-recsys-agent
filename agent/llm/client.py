from __future__ import annotations

from dataclasses import dataclass

from agent.config import Settings


@dataclass
class LLMClient:
    provider: str
    model: str = ""

    def plan(self, op, arm, parent, journal, cfg):
        raise NotImplementedError(f"LLM provider {self.provider} is not wired")


def build_llm(settings: Settings) -> LLMClient:
    return LLMClient(provider=settings.llm_provider, model=settings.llm_model)
