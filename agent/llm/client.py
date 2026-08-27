from __future__ import annotations

import json
import os
from dataclasses import dataclass

from agent.config import Settings
from agent.llm.openai_compat import LLMError, chat_completions
from agent.llm.prompts import SYSTEM, user_prompt
from agent.llm.schema import extract_json, plan_from_payload
from agent.memory.journal import Journal, Node
from agent.recsys.arms import Arm
from agent.types import Change, Hypothesis


@dataclass
class LLMClient:
    provider: str
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.3
    tokens_in: int = 0
    tokens_out: int = 0
    last_error: str = ""

    def reset_usage(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.last_error = ""

    def plan(self, op, arm, parent, journal, cfg):
        raise NotImplementedError(f"LLM provider {self.provider} is not wired")


@dataclass
class OpenAIClient(LLMClient):
    provider: str = "openai"

    def plan(
        self, op, arm: Arm, parent: Node | None, journal: Journal, cfg: dict
    ) -> tuple[Hypothesis, Change]:
        self.reset_usage()
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(op, arm, parent, journal, cfg)},
        ]
        try:
            text, tin, tout = chat_completions(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
        except LLMError as exc:
            self.last_error = str(exc)
            hyp = Hypothesis(f"LLM call failed: {exc}", arm.arm_id)
            return hyp, Change("diff", skip=True, skip_reason=str(exc))
        self.tokens_in, self.tokens_out = tin, tout
        try:
            payload = extract_json(text)
            return plan_from_payload(arm.arm_id, payload)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
            hyp = Hypothesis(f"LLM JSON parse failed: {exc}", arm.arm_id)
            return hyp, Change("diff", skip=True, skip_reason=str(exc))


def _env_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("XAI_API_KEY") or ""


def _env_base() -> str:
    if os.environ.get("OPENAI_BASE_URL"):
        return os.environ["OPENAI_BASE_URL"]
    if os.environ.get("XAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return "https://api.x.ai/v1"
    return "https://api.openai.com/v1"


def _env_model(base_url: str, configured: str) -> str:
    if os.environ.get("LLM_MODEL"):
        return os.environ["LLM_MODEL"]
    if configured:
        return configured
    if "x.ai" in base_url:
        return "grok-3-mini"
    return "gpt-4o-mini"


def build_llm(settings: Settings) -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER") or settings.llm_provider
    key = _env_key()
    base = settings.llm_base_url or _env_base()
    model = _env_model(base, settings.llm_model)
    if provider == "auto":
        provider = "openai" if key else "dummy"
    if provider in {"openai", "xai"} and key:
        return OpenAIClient(
            provider=provider,
            model=model,
            base_url=base,
            api_key=key,
            temperature=settings.llm_temperature,
        )
    return LLMClient(provider="dummy", model=model)
