from __future__ import annotations

import json
import os
from dataclasses import dataclass

from agent.config import Settings
from agent.llm.openai_compat import LLMError, chat_completions
from agent.llm.prompts import SYSTEM, user_prompt
from agent.llm.schema import default_improve, extract_json, plan_from_payload
from agent.memory.journal import Journal, Node
from agent.recsys.arms import Arm
from agent.types import Change, Hypothesis


def force_action(op: str, payload: dict) -> dict:
    out = dict(payload)
    got = str(out.get("action") or "")
    if op in {"ablate", "ensemble"}:
        out["action"] = op
    elif op == "draft":
        if got in {"research", "read_paper"}:
            out["action"] = "skip"
            out["skip_reason"] = out.get("skip_reason") or "draft cannot cheap-act"
        else:
            out["action"] = "improve"
    elif op == "improve" and got not in {"", "improve", "skip", "research", "read_paper", "diagnose"}:
        out["action"] = "skip"
        out["skip_reason"] = out.get("skip_reason") or f"policy asked improve, got {got}"
    return out


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

    def plan(self, op, arm, parent, journal, cfg, eda_text="", skill_text="", notes_text="", tried_text="", files_window=False):
        raise NotImplementedError(f"LLM provider {self.provider} is not wired")


@dataclass
class OpenAIClient(LLMClient):
    provider: str = "openai"

    def _complete(self, messages: list) -> str:
        text, tin, tout = chat_completions(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            timeout=180,
            extra_body=deepseek_extra(self.base_url),
        )
        self.tokens_in += tin
        self.tokens_out += tout
        return text

    def _fallback(self, arm: Arm, cfg: dict, err: str) -> tuple[Hypothesis, Change]:
        patch = default_improve(arm.arm_id, cfg)
        hyp = Hypothesis(f"LLM fallback after {err}", arm.arm_id)
        if patch:
            return hyp, Change("diff", config_patch=patch)
        return hyp, Change("diff", action="skip", skip_reason=err)

    def plan(
        self,
        op,
        arm: Arm,
        parent: Node | None,
        journal: Journal,
        cfg: dict,
        eda_text: str = "",
        skill_text: str = "",
        notes_text: str = "",
        tried_text: str = "",
        files_window: bool = False,
    ) -> tuple[Hypothesis, Change]:
        self.reset_usage()
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": user_prompt(
                    op,
                    arm,
                    parent,
                    journal,
                    cfg,
                    eda_text,
                    skill_text,
                    notes_text,
                    tried_text,
                    files_window=files_window,
                ),
            },
        ]
        last = ""
        for attempt in range(2):
            try:
                last = self._complete(messages)
                payload = extract_json(last)
                return plan_from_payload(
                    arm.arm_id,
                    force_action(op, payload),
                    expected_action=op,
                    data_scale=str((cfg or {}).get("data_scale") or ""),
                )
            except LLMError as exc:
                self.last_error = str(exc)
                return self._fallback(arm, cfg, str(exc))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self.last_error = str(exc)
                messages.append({"role": "assistant", "content": last[:2000]})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Invalid JSON ({exc}). Reply with the JSON object only.",
                    }
                )
        return self._fallback(arm, cfg, self.last_error or "invalid json")


def _env_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("XAI_API_KEY")
        or ""
    )


def _env_base() -> str:
    if os.environ.get("OPENAI_BASE_URL"):
        return os.environ["OPENAI_BASE_URL"]
    if os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return "https://api.deepseek.com"
    if os.environ.get("XAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return "https://api.x.ai/v1"
    return "https://api.openai.com/v1"


def _env_model(base_url: str, configured: str) -> str:
    if os.environ.get("LLM_MODEL"):
        return os.environ["LLM_MODEL"]
    if configured:
        return configured
    if "deepseek.com" in base_url:
        return "deepseek-v4-flash"
    if "x.ai" in base_url:
        return "grok-3-mini"
    return "gpt-4o-mini"


def deepseek_extra(base_url: str) -> dict:
    if "deepseek.com" not in base_url:
        return {}
    effort = os.environ.get("LLM_REASONING_EFFORT") or "max"
    return {
        "thinking": {"type": "enabled"},
        "reasoning_effort": effort,
    }


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
