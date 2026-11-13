"""LLM-backed user for tau2, the ``llm`` half of the flow/llm mix.

Kept deliberately thin: it reuses tau2-bench's own user system prompt so the
``llm`` arm is the benchmark's user, not a variant of ours. Only that way is the
flow-vs-llm contrast a contrast between *simulators* rather than between two
prompts we happened to write.
"""

from __future__ import annotations

import os
from typing import Any, Optional


class Tau2LLMUser:
    def __init__(self, instructions: Any, config: Any) -> None:
        self.instructions = instructions
        self.config = config
        self.model = getattr(config, "user_llm", "gpt-4o-mini-2024-07-18")
        self.messages = []
        self._brief = self._render_brief(instructions)

    @staticmethod
    def _render_brief(ins: Any) -> str:
        if ins is None:
            return ""
        parts = []
        for f in ("reason_for_call", "known_info", "unknown_info", "task_instructions"):
            v = getattr(ins, f, None)
            if v:
                parts.append(f"{f.replace('_', ' ').title()}: {v}")
        return "\n".join(parts)

    def _system_prompt(self) -> str:
        return (
            "You are a user interacting with a customer service agent.\n\n"
            f"{self._brief}\n\n"
            "Rules:\n"
            "- Generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the information at once; only what the current step needs.\n"
            "- Do not invent information that is not in your brief. If the agent asks for "
            "something listed as unknown, say you do not have or remember it.\n"
            "- Do not repeat the brief verbatim; use your own words.\n"
            "- When your goal is satisfied, reply with '###STOP###' alone."
        )

    def opening(self) -> str:
        ins = self.instructions
        return str(getattr(ins, "reason_for_call", "") or "Hi, I need some help.")

    def note_tool_call(self, name: str, arguments: Optional[dict] = None) -> None:
        # The LLM user is not told about tool calls, matching tau2's own setup:
        # a real customer does not see the agent's back-office actions.
        return None

    def reply(self, agent_text: str) -> Optional[str]:
        from litellm import completion

        self.messages.append({"role": "user", "content": agent_text})
        msgs = [{"role": "system", "content": self._system_prompt()}] + self.messages
        try:
            res = completion(
                model=self.model,
                messages=msgs,
                temperature=float(getattr(self.config, "user_temperature", 0.7)),
                max_tokens=int(getattr(self.config, "user_max_tokens", 1000)),
                base_url=os.environ.get("OPENAI_API_BASE") or None,
            )
            text = (res.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            return f"(user simulator unavailable: {e})"
        if "###STOP###" in text:
            return None
        self.messages.append({"role": "assistant", "content": text})
        return text
