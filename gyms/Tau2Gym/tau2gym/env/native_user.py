"""Adapt the vendored, unchanged tau2 UserSimulator to the training interface.

Only the public user scenario and allowed user tools cross this boundary. No
expected actions, simulator rules, custom prompts, or generated fallback text
are supplied to the benchmark user.
"""

from __future__ import annotations

import copy
import json
import os


class Tau2NativeUser:
    def __init__(self, scenario, tools, environment, record, *, max_tool_rounds=200):
        from tau2.config import DEFAULT_LLM_ARGS_USER, DEFAULT_LLM_USER
        from tau2.user.user_simulator import UserSimulator

        self.model = os.environ.get("EAU_NATIVE_USER_MODEL") or DEFAULT_LLM_USER
        args = json.loads(os.environ.get("EAU_NATIVE_USER_LLM_ARGS", "null"))
        if args is None:
            args = copy.deepcopy(DEFAULT_LLM_ARGS_USER)
        if not isinstance(args, dict):
            raise ValueError("EAU_NATIVE_USER_LLM_ARGS must be a JSON object")
        self.simulator = UserSimulator(
            llm=self.model, llm_args=args, instructions=str(scenario), tools=tools or None,
        )
        self.state = self.simulator.get_init_state()
        self.environment = environment
        self.record = record
        self.max_tool_rounds = max_tool_rounds

    def opening(self):
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

        greeting = copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        self.record(greeting.model_dump())
        return self._respond(greeting)

    def reply(self, agent_text):
        from tau2.data_model.message import AssistantMessage

        return self._respond(AssistantMessage(role="assistant", content=agent_text))

    def _respond(self, incoming):
        from tau2.data_model.message import MultiToolMessage

        for _ in range(self.max_tool_rounds):
            response, self.state = self.simulator.generate_next_message(incoming, self.state)
            if self.simulator.is_stop(response):
                self.record(response.model_dump())
                return None
            if not response.is_tool_call():
                # The training env records the visible user utterance once.
                return response.content or ""
            self.record(response.model_dump())
            results = []
            for call in response.tool_calls:
                result = self.environment.get_response(call)
                self.record(result.model_dump())
                results.append(result)
            incoming = (results[0] if len(results) == 1 else
                        MultiToolMessage(role="tool", tool_messages=results))
        raise RuntimeError("Native user exceeded its user-tool response budget")
