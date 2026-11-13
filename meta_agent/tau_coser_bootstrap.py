"""Fresh CoSER user plus an empty, editable flow implementation.

This is a training-interface adapter around the vendored tau2 UserSimulator.
It never consumes reference actions. The initial bundle contains no previous
flow rules, templates or helpers. Model failures propagate to the caller.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from functools import lru_cache


COSER_REPOSITORY = "Neph0s/CoSER-Llama-3.1-8B"
COSER_REVISION = "80d5c88072599026bd0b0e2eb4369a87e6a5c960"
COSER_CONFIG_SHA256 = "a6f261303b790f7e300534ad56e0b4bc4d8228af052e30288ea09663704fc9c2"
# None is the existing user ABI's stop result. An explicit sentinel is needed
# for a flow implementation to defer a turn without accidentally ending it.
DEFER_TO_COSER = object()


def user_configuration(environment=None):
    """Read explicit model transport settings; never substitute another model."""
    environment = os.environ if environment is None else environment
    model = environment.get("EAU_COSER_USER_MODEL")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("EAU_COSER_USER_MODEL must name the deployed CoSER model")
    source = environment.get("EAU_COSER_SOURCE_MODEL", COSER_REPOSITORY)
    if source != COSER_REPOSITORY:
        raise ValueError("CoSER bootstrap requires the declared CoSER source model")
    try:
        settings = json.loads(environment.get("EAU_COSER_USER_LLM_ARGS", "{}"))
    except (TypeError, ValueError):
        raise ValueError("EAU_COSER_USER_LLM_ARGS must be a JSON object") from None
    if not isinstance(settings, dict):
        raise ValueError("EAU_COSER_USER_LLM_ARGS must be a JSON object")
    settings = copy.deepcopy(settings)
    settings.setdefault("temperature", 0.7)
    settings.setdefault("max_tokens", 512)
    settings.setdefault("timeout", 120)
    settings.setdefault("num_retries", 0)
    # This CoSER checkpoint retains end_of_text as generation EOS, while its
    # actual chat turns use the existing, in-vocabulary eot_id token.
    settings.setdefault("stop", ["<|eot_id|>"])
    temperature = settings["temperature"]
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError("CoSER temperature must be finite and between zero and two")
    for key in ("max_tokens", "timeout"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"CoSER {key} must be a positive integer")
    if type(settings["num_retries"]) is not int or settings["num_retries"] < 0:
        raise ValueError("CoSER num_retries must be a nonnegative integer")
    return model, settings


@lru_cache(maxsize=1)
def _native_coser_simulator_type():
    """Keep empty tool-call content explicit across the pinned HTTP transport.

    SGLang 0.4.7 requires the nullable assistant ``content`` key, whereas
    LiteLLM omits a None value when serializing tool-only messages. An empty
    string renders identically in our serving template and survives that
    serializer. Apply this compatibility rule only to CoSER's native user
    tool calls; leave the vendored simulator and other model clients intact.
    """
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator import UserSimulator

    def explicit_tool_content(message):
        if (isinstance(message, UserMessage) and message.is_tool_call()
                and message.content is None):
            message.content = ""

    class CoSERNativeUserSimulator(UserSimulator):
        def generate_next_message(self, message, state):
            # Also cover actual flow-tool history appended before CoSER resumes.
            for previous in state.messages:
                explicit_tool_content(previous)
            response, state = super().generate_next_message(message, state)
            # Native generate appends this same response object to state.
            explicit_tool_content(response)
            return response, state

    return CoSERNativeUserSimulator


class CoSERUser:
    """Use the native tau2 prompt, state and native user-tool conversation."""

    def __init__(self, instructions, user_tools=(), *, environment=None, record=None,
                 simulator_factory=None):
        if simulator_factory is None:
            simulator_factory = _native_coser_simulator_type()
        model, settings = user_configuration()
        self.model, self.llm_args = model, settings
        self.simulator = simulator_factory(
            llm=model, llm_args=settings, instructions=str(instructions),
            tools=list(user_tools or ()) or None,
        )
        self.state = self.simulator.get_init_state()
        self.environment = environment
        self.record = record if record is not None else lambda message: None
        self.tool_runner = None
        self.stopped = False

    @staticmethod
    def first_agent_message():
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
        return copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)

    @staticmethod
    def agent_message(text):
        from tau2.data_model.message import AssistantMessage
        return AssistantMessage(role="assistant", content=text)

    def observe_incoming(self, message):
        self.state.messages.append(message)

    def observe_user(self, text):
        from tau2.data_model.message import UserMessage
        self.state.messages.append(UserMessage(role="user", content=text))

    def observe_flow_tool(self, call, result):
        """Keep actual user-tool calls available to later CoSER fallback turns."""
        from tau2.data_model.message import UserMessage
        self.state.messages.extend([
            UserMessage(role="user", content=None, tool_calls=[call]),
            result,
        ])

    def call_tool(self, call):
        from tau2.data_model.message import ToolMessage
        if self.environment is not None:
            result = self.environment.get_response(call)
        elif self.tool_runner is not None:
            result = ToolMessage(role="tool", id=call.id, requestor="user",
                                 content=str(self.tool_runner(call.name, dict(call.arguments or {}))))
        else:
            raise RuntimeError("CoSER emitted a user tool call without an environment runner")
        self.record(result.model_dump())
        return result

    def respond(self, incoming):
        from tau2.data_model.message import MultiToolMessage, ToolMessage
        if self.stopped:
            return None
        for _ in range(200):
            response, self.state = self.simulator.generate_next_message(incoming, self.state)
            if self.simulator.is_stop(response):
                self.record(response.model_dump())
                self.stopped = True
                return None
            if not response.is_tool_call():
                if not isinstance(response.content, str) or not response.content.strip():
                    # An empty end-of-turn generation must end this user
                    # episode. Reusing it as a visible utterance poisons the
                    # native role-flipped history on the following user call.
                    self.record(response.model_dump())
                    self.stopped = True
                    return None
                return response.content
            self.record(response.model_dump())
            results = []
            for call in response.tool_calls:
                results.append(self.call_tool(call))
            incoming = results[0] if len(results) == 1 else MultiToolMessage(
                role="tool", tool_messages=results,
            )
        raise RuntimeError("CoSER exceeded the native user-tool response budget")


class BootstrapUser:
    """The old factory ABI, with an empty flow and a native CoSER fallback.

The public factory ABI has no reference-action argument. A flow object may return
DEFER_TO_COSER for a turn. Its visible replies and actual tool outputs are then
retained in the CoSER history, so switching back does not erase the dialogue.
The initial factory returns None and all turns use CoSER.
"""

    tau2_public_user_api = 1

    def __init__(self, public_scenario, knobs, *, user_tools, environment, record,
                 task_id=None, create_flow_user=None, simulator_factory=None):
        self.knobs = copy.deepcopy(knobs or {})
        self.instructions = public_scenario
        self.task_id = str(task_id) if task_id is not None else None
        probability = self.knobs.get("flow_probability", 0.0)
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("flow_probability must be finite and between zero and one")
        self.flow_probability = probability
        self.coser = CoSERUser(public_scenario, user_tools, environment=environment, record=record,
                              simulator_factory=simulator_factory)
        self.flow = (create_flow_user(public_scenario, copy.deepcopy(self.knobs), list(user_tools or ()))
                     if create_flow_user is not None else None)
        if self.flow is not None and not all(callable(getattr(self.flow, name, None))
                                             for name in ("opening", "reply")):
            raise TypeError("A flow user must provide opening() and reply(agent_text)")
        self.tool_runner = None
        self._turn = 0
        self._stopped = False
        self.backend_history = []

    @property
    def tool_runner(self):
        return self._tool_runner

    @tool_runner.setter
    def tool_runner(self, runner):
        self._tool_runner = runner
        self.coser.tool_runner = runner
        if self.flow is not None:
            self.flow.tool_runner = self._flow_tool

    def _flow_tool(self, name, arguments=None):
        from tau2.data_model.message import ToolCall, UserMessage
        arguments = dict(arguments or {})
        call = ToolCall(id=uuid.uuid4().hex, name=name, arguments=arguments, requestor="user")
        self.coser.record(UserMessage(role="user", content=None, tool_calls=[call]).model_dump())
        result = self.coser.call_tool(call)
        self.coser.observe_flow_tool(call, result)
        return result.content or ""

    def _use_flow(self):
        self._turn += 1
        if self.flow is None or self.flow_probability == 0:
            return False
        digest = hashlib.sha256(f"{self.instructions}\nturn:{self._turn}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.flow_probability

    def _respond(self, incoming, *, opening=False):
        if self._stopped:
            return None
        if self._use_flow():
            self.coser.observe_incoming(incoming)
            result = self.flow.opening() if opening else self.flow.reply(incoming.content or "")
            if result is DEFER_TO_COSER:
                # Native UserSimulator skips empty input, keeping the just
                # observed incoming message and any actual flow tool results.
                result = self.coser.respond(self.coser.agent_message(""))
                backend = "coser_after_flow_defer"
            else:
                if result is not None and not isinstance(result, str):
                    raise TypeError("Flow reply must be text, None(stop), or DEFER_TO_COSER")
                if result is not None:
                    self.coser.observe_user(result)
                backend = "flow"
        else:
            result = self.coser.respond(incoming)
            backend = "coser"
        self.backend_history.append(backend)
        if result is None:
            self._stopped = True
        return result

    def opening(self):
        greeting = self.coser.first_agent_message()
        self.coser.record(greeting.model_dump())
        return self._respond(greeting, opening=True)

    def reply(self, agent_text):
        return self._respond(self.coser.agent_message(agent_text))

    def note_tool_call(self, name, arguments=None):
        # Only actual calls supplied by the environment reach a future flow.
        if self.flow is not None and callable(getattr(self.flow, "note_tool_call", None)):
            self.flow.note_tool_call(name, arguments)

    @property
    def all_done(self):
        return self._stopped


def bootstrap_bundle():
    """Return exactly three fresh UTF-8 bundle files, without legacy flow code."""
    entry = '''"""Fresh public user ABI; no reference actions are accepted."""
from meta_agent.tau_coser_bootstrap import BootstrapUser
from .flow import create_user

class Tau2FlowUser(BootstrapUser):
    def __init__(self, public_scenario, knobs, *, user_tools, environment, record, task_id=None):
        super().__init__(public_scenario, knobs, user_tools=user_tools, environment=environment,
                         record=record, task_id=task_id, create_flow_user=create_user)
'''
    flow = '''"""Empty flow implementation: CoSER handles every initial turn.

Implement create_user using the public scenario and allowed user tools only.
An optional object exposes opening(), reply(agent_text), and tool_runner.
Return DEFER_TO_COSER from a turn to retain its history and ask CoSER to reply;
None from a turn means stop. Increase tau2_flow.flow_probability to use it.
There are no inherited task rules, action plans, fact extractors or templates.
"""
from meta_agent.tau_coser_bootstrap import DEFER_TO_COSER

def create_user(instructions, knobs, user_tools):
    return None
'''
    policy = {"version": 0, "tau2_flow": {"flow_probability": 0.0}}
    return {"flow_user.py": entry.encode(), "flow.py": flow.encode(),
            "env_policy.json": (json.dumps(policy, indent=2) + "\n").encode()}
