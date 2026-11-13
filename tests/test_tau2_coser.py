"""CPU tests with the real vendored UserSimulator and scripted generation.

No dataset, platform, GPU, HTTP or real model request is used.
Run with the lightweight tau2 dependency environment:
python -m unittest discover -s tests -p test_tau2_coser.py
"""
import hashlib
import json
import inspect
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
workspace = os.environ.get("TAU2_TRAIN_WORKSPACE")
if not workspace:
    raise unittest.SkipTest("Set TAU2_TRAIN_WORKSPACE for native CoSER CPU integration")
benchmark = Path(workspace).resolve() / "benchmark"
sys.path.insert(0, str(benchmark / "src"))
sys.path.insert(0, str(ROOT / "gyms/Tau2Gym/tau2gym/env"))
os.environ.setdefault("TAU2_DATA_DIR", str(benchmark / "data"))
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from meta_agent import tau_coser_bootstrap as bootstrap
from usersim_runtime import load_usersim_bundle
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.tasks import StructuredUserInstructions, UserScenario
from tau2.environment.tool import Tool
from tau2.user.user_simulator import UserSimulator


class UnreadableReferenceActions:
    def __str__(self):
        raise AssertionError("Reference actions were read")

    def __iter__(self):
        raise AssertionError("Reference actions were iterated")


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {
            "EAU_COSER_USER_MODEL": "openai/coser-usersim",
            "EAU_COSER_SOURCE_MODEL": bootstrap.COSER_REPOSITORY,
            "EAU_COSER_USER_LLM_ARGS": json.dumps({"api_base": "http://127.0.0.1:31999/v1"}),
        })
        self.environment.start(); self.addCleanup(self.environment.stop)
        self.scenario = UserScenario(persona="A patient caller.", instructions=StructuredUserInstructions(
            domain="synthetic", reason_for_call="Restore my synthetic service.",
            known_info="My nickname is Pixel.", unknown_info="I do not know my PIN.",
            task_instructions="Use only my public scenario.",
        ))
        self.tmp = tempfile.TemporaryDirectory(prefix="tau-coser-bootstrap-unit-")
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name).resolve() / "fresh_coser_run"
        (self.run / "live").mkdir(parents=True)

    def snapshot(self, version=0, files=None):
        content = bootstrap.bootstrap_bundle() if files is None else files
        snapshot = self.run / "versions/versions" / f"v{version:06d}"
        snapshot.mkdir(parents=True)
        for relative, raw in content.items():
            (snapshot / relative).write_bytes(raw)
        files = {name: hashlib.sha256(raw).hexdigest() for name, raw in content.items()}
        pointer = dict(schema_version=1, run=self.run.name, version=version, snapshot=str(snapshot),
                       files=files, entrypoint="flow_user.py",
                       bundle_sha256=hashlib.sha256(json.dumps(files, sort_keys=True,
                                                              separators=(",", ":")).encode()).hexdigest())
        (self.run / "live/usersim_bundle.json").write_text(json.dumps(pointer))
        return pointer

    def user(self, *, knobs=None, create_flow_user=None, user_tools=(), environment=None, record=None):
        return bootstrap.BootstrapUser(self.scenario, knobs or {}, user_tools=user_tools,
                                       environment=environment, record=record or (lambda _: None),
                                       create_flow_user=create_flow_user)

    def test_explicit_real_coser_model_and_defaults_no_fallback(self):
        model, arguments = bootstrap.user_configuration()
        self.assertEqual(model, "openai/coser-usersim")
        self.assertEqual(arguments["temperature"], 0.7)
        self.assertEqual(arguments["max_tokens"], 512)
        self.assertEqual(arguments["num_retries"], 0)
        self.assertEqual(arguments["stop"], ["<|eot_id|>"])
        for environment in ({}, {"EAU_COSER_USER_MODEL": "openai/qwen", "EAU_COSER_SOURCE_MODEL": "Qwen/7B"},
                            {"EAU_COSER_USER_MODEL": "openai/coser", "EAU_COSER_USER_LLM_ARGS": "[]"},
                            {"EAU_COSER_USER_MODEL": "openai/coser", "EAU_COSER_USER_LLM_ARGS": '{"temperature": NaN}'}):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                bootstrap.user_configuration(environment)

    def test_three_file_blank_bundle_loads_native_user_without_reference_actions(self):
        content = bootstrap.bootstrap_bundle()
        self.assertEqual(set(content), {"flow_user.py", "flow.py", "env_policy.json"})
        self.assertEqual(json.loads(content["env_policy.json"]),
                         {"version": 0, "tau2_flow": {"flow_probability": 0.0}})
        self.assertNotIn(b"extract_facts", b"".join(content.values()))
        self.assertNotIn(b"tau2gym.env.flow_user", b"".join(content.values()))
        pointer = self.snapshot()
        factory, policy, metadata = load_usersim_bundle(self.run / "live/usersim_bundle.json", self.run.name)
        self.assertEqual(factory.tau2_public_user_api, 1)
        self.assertNotIn("expected_actions", inspect.signature(factory).parameters)
        user = factory(self.scenario, policy["tau2_flow"], user_tools=[], environment=None,
                       record=lambda _: None)
        self.assertIsInstance(user.coser.simulator, UserSimulator)
        with mock.patch("tau2.user.user_simulator.generate", return_value=AssistantMessage(
                role="assistant", content="Could you help restore my service?")) as generation:
            self.assertEqual(user.opening(), "Could you help restore my service?")
        kwargs = generation.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/coser-usersim")
        self.assertEqual(kwargs["temperature"], 0.7)
        system = kwargs["messages"][0].content
        self.assertIn("A patient caller", system)
        self.assertIn("My nickname is Pixel", system)
        self.assertEqual(metadata["usersim_files"], pointer["files"])
        self.assertEqual(user.backend_history, ["coser"])
        self.assertIsNone(user.flow)

    def test_native_user_tools_roundtrip_uses_actual_runner_and_native_history(self):
        def check_device() -> str:
            """Inspect the caller's synthetic device."""
            return "synthetic result"
        from tau2.data_model.message import ToolMessage
        environment = mock.Mock()
        environment.get_response.return_value = ToolMessage(
            role="tool", id="user-call-1", requestor="user", content="Device is online.", error=True)
        record = mock.Mock()
        user = self.user(user_tools=[Tool(check_device)], environment=environment, record=record)
        responses = [AssistantMessage(role="assistant", tool_calls=[
            ToolCall(id="user-call-1", name="check_device", arguments={})]),
            AssistantMessage(role="assistant", content="My device is online.")]
        with mock.patch("tau2.user.user_simulator.generate", side_effect=responses) as generation:
            self.assertEqual(user.opening(), "My device is online.")
        environment.get_response.assert_called_once()
        self.assertEqual(environment.get_response.call_args.args[0].requestor, "user")
        self.assertEqual(generation.call_count, 2)
        messages = user.coser.state.messages
        self.assertEqual(messages[-2].requestor, "user")
        self.assertEqual(messages[-2].id, "user-call-1")
        self.assertEqual(messages[-2].content, "Device is online.")
        self.assertTrue(messages[-2].error)
        self.assertEqual(messages[-3].tool_calls[0].requestor, "user")
        self.assertEqual(record.call_args_list[-1].args[0]["id"], "user-call-1")
        self.assertTrue(record.call_args_list[-1].args[0]["error"])

    def test_flow_visible_reply_and_tools_are_preserved_when_coser_resumes(self):
        class NewFlow:
            def opening(self):
                self.tool_runner("check_device", {})
                return "This is the flow's visible opening."

            def reply(self, agent_text):
                return bootstrap.DEFER_TO_COSER

        from tau2.data_model.message import ToolMessage
        class SyntheticEnvironment:
            def get_response(self, call):
                return ToolMessage(role="tool", id=call.id, requestor="user",
                                   content="Actual synthetic device output.")
        user = self.user(knobs={"flow_probability": 1.0}, create_flow_user=lambda *_: NewFlow(),
                         environment=SyntheticEnvironment())
        with mock.patch("tau2.user.user_simulator.generate", return_value=AssistantMessage(
                role="assistant", content="The coherent CoSER fallback reply.")) as generation:
            self.assertEqual(user.opening(), "This is the flow's visible opening.")
            generation.assert_not_called()
            self.assertEqual(user.reply("Please confirm the latest observation."),
                             "The coherent CoSER fallback reply.")
        kwargs = generation.call_args.kwargs
        history = kwargs["messages"]
        contents = [m.content for m in history]
        self.assertIn("This is the flow's visible opening.", contents)
        self.assertIn("Actual synthetic device output.", contents)
        self.assertEqual(contents.count("Please confirm the latest observation."), 1)
        self.assertEqual(user.backend_history, ["flow", "coser_after_flow_defer"])

    def test_empty_flow_still_coser_even_if_probability_changes(self):
        user = self.user(knobs={"flow_probability": 1.0}, create_flow_user=lambda *_: None)
        with mock.patch("tau2.user.user_simulator.generate", return_value=AssistantMessage(
                role="assistant", content="Still CoSER.")):
            self.assertEqual(user.opening(), "Still CoSER.")
        self.assertEqual(user.backend_history, ["coser"])

    def test_next_snapshot_enables_new_flow_without_replacing_existing_episode(self):
        self.snapshot()
        factory, policy, _ = load_usersim_bundle(self.run / "live/usersim_bundle.json")
        existing = factory(self.scenario, policy["tau2_flow"], user_tools=[], environment=None,
                           record=lambda _: None)
        content = bootstrap.bootstrap_bundle()
        content["flow.py"] = b'''class NewFlow:
    def opening(self): return "New flow episode."
    def reply(self, agent_text): return "New flow reply."
def create_user(instructions, knobs, user_tools): return NewFlow()
'''
        content["env_policy.json"] = b'{"version":1,"tau2_flow":{"flow_probability":1.0}}\n'
        self.snapshot(1, content)
        next_factory, next_policy, _ = load_usersim_bundle(self.run / "live/usersim_bundle.json")
        new_user = next_factory(self.scenario, next_policy["tau2_flow"], user_tools=[], environment=None,
                               record=lambda _: None)
        with mock.patch("tau2.user.user_simulator.generate", return_value=AssistantMessage(
                role="assistant", content="Existing episode remains CoSER.")) as generation:
            self.assertEqual(new_user.opening(), "New flow episode.")
            generation.assert_not_called()
            self.assertEqual(existing.opening(), "Existing episode remains CoSER.")
        self.assertIsNone(existing.flow)
        self.assertNotEqual(factory.__module__, next_factory.__module__)

    def test_model_failure_propagates_and_stop_is_not_a_fake_utterance(self):
        user = self.user()
        with mock.patch("tau2.user.user_simulator.generate", side_effect=RuntimeError("synthetic unavailable")):
            with self.assertRaisesRegex(RuntimeError, "synthetic unavailable"):
                user.opening()
        stopped = self.user()
        with mock.patch("tau2.user.user_simulator.generate", return_value=AssistantMessage(
                role="assistant", content="###STOP###")):
            self.assertIsNone(stopped.opening())
        self.assertTrue(stopped.all_done)
        self.assertIsNone(stopped.reply("No more turns."))

    def test_empty_generation_ends_episode_without_fabrication_or_reusing_invalid_history(self):
        for content in [None, '', ' \n ']:
            with self.subTest(content=content):
                user = self.user()
                with mock.patch('tau2.user.user_simulator.generate', return_value=AssistantMessage(
                        role='assistant', content=content)) as generation:
                    self.assertIsNone(user.opening())
                    self.assertTrue(user.all_done)
                    self.assertIsNone(user.reply('This must not trigger another model request.'))
                    generation.assert_called_once()

    def test_independent_users_have_fresh_native_histories(self):
        first, second = self.user(), self.user()
        self.assertIsNot(first.coser.state, second.coser.state)
        with mock.patch("tau2.user.user_simulator.generate", return_value=AssistantMessage(
                role="assistant", content="First user's reply.")):
            first.opening()
        self.assertEqual(second.coser.state.messages, [])
        self.assertEqual(second.backend_history, [])

    def test_native_no_tools_none_is_supported_for_retail_and_airline(self):
        user = self.user(user_tools=None, create_flow_user=lambda *_: None)
        self.assertIsNone(user.coser.simulator.tools)
        with mock.patch("tau2.user.user_simulator.generate", return_value=AssistantMessage(
                role="assistant", content="A user without device tools.")) as generation:
            self.assertEqual(user.opening(), "A user without device tools.")
        self.assertIsNone(generation.call_args.kwargs["tools"])


if __name__ == "__main__":
    unittest.main()
