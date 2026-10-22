"""Optional real τ³ 1.0.1 API checks with a mocked native user-model response.

Set TAU3_BENCHMARK_ROOT to an external checkout and install its core/gym
dependencies. These tests use official train tasks and the actual Gym,
UserSimulator, Orchestrator and evaluator. They measure API compatibility,
not model quality or training performance. Local HTTP fixtures make no external
network calls and perform no real model inference.
"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("TAU3_BENCHMARK_ROOT"),
                     "set TAU3_BENCHMARK_ROOT to a τ³-bench 1.0.1 checkout")
class NativeTau3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "tau3_native_adapter_under_test",
            REPO / "examples/tau3_minimal/tau_adapter.py",
        )
        cls.adapter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.adapter_module)

    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "PYTHON_DOTENV_DISABLED": "1", "OPENAI_API_KEY": "native-test-unused",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.adapter = self.adapter_module.Tau3Adapter(
            benchmark_root=os.environ["TAU3_BENCHMARK_ROOT"], domain="retail",
            reward_mode="env", user_model="openai/native-api-fixture",
            user_base_url="http://127.0.0.1:1/v1", max_turns=3,
        )
        import litellm
        telemetry = patch.object(litellm, "telemetry", False)
        telemetry.start()
        self.addCleanup(telemetry.stop)
        from loguru import logger
        logger.disable("tau2")
        self.addCleanup(logger.enable, "tau2")

    @staticmethod
    def user_response(**kwargs):
        from tau2.data_model.message import AssistantMessage
        return AssistantMessage(
            role="assistant", content="Please help me with my account.",
            raw_data={"internal_prompt": "RAW_SECRET_SENTINEL"},
        )

    @contextmanager
    def local_user_service(self, delay=0):
        """Serve canned completions; never inspect private task prompt contents."""
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append({"path": self.path, "model": body["model"],
                                 "authorization": self.headers.get("Authorization")})
                time.sleep(delay)
                payload = json.dumps({
                    "id": "chatcmpl-native-fixture", "object": "chat.completion",
                    "created": 0, "model": "native-api-fixture",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "Please help me."}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Expected when checking the native client timeout.

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=1)

    def test_real_native_train_episode_and_private_field_filtering(self):
        from tau2.registry import registry
        from tau2.user import user_simulator
        expected = registry.get_task_splits_loader("retail")()["train"]
        self.assertEqual({str(task.id) for task in self.adapter.tasks},
                         {str(task_id) for task_id in expected})

        class Policy:
            def __init__(self):
                self.calls = 0

            def sample(self, system, observation):
                self.calls += 1
                if not isinstance(system, str) or not isinstance(observation, str):
                    raise AssertionError("Policy received nonpublic native objects")
                return {"text": "Hello." if self.calls == 1 else
                        '{"name":"done","arguments":{}}'}

        with patch.object(user_simulator, "generate", side_effect=self.user_response) as generate:
            samples, reward, trajectory = self.adapter.rollout(Policy(), 0)
        self.assertEqual(len(samples), 2)
        self.assertGreaterEqual(generate.call_count, 2)
        self.assertTrue(0 <= reward <= 1)
        self.assertEqual(trajectory["split"], "train")
        self.assertEqual(trajectory["termination_reason"], "agent_stop")
        self.assertEqual(trajectory["reward_mode"], "env")
        self.assertFalse(trajectory["training_cutoff"])
        self.assertTrue(any(message["role"] == "user" for message in trajectory["messages"]))
        encoded = json.dumps(trajectory)
        for hidden in ("RAW_SECRET_SENTINEL", "evaluation_criteria", "raw_data", "user_scenario"):
            self.assertNotIn(hidden, encoded)
        for request in generate.call_args_list:
            self.assertEqual(request.kwargs["api_base"], "http://127.0.0.1:1/v1")
            self.assertEqual(request.kwargs["num_retries"], 0)
            self.assertGreater(request.kwargs["timeout"], 0)
            self.assertNotIn("api_key", request.kwargs)

    def test_native_user_timeout_is_an_error_instead_of_training_reward(self):
        from tau2.user import user_simulator
        with patch.object(user_simulator, "generate", side_effect=TimeoutError("fixture timeout")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                self.adapter.rollout(None, 0)

    def test_native_litellm_calls_explicit_local_service_with_declared_key(self):
        class DonePolicy:
            def sample(self, system, observation):
                return {"text": '{"name":"done","arguments":{}}'}

        with self.local_user_service() as (url, requests):
            with patch.dict(os.environ, {"TAU3_NATIVE_FIXTURE_KEY": "native-fixture-credential"}):
                adapter = self.adapter_module.Tau3Adapter(
                    benchmark_root=os.environ["TAU3_BENCHMARK_ROOT"], domain="retail",
                    reward_mode="env", user_model="openai/native-api-fixture", user_base_url=url,
                    user_api_key_env="TAU3_NATIVE_FIXTURE_KEY", max_turns=2,
                )
                _, reward, trajectory = adapter.rollout(DonePolicy(), 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(requests[0]["model"], "native-api-fixture")
        self.assertEqual(requests[0]["authorization"], "Bearer native-fixture-credential")
        self.assertTrue(0 <= reward <= 1)
        self.assertNotIn("native-fixture-credential", json.dumps(trajectory))
        self.assertEqual(trajectory["termination_reason"], "agent_stop")

    def test_actual_native_local_http_timeout_refuses_policy_update(self):
        with self.local_user_service(delay=0.3) as (url, requests):
            self.adapter.options["user_llm_args"].update(api_base=url, timeout=0.05)
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "failed"):
                self.adapter.rollout(None, 0)
            self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(requests)

    def test_policy_exception_unblocks_native_orchestrator_thread(self):
        from tau2.user import user_simulator
        before = {thread.ident for thread in threading.enumerate()}

        class BrokenPolicy:
            def sample(self, system, observation):
                raise ValueError("fixture policy failure")

        with patch.object(user_simulator, "generate", side_effect=self.user_response):
            with self.assertRaisesRegex(ValueError, "fixture policy failure"):
                self.adapter.rollout(BrokenPolicy(), 0)
        leaked = [thread for thread in threading.enumerate()
                  if thread.ident not in before and "_run_orchestrator" in thread.name]
        self.assertEqual(leaked, [])


if __name__ == "__main__":
    unittest.main()
