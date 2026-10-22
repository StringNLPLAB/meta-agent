"""CPU checks for the real-data recipe and native verl environment boundary."""

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("tau3_verl_prepare", "examples/tau3_verl/prepare.py")
launch = load("tau3_verl_launch", "examples/tau3_verl/launch.py")
native = load("tau3_verl_native", "gyms/Tau2Gym/tau2gym/native.py")


class RecipeTests(unittest.TestCase):
    def test_partition_is_deterministic_disjoint_and_train_only(self):
        tasks = [SimpleNamespace(id=str(index)) for index in range(30)]
        train, dev = prepare.partition_tasks(tasks, 0.1, 42)
        self.assertEqual(len(train), 27)
        self.assertEqual(len(dev), 3)
        self.assertFalse({item.id for item in train} & {item.id for item in dev})
        self.assertEqual({item.id for item in train + dev}, {item.id for item in tasks})
        self.assertEqual([task.id for task in prepare.partition_tasks(tasks, .1, 42)[0]], [task.id for task in train])
        for fraction in (0, 0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                prepare.partition_tasks(tasks, fraction, 42)

    def test_parquet_rows_do_not_include_hidden_tasks(self):
        task = SimpleNamespace(id="train_a", evaluation_criteria={"answer": "HIDDEN_MARKER"}, user_scenario="PRIVATE_USER_PROMPT")
        schemas = [
            {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "cancel", "parameters": {"type": "object", "properties": {"order_id": {"type": "integer"}}}}},
        ]
        rows = prepare.rows_for_tasks([task], policy="public policy", schemas=schemas,
                                     domain="retail", environment={"task_split": "train", "domain": "retail"}, role="train")
        encoded = json.dumps(rows)
        self.assertNotIn("HIDDEN_MARKER", encoded)
        self.assertNotIn("PRIVATE_USER_PROMPT", encoded)
        row = rows[0]
        self.assertEqual(row["reward_model"]["type"], "env_reward")
        self.assertNotIn("tool_schemas", row["extra_info"])
        self.assertEqual(row["extra_info"]["tool_names"], ["search", "cancel"])
        self.assertEqual(row["extra_info"]["tools_kwargs"]["_tau2_env_kwargs"]["task_id"], "train_a")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("PyArrow is an optional training dependency")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            self.assertEqual(pq.read_table(path).to_pylist(), rows)

    def test_actual_hydra_composition_preserves_budget_and_no_test_feedback(self):
        if not importlib.util.find_spec("hydra"):
            self.skipTest("Hydra is a training dependency")
        args = prepare.parser().parse_args([
            "--benchmark-root", "/tmp/tau3", "--model-path", "/tmp/model [local],v1",
            "--user-model", "openai/local-user", "--user-base-url", "http://localhost:9999/v1",
            "--run", "RECIPE_TEST", "--steps", "7",
        ])
        args.train_tasks = 20
        output = ROOT / "reports/tau3_verl/RECIPE_TEST"
        config = launch.composed_config({"output_dir": str(output), "overrides": prepare.overrides_for(args, output)})
        self.assertEqual(config["trainer"]["total_training_steps"], 7)
        self.assertEqual(config["actor_rollout_ref"]["model"]["path"], "/tmp/model [local],v1")
        self.assertEqual(config["algorithm"]["adv_estimator"], "info_grpo")
        self.assertEqual(config["trainer"]["test_freq"], -1)
        self.assertFalse(config["trainer"]["val_before_train"])
        self.assertTrue(config["data"]["val_files"].endswith("dev.parquet"))
        self.assertEqual(config["trainer"]["log_train_trajectories_freq"], 1)

    def test_runtime_owns_the_real_launcher_and_current_training_evidence(self):
        args = prepare.parser().parse_args([
            "--benchmark-root", "/tmp/tau3", "--model-path", "/tmp/model",
            "--user-model", "openai/user", "--user-base-url", "http://localhost:9999/v1",
            "--run", "RUNTIME_TEST", "--meta-model", "flint", "--meta-base-url", "http://localhost:8000/v1",
        ])
        output = ROOT / "reports/tau3_verl/RUNTIME_TEST"
        config = prepare.runtime_config(args, output, output / "recipe.json")
        self.assertIn(str(ROOT / "examples/tau3_verl/launch.py"), config["trainer_command"])
        self.assertIn("reports/tau3_verl/RUNTIME_TEST/checkpoints/debug_trajectories", config["supervisor_args"])
        self.assertEqual(config["failure_policy"], "stop_owned_trainer")
        self.assertEqual(config["contract"]["allowed_operations"], [])


class NativeBridgeTests(unittest.TestCase):
    def options(self, **overrides):
        return dict(benchmark_root="/benchmark", domain="retail", task_id="train_a",
                    user_model="openai/user", user_base_url="http://localhost:9999/v1", **overrides)

    def adapter(self, **kwargs):
        class Gym:
            def __init__(self, **kwargs):
                self.native_error = None
                self._simulation_done = threading.Event()
                self._simulation_run = SimpleNamespace(termination_reason=SimpleNamespace(value="user_stop"))
            def reset(self):
                return "hello", {"task": {"HIDDEN": True}}
            def step(self, action):
                if action.startswith('{"name":"done"'):
                    self._simulation_done.set()
                    return "bye", 1.0, True, False, {"task": {"HIDDEN": True}}
                return "visible reply", 0.0, False, False, {"task": {"HIDDEN": True}}
            def _get_reward(self):
                if self.native_error:
                    raise self.native_error
                return 1.0, '{"hidden":true}'
        return SimpleNamespace(tasks=[SimpleNamespace(id="train_a")], env_type=Gym,
                               options={"reward_mode": "action", "all_messages_as_observation": True})

    def test_only_official_train_task_can_start(self):
        with patch.object(native, "Tau3Adapter", side_effect=self.adapter):
            for kwargs in (self.options(task_split="test"), self.options(solo_mode=True)):
                with self.assertRaises(ValueError):
                    native.Tau2Env(**kwargs)
            options = self.options()
            options["task_id"] = "test_a"
            with self.assertRaisesRegex(ValueError, "official.*train split"):
                native.Tau2Env(**options)

    def test_public_observations_and_cutoff_never_expose_hidden_task(self):
        with patch.object(native, "Tau3Adapter", side_effect=self.adapter):
            env = native.Tau2Env(**self.options(max_turns=1))
            observation, info = env.reset()
            self.assertEqual(observation, {"feedback": "hello"})
            self.assertNotIn("HIDDEN", json.dumps(info))
            result = asyncio.run(env.step_async("hello"))
            self.assertEqual(result[1:4], (0.0, True, True))
            self.assertTrue(result[4]["training_cutoff"])
            self.assertFalse(result[4]["benchmark_score"])
            self.assertEqual(env.calculate_reward(), 0.0)

    def test_service_error_is_not_converted_to_zero_reward(self):
        with patch.object(native, "Tau3Adapter", side_effect=self.adapter):
            env = native.Tau2Env(**self.options())
            env.reset()
            env.tau2_env.native_error = RuntimeError("service down")
            with self.assertRaisesRegex(RuntimeError, "service failed"):
                env.step("hello")
            with self.assertRaisesRegex(RuntimeError, "service down"):
                env.calculate_reward()

    def test_native_done_preserves_reward_without_controller_cutoff(self):
        with patch.object(native, "Tau3Adapter", side_effect=self.adapter):
            env = native.Tau2Env(**self.options())
            env.reset()
            result = env.step('{"name":"done","arguments":{}}')
            self.assertEqual(result[1:4], (1.0, True, False))
            self.assertFalse(result[4]["training_cutoff"])
            self.assertEqual(env.calculate_reward(), 1.0)


if __name__ == "__main__":
    unittest.main()
