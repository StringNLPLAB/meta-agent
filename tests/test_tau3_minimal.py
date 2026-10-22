"""CPU optimizer smoke and native-API boundary tests; no weights or services needed."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tau3_minimal"
sys.path.insert(0, str(EXAMPLE))
import prepare
import tau_adapter
import train


@contextmanager
def temporary_run():
    with tempfile.TemporaryDirectory(prefix="tau3-minimal-test-") as directory:
        yield Path(directory)


class NativeBoundaryTests(unittest.TestCase):
    def registry(self, *, declared=("a", "b"), actual=("a", "b")):
        calls = []
        def loader(*, task_split_name):
            calls.append(task_split_name)
            return [SimpleNamespace(id=item) for item in actual]
        return SimpleNamespace(
            get_task_splits_loader=lambda domain: lambda: {"train": list(declared)},
            get_tasks_loader=lambda domain: loader, calls=calls,
        )

    def test_only_exact_official_train_split_is_accepted(self):
        registry = self.registry()
        self.assertEqual([task.id for task in tau_adapter.strict_train_tasks(registry, "retail")], ["a", "b"])
        self.assertEqual(registry.calls, ["train"])
        for registry in (self.registry(declared=()), self.registry(actual=("a", "hidden_test")),
                         self.registry(actual=("a", "a")), self.registry(declared=("a", "a"))):
            with self.assertRaises(ValueError):
                tau_adapter.strict_train_tasks(registry, "retail")

    def test_native_failure_cannot_become_zero_reward(self):
        class Base:
            def __init__(self, **kwargs):
                self.domain = "retail"
                self._simulation_run = None
                self._simulation_done = threading.Event()
        seen = []
        def evaluator(**kwargs):
            seen.append(kwargs)
            return SimpleNamespace(reward=0.75, model_dump_json=lambda: '{"reward":0.75}')
        env_class = tau_adapter.proxy_environment_class(
            Base, evaluator, SimpleNamespace(ACTION="action", ENV="env"),
        )
        task = SimpleNamespace(id="train")
        env = env_class(selected_task=task, reward_mode="env")
        self.assertIs(env._get_task(), task)
        self.assertEqual(env._get_reward()[0], 0)
        env._simulation_run = SimpleNamespace(messages=[], termination_reason=SimpleNamespace(value="agent_stop"))
        self.assertEqual(env._get_reward()[0], 0.75)
        self.assertEqual(seen[0]["evaluation_type"], "env")
        self.assertIs(seen[0]["task"], task)
        for reason in ("user_error", "infrastructure_error", "context_window_exceeded", "timeout"):
            env._simulation_run.termination_reason.value = reason
            with self.assertRaisesRegex(RuntimeError, "refusing policy update"):
                env._get_reward()
        def broken():
            raise RuntimeError("user service unreachable")
        env._orchestrator = SimpleNamespace(run=broken)
        env._run_orchestrator()
        self.assertTrue(env._simulation_done.is_set())
        with self.assertRaisesRegex(RuntimeError, "rollout failed"):
            env._get_reward()
        env.native_error, env._simulation_run = None, None
        with self.assertRaisesRegex(RuntimeError, "without a simulation"):
            env._get_reward()

    def test_visible_messages_omit_raw_requests_and_hidden_criteria(self):
        message = SimpleNamespace(role="assistant", content="visible", raw_data={"api_key": "secret"},
                                  task={"hidden": "answer"}, tool_calls=[SimpleNamespace(
                                      id="call", name="lookup", arguments={"id": 1}, requestor="assistant")])
        exported = tau_adapter.public_messages(SimpleNamespace(messages=[message]))
        self.assertEqual(exported[0]["content"], "visible")
        text = json.dumps(exported)
        self.assertNotIn("secret", text)
        self.assertNotIn("hidden", text)
        self.assertIn("lookup", text)

    def test_rollout_policy_receives_public_context_without_task(self):
        hidden = "DO_NOT_EXPOSE_CRITERION"
        task = SimpleNamespace(id="train_1", evaluation_criteria={"hidden": hidden})
        simulation = SimpleNamespace(messages=[SimpleNamespace(role="user", content="hello")],
                                     termination_reason=SimpleNamespace(value="agent_stop"))
        class Base:
            def __init__(self, **kwargs):
                self.domain = kwargs["domain"]
                self._simulation_done = threading.Event()
                self._simulation_run = None
                self._orchestrator_thread = None
            def reset(self):
                return "public greeting", {"policy": "public policy", "tools": [], "task": task}
            def step(self, action):
                self._simulation_done.set()
                self._simulation_run = simulation
                return "public response", self._get_reward()[0], True, False, {"task": task}
        adapter = tau_adapter.Tau3Adapter.__new__(tau_adapter.Tau3Adapter)
        adapter.tasks, adapter.options, adapter.max_turns = [task], {}, 2
        adapter.env_type = tau_adapter.proxy_environment_class(
            Base, lambda **kwargs: SimpleNamespace(reward=0.5, model_dump_json=lambda: "{}"),
            SimpleNamespace(ENV="env"),
        )
        adapter.options = {"domain": "retail", "reward_mode": "env"}
        observed = []
        policy = SimpleNamespace(sample=lambda system, obs: observed.append((system, obs)) or {"text": "hello"})
        samples, reward, record = adapter.rollout(policy, 0)
        self.assertEqual(reward, 0.5)
        self.assertEqual(len(samples), 1)
        self.assertNotIn(hidden, json.dumps([record, observed]))
        self.assertEqual(record["split"], "train")
        self.assertIn("public policy", record["policy_context"])

    def test_urls_refuse_embedded_credentials(self):
        for url in ("http://user:secret@localhost/v1", "http://localhost/v1?key=secret", "file:///tmp/api"):
            with self.assertRaises(ValueError):
                tau_adapter.service_url(url)

    def test_source_version_and_hashes_are_real(self):
        with temporary_run() as root:
            for name in tau_adapter.SOURCE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture\n")
            (root / "pyproject.toml").write_text('[project]\nname="tau2"\nversion="1.0.1"\n')
            identity = tau_adapter.benchmark_identity(root)
            self.assertEqual(identity["version"], "1.0.1")
            self.assertEqual(len(identity["sources"]), len(tau_adapter.SOURCE_FILES))
            (root / "pyproject.toml").write_text('[project]\nname="tau2"\nversion="9.9"\n')
            with self.assertRaisesRegex(ValueError, "1.0.1"):
                tau_adapter.benchmark_identity(root)


class PrepareTests(unittest.TestCase):
    def arguments(self, run):
        return prepare.parser().parse_args([
            "--root", str(ROOT), "--run", run, "--output", "/tmp/not-written.json", "--smoke",
            "--meta-model", "mock-flint", "--meta-base-url", "http://127.0.0.1:30000/v1",
        ])

    def test_prepared_runtime_has_distinct_owner_paths_and_read_only_contract(self):
        args = self.arguments("unit_prepare_9f80")
        config = prepare.launch_config(args)
        self.assertEqual(config["version"], 1)
        self.assertEqual(config["contract"]["allowed_operations"], [])
        self.assertEqual(config["failure_policy"], "stop_owned_trainer")
        self.assertGreater(config["final_timeout_seconds"], args.agent_timeout)
        flags = config["supervisor_args"]
        self.assertEqual(flags[flags.index("--log") + 1], "reports/runtime/unit_prepare_9f80/trainer.stdout")
        self.assertIn("--export-atif", flags)
        self.assertIn("--smoke", config["trainer_command"])
        for protected in ("--root", "--run", "--contract", "--completion-file", "--once", "--adopt-settings", "--stop-step"):
            self.assertNotIn(protected, flags)

    def test_ambiguous_smoke_and_unsafe_timeouts_are_rejected(self):
        args = self.arguments("unit_prepare_unsafe")
        args.policy_model = "/tmp/model"
        with self.assertRaisesRegex(ValueError, "must not be combined"):
            prepare.launch_config(args)
        args.policy_model = None
        args.final_timeout = args.agent_timeout
        with self.assertRaisesRegex(ValueError, "Final timeout"):
            prepare.launch_config(args)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch is not installed")
class OptimizerSmokeTests(unittest.TestCase):
    def test_sparse_saves_do_not_make_checkpoint_its_own_parent(self):
        with temporary_run() as root:
            args = train.parser().parse_args(["--smoke", "--run", "sparse_save", "--output-dir", str(root / "run"),
                                               "--steps", "4", "--episodes-per-step", "4", "--save-every", "2"])
            snapshots = []
            writer = train.write_json
            def capture(path, value):
                if Path(path).name == "manifest.json":
                    snapshots.append(dict(value))
                writer(path, value)
            with patch.object(train, "write_json", side_effect=capture):
                train.train(args)
            for manifest in snapshots:
                if manifest["checkpoint"]:
                    self.assertNotEqual(manifest["checkpoint"], manifest["parent_checkpoint"])
            step3 = next(item for item in snapshots if item["optimizer_steps"] == 3)
            self.assertEqual(step3["checkpoint"], "checkpoints/step_000002.pt")
            self.assertIsNone(step3["parent_checkpoint"])

    def test_actual_optimizer_updates_trajectories_and_checkpoint_lineage(self):
        import torch
        with temporary_run() as root:
            output = root / "training"
            command = [sys.executable, str(EXAMPLE / "train.py"), "--smoke", "--run", "smoke_test",
                       "--output-dir", str(output), "--steps", "3", "--episodes-per-step", "8",
                       "--learning-rate", "0.1"]
            result = subprocess.run(command, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([line.split(" - ")[0] for line in result.stdout.splitlines()],
                             ["step:1", "step:2", "step:3"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["kind"], "synthetic_optimizer_smoke")
            self.assertFalse(manifest["benchmark_score"])
            self.assertEqual(manifest["optimizer_steps"], 3)
            self.assertNotEqual(manifest["initial_parameter_sha256"], manifest["parameter_sha256"])
            records = [json.loads(path.read_text()) for path in sorted((output / "trajectories").glob("*.json"))]
            self.assertEqual(len(records), 24)
            for record in records:
                target = record["observation"].index(1.0)
                self.assertEqual(record["reward"], float(record["action"] == target))
                self.assertEqual(record["kind"], "synthetic_optimizer_smoke")
                self.assertTrue(record["messages"])
                self.assertIn(record["optimizer_step"], (1, 2, 3))
            checkpoint = torch.load(output / manifest["checkpoint"], weights_only=True)
            self.assertEqual(checkpoint["optimizer_step"], 3)
            self.assertEqual(checkpoint["parent_checkpoint"], "checkpoints/step_000002.pt")
            self.assertEqual(checkpoint["parameter_sha256"], manifest["parameter_sha256"])
            self.assertTrue(checkpoint["model_state_dict"]["weight"].abs().sum() > 0)
            self.assertTrue(all(int(item["step"]) == 3 for item in checkpoint["optimizer_state_dict"]["state"].values()))
            before = (output / "manifest.json").read_bytes()
            again = subprocess.run(command, text=True, capture_output=True, timeout=60)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual((output / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
