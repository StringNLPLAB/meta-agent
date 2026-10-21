"""Verify trace export at the controller boundary without calling a real model."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_supervise import AGENT, EXECUTOR

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from export_run import export_run, write_index
from supervise import Supervisor, parser


class TraceIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="trace-integration-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "agent.py").write_text(AGENT)
        (self.root / "executor.py").write_text(EXECUTOR)
        (self.root / "agent-settings.json").write_text("{}")
        (self.root / "executor-settings.json").write_text("{}")
        (self.root / "control.json").write_text('{"version":1}')
        contract = dict(objective="complete training tasks", budget="20 agent calls",
                        guardrails=["preserve useful clarification"], allowed_feedback=["training traces"],
                        excluded_feedback=["final evaluation"], evaluation_protocol="fixed reporting-only evaluator",
                        checkpoint_selection="final checkpoint", allowed_operations=["set_repeat_budget"])
        (self.root / "contract.json").write_text(json.dumps(contract))
        self.base = ["--root", str(self.root), "--run", "RUN", "--log", "train.log", "--trajectories", "trajectories",
                     "--contract", "contract.json", "--every-steps", "20", "--max-calls", "20", "--once",
                     "--agent-command", json.dumps([sys.executable, str(self.root / "agent.py")])]
        self.run_dir = self.root / "reports/supervision/RUN"
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for name in ("EAU_ROOT", "EAU_GYM_ROOT", "EAU_FIRST_REVIEW", "EAU_REVIEW_INTERVAL", "EAU_MANDATE_INTERVAL"):
            self.env.pop(name, None)

    def call(self, *extra, expected=0):
        result = subprocess.run([sys.executable, str(SCRIPTS / "supervise.py"), *self.base, *extra],
                                env=self.env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def settings(self, **values):
        (self.root / "agent-settings.json").write_text(json.dumps(values))

    def step(self, value):
        with (self.root / "train.log").open("a") as handle:
            handle.write(f"step:{value} - global_seqlen:100\n")

    def records(self):
        paths = (self.run_dir / "requests").glob("*/request.json")
        return sorted(((p.parent, json.loads(p.read_text())) for p in paths), key=lambda item: item[1]["started_at"])

    def state(self):
        return json.loads((self.run_dir / "state.json").read_text())

    def test_optional_export_records_the_meta_agent_and_keeps_training_input_separate(self):
        self.call("--export-atif")
        directory, request = self.records()[0]
        trace = json.loads((directory / "trajectory.json").read_text())
        manifest = json.loads((directory / "export.json").read_text())
        self.assertEqual(trace["trajectory_id"], request["id"])
        self.assertEqual(trace["session_id"], "RUN")
        self.assertEqual(trace["steps"][0]["source"], "user")
        self.assertIsInstance(trace["steps"][0]["timestamp"], str)
        self.assertIn("# Meta-Agent", trace["steps"][0]["message"])
        self.assertEqual(manifest["capture"], "artifacts_only")
        self.assertEqual(request["outcome"], "completed")
        self.assertEqual(request["processes"]["agent"]["returncode"], 0)
        self.assertGreaterEqual(request["finished_at"], request["started_at"])
        index = json.loads((self.run_dir / "trajectories.json").read_text())
        self.assertEqual(index["requests"][0]["outcome"], "completed")
        self.assertFalse((self.root / "trajectories").exists())

    def test_tracing_can_be_enabled_without_adopting_settings_or_repeating_discovery(self):
        self.call()
        self.assertFalse((self.records()[0][0] / "trajectory.json").exists())
        self.step(20)
        self.call("--export-atif")
        records = self.records()
        self.assertEqual([record[1]["phase"] for record in records], ["discovery", "review"])
        self.assertTrue((records[-1][0] / "trajectory.json").exists())
        self.assertFalse(self.state().get("binding_history"))

    def test_failed_agent_exports_partial_and_retry_has_a_distinct_request(self):
        self.settings(fail=True)
        self.call("--export-atif", expected=2)
        directory, request = self.records()[0]
        self.assertEqual(request["outcome"], "failed")
        self.assertEqual(request["processes"]["agent"]["returncode"], 7)
        self.assertTrue(json.loads((directory / "export.json").read_text())["coverage"]["partial"])
        self.settings()
        self.call("--export-atif")
        self.assertEqual([r[1]["outcome"] for r in self.records()], ["failed", "completed"])
        self.assertEqual(len(json.loads((self.run_dir / "trajectories.json").read_text())["requests"]), 2)

    def test_timeout_keeps_trace_and_process_outcome(self):
        self.settings(sleep=5)
        self.call("--export-atif", "--agent-timeout", "0.2", expected=2)
        directory, request = self.records()[0]
        self.assertEqual(request["processes"]["agent"]["outcome"], "timed_out")
        self.assertEqual(request["outcome"], "failed")
        self.assertTrue((directory / "trajectory.json").exists())
        self.assertEqual(self.state()["status"], "needs_reconciliation")

    def test_sigterm_exports_an_interrupted_request(self):
        self.settings(sleep=5)
        command = [sys.executable, str(SCRIPTS / "supervise.py"), *self.base, "--export-atif"]
        process = subprocess.Popen(command, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        deadline = time.monotonic() + 4
        while not (self.root / "agent-pid.txt").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue((self.root / "agent-pid.txt").exists())
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=8), 130)
        directory, request = self.records()[0]
        self.assertEqual(request["outcome"], "interrupted")
        self.assertEqual(json.loads((directory / "export.json").read_text())["status"], "interrupted")
        self.assertFalse(self.state()["supervisor"]["running"])

    def test_native_session_is_collected_from_explicit_source_for_exact_thread(self):
        self.settings(file_response=True)
        prefix = 'import json\nprint(json.dumps({"type":"thread.started","thread_id":"trace-thread"}),flush=True)\n'
        (self.root / "agent.py").write_text(prefix + AGENT)
        sessions = self.root / "sessions"
        sessions.mkdir()
        events = [{"type": "session_meta", "payload": {"id": "trace-thread", "cli_version": "fixture"}},
                  {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": "actual native observation"}]}}]
        (sessions / "rollout-trace-thread.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
        self.call("--export-atif", "--codex-sessions", str(sessions))
        directory, _ = self.records()[0]
        manifest = json.loads((directory / "export.json").read_text())
        self.assertEqual(manifest["capture"], "native")
        self.assertEqual(manifest["source"]["native_session_id"], "trace-thread")
        self.assertIn("actual native observation", (directory / "trajectory.json").read_text())

    def test_export_failure_does_not_repeat_a_successful_gate_operation(self):
        self.settings(apply=True)
        args = parser().parse_args([*self.base, "--export-atif", "--mode", "apply", "--executor-command",
                                   json.dumps([sys.executable, str(self.root / "executor.py")])])
        supervisor = Supervisor(args)
        supervisor.load()
        supervisor.tick()
        self.step(20)
        with patch("export_atif.export_request", side_effect=OSError("fixture export disk failure")):
            supervisor.tick()
        self.assertEqual(supervisor.state["status"], "watching")
        self.assertIsNone(supervisor.state["active_request"])
        self.assertEqual(json.loads((self.root / "control.json").read_text()), {"version": 2})
        supervisor.tick()
        calls = (self.root / "executor-calls.jsonl").read_text().splitlines()
        self.assertEqual(len(calls), 1)
        directory, request = self.records()[-1]
        self.assertEqual(request["outcome"], "completed")
        self.assertTrue((directory / "receipt.json").exists())

    def test_backfill_reads_saved_records_without_repeating_agent_or_mutating_state(self):
        self.call()
        state_path = self.run_dir / "state.json"
        before = state_path.read_bytes()
        calls_before = (self.root / "agent-calls.jsonl").read_bytes()
        directory, request = self.records()[0]
        request.pop("outcome")
        request.pop("finished_at")
        (directory / "request.json").write_text(json.dumps(request))
        result = export_run(self.run_dir)
        self.assertEqual(result["exported"], 1)
        self.assertFalse(result["errors"])
        self.assertEqual(state_path.read_bytes(), before)
        self.assertEqual((self.root / "agent-calls.jsonl").read_bytes(), calls_before)
        index = json.loads((self.run_dir / "trajectories.json").read_text())
        self.assertEqual(index["requests"][0]["outcome"], "completed")
        self.assertIsNotNone(index["requests"][0]["finished_at"])

    def test_backfill_cli_builds_a_viewer_job_without_agent_calls(self):
        self.call()
        calls = (self.root / "agent-calls.jsonl").read_bytes()
        output = self.root / "viewer-jobs/meta-agent"
        result = subprocess.run([sys.executable, str(SCRIPTS / "export_run.py"), "--run-dir", str(self.run_dir),
                                 "--harbor-out", str(output)], env=self.env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["exported"], 1)
        bundle = json.loads((output / "meta-agent-bundle.json").read_text())
        self.assertEqual(bundle["run"], "RUN")
        trial = output / bundle["trials"][0]["trial_name"]
        self.assertTrue((trial / "agent/trajectory.json").is_file())
        self.assertIsNone(json.loads((trial / "result.json").read_text())["verifier_result"])
        self.assertEqual((self.root / "agent-calls.jsonl").read_bytes(), calls)

    def test_rejected_gate_receipt_stays_an_unvalidated_claim_in_trace(self):
        self.settings(apply=True)
        (self.root / "executor-settings.json").write_text('{"contradictory_exit":true}')
        apply = ["--export-atif", "--mode", "apply", "--executor-command",
                 json.dumps([sys.executable, str(self.root / "executor.py")])]
        self.call(*apply)
        self.step(20)
        self.call(*apply, expected=2)
        directory, _ = self.records()[-1]
        trace = json.loads((directory / "trajectory.json").read_text())
        gate = next(step for step in trace["steps"] if step.get("extra", {}).get("actor") == "controller")
        observation = gate["observation"]["results"][0]
        self.assertFalse(observation["extra"]["supervisor_validated"])
        self.assertFalse(observation["extra"]["runtime_verified"])
        self.assertEqual(self.state()["status"], "needs_reconciliation")
        self.settings()
        self.call(*apply)
        index = json.loads((self.run_dir / "trajectories.json").read_text())
        self.assertEqual(index["requests"][-1]["previous_request_id"], index["requests"][-2]["request_id"])
        self.assertEqual(len((self.root / "executor-calls.jsonl").read_text().splitlines()), 1)

    def test_backfill_rejects_a_live_runner_lock(self):
        self.call()
        with (self.run_dir / "runner.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "supervisor owns"):
                export_run(self.run_dir)
        self.assertFalse((self.records()[0][0] / "trajectory.json").exists())

    def test_index_keeps_unfinished_requests_incomplete(self):
        self.settings(fail=True)
        self.call(expected=2)
        directory, request = self.records()[0]
        request.pop("outcome")
        (directory / "request.json").write_text(json.dumps(request))
        state = self.state()
        state.update(active_request=None, status="watching")
        (self.run_dir / "state.json").write_text(json.dumps(state))
        index = write_index(self.run_dir)
        self.assertEqual(index["requests"][0]["outcome"], "incomplete")


if __name__ == "__main__":
    unittest.main()
