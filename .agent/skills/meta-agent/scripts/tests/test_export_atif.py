"""Offline artifact export tests; never invoke Codex or inspect shared credentials."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from export_atif import export_request


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official_model = None
        if os.environ.get("HARBOR_VALIDATION_WHEEL"):
            sys.path.insert(0, os.environ["HARBOR_VALIDATION_WHEEL"])
            for directory in os.environ.get("HARBOR_VALIDATION_DEPS", "").split(os.pathsep):
                if directory:
                    sys.path.insert(0, directory)
            from harbor.models.trajectories import Trajectory
            cls.official_model = Trajectory

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="atif-export-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.request = self.root / "request"
        self.request.mkdir()
        self.sessions = self.root / "sessions"
        self.sessions.mkdir()
        self.write("request.json", {"id": "req-1", "run": "RUN", "phase": "review", "step": 20,
                                    "started_at": "2026-10-21T01:00:00Z", "outcome": "completed"})
        self.write("prompt.txt", "Inspect current training log and trajectories.")
        self.write("response.json", {"phase": "review", "summary": "Observe more clean exposure."})

    def write(self, name, value):
        path = self.request / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))

    def stdout(self, events):
        self.write("agent.stdout", "".join(json.dumps(e) + "\n" for e in events))

    def native(self, name="rollout-native-1.jsonl", events=None, identity="native-1"):
        entries = [{"type": "session_meta", "payload": {"id": identity, "cli_version": "0.163.0"}},
                   {"type": "turn_context", "payload": {"model": "recorded-model"}}]
        entries.extend(events or [])
        path = self.sessions / name
        path.write_text("".join(json.dumps(e) + "\n" for e in entries))
        return path

    def export(self, **kwargs):
        result = export_request(self.request, run="RUN", **kwargs)
        if self.official_model is not None:
            self.official_model.model_validate_json((self.request / "trajectory.json").read_text())
        return result

    def trajectory(self):
        return json.loads((self.request / "trajectory.json").read_text())

    def assertReferences(self, trajectory):
        self.assertEqual(list(range(1, len(trajectory["steps"]) + 1)), [s["step_id"] for s in trajectory["steps"]])
        for step in trajectory["steps"]:
            call_ids = {call["tool_call_id"] for call in step.get("tool_calls", [])}
            for output in step.get("observation", {}).get("results", []):
                self.assertIn(output["source_call_id"], call_ids)

    def test_custom_backend_exports_prompt_response_without_invented_tools(self):
        result = self.export()
        trajectory = self.trajectory()
        self.assertEqual("artifacts_only", result["capture"])
        self.assertEqual("unavailable", result["coverage"]["internal_tools"])
        self.assertTrue(result["coverage"]["partial"])
        self.assertTrue(result["coverage"]["lifecycle_complete"])
        self.assertEqual("RUN", trajectory["session_id"])
        self.assertEqual("req-1", trajectory["trajectory_id"])
        self.assertEqual(["user", "agent"], [s["source"] for s in trajectory["steps"]])
        self.assertTrue(all("llm_call_count" not in s for s in trajectory["steps"]))

    def test_epoch_request_timestamp_and_reconciliation_lineage(self):
        request = json.loads((self.request / "request.json").read_text())
        request.update(started_at=1000.25, previous_request={"id": "prior-attempt"})
        self.write("request.json", request)
        self.export()
        trajectory = self.trajectory()
        self.assertEqual(trajectory["steps"][0]["timestamp"], "1970-01-01T00:16:40.250000+00:00")
        self.assertEqual(trajectory["extra"]["previous_request_id"], "prior-attempt")

    def test_stdout_deduplicates_started_completed_and_preserves_incomplete_call(self):
        self.stdout([
            {"type": "thread.started", "thread_id": "native-1"},
            {"type": "item.started", "item": {"id": "cmd1", "type": "command_execution", "command": "read log", "status": "in_progress"}},
            {"type": "item.completed", "item": {"id": "cmd1", "type": "command_execution", "command": "read log", "aggregated_output": "step:20", "exit_code": 0}},
            {"type": "item.started", "item": {"id": "cmd2", "type": "command_execution", "command": "read trajectories"}},
            {"type": "item.started", "item": {"id": "m1", "type": "agent_message", "text": "partial"}},
            {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "complete"}},
            {"type": "turn.completed", "usage": {"input_tokens": 500, "cached_input_tokens": 200, "output_tokens": 80}},
        ])
        result = self.export(status="failed")
        trajectory = self.trajectory()
        tools = [s for s in trajectory["steps"] if "tool_calls" in s]
        self.assertEqual(2, len(tools))
        self.assertEqual("step:20", tools[0]["observation"]["results"][0]["content"])
        self.assertNotIn("observation", tools[1])
        self.assertEqual(1, sum(s["message"] == "complete" for s in trajectory["steps"]))
        self.assertTrue(result["coverage"]["partial"])
        self.assertEqual(500, trajectory["final_metrics"]["total_prompt_tokens"])
        self.assertNotIn("total_cost_usd", trajectory["final_metrics"])
        self.assertReferences(trajectory)

    def test_native_exact_id_and_response_items_win_over_stdout_mirrors(self):
        self.stdout([
            {"type": "thread.started", "thread_id": "native-1"},
            {"type": "item.completed", "item": {"id": "stdout1", "type": "agent_message", "text": "stdout mirror"}},
        ])
        self.native(events=[
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "event mirror"}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "actual native message"}]}},
            {"type": "response_item", "payload": {"type": "function_call", "call_id": "fn1", "name": "read_file", "arguments": '{"path":"train.log"}'}},
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "fn1", "output": "step:20"}},
        ])
        # A later unrelated session must not be selected.
        self.native(name="newer-other.jsonl", identity="unrelated", events=[
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "unrelated secret"}]}}])
        result = self.export(sessions_dir=self.sessions)
        text = (self.request / "trajectory.json").read_text()
        self.assertEqual("native", result["capture"])
        self.assertIn("actual native message", text)
        self.assertNotIn("stdout mirror", text)
        self.assertNotIn("event mirror", text)
        self.assertNotIn("unrelated secret", text)
        self.assertEqual("recorded-model", self.trajectory()["agent"]["model_name"])
        self.assertReferences(self.trajectory())

    def test_encrypted_reasoning_is_filtered_and_visible_summary_only(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        source = self.native(events=[
            {"type": "response_item", "payload": {"type": "reasoning", "encrypted_content": "opaque-NEVER-export", "summary": [{"type": "summary_text", "text": "Visible summary"}]}}
        ])
        result = self.export(sessions_dir=self.sessions)
        archive = self.request / result["source"]["path"]
        for path in (archive, self.request / "trajectory.json", self.request / "export.json"):
            self.assertNotIn("opaque-NEVER-export", path.read_text())
        self.assertNotIn("encrypted_content", archive.read_text())
        self.assertIn("Visible summary", (self.request / "trajectory.json").read_text())
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), result["source"]["source_sha256"])
        self.assertNotEqual(result["source"]["source_sha256"], result["source"]["archived_sha256"])
        self.assertEqual("filtered_native_session_snapshot", result["source"]["capture"])

    def test_archived_native_wins_and_reexport_is_identical(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        source = self.native(events=[{"type": "response_item", "payload": {"type": "reasoning", "encrypted_content": "hidden", "summary": [{"type": "summary_text", "text": "snapshot"}]}}])
        first = self.export(sessions_dir=self.sessions)
        first_bytes = (self.request / "trajectory.json").read_bytes()
        source.write_text("corrupted shared session later")
        second = self.export(sessions_dir=self.sessions)
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, (self.request / "trajectory.json").read_bytes())

    def test_multiple_matching_native_sources_fail_closed(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        self.native()
        self.native(name="duplicate.jsonl")
        result = self.export(sessions_dir=self.sessions)
        self.assertNotEqual("native", result["capture"])
        self.assertTrue(any("found 2" in w for w in result["warnings"]))
        self.assertFalse((self.request / "native").exists())

    def test_mixed_thread_ids_do_not_collect_native(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"},
                     {"type": "thread.started", "thread_id": "unrelated"}])
        self.native()
        result = self.export(sessions_dir=self.sessions)
        self.assertNotEqual("native", result["capture"])
        self.assertTrue(any("multiple stdout thread" in w for w in result["warnings"]))

    def test_mixed_native_metadata_does_not_match(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        self.native(events=[{"type": "session_meta", "payload": {"id": "unrelated"}}])
        result = self.export(sessions_dir=self.sessions)
        self.assertNotEqual("native", result["capture"])

    def test_no_native_id_means_no_shared_session_collection(self):
        self.native()
        result = self.export(sessions_dir=self.sessions)
        self.assertNotEqual("native", result["capture"])
        self.assertFalse((self.request / "native").exists())

    def test_mcp_real_output_is_linked_and_incomplete_call_has_no_output(self):
        self.stdout([
            {"type": "item.started", "item": {"id": "mcp1", "type": "mcp_tool_call", "server": "local", "tool": "read", "arguments": {"path": "log"}}},
            {"type": "item.completed", "item": {"id": "mcp1", "type": "mcp_tool_call", "server": "local", "tool": "read", "result": {"content": [{"type": "text", "text": "actual output"}]}}},
            {"type": "item.started", "item": {"id": "mcp2", "type": "mcp_tool_call", "server": "local", "tool": "wait"}},
        ])
        self.export()
        tools = [s for s in self.trajectory()["steps"] if "tool_calls" in s]
        self.assertEqual(2, len(tools))
        self.assertEqual("local.read", tools[0]["tool_calls"][0]["function_name"])
        self.assertEqual("actual output", tools[0]["observation"]["results"][0]["content"])
        self.assertNotIn("observation", tools[1])
        self.assertReferences(self.trajectory())

    def test_stdout_completion_fills_missing_started_arguments(self):
        self.stdout([
            {"type": "item.started", "item": {"id": "cmd1", "type": "command_execution"}},
            {"type": "item.completed", "item": {"id": "cmd1", "type": "command_execution", "command": "read log", "aggregated_output": "done"}},
            {"type": "item.started", "item": {"id": "m1", "type": "mcp_tool_call", "server": "local", "tool": "read"}},
            {"type": "item.completed", "item": {"id": "m1", "type": "mcp_tool_call", "arguments": {"path": "train.log"}, "result": "done"}},
        ])
        self.export()
        tools = [s["tool_calls"][0] for s in self.trajectory()["steps"] if "tool_calls" in s]
        self.assertEqual({"command": "read log"}, tools[0]["arguments"])
        self.assertEqual({"path": "train.log"}, tools[1]["arguments"])
        self.assertEqual("local.read", tools[1]["function_name"])

    def test_native_custom_tool_and_orphan_output(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        self.native(events=[
            {"type": "response_item", "payload": {"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch", "input": "*** Begin Patch"}},
            {"type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": "c1", "output": "Done"}},
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "missing", "output": "unmatched"}},
        ])
        result = self.export(sessions_dir=self.sessions)
        tools = [s for s in self.trajectory()["steps"] if "tool_calls" in s]
        self.assertEqual({"input": "*** Begin Patch"}, tools[0]["tool_calls"][0]["arguments"])
        self.assertTrue(any("no matching" in w for w in result["warnings"]))
        self.assertReferences(self.trajectory())

    def test_multimodal_is_not_claimed_as_complete_text(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        self.native(events=[
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "visible"}, {"type": "image", "url": "data:image/png;base64,NOTTEXT"}]}},
            {"type": "response_item", "payload": {"type": "future_type", "data": "raw future event"}},
        ])
        result = self.export(sessions_dir=self.sessions)
        self.assertTrue(any("multimodal" in w for w in result["warnings"]))
        self.assertTrue(any("future_type" in w for w in result["warnings"]))
        self.assertNotIn("NOTTEXT", (self.request / "trajectory.json").read_text())
        self.assertIn("NOTTEXT", (self.request / result["source"]["path"]).read_text())

    def test_controller_receipt_has_deterministic_actor_and_zero_calls(self):
        self.write("operation.json", {"operation": "set_budget", "arguments": {"budget": 3}, "request_id": "req-1"})
        self.write("receipt.json", {"application": "partial", "scope": "worker-a", "uptake": "pending"})
        self.export()
        controller = self.trajectory()["steps"][-1]
        self.assertEqual("controller", controller["extra"]["actor"])
        self.assertEqual(0, controller["llm_call_count"])
        self.assertNotIn("metrics", controller)
        self.assertNotIn("reasoning_content", controller)
        self.assertIn('"partial"', controller["observation"]["results"][0]["content"])
        self.assertReferences(self.trajectory())

    def test_controller_pending_dispatch_never_invents_receipt(self):
        self.write("operation.json", {"operation": "set_budget", "request_id": "req-1"})
        result = self.export(status="interrupted")
        controller = self.trajectory()["steps"][-1]
        self.assertEqual(0, controller["llm_call_count"])
        self.assertNotIn("observation", controller)
        self.assertEqual("prepared_dispatch_unconfirmed", controller["extra"]["dispatch_status"])
        self.assertTrue(any("outcome remains unknown" in w for w in result["warnings"]))

    def test_controller_attempt_requires_process_evidence(self):
        self.write("operation.json", {"operation": "set_budget"})
        self.write("request.json", {"id": "req-1", "run": "RUN", "processes": {"executor": {"started_at": 3, "outcome": "timed_out"}}, "outcome": "failed"})
        self.export()
        self.assertEqual("attempt_recorded", self.trajectory()["steps"][-1]["extra"]["dispatch_status"])

    def test_controller_receipt_without_operation_is_state_without_tool_call(self):
        self.write("receipt.json", {"application": "pending", "attempt": "not sent: configured end boundary reached"})
        self.export()
        controller = self.trajectory()["steps"][-1]
        self.assertEqual("system", controller["source"])
        self.assertNotIn("tool_calls", controller)
        self.assertIn("not sent", controller["message"])
        self.assertEqual("no_operation_recorded", controller["extra"]["dispatch_status"])

    def test_controller_stdout_receipt_is_explicitly_unvalidated(self):
        self.write("operation.json", {"operation": "set_budget", "request_id": "req-1"})
        self.write("executor.stdout", {"application": "fully_applied", "uptake": "executor claim"})
        result = self.export(status="failed")
        observation = self.trajectory()["steps"][-1]["observation"]["results"][0]
        self.assertFalse(observation["extra"]["supervisor_validated"])
        self.assertTrue(any("not confirmed" in w for w in result["warnings"]))

    def test_controller_validated_receipt_wins_over_stdout(self):
        self.write("operation.json", {"operation": "set_budget"})
        self.write("receipt.json", {"application": "partial"})
        self.write("executor.stdout", {"application": "fully_applied"})
        self.export()
        observation = self.trajectory()["steps"][-1]["observation"]["results"][0]
        self.assertIn('"partial"', observation["content"])
        self.assertTrue(observation["extra"]["supervisor_validated"])

    def test_malformed_partial_stdout_does_not_hide_available_trajectory(self):
        self.write("agent.stdout", '{"type":"item.completed","item":{"type":"agent_message","id":"a","text":"available"}}\n{"type":')
        result = self.export(status="failed")
        self.assertTrue(any("incomplete" in w for w in result["warnings"]))
        self.assertIn("available", (self.request / "trajectory.json").read_text())

    def test_native_unfinished_tool_has_partial_coverage_even_completed_request(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        self.native(events=[{"type": "response_item", "payload": {"type": "function_call", "call_id": "p1", "name": "read", "arguments": "{}"}}])
        result = self.export(sessions_dir=self.sessions)
        self.assertTrue(result["coverage"]["partial"])
        self.assertTrue(result["coverage"]["lifecycle_complete"])
        self.assertTrue(any("no captured output" in w for w in result["warnings"]))

    def test_native_unsupported_type_has_partial_coverage(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        self.native(events=[{"type": "response_item", "payload": {"type": "future_api_type", "opaque": "value"}}])
        result = self.export(sessions_dir=self.sessions)
        self.assertTrue(result["coverage"]["partial"])
        self.assertEqual("completed", result["status"])

    def test_native_partial_line_snapshot_reports_loss_and_reexports_stably(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        source = self.native()
        with source.open("a") as handle:
            handle.write('{"type":"response_item","payload":')
        first = self.export(sessions_dir=self.sessions, status="incomplete")
        second = self.export(sessions_dir=self.sessions, status="incomplete")
        self.assertEqual(first, second)
        self.assertTrue(any("omitted from filtered snapshot" in w for w in first["warnings"]))

    def test_artifact_hashes_match_and_do_not_include_credentials(self):
        (self.root / "auth.json").write_text("DO-NOT-COPY")
        (self.sessions / "config.toml").write_text("DO-NOT-COPY")
        result = self.export(sessions_dir=self.sessions)
        for name, digest in result["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((self.request / name).read_bytes()).hexdigest(), digest)
        self.assertNotIn("DO-NOT-COPY", (self.request / "trajectory.json").read_text())
        self.assertFalse((self.request / "auth.json").exists())

    def test_symlink_artifact_and_output_directory_are_rejected(self):
        external = self.root / "external"
        external.mkdir()
        (external / "secret").write_text("outside")
        (self.request / "agent.stdout").symlink_to(external / "secret")
        with self.assertRaises(ValueError):
            self.export()
        (self.request / "agent.stdout").unlink()
        (self.request / "native").symlink_to(external, target_is_directory=True)
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        self.native()
        with self.assertRaises(ValueError):
            self.export(sessions_dir=self.sessions)
        self.assertEqual(["secret"], [p.name for p in external.iterdir()])

    def test_native_symlink_is_not_collected(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        actual = self.native()
        relocated = self.root / "outside.jsonl"
        actual.rename(relocated)
        actual.symlink_to(relocated)
        result = self.export(sessions_dir=self.sessions)
        self.assertNotEqual("native", result["capture"])

    def test_run_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            export_request(self.request, run="OTHER")

    def test_structured_response_not_duplicated_when_native_final_message_matches(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        response = json.loads((self.request / "response.json").read_text())
        self.native(events=[{"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": json.dumps(response)}]}}])
        self.export(sessions_dir=self.sessions)
        messages = [s for s in self.trajectory()["steps"] if s["source"] == "agent"]
        self.assertEqual(1, len(messages))

    def test_prompt_not_duplicated_by_native_user_message(self):
        self.stdout([{"type": "thread.started", "thread_id": "native-1"}])
        prompt = (self.request / "prompt.txt").read_text()
        self.native(events=[{"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}}])
        self.export(sessions_dir=self.sessions)
        self.assertEqual(1, sum(s["message"] == prompt for s in self.trajectory()["steps"]))

    def test_cli_exports_without_harbor_dependency(self):
        result = subprocess.run([sys.executable, str(SCRIPTS / "export_atif.py"), "--request-dir", str(self.request), "--run", "RUN"], capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("req-1", json.loads(result.stdout)["request_id"])


if __name__ == "__main__":
    unittest.main()
