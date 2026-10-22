"""Offline OpenCode CLI fixtures; no model calls or global session reads."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from export_atif import export_request


class OpenCodeExportTests(unittest.TestCase):
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
        self.temporary = tempfile.TemporaryDirectory(prefix="opencode-atif-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.write("request.json", {"id": "req-open", "run": "RUN", "backend": "opencode",
                                    "phase": "review", "started_at": 1792530000.25,
                                    "outcome": "completed"})
        self.write("prompt.txt", "Review the captured training evidence.")
        self.write("response.json", {"summary": "Keep collecting evidence."})
        self.write("agent.backend.json", {"version": "1.19.3", "model": "local/flint",
                                         "session_id": "ses_this_request"})

    def write(self, name, value):
        (self.root / name).write_text(value if isinstance(value, str) else json.dumps(value))

    def event(self, kind, *, identifier=None, session="ses_this_request", timestamp=1792530000123, **values):
        event = {"type": kind, "timestamp": timestamp, "sessionID": session}
        if kind == "error":
            event["error"] = values
            return event
        part = {"id": identifier or "part_" + kind, "sessionID": session,
                "messageID": "msg_1", "type": {"tool_use": "tool", "step_start": "step-start",
                                               "step_finish": "step-finish"}.get(kind, kind)}
        part.update(values)
        event["part"] = part
        return event

    def stdout(self, events, suffix=""):
        self.write("agent.stdout", "".join(json.dumps(event) + "\n" for event in events) + suffix)

    def finish(self, **kwargs):
        return self.event("step_finish", reason="stop", **kwargs)

    def export(self, **kwargs):
        manifest = export_request(self.root, run="RUN", **kwargs)
        trajectory = json.loads((self.root / "trajectory.json").read_text())
        if self.official_model is not None:
            self.official_model.model_validate(trajectory)
        for step in trajectory["steps"]:
            calls = {call["tool_call_id"] for call in step.get("tool_calls", [])}
            for output in step.get("observation", {}).get("results", []):
                self.assertIn(output["source_call_id"], calls)
        self.assertEqual(list(range(1, len(trajectory["steps"]) + 1)),
                         [step["step_id"] for step in trajectory["steps"]])
        return manifest, trajectory

    def test_visible_text_reasoning_and_completed_tool_link_to_real_call(self):
        self.stdout([
            self.event("step_start"),
            self.event("reasoning", text="Inspect the exposed loss trend."),
            self.event("tool_use", callID="call_read", tool="read",
                       state={"status": "completed", "input": {"filePath": "training.log"},
                              "output": "step=20 loss=0.7"}),
            self.event("text", text='{"summary": "Keep collecting evidence."}'),
            self.finish(),
        ])
        manifest, trajectory = self.export()
        self.assertEqual("opencode_stdout", manifest["capture"])
        self.assertEqual("opencode_stdout_events", manifest["coverage"]["internal_tools"])
        self.assertTrue(manifest["coverage"]["partial"])
        self.assertTrue(manifest["coverage"]["lifecycle_complete"])
        self.assertEqual("ses_this_request", trajectory["extra"]["native_session_id"])
        self.assertEqual("local/flint", trajectory["agent"]["model_name"])
        self.assertEqual("1.19.3", trajectory["agent"]["version"])
        tools = [step for step in trajectory["steps"] if "tool_calls" in step]
        self.assertEqual(1, len(tools))
        self.assertEqual({"tool_call_id": "call_read", "function_name": "read",
                          "arguments": {"filePath": "training.log"}}, tools[0]["tool_calls"][0])
        self.assertEqual("step=20 loss=0.7", tools[0]["observation"]["results"][0]["content"])
        reasoning = next(step for step in trajectory["steps"] if "reasoning_content" in step)
        self.assertEqual("visible_backend_text", reasoning["extra"]["visibility"])
        self.assertTrue(all("llm_call_count" not in step for step in trajectory["steps"]))
        self.assertEqual(1, sum(step["message"] == '{"summary": "Keep collecting evidence."}'
                                for step in trajectory["steps"]))

    def test_millisecond_event_time_and_seconds_supervisor_time_stay_distinct(self):
        self.stdout([self.event("text", text="visible"), self.finish()])
        _, trajectory = self.export()
        self.assertEqual(datetime.fromtimestamp(1792530000.25, timezone.utc).isoformat(),
                         trajectory["steps"][0]["timestamp"])
        visible = next(step for step in trajectory["steps"] if step["message"] == "visible")
        self.assertEqual("2026-10-20T21:00:00.123000+00:00", visible["timestamp"])

    def test_repeated_part_snapshots_update_instead_of_duplicate_tools_or_messages(self):
        self.stdout([
            self.event("text", text="first snapshot"),
            self.event("text", text="final snapshot"),
            self.event("tool_use", callID="read", tool="read", state={"status": "running", "input": {"path": "log"}}),
            self.event("tool_use", callID="read", tool="read", state={"status": "completed", "input": {"path": "log"}, "output": "observed"}),
            self.finish(),
        ])
        _, trajectory = self.export()
        self.assertNotIn("first snapshot", json.dumps(trajectory))
        self.assertEqual(1, sum(step["message"] == "final snapshot" for step in trajectory["steps"]))
        self.assertEqual(1, sum("tool_calls" in step for step in trajectory["steps"]))

    def test_unique_finish_usage_is_reported_not_estimated_or_double_counted(self):
        first = self.event("step_finish", identifier="finish_a", reason="tool-calls",
                           tokens={"input": 100, "output": 20, "reasoning": 8,
                                   "cache": {"read": 30, "write": 10}}, cost=0.002)
        self.stdout([first, first, self.finish(identifier="finish_b", tokens={
            "input": 200, "output": 40, "reasoning": 9, "cache": {"read": 50, "write": 0}}, cost=0)])
        _, trajectory = self.export()
        metrics = trajectory["final_metrics"]
        usage = metrics["extra"]["opencode_usage"]
        self.assertEqual(300, usage["input"])
        self.assertEqual(60, usage["output"])
        self.assertEqual(17, usage["reasoning"])
        self.assertEqual(80, usage["cache_read"])
        self.assertEqual(0.002, usage["reported_cost_usd"])
        self.assertEqual(2, metrics["extra"]["observed_usage_steps"])
        self.assertEqual(390, metrics["total_prompt_tokens"])
        self.assertEqual(77, metrics["total_completion_tokens"])
        self.assertEqual(80, metrics["total_cached_tokens"])
        self.assertNotIn("total_cost_usd", metrics)
        # Metadata totals are not mixed into already-observed per-step usage.
        self.write("agent.backend.json", {"session_id": "ses_this_request", "usage": {"input": 99999}})
        _, updated = self.export()
        self.assertEqual(300, updated["final_metrics"]["extra"]["opencode_usage"]["input"])

    def test_unknown_version_and_incomplete_categories_keep_native_usage_only(self):
        for version in ("unknown", "1.19.3"):
            with self.subTest(version=version):
                self.write("agent.backend.json", {"session_id": "ses_this_request", "version": version})
                tokens = {"input": 100, "output": 20}
                if version == "unknown":
                    tokens.update(reasoning=5, cache={"read": 10, "write": 0})
                self.stdout([self.finish(tokens=tokens)])
                _, trajectory = self.export()
                self.assertNotIn("total_prompt_tokens", trajectory["final_metrics"])
                self.assertNotIn("total_completion_tokens", trajectory["final_metrics"])
                self.assertEqual(100, trajectory["final_metrics"]["extra"]["opencode_usage"]["input"])

    def test_absent_usage_cost_and_model_are_not_filled_with_defaults(self):
        (self.root / "agent.backend.json").unlink()
        self.stdout([self.finish()])
        _, trajectory = self.export()
        self.assertNotIn("model_name", trajectory["agent"])
        self.assertEqual("unknown", trajectory["agent"]["version"])
        self.assertNotIn("final_metrics", trajectory)

    def test_invalid_usage_is_only_retained_in_raw_artifact(self):
        self.stdout([self.finish(tokens={"input": True, "output": -2, "reasoning": 1.2,
                                        "cache": {"read": "10", "write": 0}}, cost=float("nan"))])
        manifest, trajectory = self.export()
        self.assertEqual({"cache_write": 0}, trajectory["final_metrics"]["extra"]["opencode_usage"])
        self.assertTrue(any("unsupported" in warning for warning in manifest["warnings"]))
        self.assertNotIn("NaN", (self.root / "trajectory.json").read_text())

    def test_failed_tools_keep_error_observation_and_session_error_remains_visible(self):
        self.stdout([
            self.event("tool_use", callID="bad", tool="read", state={"status": "error", "input": {"path": "missing"},
                                                                    "error": "No such file"}),
            self.event("error", name="APIError", data={"message": "endpoint disconnected"}),
        ])
        manifest, trajectory = self.export(status="failed")
        tool = next(step for step in trajectory["steps"] if "tool_calls" in step)
        self.assertEqual("No such file", tool["observation"]["results"][0]["content"])
        self.assertEqual("error", tool["observation"]["results"][0]["extra"]["status"])
        self.assertIn("endpoint disconnected", json.dumps(trajectory))
        self.assertFalse(manifest["coverage"]["lifecycle_complete"])
        self.assertTrue(any("reports error" in warning for warning in manifest["warnings"]))

    def test_truncated_ndjson_preserves_preceding_events_without_inventing_completion(self):
        self.stdout([self.event("text", text="visible before interruption")], suffix='{"type":"tool_use"')
        manifest, trajectory = self.export(status="interrupted")
        self.assertIn("visible before interruption", json.dumps(trajectory))
        self.assertFalse(manifest["coverage"]["lifecycle_complete"])
        self.assertTrue(any("incomplete/non-JSON" in warning for warning in manifest["warnings"]))
        self.assertTrue(any("truncated" in warning for warning in manifest["warnings"]))
        self.assertFalse(any("tool_calls" in step for step in trajectory["steps"]))

    def test_mixed_sessions_are_not_merged_even_if_metadata_selects_one(self):
        self.stdout([self.event("text", text="first session contents"),
                     self.event("text", session="ses_other", text="other session contents")])
        manifest, trajectory = self.export()
        rendered = json.dumps(trajectory)
        self.assertNotIn("first session contents", rendered)
        self.assertNotIn("other session contents", rendered)
        self.assertNotIn("native_session_id", trajectory["extra"])
        self.assertTrue(any("mixed/missing session" in warning for warning in manifest["warnings"]))
        self.assertEqual(hashlib.sha256((self.root / "agent.stdout").read_bytes()).hexdigest(),
                         manifest["artifact_sha256"]["agent.stdout"])
        self.assertTrue(manifest["coverage"]["partial"])

    def test_metadata_does_not_invent_session_identity_for_missing_stdout(self):
        self.stdout([])
        manifest, trajectory = self.export(status="failed")
        self.assertNotIn("native_session_id", trajectory["extra"])
        self.assertEqual("unavailable", manifest["coverage"]["internal_tools"])
        self.assertTrue(any("missing session" in warning for warning in manifest["warnings"]))

    def test_nested_part_identity_and_metadata_must_agree_with_envelope(self):
        for mismatch in ("part", "metadata"):
            with self.subTest(mismatch=mismatch):
                event = self.event("text", text="unbound contents")
                if mismatch == "part":
                    event["part"]["sessionID"] = "ses_other"
                else:
                    self.write("agent.backend.json", {"session_id": "ses_other"})
                self.stdout([event])
                _, trajectory = self.export()
                self.assertNotIn("unbound contents", json.dumps(trajectory))

    def test_unknown_events_and_missing_tool_ids_are_raw_only(self):
        self.stdout([
            self.event("future_event", text="unsupported payload"),
            self.event("tool_use", tool="read", state={"status": "completed", "input": {}, "output": "no ID"}),
            self.finish(),
        ])
        manifest, trajectory = self.export()
        self.assertTrue(any("future_event" in warning for warning in manifest["warnings"]))
        self.assertTrue(any("callID" in warning for warning in manifest["warnings"]))
        self.assertNotIn("unsupported payload", json.dumps(trajectory))
        self.assertFalse(any("tool_calls" in step for step in trajectory["steps"]))

    def test_no_global_session_lookup_and_reexport_are_deterministic(self):
        self.stdout([self.event("text", text="this request only"), self.finish()])
        with patch("export_atif._native", side_effect=AssertionError("must never read global sessions")):
            first, trajectory = self.export(sessions_dir=self.root / "unrelated-opencode-store")
            first_bytes = (self.root / "trajectory.json").read_bytes()
            second, _ = self.export()
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, (self.root / "trajectory.json").read_bytes())
        self.assertIn("agent.backend.json", first["artifact_sha256"])
        self.assertFalse((self.root / "native").exists())


if __name__ == "__main__":
    unittest.main()
