"""Verify the viewer bundle on isolated recorded requests; no harness is run."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from harbor_bundle import MARKER, write_bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="harbor-bundle-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "source" / "RUN"
        self.output = self.root / "jobs" / "meta-agent-RUN"
        self.run.mkdir(parents=True)
        self.save(self.run / "state.json", {"run": "RUN", "status": "watching"})

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def request(self, name="req-1", outcome="completed", phase="review", step=20):
        request = self.run / "requests" / name
        request.mkdir(parents=True)
        self.save(request / "request.json", {"id": name, "phase": phase, "step": step,
                  "outcome": outcome, "started_at": 1000, "finished_at": 1100 if outcome != "incomplete" else None})
        self.save(request / "export.json", {"status": outcome, "capture": "native", "coverage": {"partial": outcome != "completed"}})
        self.save(request / "trajectory.json", {"schema_version": "ATIF-v1.7", "session_id": name,
                  "agent": {"name": "codex", "version": "0.163.0", "model_name": "fixture-model"},
                  "steps": [{"step_id": 1, "source": "user", "message": "inspect recorded fixture"},
                            {"step_id": 2, "source": "agent", "message": "observe"}],
                  "final_metrics": {"total_prompt_tokens": 30, "total_completion_tokens": 4}})
        (request / "agent.stdout").write_text("visible tool output\n")
        return request

    def trial(self, manifest, index=0):
        return self.output / manifest["trials"][index]["trial_name"]

    def test_completed_pass_becomes_viewable_trial_without_rewards(self):
        request = self.request()
        self.save(request / "receipt.json", {"application": "pending", "uptake": "unconfirmed"})
        manifest = write_bundle(self.run, self.output)
        trial = self.trial(manifest)
        result = json.loads((trial / "result.json").read_text())
        job = json.loads((self.output / "result.json").read_text())
        self.assertIsNone(result["verifier_result"])
        self.assertIsNone(result["exception_info"])
        self.assertTrue(result["config"]["verifier"]["disable"])
        self.assertTrue(result["agent_result"]["metadata"]["request_pass_completed"])
        self.assertEqual(job["stats"]["n_completed_trials"], 1)
        self.assertEqual(job["stats"]["n_input_tokens"], 30)
        self.assertIsNone(job["finished_at"])
        self.assertEqual((trial / "agent/trajectory.json").read_bytes(), (request / "trajectory.json").read_bytes())
        self.assertEqual((trial / "agent/command-0/stdout.txt").read_text(), "visible tool output\n")
        self.assertTrue((trial / "artifacts/receipt.json").is_file())
        self.assertIn("not a runnable Harbor", (self.output / "README.txt").read_text())

    def test_failed_and_partial_remain_distinct(self):
        failed = self.request("failed", "failed")
        data = json.loads((failed / "request.json").read_text())
        data["error"] = "agent exited 7"
        self.save(failed / "request.json", data)
        self.request("partial", "incomplete")
        manifest = write_bundle(self.run, self.output)
        results = [json.loads((self.trial(manifest, i) / "result.json").read_text()) for i in range(2)]
        self.assertEqual(results[0]["exception_info"]["exception_message"], "agent exited 7")
        self.assertIsNone(results[1]["exception_info"])
        self.assertIsNone(results[1]["finished_at"])
        self.assertFalse(results[1]["agent_result"]["metadata"]["request_pass_completed"])
        stats = json.loads((self.output / "result.json").read_text())["stats"]
        self.assertEqual((stats["n_completed_trials"], stats["n_errored_trials"], stats["n_pending_trials"]), (1, 1, 1))

    def test_reexport_updates_owned_files_and_preserves_unrelated_file(self):
        request = self.request()
        first = write_bundle(self.run, self.output)
        (self.output / "user-note.txt").write_text("keep this")
        (request / "agent.stdout").write_text("updated trace")
        second = write_bundle(self.run, self.output)
        self.assertEqual(first["trials"][0]["trial_name"], second["trials"][0]["trial_name"])
        self.assertEqual((self.trial(second) / "agent/command-0/stdout.txt").read_text(), "updated trace")
        self.assertEqual((self.output / "user-note.txt").read_text(), "keep this")

    def test_modified_generated_file_is_never_overwritten(self):
        self.request()
        manifest = write_bundle(self.run, self.output)
        target = self.trial(manifest) / "agent/trajectory.json"
        target.write_text("user edit")
        with self.assertRaisesRegex(ValueError, "modified generated file"):
            write_bundle(self.run, self.output)
        self.assertEqual(target.read_text(), "user edit")

    def test_refuses_unowned_destination_and_different_source(self):
        self.request()
        self.output.mkdir(parents=True)
        (self.output / "private.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "Required source file"):
            write_bundle(self.run, self.output)
        (self.output / "private.txt").unlink()
        write_bundle(self.run, self.output)
        other = self.root / "other-run"
        other.mkdir()
        self.save(other / "state.json", {"run": "OTHER"})
        with self.assertRaisesRegex(ValueError, "different source run"):
            write_bundle(other, self.output)

    def test_refuses_overlapping_or_symbolic_link_destinations(self):
        self.request()
        for output in (self.run, self.run / "bundle", self.run.parent):
            with self.assertRaisesRegex(ValueError, "outside the source"):
                write_bundle(self.run, output)
        link = self.root / "linked"
        link.symlink_to(self.output.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symbolic links"):
            write_bundle(self.run, link / "bundle")

    def test_only_whitelisted_request_artifacts_are_copied(self):
        request = self.request()
        (request / "auth.json").write_text("private auth")
        (request / "training-dump.json").write_text("private training")
        private = self.root / "private-secret"
        private.write_text("never copy")
        (request / "prompt.txt").symlink_to(private)
        (request / "executor.stderr").symlink_to(private)
        manifest = write_bundle(self.run, self.output)
        trial = self.trial(manifest)
        self.assertFalse((trial / "artifacts/auth.json").exists())
        self.assertFalse((trial / "artifacts/training-dump.json").exists())
        self.assertFalse((trial / "artifacts/prompt.txt").exists())
        self.assertEqual(len(manifest["warnings"]), 2)

    def test_native_manifest_path_is_restricted_to_exact_archive_filename(self):
        request = self.request()
        digest = "a" * 64
        relative = f"native/sessions/{digest}.jsonl"
        archive = request / relative
        archive.parent.mkdir(parents=True)
        archive.write_text('{"type":"session_meta"}\n')
        export = json.loads((request / "export.json").read_text())
        export["source"] = {"path": relative}
        self.save(request / "export.json", export)
        manifest = write_bundle(self.run, self.output)
        self.assertEqual((self.trial(manifest) / "artifacts" / relative).read_bytes(), archive.read_bytes())
        export["source"]["path"] = "../../secret"
        self.save(request / "export.json", export)
        manifest = write_bundle(self.run, self.output)
        self.assertFalse((self.trial(manifest) / "artifacts" / relative).exists())

    def test_old_request_uses_run_index_completion_without_mutating_source(self):
        request = self.request()
        original = {"id": "req-1", "phase": "discovery", "step": 0, "started_at": 1000, "stage": "prepared"}
        self.save(request / "request.json", original)
        self.save(self.run / "trajectories.json", {"requests": [{"request_id": "req-1", "outcome": "completed", "finished_at": 1100}]})
        manifest = write_bundle(self.run, self.output)
        result = json.loads((self.trial(manifest) / "result.json").read_text())
        self.assertTrue(result["agent_result"]["metadata"]["request_pass_completed"])
        self.assertEqual(json.loads((request / "request.json").read_text()), original)
        self.assertIsNotNone(result["finished_at"])

    def test_missing_trajectory_is_an_explicit_partial_artifact_trial(self):
        request = self.request(outcome="incomplete")
        (request / "trajectory.json").unlink()
        manifest = write_bundle(self.run, self.output)
        self.assertFalse(manifest["trials"][0]["trajectory_available"])
        self.assertTrue((self.trial(manifest) / "result.json").exists())
        self.assertFalse((self.trial(manifest) / "agent/trajectory.json").exists())

    def test_empty_run_is_a_zero_trial_snapshot(self):
        manifest = write_bundle(self.run, self.output)
        result = json.loads((self.output / "result.json").read_text())
        self.assertEqual(manifest["trials"], [])
        self.assertEqual(result["n_total_trials"], 0)
        self.assertEqual(result["stats"]["n_completed_trials"], 0)
        self.assertIsNone(result["finished_at"])

    @unittest.skipUnless(os.environ.get("HARBOR_VALIDATION_WHEEL"), "optional pinned Harbor model validation")
    def test_official_models_and_scanner_load_bundle(self):
        wheel = os.environ["HARBOR_VALIDATION_WHEEL"]
        sys.path.insert(0, wheel)
        for directory in os.environ.get("HARBOR_VALIDATION_DEPS", "").split(os.pathsep):
            if directory:
                sys.path.insert(0, directory)
        from harbor.models.job.config import JobConfig
        from harbor.models.job.result import JobResult
        from harbor.models.trial.config import TrialConfig
        from harbor.models.trial.result import TrialResult
        from harbor.models.trajectories import Trajectory
        from harbor.viewer.scanner import JobScanner
        self.request()
        self.request("partial", "incomplete")
        self.request("failed", "failed")
        manifest = write_bundle(self.run, self.output)
        JobConfig.model_validate_json((self.output / "config.json").read_text())
        JobResult.model_validate_json((self.output / "result.json").read_text())
        scanner = JobScanner(self.output.parent)
        self.assertEqual(scanner.list_jobs(), [self.output.name])
        self.assertIsNotNone(scanner.get_job_config(self.output.name))
        self.assertIsNotNone(scanner.get_job_result(self.output.name))
        self.assertEqual(len(scanner.list_trials(self.output.name)), 3)
        for trial in manifest["trials"]:
            directory = self.output / trial["trial_name"]
            TrialConfig.model_validate_json((directory / "config.json").read_text())
            TrialResult.model_validate_json((directory / "result.json").read_text())
            Trajectory.model_validate_json((directory / "agent/trajectory.json").read_text())
            self.assertIsNotNone(scanner.get_trial_result(self.output.name, trial["trial_name"]))


if __name__ == "__main__":
    unittest.main()
