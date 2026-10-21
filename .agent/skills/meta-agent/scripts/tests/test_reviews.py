"""Offline adapter regression tests. All experiment files live in TemporaryDirectory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
HEADER = ("| run | step | tier | knob | change | evidence | prediction | verification |\n"
          "|---|---|---|---|---|---|---|---|\n")


class ReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="meta-agent-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "reports").mkdir()
        (self.root / "logs").mkdir()
        self.log = self.root / "logs/train_fixture_20261021_120000.log"
        self.log.write_text("step:20 - global_seqlen:100 - actor/entropy:0.5\n")
        self.table = self.root / "reports/change_log.md"
        self.table.write_text(HEADER)
        (self.root / "reports/run_aliases.json").write_text(json.dumps({
            "_README": "Offline fixture", "fixture20261021120000": "RUN"}))
        self.env = dict(os.environ, EAU_ROOT=str(self.root),
                        EAU_GYM_ROOT=str(self.root / "gyms"),
                        EAU_REVIEW_INTERVAL="20", PYTHONDONTWRITEBYTECODE="1")
        for key in ("EAU_MANDATE_INTERVAL", "EAU_FIRST_REVIEW"):
            self.env.pop(key, None)

    def run_script(self, name, *args, env=None):
        return subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)],
                              cwd=self.root, env=env or self.env, text=True,
                              capture_output=True, timeout=15)

    def python(self, code):
        return subprocess.run([sys.executable, "-c", code], cwd=SCRIPTS,
                              env=self.env, text=True, capture_output=True, timeout=15)

    def row(self, *, run="RUN", step=20, decision="observe", change=None,
            evidence="trajectory: sample 37 repeats one known fact", prediction="—",
            verification="H1 pending", knob="disclosure"):
        if change is None:
            change = (f"decision: {decision}; reason: H1 observation is incomplete; "
                      "window: clean steps 21-60, 200 admitted episodes; next: step 60 or 200 episodes")
        cells = [run, str(step), "user", knob, change, evidence, prediction, verification]
        with self.table.open("a") as handle:
            handle.write("| " + " | ".join(cells) + " |\n")

    def apply_row(self, status, **overrides):
        values = dict(
            decision="apply",
            change=("decision: apply; target: useful task interaction; old: v1; new: v2; "
                    "mechanism: charge redundant replies; reason: repeated facts; "
                    "next: 200 clean episodes or task regression"),
            prediction="repeat rate below baseline 0.4; window: clean steps 21-60 and 200 episodes",
            verification=(f"application: {status}; attempt: gate request id fixture-1 recorded; "
                          "recovery: restore old control, retain learned lineage"))
        values.update(overrides)
        self.row(**values)

    def check(self, step=20, *extra):
        return self.run_script("check_mandate.py", "--step", step, "--run", "RUN",
                               "--log", self.log, "--json", *extra)

    def pending(self, *extra):
        return self.run_script("pending_checkpoints.py", "--log", self.log, "--run", "RUN", *extra)

    def assert_exit(self, result, code):
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)

    def test_observe_without_version_or_new_prediction(self):
        self.row()
        result = self.check()
        self.assert_exit(result, 0)
        record = json.loads(result.stdout)
        self.assertEqual(record["configs_bumped"], [])
        self.assertFalse(record["runtime_verified"])
        self.assert_exit(self.pending(), 0)

    def test_instrument_defer_restart_and_noop(self):
        variants = {
            "instrument": "question: which fact type repeats; method: unbiased fact-state samples",
            "defer": "limitation: missing resident reader; plan: add stateful repeat counter next run",
            "restart_plan": ("limitation: Python is resident; plan: add validated repeat counter; "
                             "recovery: resume compatible checkpoint with code revision"),
            "leave_unchanged": "",
        }
        for decision, fields in variants.items():
            with self.subTest(decision=decision):
                self.table.write_text(HEADER)
                self.row(change=f"decision: {decision}; reason: current evidence needs investigation; "
                         f"next: new fact-state samples; {fields}")
                self.assert_exit(self.check(), 0)

    def test_incomplete_observe_and_instrument_fail(self):
        for decision in ("observe", "instrument"):
            with self.subTest(decision=decision):
                self.table.write_text(HEADER)
                self.row(change=f"decision: {decision}; reason: insufficient evidence; next: step 40")
                self.assert_exit(self.check(), 1)
                self.assert_exit(self.pending(), 1)

    def test_apply_pending_partial_failed_and_reverted_are_complete_reviews(self):
        for status in ("pending", "partial", "failed", "control_reverted"):
            with self.subTest(status=status):
                self.table.write_text(HEADER)
                self.apply_row(status)
                self.assert_exit(self.check(), 0)
                self.assert_exit(self.pending(), 0)

    def test_false_fully_applied_fails_even_with_version_bump(self):
        (self.root / "entropy_schedule.json").write_text('{"version": 8}')
        self.apply_row("fully_applied")
        result = self.check()
        self.assert_exit(result, 1)
        self.assertIn("missing field: uptake", result.stdout)
        self.assert_exit(self.pending(), 1)

    def test_scope_requires_all_acknowledgments(self):
        for confirmed in ("worker0", "worker0,worker1", "worker0,worker1,worker2"):
            with self.subTest(confirmed=confirmed):
                self.table.write_text(HEADER)
                verification = ("application: fully_applied; attempt: gate request fixture-1; "
                                "recovery: revert control only; boundary: episode 21; "
                                f"scope: worker0,worker1; confirmed: {confirmed}; "
                                "uptake: per-worker consumed v2 hash 123 logged for episode 21")
                self.apply_row("fully_applied", verification=verification)
                result = self.check()
                self.assert_exit(result, 0 if confirmed == "worker0,worker1" else 1)
                if result.returncode == 0:
                    self.assertFalse(json.loads(result.stdout)["runtime_verified"])

    def test_unknown_uptake_is_not_fully_applied(self):
        self.apply_row("fully_applied", verification=(
            "application: fully_applied; attempt: gate write; recovery: control only; "
            "boundary: step 21; scope: worker0; confirmed: worker0; uptake: unknown"))
        result = self.check()
        self.assert_exit(result, 1)
        self.assertIn("cannot use uptake", result.stdout)

    def test_first_check_does_not_excuse_missing_review(self):
        self.assert_exit(self.check(), 1)

    def test_baseline_does_not_complete_a_review(self):
        result = self.check(20, "--baseline")
        self.assert_exit(result, 0)
        self.assertTrue(json.loads(result.stdout)["baseline_only"])
        self.assert_exit(self.pending(), 1)
        self.assert_exit(self.check(), 1)

    def test_recent_restart_log_does_not_replace_review_evidence(self):
        self.apply_row("fully_applied", evidence=f"trajectory: unknown; {self.log.relative_to(self.root)}")
        self.assert_exit(self.check(), 1)

    def test_review_requires_reason_and_next_condition(self):
        self.row(change="decision: leave_unchanged")
        result = self.check()
        self.assert_exit(result, 1)
        self.assertIn("missing field: reason", result.stdout)
        self.assertIn("missing field: next", result.stdout)

    def test_valid_row_cannot_hide_invalid_application(self):
        self.row()
        self.apply_row("fully_applied")
        self.assert_exit(self.check(), 1)
        self.assert_exit(self.pending(), 1)

    def test_wrong_run_cannot_satisfy_current_review(self):
        self.row(run="OTHER")
        self.assert_exit(self.check(), 1)
        self.assert_exit(self.pending(), 1)
        result = self.run_script("check_mandate.py", "--step", 20, "--log", self.log, "--run", "OTHER")
        self.assert_exit(result, 2)
        self.assertIn("does not match", result.stderr)

    def test_aliases_apply_to_reviews_and_acknowledged_misses(self):
        self.row(run="fixture20261021120000")
        self.assert_exit(self.check(), 0)
        self.assert_exit(self.pending(), 0)
        self.table.write_text(HEADER)
        self.row(knob="missed checkpoint", prediction="—", evidence="session interruption, no review")
        result = self.pending()
        self.assert_exit(result, 0)
        self.assertIn("acknowledged missed reviews: 20", result.stdout)
        self.assert_exit(self.check(), 1)

    def test_missed_marker_cannot_hide_invalid_review(self):
        self.row(knob="missed", prediction="—")
        self.apply_row("fully_applied")
        self.assert_exit(self.pending(), 1)

    def test_gap_markers_require_valid_structure_and_explicit_gap_meaning(self):
        self.row(knob="missed checkpoint", evidence="interrupted | malformed cell", prediction="—")
        self.assert_exit(self.pending(), 1)
        self.table.write_text(HEADER)
        self.row(knob="missed opportunity", change="unchanged", prediction="—")
        self.assert_exit(self.pending(), 1)

    def test_gap_cannot_be_replaced_by_a_backfilled_valid_review(self):
        self.row(knob="missed", prediction="—")
        self.row()
        result = self.pending()
        self.assert_exit(result, 1)
        self.assertIn("do not backfill", result.stdout)

    def test_duplicate_fields_and_invalid_decisions_fail(self):
        for change in (
            "decision: observe; decision: leave_unchanged; reason: no evidence; next: step 40",
            "decision: adjust_any_knob; reason: review due; next: step 40",
        ):
            with self.subTest(change=change):
                self.table.write_text(HEADER)
                self.row(change=change)
                self.assert_exit(self.check(), 1)

    def test_another_log_cannot_supply_a_resume_anchor(self):
        other = self.root / "logs/train_other_20261021_130000.log"
        other.write_text("Setting global step to 225\nstep:260 - global_seqlen:100\n")
        self.assert_exit(self.pending(), 1)
        result = self.run_script("pending_checkpoints.py")
        self.assert_exit(result, 2)
        self.assertIn("Multiple matching", result.stderr)
        result = self.run_script("pending_checkpoints.py", "--run", "RUN")
        self.assert_exit(result, 1)
        self.assertIn("step 20", result.stdout)

    def test_resume_skips_only_steps_not_executed_in_selected_run(self):
        self.log.write_text("Setting global step to 225\nstep:260 - global_seqlen:100\n")
        self.row(step=240)
        result = self.pending()
        self.assert_exit(result, 1)
        self.assertIn("incomplete review steps: 260", result.stdout)
        self.assertNotIn("incomplete review steps: 20", result.stdout)

    def test_schedule_has_no_silent_default_and_accepts_legacy_spelling(self):
        env = dict(self.env)
        env.pop("EAU_REVIEW_INTERVAL")
        result = self.run_script("pending_checkpoints.py", "--log", self.log, env=env)
        self.assert_exit(result, 2)
        env["EAU_MANDATE_INTERVAL"] = "20"
        self.row()
        self.assert_exit(self.run_script("pending_checkpoints.py", "--log", self.log, env=env), 0)
        env["EAU_REVIEW_INTERVAL"] = "7"
        self.assert_exit(self.run_script("pending_checkpoints.py", "--log", self.log, env=env), 2)

    def test_custom_cadence_and_first_boundary_match_watcher(self):
        self.log.write_text("".join(f"step:{step} - global_seqlen:100 - actor/entropy:0.5\n"
                                    for step in (9, 16, 23, 30)))
        for step in (9, 16, 23, 30):
            self.row(step=step)
        self.assert_exit(self.pending("--interval", 7, "--first-review", 9), 0)
        result = self.run_script("run_watch.py", "--log", self.log, "--run", "RUN", "--interval", 7,
                                 "--first-review", 9, "--replay")
        self.assert_exit(result, 0)
        self.assertEqual(result.stdout.count("complete (record only"), 4, result.stdout)
        self.assertFalse((self.root / "reports/.mandate_state.json").exists())

    def test_replay_uses_bundled_checker_and_never_writes_snapshots(self):
        self.row()
        (self.root / "meta_agent").mkdir()
        (self.root / "meta_agent/check_mandate.py").write_text("raise RuntimeError('wrong checker imported')")
        state = self.root / "reports/.mandate_state.json"
        state.write_text('{"runs": {"OTHER": [{"step": 99, "versions": {}}]}}')
        original = state.read_bytes()
        result = self.run_script("run_watch.py", "--log", self.log, "--run", "RUN", "--replay")
        self.assert_exit(result, 0)
        self.assertIn("[REVIEW] RUN step 20: complete", result.stdout)
        self.assertEqual(state.read_bytes(), original)

    def test_replay_and_pending_share_selected_resume_anchor(self):
        self.log.write_text("Setting global step to 225\n" + "".join(
            f"step:{step} - global_seqlen:100\n" for step in (220, 240, 260)))
        self.row(step=240)
        self.row(step=260)
        result = self.run_script("run_watch.py", "--log", self.log, "--run", "RUN", "--replay")
        self.assert_exit(result, 0)
        self.assertEqual(result.stdout.count("complete (record only"), 2, result.stdout)
        self.assertNotIn("[REVIEW] RUN step 220", result.stdout)
        self.assert_exit(self.pending(), 0)

    def test_watchdog_forwards_selection_and_exits_on_declared_finish(self):
        result = subprocess.run(
            ["bash", str(SCRIPTS / "checkpoint_watchdog.sh"), "--log", str(self.log),
             "--run", "RUN", "--interval", "7", "--first-review", "6", "--finished"],
            env=self.env, cwd=self.root, text=True, capture_output=True, timeout=10)
        self.assert_exit(result, 0)
        record = (self.root / "reports/checkpoint_debt.log").read_text()
        self.assertIn("RUN at step 20", record)
        self.assertIn("review interval=7", record)
        self.assertIn("incomplete review steps: 6, 13, 20", record)
        self.assertIn("TEARDOWN", record)

    def test_same_step_recheck_and_legacy_state_are_preserved(self):
        state = self.root / "reports/.mandate_state.json"
        old = {"last_checkpoint_step": 500, "versions": {"entropy_schedule": 12}, "extra": "retain"}
        state.write_text(json.dumps(old))
        self.row()
        self.assert_exit(self.check(), 0)
        self.row(step=40)
        self.assert_exit(self.check(40), 0)
        result = self.check(40)
        self.assert_exit(result, 0)
        self.assertEqual(json.loads(result.stdout)["previous_checkpoint"], 20)
        saved = json.loads(state.read_text())
        self.assertEqual(saved["extra"], "retain")
        self.assertEqual(saved["runs"]["_legacy"][0]["step"], 500)
        self.assertEqual([s["step"] for s in saved["runs"]["RUN"]], [20, 40])

    def test_legacy_history_is_labeled_without_modification(self):
        self.row(change="observe unchanged", prediction="original prediction at registered window")
        before = self.table.read_bytes()
        result = self.check()
        self.assert_exit(result, 0)
        self.assertTrue(json.loads(result.stdout)["legacy_rows"])
        self.assertFalse(json.loads(result.stdout)["runtime_verified"])
        self.assertEqual(self.table.read_bytes(), before)

    def test_alias_state_snapshots_do_not_accumulate_duplicates(self):
        state = self.root / "reports/.mandate_state.json"
        snapshot = {"step": 20, "versions": {}}
        state.write_text(json.dumps({"runs": {
            "fixture20261021120000": [snapshot], "RUN": [snapshot]}}))
        self.row(step=40)
        for _ in range(2):
            self.assert_exit(self.check(40), 0)
        saved = json.loads(state.read_text())
        self.assertEqual([s["step"] for s in saved["runs"]["RUN"]], [20, 40])

    def test_escaped_pipes_parse_but_unescaped_extra_columns_fail(self):
        self.row(evidence=r"trajectory: reply A \| B; population collection pending")
        self.assert_exit(self.check(), 0)
        self.assert_exit(self.run_script("check_table.py"), 0)
        self.table.write_text(HEADER)
        self.row(evidence="trajectory: reply A | B")
        self.assert_exit(self.check(), 1)
        self.assert_exit(self.pending(), 1)
        self.assert_exit(self.run_script("check_table.py"), 1)

    def test_append_checks_custom_path_and_allows_no_prediction(self):
        custom = self.root / "reports/custom.md"
        custom.write_text(HEADER.replace("|---|", "| --- |"))
        code = ("from append_row import append_row\n"
                f"append_row(run='RUN', step=20, tier='user', knob='disclosure', path={str(custom)!r}, "
                "decision='observe', change='reason: H1 pending; window: steps 21-60; next: step 60 or A | B', "
                "evidence='trajectory: A | B', prediction='—')\n"
                "from check_mandate import _review_rows, review_fields\n"
                f"record = _review_rows('RUN', {str(custom)!r})[0]\n"
                "assert review_fields(record)[0]['next'] == 'step 60 or A | B'\n")
        result = self.python(code)
        self.assert_exit(result, 0)
        self.assertIn("A &#124; B", custom.read_text())
        self.assertEqual(self.table.read_text(), HEADER)
        self.assert_exit(self.run_script("check_table.py", custom), 0)

    def test_invalid_aliases_fail_instead_of_crossing_runs(self):
        path = self.root / "reports/run_aliases.json"
        for aliases in ({"fixture20261021120000": []}, {"RUN": "X", "X": "RUN"}):
            with self.subTest(aliases=aliases):
                path.write_text(json.dumps(aliases))
                self.assert_exit(self.pending(), 2)
                original = self.table.read_bytes()
                result = self.python(
                    "from append_row import append_row\n"
                    "append_row(run='RUN', step=20, tier='user', knob='observe', change='unchanged', "
                    "evidence='unavailable: fixture', prediction='—')")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.table.read_bytes(), original)

    def test_failed_atomic_append_keeps_existing_history(self):
        original = self.table.read_bytes()
        result = self.python(
            "from unittest.mock import patch\n"
            "from append_row import append_row\n"
            "with patch('append_row.os.replace', side_effect=OSError('fixture failure')):\n"
            "    try:\n"
            "        append_row(run='RUN', step=20, tier='user', knob='observe', change='unchanged', "
            "evidence='trajectory: fixture', prediction='—')\n"
            "    except OSError:\n"
            "        pass\n")
        self.assert_exit(result, 0)
        self.assertEqual(self.table.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in self.table.parent.iterdir()), ['change_log.md', 'run_aliases.json'])

    def test_finish_is_explicit_without_generic_process_detection(self):
        result = self.pending()
        self.assert_exit(result, 1)
        self.assertIn("liveness unknown", result.stdout)
        self.assertNotIn("TEARDOWN", result.stdout)
        result = self.pending("--finished")
        self.assert_exit(result, 0)
        self.assertIn("TEARDOWN", result.stdout)

    def test_manifest_selects_identity_and_labels_unobserved_process_environment(self):
        external = self.root / "external-gyms/symbolic_layer"
        external.mkdir(parents=True)
        (external / "env_policy.json").write_text('{"version": 3}')
        env = dict(self.env, EAU_GYM_ROOT=str(external.parent))
        result = self.run_script("run_manifest.py", "capture", "--run", "RUN", "--log", self.log, env=env)
        self.assert_exit(result, 0)
        record = json.loads((self.root / "reports/manifests/RUN.json").read_text())
        self.assertEqual(record["hot_json"]["env_policy"], {"version": 3})
        self.assertIsNone(record["training_pid"])
        self.assertEqual(record["env"], {})
        self.assertIn("unobserved", record["env_source"])


if __name__ == "__main__":
    unittest.main()
