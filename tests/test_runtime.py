"""Synthetic lifecycle tests: no model, tau benchmark, GPU, or system service."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime/run_experiment.py"
sys.path.insert(0, str(REPO))
from runtime import run_experiment

AGENT = r'''
import json, os, pathlib, re, sys, time
prompt = sys.stdin.read()
context = json.loads(re.search(r'<context>\s*(.*?)\s*</context>', prompt, re.S).group(1))
root = pathlib.Path(context['training_root'])
settings = json.loads((root/'fixture-settings.json').read_text())
phase = context['request']['phase']
samples = [pathlib.Path(p).read_text() for p in context['recent_trajectory_files']]
with (root/'calls.jsonl').open('a') as out:
    out.write(json.dumps({'phase': phase, 'time': time.time(), 'samples': samples,
                          'training_started': (root/'trainer-started.json').exists()})+'\n')
if phase == settings.get('fail_phase'):
    sys.exit(7)
if phase == settings.get('sleep_phase'):
    (root/'agent-pid').write_text(str(os.getpid()))
    time.sleep(30)
response = dict(phase=phase, summary='Synthetic lifecycle review',
    intervention_map='| Component | Mechanism | Applicability | Runtime | Evidence | Value | Recovery | Decision |\n| sampling | inspect predictions | synthetic fixture | readonly | current sample | observe | none | observe |\n',
    sources=dict(paper=['EAU versioned mechanism guide reused from setup'], experience=['unavailable: synthetic fixture'],
                 code=['fixture-trainer.py'], logs=[context['training_log']],
                 trajectories=samples or ['unavailable: trainer has not produced samples']),
    review=dict(decision='leave_unchanged', tier='instrumentation', knob='sampling',
        change='reason: inspect synthetic samples; next: next cadence or training completion',
        evidence='trajectory: '+('sample observed' if samples else 'unavailable before training'),
        prediction='—', verification='pending observation, no application'),
    action=dict(operation='', arguments_json='{}'), pending_predictions=[], experience_update='',
    next_condition='next cadence or completion', continue_supervision=phase != settings.get('handoff_phase'))
print(json.dumps(response))
'''

TRAINER = r'''
import json, os, pathlib, sys, time
root = pathlib.Path.cwd()
settings = json.loads((root/'fixture-settings.json').read_text())
calls = [json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
assert calls and calls[0]['phase'] == 'discovery' and not calls[0]['training_started']
(root/'trainer-started.json').write_text(json.dumps({'time': time.time(), 'pid': os.getpid()}))
samples = root/'samples'
samples.mkdir(exist_ok=True)
for step in range(settings.get('steps', 14)):
    (samples/f'step-{step}.json').write_text(json.dumps({'synthetic_sample': True, 'step': step}))
    print(f'step:{step} - synthetic loss:{1/(step+1)}', flush=True)
    time.sleep(settings.get('delay', .08))
sys.exit(settings.get('trainer_code', 0))
'''

DELAYED_WRAPPER = r'''
# Synthetic wrapper exercising the actual completion-before-exit ordering.
import json, pathlib, subprocess, sys, time
directory = pathlib.Path(sys.argv[1])
config = json.loads((directory/'launch.json').read_text())
started = time.time()
process = subprocess.Popen(config['trainer_command'])
code = process.wait()
receipt = dict(version=1, run=config['run'], returncode=code, trainer_pid=process.pid,
               started_at=started, finished_at=time.time(), stop_signal=None)
temporary = directory/'completion.tmp'
temporary.write_text(json.dumps(receipt))
temporary.replace(directory/'completion.json')
time.sleep(.7)
supervision = pathlib.Path(config['root'])/'reports/supervision'/config['run']/'state.json'
assert json.loads(supervision.read_text())['status'] == 'completed'
(directory/'wrapper-finished').write_text('normal cleanup after final review')
sys.exit(0 if code == 0 else 2)
'''


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="meta-runtime-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "fixture-agent.py").write_text(AGENT)
        (self.root / "fixture-trainer.py").write_text(TRAINER)
        self.settings()
        self.directory = self.root / "reports/runtime/SYNTHETIC"
        self.config = {"version": 1, "root": str(self.root), "run": "SYNTHETIC",
            "trainer_command": [sys.executable, str(self.root / "fixture-trainer.py")],
            "trainer_timeout_seconds": 15, "poll_seconds": .03, "final_timeout_seconds": 5,
            "contract": {"objective": "Synthetic lifecycle verification", "budget": "14 synthetic steps and 20 calls",
                "guardrails": ["No real model or benchmark"], "allowed_feedback": ["synthetic training samples"],
                "excluded_feedback": ["held-out evaluation"], "evaluation_protocol": "none in this synthetic fixture",
                "checkpoint_selection": "no model checkpoint"},
            "supervisor_args": ["--log", "reports/runtime/SYNTHETIC/trainer.stdout", "--trajectories", "samples",
                "--every-seconds", ".15", "--poll-seconds", ".03", "--max-calls", "20", "--export-atif",
                "--agent-command", json.dumps([sys.executable, str(self.root / "fixture-agent.py")])]}
        self.config_path = self.root / "launch.json"
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def settings(self, **settings):
        (self.root / "fixture-settings.json").write_text(json.dumps(settings))

    def command(self, *extra):
        self.config_path.write_text(json.dumps(self.config))
        return [sys.executable, str(RUNTIME), "--config", str(self.config_path), *extra]

    def call(self, *extra, expect=0):
        result = subprocess.run(self.command(*extra), env=self.env, capture_output=True, text=True, timeout=25)
        self.assertEqual(result.returncode, expect, result.stdout + result.stderr)
        return result

    def state(self):
        return json.loads((self.directory / "state.json").read_text())

    def calls(self):
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def wait(self, condition, timeout=15):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = condition()
            if value:
                return value
            time.sleep(.03)
        self.fail("Timed out waiting for synthetic runtime lifecycle event")

    def start(self):
        process = subprocess.Popen(self.command(), env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop, process)
        return process

    def stop(self, process):
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=15)
        else:
            process.communicate()

    def test_discovery_precedes_trainer_periodic_samples_and_final_export(self):
        self.call()
        state = self.state()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["trainer_returncode"], 0)
        self.assertFalse(state["owner"]["running"])
        self.assertTrue(all(not child["running"] for child in state["children"].values()))
        calls = self.calls()
        self.assertEqual(calls[0]["phase"], "discovery")
        self.assertFalse(calls[0]["training_started"])
        reviews = [call for call in calls if call["phase"] == "review"]
        self.assertTrue(reviews)
        self.assertTrue(all(call["samples"] for call in reviews))
        self.assertEqual(calls[-1]["phase"], "final")
        self.assertTrue(all("synthetic_sample" in sample for call in reviews for sample in call["samples"]))
        exports = list((self.root / "reports/supervision/SYNTHETIC/requests").glob("*/trajectory.json"))
        self.assertEqual(len(exports), len(calls))
        self.assertTrue((self.root / "reports/supervision/SYNTHETIC/trajectories.json").is_file())
        self.assertEqual(json.loads((self.directory / "completion.json").read_text())["returncode"], 0)

    def test_discovery_failure_never_starts_trainer(self):
        self.settings(fail_phase="discovery")
        self.call(expect=2)
        self.assertEqual(self.state()["status"], "discovery_failed")
        self.assertFalse((self.root / "trainer-started.json").exists())
        self.assertNotIn("trainer", self.state()["children"])

    def test_discovery_handoff_never_starts_trainer(self):
        self.settings(handoff_phase="discovery")
        self.call(expect=2)
        self.assertEqual(self.state()["status"], "discovery_failed")
        self.assertEqual(self.state()["supervision_status"], "agent_handoff")
        self.assertFalse((self.root / "trainer-started.json").exists())

    def test_failed_trainer_gets_final_but_stays_failed(self):
        self.settings(steps=4, trainer_code=17)
        self.call(expect=2)
        state = self.state()
        self.assertEqual(state["status"], "trainer_failed")
        self.assertEqual(state["trainer_returncode"], 17)
        self.assertEqual(state["supervision_status"], "completed")
        self.assertEqual(self.calls()[-1]["phase"], "final")

    def test_budget_exhaustion_stops_owned_trainer(self):
        self.settings(steps=300)
        self.config["supervisor_args"][self.config["supervisor_args"].index("--max-calls") + 1] = "2"
        self.call(expect=2)
        state = self.state()
        self.assertEqual(state["status"], "supervision_budget_exhausted")
        self.assertNotEqual(state["trainer_returncode"], 0)
        self.assertEqual(len(self.calls()), 2)

    def test_agent_handoff_is_not_experiment_completion(self):
        self.settings(steps=300, handoff_phase="review")
        self.call(expect=2)
        self.assertEqual(self.state()["status"], "supervision_agent_handoff")
        self.assertNotEqual(self.state()["trainer_returncode"], 0)

    def test_supervision_failure_stops_trainer_without_retry(self):
        self.settings(steps=300, fail_phase="review")
        self.call(expect=2)
        self.assertEqual(self.state()["status"], "supervision_needs_reconciliation")
        self.assertEqual(len(self.calls()), 2)
        self.assertNotEqual(self.state()["trainer_returncode"], 0)

    def test_trainer_timeout_preserves_timeout_and_allows_final(self):
        self.settings(steps=300)
        self.config["trainer_timeout_seconds"] = .4
        self.call(expect=2)
        self.assertEqual(self.state()["status"], "trainer_timeout")
        self.assertNotEqual(self.state()["trainer_returncode"], 0)
        self.assertEqual(self.calls()[-1]["phase"], "final")

    def test_final_timeout_is_not_completion(self):
        self.settings(steps=2, sleep_phase="final")
        self.config["final_timeout_seconds"] = .25
        self.call(expect=2)
        self.assertEqual(self.state()["status"], "final_timeout")
        self.assertFalse(self.state()["owner"]["running"])

    def test_existing_run_log_and_supervision_are_rejected(self):
        log = self.root / "existing.log"
        log.write_text("old unrelated log\n")
        self.config["supervisor_args"][1] = str(log)
        self.call(expect=2)
        self.assertFalse(self.directory.exists())
        self.assertEqual(log.read_text(), "old unrelated log\n")
        log.unlink()
        state_dir = self.root / "reports/supervision/SYNTHETIC"
        state_dir.mkdir(parents=True)
        self.call(expect=2)
        self.assertFalse(self.directory.exists())

    def test_internal_log_collisions_are_rejected_before_runtime_writes(self):
        for log in ("reports/runtime/SYNTHETIC/contract.json", "reports/runtime/SYNTHETIC/state.json",
                    "reports/supervision/SYNTHETIC/state.json"):
            with self.subTest(log=log):
                self.config["supervisor_args"][1] = log
                self.call(expect=2)
                self.assertFalse(self.directory.exists())

    def test_initialization_write_failure_releases_owner_without_launching(self):
        self.command()
        config, directory, parsed, flags = run_experiment.normalized_config(self.config_path)
        runtime = run_experiment.Runtime(config, directory, parsed, flags)
        original = run_experiment.atomic_json
        def fail_launch(path, value):
            if Path(path).name == "launch.json":
                raise OSError("synthetic initialization failure")
            return original(path, value)
        with mock.patch.object(run_experiment, "atomic_json", side_effect=fail_launch):
            self.assertEqual(runtime.run(), 2)
        state = self.state()
        self.assertEqual(state["status"], "runtime_failed")
        self.assertFalse(state["owner"]["running"])
        self.assertFalse(state["children"])
        self.assertFalse(run_experiment.run_lock_held(directory))
        self.assertFalse((self.root / "trainer-started.json").exists())

    def test_final_can_finish_before_trainer_wrapper_exits(self):
        self.settings(steps=1, delay=.03)
        self.command()
        config, directory, parsed, flags = run_experiment.normalized_config(self.config_path)
        runtime = run_experiment.Runtime(config, directory, parsed, flags)
        wrapper = self.root / "delayed-wrapper.py"
        wrapper.write_text(DELAYED_WRAPPER)
        original = runtime.spawn
        def spawn(name, command):
            if name == "trainer":
                command = [sys.executable, str(wrapper), str(directory)]
            return original(name, command)
        with mock.patch.object(runtime, "spawn", side_effect=spawn):
            self.assertEqual(runtime.run(), 0)
        self.assertEqual(self.state()["status"], "completed")
        self.assertEqual(self.state()["children"]["trainer"]["returncode"], 0)
        self.assertTrue((directory / "wrapper-finished").exists())

    def test_stop_step_without_actual_trainer_completion_is_still_failure(self):
        self.settings(steps=300)
        self.config["supervisor_args"].extend(["--stop-step", "1"])
        self.call(expect=2)
        self.assertEqual(self.state()["status"], "supervision_completed")
        self.assertIn("before the owned trainer", self.state()["error"])
        self.assertNotEqual(self.state()["trainer_returncode"], 0)

    def test_relative_root_is_relative_to_config_file_for_detached_run(self):
        self.settings(steps=3)
        self.config["root"] = "."
        result = self.call("--detach")
        state = self.wait(lambda: self.state() if not self.state()["owner"]["running"] else None)
        self.assertEqual(state["status"], "completed", result.stdout + result.stderr)
        launch = json.loads((self.directory / "launch.json").read_text())
        self.assertEqual(launch["root"], str(self.root))

    def test_duplicate_launch_rejected_while_owner_runs(self):
        self.settings(steps=300)
        process = self.start()
        self.wait(lambda: (self.root / "trainer-started.json").exists())
        second = self.call(expect=2)
        self.assertIn("already exists", second.stderr)
        self.assertIsNone(process.poll())

    def test_sigterm_cleans_owned_trainer_and_active_agent_only(self):
        self.settings(steps=300, sleep_phase="review")
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(self.stop, unrelated)
        process = self.start()
        agent_file = self.root / "agent-pid"
        self.wait(agent_file.exists)
        agent_pid = int(agent_file.read_text())
        process.terminate()
        process.communicate(timeout=15)
        self.assertEqual(process.returncode, 130)
        state = self.state()
        self.assertEqual(state["status"], "interrupted")
        self.assertFalse(state["owner"]["running"])
        self.assertTrue(all(not child["running"] for child in state["children"].values()))
        self.assertIsNone(unrelated.poll())
        # Exited children can briefly remain zombies until the host reaps them.
        def inactive():
            path = Path(f"/proc/{agent_pid}/stat")
            return not path.exists() or path.read_text().rsplit(") ", 1)[1].startswith("Z ")
        self.wait(inactive)

    def test_detached_owner_continues_after_launcher_returns(self):
        self.settings(steps=22)
        result = self.call("--detach")
        handshake = json.loads(result.stdout)
        self.assertEqual(handshake["status"], "started")
        self.assertEqual(handshake["owner"]["mode"], "detached")
        owner_pid = handshake["owner"]["pid"]
        def stop_detached():
            if (self.directory / "state.json").exists() and self.state()["owner"]["running"]:
                try:
                    os.kill(owner_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self.wait(lambda: not self.state()["owner"]["running"])
        self.addCleanup(stop_detached)
        state = self.wait(lambda: self.state() if not self.state()["owner"]["running"] else None)
        self.assertEqual(state["status"], "completed")
        self.assertTrue(any(call["phase"] == "review" for call in self.calls()))
        self.assertEqual(self.calls()[-1]["phase"], "final")
        self.assertTrue((self.directory / "runtime.stdout").exists())
        self.assertTrue((self.directory / "runtime.stderr").exists())

    def test_owned_flags_and_abbreviations_are_rejected(self):
        for flag in ("--root", "--run=OTHER", "--contract=other.json", "--completion-file=done",
                     "--once", "--adopt-settings", "--roo=other", "--help"):
            with self.subTest(flag=flag):
                original = list(self.config["supervisor_args"])
                self.config["supervisor_args"].append(flag)
                self.call(expect=2)
                self.assertFalse(self.directory.exists())
                self.config["supervisor_args"] = original

    def test_configuration_rejects_unbounded_or_unknown_behavior(self):
        cases = [("trainer_timeout_seconds", 0), ("trainer_timeout_seconds", float("inf")),
                 ("trainer_timeout_seconds", True), ("poll_seconds", 61), ("version", True),
                 ("failure_policy", "keep_training"), ("trainer_command", "python fixture.py"),
                 ("unknown_option", "ignored?")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                original = self.config.copy()
                self.config[key] = value
                self.call(expect=2)
                self.assertFalse(self.directory.exists())
                self.config = original


if __name__ == "__main__":
    unittest.main()
