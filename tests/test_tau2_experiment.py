"""CPU contract tests; optional native integration uses TAU2_TRAIN_WORKSPACE."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gyms/Tau2Gym"), str(ROOT / "UserRL/gyms")]
from examples.tau2.evaluate import summarize
from examples.tau2.recipe import train_command
from meta_agent.tau2_runtime import initialize, pin_update, WEIGHTS


class ExperimentTests(unittest.TestCase):
    def test_pass_estimators_use_without_replacement_trials(self):
        rows = [dict(domain="retail", task_id=str(task), trial=trial, reward=int(trial < successes))
                for task, successes in enumerate((0, 2, 4)) for trial in range(4)]
        result = summarize(rows)
        self.assertEqual(result["success_rate"], .5)
        self.assertEqual(result["pass_at_k"]["4"], 2 / 3)
        self.assertEqual(result["pass_power_k"]["4"], 1 / 3)
        self.assertAlmostEqual(result["pass_power_k"]["2"], 7 / 18)
        with self.assertRaises(ValueError):
            summarize(rows[:-1])
        with self.assertRaises(ValueError):
            summarize(rows + rows[:1])

    def test_online_budget_and_frozen_eval_contract(self):
        cmd = train_command("python", Path("/code"), Path("/runs/r1"), Path("/models/policy"), Path("/train"), 43)
        for item in ("algorithm.adv_estimator=info_grpo", "algorithm.use_intrinsic_reward=true",
                     "data.train_batch_size=16", "actor_rollout_ref.rollout.n=5", "trainer.total_training_steps=22",
                     "trainer.resume_mode=disable", "trainer.test_freq=-1", "trainer.val_before_train=false",
                     "+data.apply_chat_template_kwargs.enable_thinking=false"):
            self.assertIn(item, cmd)
        self.assertIn("data.val_files=/train/data/train.parquet", cmd)
        self.assertFalse(any("test.parquet" in arg for arg in cmd))

    def test_container_mounts_only_selected_training_inputs_and_fresh_output(self):
        from examples.tau2.container import docker_command
        argv = docker_command(image="tau2:local", workspace="/training-only", policy="/weights", output="/fresh-run", devices="2,3")
        mounts = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--mount"]
        self.assertEqual(len(mounts), 3)
        self.assertTrue(mounts[0].endswith("dst=/training,readonly"))
        self.assertTrue(mounts[1].endswith("dst=/policy,readonly"))
        self.assertTrue(mounts[2].endswith("dst=/runs"))
        self.assertFalse(any(value == "--pid" or value.startswith(("--pid=", "--privileged")) for value in argv))
        self.assertNotIn("/var/run/docker.sock", " ".join(argv))

    def test_fresh_runs_inherit_prose_only_and_native_reward_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = root / "r1", root / "r2"
            initialize(a)
            policy = a / "drafts/user/env_policy.json"
            policy.write_text('{"tau2_flow":{"flow_probability":0.5}}')
            (a / "experience.md").write_text("A prior training observation, not executable state.")
            initialize(b, native_reward=True, experience=a / "experience.md")
            self.assertEqual(json.loads((b / "drafts/user/env_policy.json").read_text())["tau2_flow"]["flow_probability"], 0)
            self.assertEqual(json.loads((b / "drafts/environment/environment_config.json").read_text())["process_reward"], {})
            self.assertEqual((a / "experience.md").read_text(), (b / "experience.md").read_text())
            self.assertFalse((b / "checkpoints").exists())

    def test_invalid_draft_cannot_replace_active_bundles(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "r1"
            initialize(run)
            original = (run / "live/environment_bundle.json").read_bytes()
            (run / "drafts/environment/environment.py").write_text("this is not Python !!")
            retained = pin_update(run, step=1)
            self.assertEqual(retained["version"], 0)
            self.assertEqual((run / "live/environment_bundle.json").read_bytes(), original)
            self.assertTrue((run / "rejected_drafts.jsonl").is_file())

    def test_update_pinning_keeps_existing_factory_and_records_new_version(self):
        from tau2gym.env.environment_runtime import load_environment_bundle
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "r1"
            initialize(run)
            old_factory, settings, _ = load_environment_bundle(run / "live/environment_bundle.json", run.name)
            before = old_factory(settings)
            envfile = run / "drafts/environment/environment.py"
            envfile.write_text(envfile.read_text().replace("self.settings = settings", "self.settings = settings; self.changed = True"))
            # A draft write has no effect until the next update boundary.
            still_old, _, _ = load_environment_bundle(run / "live/environment_bundle.json", run.name)
            self.assertIs(old_factory, still_old)
            result = pin_update(run, step=1)
            new_factory, settings, _ = load_environment_bundle(run / "live/environment_bundle.json", run.name)
            self.assertEqual(result["version"], 1)
            self.assertFalse(hasattr(before, "changed"))
            self.assertTrue(new_factory(settings).changed)

    def test_training_hooks_dispatch_editor_and_pin_its_actual_change(self):
        from meta_agent import training_audit as hooks
        editor = ("import json,sys;from pathlib import Path;"
                  "r=json.loads(Path(sys.argv[1]).read_text());"
                  "p=Path(r['editable_roots'][0])/'agent/entropy_schedule.json';"
                  "t=p.with_suffix('.pending');t.write_text(json.dumps({'enable':True,'value':0.004}));"
                  "t.replace(p);print('Set entropy to 0.004 based on training evidence.')")
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "hook_r1"
            initialize(run)
            environment = dict(EAU_TAU2_RUN=str(run), EAU_ASYNC_AUDIT_RUN=run.name,
                EAU_ASYNC_AUDIT_COMMAND=json.dumps([sys.executable, "-c", editor, "{request}"]),
                EAU_ASYNC_AUDIT_DIR=str(run / "audits"), EAU_AUDIT_CONTEXT_PATH=str(run / "audit_context.json"),
                EAU_ASYNC_AUDIT_DIRECT_ROOTS=json.dumps([str(run / "drafts")]),
                EAU_ASYNC_AUDIT_EVERY="4", EAU_ASYNC_AUDIT_STOP="22", EAU_ASYNC_AUDIT_FIRST="1")
            previous = hooks._coordinator
            hooks._coordinator = None
            try:
                with patch.dict(os.environ, environment):
                    for step in range(1, 11):
                        hooks.before_training_step(step)
                        hooks.audit_training_step(step, {"train/loss": 1 / step})
                        time.sleep(.04)
                    hooks.before_training_step(22, is_final=True)
                    value = json.loads((run / "active/entropy_schedule.json").read_text())
                    self.assertEqual(value, dict(enable=True, value=.004))
                    hooks.audit_training_step(22, {"train/loss": .01}, is_final=True)
                    self.assertTrue((run / "audits/direct_edits.jsonl").is_file())
                    self.assertFalse(list((run / "audits").rglob("response.json")))
                    hooks.close_training_audit()
            finally:
                if hooks._coordinator is not None:
                    hooks._coordinator.close(timeout=0)
                hooks._coordinator = previous


@unittest.skipUnless(os.environ.get("TAU2_TRAIN_WORKSPACE"), "prepare a train workspace to enable native integration")
class NativeIntegrationTests(unittest.TestCase):
    def test_split_manifest_real_environment_tool_and_pinned_rewards(self):
        from examples.tau2.prepare import verify
        workspace = Path(os.environ["TAU2_TRAIN_WORKSPACE"]).resolve(strict=True)
        manifest = verify(workspace, split="train")
        self.assertEqual(sum(manifest["counts"].values()), 178)
        sys.path.insert(0, str(workspace / "benchmark/src"))
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "native_r1"
            initialize(run)
            environment = dict(TAU2_DATA_DIR=str(workspace / "benchmark/data"),
                EAU_STEP_REVIEW_RUN=run.name, EAU_USER_SIM_MODE="editable", EAU_USER_SIM_BOOTSTRAP="coser_scratch_v1",
                EAU_USER_SIM_BUNDLE=str(run / "live/usersim_bundle.json"),
                EAU_ENVIRONMENT_BUNDLE=str(run / "live/environment_bundle.json"),
                EAU_COSER_USER_MODEL="openai/coser-usersim", EAU_COSER_USER_LLM_ARGS='{"api_base":"http://127.0.0.1:1/v1","api_key":"EMPTY"}',
                EAU_TAU2_REWARD_MODE="process", TAU2_EVAL_TYPE="action", LOGURU_LEVEL="ERROR",
                LITELLM_LOCAL_MODEL_COST_MAP="True", PYTHON_DOTENV_DISABLED="1")
            with patch.dict(os.environ, environment):
                from tau2.data_model.message import AssistantMessage
                from tau2gym import Tau2Env, Tau2GymConfig
                config = Tau2GymConfig(domain="retail", task_split="train", data_mode="single", data_source="0",
                                       tau2_data_dir=environment["TAU2_DATA_DIR"], max_steps=50)
                with patch("tau2.user.user_simulator.generate", return_value=AssistantMessage(role="assistant", content="I need help.")):
                    env = Tau2Env(config=config)
                    try:
                        observation, _ = env.reset()
                        self.assertIn("I need help.", str(observation))
                        _, reward, _, _, _ = env.step(json.dumps(dict(name="calculate", arguments={"expression": "2+3"})))
                        self.assertAlmostEqual(reward, WEIGHTS["first_tool_ok"])
                        self.assertEqual(env._environment_metadata["environment_bundle_version"], 0)
                        self.assertEqual(env._episode_process_weights, WEIGHTS)
                    finally:
                        env.close()
        # Import caches/bytecode must not be confused with source/data changes.
        verify(workspace, split="train")


if __name__ == "__main__":
    unittest.main()
