"""Offline checks of the actual ColBench port, boundaries and live consumers."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gyms/ColBenchGym"), str(ROOT / "UserRL/gyms")]
from colbenchgym import ColBenchCodeEnv, get_code_config
from examples.colbench.container import docker_command
from examples.colbench.recipe import train_command
from examples.colbench.train import compose, environment
from examples.data_preprocess.colbench_code_multiturn import preprocess
from meta_agent.colbench_runtime import initialize, pin_update

TASK = dict(problem_description="Implement f with the user's hidden constraints.",
            ground_truth="def f(x):\n    return x + 7\n", test_cases={"test2": "f(-1)", "test1": "f(1)"})


def flow_draft(run, answer="A selected training fact."):
    (run / "drafts/user/flow_user.py").write_text(
        "class FlowUser:\n    def __init__(self, code): self.knobs={}\n"
        f"    def reply(self, question): return {answer!r}\n")
    path = run / "drafts/user/env_policy.json"
    policy = json.loads(path.read_text())
    policy["user_strategy_mix"]["colbench"] = 1.0
    path.write_text(json.dumps(policy))


class ColBenchTests(unittest.TestCase):
    def test_converter_keeps_privileged_fields_out_of_policy_prompt(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            raw = base / "train.jsonl"
            raw.write_text(json.dumps(TASK) + "\n")
            manifest = preprocess(raw, base / "data", expected_source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
            rows = pq.read_table(base / "data/train.parquet").to_pylist()
            self.assertEqual(len(rows), 1)
            prompt = json.dumps(rows[0]["prompt"])
            self.assertNotIn(TASK["ground_truth"], prompt)
            self.assertNotIn("return x + 7", prompt)
            self.assertEqual(rows[0]["extra_info"]["tools_kwargs"]["interact_with_env"]["create_kwargs"]["task"]["test_cases"], ["f(1)", "f(-1)"])
            self.assertFalse(manifest["test_split_read"])
            self.assertEqual(pq.ParquetFile(base / "data/unused_train_loader.parquet").metadata.num_rows, 1)
            with self.assertRaises(ValueError):
                preprocess(raw, base / "wrong")

    def test_actual_environment_scores_code_and_preserves_native_metric(self):
        with patch.dict(os.environ, {"COLBENCH_USER_STRATEGY": "flow"}):
            os.environ.pop("EAU_COLBENCH_RUN", None)
            env = ColBenchCodeEnv(get_code_config(), TASK, id="fixture")
            env.reset()
            _, _, done, _, _ = env.step("I WANT TO ANSWER:\ndef f(x):\n    return x + 7")
            self.assertTrue(done)
            self.assertEqual(env.calculate_reward(), 1.0)
            self.assertTrue(env.score_detail["all_pass"])
            self.assertEqual(env.score_detail["native_pass_fraction"], 1.0)
            wrong = ColBenchCodeEnv(get_code_config(), TASK, id="wrong")
            wrong.reset()
            wrong.step("I WANT TO ANSWER:\ndef f(x):\n    return 999")
            self.assertEqual(wrong.calculate_reward(), 0.0)

    def test_real_episodes_pin_user_and_reward_code_until_next_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            initialize(run)
            flow_draft(run, "old reply")
            pin_update(run, step=1)
            with patch.dict(os.environ, {"EAU_COLBENCH_RUN": str(run), "EAU_COLBENCH_UPTAKE_DIR": str(run / "uptake")}):
                first = ColBenchCodeEnv(get_code_config(), TASK, id="same")
                first.reset()
                flow_draft(run, "new reply")
                path = run / "drafts/environment/environment.py"
                path.write_text(path.read_text().replace("return native_reward", "return native_reward * 0.25"))
                second = ColBenchCodeEnv(get_code_config(), TASK, id="same")
                second.reset()
                self.assertEqual(second.step("What is required?")[0]["feedback"], "old reply")
                pin_update(run, step=2)
                third = ColBenchCodeEnv(get_code_config(), TASK, id="same")
                third.reset()
                self.assertEqual(third.step("What is required?")[0]["feedback"], "new reply")
                self.assertEqual(first.step("What is required?")[0]["feedback"], "old reply")
                answer = "I WANT TO ANSWER:\ndef f(x):\n    return x + 7"
                first.step(answer)
                third.step(answer)
                self.assertEqual(first.calculate_reward(), 1.0)
                self.assertEqual(third.calculate_reward(), .25)
                self.assertEqual(third.score_detail["native_pass_fraction"], 1.0)
            receipts = [json.loads(line) for line in (run / "uptake/consumption.jsonl").read_text().splitlines()]
            self.assertEqual(len({x["values"]["snapshot_sha256"] for x in receipts}), 2)

    def test_invalid_edits_keep_last_good_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            initialize(run)
            initial = (run / "live/snapshot.json").read_bytes()
            config = run / "drafts/environment/settings.json"
            value = json.loads(config.read_text())
            value["reward_scale"] = float("nan")
            config.write_text(json.dumps(value))
            pin_update(run, step=1)
            self.assertEqual((run / "live/snapshot.json").read_bytes(), initial)
            value["reward_scale"] = 1.0
            config.write_text(json.dumps(value))
            (run / "drafts/user/flow_user.py").write_text("class FlowUser: pass")
            pin_update(run, step=2)
            self.assertEqual((run / "live/snapshot.json").read_bytes(), initial)
            self.assertEqual(len((run / "rejected_drafts.jsonl").read_text().splitlines()), 2)

    def test_fresh_round_inherits_only_prose(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "r1", Path(temporary) / "r2"
            initialize(first)
            flow_draft(first)
            (first / "drafts/agent/entropy_schedule.json").write_text('{"enable":true,"value":0.01}')
            (first / "experience.md").write_text("An observation from training only.")
            pin_update(first, step=1)
            initialize(second, experience=first / "experience.md")
            self.assertEqual((second / "experience.md").read_text(), "An observation from training only.")
            self.assertEqual(json.loads((second / "drafts/user/env_policy.json").read_text())["user_strategy_mix"]["colbench"], .5)
            self.assertEqual(json.loads((second / "drafts/agent/entropy_schedule.json").read_text()), {"enable": False})
            self.assertFalse((second / "checkpoints").exists())

    def test_real_hydra_recipe_has_critic_and_no_test_loader(self):
        command = train_command(sys.executable, ROOT, Path("/runs/r1"), Path("/policy"), Path("/training"))
        c = compose(command)
        self.assertEqual(c["algorithm"]["adv_estimator"], "gae")
        self.assertEqual(c["algorithm"]["multiturn_gae"]["turn_reward_mode"], "sao")
        self.assertFalse(c["algorithm"]["use_intrinsic_reward"])
        self.assertEqual(c["actor_rollout_ref"]["rollout"]["n"], 1)
        self.assertEqual(c["critic"]["ppo_epochs"], 2)
        self.assertEqual(c["trainer"]["critic_warmup"], 20)
        self.assertFalse(c["trainer"]["val_before_train"])
        self.assertEqual(c["trainer"]["test_freq"], -1)
        self.assertEqual(c["data"]["val_files"], "/training/data/unused_train_loader.parquet")
        self.assertNotIn("test.parquet", " ".join(command))

    def test_container_does_not_mount_parent_evaluation_or_docker_socket(self):
        command = docker_command(image="colbench:fixture", workspace="/train", policy="/policy", output="/fresh", devices="0,1")
        mounts = [command[i + 1] for i, x in enumerate(command) if x == "--mount"]
        self.assertEqual(len(mounts), 3)
        self.assertEqual(mounts[0], "type=bind,src=/train,dst=/training,readonly")
        self.assertEqual(mounts[2], "type=bind,src=/fresh,dst=/runs")
        self.assertIn("--read-only", command)
        self.assertNotIn("/var/run/docker.sock", " ".join(command))
        self.assertFalse(any(x == "--pid" or x.startswith(("--pid=", "--privileged")) for x in command))

    def test_actual_training_hooks_dispatch_background_editor_and_publish_later(self):
        from meta_agent import training_audit as hooks
        editor = ("import json,sys,time;from pathlib import Path;time.sleep(.2);"
                  "r=json.loads(Path(sys.argv[1]).read_text());"
                  "p=Path(r['editable_roots'][0])/'agent/entropy_schedule.json';"
                  "t=p.with_suffix('.pending');t.write_text(json.dumps({'enable':True,'value':.004}));"
                  "t.replace(p);print('Changed entropy from training evidence.')")
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            initialize(run)
            env = dict(EAU_COLBENCH_RUN=str(run), EAU_ASYNC_AUDIT_COMMAND=json.dumps([sys.executable, "-c", editor, "{request}"]),
                EAU_ASYNC_AUDIT_RUN=run.name, EAU_ASYNC_AUDIT_DIR=str(run / "audits"),
                EAU_AUDIT_CONTEXT_PATH=str(run / "audit_context.json"),
                EAU_ASYNC_AUDIT_DIRECT_ROOTS=json.dumps([str(run / "drafts")]),
                EAU_ASYNC_AUDIT_EVERY="20", EAU_ASYNC_AUDIT_STOP="22")
            previous = hooks._coordinator
            hooks._coordinator = None
            try:
                with patch.dict(os.environ, env):
                    os.environ.pop("EAU_TAU2_RUN", None)
                    for step in range(1, 12):
                        hooks.before_training_step(step)
                        hooks.audit_training_step(step, {"actor/loss": 1 / step})
                        if step == 2:
                            self.assertEqual(json.loads((run / "active/entropy_schedule.json").read_text()), {"enable": False})
                        time.sleep(.04)
                    hooks.before_training_step(22, is_final=True)
                    self.assertEqual(json.loads((run / "active/entropy_schedule.json").read_text()), dict(enable=True, value=.004))
                    hooks.audit_training_step(22, {"actor/loss": .01}, is_final=True)
                    self.assertTrue((run / "audits/direct_edits.jsonl").exists())
                    hooks.close_training_audit()
            finally:
                if hooks._coordinator is not None:
                    hooks._coordinator.close(timeout=0)
                hooks._coordinator = previous


if __name__ == "__main__":
    unittest.main()
