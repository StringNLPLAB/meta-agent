"""CPU checks for the vendored engine's packaging and configuration entrypoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class VendoredEngineTests(unittest.TestCase):
    def test_manifest_covers_the_handover_source(self):
        manifest = json.loads((ROOT / "verl/vendor_provenance.json").read_text())
        self.assertEqual(manifest["source_version"], "0.4.0.dev")
        self.assertIsNone(manifest["source_commit"])
        self.assertEqual(len(manifest["files"]), 262)
        for record in manifest["files"]:
            path = ROOT / "verl" / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(len(record["source_sha256"]), 64)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["vendored_sha256"],
                record["path"],
            )

    def test_gpu_profile_keeps_incompatible_rollout_engines_separate(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        extras = project["optional-dependencies"]
        self.assertIn("sglang[srt,openai]==0.4.7", extras["sglang"])
        self.assertIn("torch==2.7.1", extras["sglang"])
        self.assertIn("transformers==4.52.3", extras["sglang"])
        self.assertIn("flash-attn==2.7.4.post1", extras["sglang"])
        self.assertFalse(any("vllm" in spec for spec in project["dependencies"]))
        self.assertFalse(any("vllm" in spec for spec in extras["sglang"]))

    def test_installer_is_valid_bash_and_has_a_read_only_help_path(self):
        script = ROOT / "requirements/install_training.sh"
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
        result = subprocess.run(
            ["bash", str(script), "--help"],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("Installs dependencies only", result.stdout)

    @unittest.skipUnless(os.environ.get("VERL_VALIDATION_WHEEL"), "Set VERL_VALIDATION_WHEEL to inspect a built wheel")
    def test_wheel_contains_engine_configuration_and_third_party_notices(self):
        with zipfile.ZipFile(os.environ["VERL_VALIDATION_WHEEL"]) as archive:
            names = set(archive.namelist())
            self.assertIn("verl/version/version", names)
            self.assertIn("verl/trainer/config/ppo_trainer.yaml", names)
            self.assertIn("verl/trainer/runtime_env.yaml", names)
            self.assertIn("verl/vendor_provenance.json", names)
            notices = [name for name in names if ".dist-info/licenses/" in name]
            self.assertEqual(len(notices), 1, notices)
            self.assertTrue(notices[0].endswith("/NOTICE"))
            notice = archive.read(notices[0]).decode()
            self.assertIn("Apache License", notice)
            self.assertIn("BSD 3-Clause License", notice)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("hydra", "tensordict", "torch", "ray")),
        "Hydra, TensorDict, Torch and Ray are needed for the CPU configuration check",
    )
    def test_hydra_config_composes_without_starting_workers(self):
        result = subprocess.run(
            [sys.executable, "-m", "verl.trainer.main_ppo", "--cfg", "job"],
            cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("actor_rollout_ref:", result.stdout)
        self.assertIn("adv_estimator:", result.stdout)
        self.assertNotIn("Started a local Ray instance", result.stderr)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("hydra", "tensordict", "torch", "torchdata", "ray", "codetiming")),
        "Trainer core dependencies are needed for CPU trajectory export checks",
    )
    def test_review_samples_include_mixed_grpo_groups_on_the_requested_cadence(self):
        """Synthetic tensors exercise the real trainer dump, without a benchmark claim."""
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from verl import DataProto
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        batch = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[1, 2]] * 4),
                "responses": torch.tensor([[3, 4]] * 4),
                "token_level_rewards": torch.tensor([[1.0], [0.0], [0.25], [0.25]]),
            },
            non_tensors={
                "uid": np.array(["mixed", "mixed", "partial", "partial"], dtype=object),
                "data_source": np.array(["interact_tau2_retail"] * 4, dtype=object),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = object.__new__(RayPPOTrainer)
            trainer.global_steps = 2
            trainer.config = OmegaConf.create({"trainer": {
                "default_local_dir": directory,
                "debug_trajectory_freq": 10,
                "log_train_trajectories": 1,
            }})
            trainer.tokenizer = SimpleNamespace(decode=lambda ids, **_: " ".join(map(str, ids.tolist())))
            with patch.dict(os.environ, {"USERRL_DEBUG_DUMP_GYMS": ""}):
                trainer._analyze_group_reward_consistency(batch, {})
                self.assertFalse(list(Path(directory).rglob("*.json")))
                trainer.config.trainer.debug_trajectory_freq = 1
                trainer._analyze_group_reward_consistency(batch, {})
            files = sorted(Path(directory).rglob("*.json"))
            self.assertEqual(len(files), 2)
            records = [json.loads(path.read_text()) for path in files]
            self.assertEqual({entry["reward_type"] for entry in records}, {"mixed", "partial"})
            for entry in records:
                self.assertEqual(entry["sample_origin"], "training_batch")
                self.assertTrue(entry["biased_sample"])
                self.assertEqual(entry["global_step"], 2)
                self.assertEqual(len(entry["trajectories"]), 2)


if __name__ == "__main__":
    unittest.main()
