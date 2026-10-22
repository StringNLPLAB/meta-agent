#!/usr/bin/env python3
"""Validate and exec the vendored verl trainer; never install or download files."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tau3_minimal.tau_adapter import benchmark_identity, file_hash

MODULES = ("torch", "transformers", "ray", "datasets", "pyarrow", "tensordict", "torchdata",
           "hydra", "omegaconf", "sglang", "flash_attn", "gymnasium", "litellm")


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    expected = {"version", "root", "benchmark_root", "run", "output_dir", "user_api_key_env", "overrides", "artifact_hashes", "manifest_sha256"}
    if set(recipe) != expected or recipe["version"] != 1 or Path(recipe["root"]).resolve() != ROOT:
        raise ValueError("Recipe version, fields, or checkout identity is invalid; prepare it in this checkout")
    output = Path(recipe["output_dir"]).resolve(strict=True)
    if not output.is_relative_to(ROOT / "reports"):
        raise ValueError("Recipe output must be under this checkout's reports directory")
    for name, digest in recipe["artifact_hashes"].items():
        if name not in {"train.parquet", "dev.parquet"} or file_hash(output / name) != digest:
            raise ValueError("Prepared dataset identity changed; investigate before starting training")
    if set(recipe["artifact_hashes"]) != {"train.parquet", "dev.parquet"}:
        raise ValueError("Both train and train-derived dev artifacts must be bound")
    if not isinstance(recipe["overrides"], list) or not all(isinstance(item, str) for item in recipe["overrides"]):
        raise ValueError("Recipe overrides must be literal Hydra argument strings")
    identity = benchmark_identity(recipe["benchmark_root"])
    if file_hash(output / "manifest.json") != recipe["manifest_sha256"]:
        raise ValueError("Prepared manifest changed after recipe creation")
    manifest = json.loads((output / "manifest.json").read_text())
    if identity["sources"] != manifest["benchmark"]["sources"]:
        raise ValueError("Benchmark source changed after preparation")
    benchmark = Path(recipe["benchmark_root"]).resolve(strict=True)
    domain = manifest["domain"]
    if domain not in {"retail", "airline", "telecom"}:
        raise ValueError("Unsupported benchmark training domain")
    for key, directory, pattern in (
        ("domain_data_sha256", benchmark / "data/tau2/domains" / domain, "*"),
        ("domain_source_sha256", benchmark / "src/tau2/domains" / domain, "*.py"),
    ):
        actual = {path.relative_to(benchmark).as_posix(): file_hash(path)
                  for path in sorted(directory.rglob(pattern)) if path.is_file()}
        if actual != manifest["benchmark"][key]:
            raise ValueError("Benchmark domain data/source changed after preparation")
    return recipe


def configure_environment(recipe):
    benchmark = Path(recipe["benchmark_root"]).resolve(strict=True)
    paths = [str(ROOT), str(ROOT / "gyms/Tau2Gym"), str(benchmark / "src")]
    previous = os.environ.get("PYTHONPATH")
    if previous:
        paths.append(previous)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    sys.path[:0] = paths[:3]
    os.environ["TAU2_DATA_DIR"] = str(benchmark / "data")
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["LOGURU_LEVEL"] = "WARNING"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    key_name = recipe["user_api_key_env"]
    if key_name and not os.environ.get(key_name):
        raise ValueError("Declared user API key environment variable is empty")


def composed_config(recipe):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    # The inherited base config's file search path is relative to the checkout.
    os.chdir(ROOT)
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parent / "config")):
        config = compose(config_name="tau3_trainer", overrides=recipe["overrides"])
    if config.data.val_files != str(Path(recipe["output_dir"]) / "dev.parquet"):
        raise ValueError("Validation must reference the prepared train-derived dev partition")
    if config.data.train_files != str(Path(recipe["output_dir"]) / "train.parquet"):
        raise ValueError("Training must reference the prepared official train partition")
    if config.trainer.val_before_train or config.trainer.test_freq != -1:
        raise ValueError("The default training recipe must not run benchmark validation")
    if config.actor_rollout_ref.rollout.name != "sglang" or config.actor_rollout_ref.rollout.mode != "sync":
        raise ValueError("This migrated recipe targets the synchronous SGLang rollout")
    return OmegaConf.to_container(config, resolve=True)


def dependency_report():
    missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    versions = {}
    for name in ("torch", "transformers", "sglang", "gymnasium", "litellm", "flash-attn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"missing": missing, "versions": versions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Compose real Hydra configuration without CUDA/model/server startup")
    args = parser.parse_args(argv)
    try:
        if sys.version_info[:2] != (3, 12):
            raise ValueError("This tau3/verl training profile requires Python 3.12")
        recipe = load_recipe(args.recipe)
        configure_environment(recipe)
        config = composed_config(recipe)
        dependencies = dependency_report()
        command = [sys.executable, "-u", "-m", "verl.trainer.main_ppo", "--config-path", str(Path(__file__).parent / "config"),
                   "--config-name", "tau3_trainer", *recipe["overrides"]]
        if args.dry_run:
            print(json.dumps({"dry_run": True, "training_started": False, "command": command,
                              "dependencies": dependencies, "resolved_config": config}, ensure_ascii=False, indent=2))
            return 0
        if dependencies["missing"]:
            raise ValueError(f"Missing training dependencies: {', '.join(dependencies['missing'])}; see this example's README")
        import torch
        import flash_attn  # noqa: F401; catch an incompatible compiled extension before Ray starts.
        model_path = Path(config["actor_rollout_ref"]["model"]["path"])
        manifest = json.loads((Path(recipe["output_dir"]) / "manifest.json").read_text())
        if file_hash(model_path / "config.json") != manifest["policy_config_sha256"]:
            raise ValueError("Policy config changed after preparation")
        if not any(model_path.glob("*.safetensors")) and not any(model_path.glob("pytorch_model*.bin")):
            raise ValueError("Local policy directory has no supported Hugging Face weight files")
        if not torch.cuda.is_available() or torch.cuda.device_count() < config["trainer"]["n_gpus_per_node"]:
            raise ValueError("The configured number of CUDA GPUs is unavailable")
        if dependencies["versions"].get("sglang") != "0.4.7":
            raise ValueError("This migrated rollout requires the documented SGLang 0.4.7 profile")
        if dependencies["versions"].get("torch", "").split("+")[0] != "2.7.1" or dependencies["versions"].get("transformers") != "4.52.3":
            raise ValueError("Training requires the matching Torch 2.7.1 / Transformers 4.52.3 profile")
        checkpoint_path = Path(recipe["output_dir"]) / "checkpoints"
        if checkpoint_path.exists():
            raise ValueError("Checkpoint output already exists; implicit resume/retry is disabled")
        os.execv(sys.executable, command)
    except (OSError, ValueError, ImportError, KeyError) as error:
        print(f"[TAU3 LAUNCH] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
