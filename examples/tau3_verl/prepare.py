#!/usr/bin/env python3
"""Prepare real native τ³ train-only parquet and a bounded verl launch recipe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tau3_minimal.tau_adapter import Tau3Adapter, file_hash, service_url


def partition_tasks(tasks, dev_fraction, seed):
    """Partition only already-verified official train IDs, never a test/base split."""
    if not math.isfinite(dev_fraction) or not 0 < dev_fraction < 0.5:
        raise ValueError("dev_fraction must be between 0 and 0.5")
    if len(tasks) < 3:
        raise ValueError("At least three official train tasks are required")
    ordered = sorted(tasks, key=lambda item: str(item.id))
    random.Random(seed).shuffle(ordered)
    count = max(1, int(len(ordered) * dev_fraction))
    return ordered[count:], ordered[:count]


def rows_for_tasks(tasks, *, policy, schemas, domain, environment, role):
    rows = []
    for index, task in enumerate(tasks):
        rows.append({
            "data_source": "tau3_training",
            "prompt": [{"role": "system", "content": policy}],
            "ability": "tool_use",
            "reward_model": {"style": "rule", "type": "env_reward", "ground_truth": ""},
            "extra_info": {
                "index": index, "task_id": str(task.id), "split": "train", "partition": role,
                # Parquet struct inference corrupts heterogeneous JSON schemas
                # by filling unrelated properties with null. The rollout loads
                # exact native schemas by domain; retain only names as lineage.
                "domain": domain, "solo_mode": False,
                "tool_names": [schema["function"]["name"] for schema in schemas],
                "need_tools_kwargs": True,
                "tools_kwargs": {"_tau2_env_kwargs": dict(environment, task_id=str(task.id), id=str(task.id))},
            },
        })
    return rows


def overrides_for(args, output):
    def value(item):
        # Hydra's quoted scalar syntax prevents paths containing commas/brackets
        # from becoming list expressions. No shell is involved.
        return json.dumps(str(item))
    return [
        f"data.train_files={value(output / 'train.parquet')}",
        f"data.val_files={value(output / 'dev.parquet')}",
        f"data.train_batch_size={args.train_batch_size}",
        f"actor_rollout_ref.model.path={value(Path(args.model_path).resolve())}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.train_batch_size}",
        f"actor_rollout_ref.rollout.n={args.rollouts}",
        f"actor_rollout_ref.rollout.multi_turn.max_turns={args.max_turns}",
        f"actor_rollout_ref.rollout.multi_turn.model_name={value(args.user_model)}",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={value(Path(__file__).parent / 'config/tool_config/tau3_tools.yaml')}",
        f"trainer.n_gpus_per_node={args.gpus}", f"trainer.total_training_steps={args.steps}",
        f"trainer.total_epochs={max(1, math.ceil(args.steps / max(1, args.train_tasks // args.train_batch_size)))}",
        f"trainer.save_freq={args.steps}", f"trainer.experiment_name={value(args.run)}",
        f"trainer.default_local_dir={value(output / 'checkpoints')}",
    ]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--benchmark-root", required=True)
    result.add_argument("--model-path", required=True, help="Existing local Hugging Face policy directory")
    result.add_argument("--user-model", required=True, help="Explicit LiteLLM ID, e.g. openai/local-user")
    result.add_argument("--user-base-url", required=True)
    result.add_argument("--user-api-key-env", help="Credential environment variable name, never its value")
    result.add_argument("--domain", choices=("retail", "airline", "telecom"), default="retail")
    result.add_argument("--reward-mode", choices=("action", "env"), default="action")
    result.add_argument("--run", required=True)
    result.add_argument("--output-dir")
    result.add_argument("--steps", type=int, default=10)
    result.add_argument("--gpus", type=int, default=1)
    result.add_argument("--train-batch-size", type=int, default=2)
    result.add_argument("--rollouts", type=int, default=2)
    result.add_argument("--max-turns", type=int, default=8)
    result.add_argument("--dev-fraction", type=float, default=0.1)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--trainer-python", default=sys.executable)
    result.add_argument("--runtime-output", help="Also create an owned runtime launch JSON")
    result.add_argument("--backend", choices=("codex", "opencode"), default="opencode")
    result.add_argument("--meta-model")
    result.add_argument("--meta-base-url")
    result.add_argument("--meta-api-key-env")
    result.add_argument("--every-steps", type=int, default=1)
    result.add_argument("--every-seconds", type=float, default=600)
    result.add_argument("--max-calls", type=int, default=30)
    result.add_argument("--trainer-timeout", type=float, default=14400)
    return result


def runtime_config(args, output, recipe):
    if args.every_steps < 1 or args.max_calls < 3 or not math.isfinite(args.every_seconds) or args.every_seconds <= 0:
        raise ValueError("Review cadence must be positive, and call budget at least three")
    if not math.isfinite(args.trainer_timeout) or args.trainer_timeout <= 0:
        raise ValueError("trainer_timeout must be finite and positive")
    relative = output.relative_to(ROOT).as_posix()
    supervisor = [
        "--log", f"reports/runtime/{args.run}/trainer.stdout", "--trajectories", f"{relative}/checkpoints/debug_trajectories",
        "--backend", args.backend, "--mode", "propose", "--every-steps", str(args.every_steps),
        "--every-seconds", str(args.every_seconds), "--max-calls", str(args.max_calls), "--export-atif",
    ]
    if args.meta_model:
        supervisor += ["--agent-model", args.meta_model]
    if args.backend == "opencode":
        if not args.meta_model or not args.meta_base_url:
            raise ValueError("OpenCode supervision needs --meta-model and --meta-base-url")
        supervisor += ["--opencode-base-url", service_url(args.meta_base_url)]
        if args.meta_api_key_env:
            supervisor += ["--opencode-api-key-env", args.meta_api_key_env]
    return {
        "version": 1, "root": str(ROOT), "run": args.run,
        "trainer_command": [args.trainer_python, "-u", str(Path(__file__).with_name("launch.py")), "--recipe", str(recipe)],
        "contract": {
            "objective": "Improve a native tau3 training policy using the vendored Info-GRPO trainer and current training trajectories.",
            "budget": f"{args.steps} optimizer steps; {args.gpus} GPUs; at most {args.max_calls} coding-agent calls.",
            "guardrails": ["Read-only proposals; do not execute modifications without a project gate.", "Never use benchmark test/base task outcomes or hidden criteria for steering."],
            "evaluation_protocol": "Official train split only. Train-derived dev is a diagnostic partition. ACTION/ENV proxy reward is not an official benchmark score.",
            "checkpoint_selection": "Retain the final planned training checkpoint; no held-out selection.",
            "allowed_feedback": [f"reports/runtime/{args.run}/trainer.stdout", f"{relative}/manifest.json", f"{relative}/checkpoints/debug_trajectories"],
            "excluded_feedback": ["Test/base benchmark outcomes, hidden evaluator criteria, raw user prompts, and secrets."],
            "allowed_operations": [],
        },
        "supervisor_args": supervisor, "trainer_timeout_seconds": args.trainer_timeout,
        "poll_seconds": 0.2, "final_timeout_seconds": 660, "failure_policy": "stop_owned_trainer",
    }


def prepare(args):
    if sys.version_info[:2] != (3, 12):
        raise ValueError("This tau3/verl training profile requires Python 3.12")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        raise ValueError("run must be a plain path-safe identifier")
    if any(value < 1 for value in (args.steps, args.gpus, args.train_batch_size, args.max_turns)) or args.rollouts < 2:
        raise ValueError("Positive budgets and at least two GRPO rollouts are required")
    if args.train_batch_size % args.gpus or args.train_batch_size * args.rollouts % args.gpus:
        raise ValueError("Train batch and expanded rollout batch must be divisible by GPU count")
    policy_path = Path(args.model_path).resolve(strict=True)
    if not policy_path.is_dir() or not (policy_path / "config.json").is_file():
        raise ValueError("model-path must contain a local Hugging Face config.json")
    if args.user_api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.user_api_key_env):
        raise ValueError("user-api-key-env must be an environment variable name")
    output = Path(args.output_dir).resolve() if args.output_dir else ROOT / "reports/tau3_verl" / args.run
    if not output.is_relative_to(ROOT / "reports") or output.exists():
        raise ValueError("output-dir must be a new directory under this checkout's reports/")
    adapter = Tau3Adapter(
        benchmark_root=args.benchmark_root, domain=args.domain, reward_mode=args.reward_mode,
        user_model=args.user_model, user_base_url=args.user_base_url,
        user_api_key_env=args.user_api_key_env, max_turns=args.max_turns,
    )
    train_tasks, dev_tasks = partition_tasks(adapter.tasks, args.dev_fraction, args.seed)
    if len(train_tasks) < args.train_batch_size:
        raise ValueError("Official train partition is smaller than train-batch-size")
    args.train_tasks = len(train_tasks)
    from tau2.registry import registry
    domain_environment = registry.get_env_constructor(args.domain)(solo_mode=False)
    from tau2.environment.tool import as_tool
    from tau2.gym.gym_agent import done
    tools = list(domain_environment.get_tools())
    if not any(tool.name == "done" for tool in tools):
        tools.append(as_tool(done))
    environment = {
        "benchmark_root": str(Path(args.benchmark_root).resolve()), "domain": args.domain,
        "task_split": "train", "reward_mode": args.reward_mode, "user_model": args.user_model,
        "user_base_url": service_url(args.user_base_url), "max_turns": args.max_turns,
    }
    if args.user_api_key_env:
        environment["user_api_key_env"] = args.user_api_key_env
    kwargs = {"policy": domain_environment.get_policy(), "schemas": [tool.openai_schema for tool in tools],
              "domain": args.domain, "environment": environment}
    rows = {"train": rows_for_tasks(train_tasks, role="train", **kwargs),
            "dev": rows_for_tasks(dev_tasks, role="train_derived_dev", **kwargs)}
    import pyarrow as pa
    import pyarrow.parquet as pq
    recipe_path = output / "recipe.json"
    recipe = {"version": 1, "root": str(ROOT), "benchmark_root": environment["benchmark_root"],
              "run": args.run, "output_dir": str(output), "user_api_key_env": args.user_api_key_env,
              "overrides": overrides_for(args, output)}
    launch = runtime_config(args, output, recipe_path) if args.runtime_output else None
    if args.runtime_output and Path(args.runtime_output).exists():
        raise ValueError("runtime-output already exists")
    output.mkdir(parents=True)
    for name, data in rows.items():
        pq.write_table(pa.Table.from_pylist(data), output / f"{name}.parquet")
    manifest = {"version": 1, "benchmark": adapter.identity, "domain": args.domain,
                "reward_proxy": args.reward_mode, "benchmark_score": False, "seed": args.seed,
                "partitions": {name: [row["extra_info"]["task_id"] for row in data] for name, data in rows.items()},
                "files": {f"{name}.parquet": file_hash(output / f"{name}.parquet") for name in rows},
                "policy_config_sha256": file_hash(policy_path / "config.json")}
    recipe["artifact_hashes"] = manifest["files"]
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    recipe["manifest_sha256"] = file_hash(output / "manifest.json")
    (output / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")
    if launch:
        target = Path(args.runtime_output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            stream.write(json.dumps(launch, ensure_ascii=False, indent=2) + "\n")
    return recipe_path


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(prepare(args))
        return 0
    except (OSError, ValueError, ImportError) as error:
        print(f"[TAU3 PREPARE] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
