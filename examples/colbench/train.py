"""Standalone ColBench PPO training with optional non-blocking file edits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.colbench.prepare import verify
from examples.colbench.recipe import train_command
from meta_agent.colbench_runtime import initialize, write


def compose(command):
    from hydra import compose as hydra_compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "examples/colbench/config")):
        config = hydra_compose(config_name="colbench_trainer", overrides=command[5:])
    return OmegaConf.to_container(config, resolve=True)


def environment(workspace, run, args):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("EAU_", "USERRL_", "TAU_", "COLBENCH_"))}
    for name in ("EVAL_API_KEY", "EVAL_BASE_URL", "FORGE_API_KEY", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(name, None)
    paths = (ROOT, ROOT / "gyms/ColBenchGym", ROOT / "UserRL/gyms")
    env.update(PYTHONPATH=os.pathsep.join(map(str, paths)), PYTHONHASHSEED=str(args.seed),
        EAU_COLBENCH_RUN=str(run), EAU_RUN_DIR=str(run), EAU_STEP_REVIEW_RUN=run.name,
        EAU_TRAINING_SEED=str(args.seed),
        EAU_COLBENCH_UPTAKE_DIR=str(run / "uptake"),
        EAU_TRAINING_API_USAGE_FILE=str(run / "training_api_usage.jsonl"),
        USERRL_META_ARM="online", USERRL_USER_STRATEGY="llm",
        USERRL_ENV_POLICY_CONFIG=str(run / "active/env_policy.json"),
        USERRL_ENTROPY_SCHEDULE_CONFIG=str(run / "active/entropy_schedule.json"),
        USERRL_CLIP_SCHEDULE_CONFIG=str(run / "active/clip_schedule.json"),
        USERRL_DIFFICULTY_CONFIG=str(run / "active/disabled.json"),
        USERRL_REWARD_SHAPING_CONFIG=str(run / "active/disabled.json"),
        USERRL_PROCESS_REWARD_CONFIG=str(run / "active/disabled.json"),
        USERRL_ENABLE_QWEN3_THINKING="0", USERRL_STRIP_QWEN_THINK_OUTPUT="0",
        EAU_REQUIRE_NONTHINKING_POLICY="1", EAU_NONTHINKING_OUTPUT_POLICY="monitor",
        EAU_NONTHINKING_BEHAVIOR_DIR=str(run / "nonthinking_behavior"),
        USERRL_TERMINAL_TASK_REWARD_ONLY="0", USERRL_EVAL_DUMP_DIR=str(run / "training_samples"),
        USERRL_DEBUG_DUMP_GYMS="colbench", COLBENCH_USER_STRATEGY="llm",
        COLBENCH_USER_TEMPERATURE="0.7", COLBENCH_LEAK_GUARD="1",
        MULTITURN_MODEL_NAME=args.user_model, USER_MODEL_NAME=args.user_model,
        OPENAI_BASE_URL=args.user_url, OPENAI_API_BASE=args.user_url,
        OPENAI_API_KEY=os.environ.get("COLBENCH_USER_API_KEY", "EMPTY"),
        USER_ENV_STEP_TIMEOUT="180", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled", LOGURU_LEVEL="WARNING",
        LITELLM_LOCAL_MODEL_COST_MAP="True", RAY_ADDRESS="local",
        RAY_TMPDIR=tempfile.mkdtemp(prefix="colbench-ray-"), RAY_ENABLE_MEMORY_MONITOR="0",
        RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE="1", RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES="4294967296",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.audit_command:
        argv = json.loads(Path(args.audit_command).read_text())
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError("audit-command must contain a nonempty JSON argv array")
        env.update(EAU_ASYNC_AUDIT_COMMAND=json.dumps(argv), EAU_ASYNC_AUDIT_RUN=run.name,
            EAU_ASYNC_AUDIT_DIR=str(run / "audits"), EAU_AUDIT_CONTEXT_PATH=str(run / "audit_context.json"),
            EAU_ASYNC_AUDIT_DIRECT_ROOTS=json.dumps([str(run / "drafts")]),
            EAU_ASYNC_AUDIT_EVERY=str(args.audit_every), EAU_ASYNC_AUDIT_FIRST="1",
            EAU_ASYNC_AUDIT_STOP=str(args.steps or 624), EAU_ASYNC_AUDIT_FAILURE_POLICY="raise")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Train-only workspace from prepare.py")
    parser.add_argument("--output", required=True, help="Fresh run directory")
    parser.add_argument("--policy", required=True, help="Original local Qwen3-4B HF directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, choices=(2, 4), default=2)
    parser.add_argument("--user-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--user-model", default="coser-usersim")
    parser.add_argument("--steps", type=int, help="Optional shorter predeclared budget; default two epochs (at most624 updates)")
    parser.add_argument("--full-checkpoints", action="store_true", help="Include optimizer and scheduler/RNG for recovery")
    parser.add_argument("--audit-command", help="JSON argv with {request}; omitted means a fixed baseline")
    parser.add_argument("--audit-every", type=int, default=20)
    parser.add_argument("--experience", help="Prior training-only Markdown/text, never executable state")
    parser.add_argument("--dry-run", action="store_true", help="Compose the real Hydra config; no model/API calls or updates")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use the pinned Python3.12 training profile")
    if args.audit_every < 1 or (args.steps is not None and not 1 <= args.steps <= 624):
        raise ValueError("Invalid audit interval or training-step budget")
    os.chdir(ROOT)
    workspace = Path(args.workspace).resolve(strict=True)
    manifest = verify(workspace)
    run, policy = Path(args.output).resolve(), Path(args.policy).resolve(strict=True)
    if run.exists() or not (policy / "config.json").is_file():
        raise ValueError("Use a fresh output and a local HF model directory")
    command = train_command(sys.executable, ROOT, run, policy, workspace, args.seed, args.gpus,
        user_model=args.user_model, steps=args.steps, full_checkpoints=args.full_checkpoints)
    config = compose(command)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, model_calls=0, optimizer_updates=0,
                              resolved_config=config, training_rows=manifest["rows"]), indent=2))
        return
    initialize(run, experience=args.experience)
    env = environment(workspace, run, args)
    write(run / "contract.json", dict(seed=args.seed, original_policy="Qwen/Qwen3-4B",
        epochs=2, requested_updates=args.steps, prompts_per_batch=32, rollout_n=1, thinking=False,
        algorithm="PPO/multiturn GAE SAO", critic_warmup=20, policy_gpus=args.gpus,
        initial_user="50% symbolic flow /50% configured LLM, task-ID deterministic", user_temperature=.7,
        initial_reward="Native test-case pass fraction; no step penalty or added reward",
        meta_agent=bool(args.audit_command), training_manifest=manifest,
        checkpoint_selection="Final completed update; never selected by evaluation",
        checkpoint_contents="model,optimizer,extra" if args.full_checkpoints else "model",
        inheritance="Original policy and draft defaults each round; selected prose experience only"))
    write(run / "resolved_config.json", config)
    started = time.time()
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
            raise ValueError("Insufficient visible policy GPUs; the user and meta-agent are separate services")
        from examples.tau2.models import verify_policy
        verify_policy(policy)
        from torch_memory_saver import configure_subprocess
        libraries = []
        for location in site.getsitepackages():
            base = Path(location)
            libraries.extend(str(p) for p in (base / "nvidia").glob("*/lib") if p.is_dir())
            if (base / "torch/lib").is_dir():
                libraries.append(str(base / "torch/lib"))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + [env.get("LD_LIBRARY_PATH", "")])
        for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            os.environ.pop(name, None)
        with (run / "logs/training.log").open("a") as log:
            with configure_subprocess():
                env["LD_PRELOAD"] = os.environ["LD_PRELOAD"]
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise RuntimeError(f"Trainer exited{code}; inspect {run}/logs/training.log")
        tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
        final_step = int(tracker.read_text().strip())
        if not (run / f"checkpoints/global_step_{final_step}/actor").is_dir():
            raise RuntimeError("Trainer returned without its final actor checkpoint")
        if args.steps is not None and final_step != args.steps:
            raise RuntimeError("Training did not complete the requested update budget")
        write(run / "completed.json", dict(final_step=final_step,
              checkpoint=str(run / f"checkpoints/global_step_{final_step}/actor"),
              checkpoint_selection="final_update", evaluation_feedback_used=False))
    finally:
        write(run / "cost.json", dict(elapsed_seconds=time.time() - started, policy_gpus=args.gpus,
              scope="This attempt including setup and failures; external user and meta-agent service costs are separate"))


if __name__ == "__main__":
    main()
