"""Standalone local-GPU Tau2 InfoPO training and configuration validation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import site

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tau2.prepare import verify, write
from examples.tau2.recipe import train_command


def environment(workspace, run, args):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EAU_", "USERRL_", "TAU_"))}
    for name in ("EVAL_API_KEY", "EVAL_BASE_URL", "FORGE_API_KEY"):
        env.pop(name, None)
    # TorchMemorySaver rejects expandable_segments; both allocator aliases matter.
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(name, None)
    paths = (ROOT, ROOT / "gyms/Tau2Gym", ROOT / "UserRL/gyms", workspace / "benchmark/src")
    env.update(PYTHONPATH=os.pathsep.join(map(str, paths)),
        TAU2_DATA_DIR=str(workspace / "benchmark/data"), PYTHONHASHSEED=str(args.seed),
        EAU_TAU2_RUN=str(run), EAU_RUN_DIR=str(run), EAU_STEP_REVIEW_RUN=run.name,
        EAU_USER_SIM_BOOTSTRAP="coser_scratch_v1", EAU_USER_SIM_MODE="editable" if args.audit_command else "fixed_original",
        EAU_USER_SIM_BUNDLE=str(run / "live/usersim_bundle.json"),
        EAU_ENVIRONMENT_BUNDLE=str(run / "live/environment_bundle.json"),
        EAU_USER_SIM_UPTAKE_DIR=str(run / "uptake"), EAU_ENVIRONMENT_UPTAKE_DIR=str(run / "uptake"),
        EAU_COSER_USER_MODEL="openai/coser-usersim", EAU_COSER_SOURCE_MODEL="Neph0s/CoSER-Llama-3.1-8B",
        EAU_COSER_USER_LLM_ARGS=json.dumps(dict(api_base=args.coser_url, api_key="EMPTY", temperature=0.7,
            max_tokens=512, timeout=120, num_retries=0, stop=["<|eot_id|>"])),
        USERRL_USER_STRATEGY="flow", USERRL_META_ARM="online",
        USERRL_ENV_POLICY_CONFIG=str(run / "active/env_policy.json"),
        USERRL_PROCESS_REWARD_CONFIG=str(run / "active/process_reward.json"),
        USERRL_ENTROPY_SCHEDULE_CONFIG=str(run / "active/entropy_schedule.json"),
        USERRL_CLIP_SCHEDULE_CONFIG=str(run / "active/clip_schedule.json"),
        USERRL_DIFFICULTY_CONFIG=str(run / "active/disabled.json"),
        USERRL_REWARD_SHAPING_CONFIG=str(run / "active/disabled.json"),
        EAU_TAU2_REWARD_MODE="process", TAU2_EVAL_TYPE="action", EAU_STRICT_SCORING="1",
        EAU_TERMINAL_REWARD_AUDIT_DIR=str(run / "terminal_action_scores"),
        USERRL_ENABLE_QWEN3_THINKING="0", USERRL_STRIP_QWEN_THINK_OUTPUT="0",
        EAU_REQUIRE_NONTHINKING_POLICY="1", EAU_NONTHINKING_BEHAVIOR_DIR=str(run / "nonthinking_behavior"),
        USERRL_TERMINAL_TASK_REWARD_ONLY="0", USERRL_EVAL_DUMP_DIR=str(run / "training_samples"),
        USERRL_DEBUG_DUMP_GYMS="tau2", OPENAI_API_KEY="EMPTY", OPENAI_API_BASE=args.coser_url,
        PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="disabled", LOGURU_LEVEL="WARNING", LITELLM_LOCAL_MODEL_COST_MAP="True",
        RAY_ADDRESS="local", RAY_TMPDIR=tempfile.mkdtemp(prefix="tau2-ray-"),
        RAY_ENABLE_MEMORY_MONITOR="0", RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE="1",
        RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES="4294967296", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.audit_command:
        command = json.loads(Path(args.audit_command).read_text())
        if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
            raise ValueError("audit-command must contain a nonempty JSON argv array")
        env.update(EAU_ASYNC_AUDIT_COMMAND=json.dumps(command), EAU_ASYNC_AUDIT_RUN=run.name,
            EAU_ASYNC_AUDIT_DIR=str(run / "audits"), EAU_AUDIT_CONTEXT_PATH=str(run / "audit_context.json"),
            EAU_ASYNC_AUDIT_DIRECT_ROOTS=json.dumps([str(run / "drafts")]),
            EAU_ASYNC_AUDIT_EVERY=str(args.audit_every), EAU_ASYNC_AUDIT_STOP="22",
            EAU_ASYNC_AUDIT_FIRST="1", EAU_ASYNC_AUDIT_FAILURE_POLICY="raise")
    return env


def compose(command):
    from hydra import compose as hydra_compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "examples/tau2/config")):
        config = hydra_compose(config_name="tau2_trainer", overrides=command[5:])
    return OmegaConf.to_container(config, resolve=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Detached train workspace from prepare.py")
    parser.add_argument("--output", required=True, help="Fresh run directory; implicit resume is disabled")
    parser.add_argument("--policy", required=True, help="Original local Hugging Face Qwen3-4B directory")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4), default=2, help="Policy GPUs; CoSER is a separate service")
    parser.add_argument("--coser-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--native-reward", action="store_true", help="Start with ACTION only and zero intermediate reward")
    parser.add_argument("--audit-command", help="JSON argv file with {request}/{response}; omitted = fixed baseline")
    parser.add_argument("--audit-every", type=int, default=4)
    parser.add_argument("--experience", help="Prior training-only Markdown/text document; never executable state")
    parser.add_argument("--dry-run", action="store_true", help="Resolve real Hydra configuration; zero model calls/updates")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use the pinned Python 3.12 training profile")
    os.chdir(ROOT)
    workspace = Path(args.workspace).resolve(strict=True)
    verify(workspace, split="train")
    run, policy = Path(args.output).resolve(), Path(args.policy).resolve(strict=True)
    if run.exists() or not (policy / "config.json").is_file():
        raise ValueError("Use a fresh output directory and a local HF model directory")
    command = train_command(sys.executable, ROOT, run, policy, workspace, args.seed, args.gpus)
    config = compose(command)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, optimizer_updates=0, model_calls=0,
                              expected_training_episodes=1760, resolved_config=config), indent=2))
        return
    env = environment(workspace, run, args)
    for name in ("EVAL_API_KEY", "EVAL_BASE_URL", "FORGE_API_KEY"):
        os.environ.pop(name, None)
    os.environ.update(env)
    sys.path[:0] = env["PYTHONPATH"].split(os.pathsep)
    from meta_agent.tau2_runtime import initialize
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
        raise ValueError("Insufficient visible policy GPUs; launch CoSER on a separate GPU")
    if not any(policy.glob("*.safetensors")):
        raise ValueError("Policy weights are missing")
    from examples.tau2.models import verify_policy
    verify_policy(policy)
    initialize(run, native_reward=args.native_reward, experience=args.experience)
    write(run / "active/disabled.json", dict(enable=False))
    write(run / "contract.json", dict(seed=args.seed, updates=22, prompts_per_batch=16, rollout_n=5,
          episodes=1760, thinking=False, policy_gpus=args.gpus, checkpoint_selection="Final update22",
          reward_profile="native_action" if args.native_reward else "action_plus_B0_milestones",
          meta_agent=bool(args.audit_command), inheritance="Fresh policy and draft defaults; selected prose experience only",
          training_manifest_sha256=__import__("hashlib").sha256((workspace / "manifest.json").read_bytes()).hexdigest()))
    write(run / "resolved_config.json", config)
    started = time.time()
    try:
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
                result_code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if result_code:
            raise RuntimeError(f"Trainer exited {result_code}; inspect {run}/logs/training.log")
        if not (run / "checkpoints/global_step_22/actor").is_dir():
            raise RuntimeError("Trainer returned without the predeclared final actor checkpoint")
    finally:
        write(run / "cost.json", dict(elapsed_seconds=time.time() - started, policy_gpus=args.gpus,
              scope="This attempt, including failures. CoSER and meta-agent service costs are separate."))


if __name__ == "__main__":
    main()
