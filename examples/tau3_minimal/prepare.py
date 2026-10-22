#!/usr/bin/env python3
"""Translate an agent-resolved experiment specification into the runtime launch file.

This helper does not start training, a model server, or OpenCode. The outer
coding agent supplies these arguments from the user's natural-language request.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from tau_adapter import benchmark_identity, service_url
from train import parser as training_parser, validate


def parser():
    result = training_parser(require_output=False)
    result.description = __doc__
    result.add_argument("--output", required=True, help="New runtime launch JSON file")
    result.add_argument("--root", required=True, help="Training checkout, containing this example's source")
    result.add_argument("--trainer-python", default=sys.executable,
                        help="Python with training dependencies; native τ³ requires Python 3.12/3.13")
    result.add_argument("--backend", choices=("codex", "opencode"), default="opencode")
    result.add_argument("--meta-model", help="Coding model ID, distinct from trained policy and user simulator")
    result.add_argument("--meta-base-url", help="OpenAI-compatible meta-agent endpoint (OpenCode only)")
    result.add_argument("--meta-api-key-env", help="Environment variable name containing the coding-model key")
    result.add_argument("--opencode-command", default='["opencode"]', help="Literal OpenCode argv JSON array")
    result.add_argument("--meta-context", type=int, default=32768)
    result.add_argument("--meta-output", type=int, default=8192)
    result.add_argument("--every-steps", type=int, default=1)
    result.add_argument("--every-seconds", type=float, default=600)
    result.add_argument("--max-calls", type=int)
    result.add_argument("--agent-timeout", type=float, default=90)
    result.add_argument("--trainer-timeout", type=float, default=3600)
    result.add_argument("--final-timeout", type=float, default=120)
    return result


def launch_config(args):
    """Build a bounded, read-only launch without performing training or API calls."""
    validate(args, check_python=False)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        raise ValueError("Run must be a plain path-safe identifier")
    root = Path(args.root).resolve(strict=True)
    script = Path(__file__).with_name("train.py").resolve()
    if not script.is_relative_to(root):
        raise ValueError("Place this example under the training root so the coding agent can inspect it")
    output = Path(args.output_dir) if args.output_dir else Path("reports/tau3_minimal") / args.run
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if not output.is_relative_to(root / "reports") or output.exists():
        raise ValueError("Training output must be a new path under the checkout's reports directory")
    for key in ("every_steps", "every_seconds", "agent_timeout", "trainer_timeout", "final_timeout"):
        import math
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be finite and positive")
    if args.final_timeout <= args.agent_timeout:
        raise ValueError("Final timeout must exceed the agent timeout to allow final review/export")
    calls = args.max_calls if args.max_calls is not None else args.steps + 10
    if calls < 3:
        raise ValueError("Call budget needs discovery, at least one periodic review, and final review")
    if args.meta_context <= args.meta_output or args.meta_output <= 0:
        raise ValueError("Meta-agent context must exceed its positive output budget")
    trainer = [args.trainer_python, str(script), "--output-dir", str(output), "--run", args.run]
    names = ("steps", "episodes_per_step", "learning_rate", "seed", "device", "step_delay_seconds",
             "save_every", "max_turns", "max_new_tokens", "max_prompt_tokens")
    for name in names:
        trainer.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.smoke:
        trainer.append("--smoke")
    else:
        benchmark_identity(args.benchmark_root)
        for name in ("benchmark_root", "policy_model", "domain", "reward_mode", "user_model", "user_base_url"):
            value = getattr(args, name)
            if name in {"benchmark_root", "policy_model"}:
                value = str(Path(value).resolve(strict=True))
            trainer.extend(["--" + name.replace("_", "-"), str(value)])
        if args.user_api_key_env:
            trainer.extend(["--user-api-key-env", args.user_api_key_env])
    relative_output = output.relative_to(root).as_posix()
    supervisor = [
        "--log", f"reports/runtime/{args.run}/trainer.stdout",
        "--trajectories", f"{relative_output}/trajectories",
        "--backend", args.backend, "--mode", "propose",
        "--every-steps", str(args.every_steps), "--every-seconds", str(args.every_seconds),
        "--poll-seconds", "0.2", "--agent-timeout", str(args.agent_timeout),
        "--max-calls", str(calls), "--export-atif",
    ]
    if args.meta_model:
        supervisor.extend(["--agent-model", args.meta_model])
    if args.backend == "opencode":
        if not args.meta_model or not args.meta_base_url:
            raise ValueError("OpenCode requires explicit --meta-model and --meta-base-url")
        command = json.loads(args.opencode_command)
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError("--opencode-command must be a nonempty literal argv JSON array")
        supervisor.extend(["--opencode-base-url", service_url(args.meta_base_url),
                           "--opencode-command", json.dumps(command),
                           "--opencode-context", str(args.meta_context),
                           "--opencode-output", str(args.meta_output)])
        if args.meta_api_key_env:
            supervisor.extend(["--opencode-api-key-env", args.meta_api_key_env])
    elif args.meta_base_url or args.meta_api_key_env:
        raise ValueError("Meta endpoint/key settings apply only to the OpenCode backend")
    kind = "synthetic CPU optimizer smoke" if args.smoke else "native τ³-bench text train-split REINFORCE demo"
    return {
        "version": 1, "root": str(root), "run": args.run,
        "trainer_command": trainer,
        "contract": {
            "objective": f"Observe a bounded {kind}; discover code components and review actual training trajectories.",
            "budget": f"{args.steps} optimizer steps, {args.episodes_per_step} episodes per step; at most {calls} agent calls.",
            "evaluation_protocol": ("Only newly generated synthetic observations/rewards; these are not τ³ results."
                                    if args.smoke else "Only official train tasks and explicit ACTION/ENV training proxy reward; no held-out benchmark selection."),
            "checkpoint_selection": "Retain the final planned optimizer checkpoint; no evaluation-based selection.",
            "guardrails": ["Read-only supervision; propose changes without executing them.",
                           "Do not install dependencies, download weights or start inference servers.",
                           "Do not read hidden task/evaluation criteria or held-out feedback for steering.",
                           "Synthetic smoke performance must never be reported as benchmark performance."],
            "allowed_feedback": [f"reports/runtime/{args.run}/trainer.stdout",
                                 f"{relative_output}/manifest.json", f"{relative_output}/trajectories",
                                 "Current example source and saved meta-agent experience."],
            "excluded_feedback": ["Benchmark test/base task outcomes and held-out trajectories.",
                                  "Hidden evaluator criteria, private user prompts, secrets and .env files."],
            "allowed_operations": [],
            "records": {"change_log": f"reports/tau3_minimal/{args.run}_change_log.md",
                        "intervention_map": f"reports/tau3_minimal/{args.run}_intervention_map.md",
                        "experience": "reports/meta_agent_experience.md"},
        },
        "supervisor_args": supervisor,
        "trainer_timeout_seconds": args.trainer_timeout,
        "poll_seconds": 0.2, "final_timeout_seconds": args.final_timeout,
        "failure_policy": "stop_owned_trainer",
    }


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = launch_config(args)
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            stream.write(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        print(str(target))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"[PREPARE] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
