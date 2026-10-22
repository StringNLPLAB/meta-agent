#!/usr/bin/env python3
"""Bounded real local-policy training, or an explicitly synthetic optimizer smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from tau_adapter import Tau3Adapter, file_hash, service_url


def write_json(path, value):
    """Publish a complete record atomically for concurrent supervisor reads."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def state_hash(model):
    """Hash parameters rather than nondeterministic checkpoint container metadata."""
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return result.hexdigest()


class LocalPolicy:
    """Sample unmodified categorical LM probabilities and recompute their gradients."""

    def __init__(self, path, *, device="cpu", max_new_tokens=64, max_prompt_tokens=8192):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        path = Path(path).resolve(strict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
        ).to(device)
        self.model.eval()  # Disable dropout while retaining autograd for policy gradients.
        self.device = torch.device(device)
        self.max_new_tokens, self.max_prompt_tokens = max_new_tokens, max_prompt_tokens
        self.path = str(path)
        self.generation_config = GenerationConfig(
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                          else self.tokenizer.eos_token_id),
        )

    def sample(self, system, observation):
        import torch
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": observation}]
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            text = f"system: {system}\nuser: {observation}\nassistant:"
            encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        length = encoded["input_ids"].shape[1]
        context = getattr(self.model.config, "max_position_embeddings", None)
        if length > self.max_prompt_tokens or (context and length + self.max_new_tokens > context):
            raise ValueError("Policy context exceeded; no silent prompt truncation is allowed")
        with torch.no_grad():
            sequence = self.model.generate(
                **encoded, max_new_tokens=self.max_new_tokens,
                generation_config=self.generation_config,
            )[0]
        tokens = sequence[length:]
        if not tokens.numel():
            raise ValueError("Policy generated no action tokens")
        return {"text": self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "input_ids": sequence.detach(), "prompt_length": length,
                "generated_token_ids": tokens.tolist()}

    def log_probability(self, sample):
        import torch
        sequence = sample["input_ids"]
        logits = self.model(input_ids=sequence.unsqueeze(0)).logits[0, :-1].float()
        selected = torch.log_softmax(logits, dim=-1).gather(1, sequence[1:, None]).squeeze(1)
        return selected[sample["prompt_length"] - 1:].sum()


def smoke_rollout(model, index):
    """A two-action synthetic environment; never presented as τ³ task performance."""
    import torch
    target = index % 2
    observation = torch.nn.functional.one_hot(torch.tensor(target), 2).float()
    distribution = torch.distributions.Categorical(logits=model(observation))
    action = distribution.sample()
    reward = float(action.item() == target)
    return distribution.log_prob(action), reward, {
        "environment": "synthetic_two_action", "observation": observation.tolist(),
        "action": int(action), "reward": reward,
        "messages": [{"role": "user", "content": f"Synthetic target bit: {target}"},
                     {"role": "assistant", "content": str(int(action))},
                     {"role": "tool", "content": f"synthetic reward={reward}"}],
    }


def parser(require_output=True):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", required=require_output,
                        help="New directory; existing runs cannot be overwritten")
    result.add_argument("--run", required=True)
    result.add_argument("--smoke", action="store_true", help="Synthetic CPU optimizer smoke, NOT benchmark training")
    result.add_argument("--steps", type=int, default=6)
    result.add_argument("--episodes-per-step", type=int, default=8)
    result.add_argument("--learning-rate", type=float, default=1e-5)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--device", default="cpu")
    result.add_argument("--step-delay-seconds", type=float, default=0)
    result.add_argument("--save-every", type=int, default=1)
    result.add_argument("--benchmark-root")
    result.add_argument("--policy-model", help="Existing local Hugging Face directory; no downloads or remote code")
    result.add_argument("--domain", choices=("retail", "airline", "telecom"), default="retail")
    result.add_argument("--reward-mode", choices=("action", "env"), default="env")
    result.add_argument("--user-model", help="Explicit LiteLLM user ID, e.g. openai/qwen-user")
    result.add_argument("--user-base-url", help="User simulator endpoint, separate from the meta-agent endpoint")
    result.add_argument("--user-api-key-env", help="Environment variable name, never the API key itself")
    result.add_argument("--max-turns", type=int, default=8)
    result.add_argument("--max-new-tokens", type=int, default=64)
    result.add_argument("--max-prompt-tokens", type=int, default=8192)
    return result


def validate(args, check_python=True):
    for key in ("steps", "episodes_per_step", "save_every", "max_turns", "max_new_tokens", "max_prompt_tokens"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if not math.isfinite(args.step_delay_seconds) or args.step_delay_seconds < 0:
        raise ValueError("--step-delay-seconds must be finite and nonnegative")
    if not args.run.strip():
        raise ValueError("--run must be nonempty")
    if args.smoke:
        if args.benchmark_root or args.policy_model or args.user_model or args.user_base_url or args.user_api_key_env:
            raise ValueError("Synthetic smoke must not be combined with real benchmark/model service settings")
        if args.device != "cpu":
            raise ValueError("Synthetic smoke uses CPU explicitly")
    else:
        for key in ("benchmark_root", "policy_model", "user_model", "user_base_url"):
            if not getattr(args, key):
                raise ValueError(f"Real training requires --{key.replace('_', '-')}")
        if not args.user_model.startswith("openai/"):
            raise ValueError("Local user simulator ID must use the explicit openai/ provider prefix")
        service_url(args.user_base_url)
        if not Path(args.policy_model).is_dir():
            raise ValueError("Policy model must already exist as a local directory")
        if check_python and (sys.version_info < (3, 12) or sys.version_info >= (3, 14)):
            raise ValueError("τ³-bench 1.0.1 requires Python >=3.12,<3.14")


def train(args):
    import torch
    args._created_output = False
    validate(args)
    torch.manual_seed(args.seed)
    if args.smoke:
        model = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(model.weight)
        policy, adapter = None, None
    else:
        adapter = Tau3Adapter(
            benchmark_root=args.benchmark_root, domain=args.domain, reward_mode=args.reward_mode,
            user_model=args.user_model, user_base_url=args.user_base_url,
            user_api_key_env=args.user_api_key_env, max_turns=args.max_turns,
        )
        policy = LocalPolicy(args.policy_model, device=args.device, max_new_tokens=args.max_new_tokens,
                             max_prompt_tokens=args.max_prompt_tokens)
        model = policy.model
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args._created_output = True
    trajectories, checkpoints = output / "trajectories", output / "checkpoints"
    trajectories.mkdir()
    checkpoints.mkdir()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    initial_hash = state_hash(model)
    manifest = {
        "version": 1, "run": args.run,
        "kind": "synthetic_optimizer_smoke" if args.smoke else "tau3_text_training",
        "scope": "synthetic_two_action" if args.smoke else "text_half_duplex_train_split",
        "objective": "REINFORCE, zero baseline, sum of generated-action token log probabilities",
        "benchmark_score": False, "reward_mode": "synthetic" if args.smoke else args.reward_mode,
        "benchmark": None if args.smoke else adapter.identity,
        "policy_model": "synthetic_torch_linear" if args.smoke else str(Path(args.policy_model).resolve()),
        "user_model": None if args.smoke else args.user_model,
        "parameters": {
            "device": args.device, "max_turns": args.max_turns,
            "max_new_tokens": args.max_new_tokens, "max_prompt_tokens": args.max_prompt_tokens,
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "dropout": False},
            "user_base_url": None if args.smoke else service_url(args.user_base_url),
            "user_api_key_env": None if args.smoke else args.user_api_key_env,
            "user_temperature": None if args.smoke else 0.0,
            "user_max_tokens": None if args.smoke else 512,
            "user_timeout_seconds": None if args.smoke else 30,
            "save_every": args.save_every, "step_delay_seconds": args.step_delay_seconds,
        },
        "reward_metadata": {"proxy": "synthetic" if args.smoke else args.reward_mode,
                            "benchmark_score": False, "truncated_episode_reward": 0.0},
        "meta_model": "configured separately by the runtime supervisor",
        "initial_parameter_sha256": initial_hash, "parameter_sha256": initial_hash,
        "optimizer_steps": 0, "planned_steps": args.steps, "seed": args.seed,
        "seed_scope": "PyTorch policy RNG only; native user/service determinism is not guaranteed",
        "torch_version": str(torch.__version__),
        "episodes_per_step": args.episodes_per_step, "learning_rate": args.learning_rate,
        "code_sha256": {name: file_hash(Path(__file__).with_name(name))
                        for name in ("train.py", "tau_adapter.py")},
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "checkpoint": None, "parent_checkpoint": None,
    }
    write_json(output / "manifest.json", manifest)
    previous_hash, previous_checkpoint, checkpoint_parent = initial_hash, None, None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rewards, loss_value = [], 0.0
        for episode in range(args.episodes_per_step):
            index = (step - 1) * args.episodes_per_step + episode
            if args.smoke:
                log_probability, reward, record = smoke_rollout(model, index)
                samples = []
                loss = -reward * log_probability / args.episodes_per_step
                loss.backward()
                loss_value += float(loss.detach())
            else:
                samples, reward, record = adapter.rollout(policy, index)
                if not samples:
                    raise RuntimeError("Episode contained no sampled policy actions")
                for sample in samples:
                    loss = -reward * policy.log_probability(sample) / args.episodes_per_step
                    loss.backward()
                    loss_value += float(loss.detach())
            if not math.isfinite(reward):
                raise ValueError("Nonfinite environment reward")
            record.update({"version": 1, "run": args.run, "kind": manifest["kind"],
                           "stage": "rollout_before_update",
                           "optimizer_step": step, "behavior_parameter_sha256": previous_hash,
                           "episode": episode, "reward": reward,
                           "sampled_actions": [{"text": sample["text"],
                                                "prompt_length": sample["prompt_length"],
                                                "generated_token_ids": sample["generated_token_ids"]}
                                               for sample in samples]})
            write_json(trajectories / f"step_{step:06d}_episode_{episode:03d}.json", record)
            rewards.append(reward)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(gradient_norm) or not math.isfinite(loss_value):
            raise ValueError("Nonfinite policy loss/gradient; refusing optimizer update")
        optimizer.step()
        parameter_hash = state_hash(model)
        checkpoint = previous_checkpoint
        if step % args.save_every == 0 or step == args.steps:
            checkpoint_parent = previous_checkpoint
            checkpoint = f"checkpoints/step_{step:06d}.pt"
            torch.save({"version": 1, "run": args.run, "kind": manifest["kind"], "optimizer_step": step,
                        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "parameter_sha256": parameter_hash, "behavior_parameter_sha256": previous_hash,
                        "parent_checkpoint": previous_checkpoint,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []},
                       output / checkpoint)
        manifest.update(optimizer_steps=step, parameter_sha256=parameter_hash,
                        parent_checkpoint=checkpoint_parent, checkpoint=checkpoint,
                        last_mean_reward=sum(rewards) / len(rewards), last_gradient_norm=gradient_norm,
                        status="complete" if step == args.steps else "running")
        write_json(output / "manifest.json", manifest)
        print(f"step:{step} - " + json.dumps({"kind": manifest["kind"], "mean_reward": manifest["last_mean_reward"],
              "loss": loss_value, "gradient_norm": gradient_norm,
              "parameter_changed": parameter_hash != previous_hash, "optimizer_steps": step}), flush=True)
        previous_hash, previous_checkpoint = parameter_hash, checkpoint
        if args.step_delay_seconds:
            time.sleep(args.step_delay_seconds)
    return manifest


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        train(args)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        manifest_path = Path(args.output_dir).resolve() / "manifest.json"
        if getattr(args, "_created_output", False) and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                manifest.update(status="failed", error_type=type(error).__name__,
                                failed_at=datetime.now(timezone.utc).isoformat())
                write_json(manifest_path, manifest)
            except (OSError, ValueError):
                pass  # Preserve the original training failure.
        print(f"[TRAIN] {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
