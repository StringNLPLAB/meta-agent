"""New thin integration with an external τ³-bench 1.0.1 checkout.

No benchmark code or task data is vendored here. Hidden task criteria remain
inside the native environment/evaluator, never in the policy observations.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tomllib
from urllib.parse import urlsplit


SOURCE_FILES = (
    "pyproject.toml",
    "src/tau2/gym/gym_agent.py",
    "src/tau2/registry.py",
    "src/tau2/user/user_simulator.py",
    "src/tau2/evaluator/evaluator.py",
)


def file_hash(path):
    """Return a source/artifact SHA-256 without loading its entire contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def service_url(value):
    """Validate a plain HTTP API URL; credentials belong in environment variables."""
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Service URL must be HTTP(S), without credentials, query or fragment")
    return value.rstrip("/")


def benchmark_identity(root):
    """Identify the inspected native API version and hash the actual source files."""
    root = Path(root).resolve(strict=True)
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    if project.get("name") != "tau2" or project.get("version") != "1.0.1":
        raise ValueError("This adapter targets the tau2 package from τ³-bench 1.0.1 exactly")
    return {
        "name": "tau2", "version": "1.0.1", "scope": "text_half_duplex",
        "sources": {name: file_hash(root / name) for name in SOURCE_FILES},
    }


def strict_train_tasks(registry, domain):
    """Load only the official train split, checking its declared IDs independently."""
    split_loader = registry.get_task_splits_loader(domain)
    if not callable(split_loader):
        raise ValueError(f"{domain}: official split loader is unavailable")
    splits = split_loader()
    if not isinstance(splits, dict):
        raise ValueError("Official split loader did not return a split mapping")
    train_ids = splits.get("train")
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError(f"{domain}: official nonempty train split is required")
    declared = [str(item) for item in train_ids]
    if len(set(declared)) != len(declared):
        raise ValueError("Official train split has duplicate task IDs")
    tasks = registry.get_tasks_loader(domain)(task_split_name="train")
    actual = [str(task.id) for task in tasks]
    if len(actual) != len(set(actual)) or set(actual) != set(declared):
        raise ValueError("Native loader did not return exactly the official train split")
    return tasks


def proxy_environment_class(base, evaluator, evaluation_types):
    """Specialize the native Gym API without replacing its user or orchestration."""
    class TrainEnvironment(base):
        def __init__(self, *, selected_task, reward_mode, **kwargs):
            self.selected_task = selected_task
            self.reward_mode = reward_mode
            self.native_error = None
            super().__init__(**kwargs)

        def _get_task(self):
            # The native default reloads the complete base split on every call.
            return self.selected_task

        def _get_reward(self):
            if self.native_error is not None:
                raise RuntimeError("Native benchmark rollout failed") from self.native_error
            if self._simulation_run is None:
                if self._simulation_done.is_set():
                    raise RuntimeError("Native rollout ended without a simulation record")
                return 0.0, "{}"
            reason = self._simulation_run.termination_reason.value
            if reason not in {"agent_stop", "user_stop", "max_steps", "too_many_errors"}:
                raise RuntimeError(f"Native benchmark terminated with {reason}; refusing policy update")
            result = evaluator(
                simulation=self._simulation_run, task=self.selected_task,
                evaluation_type=getattr(evaluation_types, self.reward_mode.upper()),
                solo_mode=False, domain=self.domain,
            )
            reward = float(result.reward)
            if not math.isfinite(reward):
                raise ValueError("Benchmark evaluator returned a nonfinite reward")
            return reward, result.model_dump_json()

        def _run_orchestrator(self):
            # Native Gym logs and swallows these errors; training must fail instead.
            try:
                self._simulation_run = self._orchestrator.run()
            except Exception as error:
                self.native_error = error
            finally:
                self._simulation_done.set()

    return TrainEnvironment


def public_messages(simulation):
    """Export visible conversation fields; omit raw_data, prompts and hidden tasks."""
    result = []
    for message in simulation.messages:
        item = {"role": message.role, "content": message.content}
        calls = getattr(message, "tool_calls", None)
        if calls:
            item["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments,
                 "requestor": getattr(call, "requestor", "assistant")}
                for call in calls
            ]
        for key in ("id", "requestor", "error"):
            value = getattr(message, key, None)
            if value is not None:
                item[key] = value
        result.append(item)
    return result


class Tau3Adapter:
    """Expose native text episodes with an explicit local training reward proxy."""

    def __init__(self, *, benchmark_root, domain, reward_mode, user_model,
                 user_base_url, user_api_key_env=None, max_turns=8):
        if domain not in {"retail", "airline", "telecom"} or reward_mode not in {"env", "action"}:
            raise ValueError("Select a supported text domain and ENV/ACTION training reward proxy")
        root = Path(benchmark_root).resolve(strict=True)
        self.identity = benchmark_identity(root)
        data = root / "data"
        if not data.is_dir():
            raise ValueError("Benchmark checkout must contain its official data directory")
        os.environ["TAU2_DATA_DIR"] = str(data)
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        os.environ["LOGURU_LEVEL"] = "WARNING"
        sys.path.insert(0, str(root / "src"))
        package = importlib.import_module("tau2")
        location = Path(package.__file__).resolve()
        if not location.is_relative_to(root / "src" / "tau2"):
            raise ValueError("A different tau2 package is already imported")
        registry = importlib.import_module("tau2.registry").registry
        gym_module = importlib.import_module("tau2.gym.gym_agent")
        evaluator = importlib.import_module("tau2.evaluator.evaluator")
        self.tasks = strict_train_tasks(registry, domain)
        split_path = data / "tau2" / "domains" / domain / "split_tasks.json"
        self.identity["train_split_sha256"] = file_hash(split_path)
        self.identity["train_task_ids"] = [str(task.id) for task in self.tasks]
        domain_data = data / "tau2" / "domains" / domain
        domain_source = root / "src" / "tau2" / "domains" / domain
        self.identity["domain_data_sha256"] = {
            path.relative_to(root).as_posix(): file_hash(path)
            for path in sorted(domain_data.rglob("*")) if path.is_file()
        }
        self.identity["domain_source_sha256"] = {
            path.relative_to(root).as_posix(): file_hash(path)
            for path in sorted(domain_source.rglob("*.py")) if path.is_file()
        }
        self.env_type = proxy_environment_class(
            gym_module.AgentGymEnv, evaluator.evaluate_simulation, evaluator.EvaluationType,
        )
        if user_api_key_env:
            key = os.environ.get(user_api_key_env)
            if not key:
                raise ValueError("Declared user API key environment variable is empty")
            os.environ["OPENAI_API_KEY"] = key
        else:
            # OpenAI-compatible local servers often require a syntactic key.
            os.environ["OPENAI_API_KEY"] = "unused"
        self.options = {
            "domain": domain, "reward_mode": reward_mode, "solo_mode": False,
            "user_llm": user_model,
            "user_llm_args": {"api_base": service_url(user_base_url), "temperature": 0.0,
                              "max_tokens": 512, "timeout": 30, "num_retries": 0},
            "max_steps": max_turns * 2 + 2, "all_messages_as_observation": True,
        }
        self.max_turns = max_turns

    def rollout(self, policy, episode_index):
        """Run one real episode; only visible policy/tools/messages reach the policy."""
        task = self.tasks[episode_index % len(self.tasks)]
        env = self.env_type(selected_task=task, task_id=str(task.id), **self.options)
        samples, turns = [], []
        try:
            observation, info = env.reset()
            # Never serialize/pass the whole info: it contains the complete task.
            system = (info["policy"] + "\n\nAvailable tools (JSON schemas):\n"
                      + json.dumps([tool.openai_schema for tool in info["tools"]])
                      + '\nRespond with a plain message or one JSON object '
                        '{"name":"TOOL","arguments":{...}}. '
                        'Use {"name":"done","arguments":{}} to finish.')
            terminated = env._simulation_done.is_set()
            if terminated:
                env._get_reward()  # Distinguish startup failure from an ordinary zero reward.
                raise RuntimeError("Native episode ended before the first policy action")
            for _ in range(self.max_turns):
                sample = policy.sample(system, observation)
                samples.append(sample)
                action = sample["text"]
                if not action:
                    raise ValueError("Policy produced an empty decoded action; refusing an untracked parse failure")
                next_observation, reward, terminated, _, _ = env.step(action)
                if next_observation.startswith("Invalid action with error:"):
                    raise ValueError("Native Gym rejected the policy action format")
                turns.append({"observation": observation, "action": action,
                              "next_observation": next_observation})
                observation = next_observation
                if terminated:
                    break
            controller_cutoff = not terminated
            if controller_cutoff:
                env.step('{"name":"done","arguments":{}}')
            if env.native_error is not None or env._simulation_run is None:
                raise RuntimeError("Native episode did not complete successfully") from env.native_error
            reward, _ = env._get_reward()
            reason = env._simulation_run.termination_reason.value
            cutoff = controller_cutoff or reason in {"max_steps", "too_many_errors"}
            # Controller-imposed truncation does not masquerade as a normal success.
            if cutoff:
                reward = 0.0
            return samples, float(reward), {
                "task_id": str(task.id), "split": "train", "reward_mode": self.options["reward_mode"],
                "training_cutoff": cutoff, "turns": turns,
                "policy_context": system,
                "reward_metadata": {"proxy": self.options["reward_mode"], "benchmark_score": False,
                                    "controller_cutoff": controller_cutoff},
                "messages": public_messages(env._simulation_run),
                "termination_reason": reason,
            }
        finally:
            thread = getattr(env, "_orchestrator_thread", None)
            if thread and thread.is_alive():
                agent = getattr(env, "_agent", None)
                if agent is not None and agent.is_agent_turn:
                    from tau2.data_model.message import AssistantMessage
                    agent.set_action(AssistantMessage(role="assistant", content="###STOP###"))
                thread.join(timeout=1)
            if thread and thread.is_alive():
                raise RuntimeError("Native episode left a running user/orchestrator thread")
