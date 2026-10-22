"""Fail-closed bridge to native τ³ user simulation; no custom flow user."""

from __future__ import annotations

import asyncio
import math

from examples.tau3_minimal.tau_adapter import Tau3Adapter


class Tau2Env:
    """Expose only public observations to verl and keep hidden tasks inside Gym."""

    def __init__(self, *, benchmark_root, domain, task_id=None, id=None,
                 task_split="train", reward_mode="action", user_model=None,
                 user_base_url=None, user_api_key_env=None, max_turns=16,
                 model_name=None, solo_mode=False, **unknown):
        if unknown:
            raise ValueError(f"Unsupported native environment settings: {sorted(unknown)}")
        if task_split != "train" or solo_mode:
            raise ValueError("Training requires the official train split and native user simulation")
        if not user_model or not user_base_url:
            raise ValueError("Explicit user_model and user_base_url are required")
        if not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.adapter = Tau3Adapter(
            benchmark_root=benchmark_root, domain=domain, reward_mode=reward_mode,
            user_model=user_model, user_base_url=user_base_url,
            user_api_key_env=user_api_key_env, max_turns=max_turns,
        )
        selected = str(task_id if task_id is not None else id)
        task = next((task for task in self.adapter.tasks if str(task.id) == selected), None)
        if task is None:
            raise ValueError(f"Task {selected!r} is not in the official {domain} train split")
        self.task_id = selected
        options = dict(self.adapter.options)
        # Return only the newest observation: verl already retains the history.
        options["all_messages_as_observation"] = False
        self.tau2_env = self.adapter.env_type(
            selected_task=task, task_id=selected, **options,
        )
        self.max_turns = max_turns
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_complete = False
        self.training_cutoff = False
        self._step_lock = asyncio.Lock()

    def reset(self):
        observation, _ = self.tau2_env.reset()
        if self.tau2_env._simulation_done.is_set():
            self.tau2_env._get_reward()
            raise RuntimeError("Native episode ended before the first policy action")
        return {"feedback": observation}, self._public_info()

    def _public_info(self):
        return {"task_id": self.task_id, "split": "train", "training_cutoff": self.training_cutoff,
                "reward_proxy": self.adapter.options["reward_mode"], "benchmark_score": False}

    def step(self, action):
        if self.episode_complete:
            raise RuntimeError("Cannot step a completed native episode")
        self.step_count += 1
        observation, reward, terminated, truncated, _ = self.tau2_env.step(action)
        if observation.startswith("Invalid action with error:"):
            raise ValueError("Native Gym rejected the policy action format")
        if self.tau2_env.native_error is not None:
            raise RuntimeError("Native user/environment service failed") from self.tau2_env.native_error
        if terminated:
            reason = self.tau2_env._simulation_run.termination_reason.value
            self.training_cutoff = reason in {"max_steps", "too_many_errors"}
            reward = 0.0 if self.training_cutoff else reward
            self.episode_complete = True
            self.total_reward = float(reward)
        elif self.step_count >= self.max_turns:
            self.calculate_reward()
            reward, terminated, truncated = self.total_reward, True, True
        if not math.isfinite(float(reward)):
            raise ValueError("Native evaluator returned a nonfinite reward")
        return {"feedback": observation}, float(reward), terminated, truncated, self._public_info()

    async def step_async(self, action):
        # Native Gym waits for its own orchestrator thread. Keep concurrent verl
        # episodes responsive while a user model request is in flight.
        async with self._step_lock:
            return await asyncio.to_thread(self.step, action)

    def calculate_reward(self):
        if not self.episode_complete:
            self.training_cutoff = True
            self.tau2_env.step('{"name":"done","arguments":{}}')
            self.episode_complete = True
        reward, _ = self.tau2_env._get_reward()
        self.total_reward = 0.0 if self.training_cutoff else float(reward)
        return self.total_reward

    def close(self):
        thread = getattr(self.tau2_env, "_orchestrator_thread", None)
        if thread and thread.is_alive():
            agent = getattr(self.tau2_env, "_agent", None)
            if agent is not None and agent.is_agent_turn:
                from tau2.data_model.message import AssistantMessage
                agent.set_action(AssistantMessage(role="assistant", content="###STOP###"))
            thread.join(timeout=2)
        if thread and thread.is_alive():
            raise RuntimeError("Native episode left a running user/orchestrator thread")


def create_environment(**kwargs):
    env = Tau2Env(**kwargs)
    try:
        observation, _ = env.reset()
        return env, observation["feedback"]
    except BaseException:
        env.close()
        raise
