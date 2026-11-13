# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
from functools import partial

import ray

from verl import DataProto
from verl.utils.reward_score import default_compute_score


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """
    from verl.workers.reward_manager import get_reward_manager_cls

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Try to get a custom reward function based on the configuration
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


# Reserved key under which NaiveRewardManager's scalar per-gym metrics travel
# inside reward_extra_infos_dict (see compute_reward / RayPPOTrainer.fit).
GYM_METRICS_KEY = "_gym_reward_metrics"


def compute_reward(data: DataProto, reward_fn, gamma=0.9, use_intrinsic=False, intrinsic_config=None, actor_model=None, tokenizer=None, current_epoch=None, intrinsic_rewards=None, place_per_turn_rewards=False):
    """
    Compute reward for a batch of data.

    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the extrinsic reward.
        gamma: Discount factor for reward computation.
        use_intrinsic: Whether to use intrinsic rewards (Info-GRPO).
        intrinsic_config: Configuration dict for intrinsic reward computation.
        actor_model: Actor model for intrinsic reward computation.
        tokenizer: Tokenizer for intrinsic reward computation.
        current_epoch: Current training epoch (for intrinsic reward decay).

    Returns:
        reward_tensor: Combined reward tensor for backward compatibility
        reward_extra_infos_dict: Dict with:
            - extrinsic_reward: Extrinsic reward tensor (if intrinsic enabled)
            - intrinsic_reward: Intrinsic reward tensor (if intrinsic enabled)
            - Other metrics
    """
    # Compute extrinsic reward
    try:
        reward_result = reward_fn(data, return_dict=True, gamma=gamma, place_per_turn_rewards=place_per_turn_rewards)
        extrinsic_reward = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
        # Scalar per-gym metrics ride along under a reserved key: compute_reward
        # returns a 2-tuple, and the trainer pops this before treating the rest
        # of the dict as per-sample lists.
        if reward_result.get("reward_metrics"):
            reward_extra_infos_dict[GYM_METRICS_KEY] = reward_result["reward_metrics"]
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        extrinsic_reward = reward_fn(data)
        reward_extra_infos_dict = {}

    # Initialize combined reward
    reward_tensor = extrinsic_reward

    # Add intrinsic rewards if enabled
    print(f"[DEBUG compute_reward] use_intrinsic={use_intrinsic}, intrinsic_rewards={'None' if intrinsic_rewards is None else 'OK'}, actor_model={'None' if actor_model is None else 'OK'}, tokenizer={'None' if tokenizer is None else 'OK'}")  # Commented out to reduce noise

    # If intrinsic_rewards is already computed on worker side, use it
    # Otherwise, compute it here if actor_model is available (fallback for non-distributed case)
    if use_intrinsic and intrinsic_rewards is None:
        if actor_model is not None and tokenizer is not None:
            # print(f"[DEBUG compute_reward] Computing intrinsic rewards locally (fallback)...")  # Commented out to reduce noise
            from verl.trainer.ppo.intrinsic_reward import compute_intrinsic_rewards

            if intrinsic_config is None:
                intrinsic_config = {}

            try:
                # Compute intrinsic rewards (already token-level, per-turn)
                debug_intrinsic = intrinsic_config.get("debug", False)
                # print(f"[DEBUG compute_reward] debug_intrinsic={debug_intrinsic}")  # Commented out to reduce noise
                intrinsic_rewards = compute_intrinsic_rewards(
                    batch=data,
                    model=actor_model,
                    tokenizer=tokenizer,
                    config=intrinsic_config,
                    debug=debug_intrinsic,
                )
                # print(f"[DEBUG compute_reward] Intrinsic rewards computed locally: shape={intrinsic_rewards.shape}, sum={intrinsic_rewards.sum().item():.6f}, mean={intrinsic_rewards.mean().item():.6f}")  # Commented out to reduce noise
            except Exception as e:
                print(f"Warning: Failed to compute intrinsic rewards locally: {e}")
                import traceback
                traceback.print_exc()
                intrinsic_rewards = None

    if use_intrinsic and intrinsic_rewards is not None:
        # Calculate effective intrinsic weight with decay
        if intrinsic_config is None:
            intrinsic_config = {}

        # Align intrinsic reward length with extrinsic reward length (response tokens)
        if intrinsic_rewards.dim() == extrinsic_reward.dim() == 2:
            if intrinsic_rewards.shape[1] != extrinsic_reward.shape[1]:
                intrinsic_rewards = intrinsic_rewards[:, -extrinsic_reward.shape[1]:]

        base_weight = intrinsic_config.get("intrinsic_weight", 0.1)
        decay_rate = intrinsic_config.get("intrinsic_decay_rate", 0.0)

        if current_epoch is not None and decay_rate > 0:
            import math
            intrinsic_weight = base_weight * math.exp(-decay_rate * current_epoch)
        else:
            intrinsic_weight = base_weight

        # IMPORTANT: Store extrinsic and intrinsic separately
        # This allows proper credit assignment in GRPO
        reward_extra_infos_dict["extrinsic_reward"] = extrinsic_reward
        reward_extra_infos_dict["intrinsic_reward"] = intrinsic_rewards
        reward_extra_infos_dict["intrinsic_weight"] = intrinsic_weight

        # For backward compatibility, also compute combined reward
        combined_reward = extrinsic_reward + intrinsic_weight * intrinsic_rewards

        # Add stats
        reward_extra_infos_dict["intrinsic_reward_mean"] = intrinsic_rewards.mean().item()
        reward_extra_infos_dict["intrinsic_reward_std"] = intrinsic_rewards.std().item()
        reward_extra_infos_dict["extrinsic_reward_mean"] = extrinsic_reward.mean().item()
        reward_extra_infos_dict["combined_reward_mean"] = combined_reward.mean().item()

        # Return combined for backward compatibility
        # But the separate rewards are in extra_infos for GRPO to use
        reward_tensor = combined_reward
    else:
        # print(f"[DEBUG compute_reward] Skipping intrinsic rewards: use_intrinsic={use_intrinsic}, actor_model={'None' if actor_model is None else 'OK'}, tokenizer={'None' if tokenizer is None else 'OK'}")  # Commented out to reduce noise
        pass

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
