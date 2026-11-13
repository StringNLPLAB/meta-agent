# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards (for backward compatibility)
            - critic/external_rewards/mean, max, min: Statistics about external/outcome rewards
            - critic/intrinsic_rewards/mean, max, min: Statistics about intrinsic rewards (if available)
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - turns/mean, max, min, std: Statistics about number of interaction turns per conversation (if multi-turn data available)
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    # Compute average number of turns per conversation
    num_turns_per_sample = None
    if "turn_boundaries" in batch.batch:
        turn_boundaries = batch.batch["turn_boundaries"]  # (bs, response_length)
        # Count number of turns per sample (sum of 1s in each row)
        num_turns_per_sample = turn_boundaries.sum(dim=1).float()  # (bs,)

        # DEBUG: Compare with conversation_histories if available
        if "conversation_histories" in batch.non_tensor_batch:
            conversation_histories = batch.non_tensor_batch["conversation_histories"]
            unwrapped_histories = []
            for conv_hist_wrapper in conversation_histories:
                if isinstance(conv_hist_wrapper, (list, np.ndarray)) and len(conv_hist_wrapper) > 0:
                    if isinstance(conv_hist_wrapper[0], (list, np.ndarray)) and len(conv_hist_wrapper[0]) > 0:
                        if isinstance(conv_hist_wrapper[0][0], dict):
                            conv_hist = conv_hist_wrapper[0]
                        else:
                            conv_hist = conv_hist_wrapper
                    elif isinstance(conv_hist_wrapper[0], dict):
                        conv_hist = conv_hist_wrapper
                    else:
                        conv_hist = []
                elif isinstance(conv_hist_wrapper, dict):
                    conv_hist = [conv_hist_wrapper]
                else:
                    conv_hist = []
                # Ensure conv_hist is a list (not numpy array) to avoid truthiness issues
                if isinstance(conv_hist, np.ndarray):
                    conv_hist = conv_hist.tolist()
                unwrapped_histories.append(conv_hist)

            turns_from_hist = torch.tensor(
                [len(conv_hist) if len(conv_hist) > 0 else 0 for conv_hist in unwrapped_histories],
                dtype=torch.float32,
                device=num_turns_per_sample.device
            )

            # Check for mismatches
            mismatch_mask = torch.abs(num_turns_per_sample - turns_from_hist) > 0.5
            if mismatch_mask.any():
                mismatch_indices = torch.where(mismatch_mask)[0].tolist()
                import logging
                logger = logging.getLogger(__name__)
                # logger.warning(
                #     f"[TURN_COUNT_MISMATCH] Found {len(mismatch_indices)} samples with mismatched turn counts: "
                #     f"indices={mismatch_indices[:10]}, "
                #     f"turns_from_boundaries={num_turns_per_sample[mismatch_mask].tolist()[:10]}, "
                #     f"turns_from_hist={turns_from_hist[mismatch_mask].tolist()[:10]}"
                # )
                # Use conversation_histories as ground truth if there's a mismatch
                num_turns_per_sample = turns_from_hist
    elif "conversation_histories" in batch.non_tensor_batch:
        # Fallback: count turns from conversation_histories
        # NOTE: conversation_histories may be wrapped in an extra list dimension (e.g., [[turns]])
        # We need to unwrap it to get the actual conversation history list
        conversation_histories = batch.non_tensor_batch["conversation_histories"]
        unwrapped_histories = []
        for conv_hist_wrapper in conversation_histories:
            if isinstance(conv_hist_wrapper, (list, np.ndarray)) and len(conv_hist_wrapper) > 0:
                # Check if it's a nested structure like [[turns]]
                if isinstance(conv_hist_wrapper[0], (list, np.ndarray)) and len(conv_hist_wrapper[0]) > 0:
                    # It's wrapped: [[turns]] -> [turns]
                    if isinstance(conv_hist_wrapper[0][0], dict):
                        conv_hist = conv_hist_wrapper[0]
                    else:
                        conv_hist = conv_hist_wrapper
                elif isinstance(conv_hist_wrapper[0], dict):
                    # It's already [turns]
                    conv_hist = conv_hist_wrapper
                else:
                    conv_hist = []
            elif isinstance(conv_hist_wrapper, dict):
                # Single dict, wrap in list
                conv_hist = [conv_hist_wrapper]
            else:
                conv_hist = []
            # Ensure conv_hist is a list (not numpy array) to avoid truthiness issues
            if isinstance(conv_hist, np.ndarray):
                conv_hist = conv_hist.tolist()
            unwrapped_histories.append(conv_hist)

        num_turns_per_sample = torch.tensor(
            [len(conv_hist) if len(conv_hist) > 0 else 0 for conv_hist in unwrapped_histories],
            dtype=torch.float32,
            device=response_length.device
        )

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Action-token-only variants. In multiturn runs the critic is trained only on
    # agent action tokens (obs/tool tokens are masked out of vf_loss), so the
    # attention-masked explained variance above mixes in unsupervised obs values.
    # These metrics measure the fit on the tokens the critic actually trains on.
    vf_explained_var_action = None
    returns_mean_action = None
    values_mean_action = None
    if use_critic and "loss_mask" in batch.batch:
        action_response_mask = batch.batch["loss_mask"][:, -max_response_length:].bool()
        if action_response_mask.any():
            valid_returns_action = torch.masked_select(returns, action_response_mask)
            valid_values_action = torch.masked_select(values, action_response_mask)
            return_diff_var_action = torch.var(valid_returns_action - valid_values_action)
            return_var_action = torch.var(valid_returns_action)
            vf_explained_var_action = (1.0 - return_diff_var_action / (return_var_action + 1e-5)).detach().item()
            returns_mean_action = torch.mean(valid_returns_action).detach().item()
            values_mean_action = torch.mean(valid_values_action).detach().item()

    # Compute external rewards
    # If extrinsic_reward is available (Info-GRPO mode), use it; otherwise use token_level_rewards
    if "extrinsic_reward" in batch.batch:
        sequence_external_reward = batch.batch["extrinsic_reward"].sum(-1)
    else:
        sequence_external_reward = sequence_reward

    # Check if intrinsic rewards are available
    has_intrinsic_reward = "intrinsic_reward" in batch.batch
    turn_level_stats = {}  # Initialize outside if block
    if has_intrinsic_reward:
        sequence_intrinsic_reward = batch.batch["intrinsic_reward"].sum(-1)

        # Compute turn-level intrinsic rewards for analysis
        # Get max_turns from batch.meta_info if available, otherwise infer from conversation_histories or use default
        max_turns = batch.meta_info.get("max_turns", None)
        if max_turns is None:
            # Try to infer from conversation_histories
            if "conversation_histories" in batch.non_tensor_batch:
                conversation_histories = batch.non_tensor_batch["conversation_histories"]
                max_observed_turns = 0
                for conv_hist_wrapper in conversation_histories:
                    # Unwrap nested structure
                    if isinstance(conv_hist_wrapper, (list, np.ndarray)) and len(conv_hist_wrapper) > 0:
                        if isinstance(conv_hist_wrapper[0], (list, np.ndarray)) and len(conv_hist_wrapper[0]) > 0:
                            if isinstance(conv_hist_wrapper[0][0], dict):
                                conv_hist = conv_hist_wrapper[0]
                            else:
                                conv_hist = conv_hist_wrapper
                        elif isinstance(conv_hist_wrapper[0], dict):
                            conv_hist = conv_hist_wrapper
                        else:
                            conv_hist = []
                    elif isinstance(conv_hist_wrapper, dict):
                        conv_hist = [conv_hist_wrapper]
                    else:
                        conv_hist = []
                    max_observed_turns = max(max_observed_turns, len(conv_hist))
                # Use max_observed_turns but cap at a reasonable limit to avoid excessive metrics
                # Different training configs have different max_turns (e.g., 16, 50, etc.)
                # Cap at 100 to handle edge cases without creating too many metrics
                # If inferred value is reasonable (<=100), use it; otherwise cap at 100
                if max_observed_turns > 100:
                    # Data might have issues, cap at reasonable limit
                    max_turns = 100
                else:
                    # Use inferred value (could be 16, 50, or any reasonable value)
                    max_turns = max_observed_turns if max_observed_turns > 0 else 16
            else:
                max_turns = 16  # Default from training script

        # Import here to avoid circular imports
        from verl.trainer.ppo.intrinsic_reward import aggregate_intrinsic_reward_by_turn

        try:
            turn_level_intrinsic_rewards = aggregate_intrinsic_reward_by_turn(
                batch=batch,
                intrinsic_rewards=batch.batch["intrinsic_reward"],
                max_turns=max_turns,
            )  # (batch_size, max_turns)

            # Compute statistics for each turn position
            # Only consider turns that actually exist (non-zero rewards or valid turns)
            # For each turn position, compute mean/max/min across all trajectories that have that turn
            for turn_idx in range(max_turns):
                turn_rewards = turn_level_intrinsic_rewards[:, turn_idx]  # (batch_size,)

                # Only consider trajectories that have this turn (non-zero or we check conversation_histories)
                # For now, we'll compute stats on all values (including zeros for non-existent turns)
                # But we can also compute stats only on non-zero values
                turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/mean"] = torch.mean(turn_rewards).detach().item()
                turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/max"] = torch.max(turn_rewards).detach().item()
                turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/min"] = torch.min(turn_rewards).detach().item()
                turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/std"] = torch.std(turn_rewards).detach().item()

                # Also compute stats only on non-zero values (actual turns)
                non_zero_mask = turn_rewards != 0
                if non_zero_mask.any():
                    non_zero_rewards = turn_rewards[non_zero_mask]
                    turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/mean_nonzero"] = torch.mean(non_zero_rewards).detach().item()
                    # Fix: std requires at least 2 samples, otherwise set to 0
                    if non_zero_mask.sum().item() > 1:
                        turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/std_nonzero"] = torch.std(non_zero_rewards).detach().item()
                    else:
                        turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/std_nonzero"] = 0.0
                    turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/count"] = non_zero_mask.sum().item()
                else:
                    turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/mean_nonzero"] = 0.0
                    turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/std_nonzero"] = 0.0
                    turn_level_stats[f"critic/intrinsic_reward_per_turn/turn_{turn_idx}/count"] = 0

            # Compute trend: compare early turns vs late turns
            # Early turns: first 1/3, Late turns: last 1/3
            early_turn_end = max_turns // 3
            late_turn_start = max_turns - max_turns // 3

            early_turn_rewards = turn_level_intrinsic_rewards[:, :early_turn_end].sum(dim=1)  # (batch_size,)
            late_turn_rewards = turn_level_intrinsic_rewards[:, late_turn_start:].sum(dim=1)  # (batch_size,)

            early_turns_mean = torch.mean(early_turn_rewards).detach().item()
            late_turns_mean = torch.mean(late_turn_rewards).detach().item()

            turn_level_stats["critic/intrinsic_reward_per_turn/early_turns_mean"] = early_turns_mean
            turn_level_stats["critic/intrinsic_reward_per_turn/late_turns_mean"] = late_turns_mean

            # Fix: Handle division by zero more gracefully
            # If late_turns_mean is very small (< 1e-6), set ratio to a sentinel value or skip
            if abs(late_turns_mean) < 1e-6:
                # Late turns have no data or all zeros, use a sentinel value
                # Use -1 to indicate "late turns have no data"
                turn_level_stats["critic/intrinsic_reward_per_turn/early_vs_late_ratio"] = -1.0
            else:
                turn_level_stats["critic/intrinsic_reward_per_turn/early_vs_late_ratio"] = (
                    early_turns_mean / late_turns_mean
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"[TURN_LEVEL_INTRINSIC_REWARD] Failed to compute turn-level stats: {e}")
            import traceback
            traceback.print_exc()
            turn_level_stats = {}

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward (keep for backward compatibility)
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # external rewards (explicitly named)
        "critic/external_rewards/mean": torch.mean(sequence_external_reward).detach().item(),
        "critic/external_rewards/max": torch.max(sequence_external_reward).detach().item(),
        "critic/external_rewards/min": torch.min(sequence_external_reward).detach().item(),

        # intrinsic rewards (if available)
        **(
            {
                "critic/intrinsic_rewards/mean": torch.mean(sequence_intrinsic_reward).detach().item(),
                "critic/intrinsic_rewards/max": torch.max(sequence_intrinsic_reward).detach().item(),
                "critic/intrinsic_rewards/min": torch.min(sequence_intrinsic_reward).detach().item(),
            }
            if has_intrinsic_reward
            else {}
        ),
        # turn-level intrinsic rewards (if available)
        **turn_level_stats,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # advantage decomposition (Info-GRPO only)
        **(
            {
                "critic/outcome_advantage/mean": batch.meta_info["adv_stats"]["outcome_advantage_mean"],
                "critic/outcome_advantage/std": batch.meta_info["adv_stats"]["outcome_advantage_std"],
                "critic/intrinsic_advantage/mean": batch.meta_info["adv_stats"]["intrinsic_advantage_mean"],
                "critic/intrinsic_advantage/std": batch.meta_info["adv_stats"]["intrinsic_advantage_std"],
                "critic/intrinsic_contribution_ratio": batch.meta_info["adv_stats"]["intrinsic_contribution_ratio"],
                "critic/beta_t": batch.meta_info["adv_stats"]["beta_t"],
                # Additional Info-GRPO diagnostic metrics
                "critic/info_grpo/outcome_std": batch.meta_info["adv_stats"]["outcome_std"],
                "critic/info_grpo/intrinsic_std_raw": batch.meta_info["adv_stats"]["intrinsic_std_raw"],
                "critic/info_grpo/magnitude_ratio": batch.meta_info["adv_stats"]["magnitude_ratio"],
                "critic/info_grpo/intrinsic_avg_gate": batch.meta_info["adv_stats"]["intrinsic_avg_gate"],
                "critic/info_grpo/intrinsic_contribution": batch.meta_info["adv_stats"]["intrinsic_contribution"],
                # Enhanced gate diagnostics
                "critic/info_grpo/gate_min": batch.meta_info["adv_stats"]["gate_min"],
                "critic/info_grpo/gate_max": batch.meta_info["adv_stats"]["gate_max"],
                "critic/info_grpo/gate_std": batch.meta_info["adv_stats"]["gate_std"],
                # Group variance statistics (key driver of gating)
                "critic/info_grpo/group_std_mean": batch.meta_info["adv_stats"]["group_std_mean"],
                "critic/info_grpo/group_std_min": batch.meta_info["adv_stats"]["group_std_min"],
                "critic/info_grpo/group_std_max": batch.meta_info["adv_stats"]["group_std_max"],
                # Effective weights and magnitudes
                "critic/info_grpo/effective_intrinsic_weight": batch.meta_info["adv_stats"]["effective_intrinsic_weight"],
                "critic/info_grpo/abs_outcome_magnitude": batch.meta_info["adv_stats"]["abs_outcome_magnitude"],
                "critic/info_grpo/abs_intrinsic_magnitude": batch.meta_info["adv_stats"]["abs_intrinsic_magnitude"],
                "critic/info_grpo/abs_outcome_unnormalized": batch.meta_info["adv_stats"]["abs_outcome_unnormalized"],
                "critic/info_grpo/abs_intrinsic_scaled": batch.meta_info["adv_stats"]["abs_intrinsic_scaled"],
                # Hyperparameters
                "critic/info_grpo/intrinsic_gate_temperature": batch.meta_info["adv_stats"]["intrinsic_gate_temperature"],
                "critic/info_grpo/baseline_std": batch.meta_info["adv_stats"]["baseline_std"],
            }
            if "adv_stats" in batch.meta_info
            else {}
        ),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
                **(
                    {
                        "critic/vf_explained_var_action": vf_explained_var_action,
                        "critic/returns/mean_action": returns_mean_action,
                        "critic/values/mean_action": values_mean_action,
                    }
                    if vf_explained_var_action is not None
                    else {}
                ),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # Add turn statistics if available
    if num_turns_per_sample is not None:
        turns_mean = torch.mean(num_turns_per_sample).detach().item()
        turns_max = torch.max(num_turns_per_sample).detach().item()
        turns_min = torch.min(num_turns_per_sample).detach().item()
        turns_std = torch.std(num_turns_per_sample).detach().item()

        # DEBUG: Log turn statistics with detailed information
        import logging
        logger = logging.getLogger(__name__)

        # Get detailed turn information for samples with unusual turn counts
        all_turn_values = num_turns_per_sample.tolist()
        max_turn_idx = torch.argmax(num_turns_per_sample).item()
        min_turn_idx = torch.argmin(num_turns_per_sample).item()

        # Get conversation_histories for max and min turn samples
        max_turn_details = None
        min_turn_details = None
        if "conversation_histories" in batch.non_tensor_batch:
            conv_histories = batch.non_tensor_batch["conversation_histories"]
            if max_turn_idx < len(conv_histories):
                max_conv_hist = conv_histories[max_turn_idx]
                # Unwrap if nested
                if isinstance(max_conv_hist, (list, np.ndarray)) and len(max_conv_hist) > 0:
                    if isinstance(max_conv_hist[0], (list, np.ndarray)) and len(max_conv_hist[0]) > 0:
                        if isinstance(max_conv_hist[0][0], dict):
                            max_conv_hist = max_conv_hist[0]
                    elif isinstance(max_conv_hist[0], dict):
                        pass
                if isinstance(max_conv_hist, (list, np.ndarray)):
                    max_turn_details = {
                        "num_turns": len(max_conv_hist),
                        "first_turn": max_conv_hist[0] if len(max_conv_hist) > 0 else None,
                        "last_turn": max_conv_hist[-1] if len(max_conv_hist) > 0 else None,
                    }

            if min_turn_idx < len(conv_histories):
                min_conv_hist = conv_histories[min_turn_idx]
                # Unwrap if nested
                if isinstance(min_conv_hist, (list, np.ndarray)) and len(min_conv_hist) > 0:
                    if isinstance(min_conv_hist[0], (list, np.ndarray)) and len(min_conv_hist[0]) > 0:
                        if isinstance(min_conv_hist[0][0], dict):
                            min_conv_hist = min_conv_hist[0]
                    elif isinstance(min_conv_hist[0], dict):
                        pass
                if isinstance(min_conv_hist, (list, np.ndarray)):
                    min_turn_details = {
                        "num_turns": len(min_conv_hist),
                        "first_turn": min_conv_hist[0] if len(min_conv_hist) > 0 else None,
                        "last_turn": min_conv_hist[-1] if len(min_conv_hist) > 0 else None,
                    }

        logger.warning(
            f"[TURN_STATS_DEBUG] Batch turn statistics: "
            f"mean={turns_mean:.2f}, max={turns_max:.2f}, min={turns_min:.2f}, std={turns_std:.2f}, "
            f"sample_count={len(num_turns_per_sample)}, "
            f"all_values={all_turn_values[:20]}, "  # Show first 20 values
            f"max_turn_sample_idx={max_turn_idx}, max_turn_details={max_turn_details}, "
            f"min_turn_sample_idx={min_turn_idx}, min_turn_details={min_turn_details}"
        )

        metrics.update({
            "turns/mean": turns_mean,
            "turns/max": turns_max,
            "turns/min": turns_min,
            "turns/std": turns_std,
        })

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage
            - timing_pct/{name}: Percentage of total time for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor", "intrinsic_reward", "reward", "old_log_prob"]},
    }

    # Calculate total step time and stage percentages
    total_step_time = timing_raw.get("step", sum(timing_raw.values()))

    # Build timing metrics
    timing_metrics = {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section.get(name, num_overall_tokens)
           for name in timing_raw.keys() if name in num_tokens_of_section},
    }

    # Add percentage metrics
    for name, value in timing_raw.items():
        if name != "step":
            timing_metrics[f"timing_pct/{name}"] = (value / total_step_time * 100) if total_step_time > 0 else 0

    # ========== PERFORMANCE ANALYSIS REPORT ==========
    # Print detailed analysis every N steps (to reduce log spam)
    import os
    should_print_analysis = os.environ.get("VERL_TIMING_ANALYSIS", "1") == "1"

    if should_print_analysis and timing_raw:
        # Sort stages by time (descending)
        sorted_stages = sorted(timing_raw.items(), key=lambda x: x[1], reverse=True)

        print("\n" + "=" * 80)
        print("⏱️  TRAINING PERFORMANCE ANALYSIS".center(80))
        print("=" * 80)

        # Print stage-by-stage breakdown
        print(f"\n{'Stage':<20} {'Time (s)':<12} {'% of Total':<12} {'ms/token':<12} {'Status':<20}")
        print("-" * 80)

        for name, time_val in sorted_stages:
            if name == "step":
                continue
            pct = (time_val / total_step_time * 100) if total_step_time > 0 else 0
            ms_per_token = (time_val * 1000 / num_tokens_of_section.get(name, num_overall_tokens)) if name in num_tokens_of_section else 0

            # Status indicator
            if pct > 40:
                status = "🔴 BOTTLENECK"
            elif pct > 25:
                status = "🟡 HIGH"
            elif pct > 10:
                status = "🟢 NORMAL"
            else:
                status = "⚪ LOW"

            print(f"{name:<20} {time_val:<12.3f} {pct:<12.1f} {ms_per_token:<12.3f} {status:<20}")

        # Print total
        print("-" * 80)
        print(f"{'TOTAL STEP':<20} {total_step_time:<12.3f} {'100.0':<12} {'':<12} {'':<20}")

        # Performance insights
        print("\n" + "-" * 80)
        print("📊 PERFORMANCE INSIGHTS:")
        print("-" * 80)

        # Identify top 3 bottlenecks
        top_3 = sorted_stages[:3] if len(sorted_stages) > 3 else sorted_stages
        top_3 = [(n, t) for n, t in top_3 if n != "step"][:3]

        if top_3:
            print(f"\n🔝 Top 3 Time Consumers:")
            for i, (name, time_val) in enumerate(top_3, 1):
                pct = (time_val / total_step_time * 100) if total_step_time > 0 else 0
                print(f"   {i}. {name}: {time_val:.2f}s ({pct:.1f}%)")

        # Specific recommendations
        print(f"\n💡 RECOMMENDATIONS:")

        gen_time = timing_raw.get("gen", 0)
        intrinsic_time = timing_raw.get("intrinsic_reward", 0)
        update_actor_time = timing_raw.get("update_actor", 0)
        update_critic_time = timing_raw.get("update_critic", 0)
        reward_time = timing_raw.get("reward", 0)

        if gen_time / total_step_time > 0.4:
            print(f"   ⚠️  Rollout (gen) is slow ({gen_time:.2f}s, {gen_time/total_step_time*100:.1f}%)")
            print(f"      → Consider: Increase batch size, optimize generation parameters")

        if intrinsic_time / total_step_time > 0.3:
            print(f"   ⚠️  Intrinsic reward is slow ({intrinsic_time:.2f}s, {intrinsic_time/total_step_time*100:.1f}%)")
            print(f"      → Consider: Increase intrinsic_kl_batch_size (current might be too small)")
            print(f"      → Or disable intrinsic reward if not necessary")

        if (update_actor_time + update_critic_time) / total_step_time > 0.4:
            total_update = update_actor_time + update_critic_time
            print(f"   ⚠️  Gradient updates are slow ({total_update:.2f}s, {total_update/total_step_time*100:.1f}%)")
            print(f"      → Consider: Reduce ppo_mini_batch_size, optimize gradient accumulation")

        if reward_time / total_step_time > 0.2 and intrinsic_time < reward_time * 0.5:
            print(f"   ⚠️  External reward computation is slow ({reward_time:.2f}s, {reward_time/total_step_time*100:.1f}%)")
            print(f"      → Consider: Optimize reward_fn, or use async reward computation")

        # Overall assessment
        print(f"\n📈 OVERALL ASSESSMENT:")
        if gen_time / total_step_time > 0.5:
            print(f"   🎯 Focus: Optimize ROLLOUT (generation) - it's the main bottleneck")
        elif intrinsic_time / total_step_time > 0.4:
            print(f"   🎯 Focus: Optimize INTRINSIC REWARD computation")
        elif (update_actor_time + update_critic_time) / total_step_time > 0.5:
            print(f"   🎯 Focus: Optimize GRADIENT UPDATES (actor/critic)")
        else:
            print(f"   ✅ Performance is well-balanced across stages")

        print("\n" + "=" * 80 + "\n")
    # ========== END PERFORMANCE ANALYSIS ==========

    return timing_metrics


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """

    prompt_count = 0
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        # We on purpose add a suffix to the prompt to avoid the same prompt from different data sources
        prompt = sample_inputs[sample_idx] + f"_{prompt_count}"
        prompt_count += 1
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    print(f"{data_source=}, {prompt=}, {var_name=}, {var_vals=}")
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
