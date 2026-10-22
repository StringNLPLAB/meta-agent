# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ['register', "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum

import logging
import os
import numpy as np
import torch

import verl.utils.torch_functional as verl_F

import math

ADV_ESTIMATOR_REGISTRY = {}

def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}")
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator

def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]

class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GRPO_MULTITURN = "grpo_multiturn"
    INFO_GRPO = "info_grpo"
    BI_LEVEL_GAE = "bi_level_gae"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    if kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    raise NotImplementedError

@register_adv_est(AdvantageEstimator.GAE) # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_multiturn_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
    gamma: float,
    lam: float,
    turn_reward_mode: str = "raw",
):
    """Compute GAE over environment turns and broadcast it to action tokens.

    Multi-turn rollouts interleave trainable agent tokens with user and tool
    observations. Applying token-level GAE across that concatenated stream
    discounts through observations and trains the critic on non-actions. This
    implementation uses one value and reward per action turn, then assigns the
    resulting turn advantage/return to every trainable token in that turn.

    turn_reward_mode: "raw" keeps each turn's summed env reward; "terminal_only"
    zeroes every turn's reward except the last action turn, so all turn-level
    credit assignment flows through the critic's value differences; "r2g"
    replaces every turn reward with the discounted reward-to-go from that turn;
    "sao" ignores turn spans entirely and instead runs token-level GAE over
    action tokens only, skipping observation tokens (SAO-style skip-observation
    GAE, arXiv:2607.07508): the TD residual at a turn's last action token
    bridges directly to the value at the first action token of the next turn,
    and each action token gets its own advantage instead of a shared turn value.
    """
    if turn_reward_mode not in {"raw", "terminal_only", "r2g", "sao"}:
        raise ValueError(f"Invalid turn_reward_mode: {turn_reward_mode}")
    with torch.no_grad():
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        batch_size, response_length = token_level_rewards.shape

        if turn_reward_mode == "sao":
            # Skip-observation token-level GAE. The backward recursion only
            # visits action (action_mask==1) positions; observation tokens keep
            # zero advantage/return. Rewards are already placed on the last
            # action token of each turn by the reward manager, so the bridge
            # delta = r + gamma * V(a_{i+1,0}) - V(a_{i,N}) falls out naturally
            # from visiting action positions only.
            # lam <= 0 enables SAO's length-adaptive lambda (arXiv:2607.07508):
            # per-trajectory lam = 1 - 1 / (alpha * num_action_tokens), alpha=1.5.
            sao_alpha = 1.5
            for batch_idx in range(batch_size):
                action_positions = torch.where(action_mask[batch_idx, :response_length] > 0)[0].tolist()
                if lam > 0:
                    row_lam = lam
                else:
                    row_lam = 1.0 - 1.0 / (sao_alpha * max(len(action_positions), 1))
                next_value = torch.zeros((), device=values.device, dtype=values.dtype)
                last_gae = torch.zeros((), device=values.device, dtype=values.dtype)
                for p in reversed(action_positions):
                    delta = token_level_rewards[batch_idx, p] + gamma * next_value - values[batch_idx, p]
                    last_gae = delta + gamma * row_lam * last_gae
                    advantages[batch_idx, p] = last_gae
                    returns[batch_idx, p] = last_gae + values[batch_idx, p]
                    next_value = values[batch_idx, p]
            advantages = verl_F.masked_whiten(advantages, action_mask)
            return advantages, returns

        for batch_idx in range(batch_size):
            starts = torch.where(turn_boundaries[batch_idx, :response_length] == 1)[0].tolist()
            if not starts:
                starts = [0]
            starts = sorted(set(starts))
            starts.append(response_length)

            turns = []
            for turn_idx, start in enumerate(starts[:-1]):
                end = starts[turn_idx + 1]
                positions = torch.where(action_mask[batch_idx, start:end] > 0)[0] + start
                if positions.numel() == 0:
                    continue
                first_position = positions[0]
                turns.append(
                    (
                        positions,
                        token_level_rewards[batch_idx, positions].sum(),
                        values[batch_idx, first_position],
                    )
                )

            if turn_reward_mode == "terminal_only" and turns:
                turns = [
                    (positions, reward if turn_idx == len(turns) - 1 else reward * 0.0, value)
                    for turn_idx, (positions, reward, value) in enumerate(turns)
                ]
            if turn_reward_mode == "r2g" and turns:
                reward_to_go = torch.zeros((), device=values.device, dtype=values.dtype)
                turn_returns_reversed = []
                for _positions, reward, _value in reversed(turns):
                    reward_to_go = reward + gamma * reward_to_go
                    turn_returns_reversed.append(reward_to_go)
                turn_returns = list(reversed(turn_returns_reversed))
                for (positions, _reward, value), turn_return in zip(turns, turn_returns):
                    advantages[batch_idx, positions] = turn_return - value
                    returns[batch_idx, positions] = turn_return
                continue

            last_gae = torch.zeros((), device=values.device, dtype=values.dtype)
            for turn_idx in reversed(range(len(turns))):
                positions, reward, value = turns[turn_idx]
                next_value = turns[turn_idx + 1][2] if turn_idx + 1 < len(turns) else 0.0
                delta = reward + gamma * next_value - value
                last_gae = delta + gamma * lam * last_gae
                turn_return = last_gae + value
                advantages[batch_idx, positions] = last_gae
                returns[batch_idx, positions] = turn_return

        advantages = verl_F.masked_whiten(advantages, action_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO) # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.GRPO_PASSK) # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config = None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE) # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor,
                                                           epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.RLOO) # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray,
                                   epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.OPO) # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6,
                                  config=None, **kwargs):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS) # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns

@register_adv_est(AdvantageEstimator.REMAX) # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    clip_zero_out_of_range=False,
    loss_agg_mode: str = "token-mean",
    offpolicy_seq_mask_delta=None,
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        clip_zero_out_of_range (bool, optional):
            If true, tokens with ratio outside [1 - cliprange_low, 1 + cliprange_high]
            contribute zero policy loss instead of using boundary-clipped PPO.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        offpolicy_seq_mask_delta (float, optional):
            Off-Policy Sequence Masking threshold from DeepSeek-V3.2 (Sec. 3.1).
            For each sequence, compute the mean log-ratio log(pi_old/pi_theta)
            over response tokens; when it exceeds this delta, tokens with
            negative advantage in that sequence are removed from the policy loss
            (they contribute nothing and are excluded from the loss normalizer).
            Positive-advantage tokens are never masked. None disables masking.

    Returns:
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, offpolicy_seq_mask_frac
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Off-Policy Sequence Masking (DeepSeek-V3.2, Sec. 3.1): drop negative-advantage
    # tokens of sequences whose average log-ratio log(pi_old/pi_theta) exceeds
    # delta, i.e. sequences that have drifted far off the sampling policy.
    # Highly off-policy negative samples mislead optimization; on-policy mistakes
    # are kept. Positive-advantage tokens are never masked.
    offpolicy_seq_mask_frac = torch.zeros((), device=ratio.device)
    loss_mask = response_mask
    if offpolicy_seq_mask_delta is not None:
        seq_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask, axis=-1)  # (batch_size,)
        diverged = seq_divergence > offpolicy_seq_mask_delta  # (batch_size,)
        off_mask = diverged.unsqueeze(-1) & (advantages < 0)  # (batch, response_length)
        loss_mask = response_mask * (~off_mask).to(response_mask.dtype)
        offpolicy_seq_mask_frac = verl_F.masked_mean(off_mask.float(), response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    if clip_zero_out_of_range:
        lower_bound = 1 - cliprange_low
        upper_bound = 1 + cliprange_high
        in_range = (ratio >= lower_bound) & (ratio <= upper_bound)
        zero_ratio = torch.zeros_like(ratio)
        pg_losses = -advantages * torch.where(in_range, ratio, zero_ratio)
        out_of_range = torch.logical_not(in_range)
        pg_clipfrac = verl_F.masked_mean(out_of_range.float(), response_mask)
        pg_clipfrac_lower = verl_F.masked_mean(torch.lt(ratio, lower_bound).float(), response_mask)
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)
        return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, offpolicy_seq_mask_frac

    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, offpolicy_seq_mask_frac


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def compute_value_loss_bins(
    vpred_logits: torch.Tensor,
    returns: torch.Tensor,
    scalar_vpreds: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    bin_centers: torch.Tensor,
    cliprange_value: float,
    mse_aux_weight: float = 1.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Regression-as-classification value loss: cross-entropy over return bins
    with two-hot soft labels, plus an auxiliary clipped scalar MSE.

    Args:
        vpred_logits (torch.FloatTensor):
            Per-bin logits from the value head, shape (batch_size, response_length, num_bins).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        scalar_vpreds (torch.FloatTensor):
            Scalar value predictions (softmax expectation over bins, plus any
            probe correction), shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        bin_centers (torch.FloatTensor):
            Bin centers, shape (num_bins,), uniformly spaced ascending.
        cliprange_value (float):
            Clip range for the auxiliary scalar MSE (same semantics as
            `compute_value_loss`).
        mse_aux_weight (float):
            Weight of the auxiliary scalar MSE term.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor: CE + mse_aux_weight * clipped aux MSE.
        vf_clipfrac (float):
            Fraction of elements where the clipped aux MSE was used.
    """
    num_bins = bin_centers.numel()
    bin_centers = bin_centers.to(device=vpred_logits.device, dtype=torch.float32)

    # Two-hot soft label: clamp the return into [bin_min, bin_max], find the
    # surrounding bins i, i+1 and interpolate. A return exactly on a bin
    # center lands entirely on that bin.
    r = returns.float().clamp(bin_centers[0], bin_centers[-1])
    # idx = number of centers <= r, in [1, num_bins - 1] after clamping
    idx = torch.searchsorted(bin_centers, r.contiguous(), right=True).clamp(1, num_bins - 1)
    lower = idx - 1
    upper = idx
    c_lo = bin_centers[lower]
    c_hi = bin_centers[upper]
    w = (r - c_lo) / (c_hi - c_lo)
    target = torch.zeros_like(vpred_logits, dtype=torch.float32)
    target.scatter_(-1, lower.unsqueeze(-1), (1.0 - w).unsqueeze(-1))
    target.scatter_(-1, upper.unsqueeze(-1), w.unsqueeze(-1))

    # Per-token cross-entropy against the soft label.
    log_probs = torch.log_softmax(vpred_logits.float(), dim=-1)
    ce_per_token = -(target * log_probs).sum(-1)  # (batch_size, response_length)
    ce_loss = agg_loss(loss_mat=ce_per_token, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # Auxiliary clipped scalar MSE on the expectation value (keeps a dense
    # gradient path to the scalar prediction, including the probe head).
    vpredclipped = verl_F.clip_by_value(scalar_vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (scalar_vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    aux_mse_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)

    vf_loss = ce_loss + mse_aux_weight * aux_mse_loss
    return vf_loss, vf_clipfrac


# Bound for the pre-exp clamp in the k3 / low_var_kl estimator. Read ONCE at
# import: this is a hot path, and an ablation knob that can change mid-run would
# make the run it is ablating uninterpretable.
# 20.0 is the production value and is forward-identical to no clamp at all (the
# post-exp clamp already saturates at 10 for every kl >= 2.58). USERRL_KL_CLAMP
# exists only so the ablation can switch the fix off (1e30 ~= off) without
# editing this file between runs.
_KL_CLAMP = float(os.environ.get("USERRL_KL_CLAMP", "20.0"))


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-_KL_CLAMP, max=_KL_CLAMP)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_turn_credits(conversation_history, gamma=0.8, turn_level_method="Equalized", trajectory_score_method="Sum", intrinsic_rewards_per_turn=None):
    """
    Assign credits to each turn based on contribution and timing

    Args:
        conversation_history: List of conversation steps with choice/reward info
        gamma: Discount for later success (efficiency bonus)
        turn_level_method: Method for turn credit assignment
        trajectory_score_method: Method for trajectory score computation
        intrinsic_rewards_per_turn: (Optional) List of intrinsic reward values for each turn (for InfoGRPO method)

    Returns:
        turn_credits: List of credit values for each turn
        trajectory_score: Scalar trajectory score
    """

    turn_rewards = [step["reward"] for step in conversation_history]

    if trajectory_score_method == "R2G":
        trajectory_score = 0
        for turn_idx, step in enumerate(conversation_history[::-1]):
            if turn_idx == 0:
                trajectory_score = step["reward"]
            else:
                trajectory_score = step["reward"] + gamma * trajectory_score
    elif trajectory_score_method == "Sum":
        trajectory_score = round(sum(turn_rewards), 4) if turn_rewards else 0.0
    else:
        raise ValueError(f"Invalid trajectory_score_method: {trajectory_score_method}")

    if turn_level_method == "Equalized":
        turn_credits = [1.0 for _ in turn_rewards]
    elif turn_level_method == "R2G":
        turn_credits = [0.0] * len(turn_rewards)
        for turn_idx, step in enumerate(conversation_history[::-1]):
            if turn_idx == 0:
                turn_credits[turn_idx] = step["reward"]
            else:
                turn_credits[turn_idx] = step["reward"] + gamma * turn_credits[turn_idx - 1]
        turn_credits = turn_credits[::-1]
    elif turn_level_method == "EM":
        def map_reward(x, k=2.0):
            """
            Map reward x ∈ [0, 1] to higher values in [0.5, 1] using exponential scaling.
            `k` controls curvature; larger k pushes low values higher.
            """
            numerator = 1 - math.exp(-k * x)
            denominator = 1 - math.exp(-k)
            return 0.5 + 0.5 * (numerator / denominator)
        turn_credits = [round(map_reward(r), 4) for r in turn_rewards]
    elif turn_level_method == "InfoGRPO":
        # Use intrinsic rewards to weight turn credits
        if intrinsic_rewards_per_turn is None or len(intrinsic_rewards_per_turn) == 0:
            # Fallback to Equalized if no intrinsic rewards provided
            turn_credits = [1.0 for _ in turn_rewards]
        else:
            # Normalize intrinsic rewards to [0, 1] and use as weights
            intrinsic_array = np.array(intrinsic_rewards_per_turn)

            # Shift to non-negative
            min_intrinsic = intrinsic_array.min()
            if min_intrinsic < 0:
                intrinsic_array = intrinsic_array - min_intrinsic

            # Normalize to [0, 1]
            max_intrinsic = intrinsic_array.max()
            if max_intrinsic > 0:
                normalized_intrinsic = intrinsic_array / max_intrinsic
            else:
                # If all zeros, use uniform weights
                normalized_intrinsic = np.ones(len(intrinsic_array)) / len(intrinsic_array)

            # Use intrinsic weights to modulate trajectory score
            # Turns with higher intrinsic value get more credit
            turn_credits = [float(w) for w in normalized_intrinsic]
    else:
        raise ValueError(f"Invalid turn_level_method: {turn_level_method}")

    return turn_credits, trajectory_score


def compute_grpo_multiturn_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
    conversation_histories: list,
    data_sources: list,
    index: np.ndarray,
    gamma: float = 0.8,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    turn_level_method: str = "Equalized", # "Equalized" or "R2G" or "EM"
    trajectory_score_method: str = "Sum", # "Sum" or "R2G"
):
    """
    Turn-attributed GRPO with balanced action-answer credit assignment and max-reward normalization.
    This variant provides fine-grained credit assignment while maintaining GRPO's
    group baseline mechanism and preventing gaming through reward normalization.

    Args:
        token_level_rewards: (bs, response_length) - original reward tensor (may be sparse)
        response_mask: (bs, response_length) - mask for valid tokens
        turn_boundaries: (bs, response_length) - 1 at start of each turn, 0 elsewhere
        conversation_histories: List of conversation step dictionaries per batch item
        index: (bs,) - group/prompt index for GRPO baseline
        gamma: Discount factor for temporal efficiency incentive
        action_credit_ratio: Credit ratio for action turns relative to answer turns
        epsilon: Numerical stability constant
        norm_adv_by_std_in_grpo: Whether to normalize by standard deviation
        turn_level_method: Method to assign credits to turns, "Equalized" or "R2G" or "EM"
        trajectory_score_method: Method to compute trajectory score, "Sum" or "R2G"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """

    with torch.no_grad():
        advantages = torch.zeros_like(token_level_rewards)
        trajectory_scores = []
        bsz, seq_len = token_level_rewards.shape

        for b in range(bsz):
            # Compute turn credits using conversation history
            if b < len(conversation_histories) and b < len(data_sources):
                turn_credits, trajectory_score = compute_turn_credits(conversation_histories[b], gamma, turn_level_method, trajectory_score_method)
            else:
                # Fallback if no conversation history available
                turn_credits, trajectory_score = [0.0] * torch.sum(turn_boundaries[b]).item(), 0.0

            # Trajectory score is the sum of turn credits
            trajectory_scores.append(trajectory_score)

            # Find turn boundaries for this sequence
            turn_starts = torch.where(turn_boundaries[b] == 1)[0].tolist()
            if not turn_starts:
                turn_starts = [0]
            turn_starts.append(seq_len)

            # Assign credits to tokens within each turn
            for turn_idx in range(len(turn_starts) - 1):
                turn_start = turn_starts[turn_idx]
                turn_end = turn_starts[turn_idx + 1]

                if turn_idx < len(turn_credits):
                    turn_credit = turn_credits[turn_idx]
                    # All tokens in this turn get the same credit
                    advantages[b, turn_start:turn_end] = turn_credit

        # Apply GRPO group baseline (same as original GRPO)
        trajectory_scores = torch.tensor(trajectory_scores, device=advantages.device)

        # mask invalid tokens in advance
        advantages = advantages * response_mask

        # Group baseline computation (identical to original GRPO)
        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        for i in range(bsz):
            id2score[index[i]].append(trajectory_scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=advantages.device)
                id2std[idx] = torch.tensor(1.0, device=advantages.device)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)

        # Apply group baseline normalization
        for b in range(bsz):
            group_idx = index[b]
            if norm_adv_by_std_in_grpo:
                normalization_factor = (trajectory_scores[b] - id2mean[group_idx]) / (id2std[group_idx] + epsilon)
            else:
                normalization_factor = trajectory_scores[b] - id2mean[group_idx]

            episode_sum = torch.sum(advantages[b]) # trajectory_scores[b]
            if episode_sum != 0:
                num_non_zero = torch.count_nonzero(advantages[b])
                scaling_factor = normalization_factor * num_non_zero / episode_sum
                advantages[b] *= scaling_factor
            else:
                advantages[b] *= 0.0

        # For outcome supervision, returns = advantages
        returns = advantages.clone()
    logging.getLogger(__name__).debug("GRPO Multiturn Advantage: %s", tuple(advantages.shape))

    return advantages, returns


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


@register_adv_est("info_grpo")
def compute_info_grpo_advantage(
    token_level_rewards: torch.Tensor,
    token_level_intrinsic_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    intrinsic_weight: float = 0.1,
    intrinsic_decay_rate: float = 0.0,
    current_epoch: int = 0,
    # Multi-turn parameters (optional, for backward compatibility)
    turn_boundaries: torch.Tensor = None,
    conversation_histories: list = None,
    data_sources: list = None,
    gamma: float = 0.8,
    turn_level_method: str = "Equalized",
    trajectory_score_method: str = "Sum",
    **kwargs,
):
    """
    Compute advantage for Info-GRPO with orthogonal decomposition.

    This estimator computes separate advantages for:
    1. Outcome rewards (sparse task rewards) - turn-level credit assignment (if multi-turn) or trajectory-level (fallback)
    2. Intrinsic rewards (HPS-based exploration rewards) - token/turn-level

    Key improvement: Intrinsic rewards maintain token-level granularity!
    - Outcome advantage: turn-level credit assignment (if multi-turn) or broadcast to all tokens (trajectory-level signal)
    - Intrinsic advantage: per-token normalization (turn-level signal)

    This allows different turns to receive different gradient updates based on
    their information gain, rather than averaging out all turn contributions.

    Args:
        token_level_rewards: (bs, response_length) - Outcome rewards
        token_level_intrinsic_rewards: (bs, response_length) - Intrinsic rewards
        response_mask: (bs, response_length)
        index: (bs,) - Group IDs for GRPO normalization
        epsilon: Small constant for numerical stability
        norm_adv_by_std_in_grpo: Whether to normalize by std
        intrinsic_weight: Base weight for intrinsic rewards (β_0)
        intrinsic_decay_rate: Decay rate for intrinsic weight
        current_epoch: Current training epoch (for decay)
        turn_boundaries: (bs, response_length) - Optional, 1 at start of each turn, 0 elsewhere
        conversation_histories: List of conversation step dictionaries per batch item (optional)
        data_sources: List of data sources per batch item (optional)
        gamma: Discount factor for temporal efficiency incentive (default: 0.8)
        turn_level_method: Method to assign credits to turns, "Equalized" or "R2G" or "EM" (default: "Equalized")
        trajectory_score_method: Method to compute trajectory score, "Sum" or "R2G" (default: "Sum")

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    # Ensure intrinsic_weight is scalar
    if torch.is_tensor(intrinsic_weight):
        intrinsic_weight = intrinsic_weight.mean().item()

    # Compute current intrinsic weight with decay
    beta_t = intrinsic_weight * math.exp(-intrinsic_decay_rate * current_epoch)

    # 1. Compute outcome advantage using turn-level credit assignment (if multi-turn) or trajectory-level (fallback)
    with torch.no_grad():
        bsz, seq_len = token_level_rewards.shape

        # Check if we have multi-turn information
        use_multiturn = (turn_boundaries is not None and
                        conversation_histories is not None and
                        len(conversation_histories) > 0)

        if use_multiturn:
            # Multi-turn mode: Use turn-level credit assignment (same as compute_grpo_multiturn_advantage)
            advantages_outcome = torch.zeros_like(token_level_rewards)
            trajectory_scores = []

            for b in range(bsz):
                # Compute turn credits using conversation history
                if b < len(conversation_histories) and conversation_histories[b] is not None:
                    conv_hist = conversation_histories[b]
                    # Ensure it's a list/dict, not a string or other type
                    if isinstance(conv_hist, (list, tuple)) and len(conv_hist) > 0:
                        # Check if it's a list of dicts (expected format)
                        if isinstance(conv_hist[0], dict):
                            turn_credits, trajectory_score = compute_turn_credits(
                                conv_hist,
                                gamma,
                                turn_level_method,
                                trajectory_score_method
                            )
                        else:
                            # Fallback if structure is unexpected
                            turn_credits, trajectory_score = [0.0] * torch.sum(turn_boundaries[b]).item(), 0.0
                    elif isinstance(conv_hist, dict):
                        # Single dict, wrap in list
                        turn_credits, trajectory_score = compute_turn_credits(
                            [conv_hist],
                            gamma,
                            turn_level_method,
                            trajectory_score_method
                        )
                    else:
                        # Fallback if type is unexpected
                        turn_credits, trajectory_score = [0.0] * torch.sum(turn_boundaries[b]).item(), 0.0
                else:
                    # Fallback if no conversation history available
                    turn_credits, trajectory_score = [0.0] * torch.sum(turn_boundaries[b]).item(), 0.0

                # Trajectory score is used for GRPO normalization
                trajectory_scores.append(trajectory_score)

                # Find turn boundaries for this sequence
                turn_starts = torch.where(turn_boundaries[b] == 1)[0].tolist()
                if not turn_starts:
                    turn_starts = [0]
                turn_starts.append(seq_len)

                # DEBUG: Log turn_starts and turn_credits matching
                num_turn_starts = len(turn_starts) - 1  # Exclude the appended seq_len
                num_turn_credits = len(turn_credits)
                if num_turn_starts != num_turn_credits:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"[TURN_CREDIT_MISMATCH] Batch {b}: "
                        f"num_turn_starts={num_turn_starts}, num_turn_credits={num_turn_credits}, "
                        f"turn_starts={turn_starts[:10] if len(turn_starts) > 10 else turn_starts}, "
                        f"turn_credits_length={num_turn_credits}, "
                        f"seq_len={seq_len}"
                    )

                # Assign credits to tokens within each turn
                for turn_idx in range(len(turn_starts) - 1):
                    turn_start = turn_starts[turn_idx]
                    turn_end = turn_starts[turn_idx + 1]

                    if turn_idx < len(turn_credits):
                        turn_credit = turn_credits[turn_idx]
                        # All tokens in this turn get the same credit
                        advantages_outcome[b, turn_start:turn_end] = turn_credit
                    else:
                        # DEBUG: Log if turn_idx exceeds turn_credits
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"[TURN_CREDIT_MISSING] Batch {b}, turn_idx={turn_idx}: "
                            f"turn_start={turn_start}, turn_end={turn_end}, "
                            f"but only {len(turn_credits)} turn_credits available. "
                            f"This turn will have zero credit!"
                        )

            # Apply GRPO group baseline normalization (same as compute_grpo_multiturn_advantage)
            trajectory_scores = torch.tensor(trajectory_scores, device=advantages_outcome.device)

            # Mask invalid tokens
            advantages_outcome = advantages_outcome * response_mask

            # Group baseline computation
            id2score = defaultdict(list)
            id2mean = {}
            id2std = {}

            for i in range(bsz):
                id2score[index[i]].append(trajectory_scores[i])

            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0, device=advantages_outcome.device)
                    id2std[idx] = torch.tensor(1.0, device=advantages_outcome.device)
                elif len(id2score[idx]) > 1:
                    scores_tensor = torch.stack(id2score[idx])
                    id2mean[idx] = torch.mean(scores_tensor)
                    id2std[idx] = torch.std(scores_tensor)

            # Apply group baseline normalization
            for b in range(bsz):
                group_idx = index[b]
                if norm_adv_by_std_in_grpo:
                    normalization_factor = (trajectory_scores[b] - id2mean[group_idx]) / (id2std[group_idx] + epsilon)
                else:
                    normalization_factor = trajectory_scores[b] - id2mean[group_idx]

                episode_sum = torch.sum(advantages_outcome[b])
                if episode_sum != 0:
                    num_non_zero = torch.count_nonzero(advantages_outcome[b])
                    scaling_factor = normalization_factor * num_non_zero / episode_sum
                    advantages_outcome[b] *= scaling_factor
                else:
                    advantages_outcome[b] *= 0.0

            adv_outcome_broadcast = advantages_outcome  # Already token-level with turn credits

        else:
            # Single-turn mode: Use trajectory-level (backward compatible)
            scores_outcome = token_level_rewards.sum(dim=-1)  # (bs,)

            id2score_outcome = defaultdict(list)
            id2mean_outcome = {}
            id2std_outcome = {}

            # Group normalization for outcome rewards
            for i in range(bsz):
                id2score_outcome[index[i]].append(scores_outcome[i])

            for idx in id2score_outcome:
                if len(id2score_outcome[idx]) == 1:
                    id2mean_outcome[idx] = torch.tensor(0.0, device=token_level_rewards.device)
                    id2std_outcome[idx] = torch.tensor(1.0, device=token_level_rewards.device)
                elif len(id2score_outcome[idx]) > 1:
                    id2mean_outcome[idx] = torch.mean(torch.stack(id2score_outcome[idx]))
                    id2std_outcome[idx] = torch.std(torch.stack(id2score_outcome[idx]))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")

            adv_outcome = torch.zeros_like(scores_outcome)
            for i in range(bsz):
                if norm_adv_by_std_in_grpo:
                    adv_outcome[i] = (scores_outcome[i] - id2mean_outcome[index[i]]) / (id2std_outcome[index[i]] + epsilon)
                else:
                    adv_outcome[i] = scores_outcome[i] - id2mean_outcome[index[i]]

            # Broadcast outcome advantage to all tokens (trajectory-level signal)
            adv_outcome_broadcast = adv_outcome.unsqueeze(-1) * response_mask  # (bs, response_length)

    # 2. Compute intrinsic advantage (TOKEN-LEVEL, preserving turn granularity)
    # Key difference: We don't sum! We normalize each token's intrinsic reward within its group.

    # Check if normalization should be applied (ablation: normalize_intrinsic=False means no normalization at all)
    normalize_intrinsic = kwargs.get('normalize_intrinsic', True)  # Default to True for backward compatibility

    if not normalize_intrinsic:
        # Ablation: Use raw intrinsic rewards without any normalization
        # This will likely lead to unstable training due to:
        # 1. Uncontrolled magnitude (intrinsic rewards can be much larger/smaller than outcome rewards)
        # 2. No group-level centering (different groups have different baseline KL values)
        # 3. No variance normalization (high-variance groups dominate gradients)
        adv_intrinsic_token_level = token_level_intrinsic_rewards.clone()
        logging.getLogger(__name__).info(
            "[INFO-GRPO] Intrinsic reward normalization disabled (ablation): using raw intrinsic rewards."
        )
    else:
        # Collect all non-zero intrinsic rewards per group for normalization
        id2intrinsic_values = defaultdict(list)

        with torch.no_grad():
            for i in range(bsz):
                group_idx = index[i]
                # Get non-zero intrinsic rewards for this trajectory
                mask_i = response_mask[i].bool()
                intrinsic_i = token_level_intrinsic_rewards[i][mask_i]
                non_zero_intrinsic = intrinsic_i[intrinsic_i != 0]
                if len(non_zero_intrinsic) > 0:
                    id2intrinsic_values[group_idx].extend(non_zero_intrinsic.tolist())

            # Compute mean and std per group (across all non-zero intrinsic rewards)
            id2mean_intrinsic = {}
            id2std_intrinsic = {}

            for idx in id2intrinsic_values:
                if len(id2intrinsic_values[idx]) == 0:
                    id2mean_intrinsic[idx] = 0.0
                    id2std_intrinsic[idx] = 1.0
                elif len(id2intrinsic_values[idx]) == 1:
                    # Single value: center at 0, std = 1
                    id2mean_intrinsic[idx] = id2intrinsic_values[idx][0]
                    id2std_intrinsic[idx] = 1.0
                else:
                    values_tensor = torch.tensor(id2intrinsic_values[idx], device=token_level_intrinsic_rewards.device)
                    id2mean_intrinsic[idx] = values_tensor.mean().item()
                    id2std_intrinsic[idx] = values_tensor.std().item()

            # Normalize intrinsic rewards at token level
            adv_intrinsic_token_level = torch.zeros_like(token_level_intrinsic_rewards)

            for i in range(bsz):
                group_idx = index[i]
                if group_idx in id2mean_intrinsic:
                    mean_val = id2mean_intrinsic[group_idx]
                    std_val = id2std_intrinsic[group_idx]

                    if norm_adv_by_std_in_grpo:
                        # Normalize: (reward - mean) / std for non-zero rewards
                        # Zero rewards stay zero (no gradient)
                        mask_i = response_mask[i].bool()
                        non_zero_mask = (token_level_intrinsic_rewards[i] != 0) & mask_i
                        adv_intrinsic_token_level[i][non_zero_mask] = (
                            (token_level_intrinsic_rewards[i][non_zero_mask] - mean_val) / (std_val + epsilon)
                        )
                    else:
                        # Just center: reward - mean
                        mask_i = response_mask[i].bool()
                        non_zero_mask = (token_level_intrinsic_rewards[i] != 0) & mask_i
                        adv_intrinsic_token_level[i][non_zero_mask] = (
                            token_level_intrinsic_rewards[i][non_zero_mask] - mean_val
                        )

    # 3. Combine with variance-gated intrinsic motivation
    # Two-stage approach:
    #   (a) Bounded scaling: prevent magnitude EXPLOSION (downscale if too large)
    #       - If intrinsic_std >> outcome_std: scale down to match
    #       - If intrinsic_std << outcome_std: keep original (don't suppress!)
    #       - Use baseline_std=1.0 to ensure intrinsic stays active when outcome_std→0
    #   (b) Variance gating: suppress intrinsic when group has strong external signals
    #       - High group variance → gate→0 (external signal is informative)
    #       - Low group variance → gate→0.5 (need intrinsic for exploration)
    #
    # Role of beta_t: Global decay weight (only in final combination)
    #   - Early training (beta_t ≈ 0.1): intrinsic motivation is active
    #   - Late training (beta_t → 0): shift to pure external reward optimization

    mask = response_mask.bool()
    outcome_std = adv_outcome_broadcast[mask].std().item() + epsilon
    intrinsic_std_raw = adv_intrinsic_token_level[mask].std().item() + epsilon

    # Stage 1: Bounded scaling (prevents magnitude explosion, not suppression)
    # Key insight: Only DOWNSCALE intrinsic when it's too large, never suppress it
    # When outcome_std is small (weak signal), intrinsic should maintain its magnitude

    # Option 1: Use a minimum baseline to prevent over-suppression
    # When outcome_std → 0, we still want intrinsic to contribute meaningfully
    baseline_std = 1.0  # Minimum target std for intrinsic (ensures it stays active)
    target_intrinsic_std = max(outcome_std, baseline_std)
    magnitude_ratio = min(1.0, target_intrinsic_std / intrinsic_std_raw)
    adv_intrinsic_scaled = adv_intrinsic_token_level * magnitude_ratio

    # Stage 2: Variance-based gating (GRPO-aligned: use group variance as signal quality)
    # High group variance → strong external signal → suppress intrinsic
    # Low group variance → weak external signal → use intrinsic for exploration
    intrinsic_gate_temperature = kwargs.get('intrinsic_gate_temperature', 0.05)
    gate = torch.ones(bsz, 1, device=adv_outcome_broadcast.device)

    # Use id2std from GRPO computation (already computed in multi-turn or single-turn branches)
    if use_multiturn:
        # id2std already computed in multi-turn branch
        for i in range(bsz):
            group_idx = index[i]
            group_std = id2std.get(group_idx, torch.tensor(0.0)).item()
            gate_value = torch.sigmoid(torch.tensor(-group_std / intrinsic_gate_temperature))
            gate[i] = gate_value
    else:
        # Use id2std_outcome from single-turn branch
        for i in range(bsz):
            group_idx = index[i]
            group_std = id2std_outcome.get(group_idx, torch.tensor(0.0)).item()
            gate_value = torch.sigmoid(torch.tensor(-group_std / intrinsic_gate_temperature))
            gate[i] = gate_value

    # Ablation: Use intrinsic reward only (ignore outcome reward)
    use_intrinsic_only = kwargs.get('use_intrinsic_only', False)  # Default to False for backward compatibility

    if use_intrinsic_only:
        # Ablation: Only use intrinsic rewards, ignore outcome rewards
        # This tests whether intrinsic rewards alone can guide the agent to improve outcome rewards
        combined_advantages = adv_intrinsic_scaled
        logging.getLogger(__name__).info(
            "[INFO-GRPO] Intrinsic-only mode enabled (ablation): outcome rewards are ignored."
        )
    else:
        # Normal mode: Combine outcome and intrinsic rewards
        combined_advantages = adv_outcome_broadcast + (beta_t * gate) * adv_intrinsic_scaled

    # Compute detailed advantage decomposition for monitoring
    # Use adv_intrinsic_weighted for accurate contribution calculation
    valid_outcome_adv = adv_outcome_broadcast[mask]
    valid_intrinsic_adv = (beta_t * gate * adv_intrinsic_scaled)[mask]

    # Compute contribution ratio: what % of gradient comes from intrinsic?
    abs_outcome = torch.abs(valid_outcome_adv).mean().item()
    abs_intrinsic = torch.abs(valid_intrinsic_adv).mean().item()
    total_abs = abs_outcome + abs_intrinsic + 1e-10
    intrinsic_contrib = abs_intrinsic / total_abs

    # Enhanced gate statistics for better diagnostics
    avg_gate = gate.mean().item()
    min_gate = gate.min().item()
    max_gate = gate.max().item()
    std_gate = gate.std().item()

    # Compute group variance statistics (key driver of gating mechanism)
    group_stds = []
    for i in range(bsz):
        group_idx = index[i]
        if use_multiturn:
            group_std = id2std.get(group_idx, torch.tensor(0.0)).item()
        else:
            group_std = id2std_outcome.get(group_idx, torch.tensor(0.0)).item()
        group_stds.append(group_std)

    avg_group_std = np.mean(group_stds)
    min_group_std = np.min(group_stds)
    max_group_std = np.max(group_stds)

    # Compute effective scaling factors to understand Stage 1 vs Stage 2 impact
    # Stage 1 effect: magnitude_ratio
    # Stage 2 effect: gate
    # Combined effect: beta_t * gate * magnitude_ratio
    effective_intrinsic_weight = (beta_t * avg_gate * magnitude_ratio)

    # Absolute magnitude of advantages (before normalization)
    abs_outcome_unnorm = torch.abs(adv_outcome_broadcast[mask]).mean().item()
    abs_intrinsic_scaled = torch.abs(adv_intrinsic_scaled[mask]).mean().item()

    logging.getLogger(__name__).debug(
        "Info-GRPO Advantage: outcome_std=%.4f intrinsic_std_raw=%.4f magnitude_ratio=%.4f "
        "avg_gate=%.4f [%.4f-%.4f] avg_group_std=%.4f beta_t=%.4f intrinsic_contrib=%.4f "
        "effective_weight=%.4f",
        outcome_std,
        intrinsic_std_raw,
        magnitude_ratio,
        avg_gate,
        min_gate,
        max_gate,
        avg_group_std,
        beta_t,
        intrinsic_contrib,
        effective_intrinsic_weight,
    )

    adv_stats = {
        "outcome_std": outcome_std,
        "intrinsic_std_raw": intrinsic_std_raw,
        "magnitude_ratio": magnitude_ratio,
        "intrinsic_avg_gate": avg_gate,
        "intrinsic_contribution": intrinsic_contrib,
        "beta_t": beta_t,
        # Add missing keys for metric_utils.py
        "outcome_advantage_mean": valid_outcome_adv.mean().item(),
        "outcome_advantage_std": valid_outcome_adv.std().item(),
        "intrinsic_advantage_mean": valid_intrinsic_adv.mean().item(),
        "intrinsic_advantage_std": valid_intrinsic_adv.std().item(),
        "intrinsic_contribution_ratio": intrinsic_contrib,
        # Enhanced diagnostic metrics
        "gate_min": min_gate,
        "gate_max": max_gate,
        "gate_std": std_gate,
        "group_std_mean": avg_group_std,
        "group_std_min": min_group_std,
        "group_std_max": max_group_std,
        "effective_intrinsic_weight": effective_intrinsic_weight,
        "abs_outcome_magnitude": abs_outcome,
        "abs_intrinsic_magnitude": abs_intrinsic,
        "abs_outcome_unnormalized": abs_outcome_unnorm,
        "abs_intrinsic_scaled": abs_intrinsic_scaled,
        "intrinsic_gate_temperature": intrinsic_gate_temperature,
        "baseline_std": baseline_std,
    }

    return combined_advantages, combined_advantages, adv_stats


@register_adv_est(AdvantageEstimator.BI_LEVEL_GAE)
def compute_bi_level_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
    high_level_gamma: float = 0.95,
    config=None,
    **kwargs
):
    """
    Modified GAE calculation that computes two levels of advantage and return:
    - High level: per-turn wise (using high_level_gamma)
    - Low level: token wise (using gamma)

    There are two levels of MDP, where high level is the agentic MDP and low level is the token MDP.
    This is adapted from RAGEN's bi-level GAE for multi-turn RL training.

    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward (default: 0.95)
        config: (dict) optional algorithm config (for compatibility)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # Extract high_level_gamma from config if provided
    if config is not None:
        high_level_gamma = config.get("high_level_gamma", high_level_gamma)

    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        reward_mask = token_level_rewards.bool()  # Detect turn boundaries (non-zero rewards)
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()

        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions = reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]

                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0

                # Calculate delta using the next valid token
                delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]

                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam

            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]

            # Then, calculate low level advantage and return for each token using gamma,
            # assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = response_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

        advantages = verl_F.masked_whiten(advantages, response_mask)

    return advantages, returns
