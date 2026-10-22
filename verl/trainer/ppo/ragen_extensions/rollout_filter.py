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
Rollout Filtering for RAGEN Extensions

Implements uncertainty-based filtering from the RAGEN paper (StarPO-S stabilization).
Filters rollouts based on reward/entropy variance to focus training on informative samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from verl import DataProto


@dataclass
class RolloutFilterConfig:
    """Configuration for rollout filtering.

    Args:
        enable: Whether to enable rollout filtering (default: False for backward compatibility)
        ratio: Fraction of groups to keep (e.g., 0.25 keeps top 25% by variance)
        filter_type: "largest" (keep high variance) or "smallest" (keep low variance)
        metric: Metric to use for filtering ("reward_variance", "entropy_variance", etc.)
        group_size: Number of responses per prompt group
    """
    enable: bool = False
    ratio: float = 0.25
    filter_type: str = "largest"
    metric: str = "reward_variance"
    group_size: int = 16


class RolloutFilter:
    """Base class for rollout filters."""

    def __init__(self, config: RolloutFilterConfig):
        self.config = config

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, float]]:
        """Apply filtering to a batch of rollouts.

        Args:
            batch: DataProto containing rollout data

        Returns:
            filtered_batch: Filtered DataProto
            metrics: Dict of filtering metrics for logging
        """
        raise NotImplementedError

    @property
    def ratio(self) -> float:
        return self.config.ratio

    @property
    def filter_type(self) -> str:
        return self.config.filter_type

    @property
    def group_size(self) -> int:
        return self.config.group_size

    def _select_top_groups(self, scores: torch.Tensor, num_groups: int) -> torch.Tensor:
        """Select top-k groups based on scores and filter_type."""
        if self.ratio >= 1.0:
            return torch.arange(num_groups, device=scores.device)

        k = max(int(self.ratio * num_groups), 1)

        if self.filter_type == "smallest":
            top_groups = (-scores).topk(k).indices
        elif self.filter_type == "largest":
            top_groups = scores.topk(k).indices
        else:
            raise ValueError(f"Invalid rollout filter type: {self.filter_type}")

        return top_groups

    def _groups_to_mask(self, top_groups: torch.Tensor, group_size: int, num_groups: int) -> torch.Tensor:
        """Convert selected group indices to a sample-level boolean mask."""
        device = top_groups.device
        mask = torch.zeros(num_groups, dtype=torch.bool, device=device)
        if top_groups.numel() > 0:
            mask[top_groups] = True
        mask = mask.unsqueeze(1).expand(-1, group_size).reshape(-1).cpu()
        return mask

    def _apply_mask(self, batch: DataProto, mask: torch.Tensor) -> DataProto:
        """Apply boolean mask to filter batch data."""
        # Filter tensor batch
        batch.batch = batch.batch[mask]

        # Filter non-tensor batch
        if batch.non_tensor_batch is not None:
            np_mask = mask.cpu().numpy()
            filtered_non_tensor_batch = {}
            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    filtered_non_tensor_batch[key] = value[np_mask]
                elif isinstance(value, list):
                    filtered_non_tensor_batch[key] = [value[i] for i, m in enumerate(np_mask) if m]
                else:
                    # Keep scalar values as-is
                    filtered_non_tensor_batch[key] = value
            batch.non_tensor_batch = filtered_non_tensor_batch

        # Filter meta_info if it contains lists
        if batch.meta_info is not None:
            filtered_meta_info = {}
            np_mask = mask.cpu().numpy()
            for key, value in batch.meta_info.items():
                if isinstance(value, list) and len(value) == len(mask):
                    filtered_meta_info[key] = [value[i] for i, m in enumerate(np_mask) if m]
                else:
                    filtered_meta_info[key] = value
            batch.meta_info = filtered_meta_info

        return batch

    def _build_base_metrics(
        self, in_group_std: torch.Tensor, in_group_max: torch.Tensor,
        in_group_mean: torch.Tensor, top_groups: torch.Tensor
    ) -> Dict[str, float]:
        """Build basic filtering metrics for logging."""
        return {
            "rollout/filter_ratio": self.ratio,
            "rollout/num_groups_kept": len(top_groups),
            "rollout/in_group_std_mean": in_group_std.mean().item(),
            "rollout/in_group_max_mean": in_group_max.mean().item(),
            "rollout/in_group_mean_mean": in_group_mean.mean().item(),
        }


class RewardRolloutFilter(RolloutFilter):
    """Filters rollouts based on reward variance within groups.

    This implements the uncertainty-based filtering from RAGEN (StarPO-S).
    Keeps prompts with high reward variance (informative samples).
    """

    def _selection_scores(self, in_group_std: torch.Tensor, in_group_mean: torch.Tensor) -> torch.Tensor:
        """Use reward std as selection criterion."""
        return in_group_std

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, float]]:
        """Filter based on reward variance."""
        # Check if we have uid (prompt group ids)
        if "uid" not in batch.non_tensor_batch:
            raise ValueError("Rollout filtering requires 'uid' in non_tensor_batch for grouping")

        # Get rewards and group structure
        token_level_rewards = batch.batch.get("token_level_rewards", batch.batch.get("token_level_scores"))
        all_scores = token_level_rewards.sum(dim=-1)  # (batch_size,)

        # Group by uid
        uids = batch.non_tensor_batch["uid"]
        unique_uids = sorted(set(uids))
        num_groups = len(unique_uids)

        # Compute per-group statistics
        uid_to_scores = {uid: [] for uid in unique_uids}
        for i, uid in enumerate(uids):
            uid_to_scores[uid].append(all_scores[i].item())

        # Convert to tensors for statistics
        group_stds = []
        group_means = []
        group_maxs = []
        for uid in unique_uids:
            scores = torch.tensor(uid_to_scores[uid], device=all_scores.device)
            group_stds.append(scores.std().item() if len(scores) > 1 else 0.0)
            group_means.append(scores.mean().item())
            group_maxs.append(scores.max().item())

        in_group_std = torch.tensor(group_stds, device=all_scores.device)
        in_group_mean = torch.tensor(group_means, device=all_scores.device)
        in_group_max = torch.tensor(group_maxs, device=all_scores.device)

        # Select top groups by variance
        selection_scores = self._selection_scores(in_group_std, in_group_mean)
        top_groups = self._select_top_groups(selection_scores, num_groups)

        # Build metrics
        metrics = self._build_base_metrics(in_group_std, in_group_max, in_group_mean, top_groups)
        metrics.update({
            "rollout/reward_std_mean": in_group_std.mean().item(),
            "rollout/reward_max_mean": in_group_max.mean().item(),
            "rollout/reward_mean": in_group_mean.mean().item(),
            "rollout/chosen_reward_std_mean": in_group_std[top_groups].mean().item(),
        })

        # If ratio >= 1.0, keep all samples
        if self.ratio >= 1.0:
            return batch, metrics

        # Build mask for samples
        kept_uids = set([unique_uids[i] for i in top_groups.cpu().tolist()])
        mask = torch.tensor([uid in kept_uids for uid in uids], dtype=torch.bool)

        # Apply mask
        batch = self._apply_mask(batch, mask)

        return batch, metrics


class EntropyRolloutFilter(RolloutFilter):
    """Filters rollouts based on policy entropy variance within groups.

    This is an alternative filtering method that uses policy entropy instead of rewards.
    Useful for detecting echo traps (low entropy indicates mode collapse).
    """

    def __init__(self, config: RolloutFilterConfig, compute_log_prob: Callable[[DataProto], DataProto]):
        super().__init__(config)
        self._compute_log_prob = compute_log_prob

    def _selection_scores(self, in_group_std: torch.Tensor, in_group_mean: torch.Tensor) -> torch.Tensor:
        """Use entropy std as selection criterion."""
        return in_group_std

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, float]]:
        """Filter based on entropy variance."""
        # Compute log probs and entropy if not already present
        if "entropy" not in batch.batch:
            batch = self._compute_log_prob(batch)

        # Check if we have uid (prompt group ids)
        if "uid" not in batch.non_tensor_batch:
            raise ValueError("Rollout filtering requires 'uid' in non_tensor_batch for grouping")

        # Get entropy per trajectory
        response_mask = batch.batch.get("response_mask", batch.batch.get("attention_mask"))
        entropy = batch.batch["entropy"]  # (batch_size, seq_len)
        entropy_per_traj = (entropy * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)  # (batch_size,)

        # Group by uid
        uids = batch.non_tensor_batch["uid"]
        unique_uids = sorted(set(uids))
        num_groups = len(unique_uids)

        # Compute per-group statistics
        uid_to_entropy = {uid: [] for uid in unique_uids}
        for i, uid in enumerate(uids):
            uid_to_entropy[uid].append(entropy_per_traj[i].item())

        # Convert to tensors for statistics
        group_stds = []
        group_means = []
        group_maxs = []
        for uid in unique_uids:
            entropies = torch.tensor(uid_to_entropy[uid], device=entropy_per_traj.device)
            group_stds.append(entropies.std().item() if len(entropies) > 1 else 0.0)
            group_means.append(entropies.mean().item())
            group_maxs.append(entropies.max().item())

        in_group_std = torch.tensor(group_stds, device=entropy_per_traj.device)
        in_group_mean = torch.tensor(group_means, device=entropy_per_traj.device)
        in_group_max = torch.tensor(group_maxs, device=entropy_per_traj.device)

        # Select top groups by entropy variance
        selection_scores = self._selection_scores(in_group_std, in_group_mean)
        top_groups = self._select_top_groups(selection_scores, num_groups)

        # Build metrics
        metrics = self._build_base_metrics(in_group_std, in_group_max, in_group_mean, top_groups)
        metrics.update({
            "rollout/entropy_std_mean": in_group_std.mean().item(),
            "rollout/entropy_max_mean": in_group_max.mean().item(),
            "rollout/entropy_mean": in_group_mean.mean().item(),
            "rollout/chosen_entropy_std_mean": in_group_std[top_groups].mean().item(),
        })

        # If ratio >= 1.0, keep all samples
        if self.ratio >= 1.0:
            return batch, metrics

        # Build mask for samples
        kept_uids = set([unique_uids[i] for i in top_groups.cpu().tolist()])
        mask = torch.tensor([uid in kept_uids for uid in uids], dtype=torch.bool)

        # Apply mask
        batch = self._apply_mask(batch, mask)

        return batch, metrics


def create_rollout_filter(
    config: RolloutFilterConfig,
    compute_log_prob: Optional[Callable[[DataProto], DataProto]] = None
) -> Optional[RolloutFilter]:
    """Factory function to create a rollout filter based on config.

    Args:
        config: RolloutFilterConfig specifying filter settings
        compute_log_prob: Optional function to compute log probs (required for entropy filter)

    Returns:
        RolloutFilter instance or None if filtering is disabled
    """
    if not config.enable:
        return None

    metric = config.metric
    if metric in ("reward_variance", "reward"):
        return RewardRolloutFilter(config)
    elif metric in ("entropy_variance", "entropy"):
        if compute_log_prob is None:
            raise ValueError("EntropyRolloutFilter requires compute_log_prob function")
        return EntropyRolloutFilter(config, compute_log_prob)
    else:
        raise ValueError(f"Unknown filter metric: {metric}")
