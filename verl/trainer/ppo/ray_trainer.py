# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import shutil
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import GYM_METRICS_KEY, compute_reward, compute_reward_async
from verl.trainer.ppo.ragen_extensions import create_rollout_filter, RolloutFilterConfig
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path, get_best_score
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]

def _unwrap_conversation_histories(raw_histories) -> list:
    """Normalize conversation histories to a list of per-sample histories (list[dict])."""
    if raw_histories is None:
        return []

    histories = raw_histories.tolist() if isinstance(raw_histories, np.ndarray) else list(raw_histories)
    unwrapped = []
    for item in histories:
        if isinstance(item, list):
            if item and isinstance(item[0], dict):
                unwrapped.append(item)
            elif item and isinstance(item[0], (list, np.ndarray)):
                first = item[0]
                unwrapped.append(first.tolist() if isinstance(first, np.ndarray) else first)
            else:
                unwrapped.append(item)
            continue

        if isinstance(item, dict):
            unwrapped.append([item])
            continue

        if isinstance(item, str):
            try:
                parsed = json.loads(item)
                unwrapped.append(parsed if isinstance(parsed, list) else [])
            except (json.JSONDecodeError, TypeError):
                unwrapped.append([])
            continue

        if isinstance(item, np.ndarray):
            item_list = item.tolist()
            if item_list and isinstance(item_list[0], dict):
                unwrapped.append(item_list)
            elif item_list and isinstance(item_list[0], (list, np.ndarray)):
                first = item_list[0]
                unwrapped.append(first.tolist() if isinstance(first, np.ndarray) else first)
            else:
                unwrapped.append(item_list)
            continue

        unwrapped.append([])

    return unwrapped


CLIP_SCHEDULE_PATH = os.environ.get(
    "USERRL_CLIP_SCHEDULE_CONFIG",
    os.path.join(os.getcwd(), "clip_schedule.json"),
)
# hard bounds, so a typo in the file cannot destroy the run
_CLIP_HIGH_FLOOR, _CLIP_HIGH_CEIL = 0.01, 5.0
_CLIP_DEFAULT_MAX_DELTA = 0.05  # max change per step, avoids a discontinuity


_META_ARM = os.environ.get("USERRL_META_ARM", "online").strip().lower()


class _ClipScheduleConfig:
    """Stat-on-access hot reload; keeps the last good value on any error.

    Reused for every hot-reloadable scalar knob; `tag` only selects the log
    prefix so two instances are distinguishable in the training log.
    """

    def __init__(self, path: str, tag: str = "CLIP_SCHEDULE") -> None:
        self._path = path
        self._tag = tag
        self._cfg = None
        self._stamp = None

    def get(self):
        """Current override dict, or None when the file is absent/invalid."""
        if _META_ARM == "off":
            return None
        if _META_ARM == "frozen" and self._cfg is not None:
            return self._cfg
        try:
            st = os.stat(self._path)
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            return None  # no file -> hydra config wins, zero behavior change
        if stamp == self._stamp and self._cfg is not None:
            return self._cfg
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                new = json.load(f)
            if not isinstance(new, dict):
                raise ValueError(f"{self._tag} config root must be an object")
        except Exception as e:  # noqa: BLE001
            print(f"[{self._tag}] ignoring unreadable {self._path}: {e}")
            return self._cfg
        self._cfg = new
        self._stamp = stamp
        print(f"[{self._tag}] loaded {self._path}: {json.dumps(new)}")
        return new


_CLIP_SCHEDULE = _ClipScheduleConfig(CLIP_SCHEDULE_PATH)


ENTROPY_SCHEDULE_PATH = os.environ.get(
    "USERRL_ENTROPY_SCHEDULE_CONFIG",
    os.path.join(os.getcwd(), "entropy_schedule.json"),
)
# Hard bounds. A typo of 0.5 instead of 0.005 would drown the policy gradient,
# so the ceiling is enforced here and not only in the gate.
_ENTROPY_FLOOR, _ENTROPY_CEIL = 0.0, 0.05
_ENTROPY_MAX_DELTA = 0.002  # per step, so an edit ramps instead of stepping

_ENTROPY_SCHEDULE = _ClipScheduleConfig(ENTROPY_SCHEDULE_PATH, tag="ENTROPY_SCHEDULE")


def _scheduled_entropy_coeff(global_steps: int, previous):
    """Resolve entropy_coeff for this step, or None to use the hydra config.

    Shape: {"enable": true, "value": 0.005} for a constant override, or
    {"enable": true, "init": 0.0, "final": 0.01, "steps": 50} for a ramp.
    Absent/disabled file -> None -> static config, zero behavior change.
    """
    cfg = _ENTROPY_SCHEDULE.get()
    if not cfg or not cfg.get("enable", False):
        return None

    if cfg.get("value") is not None:
        target = float(cfg["value"])
    else:
        init = float(cfg.get("init", 0.0))
        final = float(cfg.get("final", 0.0))
        steps = max(int(cfg.get("steps", 50)), 2)
        frac = min(max((global_steps - 1) / (steps - 1), 0.0), 1.0)
        target = init + frac * (final - init)

    clamped = min(max(target, _ENTROPY_FLOOR), _ENTROPY_CEIL)
    if clamped != target:
        print(f"[ENTROPY_SCHEDULE] clamped entropy_coeff {target:.5f} -> {clamped:.5f} "
              f"(allowed [{_ENTROPY_FLOOR}, {_ENTROPY_CEIL}])")
    if previous is not None:
        delta = clamped - previous
        if abs(delta) > _ENTROPY_MAX_DELTA:
            clamped = previous + (_ENTROPY_MAX_DELTA if delta > 0 else -_ENTROPY_MAX_DELTA)
    return clamped


def _scheduled_clip_high(hydra_sched, global_steps: int, previous):
    """Resolve clip_ratio_high for this step.

    Precedence: hot file (when enabled) > hydra config > None (static config).
    `override` short-circuits the ramp so an operator can pin a value mid-run.
    The result is rate-limited against `previous` so editing the file changes
    the curve smoothly instead of stepping it.
    """
    hot = _CLIP_SCHEDULE.get()
    sched, source = hydra_sched, "hydra"
    if hot and hot.get("enable", False):
        sched, source = hot, "hotfile"
    if not sched or not sched.get("enable", False):
        return None, source

    override = sched.get("override")
    if override is not None:
        target = float(override)
    else:
        init = float(sched.get("init", 0.2))
        final = float(sched.get("final", 5.0))
        steps = max(int(sched.get("steps", 100)), 2)
        frac = min(max((global_steps - 1) / (steps - 1), 0.0), 1.0)
        target = init + frac * (final - init)

    clamped = min(max(target, _CLIP_HIGH_FLOOR), _CLIP_HIGH_CEIL)
    if clamped != target:
        print(f"[CLIP_SCHEDULE] clamped clip_ratio_high {target:.4f} -> {clamped:.4f} "
              f"(allowed [{_CLIP_HIGH_FLOOR}, {_CLIP_HIGH_CEIL}])")
    target = clamped
    # Rate-limit the HOT path only. The hydra ramp is a fixed curve declared
    # before the run, so smoothing it would silently change existing behavior;
    # the limiter exists for mid-run edits, which can only come from the file.
    if source == "hotfile" and previous is not None:
        max_delta = abs(float(sched.get("max_delta_per_step", _CLIP_DEFAULT_MAX_DELTA)))
        if max_delta > 0:
            target = min(max(target, previous - max_delta), previous + max_delta)
    return target, source


def _discounted_reward_to_go(rewards: list[float], gamma: float) -> float:
    total = 0.0
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
    return float(total)


def _compute_task_reward_metrics(batch: DataProto, turn_reward_mode: str = "raw", gamma: float = 1.0) -> dict[str, float]:
    """Aggregate task reward with aliases matching the active multi-turn reward mode."""
    raw_reward_scores = batch.non_tensor_batch.get("reward_scores")
    reward_items = raw_reward_scores.tolist() if raw_reward_scores is not None and isinstance(raw_reward_scores, np.ndarray) else (
        list(raw_reward_scores) if raw_reward_scores is not None else []
    )

    raw_values = []
    for item in reward_items:
        while isinstance(item, (list, tuple, np.ndarray)) and len(item) == 1:
            item = item[0]
        if not isinstance(item, dict) or "interact_with_env_original" not in item:
            continue
        try:
            value = float(item["interact_with_env_original"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            raw_values.append(value)

    histories = _unwrap_conversation_histories(batch.non_tensor_batch.get("conversation_histories"))
    effective_values = []
    raw_history_values = []
    for history in histories:
        if not history:
            continue
        rewards = []
        for turn in history:
            if not isinstance(turn, dict) or "reward" not in turn:
                continue
            try:
                reward = float(turn.get("original_env_reward", turn.get("reward", 0.0)) or 0.0)
            except (TypeError, ValueError):
                continue
            if np.isfinite(reward):
                rewards.append(reward)
        if not rewards:
            continue
        raw_sum = float(sum(rewards))
        raw_history_values.append(raw_sum)
        if turn_reward_mode == "r2g":
            effective_values.append(_discounted_reward_to_go(rewards, gamma))
        elif turn_reward_mode == "terminal_only":
            effective_values.append(float(rewards[-1]))
        else:
            effective_values.append(raw_sum)

    if not effective_values:
        effective_values = raw_values
    if not raw_history_values:
        raw_history_values = raw_values

    total = max(len(histories) if histories else len(reward_items), 1)
    metrics = {"task_reward/available_fraction": float(len(effective_values) / total)}
    if not effective_values:
        return metrics

    rewards = np.asarray(effective_values, dtype=np.float32)
    metrics.update(
        {
            "task_reward/mean": float(rewards.mean()),
            "task_reward/min": float(rewards.min()),
            "task_reward/max": float(rewards.max()),
            "task_reward/success_rate": float((rewards > 0.0).mean()),
        }
    )
    if raw_history_values:
        raw_rewards = np.asarray(raw_history_values, dtype=np.float32)
        metrics.update(
            {
                "task_reward/traj_sum_mean": float(raw_rewards.mean()),
                "task_reward/raw_mean": float(raw_rewards.mean()),
                "task_reward/raw_min": float(raw_rewards.min()),
                "task_reward/raw_max": float(raw_rewards.max()),
                "task_reward/raw_success_rate": float((raw_rewards > 0.0).mean()),
            }
        )
    return metrics


def _compute_probe_reward_metrics(batch: DataProto) -> dict[str, float]:
    """Aggregate probe levels, gates, and reward components when present."""
    raw_reward_scores = batch.non_tensor_batch.get("reward_scores")
    if raw_reward_scores is None:
        return {}

    reward_items = raw_reward_scores.tolist() if isinstance(raw_reward_scores, np.ndarray) else list(raw_reward_scores)
    field_names = {
        "probe_satisfaction_initial": "initial",
        "probe_satisfaction_final": "final",
        "probe_satisfaction_improvement": "improvement",
        "probe_satisfaction_delta_mean": "delta_mean",
        "probe_task_reward_component_sum": "task_component_sum",
        "probe_delta_reward_component_sum": "delta_component_sum",
        "probe_gate": "gate",
        "probe_task_group_std": "task_group_std",
        "probe_group_size": "group_size",
        "probe_gated_reward_component_sum": "gated_component_sum",
    }
    values_by_field = {field: [] for field in field_names}

    for item in reward_items:
        while isinstance(item, (list, tuple, np.ndarray)) and len(item) == 1:
            item = item[0]
        if not isinstance(item, dict):
            continue
        for field in field_names:
            try:
                value = float(item[field])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values_by_field[field].append(value)

    metrics = {}
    for field, label in field_names.items():
        values = values_by_field[field]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float32)
        metrics[f"probe_reward/{label}/mean"] = float(array.mean())
        metrics[f"probe_reward/{label}/min"] = float(array.min())
        metrics[f"probe_reward/{label}/max"] = float(array.max())
    return metrics


def _build_probe_feature_tensor(batch: DataProto, feature_dim: int = 2) -> torch.Tensor:
    """Build per-token privileged probe features for the asymmetric critic.

    Channel 0 holds the per-turn satisfaction probe score, spread over that
    turn's response span; channel 1 marks positions that carry a valid score.
    If enabled, channel 2 holds a format-error feature value, also spread over
    the offending response span. If feature_dim >= 4, channel 3 is the matching
    format-error mask, so the critic can distinguish "no format signal" from a
    zero-valued format feature. If feature_dim >= 6, channel 4 holds the turn's
    raw env (intermediate) reward and channel 5 is its mask; the mask is set on
    every real turn span since a zero turn reward is a meaningful "no progress"
    signal. Prompt positions and turns without a signal stay zero. Only the critic consumes these features; they never enter the
    actor-visible reward, so the policy cannot optimize privileged signals
    directly.

    Returns a float tensor of shape (batch, input_ids_seqlen, feature_dim).
    """
    input_ids = batch.batch["input_ids"]
    batch_size, seq_length = input_ids.shape
    response_length = batch.batch["responses"].size(1)
    feature_dim = max(int(feature_dim), 2)
    features = torch.zeros((batch_size, seq_length, feature_dim), dtype=torch.float32, device=input_ids.device)
    histories = _unwrap_conversation_histories(batch.non_tensor_batch.get("conversation_histories"))
    turn_boundaries = batch.batch.get("turn_boundaries")
    if turn_boundaries is None:
        return features

    response_offset = seq_length - response_length
    for batch_index in range(batch_size):
        if batch_index >= len(histories) or not isinstance(histories[batch_index], (list, tuple)):
            continue
        turn_starts = torch.where(turn_boundaries[batch_index] == 1)[0].tolist()
        if not turn_starts:
            turn_starts = [0]
        turn_starts.append(response_length)

        for turn_index, turn in enumerate(histories[batch_index]):
            if turn_index >= len(turn_starts) - 1 or not isinstance(turn, dict):
                break
            start = response_offset + turn_starts[turn_index]
            end = response_offset + turn_starts[turn_index + 1]
            try:
                score = float(turn.get("probe_satisfaction"))
            except (TypeError, ValueError):
                score = None
            if score is not None and np.isfinite(score):
                features[batch_index, start:end, 0] = score
                features[batch_index, start:end, 1] = 1.0
            if feature_dim >= 3:
                try:
                    format_signal = float(turn.get("format_error_feature", 0.0))
                except (TypeError, ValueError):
                    format_signal = 0.0
                if np.isfinite(format_signal) and format_signal != 0.0:
                    features[batch_index, start:end, 2] = format_signal
                    if feature_dim >= 4:
                        features[batch_index, start:end, 3] = 1.0
            if feature_dim >= 6:
                # Channels 4/5: the turn's raw env (intermediate) reward as a
                # privileged critic feature. Unlike satisfaction/format, a turn
                # reward of 0 is a real signal ("no progress this turn"), so the
                # mask is set whenever the turn exists, not only on non-zero.
                try:
                    turn_reward_signal = float(turn.get("reward", 0.0) or 0.0)
                except (TypeError, ValueError):
                    turn_reward_signal = 0.0
                if np.isfinite(turn_reward_signal):
                    features[batch_index, start:end, 4] = turn_reward_signal
                    features[batch_index, start:end, 5] = 1.0

    return features


def _safe_pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation, or None when it would not be meaningful."""
    if x.size < 4 or float(x.std()) < 1e-8 or float(y.std()) < 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _compute_probe_signal_metrics(batch: DataProto) -> dict:
    """Per-step diagnostics for the asymmetric critic's privileged probe signal.

    Aggregates the satisfaction probe scores of every trajectory and reports
    their correlation with (a) the trajectory return and (b) the residual the
    current value model leaves unexplained. ``corr_sat_*_residual`` is the
    number that matters for the probe head: if it stays near zero, the probe
    features carry no information beyond what the backbone critic already reads
    from the transcript, and no optimizer tuning will make the head learn.
    """
    histories = _unwrap_conversation_histories(batch.non_tensor_batch.get("conversation_histories"))
    if not histories or "returns" not in batch.batch.keys() or "values" not in batch.batch.keys():
        return {}

    keep, sat_means, sat_finals = [], [], []
    for i, history in enumerate(histories):
        if not isinstance(history, (list, tuple)):
            continue
        scores = []
        for turn in history:
            if not isinstance(turn, dict):
                continue
            try:
                score = float(turn.get("probe_satisfaction"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(score):
                scores.append(score)
        if not scores:
            continue
        keep.append(i)
        sat_means.append(float(np.mean(scores)))
        sat_finals.append(float(scores[-1]))
    if len(keep) < 4:
        return {}

    index = torch.as_tensor(keep, dtype=torch.long, device=batch.batch["returns"].device)
    traj_return = batch.batch["returns"].detach().float().index_select(0, index)[:, 0].cpu().numpy()
    traj_value = batch.batch["values"].detach().float().index_select(0, index)[:, 0].cpu().numpy()
    residual = traj_return - traj_value
    sat_means = np.asarray(sat_means, dtype=np.float64)
    sat_finals = np.asarray(sat_finals, dtype=np.float64)

    metrics = {
        "probe_signal/n_scored": float(len(keep)),
        "probe_signal/sat_mean": float(sat_means.mean()),
        "probe_signal/sat_std": float(sat_means.std()),
    }
    for label, corr in {
        "corr_sat_mean_return": _safe_pearson_corr(sat_means, traj_return),
        "corr_sat_mean_residual": _safe_pearson_corr(sat_means, residual),
        "corr_sat_final_return": _safe_pearson_corr(sat_finals, traj_return),
        "corr_sat_final_residual": _safe_pearson_corr(sat_finals, residual),
    }.items():
        if corr is not None:
            metrics[f"probe_signal/{label}"] = corr
    return metrics


def _build_probe_intrinsic_reward_tensor(batch: DataProto, source: str = "satisfaction") -> torch.Tensor:
    """Map per-turn probe values from rollout histories onto response token spans."""
    field_by_source = {
        "satisfaction": "probe_satisfaction",
        "td_delta": "probe_satisfaction_delta",
        # Per-turn environment step rewards (dense reward shaping for multiturn GAE).
        "env_step": "reward",
    }
    if source not in field_by_source:
        raise ValueError(f"Unsupported probe intrinsic source: {source}")
    probe_field = field_by_source[source]
    token_level_rewards = batch.batch["token_level_rewards"]
    probe_rewards = torch.zeros_like(token_level_rewards)
    histories = _unwrap_conversation_histories(batch.non_tensor_batch.get("conversation_histories"))
    turn_boundaries = batch.batch.get("turn_boundaries")
    if turn_boundaries is None:
        return probe_rewards

    batch_size, sequence_length = probe_rewards.shape
    for batch_index in range(batch_size):
        if batch_index >= len(histories) or not isinstance(histories[batch_index], (list, tuple)):
            continue
        turn_starts = torch.where(turn_boundaries[batch_index] == 1)[0].tolist()
        if not turn_starts:
            turn_starts = [0]
        turn_starts.append(sequence_length)

        for turn_index, turn in enumerate(histories[batch_index]):
            if turn_index >= len(turn_starts) - 1 or not isinstance(turn, dict):
                break
            try:
                score = float(turn.get(probe_field))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(score):
                continue
            probe_rewards[batch_index, turn_starts[turn_index] : turn_starts[turn_index + 1]] = score

    return probe_rewards


def _normalize_multiturn_probe_reward(
    probe_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
) -> torch.Tensor:
    """Preserve one probe reward per action turn for turn-level GAE.

    Probe scores are initially copied across a turn's response span so the
    InfoGRPO token-level implementation can consume them. Multi-turn GAE
    reduces trainable action tokens to one reward per environment turn, so the
    copied score must be divided over that turn's action tokens first.
    """
    normalized = torch.zeros_like(probe_rewards)
    batch_size, response_length = probe_rewards.shape
    for batch_idx in range(batch_size):
        starts = torch.where(turn_boundaries[batch_idx, :response_length] == 1)[0].tolist()
        if not starts:
            starts = [0]
        starts = sorted(set(starts))
        starts.append(response_length)
        for turn_idx, start in enumerate(starts[:-1]):
            end = starts[turn_idx + 1]
            positions = torch.where(action_mask[batch_idx, start:end] > 0)[0] + start
            if positions.numel() == 0:
                continue
            normalized[batch_idx, positions] = probe_rewards[batch_idx, positions] / positions.numel()
    return normalized


def _json_safe(obj):
    if hasattr(obj, "model_dump"):
        return _json_safe(obj.model_dump())
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _json_dumps_limited(obj, max_chars: int) -> str:
    text = json.dumps(_json_safe(obj), ensure_ascii=False)
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def _tensor_sequence_sums(batch: DataProto, key: str):
    if key not in batch.batch:
        return None
    values = batch.batch[key].detach().float().cpu()
    if values.ndim > 1:
        values = values.sum(dim=-1)
    return values.tolist()


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, turn_level_method="Equalized", trajectory_score_method="Sum", norm_adv_by_std_in_grpo=True, config=None, current_epoch=0):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        turn_level_method (str, optional): Method to assign credits to turns, "Equalized" or "R2G" or "EM". Defaults to "Equalized".
        trajectory_score_method (str, optional): Method to compute trajectory score, "Sum" or "R2G". Defaults to "Sum".
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        use_multiturn_gae = (
            multi_turn
            and config.get("multiturn_gae", {}).get("enable", False)
            and "turn_boundaries" in data.batch
            and "loss_mask" in data.batch
        )
        if use_multiturn_gae:
            response_length = data.batch["responses"].size(1)
            action_mask = data.batch["loss_mask"][:, -response_length:].to(data.batch["values"].dtype)
            advantages, returns = core_algos.compute_multiturn_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                action_mask=action_mask,
                turn_boundaries=data.batch["turn_boundaries"],
                gamma=gamma,
                lam=lam,
                turn_reward_mode=config.get("multiturn_gae", {}).get("turn_reward_mode", "raw"),
            )
            # The critic must regress values only on agent action tokens.
            data.batch["response_mask"] = action_mask
        else:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_MULTITURN:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        conversation_histories = _unwrap_conversation_histories(data.non_tensor_batch["conversation_histories"])
        data_sources = data.non_tensor_batch["data_source"].tolist()  # unwrap from list

        advantages, returns = core_algos.compute_grpo_multiturn_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            turn_boundaries=data.batch["turn_boundaries"],
            conversation_histories=conversation_histories,
            data_sources=data_sources,
            index=data.non_tensor_batch["uid"],
            gamma=gamma,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            turn_level_method=turn_level_method,
            trajectory_score_method=trajectory_score_method,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.INFO_GRPO:
        # Info-GRPO: separate branch for orthogonal decomposition of outcome and intrinsic rewards
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]

        # Extract intrinsic rewards (required for info_grpo)
        if "intrinsic_reward" in data.batch:
            token_level_intrinsic_rewards = data.batch["intrinsic_reward"]
        else:
            # If intrinsic rewards are not available, create zero tensor with same shape as rewards
            token_level_intrinsic_rewards = torch.zeros_like(data.batch["token_level_rewards"])

        # Get intrinsic weight from batch or config
        if "intrinsic_weight" in data.batch:
            intrinsic_weight = data.batch["intrinsic_weight"]
        elif config is not None and "intrinsic_reward" in config:
            intrinsic_weight = config["intrinsic_reward"].get("intrinsic_weight", 0.1)
        else:
            intrinsic_weight = 0.1

        # Get intrinsic decay rate from config
        if config is not None and "intrinsic_reward" in config:
            intrinsic_decay_rate = config["intrinsic_reward"].get("intrinsic_decay_rate", 0.0)
        else:
            intrinsic_decay_rate = 0.0

        turn_boundaries = None
        data_sources = None
        conversation_histories = _unwrap_conversation_histories(data.non_tensor_batch["conversation_histories"])

        if multi_turn:
            if "turn_boundaries" in data.batch:
                turn_boundaries = data.batch["turn_boundaries"]
            if "data_source" in data.non_tensor_batch:
                data_sources = data.non_tensor_batch["data_source"].tolist() if isinstance(data.non_tensor_batch["data_source"], np.ndarray) else data.non_tensor_batch["data_source"]

        # Get additional intrinsic reward parameters from config
        intrinsic_gate_temperature = 0.05
        advantage_weight_clip = 1.0
        advantage_weight_threshold = 0.1
        normalize_intrinsic = True  # Default to True for backward compatibility
        use_intrinsic_only = False  # Default to False for backward compatibility
        if config is not None and "intrinsic_reward" in config:
            intrinsic_gate_temperature = config["intrinsic_reward"].get("intrinsic_gate_temperature", 0.05)
            advantage_weight_clip = config["intrinsic_reward"].get("advantage_weight_clip", 1.0)
            advantage_weight_threshold = config["intrinsic_reward"].get("advantage_weight_threshold", 0.1)
            normalize_intrinsic = config["intrinsic_reward"].get("normalize_intrinsic", True)
            use_intrinsic_only = config["intrinsic_reward"].get("use_intrinsic_only", False)

        # Call info_grpo advantage estimator
        advantages, returns, adv_stats = core_algos.compute_info_grpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            token_level_intrinsic_rewards=token_level_intrinsic_rewards,
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            intrinsic_weight=intrinsic_weight,
            intrinsic_decay_rate=intrinsic_decay_rate,
            current_epoch=current_epoch,
            # Multi-turn parameters (if available)
            turn_boundaries=turn_boundaries,
            conversation_histories=conversation_histories,
            data_sources=data_sources,
            gamma=gamma,
            turn_level_method=turn_level_method,
            trajectory_score_method=trajectory_score_method,
            # Additional intrinsic reward parameters
            intrinsic_gate_temperature=intrinsic_gate_temperature,
            advantage_weight_clip=advantage_weight_clip,
            advantage_weight_threshold=advantage_weight_threshold,
            normalize_intrinsic=normalize_intrinsic,  # Pass normalization flag for ablation
            use_intrinsic_only=use_intrinsic_only,  # Pass intrinsic-only flag for ablation
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

        # Store advantage stats for monitoring
        # meta_info is a Dict attribute of DataProto, not part of TensorDict
        if "adv_stats" not in data.meta_info:
            data.meta_info["adv_stats"] = {}
        data.meta_info["adv_stats"] = adv_stats

    else:
        # handle all other adv estimator type other than GAE, GRPO, GRPO_MULTITURN, and info_grpo
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GRPO_MULTITURN,
        ] or self.config.algorithm.adv_estimator == AdvantageEstimator.INFO_GRPO:
            # info_grpo is a GRPO variant that doesn't need critic
            self.use_critic = False
        else:
            raise NotImplementedError(f"Unsupported adv_estimator: {self.config.algorithm.adv_estimator}")

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.best_valid_score = -1

        # RAGEN Extension: Initialize rollout filter (optional, controlled by config)
        self.rollout_filter = self._create_rollout_filter()

    def _create_rollout_filter(self):
        """Create rollout filter from config (RAGEN extension).

        Returns None if rollout filtering is not enabled in config.
        """
        # Check if RAGEN extensions are enabled in config
        if not hasattr(self.config, 'ragen') or not hasattr(self.config.ragen, 'rollout_filter'):
            return None

        filter_config_dict = self.config.ragen.rollout_filter

        # Create RolloutFilterConfig from config dict
        filter_config = RolloutFilterConfig(
            enable=filter_config_dict.get('enable', False),
            ratio=filter_config_dict.get('ratio', 0.25),
            filter_type=filter_config_dict.get('filter_type', 'largest'),
            metric=filter_config_dict.get('metric', 'reward_variance'),
            group_size=filter_config_dict.get('group_size', self.config.actor_rollout_ref.rollout.n),
        )

        # Return None if not enabled
        if not filter_config.enable:
            return None

        # Create filter (compute_log_prob needed for entropy filter, can be added later)
        rollout_filter = create_rollout_filter(filter_config, compute_log_prob=None)

        print(f"[RAGEN Extension] Rollout filtering enabled: metric={filter_config.metric}, "
              f"ratio={filter_config.ratio}, filter_type={filter_config.filter_type}")

        return rollout_filter

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            # InteractComp rollout doesn't use tool calling, so skip tool_config_path check
            is_interactcomp = "interactcomp" in config.actor_rollout_ref.rollout.name.lower()
            if not is_interactcomp:
                assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GAE, AdvantageEstimator.GRPO, AdvantageEstimator.GRPO_MULTITURN, AdvantageEstimator.INFO_GRPO], "only GAE and GRPO variants are supported for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _analyze_group_reward_consistency(self, batch: DataProto, metrics: dict):
        """Analyze reward consistency within groups and save sample trajectories.

        This function:
        1. Groups trajectories by uid (prompt group)
        2. Checks if all rewards in a group are equal
        3. Computes the ratio of groups with equal rewards
        4. Saves one example trajectory for reward=1 groups and one for reward=0 groups

        Note: This analyzes EXTERNAL rewards (not weighted with intrinsic rewards).
        """
        # Get external rewards (not weighted with intrinsic rewards)
        # If extrinsic_reward is available (Info-GRPO mode), use it; otherwise use token_level_rewards
        if "extrinsic_reward" in batch.batch:
            external_rewards = batch.batch["extrinsic_reward"]
        else:
            # Fallback to token_level_rewards if extrinsic_reward not available
            external_rewards = batch.batch["token_level_rewards"]

        sequence_rewards = external_rewards.sum(dim=-1).cpu().numpy()  # (bsz,)
        uids = batch.non_tensor_batch.get("uid", None)

        if uids is None:
            print("[DEBUG_REWARD_CONSISTENCY] No uid found in batch, skipping analysis")
            return

        bsz = len(sequence_rewards)

        # Group by uid
        uid2rewards = defaultdict(list)
        uid2indices = defaultdict(list)
        for i in range(bsz):
            uid = uids[i]
            uid2rewards[uid].append(sequence_rewards[i])
            uid2indices[uid].append(i)

        # Analyze consistency: count groups where all rewards are equal
        groups_with_equal_rewards = 0
        total_groups = len(uid2rewards)
        groups_all_one = []
        groups_all_zero = []
        groups_mixed = []
        groups_partial = []

        for uid, rewards in uid2rewards.items():
            rewards_array = np.array(rewards)
            if len(rewards_array) > 0:
                if np.all(rewards_array == rewards_array[0]):  # All rewards are equal
                    groups_with_equal_rewards += 1
                    if np.allclose(rewards_array, 1.0):
                        groups_all_one.append(uid)
                    elif np.allclose(rewards_array, 0.0):
                        groups_all_zero.append(uid)
                    else:
                        groups_partial.append(uid)
                else:
                    groups_mixed.append(uid)

        # Compute ratio
        ratio_equal_rewards = groups_with_equal_rewards / total_groups if total_groups > 0 else 0.0

        # Update metrics
        metrics["debug/group_reward_equal_ratio"] = ratio_equal_rewards
        metrics["debug/total_groups"] = total_groups
        metrics["debug/groups_equal_rewards"] = groups_with_equal_rewards
        metrics["debug/groups_all_one"] = len(groups_all_one)
        metrics["debug/groups_all_zero"] = len(groups_all_zero)
        metrics["debug/groups_mixed"] = len(groups_mixed)
        metrics["debug/groups_partial"] = len(groups_partial)

        print(f"[DEBUG_REWARD_CONSISTENCY] Step {self.global_steps}: "
              f"Total groups={total_groups}, Groups with equal rewards={groups_with_equal_rewards} "
              f"({ratio_equal_rewards*100:.2f}%), "
              f"All-1 groups={len(groups_all_one)}, All-0 groups={len(groups_all_zero)}, "
              f"Mixed groups={len(groups_mixed)}")

        # Save real training-batch samples on a configurable review cadence.
        dump_freq = int(self.config.trainer.get("debug_trajectory_freq", 10))
        save_samples = dump_freq > 0 and (self.global_steps == 1 or self.global_steps % dump_freq == 0)
        if save_samples:
            debug_dir = os.path.join(self.config.trainer.default_local_dir, "debug_trajectories")
            os.makedirs(debug_dir, exist_ok=True)

            # Decode inputs and outputs for analysis
            prompts = batch.batch["prompts"]  # (bsz, prompt_len)
            responses = batch.batch["responses"]  # (bsz, response_len)
            inputs_decoded = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in prompts]
            outputs_decoded = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]

            def _arm_of(i):
                try:
                    from verl.workers.reward_manager.naive import (
                        _resolve_user_strategy, _task_id_of,
                    )
                    from verl.workers.reward_manager.reward_shaping import gym_of
                    item = batch[i]
                    src = item.non_tensor_batch.get("data_source")
                    if hasattr(src, "__len__") and not isinstance(src, str):
                        src = src[0]
                    gym = gym_of(str(src)) if src is not None else None
                    tid = _task_id_of(item)
                    if gym is None or tid is None:
                        return None
                    return _resolve_user_strategy(gym, tid, None)
                except Exception:  # noqa: BLE001
                    return None

            n_keep = max(1, int(getattr(self.config.trainer, "log_train_trajectories", 1) or 1))

            _gym_of_index = {}
            try:
                from verl.workers.reward_manager.reward_shaping import gym_of as _gym_of
                _ds = batch.non_tensor_batch.get("data_source")
                if _ds is not None:
                    for _i in range(bsz):
                        _src = _ds[_i]
                        if hasattr(_src, "__len__") and not isinstance(_src, str):
                            _src = _src[0] if len(_src) > 0 else ""
                        _gym_of_index[_i] = _gym_of(_src)
            except Exception:  # noqa: BLE001
                _gym_of_index = {}

            def _stratify_by_gym(uid_list):
                """Round-robin the uids over their gyms, preserving order within
                each gym. Falls back to the original order when the gym of a uid
                cannot be determined."""
                if not _gym_of_index:
                    return list(uid_list)
                by_gym = defaultdict(list)
                for _uid in uid_list:
                    _g = ""
                    for _i in uid2indices.get(_uid, []):
                        _g = _gym_of_index.get(_i) or ""
                        if _g:
                            break
                    by_gym[_g].append(_uid)
                out = []
                order = sorted(by_gym)
                # Rotate the starting gym per DUMP, not per step. n_keep is 4
                # and the mix has 5 gyms, so a fixed alphabetical order starves
                # whichever gym sorts last on every dumped step -- the same
                # structural blind spot, just moved.
                # The first version of this used `self.global_steps % len(order)`
                # and was silently a no-op: dumps fire on multiples of 10 and
                # 10 % 5 == 0, so the offset was 0 every single time and turtle
                # was still never sampled. A counter that advances once per dump
                # is correct whatever the dump cadence and gym count are.
                if order:
                    _off = getattr(self, "_gym_dump_rotation", 0) % len(order)
                    order = order[_off:] + order[:_off]
                while len(out) < len(uid_list):
                    progressed = False
                    for _g in order:
                        if by_gym[_g]:
                            out.append(by_gym[_g].pop(0))
                            progressed = True
                    if not progressed:
                        break
                return out

            groups_all_one = _stratify_by_gym(groups_all_one)
            groups_all_zero = _stratify_by_gym(groups_all_zero)
            groups_partial = _stratify_by_gym(groups_partial)
            groups_mixed = _stratify_by_gym(groups_mixed)
            self._gym_dump_rotation = getattr(self, "_gym_dump_rotation", 0) + 1

            # Save examples from reward=1 groups
            for rank, sample_uid in enumerate(groups_all_one[:n_keep]):
                sample_indices = uid2indices[sample_uid]
                # Save all trajectories from this group
                sample_data = {
                    "sample_origin": "training_batch",
                    "sampling_strategy": "outcome_stratified_by_gym",
                    "biased_sample": True,
                    "uid": sample_uid,
                    "reward_type": "all_one",
                    "global_step": self.global_steps,
                    "group_size": len(sample_indices),
                    "trajectories": []
                }
                for idx in sample_indices:
                    sample_data["trajectories"].append({
                        "index": int(idx),
                        "reward": float(sequence_rewards[idx]),
                        "user_strategy": _arm_of(int(idx)),
                        "input": inputs_decoded[idx],
                        "output": outputs_decoded[idx],
                    })

                # rank 0 keeps the historical filename so existing readers and
                # meta_agent/observer.py's glob keep working unchanged.
                suffix = "" if rank == 0 else f"_{rank}"
                filename = os.path.join(debug_dir, f"step_{self.global_steps}_reward_all_one{suffix}.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                print(f"[DEBUG_REWARD_CONSISTENCY] Saved reward=1 example to {filename}")

            # Save partial-credit groups separately from outcome extremes.
            for rank, sample_uid in enumerate(groups_partial[:n_keep]):
                sample_indices = uid2indices[sample_uid]
                sample_data = {
                    "sample_origin": "training_batch",
                    "sampling_strategy": "outcome_stratified_by_gym",
                    "biased_sample": True,
                    "uid": sample_uid,
                    "reward_type": "partial",
                    "global_step": self.global_steps,
                    "group_size": len(sample_indices),
                    "trajectories": []
                }
                for idx in sample_indices:
                    sample_data["trajectories"].append({
                        "index": int(idx),
                        "reward": float(sequence_rewards[idx]),
                        "user_strategy": _arm_of(int(idx)),
                        "input": inputs_decoded[idx],
                        "output": outputs_decoded[idx],
                    })
                suffix = "" if rank == 0 else f"_{rank}"
                filename = os.path.join(debug_dir, f"step_{self.global_steps}_reward_partial{suffix}.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                print(f"[DEBUG_REWARD_CONSISTENCY] Saved partial-credit example to {filename}")

            # Mixed outcome groups are common under GRPO and carry the
            # within-prompt comparisons used to compute policy advantages.
            for rank, sample_uid in enumerate(groups_mixed[:n_keep]):
                sample_indices = uid2indices[sample_uid]
                sample_data = {
                    "sample_origin": "training_batch",
                    "sampling_strategy": "outcome_stratified_by_gym",
                    "biased_sample": True,
                    "uid": sample_uid,
                    "reward_type": "mixed",
                    "global_step": self.global_steps,
                    "group_size": len(sample_indices),
                    "trajectories": [
                        {
                            "index": int(idx),
                            "reward": float(sequence_rewards[idx]),
                            "user_strategy": _arm_of(int(idx)),
                            "input": inputs_decoded[idx],
                            "output": outputs_decoded[idx],
                        }
                        for idx in sample_indices
                    ],
                }
                suffix = "" if rank == 0 else f"_{rank}"
                filename = os.path.join(debug_dir, f"step_{self.global_steps}_reward_mixed{suffix}.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                print(f"[DEBUG_REWARD_CONSISTENCY] Saved mixed-outcome example to {filename}")

            # Save examples from reward=0 groups
            for rank, sample_uid in enumerate(groups_all_zero[:n_keep]):
                sample_indices = uid2indices[sample_uid]
                sample_data = {
                    "sample_origin": "training_batch",
                    "sampling_strategy": "outcome_stratified_by_gym",
                    "biased_sample": True,
                    "uid": sample_uid,
                    "reward_type": "all_zero",
                    "global_step": self.global_steps,
                    "group_size": len(sample_indices),
                    "trajectories": []
                }
                for idx in sample_indices:
                    sample_data["trajectories"].append({
                        "index": int(idx),
                        "reward": float(sequence_rewards[idx]),
                        "user_strategy": _arm_of(int(idx)),
                        "input": inputs_decoded[idx],
                        "output": outputs_decoded[idx],
                    })

                suffix = "" if rank == 0 else f"_{rank}"
                filename = os.path.join(debug_dir, f"step_{self.global_steps}_reward_all_zero{suffix}.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                print(f"[DEBUG_REWARD_CONSISTENCY] Saved reward=0 example to {filename}")

            # Optional targeted per-gym samples prefer failing groups.
            # Both these and outcome-stratified samples are diagnostic evidence,
            # not an unbiased estimate of the training population's success rate.
            target_gyms = [
                g.strip() for g in os.environ.get("USERRL_DEBUG_DUMP_GYMS", "").split(",") if g.strip()
            ]
            data_sources = batch.non_tensor_batch.get("data_source") if target_gyms else None
            if data_sources is not None:
                from verl.workers.reward_manager.reward_shaping import gym_of

                gym_of_index = {}
                for i in range(bsz):
                    source = data_sources[i]
                    if hasattr(source, "__len__") and not isinstance(source, str):
                        source = source[0] if len(source) > 0 else ""
                    gym_of_index[i] = gym_of(source)

                for gym in target_gyms:
                    # uids whose members belong to this gym
                    gym_uids = {
                        uid: indices for uid, indices in uid2indices.items()
                        if any(gym_of_index.get(i) == gym for i in indices)
                    }
                    if not gym_uids:
                        continue
                    # prefer a failing group -- that is the one worth reading
                    sample_uid = next(
                        (uid for uid in gym_uids if uid in set(groups_all_zero)),
                        next(iter(gym_uids)),
                    )
                    sample_indices = gym_uids[sample_uid]
                    rewards_array = np.array([sequence_rewards[i] for i in sample_indices])
                    if np.allclose(rewards_array, 0.0):
                        reward_type = "all_zero"
                    elif np.allclose(rewards_array, rewards_array[0]):
                        reward_type = "all_one"
                    else:
                        reward_type = "mixed"
                    sample_data = {
                        "sample_origin": "training_batch",
                        "sampling_strategy": "target_gym_failure_preferred",
                        "uid": sample_uid,
                        "gym": gym,
                        "reward_type": reward_type,
                        "biased_sample": True,
                        "global_step": self.global_steps,
                        "group_size": len(sample_indices),
                        "trajectories": [
                            {
                                "index": int(idx),
                                "reward": float(sequence_rewards[idx]),
                                "input": inputs_decoded[idx],
                                "output": outputs_decoded[idx],
                            }
                            for idx in sample_indices
                        ],
                    }
                    filename = os.path.join(debug_dir, f"step_{self.global_steps}_gym_{gym}.json")
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(sample_data, f, ensure_ascii=False, indent=2)
                    print(f"[DEBUG_REWARD_CONSISTENCY] Saved {gym} example ({reward_type}) to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _maybe_log_train_trajectories(self, tracker, batch: DataProto):
        trajectories_to_log = int(self.config.trainer.get("log_train_trajectories", 0) or 0)
        if trajectories_to_log <= 0:
            return
        if "wandb" not in self.config.trainer.logger:
            return
        if os.environ.get("VERL_WANDB_TRAJECTORY_TABLE", "0") in ("", "0", "false", "False"):
            return

        freq = int(self.config.trainer.get("log_train_trajectories_freq", 1) or 1)
        if freq > 1 and self.global_steps % freq != 0:
            return

        wandb = getattr(tracker, "logger", {}).get("wandb", None)
        if wandb is None:
            return

        max_chars = int(self.config.trainer.get("log_train_trajectory_max_chars", 60000) or 60000)
        bsz = len(batch.batch)
        if bsz == 0:
            return

        external_rewards = _tensor_sequence_sums(batch, "extrinsic_reward")
        total_scores = _tensor_sequence_sums(batch, "token_level_scores")
        intrinsic_rewards = _tensor_sequence_sums(batch, "intrinsic_reward")
        if external_rewards is None:
            external_rewards = total_scores

        selected = []
        if external_rewards:
            reward_order = sorted(range(bsz), key=lambda i: external_rewards[i])
            for idx in reward_order[-trajectories_to_log:]:
                if idx not in selected:
                    selected.append(idx)
            for idx in reward_order[:trajectories_to_log]:
                if idx not in selected:
                    selected.append(idx)
        for idx in range(bsz):
            if len(selected) >= trajectories_to_log:
                break
            if idx not in selected:
                selected.append(idx)
        selected = selected[:trajectories_to_log]

        conversation_histories = _unwrap_conversation_histories(batch.non_tensor_batch.get("conversation_histories", None))
        messages = batch.non_tensor_batch.get("messages", None)
        reward_scores = batch.non_tensor_batch.get("reward_scores", None)
        data_sources = batch.non_tensor_batch.get("data_source", None)
        uids = batch.non_tensor_batch.get("uid", None)

        columns = [
            "step",
            "sample_index",
            "uid",
            "data_source",
            "external_reward",
            "intrinsic_reward",
            "total_score",
            "turns",
            "prompt",
            "response",
            "trajectory_json",
            "messages_json",
            "reward_scores_json",
        ]

        if not hasattr(self, "train_trajectory_table"):
            self.train_trajectory_table = wandb.Table(columns=columns)
        table = wandb.Table(columns=columns, data=self.train_trajectory_table.data)

        for idx in selected:
            prompt = self.tokenizer.decode(batch.batch["prompts"][idx].detach().cpu(), skip_special_tokens=True)
            response = self.tokenizer.decode(batch.batch["responses"][idx].detach().cpu(), skip_special_tokens=True)
            conv_hist = conversation_histories[idx] if idx < len(conversation_histories) else []
            msg_item = None
            if messages is not None and idx < len(messages):
                msg_item = messages[idx]
                if isinstance(msg_item, dict) and "messages" in msg_item:
                    msg_item = msg_item["messages"]
            reward_item = reward_scores[idx] if reward_scores is not None and idx < len(reward_scores) else None

            table.add_data(
                self.global_steps,
                int(idx),
                str(uids[idx]) if uids is not None and idx < len(uids) else "",
                str(data_sources[idx]) if data_sources is not None and idx < len(data_sources) else "",
                float(external_rewards[idx]) if external_rewards is not None else None,
                float(intrinsic_rewards[idx]) if intrinsic_rewards is not None else None,
                float(total_scores[idx]) if total_scores is not None else None,
                len(conv_hist) if isinstance(conv_hist, list) else 0,
                prompt[:max_chars] + ("...[truncated]" if len(prompt) > max_chars else ""),
                response[:max_chars] + ("...[truncated]" if len(response) > max_chars else ""),
                _json_dumps_limited(conv_hist, max_chars),
                _json_dumps_limited(msg_item, max_chars),
                _json_dumps_limited(reward_item, max_chars),
            )

        wandb.log({"train/trajectories": table}, step=self.global_steps)
        self.train_trajectory_table = table

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            # DataProto.union preserves the left batch metadata. Keep the validation
            # marker so reward managers select paper-compatible evaluation scores
            # instead of the legacy rollout reward used during training.
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Paper Table 3 uses a micro-average over evaluation instances, rather
        # than an unweighted average of per-Gym means.
        if reward_extra_infos_dict.get("reward"):
            metric_dict["val-core/micro/reward/mean@1"] = float(np.mean(reward_extra_infos_dict["reward"]))

        # get all the metrics start with "val-core"
        core_metrics = [metric_dict[pfx] for pfx in metric_dict.keys() if pfx.startswith("val-core")]
        # get the average value of the core metrics
        avg_core_metric = metric_dict.get(
            "val-core/micro/reward/mean@1",
            sum(core_metrics) / len(core_metrics) if core_metrics else 0,
        )
        save_best_validation_checkpoint = self.config.trainer.get("save_best_validation_checkpoint", True)
        if save_best_validation_checkpoint and avg_core_metric > self.best_valid_score:
            self.best_valid_score = avg_core_metric
            self._save_checkpoint(valid_save_best=True, valid_best_score=avg_core_metric)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self, valid_save_best=False, valid_best_score=-1):
        if not valid_save_best:
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        else:
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"best_global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None)

        # Respect max_*_ckpt_to_keep from config; None means keep everything.
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps,
                                           # was hard-coded None, so max_critic_ckpt_to_keep was read
                                           # from config and then ignored: actor rotated, critic did not,
                                           # leaving 45G of critic state per saved step.
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        if not valid_save_best:
            with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
                f.write(str(self.global_steps))
        else:
            # save best score
            with open(os.path.join(self.config.trainer.default_local_dir, "best_score.txt"), "w") as f:
                f.write(str(self.global_steps) + ", " + str(valid_best_score))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load best score
        best_score = get_best_score(self.config.trainer.default_local_dir)
        if best_score is not None:
            self.best_valid_score = best_score

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        # hot-reloadable clip_ratio_high schedule state (rate limiter + logging)
        self._clip_high_last = None
        self._clip_high_source = "hydra"

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                gym_reward_metrics = {}  # filled from the reward manager below
                # Debug: Check if extra_info exists in batch_dict
                if "extra_info" in batch_dict:
                    print(f"[DATALOADER DEBUG] extra_info exists in batch_dict, type={type(batch_dict['extra_info'])}, len={len(batch_dict['extra_info']) if hasattr(batch_dict['extra_info'], '__len__') else 'N/A'}")
                    if isinstance(batch_dict['extra_info'], np.ndarray) and len(batch_dict['extra_info']) > 0:
                        print(f"[DATALOADER DEBUG] extra_info[0] type={type(batch_dict['extra_info'][0])}, keys={list(batch_dict['extra_info'][0].keys()) if isinstance(batch_dict['extra_info'][0], dict) else 'N/A'}")
                else:
                    print(f"[DATALOADER DEBUG] ❌ extra_info does NOT exist in batch_dict! Available keys: {list(batch_dict.keys())}")
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # Debug: Check if extra_info exists in batch.non_tensor_batch
                if "extra_info" in batch.non_tensor_batch:
                    print(f"[DATALOADER DEBUG] ✅ extra_info exists in batch.non_tensor_batch, type={type(batch.non_tensor_batch['extra_info'])}, len={len(batch.non_tensor_batch['extra_info'])}")
                else:
                    print(f"[DATALOADER DEBUG] ❌ extra_info does NOT exist in batch.non_tensor_batch! Available keys: {list(batch.non_tensor_batch.keys())}")

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                # CRITICAL: batch.pop() modifies the original batch and only returns the popped keys!
                # We need to preserve extra_info BEFORE calling pop()
                extra_info_to_preserve = batch.non_tensor_batch.get("extra_info", None)
                if extra_info_to_preserve is not None:
                    print(f"[DATALOADER DEBUG] ✅ Found extra_info before pop, type={type(extra_info_to_preserve)}, len={len(extra_info_to_preserve)}")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # CRITICAL: batch.pop() only returns the popped keys, not the remaining ones!
                # We need to manually preserve extra_info and other non_tensor_batch keys
                # that are needed for rollout but not popped
                if extra_info_to_preserve is not None:
                    gen_batch.non_tensor_batch["extra_info"] = extra_info_to_preserve
                    print(f"[DATALOADER DEBUG] ✅ Preserved extra_info in gen_batch, type={type(gen_batch.non_tensor_batch['extra_info'])}, len={len(gen_batch.non_tensor_batch['extra_info'])}")
                else:
                    print(f"[DATALOADER DEBUG] ⚠️ extra_info not found in batch.non_tensor_batch before pop, available keys: {list(batch.non_tensor_batch.keys())}")

                pad_id = self.tokenizer.pad_token_id
                # just after creating `gen_batch`
                max_len = self.config.data.max_prompt_length
                for key in ("input_ids", "attention_mask", "position_ids"):
                    t = gen_batch.batch[key]
                    if t.size(1) < max_len:
                        pad_val = pad_id if key != "attention_mask" else 0
                        delta = max_len - t.size(1)
                        gen_batch.batch[key] = torch.nn.functional.pad(t, (0, delta), value=pad_val)
                    else:
                        gen_batch.batch[key] = t[:, :max_len]

                print("input_ids.shape", gen_batch.batch["input_ids"].shape)
                print("max prompt len in this chunk:", gen_batch.batch["input_ids"].ne(pad_id).sum(-1).max().item())
                print("min prompt len in this chunk:", gen_batch.batch["input_ids"].ne(pad_id).sum(-1).min().item())

                # Expose the current training epoch to rollout workers so
                # epoch-aware curriculum schedules (e.g. TravelGym search
                # failure injection) can switch behavior between epochs.
                gen_batch.meta_info["current_epoch"] = epoch

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            print(f"gen_batch: {gen_batch.batch['input_ids']}")
                            print(f"gen_batch: {gen_batch.batch['attention_mask']}")
                            print(f"gen_batch: {gen_batch.batch['position_ids']}")
                            print(f"gen_batch keys: {gen_batch.batch.keys()}")
                            print(f"gen_batch shape: {gen_batch.batch['input_ids'].shape}")
                            print(f"gen_batch shape: {gen_batch.batch['attention_mask'].shape}")
                            print(f"gen_batch shape: {gen_batch.batch['position_ids'].shape}")

                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    original_batch_size = len(batch.batch)
                    rollout_n = self.config.actor_rollout_ref.rollout.n
                    expected_batch_size = original_batch_size * rollout_n
                    gen_batch_output_size = len(gen_batch_output.batch) if hasattr(gen_batch_output, 'batch') and gen_batch_output.batch is not None else 0
                    print(f"[BATCH UNION DEBUG] Before repeat: batch_size={original_batch_size}, rollout_n={rollout_n}, expected_after_repeat={expected_batch_size}, gen_batch_output_size={gen_batch_output_size}")

                    batch = batch.repeat(repeat_times=rollout_n, interleave=True)
                    after_repeat_size = len(batch.batch)
                    print(f"[BATCH UNION DEBUG] After repeat: batch_size={after_repeat_size}, expected={expected_batch_size}")

                    batch = batch.union(gen_batch_output)
                    after_union_size = len(batch.batch)
                    print(f"[BATCH UNION DEBUG] After union: batch_size={after_union_size}, expected={expected_batch_size}")

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            print(f"[DEBUG] Using async reward computation - intrinsic rewards will NOT be computed!")
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            # Prepare intrinsic reward parameters
                            use_intrinsic = self.config.algorithm.get("use_intrinsic_reward", False)
                            intrinsic_config = self.config.algorithm.get("intrinsic_reward", {})

                            print(f"[DEBUG] use_intrinsic={use_intrinsic}, intrinsic_config={intrinsic_config}")
                            print(f"[DEBUG] launch_reward_fn_async={self.config.reward_model.launch_reward_fn_async}")

                            # Compute intrinsic rewards on worker side to avoid OOM
                            intrinsic_rewards = None
                            if use_intrinsic:
                                try:
                                    with _timer("intrinsic_reward", timing_raw):
                                        # Ensure non-tensor fields are numpy arrays for DataProto.chunk
                                        for k, v in list(batch.non_tensor_batch.items()):
                                            if not isinstance(v, np.ndarray):
                                                batch.non_tensor_batch[k] = np.array(v, dtype=object)
                                        # Do NOT broadcast dict/scalar configs as non-tensor arrays; keep in meta_info
                                        # Add intrinsic config to batch for worker
                                        # Create a mutable copy of intrinsic_config to add step info
                                        from omegaconf import OmegaConf
                                        intrinsic_config_dict = OmegaConf.to_container(intrinsic_config, resolve=True)
                                        if not isinstance(intrinsic_config_dict, dict):
                                            intrinsic_config_dict = dict(intrinsic_config_dict)
                                        # Pass global_step for mask sensitivity stats (only collect on first step)
                                        intrinsic_config_dict["global_step"] = self.global_steps
                                        intrinsic_config_dict["current_epoch"] = epoch
                                        # Pass default_local_dir for file writing
                                        batch.meta_info["default_local_dir"] = self.config.trainer.default_local_dir
                                        batch.meta_info["intrinsic_config"] = intrinsic_config_dict
                                        batch.meta_info["current_epoch"] = epoch
                                        batch.meta_info["use_intrinsic_reward"] = True
                                        # Force debug on intrinsic if requested globally
                                        if self.config.algorithm.get("intrinsic_reward", {}).get("debug", False):
                                            batch.meta_info["intrinsic_config"]["debug"] = True

                                        # Compute intrinsic rewards directly on worker (model stays on worker)
                                        print(f"[DEBUG] Computing intrinsic rewards on worker side...")
                                        raw_intrinsic = self.actor_rollout_wg.compute_intrinsic_rewards(batch)

                                        # Extract mask sensitivity statistics if available
                                        # Statistics are stored in the returned DataProto from worker
                                        mask_stats_found = False

                                        # Check in the returned raw_intrinsic DataProto first
                                        if isinstance(raw_intrinsic, DataProto):
                                            if hasattr(raw_intrinsic, 'mask_sensitivity_stats') and raw_intrinsic.mask_sensitivity_stats:
                                                print(f"[MASK SENSITIVITY LOG] Found stats in raw_intrinsic.mask_sensitivity_stats: {len(raw_intrinsic.mask_sensitivity_stats)} items")
                                                for key, value in raw_intrinsic.mask_sensitivity_stats.items():
                                                    metrics[key] = value
                                                    mask_stats_found = True

                                            if hasattr(raw_intrinsic, 'meta_info') and 'mask_sensitivity_stats' in raw_intrinsic.meta_info:
                                                mask_stats = raw_intrinsic.meta_info['mask_sensitivity_stats']
                                                print(f"[MASK SENSITIVITY LOG] Found stats in raw_intrinsic.meta_info['mask_sensitivity_stats']: {len(mask_stats)} items")
                                                for key, value in mask_stats.items():
                                                    metrics[key] = value
                                                    mask_stats_found = True

                                        # Also check in original batch (for backward compatibility)
                                        if hasattr(batch, 'mask_sensitivity_stats') and batch.mask_sensitivity_stats:
                                            print(f"[MASK SENSITIVITY LOG] Found stats in batch.mask_sensitivity_stats: {len(batch.mask_sensitivity_stats)} items")
                                            for key, value in batch.mask_sensitivity_stats.items():
                                                metrics[key] = value
                                                mask_stats_found = True

                                        if hasattr(batch, 'meta_info') and 'mask_sensitivity_stats' in batch.meta_info:
                                            mask_stats = batch.meta_info['mask_sensitivity_stats']
                                            print(f"[MASK SENSITIVITY LOG] Found stats in batch.meta_info['mask_sensitivity_stats']: {len(mask_stats)} items")
                                            for key, value in mask_stats.items():
                                                metrics[key] = value
                                                mask_stats_found = True

                                        if mask_stats_found:
                                            mask_metrics_count = len([k for k in metrics.keys() if k.startswith('mask_sensitivity/')])
                                            print(f"[MASK SENSITIVITY LOG] ✓ Successfully added {mask_metrics_count} mask sensitivity metrics to logging")
                                        else:
                                            print(f"[MASK SENSITIVITY LOG] WARNING: No mask sensitivity stats found!")
                                            print(f"[MASK SENSITIVITY LOG] Debug: raw_intrinsic type={type(raw_intrinsic)}")
                                            if isinstance(raw_intrinsic, DataProto):
                                                print(f"[MASK SENSITIVITY LOG] Debug: hasattr(raw_intrinsic, 'mask_sensitivity_stats')={hasattr(raw_intrinsic, 'mask_sensitivity_stats')}")
                                                if hasattr(raw_intrinsic, 'meta_info'):
                                                    print(f"[MASK SENSITIVITY LOG] Debug: 'mask_sensitivity_stats' in raw_intrinsic.meta_info={'mask_sensitivity_stats' in raw_intrinsic.meta_info if hasattr(raw_intrinsic, 'meta_info') else 'N/A'}")

                                    # Normalize return type: could be list/tuple/DataProto/tensor
                                    intrinsic_rewards = raw_intrinsic
                                    if isinstance(intrinsic_rewards, (list, tuple)) and len(intrinsic_rewards) > 0:
                                        intrinsic_rewards = intrinsic_rewards[0]
                                    if hasattr(intrinsic_rewards, "batch") and isinstance(intrinsic_rewards.batch, dict):
                                        intrinsic_rewards = intrinsic_rewards.batch.get("intrinsic_reward", intrinsic_rewards)
                                    if hasattr(intrinsic_rewards, "tensor"):
                                        intrinsic_rewards = intrinsic_rewards.tensor
                                    if not torch.is_tensor(intrinsic_rewards) and getattr(intrinsic_rewards, "__class__", None).__name__ == "DataProto":
                                        intrinsic_rewards = intrinsic_rewards.batch.get("intrinsic_reward", None)

                                    if intrinsic_rewards is None or not torch.is_tensor(intrinsic_rewards):
                                        raise RuntimeError(f"Intrinsic reward extraction failed, got type={type(intrinsic_rewards)} from {type(raw_intrinsic)}")

                                    # print(
                                    #     f"[DEBUG] Intrinsic rewards computed on worker: "
                                    #     f"shape={intrinsic_rewards.shape}, "
                                    #     f"sum={intrinsic_rewards.sum().item():.6f}, "
                                    #     f"mean={intrinsic_rewards.mean().item():.6f}"
                                    # )  # Commented out to reduce noise
                                except Exception as e:
                                    # print(f"[DEBUG] Warning: Failed to compute intrinsic rewards on worker:")  # Commented out to reduce noise
                                    # print(f"[DEBUG] {e}")  # Commented out to reduce noise
                                    import traceback
                                    traceback.print_exc()
                                    intrinsic_rewards = None

                            # print(f"[DEBUG] Before compute_reward: use_intrinsic={use_intrinsic}, intrinsic_rewards={'None' if intrinsic_rewards is None else 'OK'}")  # Commented out to reduce noise

                            # Multiturn GAE consumes per-turn RAW env rewards placed on
                            # each turn's action span (see NaiveRewardManager); other
                            # estimators keep the legacy trajectory-scalar placement.
                            place_per_turn_rewards = (
                                self.config.algorithm.adv_estimator == AdvantageEstimator.GAE
                                and bool(self.config.algorithm.get("multiturn_gae", {}).get("enable", False))
                            )

                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch,
                                self.reward_fn,
                                gamma=self.config.algorithm.gamma,
                                use_intrinsic=use_intrinsic,
                                intrinsic_config=intrinsic_config,
                                intrinsic_rewards=intrinsic_rewards,  # Pass pre-computed rewards instead of model
                                tokenizer=self.tokenizer,
                                current_epoch=epoch,
                                place_per_turn_rewards=place_per_turn_rewards,
                            )

                            print(f"[DEBUG] After compute_reward: has intrinsic_reward in extra_infos={'intrinsic_reward' in reward_extra_infos_dict}")

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_logprob_vals = torch.masked_select(rollout_old_log_probs, response_mask.bool())
                            actor_logprob_vals = torch.masked_select(actor_old_log_probs, response_mask.bool())
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_logprob_mean": rollout_logprob_vals.mean().detach().item(),
                                    "training/rollout_logprob_min": rollout_logprob_vals.min().detach().item(),
                                    "training/rollout_logprob_max": rollout_logprob_vals.max().detach().item(),
                                    "training/actor_logprob_mean": actor_logprob_vals.mean().detach().item(),
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        # Asymmetric critic: attach privileged probe features (user
                        # satisfaction, per env turn) for the value model only. These
                        # features never enter the reward, so the policy cannot hack
                        # them; they only improve the baseline's credit assignment.
                        if self.config.critic.model.get("use_probe_features", False):
                            feature_dim = int(self.config.critic.model.get("probe_feature_dim", 2))
                            batch.batch["probe_features"] = _build_probe_feature_tensor(batch, feature_dim=feature_dim)
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        # scalar per-gym reward metrics: pop before the loop below
                        # treats every remaining entry as a per-sample list
                        gym_reward_metrics = reward_extra_infos_dict.pop(GYM_METRICS_KEY, {}) \
                            if reward_extra_infos_dict else {}

                        # Add intrinsic and extrinsic rewards to batch if available (Info-GRPO)
                        if "extrinsic_reward" in reward_extra_infos_dict:
                            batch.batch["extrinsic_reward"] = reward_extra_infos_dict["extrinsic_reward"]
                        if "intrinsic_reward" in reward_extra_infos_dict:
                            batch.batch["intrinsic_reward"] = reward_extra_infos_dict["intrinsic_reward"]
                            # Ensure intrinsic_weight matches batch dim for TensorDict
                            weight_val = reward_extra_infos_dict.get("intrinsic_weight", 0.1)
                            bsz = batch.batch.batch_size[0]
                            device = batch.batch["token_level_scores"].device
                            batch.batch["intrinsic_weight"] = torch.full((bsz, 1), float(weight_val), device=device)

                        # print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            # Filter out fields already added as tensors to avoid duplicate/scalar issues
                            tensor_fields = {'intrinsic_reward', 'extrinsic_reward', 'intrinsic_weight'}
                            for k, v in reward_extra_infos_dict.items():
                                if k in tensor_fields:
                                    continue
                                # Convert to numpy array and check if it's splittable (not 0-D)
                                arr = np.array(v) if not isinstance(v, np.ndarray) else v
                                if arr.ndim > 0:  # Only add non-scalar arrays (1-D or higher)
                                    # Samples whose score dict lacks this key leave the list
                                    # shorter than the batch; such ragged fields crash
                                    # DataProto.chunk during worker dispatch, so drop them.
                                    if arr.shape[0] != batch.batch.batch_size[0]:
                                        print(f"[DEBUG] Skipping ragged field '{k}' with length {arr.shape[0]} != batch size {batch.batch.batch_size[0]} (would cause chunk error)")
                                        continue
                                    batch.non_tensor_batch[k] = arr
                                else:
                                    print(f"[DEBUG] Skipping scalar field '{k}' with value {v} (would cause chunk error)")

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        probe_intrinsic_config = self.config.algorithm.get("probe_intrinsic_reward", {})
                        if (
                            self.config.algorithm.adv_estimator in [AdvantageEstimator.INFO_GRPO, AdvantageEstimator.GAE]
                            and probe_intrinsic_config.get("enable", False)
                        ):
                            # Keep the environment reward as the InfoPO outcome branch and
                            # source the intrinsic branch exclusively from the user-satisfaction probe.
                            batch.batch["extrinsic_reward"] = batch.batch["token_level_rewards"].clone()
                            batch.batch["intrinsic_reward"] = _build_probe_intrinsic_reward_tensor(
                                batch,
                                source=str(probe_intrinsic_config.get("source", "satisfaction")),
                            )
                            batch.batch["intrinsic_weight"] = torch.full(
                                (batch.batch.batch_size[0], 1),
                                float(probe_intrinsic_config.get("weight", 0.5)),
                                device=batch.batch["token_level_rewards"].device,
                            )
                            if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
                                multiturn_gae_config = self.config.algorithm.get("multiturn_gae", {})
                                if (
                                    multiturn_gae_config.get("enable", False)
                                    and "loss_mask" in batch.batch
                                    and "turn_boundaries" in batch.batch
                                ):
                                    response_length = batch.batch["responses"].size(1)
                                    action_mask = batch.batch["loss_mask"][:, -response_length:]
                                    batch.batch["intrinsic_reward"] = _normalize_multiturn_probe_reward(
                                        batch.batch["intrinsic_reward"],
                                        action_mask,
                                        batch.batch["turn_boundaries"],
                                    )
                                batch.batch["token_level_rewards"] = (
                                    batch.batch["extrinsic_reward"]
                                    + batch.batch["intrinsic_reward"] * batch.batch["intrinsic_weight"]
                                )

                        # Dense per-turn environment rewards: map each turn's env step
                        # reward (already stored in conversation_histories) onto that
                        # turn's token span and add it on top of the terminal outcome
                        # reward, so multiturn GAE gets a per-turn signal instead of
                        # terminal-only supervision.
                        env_turn_reward_config = self.config.algorithm.get("env_turn_reward", {})
                        if (
                            self.config.algorithm.adv_estimator == AdvantageEstimator.GAE
                            and env_turn_reward_config.get("enable", False)
                        ):
                            env_turn_rewards = _build_probe_intrinsic_reward_tensor(batch, source="env_step")
                            multiturn_gae_config = self.config.algorithm.get("multiturn_gae", {})
                            if (
                                multiturn_gae_config.get("enable", False)
                                and "loss_mask" in batch.batch
                                and "turn_boundaries" in batch.batch
                            ):
                                response_length = batch.batch["responses"].size(1)
                                action_mask = batch.batch["loss_mask"][:, -response_length:]
                                env_turn_rewards = _normalize_multiturn_probe_reward(
                                    env_turn_rewards,
                                    action_mask,
                                    batch.batch["turn_boundaries"],
                                )
                            env_turn_weight = float(env_turn_reward_config.get("weight", 1.0))
                            batch.batch["env_turn_reward"] = env_turn_rewards
                            batch.batch["token_level_rewards"] = (
                                batch.batch["token_level_rewards"] + env_turn_rewards * env_turn_weight
                            )
                            per_traj_sum = env_turn_rewards.sum(-1).float()
                            metrics["env_turn_reward/sum_mean"] = float(per_traj_sum.mean().item())
                            metrics["env_turn_reward/sum_max"] = float(per_traj_sum.max().item())

                        # Debug: Analyze group reward consistency and save sample trajectories
                        with _timer("debug_reward_consistency", timing_raw):
                            self._analyze_group_reward_consistency(batch, metrics)

                        # Compute pre-filter reward statistics (before filtering removes samples)
                        # This ensures we record the true batch average before filtering
                        prefilter_metrics = {}
                        if "extrinsic_reward" in batch.batch:
                            prefilter_scores = batch.batch["extrinsic_reward"].sum(dim=-1)
                            prefilter_mean = torch.mean(prefilter_scores).detach().item()
                            prefilter_metrics.update({
                                "critic/prefilter_external_rewards/mean": prefilter_mean,
                                "critic/prefilter_external_rewards/max": torch.max(prefilter_scores).detach().item(),
                                "critic/prefilter_external_rewards/min": torch.min(prefilter_scores).detach().item(),
                                "critic/prefilter_external_rewards/std": torch.std(prefilter_scores).detach().item(),
                            })
                            print(f"[DEBUG pre-filter] Using extrinsic_reward, prefilter_mean={prefilter_mean:.4f}, batch_size={len(prefilter_scores)}")
                        elif "token_level_rewards" in batch.batch:
                            prefilter_scores = batch.batch["token_level_rewards"].sum(dim=-1)
                            prefilter_mean = torch.mean(prefilter_scores).detach().item()
                            prefilter_metrics.update({
                                "critic/prefilter_external_rewards/mean": prefilter_mean,
                                "critic/prefilter_external_rewards/max": torch.max(prefilter_scores).detach().item(),
                                "critic/prefilter_external_rewards/min": torch.min(prefilter_scores).detach().item(),
                                "critic/prefilter_external_rewards/std": torch.std(prefilter_scores).detach().item(),
                            })
                            print(f"[DEBUG pre-filter] Using token_level_rewards, prefilter_mean={prefilter_mean:.4f}, batch_size={len(prefilter_scores)}")
                        elif "token_level_scores" in batch.batch:
                            prefilter_scores = batch.batch["token_level_scores"].sum(dim=-1)
                            prefilter_mean = torch.mean(prefilter_scores).detach().item()
                            prefilter_metrics.update({
                                "critic/prefilter_external_rewards/mean": prefilter_mean,
                                "critic/prefilter_external_rewards/max": torch.max(prefilter_scores).detach().item(),
                                "critic/prefilter_external_rewards/min": torch.min(prefilter_scores).detach().item(),
                                "critic/prefilter_external_rewards/std": torch.std(prefilter_scores).detach().item(),
                            })
                            print(f"[DEBUG pre-filter] Using token_level_scores, prefilter_mean={prefilter_mean:.4f}, batch_size={len(prefilter_scores)}")
                        else:
                            print(f"[DEBUG pre-filter] WARNING: Neither extrinsic_reward, token_level_rewards, nor token_level_scores found in batch.batch.keys()={list(batch.batch.keys())[:10]}...")

                        # RAGEN Extension: Apply rollout filtering (uncertainty-based sample selection)
                        # This implements the StarPO-S stabilization mechanism from RAGEN paper
                        if hasattr(self, 'rollout_filter') and self.rollout_filter is not None:
                            with _timer("rollout_filter", timing_raw):
                                batch, filter_metrics = self.rollout_filter.filter(batch)
                            metrics.update(filter_metrics)
                            metrics.update(prefilter_metrics)

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            turn_level_method=self.config.actor_rollout_ref.rollout.multi_turn.turn_level_method,
                            trajectory_score_method=self.config.actor_rollout_ref.rollout.multi_turn.trajectory_score_method,
                            config=self.config.algorithm,
                            current_epoch=epoch,
                        )

                        # Asymmetric-critic diagnostics: per-trajectory correlation
                        # between the satisfaction probe and the return / value
                        # residual. Tells whether the probe head has any signal to
                        # learn before touching optimizer settings.
                        if self.config.critic.model.get("use_probe_features", False):
                            metrics.update(_compute_probe_signal_metrics(batch))
                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Optional clip-high scheduler: linearly ramp
                        # clip_ratio_high from `init` to `final` over `steps`
                        # training steps, then hold. Config:
                        #   actor_rollout_ref.actor.clip_ratio_high_schedule:
                        #     {enable: true, init: 0.2, final: 5.0, steps: 100}
                        # Resolved through the hot-reloadable override (see
                        # _scheduled_clip_high); falls back to this hydra block
                        # when no override file is present.
                        hydra_sched = self.config.actor_rollout_ref.actor.get("clip_ratio_high_schedule", None)
                        clip_high, clip_source = _scheduled_clip_high(
                            hydra_sched, self.global_steps, self._clip_high_last)
                        if clip_high is not None:
                            batch.meta_info["clip_ratio_high"] = clip_high
                            if clip_source != self._clip_high_source:
                                print(f"[CLIP_SCHEDULE] step {self.global_steps}: source "
                                      f"{self._clip_high_source} -> {clip_source}, value {clip_high:.4f}")
                                self._clip_high_source = clip_source
                            self._clip_high_last = clip_high

                        # Hot-reloadable entropy bonus (see
                        # _scheduled_entropy_coeff). Resolved here, once, and
                        # passed down through meta_info so every worker uses the
                        # same value for the step.
                        ent_coeff = _scheduled_entropy_coeff(
                            self.global_steps, getattr(self, "_entropy_coeff_last", None))
                        if ent_coeff is not None:
                            batch.meta_info["entropy_coeff"] = ent_coeff
                            if getattr(self, "_entropy_coeff_last", None) != ent_coeff:
                                print(f"[ENTROPY_SCHEDULE] step {self.global_steps}: "
                                      f"entropy_coeff -> {ent_coeff:.5f}")
                            self._entropy_coeff_last = ent_coeff
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    val_start_step = int(self.config.trainer.get("val_start_step", 0) or 0)
                    should_validate = (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps >= val_start_step
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    )
                    if should_validate:
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint(valid_save_best=False)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                data_metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
                postfilter_external_rewards_mean = data_metrics.get("critic/external_rewards/mean", None)
                postfilter_str = f"{postfilter_external_rewards_mean:.4f}" if postfilter_external_rewards_mean is not None else "None"
                print(f"[DEBUG post-filter] After compute_data_metrics, critic/external_rewards/mean={postfilter_str}")
                metrics.update(data_metrics)
                metrics.update(
                    _compute_task_reward_metrics(
                        batch,
                        turn_reward_mode=self.config.algorithm.get("multiturn_gae", {}).get("turn_reward_mode", "raw"),
                        gamma=float(self.config.algorithm.gamma),
                    )
                )
                metrics.update(_compute_probe_reward_metrics(batch))
                metrics.update(gym_reward_metrics)

                # Override critic/external_rewards/* with pre-filter values if available
                # This ensures we record the true batch average before filtering
                if "critic/prefilter_external_rewards/mean" in metrics:
                    prefilter_mean = metrics["critic/prefilter_external_rewards/mean"]
                    print(f"[DEBUG override] Found prefilter_external_rewards/mean={prefilter_mean:.4f}, overriding critic/external_rewards/mean from {postfilter_str}")
                    metrics["critic/external_rewards/mean"] = metrics["critic/prefilter_external_rewards/mean"]
                    metrics["critic/external_rewards/max"] = metrics["critic/prefilter_external_rewards/max"]
                    metrics["critic/external_rewards/min"] = metrics["critic/prefilter_external_rewards/min"]
                    print(f"[DEBUG override] Final critic/external_rewards/mean={metrics['critic/external_rewards/mean']:.4f}")
                else:
                    print(f"[DEBUG override] WARNING: critic/prefilter_external_rewards/mean not found in metrics.keys()={list(metrics.keys())[:10]}...")

                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                self._maybe_log_train_trajectories(logger, batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
