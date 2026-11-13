import re

# A role header paired with the WRONG block: the template emits
# "user\n<tool_response>" and "assistant\n<tool_call>", so the two orderings
# below can only come from the policy writing template scaffolding itself.
_ROLE_MISMATCH_RE = re.compile(r"\buser\n<tool_call>|\bassistant\n<tool_response>")
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

from collections import defaultdict
import logging
import ast
from typing import Optional
import os
import sys
import json
from datetime import datetime
from uuid import uuid4
import numpy as np

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.reward_shaping import apply_reward_shaping, gym_of


# The flow/llm user-simulator draw is a deterministic function of the task id
# (symbolic_layer.env_policy.resolve_user_strategy), so the split can be
# recomputed here instead of plumbing a new field out of every env. Without it
# there is no way to confirm from the metrics that a configured mix is actually
# splitting traffic -- the flow renderer only logs when it *fails*, so log
# density says nothing about how often it ran.
try:
    from symbolic_layer.env_policy import resolve_user_strategy as _resolve_user_strategy
    from symbolic_layer.env_policy import content_task_id as _content_task_id
except Exception:  # noqa: BLE001
    def _content_task_id(text):
        import hashlib as _h
        return "sha:" + _h.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]

    def _resolve_user_strategy(gym, task_id, fallback):
        return fallback


def _task_id_of(data_item) -> Optional[str]:
    """Task id as the env sees it: extra_info -> tools_kwargs -> create_kwargs["id"]."""
    try:
        extra = data_item.non_tensor_batch.get("extra_info") or {}
        if isinstance(extra, np.ndarray):
            extra = extra.item() if extra.shape == () else extra[0]
        tk = extra.get("tools_kwargs") or {}
        if isinstance(tk, str):
            tk = ast.literal_eval(tk)
        # tools_kwargs is not uniformly {tool: {create_kwargs: ...}}: ColBench adds
        # a "_colbench_direct_text" flag alongside the real tool entry. Calling
        # .get on that non-dict raised AttributeError, which the outer except
        # swallowed, so the whole scan returned None and every episode looked
        # like "strategy unknown".
        for spec in tk.values():
            if not isinstance(spec, dict):
                continue
            ck = spec.get("create_kwargs")
            if not isinstance(ck, dict):
                continue
            if ck.get("id") is not None:
                return str(ck["id"])
        # tau2 nests its id one level differently: tools_kwargs holds a single
        # "_tau2_env_kwargs" entry with the id directly on it, and no
        # create_kwargs at all. Without this branch the scan fell through, the
        # per-strategy buckets stayed empty, and there was no metric able to say
        # whether a failure came from the flow user or the LLM one.
        for key, spec in tk.items():
            if isinstance(spec, dict) and spec.get("id") is not None:
                return str(spec["id"])
        # Datasets with no per-task id (ColBench) fall back to the same
        # content-derived id the environment uses, so both sides agree on which
        # simulator an episode faced. Without this the split metrics are absent
        # and the mix itself collapses to a single constant draw.
        for spec in tk.values():
            if not isinstance(spec, dict):
                continue
            ck = spec.get("create_kwargs")
            if not isinstance(ck, dict):
                continue
            task = ck.get("task")
            if isinstance(task, dict) and task.get("problem_description"):
                return _content_task_id(task["problem_description"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _new_gym_stat():
    return {"n": 0, "reward_sum": 0.0, "reward_max": float("-inf"), "n_positive": 0,
            "turns": 0, "raw_turn_sum": 0.0, "shaped_turn_sum": 0.0,
            "n_flow": 0, "n_strategy_known": 0,
            "n_open_calls": 0, "n_unclosed_calls": 0, "n_ep_unclosed": 0,
            "n_ep_no_parsed_call": 0, "n_ep_newline_flood": 0,
            "n_role_header_mismatch": 0, "n_ep_role_header_mismatch": 0,
            "flow": {"n": 0, "reward_sum": 0.0, "n_positive": 0, "raw_turn_sum": 0.0},
            "llm": {"n": 0, "reward_sum": 0.0, "n_positive": 0, "raw_turn_sum": 0.0}}


def summarize_gym_rewards(stats):
    """Flatten per-gym accumulators into scalar metrics.

    The trajectory reward is the discounted SUM of turn rewards, so a gym can
    inflate its reward by taking more turns rather than by succeeding. Reporting
    turns_mean and raw_turn_sum_mean next to the reward makes that visible, and
    shaping_delta_mean shows exactly what the reward-shaping layer is doing to
    each gym.
    """
    metrics = {}
    for gym, s in stats.items():
        n = s["n"]
        if not n:
            continue
        prefix = f"task_reward/by_gym/{gym}"
        metrics[f"{prefix}/n"] = float(n)
        metrics[f"{prefix}/mean"] = s["reward_sum"] / n
        metrics[f"{prefix}/max"] = s["reward_max"]
        metrics[f"{prefix}/success_rate"] = s["n_positive"] / n
        metrics[f"{prefix}/turns_mean"] = s["turns"] / n
        metrics[f"{prefix}/raw_turn_sum_mean"] = s["raw_turn_sum"] / n
        metrics[f"{prefix}/shaping_delta_mean"] = (s["shaped_turn_sum"] - s["raw_turn_sum"]) / n
        metrics[f"{prefix}/unclosed_ep_frac"] = s["n_ep_unclosed"] / n
        metrics[f"{prefix}/no_parsed_call_ep_frac"] = s["n_ep_no_parsed_call"] / n
        metrics[f"{prefix}/newline_flood_ep_frac"] = s["n_ep_newline_flood"] / n
        metrics[f"{prefix}/role_header_mismatch_ep_frac"] = s["n_ep_role_header_mismatch"] / n
        if s["n_open_calls"]:
            metrics[f"{prefix}/unclosed_call_frac"] = (
                s["n_unclosed_calls"] / s["n_open_calls"])
        if s["n_strategy_known"]:
            metrics[f"{prefix}/flow_frac"] = s["n_flow"] / s["n_strategy_known"]
        # Split the reward by which simulator the episode actually faced. Under a
        # user_strategy_mix the two populations are different environments, so a
        # knob that only touches one of them shows up in the blended mean as a
        # halved, ambiguous effect -- and a knob verified on the blend cannot be
        # told apart from noise. Each side carries its own n so a thin bucket is
        # visible rather than silently over-read.
        for side in ("flow", "llm"):
            b = s.get(side) or {}
            bn = b.get("n", 0)
            if not bn:
                continue
            metrics[f"{prefix}/{side}/n"] = float(bn)
            metrics[f"{prefix}/{side}/mean"] = b["reward_sum"] / bn
            metrics[f"{prefix}/{side}/success_rate"] = b["n_positive"] / bn
            metrics[f"{prefix}/{side}/raw_turn_sum_mean"] = b["raw_turn_sum"] / bn
    # One number to watch, aggregated over every gym: a run-level alarm should
    # not require reading five per-gym series.
    _tot_ep = sum(s["n"] for s in stats.values())
    if _tot_ep:
        metrics["task_reward/unclosed_ep_frac"] = (
            sum(s["n_ep_unclosed"] for s in stats.values()) / _tot_ep)
        metrics["task_reward/no_parsed_call_ep_frac"] = (
            sum(s["n_ep_no_parsed_call"] for s in stats.values()) / _tot_ep)
        metrics["task_reward/newline_flood_ep_frac"] = (
            sum(s["n_ep_newline_flood"] for s in stats.values()) / _tot_ep)
        metrics["task_reward/role_header_mismatch_ep_frac"] = (
            sum(s["n_ep_role_header_mismatch"] for s in stats.values()) / _tot_ep)
    _tot_open = sum(s["n_open_calls"] for s in stats.values())
    if _tot_open:
        metrics["task_reward/unclosed_call_frac"] = (
            sum(s["n_unclosed_calls"] for s in stats.values()) / _tot_open)
    return metrics


def make_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable format.
    Handles numpy arrays, torch tensors, and nested structures.
    """
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):
        # Handle objects with attributes
        return {key: make_json_serializable(value) for key, value in obj.__dict__.items()}
    else:
        # Fallback: convert to string
        return str(obj)


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False, gamma=0.9, place_per_turn_rewards=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        # per-gym reward accumulators -> scalar metrics in the return dict
        gym_reward_stats = defaultdict(_new_gym_stat)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            reward_diction = data_item.non_tensor_batch.get("reward_model", {}) or {}
            ground_truth = reward_diction.get("ground_truth")
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "")
            if hasattr(data_source, "__len__") and not isinstance(data_source, str):
                data_source = data_source[0] if len(data_source) > 0 else ""
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # Per-turn raw env rewards, only set by the interact gym branch below.
            turn_rewards = None

            # customized for interaction gym reward computation (already calculated in the rollout)
            # Note that the score here is only for logging purpose, not for training purpose
            # This handles InteractComp (interact_interactcomp) and ColBench (interact_colbench_code)
            if data_source.startswith("interact"):
                # Upstream UserRL semantics: the training score is the SUM of
                # per-turn env rewards from conversation_histories (not the
                # rollout's legacy max-turn reward).
                # conversation_histories for one sample may be the turn list
                # itself (1-D array of dicts) or wrapped in a length-1 outer
                # container; unwrap robustly instead of assuming the shape.
                conversation_histories = []
                conv_hist_data = data_item.non_tensor_batch.get("conversation_histories")
                if conv_hist_data is not None:
                    if isinstance(conv_hist_data, np.ndarray):
                        conv_hist_data = conv_hist_data.tolist()
                    if isinstance(conv_hist_data, dict):
                        conversation_histories = [conv_hist_data]
                    elif isinstance(conv_hist_data, (list, tuple)):
                        if (
                            len(conv_hist_data) == 1
                            and isinstance(conv_hist_data[0], (list, tuple, np.ndarray))
                            and not isinstance(conv_hist_data[0], dict)
                        ):
                            conversation_histories = list(conv_hist_data[0])
                        else:
                            conversation_histories = list(conv_hist_data)
                turn_rewards = [
                    float(step.get("reward", 0.0) or 0.0)
                    for step in conversation_histories
                    if isinstance(step, dict) and step.get("reward") is not None
                ]
                turn_choices = [
                    step.get("choice")
                    for step in conversation_histories
                    if isinstance(step, dict) and step.get("reward") is not None
                ]
                # Optional bounded reward shaping (UserDirector). Disabled by
                # default (config enabled=false) -> returns turn_rewards as-is.
                pre_shaping_turn_rewards = list(turn_rewards)
                turn_rewards = apply_reward_shaping(turn_rewards, turn_choices, data_source)
                raw_sum_reward = float(sum(turn_rewards)) if turn_rewards else 0.0
                reward_to_go = 0.0
                for turn_reward in reversed(turn_rewards):
                    reward_to_go = float(turn_reward) + float(gamma) * reward_to_go

                # Training uses the UserRL-style discounted reward-to-go as
                # the trajectory-level extrinsic reward. Validation can still
                # report a paper-compatible Gym metric when the rollout
                # provides one explicitly.
                final_reward = float(reward_to_go) if turn_rewards else 0.0
                score_max = float(max(turn_rewards)) if turn_rewards else 0.0
                reward_scores = data_item.non_tensor_batch.get("reward_scores", [{}])
                if isinstance(reward_scores, np.ndarray):
                    reward_scores = reward_scores[0]
                elif isinstance(reward_scores, list):
                    reward_scores = reward_scores[0] if reward_scores else {}
                is_validation = bool(data.meta_info.get("validate", False))
                eval_reward = final_reward
                if is_validation and isinstance(reward_scores, dict):
                    eval_reward = float(reward_scores.get("interact_with_env_eval_score", final_reward))

                score = {
                    "score": eval_reward,
                    "score_max": score_max,
                    "interact_with_env_raw_sum": raw_sum_reward,
                    "interact_with_env_r2g": final_reward,
                }
                if isinstance(reward_scores, dict):
                    for key, value in reward_scores.items():
                        if key.startswith("interact_with_env_") and key not in score:
                            score[key] = value

                gym_key = gym_of(data_source) or data_source
                # The env draws its simulator from create_kwargs["id"] (see
                # env_manager), so recompute from the same field; reward_model's
                # copy is a fallback because the rollout pipeline does not always
                # carry every key through non_tensor_batch.
                task_id = _task_id_of(data_item) or reward_diction.get("id")
                if task_id is None:
                    # Last resort only. reward_model arrives EMPTY at this point
                    # in the ColBench pipeline (verified at runtime), so this
                    # never fires there; it is kept for datasets that do carry it.
                    _pd = reward_diction.get("problem_description")
                    if _pd:
                        task_id = _content_task_id(_pd)
                if task_id is not None:
                    strategy = _resolve_user_strategy(gym_key, task_id, None)
                    if strategy is not None:
                        gym_reward_stats[gym_key]["n_strategy_known"] += 1
                        gym_reward_stats[gym_key]["n_flow"] += int(strategy == "flow")
                        bucket = gym_reward_stats[gym_key].get(
                            "flow" if strategy == "flow" else "llm")
                        if bucket is not None:
                            bucket["n"] += 1
                            bucket["reward_sum"] += final_reward
                            bucket["n_positive"] += int(final_reward > 0.0)
                            bucket["raw_turn_sum"] += float(sum(pre_shaping_turn_rewards))
                gym_stat = gym_reward_stats[gym_key]
                gym_stat["n"] += 1
                gym_stat["reward_sum"] += final_reward
                gym_stat["reward_max"] = max(gym_stat["reward_max"], final_reward)
                gym_stat["n_positive"] += int(final_reward > 0.0)
                gym_stat["turns"] += len(turn_rewards)
                gym_stat["raw_turn_sum"] += float(sum(pre_shaping_turn_rewards))
                gym_stat["shaped_turn_sum"] += raw_sum_reward
                _n_open = response_str.count("<tool_call>")
                _n_close = response_str.count("</tool_call>")
                _n_unclosed = max(0, _n_open - _n_close)
                gym_stat["n_open_calls"] += _n_open
                gym_stat["n_unclosed_calls"] += _n_unclosed
                gym_stat["n_ep_unclosed"] += int(_n_unclosed > 0)
                gym_stat["n_ep_no_parsed_call"] += int(_n_close == 0)
                _run = 0
                _max_run = 0
                for _ch in response_str:
                    if _ch == "\n":
                        _run += 1
                        if _run > _max_run:
                            _max_run = _run
                    else:
                        _run = 0
                gym_stat["n_ep_newline_flood"] += int(_max_run > 10)

                _bad_role = len(_ROLE_MISMATCH_RE.findall(response_str))
                gym_stat["n_role_header_mismatch"] += _bad_role
                gym_stat["n_ep_role_header_mismatch"] += int(_bad_role > 0)

                # conversation_histories already extracted above (used for both
                # the training score and logging below).
            elif data_source == "interdimension" or reward_diction.get("type") == "env_reward":
                reward_scores = data_item.non_tensor_batch.get("reward_scores", [{}])
                if isinstance(reward_scores, np.ndarray):
                    reward_scores = reward_scores[0]
                elif isinstance(reward_scores, list):
                    reward_scores = reward_scores[0] if reward_scores else {}

                final_reward = 0.0
                if isinstance(reward_scores, dict):
                    final_reward = float(reward_scores.get("interact_with_env", final_reward))
                else:
                    try:
                        final_reward = float(reward_scores)
                    except (TypeError, ValueError):
                        logging.getLogger(__name__).warning(
                            "Unexpected reward_scores format for interdimension: %s", type(reward_scores)
                        )

                score = {
                    "score": final_reward,
                    "score_max": final_reward,
                }
                conversation_histories = []
                if "conversation_histories" in data_item.non_tensor_batch:
                    conv_hist_data = data_item.non_tensor_batch["conversation_histories"]
                    if isinstance(conv_hist_data, np.ndarray):
                        conversation_histories = conv_hist_data[0] if len(conv_hist_data) > 0 else []
                    elif isinstance(conv_hist_data, list):
                        conversation_histories = conv_hist_data[0] if conv_hist_data else []
                    else:
                        conversation_histories = conv_hist_data
                ground_truth = ground_truth if ground_truth is not None else ""
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=reward_diction,
                    extra_info=extra_info,
                )

            # Note that the score here is only for logging purpose, not for training purpose
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            if (
                place_per_turn_rewards
                and turn_rewards
                and "turn_boundaries" in data.batch.keys()
                and "loss_mask" in data.batch.keys()
            ):
                # Multiturn GAE reads each turn's reward only from action
                # (loss_mask==1) positions inside that turn's span, so place every
                # turn's RAW env reward on the last action token of its span.
                # Raw (undiscounted) rewards are used on purpose: the r2g/raw
                # modes in compute_multiturn_gae_advantage_return apply gamma
                # themselves; placing the pre-discounted trajectory scalar would
                # double-discount. Rewards of turns beyond the (possibly
                # truncated) training window are dropped.
                response_length = data.batch["responses"].shape[-1]
                boundary_row = data.batch["turn_boundaries"][i, :response_length]
                loss_mask_row = data.batch["loss_mask"][i, -response_length:]
                turn_starts = torch.where(boundary_row == 1)[0].tolist()
                if not turn_starts:
                    turn_starts = [0]
                turn_starts = sorted(set(turn_starts))
                turn_starts.append(response_length)
                placed_any = False
                # USERRL_TERMINAL_TASK_REWARD_ONLY=1: the actor only sees
                # terminal-task rewards — per-turn shaping rewards (search /
                # preference elicitation) are withheld and only exist as
                # privileged critic features. Keep rewards on answer turns
                # (final-choice correctness); for answer-less gyms, keep the
                # last turn's reward as the terminal-state signal.
                terminal_only = os.environ.get("USERRL_TERMINAL_TASK_REWARD_ONLY", "0") == "1"
                has_answer_turn = any(c == "answer" for c in turn_choices)
                for turn_idx, turn_reward in enumerate(turn_rewards):
                    if terminal_only:
                        is_answer = turn_idx < len(turn_choices) and turn_choices[turn_idx] == "answer"
                        is_last_turn = turn_idx == len(turn_rewards) - 1
                        if not is_answer and not (is_last_turn and not has_answer_turn):
                            continue
                    if turn_idx >= len(turn_starts) - 1:
                        break
                    span_start = turn_starts[turn_idx]
                    span_end = min(turn_starts[turn_idx + 1], valid_response_length)
                    if span_start >= valid_response_length:
                        break
                    action_positions = torch.where(loss_mask_row[span_start:span_end] > 0)[0]
                    if action_positions.numel() == 0:
                        continue
                    reward_tensor[i, span_start + action_positions[-1].item()] = float(turn_reward)
                    placed_any = True
                if not placed_any and not terminal_only:
                    reward_tensor[i, valid_response_length - 1] = reward
            else:
                reward_tensor[i, valid_response_length - 1] = reward

            # Optional: dump evaluation trajectories for debugging when enabled
            dump_dir = os.environ.get("USERRL_EVAL_DUMP_DIR", None)
            if dump_dir and os.access(os.path.dirname(dump_dir) or ".", os.W_OK):
                try:
                    os.makedirs(dump_dir, exist_ok=True)

                    # Extract messages for detailed prompt inspection
                    messages = None
                    if "messages" in data_item.non_tensor_batch:
                        messages_data = data_item.non_tensor_batch["messages"]
                        if isinstance(messages_data, dict) and "messages" in messages_data:
                            messages = messages_data["messages"]
                        elif hasattr(messages_data, "item") and callable(messages_data.item):
                            msg_obj = messages_data.item()
                            if isinstance(msg_obj, dict) and "messages" in msg_obj:
                                messages = msg_obj["messages"]

                    record = {
                        "id": str(uuid4()),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "data_source": data_source,
                        "ground_truth": str(ground_truth) if not isinstance(ground_truth, (str, int, float, bool, type(None))) else ground_truth,
                        "prompt": prompt_str,
                        "response": response_str,
                        "reward": float(reward),
                        "prompt_length": len(valid_prompt_ids),
                        "response_length": len(valid_response_ids),
                    }
                    if data_source.startswith("interact"):
                        # Make conversation_histories JSON serializable (covers InteractComp and ColBench)
                        record["conversation_histories"] = make_json_serializable(conversation_histories)
                        record["num_turns"] = len(conversation_histories)
                    elif data_source == "interdimension":
                        record["conversation_histories"] = make_json_serializable(conversation_histories)
                        record["num_turns"] = len(conversation_histories)
                    if isinstance(score, dict):
                        record["score_detail"] = make_json_serializable(score)
                    if messages:
                        # Save messages for detailed prompt inspection
                        record["messages"] = [
                            {"role": msg.role if hasattr(msg, "role") else msg.get("role", "unknown"),
                             "content": str(msg.content if hasattr(msg, "content") else msg.get("content", ""))}
                            for msg in messages
                        ]

                    # Save structured JSONL for machine processing
                    with open(os.path.join(dump_dir, "eval_traj.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Save human-readable format for easy inspection
                    with open(os.path.join(dump_dir, "readable_logs.txt"), "a", encoding="utf-8") as f:
                        f.write("\n" + "="*100 + "\n")
                        f.write(f"ID: {record['id']} | Timestamp: {record['timestamp']}\n")
                        f.write(f"Data Source: {data_source} | Reward: {reward}\n")
                        f.write("="*100 + "\n")

                        # Write messages if available (shows each turn's prompt)
                        if messages:
                            f.write("\n[MESSAGES HISTORY]\n")
                            f.write("-"*100 + "\n")
                            for idx, msg in enumerate(messages):
                                msg_role = msg.role if hasattr(msg, "role") else msg.get("role", "unknown")
                                msg_content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                                f.write(f"\n[{idx}] Role: {msg_role}\n")
                                f.write(f"{msg_content}\n")
                                f.write("-"*100 + "\n")

                        f.write(f"\n[FULL PROMPT]\n{prompt_str}\n")
                        f.write(f"\n[FULL RESPONSE]\n{response_str}\n")
                        f.write(f"\n[GROUND TRUTH]\n{ground_truth}\n")

                        if data_source.startswith("interact") and len(conversation_histories) > 0:
                            f.write(f"\n[CONVERSATION HISTORIES - {len(conversation_histories)} turns] (InteractComp/ColBench)\n")
                        elif data_source == "interdimension" and len(conversation_histories) > 0:
                            f.write(f"\n[INTERDIMENSION HISTORIES - {len(conversation_histories)} turns]\n")

                        if isinstance(score, dict):
                            f.write(f"\n[SCORE DETAILS]\n")
                            for key, value in score.items():
                                f.write(f"  {key}: {value}\n")

                        f.write("\n" + "="*100 + "\n\n")

                except Exception as e:
                    print(f"[WARNING] Failed to dump evaluation trajectory: {e}")
                    import traceback
                    traceback.print_exc()

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "reward_metrics": summarize_gym_rewards(gym_reward_stats),
            }
        else:
            return reward_tensor
