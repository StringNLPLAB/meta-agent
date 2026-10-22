"""
Info-GRPO: Intrinsic Reward Computation via Hindsight Policy Shift (HPS)

This module implements intrinsic rewards for sparse-reward environments by computing
the counterfactual policy divergence (KL) between:
1. Posterior: P(next_action | context, observation)
2. Prior: P(next_action | context, <no_observation>)

Multi-turn design:
- Each turn = agent action → environment observation (feedback)
- Observation from turn t influences action at turn t+1
- Simply iterate through turns to compute intrinsic rewards

Key Insight - Reward Attribution:
- action_t → produces obs_t → influences action_{t+1}
- KL(P(action_{t+1} | obs_t) || P(action_{t+1} | no_obs)) measures information gain
- This information gain is attributed to action_t (not action_{t+1})
- Rationale: We want to encourage actions that lead to INFORMATIVE observations
- If we rewarded action_{t+1}, we'd only reward "using information", not "getting information"

Observation Detection Strategy (ACCURATE):
- NO MORE HEURISTICS! Boundaries are now recorded precisely during rollout
- During data generation (sglang_rollout_customized.py):
  * action_start: recorded before generating action (len(_req.input_ids))
  * action_end: recorded after add_assistant_message(len(_req.input_ids))
  * obs_start: recorded before add_tool_response_messages(len(_req.input_ids))
  * obs_end: recorded after add_tool_response_messages(len(_req.input_ids))
- These boundaries are stored in conversation_histories and passed to this module
- No tokenization matching, no guessing, 100% accurate

Sequence Structure:
```
[prompt] | [action_0] [obs_0] | [action_1] [obs_1] | [action_2] [obs_2] | ...
         ^          ^         ^^          ^         ^
    action_start  action_end  action_start        ...
         (turn 0)  obs_start   (turn 1)
                   obs_end
```

How it works:
1. For each turn t with observation:
   - Action tokens: [action_start_t, action_end_t)
   - Observation tokens: [obs_start_t, obs_end_t)
2. Compute KL(P(action_{t+1} | obs_t) || P(action_{t+1} | no_obs))
3. Assign this KL as intrinsic reward to action_t tokens (not action_{t+1}!)

Benefits of this approach:
- 100% accurate, no approximations
- Works for ALL gym environments without modification
- No dependency on tokenization quirks
- Clean separation of concerns: rollout records, trainer uses
- Correct attribution: rewards exploration, not just exploitation

Author: Info-GRPO Implementation
Date: 2025-12-03
Updated: 2025-12-07 - Switched to accurate boundary recording + correct reward attribution
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from verl.protocol import DataProto
from transformers import PreTrainedTokenizer

def get_turn_info_from_batch(
    batch: DataProto,
    debug: bool = False,
) -> List[List[Dict]]:
    """
    Extract turn information from conversation_histories.

    Now with ACCURATE boundaries recorded during rollout:
    - action_start: position where action generation starts
    - action_end: position after action is added to sequence
    - obs_start: position where observation starts (before add_tool_response_messages)
    - obs_end: position after observation is added to sequence

    Args:
        batch: DataProto with conversation_histories containing boundary info
        debug: Whether to print debug information

    Returns:
        List of turn info dicts per batch item
    """
    conversation_histories = batch.non_tensor_batch.get("conversation_histories", [])

    batch_turn_info = []

    for batch_idx, conv_hist_wrapper in enumerate(conversation_histories):
        # Handle potential wrapper structure from rollout
        # conv_hist_wrapper might be: np.ndarray / list / [[turns]] / [turns] / turns / empty
        conv_hist = conv_hist_wrapper

        # Normalize numpy arrays early to avoid ambiguous truth-value checks
        if isinstance(conv_hist, np.ndarray):
            conv_hist = conv_hist.tolist()

        # Unwrap one level if needed (common: [[turns]] from rollout packing)
        if isinstance(conv_hist, list) and len(conv_hist) > 0 and isinstance(conv_hist[0], np.ndarray):
            conv_hist[0] = conv_hist[0].tolist()
        if isinstance(conv_hist, list) and len(conv_hist) > 0 and isinstance(conv_hist[0], list):
            # Heuristic: if it's a wrapper like [[turn0, turn1, ...]], take the inner list
            # (but keep as-is if it's already a list of dicts)
            if len(conv_hist) == 1 or (len(conv_hist) > 0 and len(conv_hist[0]) > 0 and isinstance(conv_hist[0][0], dict)):
                conv_hist = conv_hist[0]

        # Final safety: ensure conv_hist is a list of turns
        if conv_hist is None:
            conv_hist = []
        elif isinstance(conv_hist, np.ndarray):
            conv_hist = conv_hist.tolist()
        elif not isinstance(conv_hist, list):
            # Unexpected type; treat as empty to avoid crashing intrinsic reward computation
            conv_hist = []

        if len(conv_hist) == 0:
            batch_turn_info.append([])
            continue

        turns_info = []
        for turn_idx, turn in enumerate(conv_hist):
            # Directly use the accurate boundaries recorded during rollout
            turn_info = {
                "turn_idx": turn_idx,
                "content": turn.get("content", ""),
                "env_feedback": turn.get("env_feedback", ""),
                "action_start": turn.get("action_start", None),
                "action_end": turn.get("action_end", None),
                "obs_start": turn.get("obs_start", None),
                "obs_end": turn.get("obs_end", None),
            }
            turns_info.append(turn_info)

        batch_turn_info.append(turns_info)

    return batch_turn_info

def compute_batched_kl_divergence(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tasks: List[Dict],
    tokenizer: PreTrainedTokenizer,
    placeholder_text: str = "No information found.",
    max_batch_size: int = 32,
    debug: bool = False,
) -> List[float]:
    """
    Compute KL divergence for multiple tasks in batched mode for efficiency.

    This function batches multiple KL computations together to reduce the number of
    model forward passes and improve GPU utilization.

    Args:
        model: Actor model
        input_ids: (batch_size, seq_len) full sequences with observations
        attention_mask: (batch_size, seq_len) attention masks
        tasks: List of task dictionaries, each containing:
            - batch_idx: index in the batch
            - action_start, action_end: boundaries of the action to evaluate
            - obs_start, obs_end: boundaries of the observation tokens
            - is_dummy: whether this is a dummy task for padding
        tokenizer: Tokenizer
        placeholder_text: Placeholder for masked observation
        max_batch_size: Maximum batch size for each forward pass
        debug: Whether to print debug information

    Returns:
        List of KL divergence values, one per task
    """
    if len(tasks) == 0:
        return []

    # Get rank for debugging
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0

    # CRITICAL FIX: Save and temporarily change tokenizer padding side for Flash Attention
    original_padding_side = tokenizer.padding_side
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        model_type = model.config.model_type.lower()
        # Flash Attention models (Qwen2, Llama, etc.) require left padding
        if any(name in model_type for name in ['qwen', 'llama', 'mistral', 'gemma']):
            tokenizer.padding_side = 'left'

    # Encode placeholder once
    placeholder_tokens = tokenizer.encode(placeholder_text, add_special_tokens=False)

    # Process tasks in mini-batches to control memory usage
    all_kl_values = []

    for batch_start in range(0, len(tasks), max_batch_size):
        batch_end = min(batch_start + max_batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]
        current_batch_size = len(batch_tasks)

        # Find max sequence length in this mini-batch
        max_seq_len = max(task['action_end'] for task in batch_tasks)

        # CRITICAL FIX: Use LEFT padding for Flash Attention
        # Instead of torch.zeros, we'll collect sequences and use proper padding
        sequences_with_obs = []
        sequences_without_obs = []
        valid_tasks = []

        # Store metadata for each task in batch
        task_metadata = []

        for i, task in enumerate(batch_tasks):
            batch_idx = task['batch_idx']
            action_start = task['action_start']
            action_end = task['action_end']
            obs_start = task['obs_start']
            obs_end = task['obs_end']

            # Clip boundaries to sequence length
            seq_len = len(input_ids[batch_idx])
            action_end = min(action_end, seq_len)
            action_start = min(action_start, seq_len)
            obs_end = min(obs_end, seq_len)
            obs_start = min(obs_start, seq_len)

            # Validate boundaries
            action_len = action_end - action_start
            obs_len = obs_end - obs_start

            if action_len <= 0 or action_start == 0 or obs_len <= 0:
                # Invalid task, will return 0.0
                task_metadata.append({
                    'valid': False,
                    'action_start': action_start,
                    'action_end': action_end,
                })
                # Add dummy sequences for padding
                dummy_seq = torch.zeros(1, dtype=input_ids.dtype, device=input_ids.device)
                sequences_with_obs.append(dummy_seq)
                sequences_without_obs.append(dummy_seq)
                continue

            # Get sequence WITH observation (original)
            seq_with_obs = input_ids[batch_idx][:action_end]  # (action_end,)
            sequences_with_obs.append(seq_with_obs)

            # Create counterfactual sequence by replacing observation with placeholder
            counterfactual_input_ids = input_ids[batch_idx].clone()
            obs_tokens = input_ids[batch_idx][obs_start:obs_end].tolist()

            # Replace observation tokens with placeholder tokens (same logic as original)
            if len(placeholder_tokens) == len(obs_tokens):
                counterfactual_input_ids[obs_start:obs_end] = torch.tensor(
                    placeholder_tokens, device=input_ids.device
                )
            elif len(placeholder_tokens) < len(obs_tokens):
                # Pad placeholder to same length by repeating last token
                padded_placeholder = placeholder_tokens + [placeholder_tokens[-1]] * (len(obs_tokens) - len(placeholder_tokens))
                counterfactual_input_ids[obs_start:obs_end] = torch.tensor(
                    padded_placeholder, device=input_ids.device
                )
            else:
                # Truncate placeholder to same length
                counterfactual_input_ids[obs_start:obs_end] = torch.tensor(
                    placeholder_tokens[:len(obs_tokens)], device=input_ids.device
                )

            # Get sequence WITHOUT observation (counterfactual)
            seq_without_obs = counterfactual_input_ids[:action_end]  # (action_end,)
            sequences_without_obs.append(seq_without_obs)

            task_metadata.append({
                'valid': True,
                'action_start': action_start,
                'action_end': action_end,
                'batch_idx': batch_idx,
            })

        # CRITICAL FIX: Use torch.nn.utils.rnn.pad_sequence with LEFT padding
        # For left padding, we need to reverse, pad, then reverse back
        # But pad_sequence doesn't support left padding directly for 1D tensors
        # So we manually implement left padding

        def left_pad_sequences(sequences, padding_value=0):
            """Manually implement left padding"""
            max_len = max(len(seq) for seq in sequences)
            padded = []
            masks = []
            for seq in sequences:
                seq_len = len(seq)
                padding_len = max_len - seq_len
                if padding_len > 0:
                    # Left padding: [pad, pad, ..., seq]
                    padded_seq = torch.cat([
                        torch.full((padding_len,), padding_value, dtype=seq.dtype, device=seq.device),
                        seq
                    ])
                    mask = torch.cat([
                        torch.zeros(padding_len, dtype=torch.long, device=seq.device),
                        torch.ones(seq_len, dtype=torch.long, device=seq.device)
                    ])
                else:
                    padded_seq = seq
                    mask = torch.ones(seq_len, dtype=torch.long, device=seq.device)
                padded.append(padded_seq)
                masks.append(mask)
            return torch.stack(padded), torch.stack(masks)

        # Apply left padding
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        batched_with_obs, batched_masks_with = left_pad_sequences(sequences_with_obs, padding_value=pad_token_id)
        batched_without_obs, batched_masks_without = left_pad_sequences(sequences_without_obs, padding_value=pad_token_id)

        # Batched forward passes (CRITICAL: both must be called even if some tasks are invalid)
        with torch.no_grad():
            try:
                # Forward pass WITH observation
                outputs_with = model(
                    input_ids=batched_with_obs,
                    attention_mask=batched_masks_with,
                )
                logits_with = outputs_with.logits  # (current_batch_size, max_seq_len, vocab_size)
            except Exception as e:
                logits_with = None

            try:
                # Forward pass WITHOUT observation
                outputs_without = model(
                    input_ids=batched_without_obs,
                    attention_mask=batched_masks_without,
                )
                logits_without = outputs_without.logits  # (current_batch_size, max_seq_len, vocab_size)
            except Exception as e:
                logits_without = None

        # Extract KL values for each task
        batch_kl_values = []
        for i, (task, metadata) in enumerate(zip(batch_tasks, task_metadata)):
            if not metadata['valid'] or logits_with is None or logits_without is None:
                batch_kl_values.append(0.0)
                continue

            action_start = metadata['action_start']
            action_end = metadata['action_end']
            batch_idx = metadata['batch_idx']

            # CRITICAL FIX: Account for left padding when extracting logits
            # Find where the actual sequence starts (after padding)
            seq_actual_len = action_end  # Length of actual sequence
            max_len = batched_with_obs.shape[1]
            padding_len = max_len - seq_actual_len

            # The actual sequence starts at padding_len and ends at max_len
            # For teacher forcing, we need logits at positions [padding_len + action_start - 1 : padding_len + action_end - 1]
            start_pos = padding_len + action_start - 1
            end_pos = padding_len + action_end - 1

            # Get logits for action positions
            task_logits_with = logits_with[i, start_pos:end_pos, :]  # (action_len, vocab_size)
            task_logits_without = logits_without[i, start_pos:end_pos, :]  # (action_len, vocab_size)

            # Get target tokens (the action tokens we're predicting) - TEACHER FORCING
            target_tokens = input_ids[batch_idx][action_start:action_end]  # (action_len,)

            # Safety check: Verify dimensions match
            actual_len = min(task_logits_with.shape[0], len(target_tokens))
            if task_logits_with.shape[0] != len(target_tokens):
                task_logits_with = task_logits_with[:actual_len]
                task_logits_without = task_logits_without[:actual_len]
                target_tokens = target_tokens[:actual_len]

            if actual_len == 0:
                batch_kl_values.append(0.0)
                continue

            # Compute log probabilities
            log_probs_with = F.log_softmax(task_logits_with, dim=-1)  # (action_len, vocab_size)
            log_probs_without = F.log_softmax(task_logits_without, dim=-1)  # (action_len, vocab_size)

            # Gather log probs for target tokens
            log_p_with = log_probs_with[torch.arange(actual_len), target_tokens]  # (actual_len,)
            log_p_without = log_probs_without[torch.arange(actual_len), target_tokens]  # (actual_len,)

            kl_contributions = (log_p_with - log_p_without).cpu().tolist()
            total_kl = sum(kl_contributions)
            kl_value = total_kl / actual_len if actual_len > 0 else 0.0

            batch_kl_values.append(kl_value)

        all_kl_values.extend(batch_kl_values)

    # CRITICAL FIX: Restore original padding side
    tokenizer.padding_side = original_padding_side

    return all_kl_values

def compute_intrinsic_rewards(
    batch: DataProto,
    model,
    tokenizer: PreTrainedTokenizer,
    config: Optional[Dict] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    Compute intrinsic rewards using Hindsight Policy Shift (HPS).

    Simple turn-based algorithm:
    1. For each trajectory, iterate through turns
    2. For each turn with env_feedback (observation):
       - Compute KL between P(next_action | context + obs) vs P(next_action | context + placeholder)
       - Assign this KL as intrinsic reward to the next action tokens
    3. Return intrinsic reward tensor

    Args:
        batch: DataProto with conversation_histories and sequences
        model: Actor model
        tokenizer: Tokenizer
        config: Configuration dict (optional)
        debug: Whether to print debug information

    Returns:
        intrinsic_rewards: (bs, seq_len) tensor
    """
    if config is None:
        config = {}

    # Get rank for progress tracking
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    placeholder_text = config.get("observation_placeholder", "No information found.")

    # Extract turn information
    batch_turn_info = get_turn_info_from_batch(batch, debug=False)  # Disable debug in sub-function

    # Get input tensors
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]

    # Initialize intrinsic rewards
    batch_size, seq_len = input_ids.shape
    intrinsic_rewards = torch.zeros(batch_size, seq_len, device=input_ids.device)

    # Get max_batch_size for batched computation
    max_batch_size = config.get("intrinsic_kl_batch_size", 32)

    num_real_tasks_prescan = 0

    for batch_idx, turns_info in enumerate(batch_turn_info):
        if not turns_info:
            continue
        for turn_idx in range(len(turns_info)):
            current_turn = turns_info[turn_idx]
            env_feedback = current_turn.get("env_feedback", "")

            # Handle env_feedback which might be a dict or string
            # AnswerAction in InteractComp returns {"answer": "...", "confidence": "..."}
            if isinstance(env_feedback, dict):
                # Extract meaningful text from dict (prefer "answer" field)
                env_feedback_str = str(env_feedback.get("answer", env_feedback)) if env_feedback else ""
            elif isinstance(env_feedback, str):
                env_feedback_str = env_feedback
            else:
                env_feedback_str = str(env_feedback) if env_feedback else ""

            # Check if this turn will trigger a KL computation
            if not env_feedback_str or not env_feedback_str.strip():
                continue
            if turn_idx + 1 >= len(turns_info):
                continue

            # Validate boundaries (same logic as main loop)
            current_action_start = current_turn.get("action_start", None)
            current_action_end = current_turn.get("action_end", None)
            obs_start = current_turn.get("obs_start", None)
            obs_end = current_turn.get("obs_end", None)

            if (current_action_start is None or current_action_end is None or
                current_action_start >= current_action_end or
                obs_start is None or obs_end is None or
                obs_start >= obs_end):
                continue

            # Count real tasks
            num_real_tasks_prescan += 1

    # In batched mode: num_batches = ceil(num_tasks / max_batch_size)
    # Each batch calls model() 2 times (WITH obs and WITHOUT obs)
    import math
    num_batches = math.ceil(num_real_tasks_prescan / max_batch_size) if num_real_tasks_prescan > 0 else 0

    # Collect expected calls (num_batches) from all workers and find max
    max_num_batches = num_batches
    if torch.distributed.is_initialized():
        all_num_batches = [torch.tensor([0], device='cuda') for _ in range(world_size)]
        my_num_batches = torch.tensor([num_batches], device='cuda')
        torch.distributed.all_gather(all_num_batches, my_num_batches)

        all_num_batches_list = [t.item() for t in all_num_batches]
        max_num_batches = max(all_num_batches_list)

    max_expected_calls = max_num_batches * 2
    # ==================================================================

    total_turns = sum(len(t) if t else 0 for t in batch_turn_info)
    num_trajectories = len(batch_turn_info)
    avg_turns_per_traj = total_turns / num_trajectories if num_trajectories > 0 else 0

    # Statistics
    total_kl_computed = 0
    total_kl_value = 0.0
    total_tokens_assigned = 0

    # ========== BATCHED KL COMPUTATION: Collect tasks ==========
    # Collect all real KL computation tasks
    kl_tasks = []
    for batch_idx, turns_info in enumerate(batch_turn_info):
        if not turns_info:
            continue

        for turn_idx in range(len(turns_info)):
            current_turn = turns_info[turn_idx]
            env_feedback = current_turn.get("env_feedback", "")

            # Handle env_feedback which might be a dict or string
            # AnswerAction in InteractComp returns {"answer": "...", "confidence": "..."}
            if isinstance(env_feedback, dict):
                # Extract meaningful text from dict (prefer "answer" field)
                env_feedback_str = str(env_feedback.get("answer", env_feedback)) if env_feedback else ""
            elif isinstance(env_feedback, str):
                env_feedback_str = env_feedback
            else:
                env_feedback_str = str(env_feedback) if env_feedback else ""

            # Only process if this turn has an observation
            if not env_feedback_str or not env_feedback_str.strip():
                continue

            # Find the next turn (which is influenced by this observation)
            if turn_idx + 1 >= len(turns_info):
                continue

            # Get current turn's action boundaries
            current_action_start = current_turn.get("action_start", None)
            current_action_end = current_turn.get("action_end", None)

            # Get observation boundaries from current turn
            obs_start = current_turn.get("obs_start", None)
            obs_end = current_turn.get("obs_end", None)

            # Validate boundaries
            if (current_action_start is None or current_action_end is None or
                current_action_start >= current_action_end or
                obs_start is None or obs_end is None or
                obs_start >= obs_end):
                continue

            # We need next turn to compute KL
            next_turn = turns_info[turn_idx + 1]
            next_action_start = next_turn["action_start"]
            next_action_end = next_turn["action_end"]

            # Add this as a real task
            kl_tasks.append({
                'is_dummy': False,
                'batch_idx': batch_idx,
                'turn_idx': turn_idx,
                'current_action_start': current_action_start,
                'current_action_end': current_action_end,
                'obs_start': obs_start,
                'obs_end': obs_end,
                'action_start': next_action_start,  # For batched KL computation
                'action_end': next_action_end,      # For batched KL computation
            })

    num_real_tasks = len(kl_tasks)

    # Pad tasks to ensure all workers process the same number of batches
    num_dummy_tasks_needed = max_num_batches * max_batch_size - num_real_tasks

    if num_dummy_tasks_needed > 0:
        if num_real_tasks > 0:
            # Reuse first task as dummy padding
            for _ in range(num_dummy_tasks_needed):
                dummy_task = kl_tasks[0].copy()
                dummy_task['is_dummy'] = True
                kl_tasks.append(dummy_task)
        else:
            # Worker has NO real tasks, create synthetic dummy tasks

            if batch_size > 0 and len(batch_turn_info) > 0:
                # Find the first batch with at least one turn
                synthetic_batch_idx = 0
                synthetic_turn_info = None
                for b_idx, turns_info in enumerate(batch_turn_info):
                    if turns_info and len(turns_info) > 0:
                        synthetic_batch_idx = b_idx
                        synthetic_turn_info = turns_info[0]
                        break

                if synthetic_turn_info is not None:
                    for _ in range(num_dummy_tasks_needed):
                        dummy_task = {
                            'is_dummy': True,
                            'batch_idx': synthetic_batch_idx,
                            'turn_idx': 0,
                            'current_action_start': synthetic_turn_info.get('action_start', 1),
                            'current_action_end': synthetic_turn_info.get('action_end', 2),
                            'obs_start': synthetic_turn_info.get('obs_start', 2),
                            'obs_end': synthetic_turn_info.get('obs_end', 3),
                            'action_start': synthetic_turn_info.get('action_start', 1),
                            'action_end': synthetic_turn_info.get('action_end', 2),
                        }
                        kl_tasks.append(dummy_task)

    # ========== BATCHED KL COMPUTATION: Execute in batches ==========
    # Process tasks in mini-batches
    for batch_idx in range(max_num_batches):
        batch_start = batch_idx * max_batch_size
        batch_end = min(batch_start + max_batch_size, len(kl_tasks))
        batch_tasks = kl_tasks[batch_start:batch_end]

        if len(batch_tasks) == 0:
            # This shouldn't happen with proper padding, but handle it gracefully
            continue

        # Log progress
        # Intentionally no logging/progress output for submission version.
        # Call batched KL divergence computation
        kl_values = compute_batched_kl_divergence(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tasks=batch_tasks,
            tokenizer=tokenizer,
            placeholder_text=placeholder_text,
            max_batch_size=len(batch_tasks),  # Process all tasks in this batch at once
            debug=debug,
        )

        # Assign rewards for real tasks (not dummy)
        for task, kl_value in zip(batch_tasks, kl_values):
            if not task['is_dummy']:
                total_kl_computed += 1
                total_kl_value += kl_value

                # Assign intrinsic reward to current_action (which produced the observation)
                batch_idx_task = task['batch_idx']
                current_action_start = task['current_action_start']
                current_action_end = task['current_action_end']

                current_action_len = current_action_end - current_action_start
                if current_action_len > 0:
                    # Normalize by action length (each turn separately)
                    # This ensures fairness: longer actions don't get higher rewards
                    # KL / action_len gives the turn-level reward per token
                    reward_per_token = kl_value / current_action_len
                    intrinsic_rewards[batch_idx_task, current_action_start:current_action_end] = reward_per_token
                    total_tokens_assigned += current_action_len

    return intrinsic_rewards


def aggregate_intrinsic_reward_by_turn(
    batch: DataProto,
    intrinsic_rewards: torch.Tensor,
    max_turns: int = 16,
) -> torch.Tensor:
    """
    Aggregate token-level intrinsic rewards to turn-level rewards.

    For each trajectory, compute the sum of intrinsic rewards for each turn's action tokens.
    Returns a tensor of shape (batch_size, max_turns) where each element is the
    total intrinsic reward for that turn (padded with 0 for non-existent turns).

    Args:
        batch: DataProto with conversation_histories containing turn boundaries
        intrinsic_rewards: (batch_size, seq_len) tensor of token-level intrinsic rewards
                          This is the FULL sequence (prompt + response), possibly truncated
                          conversation_histories boundaries are absolute positions in this sequence
        max_turns: Maximum number of turns to pad to (default: 16)

    Returns:
        turn_level_rewards: (batch_size, max_turns) tensor of turn-level intrinsic rewards
    """
    batch_size, seq_len = intrinsic_rewards.shape
    turn_level_rewards = torch.zeros(batch_size, max_turns, device=intrinsic_rewards.device)

    # Extract turn information
    batch_turn_info = get_turn_info_from_batch(batch, debug=False)

    for batch_idx, turns_info in enumerate(batch_turn_info):
        if not turns_info:
            continue

        for turn_idx, turn_info in enumerate(turns_info):
            if turn_idx >= max_turns:
                break  # Skip turns beyond max_turns

            action_start = turn_info.get("action_start", None)
            action_end = turn_info.get("action_end", None)

            if action_start is None or action_end is None:
                continue

            # conversation_histories boundaries are absolute positions in the full sequence (prompt + response)
            # intrinsic_rewards has the same coordinate system (full sequence, possibly truncated)
            # So we can use absolute positions directly, but need to check if they're within the truncated range

            # Check if this turn's action is within the truncated sequence
            if action_start >= seq_len:
                # This turn is beyond the truncated sequence, skip it
                continue

            # Clip boundaries to sequence length (truncated sequence)
            action_start_clipped = min(action_start, seq_len - 1)
            action_end_clipped = min(action_end, seq_len)

            if action_start_clipped >= action_end_clipped:
                continue

            # Sum intrinsic rewards for this turn's action tokens
            # Use absolute positions directly since intrinsic_rewards is full sequence (prompt + response)
            turn_reward = intrinsic_rewards[batch_idx, action_start_clipped:action_end_clipped].sum()
            turn_level_rewards[batch_idx, turn_idx] = turn_reward

    return turn_level_rewards
