"""
Debug logger for saving training model inputs and outputs.
用于保存训练模型的输入输出，便于调试prompt问题。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelIOLogger:
    """Logger for saving model inputs/outputs during training."""

    def __init__(self, log_dir: Optional[str] = None, enabled: bool = True):
        """
        Initialize the logger.

        Args:
            log_dir: Directory to save logs. If None, uses USERRL_DEBUG_LOG_DIR env var
                    or defaults to ./debug_logs
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        if not enabled:
            return

        if log_dir is None:
            log_dir = os.environ.get("USERRL_DEBUG_LOG_DIR", "./debug_logs")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create separate files for different types of logs
        self.turn_log_file = self.log_dir / "model_turns.jsonl"
        self.conversation_log_file = self.log_dir / "conversations.jsonl"
        self.readable_log_file = self.log_dir / "readable_logs.txt"

        # Clear previous logs or append based on environment variable
        clear_logs = os.environ.get("USERRL_DEBUG_CLEAR_LOGS", "false").lower() == "true"
        if clear_logs:
            for log_file in [self.turn_log_file, self.conversation_log_file, self.readable_log_file]:
                if log_file.exists():
                    log_file.unlink()

    def log_turn(
        self,
        request_id: str,
        turn_idx: int,
        prompt_ids: List[int],
        prompt_text: str,
        response_text: str,
        response_ids: List[int],
        reward: float = 0.0,
        choice: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single turn of model interaction.

        Args:
            request_id: Unique identifier for the conversation
            turn_idx: Turn index in the conversation
            prompt_ids: Token IDs of the prompt
            prompt_text: Decoded prompt text (natural language)
            response_text: Decoded response text (natural language)
            response_ids: Token IDs of the response
            reward: Reward for this turn
            choice: Choice type (action/answer/ask/search/finish)
            messages: Full message history up to this point
            extra_info: Additional information to log
        """
        if not self.enabled:
            return

        try:
            # Structured log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_id": request_id,
                "turn_idx": turn_idx,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "prompt_length": len(prompt_ids),
                "response_length": len(response_ids),
                "reward": reward,
                "choice": choice,
            }

            if messages:
                log_entry["messages"] = messages

            if extra_info:
                log_entry["extra_info"] = extra_info

            # Save to JSONL for machine processing
            with open(self.turn_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            # Save readable format for human inspection
            self._write_readable_turn(request_id, turn_idx, prompt_text, response_text, reward, choice)

        except Exception as e:
            print(f"[DEBUG_LOGGER ERROR] Failed to log turn: {e}")

    def _write_readable_turn(self, request_id: str, turn_idx: int, prompt_text: str,
                            response_text: str, reward: float, choice: str):
        """Write a human-readable format of the turn."""
        try:
            with open(self.readable_log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"Request ID: {request_id} | Turn: {turn_idx} | Choice: {choice} | Reward: {reward}\n")
                f.write("="*80 + "\n")
                f.write(f"\n[PROMPT]\n{prompt_text}\n")
                f.write(f"\n[RESPONSE]\n{response_text}\n")
                f.write("="*80 + "\n\n")
        except Exception as e:
            print(f"[DEBUG_LOGGER ERROR] Failed to write readable turn: {e}")

    def log_conversation(
        self,
        request_id: str,
        all_messages: List[Dict[str, Any]],
        conversation_histories: List[Dict[str, Any]],
        total_reward: float,
        final_prompt: str,
        final_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a complete conversation after all turns are finished.

        Args:
            request_id: Unique identifier for the conversation
            all_messages: Complete message history
            conversation_histories: Turn-by-turn history with rewards
            total_reward: Sum of all rewards
            final_prompt: The complete final prompt (all turns)
            final_response: The complete final response (all turns)
            metadata: Additional metadata (data_source, ground_truth, etc.)
        """
        if not self.enabled:
            return

        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_id": request_id,
                "num_turns": len(conversation_histories),
                "total_reward": total_reward,
                "messages": all_messages,
                "conversation_histories": conversation_histories,
                "final_prompt": final_prompt,
                "final_response": final_response,
            }

            if metadata:
                log_entry["metadata"] = metadata

            with open(self.conversation_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[DEBUG_LOGGER ERROR] Failed to log conversation: {e}")


# Global logger instance
_global_logger: Optional[ModelIOLogger] = None


def get_logger(log_dir: Optional[str] = None, enabled: Optional[bool] = None) -> ModelIOLogger:
    """
    Get the global logger instance.

    Args:
        log_dir: Directory to save logs (only used on first call)
        enabled: Whether to enable logging. If None, checks USERRL_DEBUG_LOGGING env var

    Returns:
        ModelIOLogger instance
    """
    global _global_logger

    if _global_logger is None:
        if enabled is None:
            enabled = os.environ.get("USERRL_DEBUG_LOGGING", "false").lower() == "true"
        _global_logger = ModelIOLogger(log_dir=log_dir, enabled=enabled)

    return _global_logger


def log_model_turn(
    request_id: str,
    turn_idx: int,
    prompt_ids: List[int],
    prompt_text: str,
    response_text: str,
    response_ids: List[int],
    **kwargs
):
    """
    Convenience function to log a model turn using the global logger.

    Usage:
        from verl.utils.debug_logger import log_model_turn

        log_model_turn(
            request_id=request_id,
            turn_idx=0,
            prompt_ids=prompt_ids,
            prompt_text=decoded_prompt,
            response_text=model_response,
            response_ids=response_ids,
            reward=0.5,
            choice="action"
        )
    """
    logger = get_logger()
    logger.log_turn(request_id, turn_idx, prompt_ids, prompt_text, response_text, response_ids, **kwargs)
