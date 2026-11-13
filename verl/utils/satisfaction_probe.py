from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are simulating the user in a multi-turn conversation. "
    "Use the user's private goal and the conversation so far to continue as the user. "
    "Your hidden state before responding is used to estimate how satisfied the user is."
)


@dataclass
class ProbeTurnResult:
    request_index: int
    turn_index: int
    score: float


def _as_plain_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(config, resolve=True) or {}
    except Exception:
        return dict(config)


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _clean_text(text: Any) -> str:
    text = str(text or "")
    text = re.sub(r"\nReward:\s*[-+0-9.eE]+.*$", "", text, flags=re.DOTALL).strip()
    return text


def _dtype_from_name(name: str) -> torch.dtype:
    name = str(name or "bfloat16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported satisfaction probe dtype: {name}")


class SatisfactionProbeScorer:
    """Batch scorer for the CoSER user-satisfaction linear probe.

    The scorer builds the user simulator's view of each turn:
    - system: probe/user-sim instruction plus the initial user goal
    - user: agent utterance/action
    - assistant: previous user-sim/environment response

    It then takes the last-token hidden state at the configured layer and applies
    the calibrated linear probe from `calibrated_probe.npz`.
    """

    def __init__(self, config: Any):
        cfg = _as_plain_dict(config)
        self.model_path = cfg.get("model_path")
        self.probe_path = cfg.get("probe_path")
        if not self.model_path or not self.probe_path:
            raise ValueError("Satisfaction probe requires explicit model_path and probe_path")
        self.layer = int(cfg.get("layer", -1))
        self.batch_size = int(cfg.get("batch_size", 8))
        self.max_length = int(cfg.get("max_length", 4096))
        self.device_name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = _dtype_from_name(str(cfg.get("dtype", "bfloat16")))
        self.system_prompt = str(cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        self.max_context_turns = int(cfg.get("max_context_turns", 12))
        self.include_choice_prefix = bool(cfg.get("include_choice_prefix", False))
        self.score_all_turns = bool(cfg.get("score_all_turns", True))
        scored_choices = cfg.get("scored_choices", None)
        self.scored_choices = None if scored_choices is None else {str(x) for x in scored_choices}
        self.truncate_to_probe_layer = bool(cfg.get("truncate_to_probe_layer", True))

        self._tokenizer = None
        self._model = None
        self._base_model = None
        self._probe_layer_hook = None
        self._captured_layer_hidden = None
        self._uses_truncated_forward = False
        self._mu = None
        self._w = None
        self._platt_a = None
        self._platt_b = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not os.path.exists(self.probe_path):
            raise FileNotFoundError(f"Satisfaction probe not found: {self.probe_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Satisfaction probe model not found: {self.model_path}")

        bundle = np.load(self.probe_path)
        if self.layer < 0:
            self.layer = int(bundle["layer"])

        self._mu = torch.tensor(bundle["mu"], dtype=torch.float32, device=self.device_name)
        self._w = torch.tensor(bundle["w"], dtype=torch.float32, device=self.device_name)
        self._platt_a = float(bundle["platt_a"])
        self._platt_b = float(bundle["platt_b"])

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=False)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._tokenizer.truncation_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=False,
        )
        self._model.to(self.device_name)
        self._model.eval()
        self._configure_truncated_forward()

    def _configure_truncated_forward(self) -> None:
        """Capture the probe layer and skip transformer layers the probe cannot use."""
        if not self.truncate_to_probe_layer:
            return

        base_model = getattr(self._model, "model", None)
        layers = getattr(base_model, "layers", None)
        if base_model is None or layers is None or not 1 <= self.layer <= len(layers):
            return

        def capture_probe_layer(_module, _inputs, output):
            self._captured_layer_hidden = output[0] if isinstance(output, tuple) else output

        self._base_model = base_model
        self._probe_layer_hook = layers[self.layer - 1].register_forward_hook(capture_probe_layer)
        # LlamaModel iterates up to config.num_hidden_layers. The hook captures the
        # output before its final RMSNorm, matching output_hidden_states[layer].
        self._base_model.config.num_hidden_layers = self.layer
        self._uses_truncated_forward = True

    def score_requests(
        self,
        request_histories: list[dict[str, Any]],
        include_initial_state: bool = False,
    ) -> (
        tuple[list[list[float | None]], dict[str, float]]
        | tuple[list[list[float | None]], list[float | None], dict[str, float]]
    ):
        self._load()
        score_started_at = time.perf_counter()

        examples: list[dict[str, Any]] = []
        scores_by_request: list[list[float | None]] = []
        initial_scores: list[float | None] = [None] * len(request_histories)

        for request_index, item in enumerate(request_histories):
            history = item.get("conversation_history") or []
            initial_messages = item.get("initial_messages") or []
            per_turn_scores: list[float | None] = [None] * len(history)
            scores_by_request.append(per_turn_scores)

            if include_initial_state:
                examples.append(
                    {
                        "request_index": request_index,
                        "turn_index": -1,
                        "messages": self._build_probe_messages(initial_messages, history, -1),
                    }
                )

            for turn_index, turn in enumerate(history):
                choice = str(turn.get("choice", ""))
                if not self.score_all_turns:
                    if self.scored_choices is None or choice not in self.scored_choices:
                        continue
                messages = self._build_probe_messages(initial_messages, history, turn_index)
                examples.append(
                    {
                        "request_index": request_index,
                        "turn_index": turn_index,
                        "messages": messages,
                    }
                )

        flat_results = self._score_examples(examples)
        values = []
        for result in flat_results:
            if result.turn_index < 0:
                initial_scores[result.request_index] = result.score
            else:
                scores_by_request[result.request_index][result.turn_index] = result.score
            values.append(result.score)

        stats = {}
        if values:
            arr = np.asarray(values, dtype=np.float32)
            stats = {
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "std": float(arr.std()),
                "count": float(arr.size),
                "seconds": time.perf_counter() - score_started_at,
            }
        else:
            stats = {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "count": 0.0,
                "seconds": time.perf_counter() - score_started_at,
            }
        if include_initial_state:
            return scores_by_request, initial_scores, stats
        return scores_by_request, stats

    def _score_examples(self, examples: list[dict[str, Any]]) -> list[ProbeTurnResult]:
        if not examples:
            return []

        results: list[ProbeTurnResult] = []
        tokenizer = self._tokenizer
        assert tokenizer is not None
        assert self._model is not None

        for start in range(0, len(examples), self.batch_size):
            batch_examples = examples[start : start + self.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch_examples
            ]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device_name) for k, v in encoded.items()}
            with torch.inference_mode():
                if self._uses_truncated_forward:
                    assert self._base_model is not None
                    self._captured_layer_hidden = None
                    self._base_model(
                        **encoded,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )
                    layer_hidden = self._captured_layer_hidden
                    if layer_hidden is None:
                        raise RuntimeError("Probe-layer forward hook did not capture a hidden state")
                else:
                    output = self._model(
                        **encoded,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    hidden_states = output.hidden_states
                    if self.layer >= len(hidden_states):
                        raise ValueError(
                            f"Probe layer {self.layer} is out of range for model with "
                            f"{len(hidden_states) - 1} transformer layers"
                        )
                    layer_hidden = hidden_states[self.layer]

                layer_hidden = layer_hidden.float()
                if tokenizer.padding_side == "left":
                    last_indices = torch.full(
                        (layer_hidden.shape[0],),
                        encoded["attention_mask"].shape[1] - 1,
                        dtype=torch.long,
                        device=layer_hidden.device,
                    )
                else:
                    last_indices = encoded["attention_mask"].sum(dim=1) - 1
                batch_indices = torch.arange(layer_hidden.shape[0], device=layer_hidden.device)
                h = layer_hidden[batch_indices, last_indices]
                proj = (h - self._mu) @ self._w
                logits = self._platt_a * proj + self._platt_b
                probs = torch.sigmoid(logits).detach().cpu().tolist()
                self._captured_layer_hidden = None

            for item, score in zip(batch_examples, probs):
                results.append(
                    ProbeTurnResult(
                        request_index=int(item["request_index"]),
                        turn_index=int(item["turn_index"]),
                        score=float(score),
                    )
                )

        return results

    def _build_probe_messages(
        self,
        initial_messages: list[Any],
        history: list[dict[str, Any]],
        turn_index: int,
    ) -> list[dict[str, str]]:
        initial_user_context = self._initial_user_context(initial_messages)
        system_content = self.system_prompt
        if initial_user_context:
            system_content += "\n\nInitial user goal/request:\n" + initial_user_context

        messages = [{"role": "system", "content": system_content}]
        start_turn = max(0, turn_index + 1 - self.max_context_turns)
        for idx in range(start_turn, turn_index + 1):
            turn = history[idx]
            agent_text = self._agent_turn_text(turn)
            if not agent_text:
                agent_text = "(The assistant did not provide a substantive response.)"
            messages.append({"role": "user", "content": agent_text})

            if idx < turn_index:
                user_text = _clean_text(turn.get("env_feedback", ""))
                if user_text:
                    messages.append({"role": "assistant", "content": user_text})

        return messages

    def _initial_user_context(self, initial_messages: list[Any]) -> str:
        parts = []
        for message in initial_messages:
            role = _message_role(message)
            content = _clean_text(_message_content(message))
            if not content or role == "system":
                continue
            if role == "user":
                parts.append(content)
        return "\n".join(parts).strip()

    def _agent_turn_text(self, turn: dict[str, Any]) -> str:
        content = _clean_text(turn.get("content", ""))
        if self.include_choice_prefix:
            choice = str(turn.get("choice", "") or "").strip()
            if choice:
                return f"[{choice}] {content}".strip()
        return content


def apply_satisfaction_probe_rewards(
    request_histories: list[dict[str, Any]],
    config: Any,
) -> tuple[list[list[float | None]], dict[str, float]]:
    scorer = SatisfactionProbeScorer(config)
    return scorer.score_requests(request_histories)
