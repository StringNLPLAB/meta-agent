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
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BASE_CHAT_HISTORY = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "I am a user."}]


def _is_qwen3_model(tokenizer: PreTrainedTokenizer) -> bool:
    """Check if the tokenizer belongs to a Qwen3 model."""
    if hasattr(tokenizer, 'name_or_path') and 'qwen3' in tokenizer.name_or_path.lower():
        return True
    if hasattr(tokenizer, 'model_name') and 'qwen3' in str(tokenizer.model_name).lower():
        return True
    return False


def _enable_thinking(tokenizer: PreTrainedTokenizer) -> bool:
    """Match upstream UserRL: use the model template's default (thinking for
    Qwen3). USERRL_ENABLE_QWEN3_THINKING=0 explicitly disables it."""
    if not _is_qwen3_model(tokenizer):
        return True
    return os.getenv("USERRL_ENABLE_QWEN3_THINKING", "0") == "1"


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"



_USE_SAMPLED_IDS = os.environ.get("USERRL_USE_SAMPLED_IDS", "1") != "0"
_SAMPLED_IDS_STATS = {"used": 0, "fallback": 0}


def _stop_token_ids(tokenizer) -> set:
    out = set()
    for t in (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)):
        if isinstance(t, int) and t >= 0:
            out.add(t)
    for name in ("<|im_end|>", "<|endoftext|>"):
        try:
            i = tokenizer.convert_tokens_to_ids(name)
        except Exception:  # noqa: BLE001
            i = None
        if isinstance(i, int) and i >= 0:
            out.add(i)
    return out


def _splice_sampled_ids(tokenizer, tail_text: str, content: str, sampled_ids):
    """Return sampled content ids + template suffix ids, or None to fall back.

    `tail_text` is what the old path encoded: the assistant content plus
    whatever the chat template appends after it. SGLang's `output["text"]`
    excludes the stop token while `sampled_ids` includes it, so trailing stop
    tokens are dropped before the suffix is appended.
    """
    if not sampled_ids:
        return None
    try:
        ids = list(sampled_ids)
        stop = _stop_token_ids(tokenizer)
        while ids and ids[-1] in stop:
            ids.pop()
        if not ids:
            return None
        # Compare against TAIL_TEXT, not against `content`. On the tool-call
        # path `content` is `normed_content` -- the <tool_call> block has been
        # lifted out into structured tool_calls and re-rendered by the chat
        # template -- so the sampled ids (which cover the model's whole output,
        # tool call included) can never equal it. Comparing to `content` made
        # the splice fall back on 72% of turns for no reason. The rendered tail
        # does start with the model's raw output whenever nothing rewrote it,
        # which is the case that matters.
        decoded = tokenizer.decode(ids)
        if not decoded or not tail_text.startswith(decoded):
            # text was genuinely rewritten after generation (think-stripping,
            # a repaired tool call, ...) -- the sampled ids no longer describe
            # it, so re-encoding is the only correct choice.
            return None
        suffix_ids = tokenizer.encode(tail_text[len(decoded):], add_special_tokens=False)
        out = ids + suffix_ids
        if tokenizer.decode(out) != tail_text:
            return None
        return out
    except Exception:  # noqa: BLE001
        return None


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    reward_scores: Dict[str, float]
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: Dict[str, List[Any]] = {}
    search_action_count: int = 0
    # Ids SGLang sampled for the turn being appended; consumed and cleared by
    # add_assistant_message. None when the engine did not return logprobs.
    last_sampled_ids: Optional[List[int]] = None

    turn_boundaries: List[int] = []
    conversation_histories: List[Dict[str, Any]] = []

    use_inference_chat_template: bool
    enable_tokenization_sanity_check: bool
    generation_prompt_ids: List[int]
    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        messages = values.get("messages")

        # For InteractComp: messages can be empty (wrapper builds them dynamically)
        # Check if this is an InteractComp request by presence of interact_with_env in tools_kwargs
        is_interactcomp = "interact_with_env" in values.get("tools_kwargs", {})

        if messages is None:
            raise ValueError("messages field is required for AsyncRolloutRequest initialization")

        if not messages and not is_interactcomp:
            raise ValueError("messages is required for AsyncRolloutRequest initialization (unless using InteractComp)")

        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (tokenizer := values.pop("tokenizer", None)):
            raise ValueError("tokenizer is required for AsyncRolloutRequest initialization")

        # For InteractComp with empty messages, skip chat template initialization
        if not messages and is_interactcomp:
            values["messages"] = []
            # Initialize with empty values - will be populated by wrapper
            if not values.get("input_ids"):
                values["input_ids"] = []
            if not values.get("attention_mask"):
                values["attention_mask"] = []
            values["prompt_ids"] = values["input_ids"]
            values["prompt_attention_mask"] = values["attention_mask"]
            values["position_ids"] = values["prompt_position_ids"] = []
            values["loss_mask"] = values["prompt_loss_mask"] = []
            values["generation_prompt_ids"] = []
            values["base_conv_wo_gen_prompt_end_pos"] = 0
            values["base_conv_with_gen_prompt_end_pos"] = 0
            return values

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        # Convert tool_schemas to dict format for tokenizer
        tool_schemas = values.get("tool_schemas", [])
        if tool_schemas:
            tools = []
            for schema in tool_schemas:
                if hasattr(schema, 'model_dump'):
                    tools.append(schema.model_dump())
                elif hasattr(schema, 'dict'):
                    tools.append(schema.dict())
                elif isinstance(schema, dict):
                    tools.append(schema)
                else:
                    # Try to validate as OpenAIFunctionToolSchema
                    from verl.tools.schemas import OpenAIFunctionToolSchema
                    try:
                        validated_schema = OpenAIFunctionToolSchema.model_validate(schema)
                        tools.append(validated_schema.model_dump())
                    except Exception:
                        # Skip invalid schemas
                        continue
        else:
            tools = None

        enable_thinking = _enable_thinking(tokenizer)

        tokens_without_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True, enable_thinking=enable_thinking)
        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True, return_dict=True, enable_thinking=enable_thinking)
            values["input_ids"], values["attention_mask"] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
                logger.warning(f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len {max_prompt_len} after applied chat template with tools.")

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(torch.tensor(values["attention_mask"])).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        values["generation_prompt_ids"] = values["input_ids"][len(tokens_without_prompt) :]
        # Convert tool_schemas for BASE_CHAT_HISTORY as well
        base_tools = tools  # Already converted above
        values["base_conv_wo_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=base_tools, add_generation_prompt=False, tokenize=False, enable_thinking=enable_thinking))
        values["base_conv_with_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=base_tools, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking))
        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask_list = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask_list
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)

        # Handle position_ids: if empty (first call), start from 0; otherwise continue from last position
        if len(self.position_ids) == 0:
            new_position_ids = compute_position_id_with_mask(torch.tensor(attention_mask_list)).tolist()
        else:
            new_position_ids = (compute_position_id_with_mask(torch.tensor(attention_mask_list)) + (self.position_ids[-1] + 1)).tolist()
        self.position_ids += new_position_ids

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=},
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    format_config: dict = {
        "chatml": {
            "assistant_prefix_msg": "\n<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>",
            "tool_prefix_msg": "\n<|im_start|>tool\n",
            "tool_suffix_msg": "<|im_end|>",
        },
        "qwen": {
            "assistant_prefix_msg": "\n<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>",
            "merge_tool_response": True,
            "tool_prefix_msg": "\n<|im_start|>user",
            "tool_suffix_msg": "<|im_end|>",
            "tool_response_prefix_msg": "\n<tool_response>\n",
            "tool_response_suffix_msg": "\n</tool_response>",
        }
    }

    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        generation_prompt_ids = [] if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids else self.generation_prompt_ids
        if generation_prompt_ids:
            self._update_input_ids(generation_prompt_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            enable_thinking = _enable_thinking(tokenizer)
            return tokenizer.apply_chat_template(
                [msg.model_dump() for msg in self.messages],
                tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None),
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=enable_thinking
            )
        else:
            return self.input_ids

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> None:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        raw_content = content  # apply_chat_template rebinds `content` below
        enable_thinking = _enable_thinking(tokenizer)
        content = tokenizer.apply_chat_template(
            [*BASE_CHAT_HISTORY, self.messages[-1]],
            tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None),
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=enable_thinking
        )
        tail_text = content[self.base_conv_with_gen_prompt_end_pos :]
        content_ids = None
        if _USE_SAMPLED_IDS:
            content_ids = _splice_sampled_ids(tokenizer, tail_text, raw_content, self.last_sampled_ids)
        self.last_sampled_ids = None
        if content_ids is None:
            _SAMPLED_IDS_STATS["fallback"] += 1
            content_ids = tokenizer.encode(tail_text, add_special_tokens=False)
        else:
            _SAMPLED_IDS_STATS["used"] += 1
        _n = _SAMPLED_IDS_STATS["used"] + _SAMPLED_IDS_STATS["fallback"]
        if _n % 200 == 0:
            print("[SAMPLED_IDS] n=%d 用采样id=%d (%.1f%%) 回退重编码=%d (%.1f%%)" % (
                _n, _SAMPLED_IDS_STATS["used"], 100.0 * _SAMPLED_IDS_STATS["used"] / _n,
                _SAMPLED_IDS_STATS["fallback"], 100.0 * _SAMPLED_IDS_STATS["fallback"] / _n))
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)

    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
    ) -> None:
        """Add a user message and update input_ids (for multi-turn conversations).

        User messages don't contribute to loss (loss_mask=False).
        """
        self.messages.append(Message(role="user", content=content))
        enable_thinking = _enable_thinking(tokenizer)
        content = tokenizer.apply_chat_template(
            [*BASE_CHAT_HISTORY, self.messages[-1]],
            tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None),
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=enable_thinking
        )
        content_ids = tokenizer.encode(content[self.base_conv_with_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> None:
        if not contents:
            return
        self.messages.extend([Message(role="tool", content=content) for content in contents])
        enable_thinking = _enable_thinking(tokenizer)
        content = tokenizer.apply_chat_template(
            [*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]],
            tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None),
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=enable_thinking
        )
        content_ids = tokenizer.encode(content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        turn_boundaries: List[int] = [],
        conversation_histories: List[Dict[str, Any]] = [],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        self.turn_boundaries = turn_boundaries
        self.conversation_histories = conversation_histories
        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        self.truncate_output_ids(tokenizer)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=},\x20
\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20{len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
