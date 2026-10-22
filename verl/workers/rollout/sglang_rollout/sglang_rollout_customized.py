# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import re
import time
from contextlib import contextmanager
from copy import deepcopy
import json
from json import JSONDecodeError
from typing import List, Optional, Tuple
from uuid import uuid4

_FORMAT_FEEDBACK_MAX = int(os.environ.get("USERRL_FORMAT_FEEDBACK_MAX", "8"))


import numpy as np
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sglang.srt.managers.tokenizer_manager import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.openai_api.protocol import Tool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    get_ip,
    get_open_port,
    is_cuda,
    maybe_set_triton_cache_manager,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolCall,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.net_utils import is_ipv6
from verl.utils.torch_functional import (
    get_response_mask,
    pad_sequence_to_length,
)
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ===================== Qwen3 "thinking" sanitization =====================
# Qwen3 may still inject an empty "<think>...</think>" block in the rendered prompt even when
# enable_thinking=False. Generated content typically shouldn't contain it, but we strip any <think> blocks
# from generated assistant text before using it downstream to ensure clean content.
_STRIP_QWEN_THINK_OUTPUT = os.getenv("USERRL_STRIP_QWEN_THINK_OUTPUT", "0") == "1"


def _strip_think_blocks(text: str) -> str:
    """Remove any <think>...</think> blocks from text (whitespace-tolerant)."""
    if not isinstance(text, str) or not text:
        return text
    return re.sub(r"\s*<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE).strip()

# =================== end Qwen3 "thinking" sanitization ===================

# Enable detailed rollout trace for debugging
# Set VERL_ROLLOUT_TRACE=1 to enable detailed trace output for first N requests
ROLLOUT_TRACE_ENABLED = os.getenv("VERL_ROLLOUT_TRACE", "0") == "1"
ROLLOUT_TRACE_MAX_REQUESTS = int(os.getenv("VERL_ROLLOUT_TRACE_MAX", "3"))  # Trace first 3 requests by default


def find_token_subsequence(sequence: list, subsequence: list) -> Optional[Tuple[int, int]]:
    """
    Find the first occurrence of subsequence within sequence.

    Args:
        sequence: List of token IDs to search in
        subsequence: List of token IDs to search for

    Returns:
        (start_idx, end_idx) if found, where end_idx is exclusive (Python slice style)
        None if not found
    """
    if not subsequence:
        return None

    subseq_len = len(subsequence)
    seq_len = len(sequence)

    for i in range(seq_len - subseq_len + 1):
        if sequence[i:i + subseq_len] == subsequence:
            return (i, i + subseq_len)

    return None



# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class

# ---------------------------------------------------------------------------
# USERRL_RETOK_PROBE -- decode/re-encode fidelity check. Default OFF.
#
# The multi-turn path rebuilds the training sequence from the generated TEXT
# (`output["text"]` -> apply_chat_template -> tokenizer.encode), NOT from the
# token ids SGLang actually sampled. Re-encoding is not guaranteed to reproduce
# the original ids, so the tokens being trained on need not be the tokens that
# were sampled -- which would explain a `loss_mask=1` position scored at
# log_prob ~ -87 by the training forward (defect B15).
#
# Turning this on sets return_logprob=True so the sampled ids come back, and
# compares them against the re-encoded ids. It costs an extra logprob payload
# per generation, so it is a diagnostic switch, not a default.
_RETOK_PROBE = os.environ.get("USERRL_RETOK_PROBE", "0") != "0"
# Single source of truth: schemas.py owns the switch, this file only reads it.
from verl.workers.rollout.schemas import _USE_SAMPLED_IDS
_RETOK_REPORTS = int(os.environ.get("USERRL_RETOK_REPORTS", "30"))
_RETOK_STATS = {"n": 0, "len_diff": 0, "id_diff": 0}



def _sampled_ids_from(output):
    """The token ids SGLang actually sampled, or None when logprobs were off."""
    try:
        otl = (output.get("meta_info") or {}).get("output_token_logprobs")
        if not otl:
            return None
        return [t[1] for t in otl]
    except Exception:  # noqa: BLE001
        return None


def _retok_check(tokenizer, output, content):
    """Compare the ids SGLang sampled against the ids re-encoding produces.

    Diagnostic only, and wrapped whole: an instrument must never be able to kill
    the run it is instrumenting (a previous probe did exactly that).
    """
    global _RETOK_REPORTS
    if not _RETOK_PROBE:
        return
    try:
        mi = output.get("meta_info") or {}
        otl = mi.get("output_token_logprobs")
        if not otl:
            return
        sampled = [t[1] for t in otl]
        retok = tokenizer.encode(content, add_special_tokens=False)
        _RETOK_STATS["n"] += 1
        same_len = len(sampled) == len(retok)
        same_ids = same_len and sampled == retok
        if not same_len:
            _RETOK_STATS["len_diff"] += 1
        if not same_ids:
            _RETOK_STATS["id_diff"] += 1
        if not same_ids and _RETOK_REPORTS > 0:
            _RETOK_REPORTS -= 1
            k = 0
            for k in range(min(len(sampled), len(retok))):
                if sampled[k] != retok[k]:
                    break
            else:
                k = min(len(sampled), len(retok))
            lo, hi = max(0, k - 3), k + 4
            print(
                "[RETOK_MISMATCH] n_sampled=%d n_retok=%d first_diff_idx=%d "
                "sampled_ids=%s retok_ids=%s sampled_txt=%r retok_txt=%r"
                % (len(sampled), len(retok), k, sampled[lo:hi], retok[lo:hi],
                   tokenizer.decode(sampled[lo:hi]), tokenizer.decode(retok[lo:hi]))
            )
        if _RETOK_STATS["n"] % 50 == 0:
            print("[RETOK_STATS] n=%d 长度不同=%d (%.1f%%) id不同=%d (%.1f%%)" % (
                _RETOK_STATS["n"], _RETOK_STATS["len_diff"],
                100.0 * _RETOK_STATS["len_diff"] / _RETOK_STATS["n"],
                _RETOK_STATS["id_diff"],
                100.0 * _RETOK_STATS["id_diff"] / _RETOK_STATS["n"]))
    except Exception as e:  # noqa: BLE001
        globals()["_RETOK_PROBE"] = False
        print("[RETOK_MISMATCH] probe disabled after %r" % (e,))


class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

    async def release_memory_occupation(self):
        """Release GPU occupation temporarily."""
        obj = ReleaseMemoryOccupationReqInput()
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self):
        """Resume GPU occupation."""

        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False

        obj = ResumeMemoryOccupationReqInput()
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    async def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],  # noqa: UP006
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[MultiprocessingSerializer.serialize(named_tensors) for _ in range(self.server_args.tp_size)],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return await self.tokenizer_manager.update_weights_from_tensor(obj, None)

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()


# NOTE(sgm): add for verl. We can optimize it by making
#  the dataloader yield List[int] without padding.
def _pre_process_inputs(
    pad_token_id,
    prompt_token_ids: torch.Tensor,
) -> list[int]:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# NOTE(linjunrong): adhoc
def _post_process_outputs(tokenizer, output):
    def _map_each_response(resp):
        output_token_logprobs = resp["meta_info"]["output_token_logprobs"]
        log_probs, output_token_ids = zip(*[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs])
        return torch.tensor(output_token_ids), torch.tensor(log_probs)

    out_map = map(lambda x: _map_each_response(x), output)
    batched_output_token_ids = []
    batched_logprobs = []
    for output_token_ids, log_probs in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=pad_token_id)
    return batched_output_token_ids, batched_logprobs


def get_tool_call_parser_type(tokenizer: PreTrainedTokenizer) -> str:
    items = FunctionCallParser.ToolCallParserEnum.items()
    for parser_type, parser_cls in items:
        parser = parser_cls()
        if parser.bot_token.strip() in tokenizer.get_vocab() and (parser.eot_token == "" or parser.eot_token.strip() in tokenizer.get_vocab()):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for tokenizer {tokenizer}")


class ManualParsedToolCall:
    def __init__(self, name: str, arguments: dict, tool_index: int = 0):
        self.name = name
        self.parameters = arguments
        self.tool_index = tool_index


def _json_object_balance(text: str) -> int:
    balance = 0
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            balance += 1
        elif ch == "}":
            balance -= 1
    return balance


def _loads_json_with_brace_repair(text: str):
    stripped = text.strip()
    candidates = [stripped]
    balance = _json_object_balance(stripped)
    if 0 < balance <= 8:
        candidates.append(stripped + ("}" * balance))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _build_tool_indices(function_call_parser, tool_map: dict | None) -> dict[str, int]:
    if function_call_parser is not None and hasattr(function_call_parser, "tools") and function_call_parser.tools:
        return {
            tool.function.name: i
            for i, tool in enumerate(function_call_parser.tools)
            if tool.function and tool.function.name
        }
    if tool_map:
        return {name: i for i, name in enumerate(tool_map.keys())}
    return {}


def _parse_tool_calls_json_fallback(content: str, tool_indices: dict[str, int]) -> list[ManualParsedToolCall]:
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    manual_tool_calls = []

    for fallback_index, match in enumerate(tool_call_pattern.findall(content)):
        parsed = _loads_json_with_brace_repair(match)
        if not isinstance(parsed, dict):
            continue

        name = parsed.get("name")
        arguments = parsed.get("arguments", parsed.get("parameters"))
        if isinstance(arguments, str):
            decoded_arguments = _loads_json_with_brace_repair(arguments)
            if isinstance(decoded_arguments, dict):
                arguments = decoded_arguments

        if not name or not isinstance(arguments, dict):
            continue

        if name not in tool_indices and "interact_with_env" in tool_indices and name in {"action", "answer", "search"}:
            arguments = {
                "choice": arguments.get("choice", name),
                "content": arguments.get("content", ""),
            }
            name = "interact_with_env"

        manual_tool_calls.append(
            ManualParsedToolCall(
                name=name,
                arguments=arguments,
                tool_index=tool_indices.get(name, fallback_index),
            )
        )

    return manual_tool_calls


class SGLangRollout(BaseRollout):
    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Synchronized SGLang rollout engine.

        Args:
            actor_module: Huggingface model name or path to the model. The
                model should be supported by SGLang.
            config: A DictConfig object containing SGLang-specific operational
                parameters and rollout settings.
                Refer to https://docs.sglang.ai/backend/server_arguments.html
            tokenizer: The tokenizer instance compatible with the actor_module.
            model_hf_config: The Hugging Face model's configuration (e.g.,
                `transformers.PretrainedConfig`). It provides architectural
                details and hyperparameters like `max_position_embeddings`,
                used by SGLang for correct model initialization. This is
                the model's inherent design, not SGLang's runtime behavior.
            port: Optional port for multi-node initialization when nnodes > 1.
            trust_remote_code: Whether or not to allow for custom models
                defined on the Hub in their own modeling files.
            device_mesh: Optional `DeviceMesh` object for distributed setup.
            **kwargs: Additional keyword arguments, primarily `train_tp` for
                Megatron Backend integration to initialize hybrid engine
                process groups.
        """
        super().__init__()
        self.config = config
        self._device_mesh_cpu = device_mesh
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")

        (
            self._tool_schemas,
            self._tool_map,
            self._tool_call_parser_type,
            self._sgl_tools,
            self._function_call_parser,
        ) = self._initialize_tools(config, tokenizer)
        # If turn on `free_cache_engine`, SGLang engine's KV cache
        # will be freed after each `generate_sequences` call.
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tool_names = [tool.get("function", {}).get("name", "unknown") if isinstance(tool, dict) else getattr(tool, "function", {}).get("name", "unknown") for tool in self._tool_schemas] if self._tool_schemas else []

        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)

        self._verify_config(model_hf_config=model_hf_config)
        # initialize the inference engine
        self._init_inference_engine(trust_remote_code, actor_module, port)

        self._init_sampling_params(**kwargs)

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self._satisfaction_probe_config = config.multi_turn.get("satisfaction_probe", {})
        self._satisfaction_probe_scorer = None
        # Silent repair of malformed tool calls (fallback parser + content
        # rewrite) is a LOCAL addition (upstream UserRL has none): it rescues
        # episodes but rewrites the assistant message into the parser's dialect,
        # polluting the model's own context. Set tool_call_repair=false to
        # restore upstream behavior (unparseable tool call = episode ends).
        self._tool_call_repair = bool(config.multi_turn.get("tool_call_repair", True))
        # Local addition: per-request rollout failures are converted to
        # zero-reward synthetic samples; false = upstream fail-loud behavior.
        self._rollout_failure_fallback = bool(config.get("rollout_failure_fallback", False))
        # Format-cost shaping: make tool-protocol quality visible to the reward.
        # Repaired tool calls cost a small penalty, unparseable ones (episode is
        # terminated) cost a larger one. Both default to 0 (upstream behavior).
        # no_tool_call_penalty prices plain-text responses with no tool call at
        # all (defaults to the fatal penalty); multi_tool_penalty prices every
        # tool call beyond the first in a single turn (defaults to the repair
        # penalty).
        self._format_repair_penalty = float(config.multi_turn.get("format_repair_penalty", 0.0))
        self._format_fatal_penalty = float(config.multi_turn.get("format_fatal_penalty", 0.0))
        self._format_no_tool_call_penalty = float(config.multi_turn.get("format_no_tool_call_penalty", self._format_fatal_penalty))
        self._format_multi_tool_penalty = float(config.multi_turn.get("format_multi_tool_penalty", self._format_repair_penalty))
        self._format_error_feature_value = float(config.multi_turn.get("format_error_feature_value", 0.0))
        self._format_error_feedback = bool(config.multi_turn.get("format_error_feedback", True))
        self._format_repair_pending = set()
        self._fmt_stats = {"repair": 0, "fatal": 0, "no_tool_call": 0, "multi_tool": 0,
                           "raw_reward_sum": 0.0, "penalty_sum": 0.0, "shaped_reward_sum": 0.0}

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp"))
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

    def _verify_config(self, model_hf_config):
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length
        assert self.config.max_model_len >= self.config.prompt_length + self.config.response_length, f"""max_model_len should be greater than total sequence length (prompt_length + response_length):\x20
            {self.config.max_model_len} >= {self.config.prompt_length} + {self.config.response_length}"""
        assert model_hf_config.max_position_embeddings >= self.config.max_model_len, "model context length should be greater than total sequence length"
        # currently max_turns stand for max number of tool calls
        if self.config.multi_turn.max_turns is None:
            self.config.multi_turn.max_turns = self.config.max_model_len // 3

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            rank = dist.get_rank()
            rollout_port = self._get_sglang_rollout_port(rank)
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = AsyncEngine(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                port=rollout_port,
            )
        else:
            self._engine = None

        self.sharding_manager = None
        self.is_sleep = True

    @staticmethod
    def _get_sglang_rollout_port(rank: int) -> int:
        base_port = int(os.environ.get("SGLANG_ROLLOUT_PORT_BASE", "30000"))
        stride = max(int(os.environ.get("SGLANG_ROLLOUT_PORT_STRIDE", "1000")), 1)
        pid_offset = os.getpid() % stride
        return base_port + rank * stride + pid_offset

    def _init_sampling_params(self, **kwargs):
        kwargs = dict(
            n=1,
            max_new_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
        # supporting adding any sampling params from the config file
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = self.config.get(k)
        self.sampling_params = kwargs

    def _initialize_tools(self, config, tokenizer):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool-related settings,
                    specifically `config.multi_turn.tool_config_path`.
            tokenizer: The tokenizer instance used for parsing tool calls from
                       the model's generated text.

        Returns:
            tuple: A tuple containing:
                - tool_schemas (list[dict]): OpenAI-formatted JSON schemas
                  defining each tool's capabilities.
                - tool_map (dict[str, BaseTool]): A dictionary mapping tool
                  names to their executable `BaseTool` objects.
                - tool_call_parser_type (str): The identifier for the specific
                  parser type (e.g., 'json_mode', 'tool_code') used to extract
                  tool calls.
                - sgl_tools (list[sglang.srt.openai_api.protocol.Tool]): Tool
                  definitions optimized for SGLang's internal engine.
                - function_call_parser (sglang.srt.function_call_parser.FunctionCallParser):
                  The active parser instance responsible for extracting
                  structured tool calls from model outputs.
        """
        if config.multi_turn.tool_config_path is None:
            return [], {}, None, [], None

        import importlib.util
        import sys

        from omegaconf import OmegaConf

        from verl.tools.schemas import OpenAIFunctionToolSchema

        def initialize_tools_from_config(tools_config) -> list:
            tool_list = []

            for tool_config in tools_config.tools:
                cls_name = tool_config.class_name
                module_name, class_name = cls_name.rsplit(".", 1)

                if module_name not in sys.modules:
                    spec = importlib.util.find_spec(module_name)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = sys.modules[module_name]

                tool_cls = getattr(module, class_name)

                tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)

                tool = tool_cls(
                    config=OmegaConf.to_container(tool_config.config, resolve=True),
                    tool_schema=tool_schema,
                )
                tool_list.append(tool)

            return tool_list

        tools_config_file = config.multi_turn.tool_config_path
        tools_config = OmegaConf.load(tools_config_file)
        tool_list = initialize_tools_from_config(tools_config)
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(tokenizer)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """
        Temporarily updates the model's sampling parameters for the
        duration of a `with` block. Parameters are automatically fall
          back to their original values upon exiting the block.

        Args:
            **kwargs: Keyword arguments representing sampling parameters
                    to be updated. Only parameters that already exist in
                    `self.sampling_params` will be updated.
        """
        # Store original values of parameters that will be updated
        old_sampling_params_args = {key: self.sampling_params[key] for key in kwargs if key in self.sampling_params}

        # Update sampling parameters with new values
        for key, value in kwargs.items():
            if key in self.sampling_params:
                self.sampling_params[key] = value

        try:
            yield
        finally:
            for key, value in old_sampling_params_args.items():
                self.sampling_params[key] = value

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self.config.multi_turn.enable:
            return self._req_level_generate_sequences(prompts, **kwargs)
        return self._batch_level_generate_sequences(prompts, **kwargs)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generates sequences for a batch of prompts.
        For single-turn generation, all prompts are processed in one request.
        For multi-turn generation, each prompt is processed separately via
        `_generate_req_level_sequences` for better tool calling control.
        `_generate_batch_level_sequences` involves:
        1.  Extracting and pre-processing prompt token IDs from the input
            `prompts`. This includes handling padding and preparing raw
            token ID lists.
        2.  Preparing inputs for the SGLang engine, including multi-modal
            data if present.
        3.  Invoking the SGLang engine (`self._engine.async_generate`,
            an async coroutine) with the batch of processed inputs and
            specified sampling parameters on the master TP rank.
        4.  Broadcasting the results from the master TP rank to all
            other TP ranks.
        5.  Post-processing the engine's output to format the generated
            token IDs and (if applicable) log probabilities.
        6.  Constructing the final sequences by concatenating original
            prompts with the generated responses.
        7.  Updating attention masks and position IDs to reflect the full
            concatenated sequences.
        8.  If `self.config.free_cache_engine` is true, the SGLang engine's
            KV cache is flushed after generation on the master TP rank.
        Args:
            prompts: A `DataProto` object containing the batch of
              input prompts, including tensor data (like `input_ids`,
              `attention_mask`) and meta-information (like `eos_token_id`,
              `do_sample`).
            **kwargs: Additional keyword arguments that can override the
              default sampling parameters (e.g., `temperature`, `top_p`,
              `max_new_tokens`). These are temporarily applied using
              `update_sampling_params`.
        Returns:
            DataProto: A `DataProto` object containing the batch of
              generated sequences. This includes tensors for `prompts`
              (original input IDs), `responses` (generated token IDs),
              `input_ids` (concatenated prompt and response),
              `attention_mask`, and `position_ids` for the full
              sequences.
        Note that when `n > 1`, each prompt generates multiple sequences,
        so we need to replicate its non-tensor data (i.e. raw prompts,
        messages, reward scores, etc.) n times to match the expanded
        tensor data. This is done in the `_non_tensor_batch` dictionary.
        """
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None),
                    }
                )
        else:
            sglang_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # Ensure token IDs are lists or numpy arrays
        for input_data in sglang_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            kwargs = dict(
                top_k=self.config.val_kwargs.top_k,
                top_p=self.config.val_kwargs.top_p,
                temperature=self.config.val_kwargs.temperature,
                n=1,  # if validate, already repeat in ray_trainer
            )

        with self.update_sampling_params(**kwargs):
            if self._tp_rank == 0:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=None,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        return_logprob=True,
                        input_ids=idx_list,
                        image_data=image_list,
                    )
                )
            else:
                output = None

            [output] = broadcast_pyobj(
                data=[output],
                rank=self._rank,
                dist_group=self._device_mesh_cpu["tp"].get_group(),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            out = _post_process_outputs(self.tokenizer, output)

            response = out[0].to(idx.device)
            rollout_log_probs = out[1].to(idx.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                rollout_log_probs = pad_sequence_to_length(rollout_log_probs, self.config.response_length, self.pad_token_id)

            # utilize current sampling params
            if self.sampling_params.get("n", 1) > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params["n"], dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params["n"], dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params["n"], dim=0)
                batch_size = batch_size * self.sampling_params["n"]
                _non_tensor_batch = {}
                for key, val in non_tensor_batch.items():
                    _non_tensor_batch[key] = np.repeat(val, self.sampling_params["n"], axis=0)
            else:
                _non_tensor_batch = non_tensor_batch
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free cache engine
        if self.config.free_cache_engine and self._engine is not None:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        return DataProto(batch=batch, non_tensor_batch=_non_tensor_batch)

    async def _semaphore_wrapped_rollout(self, sem, req, do_sample, is_validate, **kwargs):
        async with sem:
            try:
                return await self._async_rollout_a_request(req, do_sample, is_validate, **kwargs)
            except Exception as exc:
                # Local addition (upstream UserRL lets the exception crash the
                # rollout): any per-request failure (env crash, user-sim
                # timeout, tool error) is silently converted into a fabricated
                # zero-reward trajectory that enters the training batch as a
                # synthetic negative sample. Set rollout_failure_fallback=false
                # to restore upstream fail-loud behavior.
                if "_tau2_env_kwargs" in req.tools_kwargs or not getattr(self, "_rollout_failure_fallback", False):
                    raise
                logger.exception(
                    "[ROLLOUT_FALLBACK] Request %s failed; returning a zero-reward fallback sample",
                    req.request_id[:8],
                )
                await self._cleanup_failed_rollout(req)
                return self._finalize_failed_rollout_request(req, exc)
            finally:
                if isinstance(req.tools_kwargs, dict) and "_tau2_env_kwargs" in req.tools_kwargs:
                    from verl.tools.env_manager import get_environment_manager
                    get_environment_manager().release_environment(req.request_id)

    async def _cleanup_failed_rollout(self, req: AsyncRolloutRequest) -> None:
        """Best-effort cleanup for request-level rollout failures."""
        try:
            from verl.tools.env_manager import get_environment_manager

            get_environment_manager().release_environment(req.request_id)
        except Exception:
            logger.debug("[ROLLOUT_FALLBACK] Failed to release environment for %s", req.request_id[:8], exc_info=True)

        if not isinstance(req.tools_kwargs, dict):
            return

        dynamic_tool_map = req.tools_kwargs.get("_dynamic_tool_map", None)
        tool_map_to_use = dynamic_tool_map if dynamic_tool_map else self._tool_map
        for name in [key for key in req.tools_kwargs.keys() if not key.startswith("_")]:
            tool = tool_map_to_use.get(name) if tool_map_to_use else None
            if tool is None or not hasattr(tool, "release"):
                continue
            try:
                await tool.release(req.request_id, **req.tools_kwargs.get(name, {}).get("release_kwargs", {}))
            except Exception:
                logger.debug(
                    "[ROLLOUT_FALLBACK] Failed to release tool %s for %s",
                    name,
                    req.request_id[:8],
                    exc_info=True,
                )

    def _finalize_failed_rollout_request(self, req: AsyncRolloutRequest, exc: Exception) -> AsyncRolloutRequest:
        fallback_content = os.getenv(
            "VERL_ROLLOUT_FAILURE_FALLBACK_TEXT",
            "I could not complete this interaction.",
        )
        error_summary = f"Rollout failed before completion: {type(exc).__name__}: {exc}"

        if len(req.input_ids) + 1 >= req.max_model_len:
            keep_len = max(len(req.prompt_ids), req.max_model_len - 1)
            req.input_ids = req.input_ids[:keep_len]
            req.attention_mask = req.attention_mask[:keep_len]
            req.position_ids = req.position_ids[:keep_len]
            req.loss_mask = req.loss_mask[:keep_len]

        action_start = len(req.input_ids)
        try:
            req.add_assistant_message(self.tokenizer, fallback_content)
        except Exception:
            logger.debug(
                "[ROLLOUT_FALLBACK] add_assistant_message failed for %s; appending raw fallback tokens",
                req.request_id[:8],
                exc_info=True,
            )
            fallback_ids = self.tokenizer.encode(fallback_content, add_special_tokens=False)
            if not fallback_ids:
                fallback_ids = [self.tokenizer.eos_token_id or self.pad_token_id]
            remaining = max(1, req.max_model_len - len(req.input_ids))
            req._update_input_ids(fallback_ids[:remaining], attention_mask=True, loss_mask=True)
            req.messages.append(Message(role="assistant", content=fallback_content))

        action_end = len(req.input_ids)
        reward_scores = {"interact_with_env": 0.0}
        conversation_histories = [
            {
                "reward": 0.0,
                "choice": "action",
                "content": fallback_content,
                "env_feedback": error_summary,
                "turn_idx": 0,
                "action_start": action_start,
                "action_end": action_end,
                "obs_start": None,
                "obs_end": None,
                "reward_scores": reward_scores,
                "rollout_error": error_summary,
            }
        ]
        turn_boundaries = [action_start] if action_start < req.max_model_len else []
        req.finalize(
            self.tokenizer,
            reward_scores,
            turn_boundaries,
            conversation_histories,
            FinishReasonTypeEnum.STOP,
        )
        return req

    def _satisfaction_probe_enabled(self) -> bool:
        cfg = self._satisfaction_probe_config
        if cfg is None:
            return False
        try:
            return bool(cfg.get("enable", False))
        except AttributeError:
            return False

    def _get_satisfaction_probe_scorer(self):
        if self._satisfaction_probe_scorer is None:
            from verl.utils.satisfaction_probe import SatisfactionProbeScorer

            probe_config = dict(self._satisfaction_probe_config)
            configured_device = str(probe_config.get("device", "cuda"))
            if configured_device == "cuda" and torch.cuda.is_available():
                visible_devices = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
                if len(visible_devices) <= 1:
                    device_index = 0
                else:
                    device_index = self._rank % torch.cuda.device_count()
                probe_config["device"] = f"cuda:{device_index}"
            logger.info(
                "[SAT_PROBE] initializing scorer for rollout rank %d on %s",
                self._rank,
                probe_config.get("device", configured_device),
            )
            self._satisfaction_probe_scorer = SatisfactionProbeScorer(probe_config)
        return self._satisfaction_probe_scorer

    @staticmethod
    def _message_role_for_probe(message) -> str:
        if isinstance(message, dict):
            return str(message.get("role", ""))
        return str(getattr(message, "role", ""))

    @staticmethod
    def _message_content_for_probe(message) -> str:
        if isinstance(message, dict):
            return str(message.get("content", "") or "")
        return str(getattr(message, "content", "") or "")

    def _extract_initial_messages_for_probe(self, req: AsyncRolloutRequest) -> list:
        messages = list(req.messages or [])
        histories = list(req.conversation_histories or [])
        if not messages:
            return []

        if histories:
            first_content = str(histories[0].get("content", "") or "").strip()
            if first_content:
                for idx, message in enumerate(messages):
                    role = self._message_role_for_probe(message)
                    content = self._message_content_for_probe(message).strip()
                    if role == "assistant" and content == first_content:
                        return messages[:idx]

        initial_messages = []
        for message in messages:
            role = self._message_role_for_probe(message)
            if role in {"assistant", "tool"}:
                break
            initial_messages.append(message)
        return initial_messages

    def _log_format_stats(self):
        """Emit and reset per-batch format-cost counters (grep-able in logs)."""
        s = self._fmt_stats
        if not any([s["repair"], s["fatal"], s["no_tool_call"], s["multi_tool"], s["penalty_sum"]]):
            return
        logger.warning(
            "[FORMAT_COST] repair=%d fatal=%d no_tool_call=%d multi_tool=%d raw_reward_sum=%.3f penalty_sum=%.3f shaped_reward_sum=%.3f",
            s["repair"], s["fatal"], s["no_tool_call"], s["multi_tool"],
            s["raw_reward_sum"], s["penalty_sum"], s["shaped_reward_sum"],
        )
        self._fmt_stats = {"repair": 0, "fatal": 0, "no_tool_call": 0, "multi_tool": 0,
                           "raw_reward_sum": 0.0, "penalty_sum": 0.0, "shaped_reward_sum": 0.0}

    def _apply_satisfaction_probe_rewards_to_requests(self, req_list: list[AsyncRolloutRequest]) -> None:
        if not self._satisfaction_probe_enabled() or not req_list:
            return

        cfg = self._satisfaction_probe_config
        reward_mode = str(cfg.get("reward_mode", "replace"))
        probe_weight = float(cfg.get("probe_weight", 1.0))
        task_reward_weight = float(cfg.get("task_reward_weight", 1.0))
        delta_gamma = float(cfg.get("delta_gamma", 1.0))
        delta_clip = cfg.get("delta_clip", None)
        if delta_clip is not None:
            delta_clip = float(delta_clip)
            if delta_clip <= 0:
                delta_clip = None
        use_td_delta = reward_mode in {"td_delta", "td_delta_add_task", "td_advantage_only"}
        use_gated_probe_mean = reward_mode == "task_plus_gated_probe_mean"
        gate_temperature = float(cfg.get("gate_temperature", 0.5))
        gate_min = float(cfg.get("gate_min", 0.0))
        gate_max = float(cfg.get("gate_max", 1.0))
        if use_gated_probe_mean:
            if gate_temperature <= 0:
                raise ValueError("satisfaction probe gate_temperature must be positive")
            if not 0.0 <= gate_min <= gate_max <= 1.0:
                raise ValueError("satisfaction probe gate bounds must satisfy 0 <= gate_min <= gate_max <= 1")
        trajectory_score = str(cfg.get("trajectory_score", "sum"))
        missing_score_reward = str(cfg.get("missing_score_reward", "original"))

        request_histories = []
        scored_req_indices = []
        for req_idx, req in enumerate(req_list):
            histories = req.conversation_histories or []
            if not histories:
                continue
            request_histories.append(
                {
                    "initial_messages": self._extract_initial_messages_for_probe(req),
                    "conversation_history": histories,
                }
            )
            scored_req_indices.append(req_idx)

        if not request_histories:
            return

        scorer = self._get_satisfaction_probe_scorer()
        if use_td_delta:
            scores_by_scored_request, initial_scores, stats = scorer.score_requests(
                request_histories,
                include_initial_state=True,
            )
        else:
            scores_by_scored_request, stats = scorer.score_requests(request_histories)
            initial_scores = [None] * len(request_histories)

        # InfoPO-style auxiliary gate: use the group-level task-reward variance
        # to decide whether the probe should supplement a sparse task signal.
        # batch_data_id identifies all n rollouts generated for one prompt.
        gated_probe_metadata = {}
        if use_gated_probe_mean:
            grouped_request_indices = {}
            for local_idx, req_idx in enumerate(scored_req_indices):
                batch_data_id = req_list[req_idx].batch_data_id
                grouped_request_indices.setdefault(batch_data_id, []).append(local_idx)

            for local_indices in grouped_request_indices.values():
                task_totals = []
                for local_idx in local_indices:
                    histories = req_list[scored_req_indices[local_idx]].conversation_histories or []
                    task_totals.append(sum(float(turn.get("reward", 0.0) or 0.0) for turn in histories))

                task_std = float(np.std(np.asarray(task_totals, dtype=np.float32))) if len(task_totals) > 1 else 0.0
                gate = float(1.0 / (1.0 + np.exp(task_std / gate_temperature)))
                gate = float(np.clip(gate, gate_min, gate_max))
                for local_idx in local_indices:
                    gated_probe_metadata[local_idx] = {
                        "gate": gate,
                        "task_group_std": task_std,
                        "group_size": len(local_indices),
                    }

        for local_idx, req_idx in enumerate(scored_req_indices):
            req = req_list[req_idx]
            histories = req.conversation_histories or []
            probe_scores = scores_by_scored_request[local_idx]
            updated_turn_rewards = []
            observed_probe_scores = []
            observed_probe_deltas = []
            initial_probe_score = initial_scores[local_idx]
            previous_probe_score = float(initial_probe_score or 0.0)
            task_reward_component_sum = 0.0
            probe_reward_component_sum = 0.0
            scored_turn_indices = []

            for turn, probe_score in zip(histories, probe_scores):
                original_reward = float(turn.get("reward", 0.0) or 0.0)
                turn["original_env_reward"] = original_reward
                if probe_score is not None:
                    probe_score = float(probe_score)
                    turn["probe_satisfaction"] = probe_score
                    observed_probe_scores.append(probe_score)
                    if reward_mode == "replace":
                        new_reward = probe_score
                    elif reward_mode == "add":
                        new_reward = original_reward + probe_weight * probe_score
                    elif use_gated_probe_mean:
                        # The gated trajectory-level probe component is distributed
                        # after all turn scores are known, so Sum is length-invariant.
                        new_reward = original_reward
                        scored_turn_indices.append(len(updated_turn_rewards))
                    elif reward_mode == "advantage_only":
                        # Keep task reward untouched. The trainer later maps this
                        # turn's probe score to a separate InfoPO intrinsic-advantage branch.
                        new_reward = original_reward
                    elif reward_mode == "blend":
                        new_reward = (1.0 - probe_weight) * original_reward + probe_weight * probe_score
                    elif use_td_delta:
                        probe_delta = delta_gamma * probe_score - previous_probe_score
                        if delta_clip is not None:
                            probe_delta = float(np.clip(probe_delta, -delta_clip, delta_clip))
                        turn["probe_satisfaction_delta"] = float(probe_delta)
                        observed_probe_deltas.append(float(probe_delta))
                        probe_component = probe_weight * probe_delta
                        task_component = task_reward_weight * original_reward if reward_mode == "td_delta_add_task" else 0.0
                        new_reward = original_reward if reward_mode == "td_advantage_only" else task_component + probe_component
                        task_reward_component_sum += task_component
                        probe_reward_component_sum += probe_component
                        previous_probe_score = probe_score
                    else:
                        raise ValueError(f"Unsupported satisfaction probe reward_mode: {reward_mode}")
                else:
                    turn["probe_satisfaction"] = None
                    if use_td_delta:
                        turn["probe_satisfaction_delta"] = None
                        task_component = task_reward_weight * original_reward if reward_mode == "td_delta_add_task" else 0.0
                        new_reward = original_reward if reward_mode == "td_advantage_only" else task_component
                        task_reward_component_sum += task_component
                    elif missing_score_reward == "zero":
                        new_reward = 0.0
                    elif missing_score_reward == "original":
                        new_reward = original_reward
                    else:
                        new_reward = float(missing_score_reward)

                turn["reward"] = float(new_reward)
                updated_turn_rewards.append(float(new_reward))

            if use_gated_probe_mean:
                gate_info = gated_probe_metadata.get(local_idx, {"gate": 1.0, "task_group_std": 0.0, "group_size": 1})
                probe_mean = float(np.mean(observed_probe_scores)) if observed_probe_scores else 0.0
                probe_reward_component_sum = probe_weight * float(gate_info["gate"]) * probe_mean
                task_reward_component_sum = float(sum(float(turn.get("original_env_reward", 0.0) or 0.0) for turn in histories))
                if scored_turn_indices:
                    per_turn_component = probe_reward_component_sum / len(scored_turn_indices)
                    for turn_index in scored_turn_indices:
                        updated_turn_rewards[turn_index] += per_turn_component
                        histories[turn_index]["probe_gated_reward_component"] = float(per_turn_component)
                        histories[turn_index]["reward"] = float(updated_turn_rewards[turn_index])

            if trajectory_score == "mean":
                final_reward = float(np.mean(updated_turn_rewards)) if updated_turn_rewards else 0.0
            elif trajectory_score == "max":
                final_reward = float(np.max(updated_turn_rewards)) if updated_turn_rewards else 0.0
            elif trajectory_score == "last":
                final_reward = float(updated_turn_rewards[-1]) if updated_turn_rewards else 0.0
            elif trajectory_score == "sum":
                final_reward = float(np.sum(updated_turn_rewards)) if updated_turn_rewards else 0.0
            else:
                raise ValueError(f"Unsupported satisfaction probe trajectory_score: {trajectory_score}")

            reward_scores = dict(req.reward_scores or {})
            if "interact_with_env" in reward_scores:
                reward_scores.setdefault("interact_with_env_original", reward_scores["interact_with_env"])
            reward_scores["interact_with_env"] = final_reward
            if observed_probe_scores:
                reward_scores["probe_satisfaction_mean"] = float(np.mean(observed_probe_scores))
                reward_scores["probe_satisfaction_min"] = float(np.min(observed_probe_scores))
                reward_scores["probe_satisfaction_max"] = float(np.max(observed_probe_scores))
                reward_scores["probe_satisfaction_final"] = float(observed_probe_scores[-1])
            if use_td_delta:
                reward_scores["probe_satisfaction_initial"] = float(initial_probe_score or 0.0)
                reward_scores["probe_satisfaction_improvement"] = float(previous_probe_score - float(initial_probe_score or 0.0))
                reward_scores["probe_task_reward_component_sum"] = float(task_reward_component_sum)
                reward_scores["probe_delta_reward_component_sum"] = float(probe_reward_component_sum)
                if observed_probe_deltas:
                    reward_scores["probe_satisfaction_delta_mean"] = float(np.mean(observed_probe_deltas))
                    reward_scores["probe_satisfaction_delta_min"] = float(np.min(observed_probe_deltas))
                    reward_scores["probe_satisfaction_delta_max"] = float(np.max(observed_probe_deltas))
            if use_gated_probe_mean:
                gate_info = gated_probe_metadata.get(local_idx, {"gate": 1.0, "task_group_std": 0.0, "group_size": 1})
                reward_scores["probe_gate"] = float(gate_info["gate"])
                reward_scores["probe_task_group_std"] = float(gate_info["task_group_std"])
                reward_scores["probe_group_size"] = float(gate_info["group_size"])
                reward_scores["probe_task_reward_component_sum"] = float(task_reward_component_sum)
                reward_scores["probe_gated_reward_component_sum"] = float(probe_reward_component_sum)
            req.reward_scores = reward_scores

        logger.warning(
            "[SAT_PROBE] scored %d requests, %.0f turns in %.2fs, mean=%.4f min=%.4f max=%.4f",
            len(scored_req_indices),
            stats.get("count", 0.0),
            stats.get("seconds", 0.0),
            stats.get("mean", 0.0),
            stats.get("min", 0.0),
            stats.get("max", 0.0),
        )

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"

        # CRITICAL: Remove _dynamic_tool_map from tools_kwargs before deepcopy
        # _dynamic_tool_map contains tool objects that may have unpicklable objects (thread locks, etc.)
        _dynamic_tool_map_backup = None
        if req.tools_kwargs and "_dynamic_tool_map" in req.tools_kwargs:
            _dynamic_tool_map_backup = req.tools_kwargs.pop("_dynamic_tool_map")

        _req = deepcopy(req)

        # Restore _dynamic_tool_map after deepcopy
        if _dynamic_tool_map_backup is not None:
            _req.tools_kwargs["_dynamic_tool_map"] = _dynamic_tool_map_backup
        # Dynamic domains need a request-local parser populated with their
        # exact native schemas. Never mutate a parser shared by other episodes.
        function_call_parser = self._function_call_parser
        if _dynamic_tool_map_backup is not None:
            request_tools = [
                Tool.model_validate(schema.model_dump() if hasattr(schema, "model_dump") else schema)
                for schema in _req.tool_schemas
            ]
            function_call_parser = FunctionCallParser(request_tools, self._tool_call_parser_type)
        finish_reason_type = None
        output = None

        turn_boundaries = []
        conversation_histories = []

        current_turns = 0

        request_short_id = _req.request_id[:8]
        while current_turns < self.config.multi_turn.max_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls

                    obs_block_start = len(_req.input_ids)

                    dynamic_tool_map = _req.tools_kwargs.get("_dynamic_tool_map", None)
                    tool_map_to_use = dynamic_tool_map if dynamic_tool_map else self._tool_map

                    # Execute tool calls
                    async def _execute_single_tool_call(tool_call):
                        tool_name = tool_call.function.name
                        if tool_name not in tool_map_to_use:
                            return (
                                f"Error: Tool '{tool_name}' not found",
                                0.0,
                                False,
                                "action",
                                "",
                                {}
                            )

                        try:
                            result = await tool_map_to_use[tool_name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                current_turns,
                                **_req.tools_kwargs.get(tool_name, {}).get("execute_kwargs", {}),
                            )
                            # Handle different return formats
                            if len(result) == 3:
                                resp, reward, metrics = result
                                is_done = bool(metrics.get("terminated") or metrics.get("truncated"))
                                choice, content_param = "action", ""
                            elif len(result) >= 6:
                                resp, reward, is_done, choice, content_param, metrics = result[:6]
                            else:
                                resp, reward = result[0], result[1]
                                is_done, choice, content_param, metrics = False, "action", "", {}

                            return (resp, reward, is_done, choice, content_param, metrics)
                        except Exception as e:
                            if "_tau2_env_kwargs" in _req.tools_kwargs:
                                raise RuntimeError("Native tau3 tool execution failed") from e
                            import traceback
                            traceback.print_exc()
                            return (
                                f"Error executing tool '{tool_name}': {str(e)}",
                                0.0,
                                False,
                                "action",
                                "",
                                {}
                            )

                    tool_call_results = await asyncio.gather(
                        *[_execute_single_tool_call(tc) for tc in parsed_tool_calls]
                    )

                    # Add tool responses to conversation
                    _req.add_tool_response_messages(self.tokenizer, [resp for resp, _, _, _, _, _ in tool_call_results])

                    # Record the end position after tool responses are added
                    obs_block_end = len(_req.input_ids)

                    overall_stop = False
                    multi_penalty_applied = False
                    for tool_call, (resp, reward, is_done, choice, content, metrics) in zip(parsed_tool_calls, tool_call_results):
                        _req.update_metrics(metrics, tool_call.function.name)
                        # Format cost bookkeeping: price protocol defects into the
                        # turn reward and keep raw/shaped sums for batch metrics.
                        self._fmt_stats["raw_reward_sum"] += float(reward)
                        turn_penalty = 0.0
                        turn_format_feature = 0.0
                        if _req.request_id in self._format_repair_pending:
                            self._format_repair_pending.discard(_req.request_id)
                            turn_penalty += self._format_repair_penalty
                            self._fmt_stats["repair"] += 1
                            turn_format_feature = self._format_error_feature_value
                        extra_tool_calls = max(0, len(parsed_tool_calls) - 1)
                        if extra_tool_calls > 0 and not multi_penalty_applied and (self._format_multi_tool_penalty > 0 or self._format_error_feature_value != 0.0):
                            multi_penalty_applied = True
                            turn_penalty += extra_tool_calls * self._format_multi_tool_penalty
                            self._fmt_stats["multi_tool"] += extra_tool_calls
                            turn_format_feature = self._format_error_feature_value
                        if turn_penalty > 0:
                            reward = float(reward) - turn_penalty
                            self._fmt_stats["penalty_sum"] += turn_penalty
                        self._fmt_stats["shaped_reward_sum"] += float(reward)
                        conversation_histories[-1]["choice"] = choice
                        conversation_histories[-1]["reward"] = reward
                        conversation_histories[-1]["content"] = content
                        conversation_histories[-1]["env_feedback"] = resp  # Save environment feedback (responder reply, search results, etc.)
                        if turn_format_feature != 0.0:
                            conversation_histories[-1]["format_error_feature"] = turn_format_feature

                        # Get character-level offsets from interact_tool
                        obs_char_start = metrics.get("obs_char_start", None)
                        obs_char_end = metrics.get("obs_char_end", None)
                        pure_obs = metrics.get("pure_observation", None)

                        if obs_char_start is not None and obs_char_end is not None and pure_obs:
                            # Decode the added token block to text
                            added_tokens = _req.input_ids[obs_block_start:obs_block_end]
                            added_text = self.tokenizer.decode(added_tokens, skip_special_tokens=False)

                            # Find where resp appears in added_text (after chat template prefix)
                            resp_start_in_added = added_text.find(resp)

                            if resp_start_in_added != -1:
                                # Calculate absolute character positions in added_text
                                obs_char_start_abs = resp_start_in_added + obs_char_start
                                obs_char_end_abs = resp_start_in_added + obs_char_end

                                # Tokenize prefix (everything before observation)
                                prefix_text = added_text[:obs_char_start_abs]
                                prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)

                                # Tokenize prefix + observation
                                prefix_plus_obs_text = added_text[:obs_char_end_abs]
                                prefix_plus_obs_tokens = self.tokenizer.encode(prefix_plus_obs_text, add_special_tokens=False)

                                # Record absolute token positions
                                conversation_histories[-1]["obs_start"] = obs_block_start + len(prefix_tokens)
                                conversation_histories[-1]["obs_end"] = obs_block_start + len(prefix_plus_obs_tokens)

                            else:
                                # resp not found in added text - use block boundaries as fallback
                                logger.warning(f"Response text not found in added tokens. Using block boundaries.")
                                conversation_histories[-1]["obs_start"] = obs_block_start
                                conversation_histories[-1]["obs_end"] = obs_block_end
                        else:
                            # No offset information available - use block boundaries
                            conversation_histories[-1]["obs_start"] = obs_block_start
                            conversation_histories[-1]["obs_end"] = obs_block_end

                        if is_done:
                            overall_stop = True

                    if overall_stop or len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # Only continue the conversation if the prompt length is not greater than max_model_len - 1,
                # since SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra token accounts for the EOS token).
                if len(_req.get_generation_prompt_ids(self.tokenizer)) + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break
                turn_boundaries.append(len(_req.input_ids))
                conversation_histories.append({
                    "reward": 0.0,
                    "choice": "action",
                    "content": "",
                    "env_feedback": "",  # Will be filled after environment step
                    "turn_idx": len(conversation_histories),
                    "action_start": len(_req.input_ids),
                    "action_end": None,
                    "obs_start": None,
                    "obs_end": None,
                })

                output = await self._handle_engine_call(_req, do_sample, is_validate, **kwargs)
                content = output["text"]
                _retok_check(self.tokenizer, output, content)
                # Consumed (and cleared) by add_assistant_message so the turn is
                # appended with the tokens the model sampled, not a re-encoding.
                _req.last_sampled_ids = _sampled_ids_from(output)
                if _STRIP_QWEN_THINK_OUTPUT and isinstance(content, str) and ("<think>" in content or "</think>" in content):
                    content = _strip_think_blocks(content)
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1

                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.tokenizer, content)
                    conversation_histories[-1]["action_end"] = len(_req.input_ids)
                    break

                # ========== COLBENCH & TAU2GYM SPECIAL HANDLING: Direct text interaction (like sweet_rl) ==========
                # Check if this is ColBench or Tau2Gym environment - use direct text interaction without tool call parsing
                # BUT: For Tau2Gym, we still need to check for tool calls first, because tau2-bench supports tool calling
                is_colbench = False
                is_tau2gym = False
                dynamic_tool_map = _req.tools_kwargs.get("_dynamic_tool_map", None)

                # Check for Tau2Gym (has dynamic_tool_map)
                if dynamic_tool_map and "_tau2_env_kwargs" in _req.tools_kwargs:
                    is_tau2gym = True

                # Check for ColBench
                for tool_name in _req.tools_kwargs.keys():
                    if tool_name.startswith("_"):
                        continue  # Skip internal keys
                    tool = self._tool_map.get(tool_name)
                    if tool and hasattr(tool, '_conversation_data'):
                        create_kwargs = _req.tools_kwargs[tool_name].get("create_kwargs", {})
                        env_name = create_kwargs.get("env_name", "")
                        if env_name == "ColBenchCodeEnv":
                            is_colbench = True
                            break

                # For Tau2Gym, check if there are tool calls first
                # If tool calls are present, use the standard tool call execution path (below)
                # If no tool calls, use direct text interaction path
                if is_tau2gym and function_call_parser and function_call_parser.has_tool_call(content):
                    # Tau2Gym with tool calls: fall through to standard tool call parsing below
                    pass
                elif is_colbench or is_tau2gym:
                    # ColBench/Tau2Gym: Direct text interaction (like sweet_rl)
                    # Agent's response is pure text, directly pass to environment without tool call parsing
                    _req.add_assistant_message(self.tokenizer, content)
                    conversation_histories[-1]["action_end"] = len(_req.input_ids)
                    conversation_histories[-1]["content"] = content

                    obs_block_start = len(_req.input_ids)

                    if is_tau2gym:
                        # Tau2Gym: Use environment directly via env_manager
                        from verl.tools.env_manager import get_environment_manager
                        env_manager = get_environment_manager()
                        tau2_env = env_manager.get_environment(_req.request_id)

                        if tau2_env is None:
                            raise RuntimeError("Native tau3 environment was not initialized")
                        elif hasattr(tau2_env, 'episode_complete') and tau2_env.episode_complete:
                            resp = "Episode is complete"
                            reward, is_done, choice, content_param, metrics = 0.0, True, "action", content, {}
                        else:
                            try:
                                # Send plain text to tau2-bench environment
                                # tau2-bench's parse_action_string handles plain text natively
                                observation, step_reward, terminated, truncated, info = await tau2_env.step_async(content)

                                # Format response
                                if isinstance(observation, dict):
                                    resp = observation.get("feedback", str(observation))
                                else:
                                    resp = str(observation)

                                reward = float(step_reward)
                                is_done = terminated or truncated
                                choice = "action"
                                content_param = content
                                metrics = {
                                    "terminated": terminated,
                                    "truncated": truncated,
                                    "info": info,
                                }
                            except Exception as e:
                                raise RuntimeError("Native tau3 user/environment step failed") from e
                    else:
                        # ColBench: Use tool from tool_map
                        # Filter out internal keys (starting with _) - these are not tool names
                        tool_names = [k for k in _req.tools_kwargs.keys() if not k.startswith("_")]
                        resp = None
                        reward, is_done, choice, content_param, metrics = 0.0, False, "action", content, {}

                        if not tool_names:
                            logger.error(f"[COLBENCH_ERROR] Request {request_short_id}: No tool found in tools_kwargs (only internal keys)")
                            resp = "Error: No tool configuration found"
                            reward, is_done, choice, content_param, metrics = 0.0, True, "action", content, {}
                        else:
                            tool_name = tool_names[0]  # ColBench uses single tool
                            tool = self._tool_map.get(tool_name)
                            if tool is None:
                                logger.error(f"[COLBENCH_ERROR] Request {request_short_id}: Tool '{tool_name}' not found in tool_map")
                                resp = f"Error: Tool '{tool_name}' not found"
                                reward, is_done, choice, content_param, metrics = 0.0, True, "action", content, {}
                            else:
                                # For ColBench, pass content directly as action (no tool call parameters)
                                # The execute method will handle it specially for ColBench
                                try:
                                    resp, reward, is_done, choice, content_param, metrics = await tool.execute(
                                        _req.request_id,
                                        {"choice": "action", "content": content},  # Direct text content
                                        current_turns,
                                        **_req.tools_kwargs[tool_name].get("execute_kwargs", {}),
                                    )
                                except Exception as e:
                                    logger.error(f"[COLBENCH_ERROR] Request {request_short_id}: Tool execution failed: {e}")
                                    resp = f"Error: {str(e)}"
                                    reward, is_done, choice, content_param, metrics = 0.0, False, "action", content, {}

                        # Skip to response handling if tool execution failed
                        if resp is not None and "Error:" in resp:
                            # Add error response to conversation
                            _req.add_tool_response_messages(self.tokenizer, [resp])
                            obs_block_end = len(_req.input_ids)
                            conversation_histories[-1]["choice"] = choice
                            conversation_histories[-1]["reward"] = reward
                            conversation_histories[-1]["env_feedback"] = resp
                            conversation_histories[-1]["obs_start"] = obs_block_start
                            conversation_histories[-1]["obs_end"] = obs_block_end
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            break

                    # Add tool response to conversation
                    _req.add_tool_response_messages(self.tokenizer, [resp])
                    obs_block_end = len(_req.input_ids)

                    # Update conversation history
                    conversation_histories[-1]["choice"] = choice
                    conversation_histories[-1]["reward"] = reward
                    conversation_histories[-1]["env_feedback"] = resp

                    # Record observation boundaries
                    obs_char_start = metrics.get("obs_char_start", None)
                    obs_char_end = metrics.get("obs_char_end", None)
                    pure_obs = metrics.get("pure_observation", None)

                    if obs_char_start is not None and obs_char_end is not None and pure_obs:
                        added_tokens = _req.input_ids[obs_block_start:obs_block_end]
                        added_text = self.tokenizer.decode(added_tokens, skip_special_tokens=False)
                        resp_start_in_added = added_text.find(resp)
                        if resp_start_in_added != -1:
                            obs_char_start_abs = resp_start_in_added + obs_char_start
                            obs_char_end_abs = resp_start_in_added + obs_char_end
                            prefix_text = added_text[:obs_char_start_abs]
                            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                            prefix_plus_obs_text = added_text[:obs_char_end_abs]
                            prefix_plus_obs_tokens = self.tokenizer.encode(prefix_plus_obs_text, add_special_tokens=False)
                            conversation_histories[-1]["obs_start"] = obs_block_start + len(prefix_tokens)
                            conversation_histories[-1]["obs_end"] = obs_block_start + len(prefix_plus_obs_tokens)
                        else:
                            conversation_histories[-1]["obs_start"] = obs_block_start
                            conversation_histories[-1]["obs_end"] = obs_block_end
                    else:
                        conversation_histories[-1]["obs_start"] = obs_block_start
                        conversation_histories[-1]["obs_end"] = obs_block_end

                    # Check if episode is done
                    if is_done:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break

                    # Continue to next turn (don't break)
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                    continue
                # ========== END COLBENCH SPECIAL HANDLING ==========

                if function_call_parser and function_call_parser.has_tool_call(content):
                    finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                    _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING

                    try:
                        normed_content, tool_calls = function_call_parser.parse_non_stream(content)

                        if self._tool_call_repair and len(tool_calls) == 0 and "<tool_call>" in content and "</tool_call>" in content:
                            tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
                            matches = tool_call_pattern.findall(content)
                            dynamic_tool_map = _req.tools_kwargs.get("_dynamic_tool_map", None)
                            tool_map_for_indices = dynamic_tool_map if dynamic_tool_map else self._tool_map
                            tool_indices = _build_tool_indices(function_call_parser, tool_map_for_indices)
                            manual_tool_calls = _parse_tool_calls_json_fallback(content, tool_indices)

                            if len(manual_tool_calls) == 0:
                                for match in matches:
                                    try:
                                        # Use tau2's parse_action_string to parse the tool call JSON when available.
                                        from tau2.utils.tools import parse_action_string
                                        tool_call_content = match.strip()
                                        parsed_message = parse_action_string(tool_call_content, requestor="assistant")

                                        if parsed_message.tool_calls and len(parsed_message.tool_calls) > 0:
                                            for tau2_tool_call in parsed_message.tool_calls:
                                                manual_tool_calls.append(
                                                    ManualParsedToolCall(
                                                        name=tau2_tool_call.name,
                                                        arguments=tau2_tool_call.arguments,
                                                        tool_index=tool_indices.get(tau2_tool_call.name, len(manual_tool_calls)),
                                                    )
                                                )
                                        else:
                                            logger.warning(
                                                f"[TOOL_CALL_TAU2_PARSE_WARNING] Request {_req.request_id[:8]}: "
                                                f"tau2 parse_action_string returned no tool_calls for content: {tool_call_content[:200]}..."
                                            )
                                    except ModuleNotFoundError:
                                        logger.debug(
                                            f"[TOOL_CALL_TAU2_PARSE_SKIPPED] Request {_req.request_id[:8]}: tau2 is not installed"
                                        )
                                        break
                                    except Exception as te:
                                        logger.warning(
                                            f"[TOOL_CALL_TAU2_PARSE_ERROR] Request {_req.request_id[:8]}: "
                                            f"tau2 parse_action_string failed: {type(te).__name__}: {te}, "
                                            f"Content: {match[:200]}..."
                                        )
                                        import traceback
                                        traceback.print_exc()

                            if len(manual_tool_calls) > 0:
                                tool_calls = manual_tool_calls
                                # Remove tool_call tags from normed_content
                                normed_content = tool_call_pattern.sub("", content).strip()
                            else:
                                logger.warning(
                                    f"[TOOL_CALL_TAU2_PARSE_FAILED] Request {_req.request_id[:8]}: "
                                    f"fallback parsing failed to extract any tool calls from {len(matches)} <tool_call> tags"
                                )
                    except JSONDecodeError as e:
                        dynamic_tool_map = _req.tools_kwargs.get("_dynamic_tool_map", None)
                        tool_map_for_indices = dynamic_tool_map if dynamic_tool_map else self._tool_map
                        tool_indices = _build_tool_indices(function_call_parser, tool_map_for_indices)
                        manual_tool_calls = _parse_tool_calls_json_fallback(content, tool_indices) if self._tool_call_repair else []
                        if len(manual_tool_calls) > 0:
                            tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
                            normed_content = tool_call_pattern.sub("", content).strip()
                            tool_calls = manual_tool_calls
                            logger.warning(
                                f"[TOOL_CALL_PARSE_REPAIRED] Request {_req.request_id[:8]}: "
                                f"recovered {len(tool_calls)} tool call(s) after JSONDecodeError: {e}"
                            )
                            if self._format_repair_penalty > 0:
                                self._format_repair_pending.add(_req.request_id)
                        else:
                            logger.warning(
                                f"[TOOL_CALL_PARSE_ERROR] Request {_req.request_id[:8]}: JSONDecodeError: {e}, "
                                f"Content (last 500 chars): {content[-500:]}"
                            )
                            normed_content = content
                            tool_calls = []
                    except AttributeError as e:
                        logger.warning(
                            f"[TOOL_CALL_PARSE_ERROR] Request {_req.request_id[:8]}: AttributeError: {e}, "
                            f"Content (last 500 chars): {content[-500:]}"
                        )
                        normed_content = content
                        tool_calls = []
                    except Exception as e:
                        logger.warning(
                            f"[TOOL_CALL_PARSE_ERROR] Request {_req.request_id[:8]}: Unexpected error: {type(e).__name__}: {e}, "
                            f"Content (last 500 chars): {content[-500:]}"
                        )
                        import traceback
                        traceback.print_exc()
                        normed_content = content
                        tool_calls = []

                    # Convert to parsed tool calls
                    parsed_tool_calls = []
                    decode_error_count = 0
                    for tool_call in tool_calls:
                        # Convert parameters to JSON string if it's a dict (for ManualToolCall from tau2 parse)
                        if isinstance(tool_call.parameters, dict):
                            parameters_str = json.dumps(tool_call.parameters, ensure_ascii=False)
                        else:
                            # Already a string (from SGLang parser)
                            parameters_str = tool_call.parameters

                        function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                            OpenAIFunctionParsedSchema(
                                name=tool_call.name,
                                arguments=parameters_str,
                            )
                        )
                        # Drop the tool call if its arguments has decode error
                        if has_decode_error:
                            decode_error_count += 1
                            logger.warning(
                                f"[TOOL_CALL_DECODE_ERROR] Request {_req.request_id[:8]}: "
                                f"Tool '{tool_call.name}' decode failed, skipping"
                            )
                            continue
                        parsed_tool_calls.append(
                            OpenAIFunctionToolCall(
                                id=str(tool_call.tool_index),
                                function=function,
                            )
                        )

                    if len(parsed_tool_calls) > 0:
                        _req.add_assistant_message(self.tokenizer, normed_content, tool_calls=parsed_tool_calls)
                        conversation_histories[-1]["action_end"] = len(_req.input_ids)
                        conversation_histories[-1]["content"] = normed_content
                        # CRITICAL: Continue to next iteration to execute tool calls
                        # The next iteration will detect TOOL_CALLING state and execute the tool calls
                        continue
                    else:
                        # Parsing failed - check if this is Tau2Gym and should fallback to direct text interaction
                        dynamic_tool_map = _req.tools_kwargs.get("_dynamic_tool_map", None)
                        is_tau2gym = dynamic_tool_map and "_tau2_env_kwargs" in _req.tools_kwargs

                        if is_tau2gym:
                            # For Tau2Gym, if tool call parsing fails, fallback to direct text interaction
                            # tau2-bench's parse_action_string can handle mixed format (text + tool call)
                            logger.warning(
                                f"[TOOL_CALL_FALLBACK] Request {_req.request_id[:8]}: "
                                f"Found tool call markers but parsing failed (raw_tool_calls={len(tool_calls)}, "
                                f"decode_errors={decode_error_count}). "
                                f"Falling back to Tau2Gym direct text interaction. "
                                f"Content preview: {content[:200]}..."
                            )
                            # Fallback to Tau2Gym direct text interaction path
                            _req.add_assistant_message(self.tokenizer, content)
                            conversation_histories[-1]["action_end"] = len(_req.input_ids)
                            conversation_histories[-1]["content"] = content

                            obs_block_start = len(_req.input_ids)

                            from verl.tools.env_manager import get_environment_manager
                            env_manager = get_environment_manager()
                            tau2_env = env_manager.get_environment(_req.request_id)

                            if tau2_env is None:
                                raise RuntimeError("Native tau3 environment was not initialized")
                            elif hasattr(tau2_env, 'episode_complete') and tau2_env.episode_complete:
                                # Episode already complete, don't try to step
                                logger.warning(
                                    f"[TAU2GYM_EPISODE_COMPLETE] Request {request_short_id}: "
                                    f"Episode is already complete, skipping step"
                                )
                                resp = "Episode is complete"
                                reward, is_done, choice, content_param, metrics = 0.0, True, "action", content, {}
                            else:
                                try:
                                    # Send content (may contain tool call text) to tau2-bench
                                    # tau2-bench's parse_action_string will try to parse it
                                    observation, step_reward, terminated, truncated, info = await tau2_env.step_async(content)

                                    if isinstance(observation, dict):
                                        resp = observation.get("feedback", str(observation))
                                    else:
                                        resp = str(observation)

                                    reward = float(step_reward)
                                    is_done = terminated or truncated
                                    choice = "action"
                                    content_param = content
                                    metrics = {
                                        "terminated": terminated,
                                        "truncated": truncated,
                                        "info": info,
                                    }
                                except Exception as e:
                                    raise RuntimeError("Native tau3 user/environment step failed") from e

                            # Add tool response to conversation
                            _req.add_tool_response_messages(self.tokenizer, [resp])
                            obs_block_end = len(_req.input_ids)

                            # Update conversation history
                            conversation_histories[-1]["choice"] = choice
                            conversation_histories[-1]["reward"] = reward
                            conversation_histories[-1]["env_feedback"] = resp
                            conversation_histories[-1]["obs_start"] = obs_block_start
                            conversation_histories[-1]["obs_end"] = obs_block_end

                            # Check if episode is done
                            if is_done:
                                finish_reason_type = FinishReasonTypeEnum.STOP
                                break

                            # Continue to next turn
                            _req.state = AsyncRolloutRequestStateEnum.RUNNING
                            continue
                        else:
                            # For non-Tau2Gym environments, treat as regular message
                            logger.warning(
                                f"[TOOL_CALL_FAILED] Request {_req.request_id[:8]}: "
                                f"Found tool call markers but parsing failed. "
                                f"Raw tool_calls={len(tool_calls)}, decode_errors={decode_error_count}, "
                                f"parsed_tool_calls={len(parsed_tool_calls)}. Treating as regular message. "
                                f"Content (last 500 chars): {content[-500:]}"
                            )
                            _req.add_assistant_message(self.tokenizer, content)
                            conversation_histories[-1]["action_end"] = len(_req.input_ids)
                            conversation_histories[-1]["content"] = content
                            feedback = (
                                "Format error: your previous response contained tool-call markers, "
                                "but the tool call could not be parsed. Call interact_with_env exactly once "
                                "with valid JSON arguments containing both choice and content. choice must be "
                                "one of action, search, or answer. content must be a string. For answer, put "
                                "the internal tool call JSON as that string."
                            )
                            if self._format_fatal_penalty > 0:
                                # Format cost: unparseable tool call terminates the
                                # episode; make the protocol failure itself costly
                                # instead of silently zero-rewarding it.
                                conversation_histories[-1]["reward"] = -self._format_fatal_penalty
                                conversation_histories[-1]["env_feedback"] = feedback
                                self._fmt_stats["penalty_sum"] += self._format_fatal_penalty
                                self._fmt_stats["shaped_reward_sum"] += -self._format_fatal_penalty
                            self._fmt_stats["fatal"] += 1
                            if self._format_error_feature_value != 0.0:
                                conversation_histories[-1]["format_error_feature"] = self._format_error_feature_value
                                conversation_histories[-1].setdefault("env_feedback", feedback)
                            if (
                                self._format_error_feedback
                                and current_turns < self.config.multi_turn.max_turns
                                and getattr(_req, "_format_error_feedback_n", 0) < _FORMAT_FEEDBACK_MAX
                            ):
                                setattr(_req, "_format_error_feedback_n",
                                        getattr(_req, "_format_error_feedback_n", 0) + 1)
                                obs_block_start = len(_req.input_ids)
                                _req.add_tool_response_messages(self.tokenizer, [feedback])
                                conversation_histories[-1]["env_feedback"] = feedback
                                conversation_histories[-1]["obs_start"] = obs_block_start
                                conversation_histories[-1]["obs_end"] = len(_req.input_ids)
                                _req.state = AsyncRolloutRequestStateEnum.RUNNING
                                continue
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                else:
                    # No tool call; the feedback will come via tool manager. Record a placeholder obs window
                    obs_start_pos = len(_req.input_ids)
                    _req.add_assistant_message(self.tokenizer, content)
                    conversation_histories[-1]["action_end"] = len(_req.input_ids)
                    conversation_histories[-1]["content"] = content
                    conversation_histories[-1]["obs_start"] = obs_start_pos
                    conversation_histories[-1]["obs_end"] = len(_req.input_ids)
                    feedback = (
                        "Format error: your previous response did not call interact_with_env. "
                        "Call interact_with_env exactly once with valid JSON arguments containing both "
                        "choice and content. choice must be one of action, search, or answer. content "
                        "must be a string. For answer, put the internal tool call JSON as that string."
                    )
                    if self._format_no_tool_call_penalty > 0:
                        # Format cost: plain-text / self-made <action> response with
                        # no tool call at all also terminates the episode here —
                        # price it like a fatal protocol failure.
                        conversation_histories[-1]["reward"] = -self._format_no_tool_call_penalty
                        conversation_histories[-1]["env_feedback"] = feedback
                        self._fmt_stats["penalty_sum"] += self._format_no_tool_call_penalty
                        self._fmt_stats["shaped_reward_sum"] += -self._format_no_tool_call_penalty
                    self._fmt_stats["no_tool_call"] += 1
                    if self._format_error_feature_value != 0.0:
                        conversation_histories[-1]["format_error_feature"] = self._format_error_feature_value
                        conversation_histories[-1].setdefault("env_feedback", feedback)
                    if (
                        self._format_error_feedback
                        and current_turns < self.config.multi_turn.max_turns
                        and getattr(_req, "_format_error_feedback_n", 0) < _FORMAT_FEEDBACK_MAX
                    ):
                        setattr(_req, "_format_error_feedback_n",
                                getattr(_req, "_format_error_feedback_n", 0) + 1)
                        obs_block_start = len(_req.input_ids)
                        _req.add_tool_response_messages(self.tokenizer, [feedback])
                        conversation_histories[-1]["env_feedback"] = feedback
                        conversation_histories[-1]["obs_start"] = obs_block_start
                        conversation_histories[-1]["obs_end"] = len(_req.input_ids)
                        _req.state = AsyncRolloutRequestStateEnum.RUNNING
                        continue
                    break

        if current_turns >= self.config.multi_turn.max_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP


        # Calculate the reward for each tool
        # Check if using dynamic tools (Tau2Gym)
        dynamic_tool_map = _req.tools_kwargs.get("_dynamic_tool_map", None)
        tool_map_to_use = dynamic_tool_map if dynamic_tool_map else self._tool_map

        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs.get(name, {}).get("calc_reward_kwargs", {}))
            evaluation_metrics = {}
            if hasattr(tool, "get_evaluation_metrics"):
                evaluation_metrics = await tool.get_evaluation_metrics(_req.request_id)
            await tool.release(_req.request_id, **_req.tools_kwargs.get(name, {}).get("release_kwargs", {}))
            return name, reward, evaluation_metrics

        tool_reward_tasks = []

        # ColBench direct-text mode: compute final reward from env directly (no tool schema shown to model).
        if isinstance(_req.tools_kwargs, dict) and _req.tools_kwargs.get("_colbench_direct_text", False):
            try:
                from verl.tools.env_manager import get_environment_manager
                env_manager = get_environment_manager()
                env = env_manager.get_environment(_req.request_id)
                if env is not None and hasattr(env, "calculate_reward"):
                    final_reward = float(env.calculate_reward())
                    tool_reward_scores = {"interact_with_env": final_reward}
                else:
                    tool_reward_scores = {}
            except Exception as e:
                logger.error(f"[COLBENCH_REWARD_ERROR] Request {_req.request_id[:8]}: Failed to calculate ColBench reward: {e}")
                tool_reward_scores = {}
        # Native tau3 scoring is centralized in the wrapper: cutoff is zero,
        # ACTION/ENV is explicit, and service/evaluator errors abort the batch.
        elif dynamic_tool_map and "_tau2_env_kwargs" in _req.tools_kwargs:
            from verl.tools.env_manager import get_environment_manager
            tau2_env = get_environment_manager().get_environment(_req.request_id)
            if tau2_env is None:
                raise RuntimeError("Native tau3 environment disappeared before scoring")
            final_reward = await asyncio.to_thread(tau2_env.calculate_reward)
            tool_reward_scores = {"interact_with_env": final_reward}
        else:
            # Standard tool reward calculation
            # Filter out internal keys (starting with _) - these are not tool names
            tool_names = [name for name in _req.tools_kwargs.keys() if not name.startswith("_")]
            for name in tool_names:
                if name in tool_map_to_use:
                    tool = tool_map_to_use[name]
                    tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
            tool_reward_results = await asyncio.gather(*tool_reward_tasks) if tool_reward_tasks else []
            tool_reward_scores = {name: reward for name, reward, _ in tool_reward_results}
            for _, _, evaluation_metrics in tool_reward_results:
                tool_reward_scores.update(evaluation_metrics)

        # Keep the raw task/environment reward for metrics even when a later
        # reward transform replaces the training reward (for example, a probe).
        if "interact_with_env" in tool_reward_scores:
            tool_reward_scores = dict(tool_reward_scores)
            tool_reward_scores.setdefault(
                "interact_with_env_original",
                float(tool_reward_scores["interact_with_env"]),
            )

        # Only terminal-reward environments need their final score injected into history. Regular
        # UserGym environments already expose dense per-turn rewards; overwriting their last turn
        # with a generic max reward duplicates credit and disagrees with the paper's Sum metric.
        terminal_reward_env = bool(
            isinstance(_req.tools_kwargs, dict)
            and (_req.tools_kwargs.get("_colbench_direct_text", False) or dynamic_tool_map)
        )
        if terminal_reward_env and len(conversation_histories) > 0:
            # Get the final reward from tool_reward_scores (typically "interact_with_env" for ColBench)
            final_reward = 0.0
            for tool_name, reward in tool_reward_scores.items():
                if reward is not None:
                    final_reward = float(reward)
                    break  # Use first non-None reward

            # Update the last turn's reward in conversation_histories
            # This ensures Info-GRPO's compute_turn_credits can extract the correct trajectory_score
            conversation_histories[-1]["reward"] = final_reward

            # Also update reward_scores for logging (if it exists in conversation_histories)
            if "reward_scores" not in conversation_histories[-1]:
                conversation_histories[-1]["reward_scores"] = tool_reward_scores

        _req.finalize(self.tokenizer, tool_reward_scores, turn_boundaries, conversation_histories, finish_reason_type)

        # Verify state was set correctly
        if _req.state != AsyncRolloutRequestStateEnum.COMPLETED:
            logger.error(f"[FINALIZE ERROR] Request {_req.request_id}: State is {_req.state} after finalize, expected COMPLETED!")
        else:
            logger.debug(f"[FINALIZE DEBUG] Request {_req.request_id}: Successfully set to COMPLETED state")

        return _req

    async def _handle_engine_call(self, _req: AsyncRolloutRequest, do_sample: bool, is_validate: bool, override_n: bool = True, **kwargs) -> dict:
        generation_prompt_ids = _req.get_generation_prompt_ids(self.tokenizer)
        # Adjust max_new_tokens to ensure it is not greater than max_model_len - 1
        # SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra token accounts for the EOS token).
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(generation_prompt_ids) - 1)

        # Ensure max_new_tokens is at least 1
        max_new_tokens = max(max_new_tokens, 1)

        # Apply response_length_one_turn limit if configured (for multi-turn conversations)
        # This limits each turn's generation length, while allowing total response to accumulate
        if self.config.get("response_length_one_turn", None):
            max_new_tokens = min(self.config.response_length_one_turn, max_new_tokens)

        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=max_new_tokens,  # Use the computed max_new_tokens instead of self.config.response_length
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # In training, use configurable temperature (default 0.6 for Qwen3, fallback to 1.0)
            # For Qwen3 models, lower temperature (0.6) helps reduce thinking tokens
            training_temperature = self.config.get("training_temperature", None)
            if training_temperature is None:
                # Auto-detect Qwen3 and use 0.6, otherwise use 1.0
                if hasattr(self.tokenizer, 'name_or_path') and 'qwen3' in self.tokenizer.name_or_path.lower():
                    training_temperature = 0.6
                elif hasattr(self.tokenizer, 'model_name') and 'qwen3' in str(self.tokenizer.model_name).lower():
                    training_temperature = 0.6
                else:
                    training_temperature = 1.0
            kwargs = {
                "top_k": -1,
                "top_p": 1,
                "temperature": training_temperature,
                "n": 1,
            }
        kwargs["max_new_tokens"] = max_new_tokens
        if "n" not in kwargs or (kwargs["n"] > 1 and override_n):  # group size is supported in preprocess
            kwargs["n"] = 1
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = await self._engine.async_generate(
                input_ids=generation_prompt_ids,
                sampling_params=self.sampling_params,
                return_logprob=(_RETOK_PROBE or _USE_SAMPLED_IDS),
            )
        return output

    async def _handle_pending_state(self, _req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        # ===================== ColBench prompt alignment (TRAIN == EVAL) =====================
        # We intentionally hide tool schemas from the MODEL for ColBench (pure-text like eval),
        # but we still need to create the environment before stepping.
        if isinstance(_req.tools_kwargs, dict) and _req.tools_kwargs.get("_colbench_direct_text", False):
            try:
                from verl.tools.env_manager import get_environment_manager
                env_manager = get_environment_manager()
                if env_manager.get_environment(_req.request_id) is None:
                    # Extract create_kwargs from tools_kwargs (preserved even when tool_schemas=None)
                    create_kwargs = {}
                    if "interact_with_env" in _req.tools_kwargs:
                        tool_kwargs = _req.tools_kwargs.get("interact_with_env", {})
                        if isinstance(tool_kwargs, dict) and "create_kwargs" in tool_kwargs:
                            create_kwargs = dict(tool_kwargs["create_kwargs"])

                    # Avoid passing env_name twice (positional + keyword) for backward-compatible datasets.
                    # Many parquet entries include create_kwargs["env_name"]="ColBenchCodeEnv".
                    create_kwargs.pop("env_name", None)
                    # Ensure required fields are set (use config defaults if missing)
                    create_kwargs["max_turns"] = create_kwargs.get("max_turns", self.config["multi_turn"]["max_turns"])
                    create_kwargs["model_name"] = create_kwargs.get("model_name", self.config["multi_turn"]["model_name"])

                    # Create environment
                    env_manager.create_environment(_req.request_id, "ColBenchCodeEnv", **create_kwargs)

                    # CRITICAL: Also initialize InteractTool's conversation_data (needed for tool.execute() check)
                    # Even though we're in "direct text" mode, tool.execute() still checks _conversation_data
                    if "interact_with_env" in self._tool_map:
                        interact_tool = self._tool_map["interact_with_env"]
                        if hasattr(interact_tool, '_conversation_data'):
                            if _req.request_id not in interact_tool._conversation_data:
                                # Initialize conversation state (matching InteractTool.create() behavior)
                                interact_tool._conversation_data[_req.request_id] = {
                                    "history": [],
                                    "reward": 0.0,
                                    "ground_truth": create_kwargs.get("ground_truth"),
                                    "env_name": "ColBenchCodeEnv",
                                }

                    logger.debug(f"[COLBENCH_CREATE] Request {_req.request_id[:8]}: Created ColBench env and initialized conversation state")
            except Exception as e:
                logger.error(f"[COLBENCH_CREATE_ERROR] Request {_req.request_id[:8]}: Failed to create ColBench env: {e}")
                import traceback
                traceback.print_exc()
        # =================== end ColBench prompt alignment ===================

        if _req.tool_schemas is not None:
            # Check if using dynamic tools (Tau2Gym)
            dynamic_tool_map = _req.tools_kwargs.get("_dynamic_tool_map", None)
            tool_map_to_use = dynamic_tool_map if dynamic_tool_map else self._tool_map

            # For Tau2Gym with dynamic tools, create environment instead of individual tools
            if dynamic_tool_map and "_tau2_env_kwargs" in _req.tools_kwargs:
                from verl.tools.env_manager import get_environment_manager
                env_manager = get_environment_manager()
                _tau2_env_kwargs = _req.tools_kwargs["_tau2_env_kwargs"].copy()
                _tau2_env_kwargs["max_turns"] = self.config["multi_turn"]["max_turns"]
                _tau2_env_kwargs["model_name"] = self.config["multi_turn"]["model_name"]
                # Remove env_name from kwargs if present to avoid conflict with positional argument
                _tau2_env_kwargs.pop("env_name", None)
                # Create environment (only once for all tools)
                if _req.request_id not in env_manager._environments:
                    initial_obs = env_manager.create_environment(_req.request_id, "Tau2Gym", **_tau2_env_kwargs)
                    # Tau2Gym's create_environment returns initial observation as string
                    # Add it as user message (the initial user message from tau2-bench)
                    if not isinstance(initial_obs, str) or not initial_obs:
                        raise RuntimeError("Native tau3 environment returned no initial user observation")
                    # The dataset contains only public policy. Rebuild the
                    # complete prompt once the real user speaks, before any
                    # response tokens exist. Appending to an existing assistant
                    # generation prefix would corrupt the first user role/text.
                    values = {name: getattr(_req, name) for name in type(_req).model_fields}
                    values["messages"] = [message.model_dump() for message in _req.messages] + [
                        {"role": "user", "content": initial_obs}
                    ]
                    values["input_ids"], values["attention_mask"] = [], []
                    rebuilt = type(_req)(tokenizer=self.tokenizer, **values)
                    if len(rebuilt.prompt_ids) > rebuilt.max_prompt_len or len(rebuilt.prompt_ids) >= rebuilt.max_model_len:
                        raise ValueError("Native policy, tools, and initial user observation exceed the prompt/context budget")
                    for name in type(_req).model_fields:
                        setattr(_req, name, getattr(rebuilt, name))
                    logger.info("Native tau3 prompt initialized with its public user observation")
                # No individual tool creation needed for Tau2Gym - tools are executed directly
            else:
                # Standard tool creation (non-Tau2Gym)
                tool_creation_coroutines = []
                for tool_schema in _req.tool_schemas:
                    # Handle both dict and object formats
                    if hasattr(tool_schema, 'function'):
                        tool_name = tool_schema.function.name
                    elif isinstance(tool_schema, dict):
                        tool_name = tool_schema.get("function", {}).get("name")
                    else:
                        continue

                    if tool_name not in tool_map_to_use:
                        continue
                    tool = tool_map_to_use[tool_name]

                    create_kwargs = _req.tools_kwargs.get(tool.name, {}).get("create_kwargs", {})
                    create_kwargs["max_turns"] = self.config["multi_turn"]["max_turns"]
                    create_kwargs["model_name"] = self.config["multi_turn"]["model_name"]
                    tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))

                # Execute all tool creation coroutines
                if tool_creation_coroutines:
                    creation_results = await asyncio.gather(*tool_creation_coroutines)
                else:
                    creation_results = []

                # Note: For Tau2Gym with dynamic tools, environment creation and initial observation
                # handling is done separately in the dynamic tool loading section above.
                # This section handles standard tool creation for non-Tau2Gym environments.
        return _req

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences_with_tools(self, prompts: DataProto, **kwargs) -> DataProto:
        logger.warning(
            "`generate_sequences_with_tools` is deprecated, please use `generate_sequences(...)`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._req_level_generate_sequences(prompts, **kwargs)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # Async rollout with tools support
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        sorted_output_req_list = None

        try:
            if self._tp_rank == 0:

                req_list = self._preprocess_prompt_to_async_rollout_requests(
                    prompts,
                    n=1 if is_validate else self.config.n,
                )

                sem = asyncio.Semaphore(128)
                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(
                    asyncio.gather(
                        *[
                            self._semaphore_wrapped_rollout(sem, req, do_sample, is_validate, **kwargs)
                            for req in req_list
                        ]
                    )
                )

                sorted_output_req_list = sorted(
                    output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset)
                )
                self._apply_satisfaction_probe_rewards_to_requests(sorted_output_req_list)
                self._log_format_stats()


        except Exception as e:
            logger.exception(f"[Rank {self._rank}] Rollout failed: {e}")
            sorted_output_req_list = None

        [sorted_output_req_list] = broadcast_pyobj(
            data=[sorted_output_req_list],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )

        # Check if rollout failed (sorted_output_req_list is None or empty)
        if sorted_output_req_list is None:
            raise RuntimeError("Distributed rollout failed; refusing an empty or fabricated training batch")

        if len(sorted_output_req_list) == 0:
            logger.error(f"[Rank {self._rank}] Rollout returned empty list: 0 requests completed")
            logger.error(f"[Rank {self._rank}] Input prompts batch_size: {prompts.batch.batch_size[0] if hasattr(prompts.batch, 'batch_size') else 'N/A'}")
            logger.error(f"[Rank {self._rank}] This may indicate all requests failed or timed out")
            # Return empty DataProto
            empty_batch = TensorDict(
                {
                    "prompts": torch.empty(0, 0, dtype=torch.long, device=tgt_device),
                    "responses": torch.empty(0, 0, dtype=torch.long, device=tgt_device),
                    "input_ids": torch.empty(0, 0, dtype=torch.long, device=tgt_device),
                    "attention_mask": torch.empty(0, 0, dtype=torch.long, device=tgt_device),
                    "position_ids": torch.empty(0, 0, dtype=torch.long, device=tgt_device),
                    "loss_mask": torch.empty(0, 0, dtype=torch.long, device=tgt_device),
                    "turn_boundaries": torch.empty(0, 0, dtype=torch.int, device=tgt_device),
                },
                batch_size=0,
            )
            return DataProto(
                batch=empty_batch,
                non_tensor_batch={
                    "messages": np.array([], dtype=object),
                    "conversation_histories": np.array([], dtype=object),
                    "reward_scores": np.array([], dtype=float),
                    "data_source": np.array([], dtype=object),
                    "reward_model": np.array([], dtype=object),
                    "extra_info": np.array([], dtype=object),
                },
            )

        # Construct the batch data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        conversation_histories = []  # Add conversation histories for new algorithm
        reward_scores = []
        turn_boundaries_list = []

        for req in sorted_output_req_list:
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), f"""Request {req.request_id} has different length of\x20
                {len(req.input_ids)=}, {len(req.attention_mask)=}, {len(req.position_ids)=}, {len(req.loss_mask)=}"""
            error_message_lines = [
                f"""Request {req.request_id} has input_ids length {len(req.input_ids)}
                    greater than max_model_len {self.config.max_model_len}""",
                f"Decoded input_ids: {self.tokenizer.decode(req.input_ids)}",
                f"Decoded prompt_ids: {self.tokenizer.decode(req.prompt_ids)}",
                f"Decoded response_ids: {self.tokenizer.decode(req.response_ids)}",
                f"Messages: {req.messages}",
                f"Max model length: {req.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            assert len(req.input_ids) <= self.config.max_model_len, error_message

            prompt_ids.append(torch.tensor(req.prompt_ids, dtype=torch.int, device=tgt_device))
            response_ids.append(torch.tensor(req.response_ids, dtype=torch.int, device=tgt_device))
            if len(req.response_ids) > self.config.response_length:
                logger.warning(
                    f"""{req.request_id=} has response_ids length {len(req.response_ids)}\x20
                    greater than max_response_len {self.config.response_length},\n{req=}"""
                )
            prompt_attention_mask.append(torch.tensor(req.prompt_attention_mask, dtype=torch.int, device=tgt_device))
            response_attention_mask.append(torch.tensor(req.response_attention_mask, dtype=torch.int, device=tgt_device))
            prompt_position_ids.append(torch.tensor(req.prompt_position_ids, dtype=torch.int, device=tgt_device))
            response_position_ids.append(torch.tensor(req.response_position_ids, dtype=torch.int, device=tgt_device))
            prompt_loss_mask.append(torch.tensor(req.prompt_loss_mask, dtype=torch.int, device=tgt_device))
            response_loss_mask.append(torch.tensor(req.response_loss_mask, dtype=torch.int, device=tgt_device))
            messages.append({"messages": req.messages})

            # Store conversation histories (will adjust boundaries after padding)
            conversation_histories.append(req.conversation_histories)
            reward_scores.append(req.reward_scores)

            # Convert turn boundaries to tensor format
            # CRITICAL: Must match response_ids final padded length for distributed compatibility
            # response_ids will be padded to config.response_length, so turn_boundaries must match
            prompt_length = len(req.prompt_ids)
            actual_response_length = len(req.input_ids) - prompt_length  # Actual response length before truncation

            # Use config.response_length to ensure cross-worker compatibility
            # All workers must produce same-sized tensors for torch.cat to work
            response_length_for_tensor = self.config.response_length
            turn_boundary_tensor = torch.zeros(response_length_for_tensor, dtype=torch.int, device=tgt_device)

            actual_turns_from_hist = len(req.conversation_histories) if hasattr(req, 'conversation_histories') and req.conversation_histories else 0
            original_turn_boundaries_count = len(req.turn_boundaries) if hasattr(req, 'turn_boundaries') and req.turn_boundaries else 0

            # CRITICAL FIX: Reconstruct turn_boundaries from conversation_histories if available
            # This ensures we capture ALL turns, even if req.turn_boundaries was truncated
            use_hist_reconstruction = False
            if hasattr(req, 'conversation_histories') and req.conversation_histories:
                conv_hist = req.conversation_histories
                # Unwrap if nested
                if isinstance(conv_hist, (list, np.ndarray)) and len(conv_hist) > 0:
                    if isinstance(conv_hist[0], (list, np.ndarray)) and len(conv_hist[0]) > 0:
                        if isinstance(conv_hist[0][0], dict):
                            conv_hist = conv_hist[0]
                    elif isinstance(conv_hist[0], dict):
                        pass  # Already correct format

                # Check if we can reconstruct from conversation_histories
                if isinstance(conv_hist, (list, np.ndarray)) and len(conv_hist) > 0:
                    if isinstance(conv_hist[0], dict) and "action_start" in conv_hist[0]:
                        use_hist_reconstruction = True
                        valid_boundaries_from_hist = 0
                        invalid_boundaries_from_hist = 0
                        missing_action_start = 0
                        boundary_positions = []
                        invalid_positions = []
                        missing_turns = []

                        for turn_idx, turn in enumerate(conv_hist):
                            if isinstance(turn, dict):
                                if "action_start" in turn and turn["action_start"] is not None:
                                    action_start = turn["action_start"]
                                    # Convert from input_ids space to response_ids space
                                    response_pos = action_start - prompt_length
                                    if 0 <= response_pos < response_length_for_tensor:
                                        turn_boundary_tensor[response_pos] = 1
                                        valid_boundaries_from_hist += 1
                                        boundary_positions.append((turn_idx, response_pos, action_start))
                                    else:
                                        invalid_boundaries_from_hist += 1
                                        invalid_positions.append((turn_idx, response_pos, action_start, prompt_length, actual_response_length))
                                else:
                                    missing_action_start += 1
                                    missing_turns.append(turn_idx)
                            else:
                                missing_action_start += 1
                                missing_turns.append(turn_idx)

                        # Ensure first turn always starts at position 0 ONLY if the first turn's action_start equals prompt_length
                        # This means the first turn actually starts at position 0 in response space
                        if (response_length_for_tensor > 0 and
                            turn_boundary_tensor[0] == 0 and
                            len(conv_hist) > 0 and
                            isinstance(conv_hist[0], dict) and
                            "action_start" in conv_hist[0] and
                            conv_hist[0]["action_start"] is not None):
                            first_action_start = conv_hist[0]["action_start"]
                            # Only add boundary at position 0 if first turn starts right after prompt
                            if first_action_start == prompt_length:
                                turn_boundary_tensor[0] = 1
                                # Only count if we didn't already set it
                                if not any(pos[1] == 0 for pos in boundary_positions):
                                    valid_boundaries_from_hist += 1

            if not use_hist_reconstruction:
                if hasattr(req, 'turn_boundaries') and req.turn_boundaries:
                    # Convert turn boundaries from input_ids space to response_ids space
                    valid_boundaries = 0
                    invalid_boundaries = 0
                    first_boundary_pos = req.turn_boundaries[0] if req.turn_boundaries else None
                    first_boundary_response_pos = None

                    for idx, boundary_pos in enumerate(req.turn_boundaries):
                        # Turn boundaries are positions in input_ids, convert to response_ids positions
                        response_pos = boundary_pos - prompt_length
                        if 0 <= response_pos < response_length_for_tensor:
                            turn_boundary_tensor[response_pos] = 1
                            valid_boundaries += 1
                            if idx == 0:
                                first_boundary_response_pos = response_pos
                        else:
                            invalid_boundaries += 1
                            if idx == 0:
                                first_boundary_response_pos = response_pos  # Record even if invalid

                    truncated_response_length = len(req.response_ids)  # May be truncated to max_response_len
                    if invalid_boundaries > 0 or valid_boundaries != actual_turns_from_hist:
                        logger.warning(
                            f"[TURN_BOUNDARY_DEBUG] Request {req.request_id}: "
                            f"original_boundaries={original_turn_boundaries_count}, "
                            f"valid_in_response={valid_boundaries}, invalid={invalid_boundaries}, "
                            f"actual_turns_from_hist={actual_turns_from_hist}, "
                            f"actual_response_length={actual_response_length}, "
                            f"truncated_response_length={truncated_response_length}, "
                            f"response_length_for_tensor={response_length_for_tensor}, "
                            f"prompt_length={prompt_length}, "
                            f"input_ids_length={len(req.input_ids) if hasattr(req, 'input_ids') else 0}, "
                            f"first_boundary_pos_in_input={first_boundary_pos}, "
                            f"first_boundary_pos_in_response={first_boundary_response_pos}, "
                            f"position_0_has_boundary={turn_boundary_tensor[0].item() == 1}, "
                            f"max_response_len={req.max_response_len if hasattr(req, 'max_response_len') else 'N/A'}, "
                            f"max_model_len={req.max_model_len if hasattr(req, 'max_model_len') else 'N/A'}"
                        )

                    # Only add boundary at position 0 if the first turn_boundary equals prompt_length
                    # This means the first turn actually starts at position 0 in response space
                    # We should NOT add a boundary at position 0 if the first turn starts later (e.g., after initial observation)
                    if (response_length_for_tensor > 0 and
                        turn_boundary_tensor[0] == 0 and
                        req.turn_boundaries and
                        len(req.turn_boundaries) > 0):
                        # Only add boundary at position 0 if first turn starts right after prompt
                        if first_boundary_pos == prompt_length:
                            turn_boundary_tensor[0] = 1
                            logger.warning(
                                f"[TURN_BOUNDARY_FIX] Request {req.request_id}: "
                                f"Added boundary at position 0 because first_boundary_pos={first_boundary_pos} == prompt_length={prompt_length}"
                            )
                elif response_length_for_tensor > 0:
                    # If no turn_boundaries recorded, assume single turn starting at position 0
                    turn_boundary_tensor[0] = 1

            final_turn_count_in_tensor = turn_boundary_tensor.sum().item()
            truncated_response_length = len(req.response_ids) if hasattr(req, 'response_ids') else 0
            if final_turn_count_in_tensor != actual_turns_from_hist:
                # Get detailed information about missing turns
                turn_boundary_positions = torch.where(turn_boundary_tensor == 1)[0].tolist()

                # Check conversation_histories for missing action_start or out-of-range positions
                missing_details = []
                if hasattr(req, 'conversation_histories') and req.conversation_histories:
                    conv_hist = req.conversation_histories
                    # Unwrap if nested
                    if isinstance(conv_hist, (list, np.ndarray)) and len(conv_hist) > 0:
                        if isinstance(conv_hist[0], (list, np.ndarray)) and len(conv_hist[0]) > 0:
                            if isinstance(conv_hist[0][0], dict):
                                conv_hist = conv_hist[0]
                        elif isinstance(conv_hist[0], dict):
                            pass

                    if isinstance(conv_hist, (list, np.ndarray)):
                        for turn_idx, turn in enumerate(conv_hist):
                            if isinstance(turn, dict):
                                action_start = turn.get("action_start")
                                action_end = turn.get("action_end")
                                if action_start is None:
                                    missing_details.append({
                                        "turn_idx": turn_idx,
                                        "action_start": None,
                                        "action_end": action_end,
                                        "has_action_start": False
                                    })
                                else:
                                    response_pos = action_start - prompt_length
                                    if response_pos < 0 or response_pos >= response_length_for_tensor:
                                        missing_details.append({
                                            "turn_idx": turn_idx,
                                            "action_start": action_start,
                                            "action_end": action_end,
                                            "response_pos": response_pos,
                                            "prompt_length": prompt_length,
                                            "out_of_range": True,
                                            "actual_response_length": actual_response_length
                                        })

            turn_boundaries_list.append(turn_boundary_tensor)

        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        if prompt_ids.shape[1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)

        # MEMORY OPTIMIZATION (with distributed compatibility):
        # - In single-worker mode: pad to actual batch max (saves memory)
        # - In multi-worker mode: must pad to config.response_length (ensures concatenation works)
        # This is because different workers may have different max lengths, breaking torch.cat()
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        actual_max_response_length = response_ids.shape[1]

        # Always pad to config.response_length for distributed compatibility
        # Without this, different workers return different shapes → torch.cat fails
        if response_ids.shape[1] < self.config.response_length:
            if actual_max_response_length < self.config.response_length:
                logger.info(f"[PADDING] Padding response from {actual_max_response_length} to {self.config.response_length} for cross-worker compatibility")
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)

        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        if prompt_attention_mask.shape[1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, self.config.prompt_length, 0, left_pad=True)

        # Pad attention_mask to match response_ids length (for distributed compatibility)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)

        prompt_position_ids = pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        if prompt_position_ids.shape[1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(prompt_position_ids, self.config.prompt_length, 0, left_pad=True)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=response_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(len(sorted_output_req_list), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_loss_mask.shape[1] < self.config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)

        # Pad loss_mask to match response_ids length (for distributed compatibility)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)

        # ---- B31：把长度**钉死**在配置值上（上面每一处都只处理「短了」，不处理「长了」）----
        # 本文件自己的注释已经写明「different workers may have different max lengths,
        # breaking torch.cat()」，但补丁只写了 `if < config` 的一半。多轮 rollout 里
        # 最后一轮的环境反馈会把 response 顶过 response_length，各 rank 顶过的幅度不同，
        # `DataProto.concat` 就炸（实测 9344 vs 9390，1152+8192=9344）。
        # 处置：超长就截断——prompt 左填充保右端、response 右填充保左端，
        # 并行张量同样处理；turn_boundaries 在下面会按 response_ids 的实际长度对齐，
        # 所以必须在它之前钉死。**每次截断都打印**，不做静默丢弃。
        _pl, _rl = self.config.prompt_length, self.config.response_length
        if prompt_ids.shape[1] > _pl:
            print(f"[len-clamp] prompt {prompt_ids.shape[1]} -> {_pl}", flush=True)
            prompt_ids = prompt_ids[:, -_pl:]
            prompt_attention_mask = prompt_attention_mask[:, -_pl:]
            prompt_position_ids = prompt_position_ids[:, -_pl:]
            prompt_loss_mask = prompt_loss_mask[:, -_pl:]
        if response_ids.shape[1] > _rl:
            _n = int((response_attention_mask.sum(dim=-1) > _rl).sum().item())
            print(f"[len-clamp] response {response_ids.shape[1]} -> {_rl} "
                  f"(超出 {response_ids.shape[1]-_rl}，本批 {_n}/{response_ids.shape[0]} 条真的被切到)", flush=True)
            response_ids = response_ids[:, :_rl]
            response_attention_mask = response_attention_mask[:, :_rl]
            response_loss_mask = response_loss_mask[:, :_rl]
        # response_position_ids 依赖上面两者，钉死之后必须重算
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=response_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(len(sorted_output_req_list), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
        assert prompt_ids.shape[1] == _pl and response_ids.shape[1] == _rl, (
            f"[len-clamp] 钉死失败: prompt={prompt_ids.shape[1]} (期望 {_pl}), "
            f"response={response_ids.shape[1]} (期望 {_rl})")
        # ---- B31 end ----

        # Pad turn boundaries to match response sequence length
        # CRITICAL: In multi-turn conversations, actual_response_length can be much larger than config.response_length
        # config.response_length is the max length per turn, not the total response length across all turns
        # We should NOT truncate turn_boundaries to config.response_length, as this would lose boundaries for later turns
        # Instead, we pad/truncate to match the actual response_ids length (which may be truncated to max_response_len)
        max_turn_boundary_length = max([tb.shape[0] for tb in turn_boundaries_list]) if turn_boundaries_list else 0
        # Use the actual response_ids length after padding (not config.response_length)
        actual_response_ids_length = response_ids.shape[1] if len(response_ids) > 0 else self.config.response_length
        target_length = max(max_turn_boundary_length, actual_response_ids_length)
        turn_boundaries = pad_sequence(turn_boundaries_list, batch_first=True, padding_value=0)
        if turn_boundaries.shape[1] < target_length:
            turn_boundaries = pad_sequence_to_length(turn_boundaries, target_length, 0)
        # Only truncate if turn_boundaries is longer than actual response_ids
        # After optimization (not forcing pad to config.response_length), this should rarely happen
        # This ensures turn_boundaries matches response_ids length
        if turn_boundaries.shape[1] > actual_response_ids_length:
            # This is expected when batch has varying sequence lengths
            # The longer sequences get truncated to match the padded response_ids length
            logger.info(
                f"[TURN_BOUNDARY_PADDING] Aligning turn_boundaries length from {turn_boundaries.shape[1]} "
                f"to response_ids length {actual_response_ids_length} (batch size: {len(turn_boundaries_list)})"
            )
            turn_boundaries = turn_boundaries[:, :actual_response_ids_length]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        # Adjust conversation_histories boundaries for left padding
        # prompt_ids may have been left-padded, need to adjust all absolute positions
        for batch_idx, (req, conv_hist) in enumerate(zip(sorted_output_req_list, conversation_histories)):
            original_prompt_len = len(req.prompt_ids)
            padded_prompt_len = prompt_ids.shape[1]
            padding_offset = padded_prompt_len - original_prompt_len

            if padding_offset > 0:
                # Adjust all boundaries by adding the padding offset
                for turn in conv_hist:
                    if "action_start" in turn and turn["action_start"] is not None:
                        turn["action_start"] += padding_offset
                    if "action_end" in turn and turn["action_end"] is not None:
                        turn["action_end"] += padding_offset
                    if "obs_start" in turn and turn["obs_start"] is not None:
                        turn["obs_start"] += padding_offset
                    if "obs_end" in turn and turn["obs_end"] is not None:
                        turn["obs_end"] += padding_offset



        # Construct the batch data
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "turn_boundaries": turn_boundaries,  # Add turn boundaries to batch
            },
            batch_size=len(sorted_output_req_list),
        )

        # free cache engine
        if self.config.free_cache_engine and self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        # Extract fields from input prompts to pass through
        # These fields are needed by reward_manager and advantage calculation
        if "data_source" in prompts.non_tensor_batch:
            input_data_sources = prompts.non_tensor_batch["data_source"]
            data_sources = []
            for req in sorted_output_req_list:
                batch_data_id = req.batch_data_id
                data_sources.append(input_data_sources[batch_data_id])
            data_sources = np.array(data_sources)
        else:
            # Default to empty string if not provided
            data_sources = np.array([""] * len(sorted_output_req_list))

        # Extract reward_model field (required by reward manager)
        if "reward_model" in prompts.non_tensor_batch:
            input_reward_models = prompts.non_tensor_batch["reward_model"]
            reward_models = []
            for req in sorted_output_req_list:
                batch_data_id = req.batch_data_id
                reward_models.append(input_reward_models[batch_data_id])
            reward_models = np.array(reward_models, dtype=object)
        else:
            # Default empty dict if not provided
            reward_models = np.array([{}] * len(sorted_output_req_list), dtype=object)

        # Extract extra_info field (optional, for logging)
        if "extra_info" in prompts.non_tensor_batch:
            input_extra_infos = prompts.non_tensor_batch["extra_info"]
            extra_infos = []
            for req in sorted_output_req_list:
                batch_data_id = req.batch_data_id
                extra_infos.append(input_extra_infos[batch_data_id])
            extra_infos = np.array(extra_infos, dtype=object)
        else:
            extra_infos = np.array([{}] * len(sorted_output_req_list), dtype=object)

        conv_hist_array = np.array([[x] for x in conversation_histories], dtype=object)

        # Clean any unpicklable objects from non_tensor_batch before returning
        # This ensures Ray serialization works correctly
        from verl.workers.rollout.sglang_rollout.utils import _clean_dict_recursively, _is_picklable

        cleaned_non_tensor_batch = {
            "messages": np.array(messages),
            "conversation_histories": conv_hist_array,
            "reward_scores": np.array(reward_scores),
            "data_source": data_sources,
            "reward_model": reward_models,
            "extra_info": extra_infos,
        }

        # Clean extra_info if it contains unpicklable objects
        if extra_infos is not None and len(extra_infos) > 0:
            cleaned_extra_infos = []
            for info in extra_infos:
                if isinstance(info, dict):
                    cleaned_info = _clean_dict_recursively(info)
                    cleaned_extra_infos.append(cleaned_info)
                elif _is_picklable(info):
                    cleaned_extra_infos.append(info)
                else:
                    # Skip unpicklable items
                    cleaned_extra_infos.append({})
            cleaned_non_tensor_batch["extra_info"] = np.array(cleaned_extra_infos, dtype=object)

        return DataProto(
            batch=batch,
            non_tensor_batch=cleaned_non_tensor_batch,
        )


    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int) -> list[AsyncRolloutRequest]:
        assert "raw_prompt" in prompts.non_tensor_batch, "need data.return_raw_chat=True, due to no official way do parse_messages"
        req_list = []
        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        for data_idx, raw_prompt in enumerate(raw_prompts):
            for rollout_offset in range(n):
                # Check if this is Tau2Gym with dynamic tool schemas
                extra_info = prompts.non_tensor_batch.get("extra_info", [])

                # Handle extra_info: it might be a numpy array of dicts
                # Use explicit None check and length check to avoid numpy array truth value ambiguity
                has_extra_info = extra_info is not None and hasattr(extra_info, '__len__') and len(extra_info) > 0
                if has_extra_info and data_idx < len(extra_info):
                    current_extra_info = extra_info[data_idx]

                    # Handle both dict and object types
                    if isinstance(current_extra_info, dict):
                        dynamic_tool_schemas_raw = current_extra_info.get("tool_schemas", None)
                        domain = current_extra_info.get("domain", None)
                    else:
                        # Try to access as object attributes
                        dynamic_tool_schemas_raw = getattr(current_extra_info, "tool_schemas", None)
                        domain = getattr(current_extra_info, "domain", None)

                    # Convert numpy array to list if needed
                    if dynamic_tool_schemas_raw is not None:
                        if isinstance(dynamic_tool_schemas_raw, np.ndarray):
                            # Convert numpy array to list, handling empty arrays
                            if dynamic_tool_schemas_raw.size > 0:
                                dynamic_tool_schemas = dynamic_tool_schemas_raw.tolist()
                            else:
                                dynamic_tool_schemas = []
                        elif isinstance(dynamic_tool_schemas_raw, (list, tuple)):
                            dynamic_tool_schemas = list(dynamic_tool_schemas_raw)
                        else:
                            dynamic_tool_schemas = [dynamic_tool_schemas_raw] if dynamic_tool_schemas_raw else []
                    else:
                        dynamic_tool_schemas = None

                    # Check conditions for dynamic tool loading
                    has_tool_schemas = dynamic_tool_schemas is not None and len(dynamic_tool_schemas) > 0
                    has_domain = domain is not None and domain != ""

                    # CRITICAL: If this is Tau2Gym data (has _tau2_env_kwargs) but domain is missing, try to recover from _tau2_env_kwargs
                    _tools_kwargs_check = prompts.non_tensor_batch.get("tools_kwargs", [{}])[data_idx] if data_idx < len(prompts.non_tensor_batch.get("tools_kwargs", [])) else {}
                    is_tau2gym_data = isinstance(_tools_kwargs_check, dict) and "_tau2_env_kwargs" in _tools_kwargs_check
                    if is_tau2gym_data and not has_domain:
                        _tau2_env_kwargs = _tools_kwargs_check.get("_tau2_env_kwargs", {})
                        domain = _tau2_env_kwargs.get("domain", None)
                        if domain:
                            has_domain = domain is not None and domain != ""
                        else:
                            logger.error(f"[TOOL_LOADING_ERROR] Tau2Gym data detected but domain is missing in both extra_info and _tau2_env_kwargs!")

                    # Tau2Gym: Use dynamic tool schemas from data instead of static tool_config
                    # For Tau2Gym, we can load tools from domain even if tool_schemas is missing
                    # Check if: (1) has tool_schemas and domain, OR (2) is Tau2Gym data and has domain
                    should_load_tau2_tools = False
                    if is_tau2gym_data and has_domain:
                        # Tau2Gym data: Load tools from domain (tool_schemas optional)
                        should_load_tau2_tools = True
                    elif dynamic_tool_schemas and len(dynamic_tool_schemas) > 0 and domain:
                        # Standard case: has tool_schemas and domain
                        should_load_tau2_tools = True

                    if not should_load_tau2_tools and is_tau2gym_data:
                        # Log why we're not loading Tau2Gym tools
                        logger.error(f"[TOOL_LOADING_ERROR] Tau2Gym data detected but not loading tools! is_tau2gym_data={is_tau2gym_data}, has_domain={has_domain}, domain={domain}, has_tool_schemas={has_tool_schemas}")

                    if should_load_tau2_tools:
                        # Dynamically create tools from schemas stored in data
                        # The tool_schemas are already in OpenAI format from data preprocessing
                        try:
                            from verl.tools.tau2_tool_manager import get_tools_for_domain
                            from verl.tools.tau2_tool_wrapper import convert_tau2_tool_to_userrl
                            from verl.tools.schemas import OpenAIFunctionToolSchema

                            solo_mode = current_extra_info.get("solo_mode", False) if isinstance(current_extra_info, dict) else getattr(current_extra_info, "solo_mode", False)
                            tau2_tools = get_tools_for_domain(domain, solo_mode=solo_mode)

                            # Create UserRL tools
                            _tool_map_dynamic = {}
                            _tool_schemas_dynamic = []

                            # Add all tau2 tools
                            for idx, tau2_tool in enumerate(tau2_tools):
                                try:
                                    userrl_tool = convert_tau2_tool_to_userrl(tau2_tool)
                                    _tool_map_dynamic[tau2_tool.name] = userrl_tool
                                    tool_schema = userrl_tool.get_openai_tool_schema()
                                    _tool_schemas_dynamic.append(tool_schema)
                                except Exception as e:
                                    logger.error(f"[TOOL_LOADING_ERROR] Failed to convert tool {tau2_tool.name}: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    raise

                            # Note: We don't add send_message tool because:
                            # 1. tau2-bench's parse_action_string() handles plain text natively
                            # 2. In evaluation, models generate plain text directly (not via a tool)
                            # 3. Plain text is sent via env.step() which calls parse_action_string()
                            # This matches the evaluation setup exactly.

                            # Convert schemas to OpenAIFunctionToolSchema objects for AsyncRolloutRequest
                            # AsyncRolloutRequest expects List[OpenAIFunctionToolSchema]
                            # But the initialize_request method will convert them to dicts for tokenizer
                            _tool_schemas = _tool_schemas_dynamic  # Keep as OpenAIFunctionToolSchema objects

                            _tools_kwargs = prompts.non_tensor_batch.get("tools_kwargs", [{}])[data_idx]

                            # Store domain info for tool execution
                            if "_tau2_env_kwargs" in _tools_kwargs:
                                _tau2_env_kwargs = _tools_kwargs["_tau2_env_kwargs"]
                                _tau2_env_kwargs["max_turns"] = self.config["multi_turn"]["max_turns"]

                            # Store dynamic tool_map in tools_kwargs so we can access it during execution
                            _input_ids = None
                            _attention_mask = None
                            _tools_kwargs["_dynamic_tool_map"] = _tool_map_dynamic

                        except Exception as e:
                            logger.error(f"[TOOL_LOADING_ERROR] CRITICAL ERROR loading dynamic tools for domain {domain}: {e}")
                            import traceback
                            traceback.print_exc()
                            # DO NOT fallback silently - raise the error to prevent silent failures
                            # This ensures we catch tool loading issues early
                            raise RuntimeError(f"Failed to load dynamic tools for domain {domain}: {e}") from e
                    elif self._tool_schemas:
                        # Standard tool loading (non-Tau2Gym)
                        _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                        # Filter out internal keys (starting with _) - these are not tool names
                        tool_names = [k for k in _tools_kwargs.keys() if not k.startswith("_")]
                        _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in tool_names if k in self._tool_map]
                        _input_ids = None
                        _attention_mask = None

                        # Pass max_turns to tool creation (environment will be created in tool.create())
                        for tool_name in tool_names:
                            if tool_name in _tools_kwargs and isinstance(_tools_kwargs[tool_name], dict):
                                if "create_kwargs" not in _tools_kwargs[tool_name]:
                                    _tools_kwargs[tool_name]["create_kwargs"] = {}
                                _tools_kwargs[tool_name]["create_kwargs"]["max_turns"] = self.config["multi_turn"]["max_turns"]
                    else:
                        # No tools
                        _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                        _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                        _tools_kwargs = {}
                        _tool_schemas = None
                elif self._tool_schemas:
                    # Standard tool loading (no extra_info or no dynamic schemas)
                    _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                    # Filter out internal keys (starting with _) - these are not tool names
                    tool_names = [k for k in _tools_kwargs.keys() if not k.startswith("_")]
                    _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in tool_names if k in self._tool_map]
                    _input_ids = None
                    _attention_mask = None

                    # Pass max_turns to tool creation (environment will be created in tool.create())
                    for tool_name in tool_names:
                        if tool_name in _tools_kwargs and isinstance(_tools_kwargs[tool_name], dict):
                            if "create_kwargs" not in _tools_kwargs[tool_name]:
                                _tools_kwargs[tool_name]["create_kwargs"] = {}
                            _tools_kwargs[tool_name]["create_kwargs"]["max_turns"] = self.config["multi_turn"]["max_turns"]

                    # Epoch-aware TravelGym search failure schedule: inject the
                    # failure interval for the current training epoch so a
                    # curriculum (e.g. failure injection in epoch 0, disabled
                    # afterwards) works within a single run. The schedule comes
                    # from rollout.multi_turn.search_failure_schedule.
                    sf_schedule = self.config["multi_turn"].get("search_failure_schedule", None)
                    if sf_schedule:
                        current_epoch = int(prompts.meta_info.get("current_epoch", 0))
                        epoch_idx = min(current_epoch, len(sf_schedule) - 1)
                        sf_interval = int(sf_schedule[epoch_idx])
                        for tool_name in tool_names:
                            if tool_name in _tools_kwargs and isinstance(_tools_kwargs[tool_name], dict):
                                _tools_kwargs[tool_name].setdefault("create_kwargs", {})["search_failure_interval"] = sf_interval
                        if data_idx == 0 and rollout_offset == 0:
                            logger.info(f"[SF_SCHEDULE] current_epoch={current_epoch} -> search_failure_interval={sf_interval}")
                else:
                    _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                    _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                    # CRITICAL: Even if no tool_schemas, we may still need tools_kwargs for ColBench env creation
                    # Try to preserve tools_kwargs from data if available (for ColBench direct-text mode)
                    if "tools_kwargs" in prompts.non_tensor_batch and data_idx < len(prompts.non_tensor_batch["tools_kwargs"]):
                        _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx] or {}
                    else:
                        _tools_kwargs = {}
                    _tool_schemas = None

                # ===================== ColBench prompt alignment (TRAIN == EVAL) =====================
                # ColBench evaluation uses pure-text chat (no tool schemas shown to the model).
                # For training, we may still need tools/env internally, but MUST keep the model prompt identical
                # to eval by NOT providing tool_schemas to AsyncRolloutRequest.
                # Additionally, replace the system prompt with the eval prompt (llm_agent_code_prompt.txt)
                # to ensure consistency.
                try:
                    if isinstance(_tools_kwargs, dict) and "interact_with_env" in _tools_kwargs:
                        create_kwargs = (_tools_kwargs.get("interact_with_env", {}) or {}).get("create_kwargs", {}) or {}
                        if create_kwargs.get("env_name", "") == "ColBenchCodeEnv":
                            _tools_kwargs["_colbench_direct_text"] = True
                            _tool_schemas = None

                            # Replace system message in raw_prompt with eval prompt (llm_agent_code_prompt.txt)
                            # This ensures training prompt matches evaluation prompt exactly
                            if isinstance(raw_prompt, (list, np.ndarray)) and len(raw_prompt) > 0:
                                from pathlib import Path
                                prompt_dir = Path(__file__).parent.parent.parent.parent.parent / "gyms" / "ColBenchGym" / "colbenchgym" / "prompts"
                                eval_prompt_path = prompt_dir / "llm_agent_code_prompt.txt"

                                if eval_prompt_path.exists():
                                    with open(eval_prompt_path, "r") as f:
                                        eval_system_prompt = f.read().strip()

                                    # Convert raw_prompt to list if it's numpy array
                                    messages_list = raw_prompt.tolist() if isinstance(raw_prompt, np.ndarray) else list(raw_prompt)

                                    # Replace system message (first message with role="system")
                                    # Keep user message and other messages unchanged
                                    new_messages = []
                                    system_replaced = False
                                    for msg in messages_list:
                                        if isinstance(msg, dict):
                                            msg_role = msg.get("role", "")
                                        else:
                                            # Handle Message objects or other types
                                            msg_role = getattr(msg, "role", "") if hasattr(msg, "role") else ""

                                        if msg_role == "system" and not system_replaced:
                                            # Replace with eval prompt (remove {dialogue_history} placeholder)
                                            eval_prompt_clean = eval_system_prompt.replace("{dialogue_history}", "").strip()
                                            new_messages.append({"role": "system", "content": eval_prompt_clean})
                                            system_replaced = True
                                        else:
                                            # Keep other messages unchanged (convert to dict if needed)
                                            if isinstance(msg, dict):
                                                new_messages.append(msg)
                                            else:
                                                # Convert Message object or other types to dict
                                                if hasattr(msg, "role") and hasattr(msg, "content"):
                                                    new_messages.append({"role": msg.role, "content": msg.content})
                                                else:
                                                    # Fallback: keep original
                                                    new_messages.append(msg)

                                    # If no system message found, prepend it
                                    if not system_replaced:
                                        eval_prompt_clean = eval_system_prompt.replace("{dialogue_history}", "").strip()
                                        new_messages.insert(0, {"role": "system", "content": eval_prompt_clean})

                                    # Update raw_prompt with new messages
                                    raw_prompt = new_messages
                                else:
                                    logger.warning(f"[COLBENCH_PROMPT] Eval prompt file not found: {eval_prompt_path}, using original prompt")
                except Exception as e:
                    logger.warning(f"[COLBENCH_PROMPT] Failed to replace ColBench system prompt: {e}")
                    import traceback
                    traceback.print_exc()
                # =================== end ColBench prompt alignment ===================

                # Convert raw_prompt to list if needed (after potential ColBench prompt replacement)
                if isinstance(raw_prompt, np.ndarray):
                    messages_for_request = raw_prompt.tolist()
                elif isinstance(raw_prompt, list):
                    messages_for_request = raw_prompt
                else:
                    messages_for_request = list(raw_prompt)

                req = AsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=rollout_offset,
                    request_id=str(uuid4()),
                    state=AsyncRolloutRequestStateEnum.PENDING,
                    messages=messages_for_request,
                    tool_schemas=_tool_schemas,
                    tools_kwargs=_tools_kwargs,
                    input_ids=_input_ids,
                    response_ids=[],
                    attention_mask=_attention_mask,
                    response_attention_mask=[],
                    response_position_ids=[],
                    response_loss_mask=[],
                    reward_scores={},
                    max_prompt_len=self.config.prompt_length,
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                    use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                    enable_tokenization_sanity_check=self.config.multi_turn.enable_tokenization_sanity_check,
                    tokenizer=self.tokenizer,
                )

                error_message = f"Request {req.request_id} has mismatched lengths: input_ids={len(req.input_ids)}, attention_mask={len(req.attention_mask)}, position_ids={len(req.position_ids)}, loss_mask={len(req.loss_mask)}"
                assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), error_message
                req_list.append(req)
        return req_list

    async def chat_completion(self, json_request):
        assert self._tp_rank == 0, "only called in tp rank 0"
        _input_ids = []
        _attention_mask = []
        _position_ids = []
        _tool_schemas = []
        _tools_kwargs = {}

        req = AsyncRolloutRequest(
            request_id=str(uuid4()),
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=[Message.model_validate(msg) for msg in json_request["messages"]],
            tools=_tool_schemas,
            tools_kwargs=_tools_kwargs,
            input_ids=_input_ids,
            prompt_ids=_input_ids,
            response_ids=[],
            attention_mask=_attention_mask,
            prompt_attention_mask=_attention_mask,
            response_attention_mask=[],
            position_ids=_position_ids,
            prompt_position_ids=_position_ids,
            response_position_ids=[],
            loss_mask=[0] * len(_input_ids),
            prompt_loss_mask=[0] * len(_input_ids),
            response_loss_mask=[],
            reward_scores={},
            max_response_len=self.config.response_length,
            max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
        )

        # json_request already contains sampling_params
        output = await self._handle_engine_call(req, True, False, False, **json_request)
        # it can be Dict or AsyncIterator[Dict]
        if isinstance(output, dict):
            outputs = [output]
        else:
            outputs = output

        # build openai chat completion format
        choices = []
        id = None
        for i, content in enumerate(outputs):
            choices.append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": content["text"],
                    },
                    "finish_reason": content["meta_info"]["finish_reason"]["type"],
                }
            )
            id = content["meta_info"]["id"]

        return {
            "id": "chatcmpl-" + id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": json_request.get("model", "sglang_model"),
            "choices": choices,
        }

        # this function is left for uniform train-inference resharding

    async def wake_up(self):
        if not self.is_sleep:
            return
        await self.sharding_manager.wake_up()  # pylint: disable=C2801
        self.is_sleep = False

    # this function is left for uniform train-inference resharding
    async def sleep(self):
        if self.is_sleep:
            return
        await self.sharding_manager.sleep()
        self.is_sleep = True
