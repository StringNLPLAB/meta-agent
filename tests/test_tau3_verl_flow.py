"""CPU regressions for native observations and complete public tool schemas.

The pending method is loaded verbatim from its AST to avoid importing CUDA
SGLang engines. Requests, tokenization, and schema validation use real classes.
These checks do not run model inference or distributed training.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verl/workers/rollout/sglang_rollout/sglang_rollout_customized.py"
HAS_DEPENDENCIES = all(importlib.util.find_spec(name) for name in (
    "torch", "tensordict", "ray", "omegaconf", "transformers", "tokenizers",
))


def load_pending_method():
    module = ast.parse(SOURCE.read_text())
    owner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SGLangRollout")
    method = next(node for node in owner.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_pending_state")
    wrapper = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
                         type_ignores=[])
    namespace = {"asyncio": asyncio, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(wrapper), str(SOURCE), "exec"), namespace)
    return namespace["_handle_pending_state"]


@unittest.skipUnless(HAS_DEPENDENCIES, "Install the verl CPU validation dependencies")
class NativePromptFlowTests(unittest.TestCase):
    @staticmethod
    def tokenizer():
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        words = ("[UNK]", "[PAD]", "[EOS]", "system", "user", "assistant", "POLICY",
                 "FIRST_USER_SENTINEL", "help", "me")
        backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        result = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        result.chat_template = (
            "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} [EOS] "
            "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        )
        return result

    def fixture(self, max_prompt=128):
        from omegaconf import OmegaConf
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum

        tokenizer = self.tokenizer()
        schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {"name": "lookup", "description": "A public tool", "parameters": {
                "type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"],
            }},
        })
        request = AsyncRolloutRequest(
            batch_data_id=0, rollout_offset=0, request_id="synthetic-native-prompt",
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=[{"role": "system", "content": "POLICY"}], tool_schemas=[schema],
            tools_kwargs={"_tau2_env_kwargs": {"domain": "retail", "task_id": "synthetic_train"},
                          "_dynamic_tool_map": {"lookup": object()}},
            input_ids=None, response_ids=[], attention_mask=None,
            response_attention_mask=[], response_position_ids=[], response_loss_mask=[], reward_scores={},
            max_prompt_len=max_prompt, max_response_len=128, max_model_len=256,
            use_inference_chat_template=False, enable_tokenization_sanity_check=True, tokenizer=tokenizer,
        )

        class Manager:
            def __init__(self):
                self._environments = {}
                self.creates = 0

            def create_environment(self, request_id, env_name, **kwargs):
                self.creates += 1
                self._environments[request_id] = object()
                return "FIRST_USER_SENTINEL help me"

        controller = SimpleNamespace(
            tokenizer=tokenizer, _tool_map={},
            config=OmegaConf.create({"multi_turn": {"max_turns": 2, "model_name": "openai/synthetic-user"}}),
        )
        return tokenizer, request, controller, Manager()

    def test_first_native_observation_is_complete_in_prompt_once(self):
        from verl.tools import env_manager

        tokenizer, request, controller, manager = self.fixture()
        pending = load_pending_method()
        with patch.object(env_manager, "get_environment_manager", return_value=manager):
            asyncio.run(pending(controller, request))
            expected_messages = [{"role": "system", "content": "POLICY"},
                                 {"role": "user", "content": "FIRST_USER_SENTINEL help me"}]
            expected = tokenizer.apply_chat_template(expected_messages, add_generation_prompt=True)
            self.assertEqual(request.input_ids, expected)
            self.assertEqual(request.prompt_ids, expected)
            self.assertEqual(request.get_generation_prompt_ids(tokenizer), expected)
            self.assertEqual([message.model_dump(exclude_none=True) for message in request.messages], expected_messages)
            self.assertEqual(request.response_ids, [])
            self.assertFalse(any(request.loss_mask))
            asyncio.run(pending(controller, request))
            self.assertEqual(manager.creates, 1)
            self.assertEqual(request.input_ids, expected)
            self.assertEqual(len(request.messages), 2)

    def test_complete_initial_observation_must_fit_the_prompt_budget(self):
        from verl.tools import env_manager

        _, request, controller, manager = self.fixture(max_prompt=8)
        with patch.object(env_manager, "get_environment_manager", return_value=manager):
            with self.assertRaisesRegex(ValueError, "prompt/context budget"):
                asyncio.run(load_pending_method()(controller, request))

    @unittest.skipUnless(os.environ.get("TAU3_BENCHMARK_ROOT"), "Set TAU3_BENCHMARK_ROOT for native schema loading")
    def test_native_nested_tool_definitions_survive_verl_conversion(self):
        from examples.tau3_minimal.tau_adapter import Tau3Adapter

        with patch.dict(os.environ):
            Tau3Adapter(benchmark_root=os.environ["TAU3_BENCHMARK_ROOT"], domain="retail", reward_mode="action",
                        user_model="openai/offline-fixture", user_base_url="http://127.0.0.1:1/v1", max_turns=2)
            from verl.tools.tau2_tool_manager import get_tools_for_domain
            from verl.tools.tau2_tool_wrapper import convert_tau2_tool_to_userrl

            tokenizer, request, _, _ = self.fixture()
            for domain in ("retail", "airline", "telecom"):
                tools = get_tools_for_domain(domain, solo_mode=False)
                self.assertEqual(sum(tool.name == "done" for tool in tools), 1)
                for tool in tools:
                    with self.subTest(domain=domain, tool=tool.name):
                        original = tool.openai_schema
                        converted = convert_tau2_tool_to_userrl(tool).get_openai_tool_schema()
                        values = {name: getattr(request, name) for name in type(request).model_fields}
                        values["tool_schemas"] = [converted]
                        rebuilt = type(request)(tokenizer=tokenizer, **values)
                        restored = rebuilt.tool_schemas[0].model_dump(exclude_none=True, exclude_unset=True)
                        original_parameters = original["function"]["parameters"]
                        restored_parameters = restored["function"]["parameters"]
                        # Optional empty `required` is normalized by the wrapper.
                        expected = dict(original_parameters)
                        if not expected.get("required"):
                            expected.pop("required", None)
                        if not restored_parameters.get("required"):
                            restored_parameters.pop("required", None)
                        self.assertEqual(restored_parameters, expected)


if __name__ == "__main__":
    unittest.main()
