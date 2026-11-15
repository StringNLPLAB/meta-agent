"""Regression checks against the actual prompt decision and engine-call code."""
import ast
import asyncio
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load('_nonthinking_gate_test', ROOT / 'verl/utils/nonthinking_policy.py')
budget = load('_nonthinking_budget_test', ROOT / 'verl/utils/sglang_request_budget.py')


class Config(dict):
    __getattr__ = dict.__getitem__


class NonthinkingTests(unittest.TestCase):
    def test_explicit_switch_survives_generic_local_tokenizer_paths(self):
        tree = ast.parse((ROOT / 'verl/workers/rollout/schemas.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {'_enable_thinking', '_is_qwen3_model'}]
        scope = dict(os=os, PreTrainedTokenizer=object)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'actual_mode_decision', 'exec'), scope)
        tokenizer = SimpleNamespace(name_or_path='/scratch/download/base')
        with patch.dict(os.environ, {'USERRL_ENABLE_QWEN3_THINKING': '0'}):
            self.assertFalse(scope['_enable_thinking'](tokenizer))
        with patch.dict(os.environ, {'USERRL_ENABLE_QWEN3_THINKING': '1'}):
            self.assertTrue(scope['_enable_thinking'](tokenizer))
        with patch.dict(os.environ, {'USERRL_ENABLE_QWEN3_THINKING': 'false'}):
            with self.assertRaises(ValueError):
                scope['_enable_thinking'](tokenizer)
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(scope['_enable_thinking'](tokenizer))
            tokenizer.name_or_path = 'Qwen3-4B'
            self.assertFalse(scope['_enable_thinking'](tokenizer))

    def engine_case(self, prompt, output):
        tree = ast.parse((ROOT / 'verl/workers/rollout/sglang_rollout/sglang_rollout_customized.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == '_handle_engine_call')
        scope = dict(AsyncRolloutRequest=object, generation_budget=budget.generation_budget,
            check_nonthinking_input=gate.check_input, check_nonthinking_output=gate.check_output,
            _RETOK_PROBE=False, _USE_SAMPLED_IDS=False)
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'actual_engine_call', 'exec'), scope)
        calls = []
        async def generate(**kwargs):
            calls.append(kwargs)
            return output
        owner = SimpleNamespace(config=Config(max_model_len=40960, response_length=32768,
            response_length_one_turn=1024, training_temperature=.6),
            tokenizer=SimpleNamespace(encode=lambda text, **kwargs: list(text.encode())),
            _engine=SimpleNamespace(scheduler_info={'max_req_input_len':40954}, async_generate=generate))
        @contextmanager
        def sampling(**kwargs):
            owner.sampling_params = kwargs
            yield
        owner.update_sampling_params = sampling
        request = SimpleNamespace(get_generation_prompt_ids=lambda tokenizer:list(prompt.encode()))
        async def execute():
            return await scope['_handle_engine_call'](owner, request, True, False)
        return execute, calls

    def test_actual_missing_prefix_blocks_generation(self):
        execute, calls = self.engine_case('assistant\n', {'text':'OK'})
        with patch.dict(os.environ, {'EAU_REQUIRE_NONTHINKING_POLICY':'1','USERRL_ENABLE_QWEN3_THINKING':'0'}):
            with self.assertRaises(RuntimeError):
                asyncio.run(execute())
        self.assertEqual(calls, [])

    def test_reasoning_fails_without_stripping_or_retrying(self):
        execute, calls = self.engine_case('assistant\n'+gate.EMPTY_THINKING_SUFFIX, {'text':'<think>reasoning</think>OK'})
        with patch.dict(os.environ, {'EAU_REQUIRE_NONTHINKING_POLICY':'1','USERRL_ENABLE_QWEN3_THINKING':'0'}):
            with self.assertRaises(RuntimeError):
                asyncio.run(execute())
        self.assertEqual(len(calls), 1)

    def test_valid_generation_retains_sampling_and_raw_completion(self):
        import tempfile
        output = {'text':'OK'}
        execute, calls = self.engine_case('assistant\n'+gate.EMPTY_THINKING_SUFFIX, output)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'EAU_REQUIRE_NONTHINKING_POLICY':'1','USERRL_ENABLE_QWEN3_THINKING':'0',
                'EAU_NONTHINKING_BEHAVIOR_DIR':directory}):
            gate._COUNTS.update(input_calls_checked=0, sampled_completions_checked=0)
            self.assertIs(asyncio.run(execute()), output)
            self.assertTrue(list(Path(directory).glob('*.json')))
        self.assertEqual(calls[0]['sampling_params']['temperature'], .6)
        self.assertEqual(calls[0]['sampling_params']['max_new_tokens'], 1024)
        self.assertEqual(len(calls), 1)

    def test_malformed_reasoning_is_also_rejected(self):
        with patch.dict(os.environ, {'EAU_REQUIRE_NONTHINKING_POLICY':'1'}):
            gate._COUNTS.update(input_calls_checked=10, sampled_completions_checked=0)
            for text in ['<think>unfinished', 'reasoning</think>', '<THINK>reasoning</THINK>']:
                with self.subTest(text=text), self.assertRaises(RuntimeError):
                    gate.check_output({'text':text})
            gate.check_output({'text':'<think>\n\n</think>\n\nOK'})


if __name__ == '__main__':
    unittest.main()
