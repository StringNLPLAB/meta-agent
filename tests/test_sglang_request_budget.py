"""CPU regression for the actual rollout method at the live scheduler boundary."""
import ast
import asyncio
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('request_budget', ROOT / 'verl/utils/sglang_request_budget.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)
spec = importlib.util.spec_from_file_location('nonthinking_policy', ROOT / 'verl/utils/nonthinking_policy.py')
nonthinking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nonthinking)


class Config(dict):
    __getattr__ = dict.__getitem__


class BudgetTests(unittest.TestCase):
    def test_limits_use_the_actual_scheduler_not_nominal_context(self):
        arguments = dict(context_length=40960, response_length=32768, scheduler_info={'max_req_input_len':40954}, turn_limit=1024)
        self.assertEqual(budget.generation_budget(input_length=40958, **arguments), 0)
        self.assertEqual(budget.generation_budget(input_length=40954, **arguments), 5)
        self.assertEqual(budget.generation_budget(input_length=30000, **arguments), 1024)
        with self.assertRaises(ValueError):
            budget.generation_budget(context_length=40960,input_length=1,response_length=5,scheduler_info={})

    def test_real_engine_call_never_submits_the_crashing_input_or_changes_valid_sampling(self):
        source=ast.parse((ROOT/'verl/workers/rollout/sglang_rollout/sglang_rollout_customized.py').read_text())
        node=next(n for n in ast.walk(source) if isinstance(n,ast.AsyncFunctionDef) and n.name=='_handle_engine_call')
        namespace=dict(generation_budget=budget.generation_budget,AsyncRolloutRequest=object,_RETOK_PROBE=False,_USE_SAMPLED_IDS=False,
            check_nonthinking_input=nonthinking.check_input,check_nonthinking_output=nonthinking.check_output)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_rollout_method','exec'),namespace)
        requests=[]
        async def generate(**kwargs):
            requests.append(kwargs);return {'text':'sampled output'}
        owner=SimpleNamespace(config=Config(max_model_len=40960,response_length=32768,response_length_one_turn=1024,training_temperature=.6),
            tokenizer=SimpleNamespace(name_or_path='Qwen3-4B'),_engine=SimpleNamespace(scheduler_info={'max_req_input_len':40954},async_generate=generate))
        @contextmanager
        def sampling(**kwargs):
            owner.sampling_params=kwargs;yield
        owner.update_sampling_params=sampling
        req=SimpleNamespace(get_generation_prompt_ids=lambda tokenizer:list(range(40958)))
        with self.assertRaises(ValueError):
            asyncio.run(namespace['_handle_engine_call'](owner,req,True,False))
        self.assertEqual(requests,[])
        req.get_generation_prompt_ids=lambda tokenizer:list(range(30000))
        self.assertEqual(asyncio.run(namespace['_handle_engine_call'](owner,req,True,False)),{'text':'sampled output'})
        self.assertEqual(requests[0]['sampling_params']['temperature'],.6)
        self.assertEqual(requests[0]['sampling_params']['max_new_tokens'],1024)
        self.assertEqual(len(requests[0]['input_ids']),30000)


if __name__=='__main__':unittest.main()
