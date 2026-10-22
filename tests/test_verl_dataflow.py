"""CPU migration checks using real verl data loading and optimization code.

The default cases use synthetic tasks, a locally constructed tokenizer, and
small tensors. They do not run a model, benchmark episode, Ray, or GPU trainer.
TAU3_BENCHMARK_ROOT additionally checks the official train task/data boundary.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    "torch", "tensordict", "torchdata", "ray", "datasets", "pyarrow",
    "omegaconf", "transformers", "tokenizers",
)
HAS_DEPENDENCIES = all(importlib.util.find_spec(name) for name in DEPENDENCIES)


@unittest.skipUnless(HAS_DEPENDENCIES, "Install the verl CPU validation dependencies")
class VerlDataFlowTests(unittest.TestCase):
    @staticmethod
    def tokenizer():
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "system": 3,
             "assistant": 4, "public": 5, "policy": 6}, unk_token="[UNK]",
        ))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
            "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        )
        return tokenizer

    def data_roundtrip(self, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch
        from omegaconf import OmegaConf
        from verl import DataProto
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        with tempfile.TemporaryDirectory(prefix="meta-verl-dataflow-") as directory:
            path = Path(directory) / "train.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            config = OmegaConf.create({
                "cache_dir": str(Path(directory) / "cache"), "max_prompt_length": 32768,
                "return_raw_chat": True, "filter_overlong_prompts": True,
                "filter_overlong_prompts_workers": 1, "need_tools_kwargs": True,
                "truncation": "error",
            })
            dataset = RLHFDataset(str(path), self.tokenizer(), config)
            self.assertEqual(len(dataset), len(rows))
            batch = DataProto.from_single_dict(collate_fn([dataset[index] for index in range(len(dataset))]))
            self.assertEqual(batch.batch["input_ids"].shape, (len(rows), 32768))
            self.assertEqual(batch.batch["attention_mask"].shape, batch.batch["input_ids"].shape)
            self.assertTrue(torch.isfinite(batch.batch["input_ids"]).all())
            for index, original in enumerate(rows):
                info = batch.non_tensor_batch["extra_info"][index]
                expected = original["extra_info"]
                self.assertEqual(info["task_id"], expected["task_id"])
                self.assertEqual(info["split"], "train")
                self.assertEqual(info["domain"], expected["domain"])
                self.assertEqual(
                    batch.non_tensor_batch["tools_kwargs"][index]["_tau2_env_kwargs"],
                    expected["tools_kwargs"]["_tau2_env_kwargs"],
                )
                self.assertEqual(batch.non_tensor_batch["reward_model"][index]["type"], "env_reward")
            return batch

    def test_prepared_parquet_reaches_real_verl_batch_without_hidden_task_fields(self):
        from examples.tau3_verl.prepare import rows_for_tasks

        tasks = [SimpleNamespace(id=f"synthetic_train_{index}",
                                 user_scenario="HIDDEN_USER_SENTINEL",
                                 evaluation_criteria={"answer": "HIDDEN_EVALUATOR_SENTINEL"})
                 for index in range(2)]
        # Heterogeneous public schemas must not be corrupted by Arrow's struct
        # unification. The rollout loads canonical schemas by domain itself.
        schemas = [
            {"type": "function", "function": {"name": "lookup", "parameters": {
                "type": "object", "properties": {"user_id": {"type": "string"}},
            }}},
            {"type": "function", "function": {"name": "calculate", "parameters": {
                "type": "object", "properties": {"expression": {"type": "string"}},
            }}},
        ]
        rows = rows_for_tasks(tasks, policy="public policy", schemas=schemas, domain="retail",
                              environment={"domain": "retail", "task_split": "train"}, role="train")
        encoded = json.dumps(rows)
        self.assertNotIn("HIDDEN_USER_SENTINEL", encoded)
        self.assertNotIn("HIDDEN_EVALUATOR_SENTINEL", encoded)
        batch = self.data_roundtrip(rows)
        self.assertEqual(len(batch), 2)

    def test_multiturn_info_grpo_advantage_drives_finite_nonzero_policy_update(self):
        """Synthetic rollouts exercise the actual migrated advantage/loss code."""
        import numpy as np
        import torch
        from verl.trainer.ppo.core_algos import compute_info_grpo_advantage, compute_policy_loss

        response_mask = torch.tensor([[1., 1., 1., 1., 1., 0.]] * 4)
        terminal_rewards = torch.zeros(4, 6)
        terminal_rewards[:, 4] = torch.tensor([0., 1., 0.25, 0.75])
        intrinsic_rewards = torch.tensor([
            [.01, .03, .02, .07, .04, 0.], [.04, .02, .06, .03, .08, 0.],
            [.03, .01, .05, .02, .04, 0.], [.08, .06, .02, .09, .07, 0.],
        ])
        boundaries = torch.tensor([[1, 0, 0, 1, 0, 0]] * 4)
        histories = [[{"reward": 0.0, "choice": "action"}, {"reward": value, "choice": "answer"}]
                     for value in (0., 1., .25, .75)]
        advantages, returns, *_ = compute_info_grpo_advantage(
            terminal_rewards, intrinsic_rewards, response_mask,
            np.array(["synthetic_task_a", "synthetic_task_a", "synthetic_task_b", "synthetic_task_b"]),
            turn_boundaries=boundaries, conversation_histories=histories,
            data_sources=["tau3_training"] * 4, intrinsic_weight=.1,
        )
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertTrue(torch.isfinite(returns).all())
        self.assertGreater(advantages.abs().sum().item(), 0.)
        self.assertEqual(advantages[:, -1].abs().sum().item(), 0.)

        # Each sampled context has its own tiny trainable categorical policy.
        # This is a gradient connectivity check, not benchmark training.
        logits = torch.nn.Parameter(torch.zeros(4, 6, 3))
        optimizer = torch.optim.SGD([logits], lr=.05)
        labels = torch.arange(24).reshape(4, 6) % 3
        log_probs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        previous = logits.detach().clone()
        loss, *_ = compute_policy_loss(
            log_probs.detach(), log_probs, advantages, response_mask, cliprange=.2,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.)
        self.assertEqual(logits.grad[:, -1].abs().sum().item(), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(previous, logits.detach()))

    @unittest.skipUnless(os.environ.get("TAU3_BENCHMARK_ROOT"), "Set TAU3_BENCHMARK_ROOT for native task loading")
    def test_native_official_train_tasks_reach_the_same_parquet_pipeline(self):
        from examples.tau3_minimal.tau_adapter import Tau3Adapter
        from examples.tau3_verl.prepare import rows_for_tasks

        for domain in ("retail", "airline", "telecom"):
            with self.subTest(domain=domain), patch.dict(os.environ):
                adapter = Tau3Adapter(
                    benchmark_root=os.environ["TAU3_BENCHMARK_ROOT"], domain=domain,
                    reward_mode="action", user_model="openai/offline-fixture",
                    user_base_url="http://127.0.0.1:1/v1", max_turns=2,
                )
                from tau2.registry import registry
                environment = registry.get_env_constructor(domain)(solo_mode=False)
                rows = rows_for_tasks(
                    adapter.tasks[:2], policy=environment.get_policy(),
                    schemas=[tool.openai_schema for tool in environment.get_tools()], domain=domain,
                    environment={"domain": domain, "task_split": "train"}, role="train",
                )
                actual_ids = {row["extra_info"]["task_id"] for row in rows}
                self.assertTrue(actual_ids <= set(adapter.identity["train_task_ids"]))
                self.data_roundtrip(rows)


if __name__ == "__main__":
    unittest.main()
