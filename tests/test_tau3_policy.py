"""Exercise the real HF policy with tiny random local weights, without downloads."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tau3_minimal"
sys.path.insert(0, str(EXAMPLE))
from train import LocalPolicy, state_hash


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")
                     and importlib.util.find_spec("tokenizers"),
                     "local HF policy checks require torch, transformers, and tokenizers")
class LocalHFPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        cls.temporary = tempfile.TemporaryDirectory(prefix="tau3-random-local-policy-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.path = Path(cls.temporary.name)
        cls.offline = patch.dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        cls.offline.start()
        cls.addClassCleanup(cls.offline.stop)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(17)
        words = ["[PAD]", "[UNK]", "system", "user", "assistant", ":", "public", "policy",
                 "hello", "read", "tools", "response", "yes", "no", "done", "sample"]
        tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
        fast.save_pretrained(cls.path)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(words), n_embd=16, n_layer=1,
            n_head=2, n_positions=64, bos_token_id=None, eos_token_id=None, pad_token_id=0))
        # The adapter must not inherit recipe-specific sampling transformations.
        model.generation_config.do_sample = True
        model.generation_config.top_k = 1
        model.save_pretrained(cls.path)

    def test_sampled_action_log_probability_has_real_parameter_gradients(self):
        import torch
        torch.manual_seed(23)
        policy = LocalPolicy(self.path, max_new_tokens=4, max_prompt_tokens=32)
        self.assertFalse(policy.model.training)
        self.assertEqual(policy.generation_config.top_k, 0)
        self.assertEqual(policy.generation_config.top_p, 1.0)
        self.assertEqual(policy.generation_config.temperature, 1.0)
        sample = policy.sample("public policy", "hello")
        self.assertEqual(len(sample["generated_token_ids"]), 4)
        sequence = sample["input_ids"]
        with torch.no_grad():
            logits = policy.model(input_ids=sequence.unsqueeze(0)).logits[0, :-1].float()
            selected = torch.log_softmax(logits, dim=-1).gather(1, sequence[1:, None]).squeeze(1)
            expected = selected[sample["prompt_length"] - 1:].sum()
        actual = policy.log_probability(sample)
        torch.testing.assert_close(actual.detach(), expected)
        before = state_hash(policy.model)
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=0.01, weight_decay=0)
        optimizer.zero_grad()
        (-actual).backward()  # A positive observed training reward.
        gradient = sum(float(parameter.grad.abs().sum()) for parameter in policy.model.parameters()
                       if parameter.grad is not None)
        self.assertGreater(gradient, 0)
        optimizer.step()
        self.assertNotEqual(state_hash(policy.model), before)

    def test_context_overflow_is_rejected_without_truncating_task_instructions(self):
        policy = LocalPolicy(self.path, max_new_tokens=4, max_prompt_tokens=1)
        with self.assertRaisesRegex(ValueError, "no silent prompt truncation"):
            policy.sample("public policy", "hello")


if __name__ == "__main__":
    unittest.main()
