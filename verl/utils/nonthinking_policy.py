"""Check the actual Qwen3 prompt and completion for an explicitly strict run.

This guard does not strip reasoning, retry policy outputs, or modify sampling.
Every call is checked before its result can enter a training trajectory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

EMPTY_THINKING_SUFFIX = "<think>\n\n</think>\n\n"
_COUNTS = {"input_calls_checked": 0, "sampled_completions_checked": 0}


def required():
    return os.environ.get("EAU_REQUIRE_NONTHINKING_POLICY") == "1"


def check_input(tokenizer, input_ids):
    if not required():
        return
    if os.environ.get("USERRL_ENABLE_QWEN3_THINKING") != "0":
        raise RuntimeError("Strict non-thinking training requires an explicit disabled thinking mode")
    expected = tokenizer.encode(EMPTY_THINKING_SUFFIX, add_special_tokens=False)
    if not expected or list(input_ids[-len(expected):]) != expected:
        raise RuntimeError("Actual policy generation prompt lacks the closed empty thinking suffix")
    _COUNTS["input_calls_checked"] += 1


def check_output(output):
    if not required():
        return
    text = output.get("text") if isinstance(output, dict) else None
    if not isinstance(text, str):
        raise RuntimeError("Strict non-thinking policy completion must contain text")
    # Reject nonempty or malformed reasoning markers rather than hiding them.
    without_empty = re.sub(r"<think>\s*</think>", "", text, flags=re.IGNORECASE)
    if re.search(r"</?think\b", without_empty, flags=re.IGNORECASE):
        raise RuntimeError("Policy emitted reasoning despite the non-thinking training contract")
    if _COUNTS["input_calls_checked"] <= _COUNTS["sampled_completions_checked"]:
        raise RuntimeError("Policy completion lacks a verified non-thinking generation prompt")
    _COUNTS["sampled_completions_checked"] += 1
    number = _COUNTS["sampled_completions_checked"]
    if number == 1 or number % 100 == 0:
        receipt = dict(status="verified", pid=os.getpid(), **_COUNTS,
            actual_input_suffix_verified=True, nonempty_reasoning_accepted=0,
            reasoning_stripped=False, policy_output_retries=0)
        directory = os.environ.get("EAU_NONTHINKING_BEHAVIOR_DIR")
        if directory:
            target = Path(directory) / f"{os.getpid()}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            temporary.replace(target)
        if number == 1:
            print("EAU_NONTHINKING_POLICY_VERIFIED " + json.dumps(receipt, sort_keys=True), flush=True)
