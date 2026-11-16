"""Enforce the actual Qwen3 input mode and monitor raw policy completions.

This guard does not strip reasoning, retry policy outputs, or modify sampling.
Every input is checked. Output deviations are retained and counted by default;
an explicit error policy is available for callers that require a hard stop.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import hashlib

EMPTY_THINKING_SUFFIX = "<think>\n\n</think>\n\n"
_COUNTS = {"input_calls_checked": 0, "sampled_completions_checked": 0,
    "thinking_marker_outputs": 0, "nonempty_thinking_block_outputs": 0}
_LAST_DEVIATION = None


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


def flush_behavior():
    if not required() or _COUNTS["sampled_completions_checked"] < 1:
        return
    policy = os.environ.get("EAU_NONTHINKING_OUTPUT_POLICY", "monitor")
    receipt = dict(status="verified", pid=os.getpid(), **_COUNTS,
        actual_input_suffix_verified=True, output_policy=policy,
        raw_completions_preserved=policy == "monitor", reasoning_stripped=False,
        policy_output_retries=0, last_deviation=_LAST_DEVIATION)
    directory = os.environ.get("EAU_NONTHINKING_BEHAVIOR_DIR")
    if directory:
        target = Path(directory) / f"{os.getpid()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        temporary.replace(target)
    return receipt


def check_output(output):
    global _LAST_DEVIATION
    if not required():
        return
    text = output.get("text") if isinstance(output, dict) else None
    if not isinstance(text, str):
        raise RuntimeError("Strict non-thinking policy completion must contain text")
    policy = os.environ.get("EAU_NONTHINKING_OUTPUT_POLICY", "monitor")
    if policy not in {"monitor", "error"}:
        raise ValueError("EAU_NONTHINKING_OUTPUT_POLICY must be monitor or error")
    # Preserve the on-policy distribution, including model deviations or
    # literal task text containing markers. Never strip or resample a reply.
    without_empty = re.sub(r"<think>\s*</think>", "", text, flags=re.IGNORECASE)
    marker = bool(re.search(r"</?think\b", without_empty, flags=re.IGNORECASE))
    nonempty = any(value.strip() for value in re.findall(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL))
    if _COUNTS["input_calls_checked"] <= _COUNTS["sampled_completions_checked"]:
        raise RuntimeError("Policy completion lacks a verified non-thinking generation prompt")
    _COUNTS["sampled_completions_checked"] += 1
    _COUNTS["thinking_marker_outputs"] += int(marker)
    _COUNTS["nonempty_thinking_block_outputs"] += int(nonempty)
    if marker:
        _LAST_DEVIATION = dict(sample=_COUNTS["sampled_completions_checked"],
            output_sha256=hashlib.sha256(text.encode()).hexdigest(),
            characters=len(text), nonempty_thinking_block=nonempty)
    number = _COUNTS["sampled_completions_checked"]
    if number == 1 or number % 100 == 0 or marker:
        receipt = flush_behavior()
        if number == 1:
            print("EAU_NONTHINKING_POLICY_VERIFIED " + json.dumps(receipt, sort_keys=True), flush=True)
    if marker and policy == "error":
        raise RuntimeError("Policy emitted reasoning despite the non-thinking training contract")
