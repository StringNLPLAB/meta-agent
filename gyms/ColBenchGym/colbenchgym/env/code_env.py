"""ColBench backend-programming environment for the verl interaction tool.

The policy asks plain-text clarification questions and submits Python code.
An LLM or symbolic user holds the private training reference. The environment
scores answers against test-case expressions in time-limited subprocesses.
Optional training hooks load one immutable snapshot at each episode reset.
Subprocess execution is not an OS sandbox; deployment boundaries are external.
"""

import ast
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .flow_user import FlowUser

import sys as _sys
from pathlib import Path as _Path
_SYMBOLIC = str(_Path(__file__).resolve().parents[4] / "UserRL" / "gyms")
if _SYMBOLIC not in _sys.path:
    _sys.path.insert(0, _SYMBOLIC)
try:
    from symbolic_layer.env_policy import (
        resolve_user_strategy, get_colbench_flow, content_task_id,
    )
except Exception:  # noqa: BLE001
    def resolve_user_strategy(gym, task_id, fallback):
        return fallback

    def get_colbench_flow():
        return {}

    def content_task_id(text):
        import hashlib as _h
        return "sha:" + _h.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]

_PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
_HUMAN_PROMPT_PATH = os.path.join(_PROMPT_DIR, "human_simulator_code_prompt.txt")

# Answer markers, in the several spellings the model actually emits. Order
# matters only for reporting; the first hit wins.
_ANSWER_PATTERNS = [
    "I WANT TO ANSWER:",
    "I WANT_TO_ANSWER:",
    "I WANT_TO ANSWER:",
    "i want to answer:",
    "i want_to_answer:",
]

# Executing model-generated code: a task gets this long for all of its test
# cases put together, then the process is killed.
_EXEC_TIMEOUT_S = float(os.environ.get("COLBENCH_EXEC_TIMEOUT", "20"))

# Said when the agent submits an answer before asking anything. The problem
# statement is deliberately under-specified, so a person holding the hidden
# requirements would not accept a solution from someone who never asked.
_TOO_EARLY_REPLY = (
    "Hold on -- you have not asked me anything yet, and there are details I have "
    "not told you. What do you need to know?"
)
_FLOAT_RTOL = 1e-6

# The simulated human holds the reference solution as "hidden information" and
# is meant to describe it in prose, two sentences at a time. Weaker simulators
# instead paste the solution back verbatim, which turns "interview the user" into
# "copy the reply" and makes the reward uninformative about the policy. The
# guard removes code from the reply while keeping any prose around it, so the
# information channel stays open and only the giveaway is closed.
_LEAK_GUARD = os.environ.get("COLBENCH_LEAK_GUARD", "1") not in ("0", "false", "False")
_LEAK_NGRAM = 40
_NO_CODE_REPLY = (
    "I am not able to write the code for you. Ask me about a specific behaviour "
    "and I will describe what I expect."
)
_EMPTY_REPLY = "Sorry, I am not sure about that."
_DEF_RE = re.compile(r"def\s+\w+\s*\(", re.M)
_FENCE_RE = re.compile(r"```.*?```", re.S)

# Reference solutions are deterministic, so their outputs are computed once per
# distinct task rather than once per rollout (10k tasks x n rollouts x epochs).
_GT_CACHE: Dict[str, List[str]] = {}
_GT_CACHE_CAP = 20000


_MIN_QUESTION_CHARS = 25


def leading_question(response: str) -> str:
    """The question a turn asked before it began answering, or "" if none.

    A turn may contain both ("Should it return inf or raise? I WANT TO ANSWER:
    ..."). Everything before the first answer marker is the question part.
    """
    text = response or ""
    lowered = text.lower()
    cut = len(text)
    for pattern in _ANSWER_PATTERNS:
        idx = lowered.find(pattern.lower())
        if idx != -1:
            cut = min(cut, idx)
    prefix = text[:cut].strip()
    if "?" in prefix or len(prefix) >= _MIN_QUESTION_CHARS:
        return prefix
    return ""


def check_and_extract_answer(response: str) -> Tuple[bool, str]:
    """(has_answer, answer_text) -- mirrors run_simulate_api.check_and_extract_answer."""
    if not response:
        return False, ""
    lowered = response.lower()
    for pattern in _ANSWER_PATTERNS:
        pl = pattern.lower()
        if pl in lowered:
            if pattern in response:
                idx = response.find(pattern)
            else:
                idx = lowered.find(pl)
            return True, response[idx + len(pattern):].strip()
    return False, ""


def _strip_code_fences(text: str) -> str:
    """Drop a leading ```python fence if the model wrapped its answer in one."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


_STR_LITERAL_RE = re.compile(r"(['\"])(?:\\.|(?!\1).)*\1", re.S)


def _strip_str_literals(text: str) -> str:
    """Blank out quoted literals before comparing a reply against the solution.

    A reference solution often returns natural-language strings ("Check if the
    BIOS is set to boot from the correct disk"). That English is exactly what
    the simulated human is supposed to paraphrase, so leaving it in the corpus
    makes a correct, prose-only reply look like a verbatim leak -- which is how
    the guard first flagged a legitimate answer. Only the code skeleton counts.
    """
    return _STR_LITERAL_RE.sub(" ", text or "")


def _looks_like_code(line: str) -> bool:
    """Is this line pasted code rather than prose?

    ``def`` is matched at the start of the line only. The function signature is
    public -- it is printed in the problem description -- so a human who refers
    to it mid-sentence is not leaking anything, and blocking that reply would
    close a legitimate channel.
    """
    st = line.strip()
    if not st:
        return False
    if _DEF_RE.match(st):
        return True
    if line[:1] in (" ", "\t") and any(
        st.startswith(k) for k in ("return ", "if ", "elif ", "else:", "for ", "while ", "try:", "except")
    ):
        return True
    return st.startswith(("import ", "from ", "class "))


def _sanitize_human_reply(reply: str, hidden: str) -> Tuple[str, bool]:
    """Strip pasted solution code out of a human-simulator reply.

    Returns (clean_reply, blocked). Prose is preserved; only fenced blocks and
    code-looking lines are dropped. A reply that shares a long verbatim run with
    the hidden solution is discarded outright even if it survives the line
    filter, which catches a solution reflowed onto one line.
    """
    if not _LEAK_GUARD or not reply:
        return reply, False
    original = reply
    cleaned = _FENCE_RE.sub(" ", reply)
    kept = [ln for ln in cleaned.splitlines() if not _looks_like_code(ln)]
    cleaned = "\n".join(kept).strip()

    hid = _norm_ws(_strip_str_literals(hidden))
    body = hid.split(":", 1)[1] if ":" in hid else hid
    if len(body) >= _LEAK_NGRAM:
        norm = _norm_ws(_strip_str_literals(cleaned))
        for k in range(0, len(body) - _LEAK_NGRAM, 8):
            if body[k:k + _LEAK_NGRAM] in norm:
                cleaned = ""
                break

    blocked = _norm_ws(cleaned) != _norm_ws(original)
    if not cleaned.strip():
        cleaned = _NO_CODE_REPLY
    return cleaned, blocked


_HARNESS = r'''
import json, sys
spec = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
results = []
ns = {}
try:
    exec(spec["code"], ns)
    compiled = True
except BaseException as e:
    compiled = False
    err = type(e).__name__
for expr in spec["tests"]:
    if not compiled:
        results.append("__NOCOMPILE__:" + err)
        continue
    try:
        results.append(repr(eval(expr, ns)))
    except BaseException as e:
        results.append("__ERROR__:" + type(e).__name__)
with open(out_path, "w") as f:
    json.dump(results, f)
'''


def _run_code(code: str, tests: List[str]) -> List[str]:
    """Run ``code``, evaluate each test expression, return one repr per test."""
    if not code.strip() or not tests:
        return ["__NOCOMPILE__:Empty"] * len(tests)
    tmpdir = tempfile.mkdtemp(prefix="colbench_exec_")
    spec_path = os.path.join(tmpdir, "spec.json")
    out_path = os.path.join(tmpdir, "out.json")
    harness_path = os.path.join(tmpdir, "harness.py")
    try:
        with open(spec_path, "w") as f:
            json.dump({"code": code, "tests": tests}, f)
        with open(harness_path, "w") as f:
            f.write(_HARNESS)
        try:
            subprocess.run(
                [sys.executable, harness_path, spec_path, out_path],
                timeout=_EXEC_TIMEOUT_S,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return ["__TIMEOUT__"] * len(tests)
        try:
            with open(out_path, "r") as f:
                results = json.load(f)
        except Exception:  # noqa: BLE001
            return ["__CRASH__"] * len(tests)
        if not isinstance(results, list) or len(results) != len(tests):
            return ["__CRASH__"] * len(tests)
        return [str(r) for r in results]
    finally:
        for p in (spec_path, out_path, harness_path):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def _tolerant_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        scale = max(abs(a), abs(b), 1.0)
        return abs(a - b) <= _FLOAT_RTOL * scale
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_tolerant_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        return all(_tolerant_equal(x, y) for x, y in zip(a, b))
    return a == b


def _results_match(got: str, want: str) -> bool:
    """One test case: did the agent's output match the reference?

    A reference solution that itself errors or times out cannot be matched --
    that test case is unscoreable and counted as a miss rather than a free pass.
    """
    if want.startswith(("__ERROR__", "__NOCOMPILE__", "__TIMEOUT__", "__CRASH__")):
        return False
    if got == want:
        return True
    try:
        return _tolerant_equal(ast.literal_eval(got), ast.literal_eval(want))
    except Exception:  # noqa: BLE001
        return False


class ColBenchCodeEnv:
    """Multi-turn collaborative-coding environment.

    The agent talks to a simulated human who holds the reference solution as
    hidden information and answers in at most two sentences. The episode ends
    when the agent emits an answer marker or ``max_steps`` turns elapse; the
    reward is then the fraction of the task's test cases on which the agent's
    code reproduces the reference's output.
    """

    def __init__(
        self,
        config,
        task: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        id: Optional[Any] = None,  # noqa: A002 -- name fixed by env_manager's call
    ) -> None:
        self.config = config
        self.category = category
        self.id = id
        self.max_steps = int(getattr(config, "max_steps", 10) or 10)
        self.verbose = bool(getattr(config, "verbose", False))
        self.char_limit = int(getattr(config, "human_response_char_limit", 400) or 400)
        self.reward_scale = float(getattr(config, "reward_scale", 1.0) or 1.0)
        self.step_penalty = float(getattr(config, "step_penalty", 0.0) or 0.0)

        self.model_name = getattr(config, "env_model_name", "") or ""
        self.base_url = getattr(config, "env_base_url", "") or ""
        self.api_key = getattr(config, "env_api_key", "") or "EMPTY"

        try:
            with open(_HUMAN_PROMPT_PATH, "r", encoding="utf-8") as f:
                self.human_prompt = f.read()
        except OSError as e:
            raise RuntimeError(f"human simulator prompt missing at {_HUMAN_PROMPT_PATH}: {e}")

        self._sync_client = None
        self._async_client = None

        self.problem_description = ""
        self.hidden_information = ""
        self.test_cases: List[str] = []
        self.dialogue_history: List[Dict[str, Any]] = []
        self.answer = "No answer"
        self.agent_answer = "No answer"
        self.step_count = 0
        self.done = False
        self.episode_complete = False
        self._reward: Optional[float] = None
        self.leak_blocks = 0
        self.user_strategy = "llm"
        self.questions_asked = 0
        self.answers_refused = 0
        self._flow: Optional[FlowUser] = None
        self._last_score_detail: Dict[str, Any] = {}
        self._mechanism = None
        self._episode_flow_knobs = None
        self._runtime_receipt = None

        if task:
            self._load_task(task)

    # ------------------------------------------------------------------ task

    def _load_task(self, task: Dict[str, Any]) -> None:
        self.problem_description = str(task.get("problem_description", "") or "")
        self.hidden_information = str(task.get("ground_truth", "") or "")
        self.test_cases = self._normalize_test_cases(task.get("test_cases"))

    @staticmethod
    def _normalize_test_cases(raw: Any) -> List[str]:
        """Test cases reach us as {name: call-expression}, ordered by name.

        The parquet round-trip turns them into a struct whose fields are the
        union of every name in the dataset, so most entries of any given row
        are None -- those are dropped, not scored as empty.
        """
        if raw is None:
            return []
        if isinstance(raw, dict):
            items = [(k, v) for k, v in raw.items() if isinstance(v, str) and v.strip()]

            def _key(kv):
                name = str(kv[0])
                digits = "".join(ch for ch in name if ch.isdigit())
                return (int(digits) if digits else 0, name)

            return [v.strip() for _, v in sorted(items, key=_key)]
        if isinstance(raw, (list, tuple)):
            return [str(v).strip() for v in raw if str(v).strip()]
        return []

    # ----------------------------------------------------------------- reset

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        options = options or {}
        task = options.get("task")
        if task:
            self._load_task(task)

        self.dialogue_history = [{"role": "user", "content": self.problem_description}]
        self.answer = "No answer"
        self.agent_answer = "No answer"
        self.step_count = 0
        self.done = False
        self.episode_complete = False
        self._reward = None
        self.leak_blocks = 0
        self.questions_asked = 0
        self.answers_refused = 0
        self._last_score_detail = {}

        # Which simulator this episode faces. Seeded by task id like TravelGym,
        # so the mix is spread across the dataset and is never a source of
        # run-to-run variance.
        fallback = os.environ.get("COLBENCH_USER_STRATEGY", "llm")
        try:
            self.resolved_task_id = (
                self.id if self.id is not None else content_task_id(self.problem_description)
            )
            self.user_strategy = resolve_user_strategy(
                "colbench", self.resolved_task_id, fallback)
        except Exception:  # noqa: BLE001
            self.user_strategy = fallback
        flow_factory = FlowUser
        run = os.environ.get("EAU_COLBENCH_RUN")
        if run:
            from meta_agent.colbench_runtime import load_episode
            flow_factory, self._mechanism, policy, settings, receipt = load_episode(run)
            self._runtime_receipt = receipt
            self._episode_flow_knobs = policy.get("colbench_flow", {})
            probability = policy["user_strategy_mix"]["colbench"]
            draw = int(hashlib.sha256(str(self.resolved_task_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            self.user_strategy = "flow" if draw < probability else "llm"
            self.max_steps = settings["max_steps"]
            self.char_limit = settings["human_response_char_limit"]
            self.config.min_questions = settings["min_questions"]
            self.reward_scale = settings["reward_scale"]
            self.step_penalty = settings["step_penalty"]
        self._flow = flow_factory(self.hidden_information) if self.user_strategy == "flow" else None
        if self._mechanism:
            self._mechanism.on_reset(self)

        if os.environ.get("EAU_COLBENCH_UPTAKE_DIR"):
            from meta_agent.colbench_uptake import record
            record("environment_reset", {
                "task_id": str(self.resolved_task_id), "user_strategy": self.user_strategy,
                "max_steps": self.max_steps,
                "min_questions": int(getattr(self.config, "min_questions", 0)),
                "human_response_char_limit": self.char_limit,
                "flow_knobs_at_reset": self._episode_flow_knobs if run else get_colbench_flow(),
                "snapshot_version": self._runtime_receipt["version"] if run else None,
                "snapshot_sha256": self._runtime_receipt["sha256"] if run else None,
            })

        obs = {
            "feedback": self.problem_description,
            "last_message": self.problem_description,
            "step_count": self.step_count,
            "dialogue_history": self.get_dialogue_history(),
        }
        return obs, {"task_id": self.id, "category": self.category,
                     "num_test_cases": len(self.test_cases)}

    def get_dialogue_history(self) -> List[Dict[str, str]]:
        return [{"role": d["role"], "content": d["content"]} for d in self.dialogue_history]

    def _str_dialogue_history(self) -> str:
        out = ""
        for d in self.dialogue_history:
            out += f"{d['role']}:{d['content']}\n\n\n\n"
        return out + "agent:"

    # ------------------------------------------------------------ human sim

    def _human_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.human_prompt.format(
                problem_description=self.problem_description,
                hidden_information=self.hidden_information,
                dialogue_history=self._str_dialogue_history(),
            )},
        ]

    def _request_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._human_messages(),
            "max_tokens": 512,
            "temperature": float(os.environ.get("COLBENCH_USER_TEMPERATURE", "0") or 0),
            "timeout": 60.0,
        }
        if "qwen3" in self.model_name.lower():
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return params

    def _get_sync_client(self):
        if self._sync_client is None:
            from openai import OpenAI
            self._sync_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._sync_client

    def _get_async_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._async_client

    def _finalize_reply(self, text: str) -> str:
        clean, blocked = _sanitize_human_reply(text or "", self.hidden_information)
        if blocked:
            self.leak_blocks += 1
            if self.verbose:
                print(f"[colbench] leak guard fired on task {self.id} (step {self.step_count})")
        if not clean.strip():
            clean = _EMPTY_REPLY
        return clean

    def _flow_reply(self) -> str:
        """Symbolic reply for this turn, or "" when the flow user is not active.

        Knobs are re-read here rather than captured at reset, so a mid-run write
        to env_policy.json reaches episodes that are already in flight.
        """
        if self._flow is None:
            return ""
        try:
            self._flow.knobs = (self._episode_flow_knobs if self._episode_flow_knobs is not None
                                else (get_colbench_flow() or {}))
        except Exception:  # noqa: BLE001
            self._flow.knobs = {}
        last = ""
        for msg in reversed(self.dialogue_history):
            if msg.get("role") == "assistant":
                last = msg.get("content") or ""
                break
        return self._flow.reply(last)

    def _invoke_human(self) -> str:
        if self._flow is not None:
            return self._flow_reply()
        for attempt in range(3):
            try:
                completion = self._get_sync_client().chat.completions.create(**self._request_params())
                self._record_api_usage(completion, attempt=attempt + 1)
                return self._finalize_reply((completion.choices[0].message.content or "")[:self.char_limit])
            except Exception as e:  # noqa: BLE001
                self._record_api_usage(error=e, attempt=attempt + 1)
                if self.verbose or attempt == 2:
                    print(f"[colbench] human sim error (attempt {attempt + 1}/3): {e}")
        if os.environ.get("EAU_COLBENCH_RUN"):
            raise RuntimeError("Training user requests failed; refusing a fabricated simulator reply")
        return "No response."

    async def _invoke_human_async(self) -> str:
        if self._flow is not None:
            return self._flow_reply()
        for attempt in range(3):
            try:
                completion = await self._get_async_client().chat.completions.create(**self._request_params())
                self._record_api_usage(completion, attempt=attempt + 1)
                return self._finalize_reply((completion.choices[0].message.content or "")[:self.char_limit])
            except Exception as e:  # noqa: BLE001
                self._record_api_usage(error=e, attempt=attempt + 1)
                if self.verbose or attempt == 2:
                    print(f"[colbench] human sim error (attempt {attempt + 1}/3): {e}")
        if os.environ.get("EAU_COLBENCH_RUN"):
            raise RuntimeError("Training user requests failed; refusing a fabricated simulator reply")
        return "No response."

    def _record_api_usage(self, completion=None, *, attempt, error=None):
        if os.environ.get("EAU_TRAINING_API_USAGE_FILE"):
            from meta_agent.training_api_usage import record_call
            record_call(completion, model_requested=self.model_name,
                        task_id=getattr(self, "resolved_task_id", self.id), attempt=attempt, error=error)

    # ------------------------------------------------------------------ step

    def _min_questions(self) -> int:
        """How many questions must precede an accepted answer.

        Zero accepts immediate answers. A positive value requires interaction
        before submission, without adding a question bonus to the task score.
        """
        try:
            return int(getattr(self.config, "min_questions", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _pre_step(self, action: str) -> Tuple[str, bool]:
        """Record the agent turn. Returns (response, episode_now_done)."""
        self.step_count += 1
        response = action or ""
        if self._mechanism:
            response = self._mechanism.before_action(self, response)
            if not isinstance(response, str):
                raise ValueError("before_action must return text")
        if "OUTPUT:" in response:
            response = response.split("OUTPUT:")[1]

        has_answer, answer_text = check_and_extract_answer(response)

        # Too early: keep the episode going instead of accepting. Only while the
        # turn budget still allows it, so this can never prevent an answer from
        # ever being recorded.
        if (has_answer and self.questions_asked < self._min_questions()
                and self.step_count < self.max_steps):
            # A turn that asks *and* answers has still asked. Counting it only
            # as an answer deadlocked the episode: the refusal tells the agent
            # "you have not asked me anything yet", which is false, so no reply
            # it can produce satisfies the guard. Credit the question and let
            # the human answer it before the next submission.
            if leading_question(response):
                self.questions_asked += 1
                self.dialogue_history.append({"role": "assistant", "content": response})
                return response, False
            self.answers_refused += 1
            self.dialogue_history.append({"role": "assistant", "content": response})
            return response, False

        if not has_answer:
            self.questions_asked += 1

        if has_answer or self.step_count >= self.max_steps:
            self.done = True
            self.episode_complete = True
            self.answer = answer_text if has_answer else response
            self.agent_answer = self.answer

        self.dialogue_history.append({"role": "assistant", "content": response})
        return response, self.done

    def _post_step(self, human_reply: Optional[str]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if human_reply is not None:
            self.dialogue_history.append({"role": "user", "content": human_reply})
        feedback = human_reply if human_reply is not None else "Conversation ended."
        obs = {
            "feedback": feedback,
            "last_message": feedback,
            "step_count": self.step_count,
            "dialogue_history": self.get_dialogue_history(),
        }
        terminated = bool(self.done)
        truncated = False
        info = {
            "agent_answer": self.agent_answer,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "leak_blocks": self.leak_blocks,
            "user_strategy": self.user_strategy,
            "questions_asked": self.questions_asked,
            "answers_refused": self.answers_refused,
            "task_id": self.id,
        }
        result = (obs, 0.0, terminated, truncated, info)
        if self._mechanism:
            result = self._mechanism.after_action(self, *result)
            if (not isinstance(result, (list, tuple)) or len(result) != 5
                    or not isinstance(result[0], dict) or not isinstance(result[4], dict)
                    or not isinstance(result[1], (int, float)) or not math.isfinite(result[1])
                    or type(result[2]) is not bool or type(result[3]) is not bool):
                raise ValueError("after_action must return a valid finite Gym step")
        return result

    def step(self, action: str):
        if self.done:
            return self._post_step(None)
        refused_before = self.answers_refused
        _, finished = self._pre_step(action)
        if self.answers_refused > refused_before:
            return self._post_step(_TOO_EARLY_REPLY)
        return self._post_step(None if finished else self._invoke_human())

    async def step_async(self, action: str):
        if self.done:
            return self._post_step(None)
        refused_before = self.answers_refused
        _, finished = self._pre_step(action)
        if self.answers_refused > refused_before:
            return self._post_step(_TOO_EARLY_REPLY)
        if finished:
            return self._post_step(None)
        return self._post_step(await self._invoke_human_async())

    # ---------------------------------------------------------------- reward

    def extract_answer(self) -> str:
        """The answer to score, with run_simulate_api's fallback chain.

        A rollout can end without the agent ever emitting the marker (turn
        budget, truncation), so a trailing message that merely looks like code
        still counts -- otherwise those episodes score 0 for a formatting
        reason rather than a correctness one.
        """
        if self.answer and self.answer != "No answer":
            return self.answer.strip()
        for msg in reversed(self.dialogue_history):
            if msg.get("role") != "assistant":
                continue
            last = (msg.get("content") or "").strip()
            if not last:
                continue
            has_answer, answer_text = check_and_extract_answer(last)
            if has_answer and answer_text:
                return answer_text
            if len(last) > 20 and any(k in last for k in ("def ", "import ", "class ", "return ", "=")):
                return last
            if self.done and len(last) > 10:
                return last
            break
        return self.answer.strip() if self.answer else "No answer"

    def _reference_outputs(self) -> List[str]:
        key = hashlib.sha256(
            (self.hidden_information + "\x00" + "\x00".join(self.test_cases)).encode("utf-8")
        ).hexdigest()
        cached = _GT_CACHE.get(key)
        if cached is not None:
            return cached
        outputs = _run_code(self.hidden_information, self.test_cases)
        if len(_GT_CACHE) < _GT_CACHE_CAP:
            _GT_CACHE[key] = outputs
        return outputs

    def calculate_reward(self) -> float:
        """Fraction of test cases whose output matches the reference solution.

        Cached per episode: interact_tool may call this more than once, and each
        call is two subprocess executions.
        """
        if self._reward is not None:
            return self._reward

        answer = _strip_code_fences(self.extract_answer())
        self.agent_answer = answer

        if not self.test_cases or not self.hidden_information.strip():
            self._reward = 0.0
            self._last_score_detail = {"reason": "no test cases or no reference solution"}
            return self._reward
        if not answer or answer == "No answer":
            self._reward = 0.0
            self._last_score_detail = {"reason": "no answer extracted"}
            return self._reward

        want = self._reference_outputs()
        got = _run_code(answer, self.test_cases)
        passed = sum(1 for g, w in zip(got, want) if _results_match(g, w))
        total = len(self.test_cases)
        frac = passed / total if total else 0.0

        self._last_score_detail = {
            "passed": passed,
            "total": total,
            "all_pass": passed == total and total > 0,
            "compiled": not got[0].startswith("__NOCOMPILE__") if got else False,
            "reference_ok": not any(
                w.startswith(("__ERROR__", "__NOCOMPILE__", "__TIMEOUT__", "__CRASH__")) for w in want
            ),
        }
        reward = frac * self.reward_scale - self.step_penalty * self.step_count
        if self._mechanism:
            reward = self._mechanism.terminal_reward(self, reward)
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                raise ValueError("terminal_reward must return a finite scalar")
        self._last_score_detail["native_pass_fraction"] = frac
        self._reward = float(reward)
        if self.verbose:
            print(f"[colbench] task={self.id} reward={self._reward:.3f} {self._last_score_detail}")
        return self._reward

    @property
    def score_detail(self) -> Dict[str, Any]:
        return dict(self._last_score_detail)

    def close(self) -> None:
        self._sync_client = None
        self._async_client = None
