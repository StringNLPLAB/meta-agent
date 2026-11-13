"""Tau2Gym environment -- the module ``tau2gym/__init__.py`` imports but that
upstream never committed.

Reconstructed against four sources so the contract is not guessed:
  * ``tau2gym/config.py`` (present) -- fields this env must honour;
  * ``verl/tools/tau2_tool_manager.py`` -- how the trainer builds tools and
    what it expects ``step`` to accept;
  * ``examples/tau2/config/tool_config/tau2_tool_config.yaml`` -- the action
    protocol: ``interact_with_env(choice in {message, tool_call, done}, content)``;
  * ``TauGym/taugym/env/tau_env.py`` -- the sibling gym's ``reset``/``step``
    signature, which ``verl/tools/env_manager.py`` relies on.

Native task scoring is delegated to tau2-bench's ``evaluate_simulation``.
An explicitly configured training-environment bundle may change training
rewards and dynamics; the raw native task score remains a separate metric.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - gymnasium is a hard dep in training
    gym = None

from ..config import Tau2GymConfig


def _ensure_tau2_importable(data_dir: Optional[str]) -> None:
    """Put the vendored tau2-bench on the path and point it at its data.

    tau2 reads ``TAU2_DATA_DIR`` at import time, so this has to run before the
    first ``import tau2``. The vendored copy lives at ``<repo>/tau2-bench``;
    an already-installed ``tau2`` package wins if there is one.
    """
    if data_dir and not os.environ.get("TAU2_DATA_DIR"):
        os.environ["TAU2_DATA_DIR"] = data_dir
    try:
        import tau2  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    src = os.path.join(repo, "tau2-bench", "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _add_userrl_gyms_to_path() -> None:
    """Make ``symbolic_layer`` importable, preferring the vendored copy.

    The hot layers (env_policy / difficulty / process_reward) live in UserRL's
    ``symbolic_layer``. Prefer the copy vendored at ``<repo>/UserRL/gyms`` so a
    fresh clone is self-contained; fall back to the training host's absolute
    path if the vendored tree is absent.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    for cand in (os.path.join(repo, "UserRL", "gyms") ,):
        if os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return


# ----------------------------------------------------------------------------
# hot-reloadable layers (absent = pre-existing behaviour, same as the other gyms)
# ----------------------------------------------------------------------------

def _hot(name: str, default):
    try:
        _add_userrl_gyms_to_path()
        from symbolic_layer import env_policy  # type: ignore
        return getattr(env_policy, name)
    except Exception:  # noqa: BLE001
        return default


def _tau2_flow_knobs() -> Dict[str, Any]:
    fn = _hot("get_tau2_flow", None)
    if fn is None:
        return {}
    try:
        return fn() or {}
    except Exception:  # noqa: BLE001
        return {}


def _resolve_strategy(task_id: Any, fallback: str) -> str:
    fn = _hot("resolve_user_strategy", None)
    if fn is None:
        return fallback
    try:
        return fn("tau2", task_id, fallback)
    except Exception:  # noqa: BLE001
        return fallback


_FLOW_RUNTIME_DEFAULTS = {
    "facts_per_reply": 1, "match_threshold": 0.34, "guide_interval": 2,
    "max_reveals": 0, "reveal_identifiers": True, "tool_threshold": 0.30,
    "tool_strong_threshold": 0.6, "stall_cap": 8,
}


def _review_policy_snapshot():
    """Read bytes once, then derive both provenance and applied reset knobs."""
    _add_userrl_gyms_to_path()
    from symbolic_layer import env_policy  # type: ignore

    if env_policy._META_ARM != "online":
        raise RuntimeError("User-simulator uptake instrumentation requires USERRL_META_ARM=online")
    source = os.path.abspath(env_policy._CONFIG._path)
    with open(source, "rb") as handle:
        raw = handle.read()
    configured_policy = json.loads(raw.decode("utf-8"))
    if not isinstance(configured_policy, dict):
        raise RuntimeError("User-simulator policy root must be a JSON object")
    snapshot = copy.deepcopy(env_policy._DEFAULTS)
    snapshot.update(configured_policy)
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    metadata = {
        "config_version": snapshot.get("version"),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "policy_sha256": hashlib.sha256(raw).hexdigest(),
        "policy_snapshot_sha256": hashlib.sha256(encoded).hexdigest(),
        "policy_sha256_encoding": "raw config bytes",
        "config_path": source,
        "meta_arm": env_policy._META_ARM,
    }
    # The native accessor admits exactly these keys. Preserve that behavior,
    # including its defaults, without triggering another hot-loader read.
    knobs = dict(env_policy._TAU2_FLOW_DEFAULTS)
    configured = snapshot.get("tau2_flow") or {}
    if isinstance(configured, dict):
        knobs.update({key: value for key, value in configured.items() if key in knobs})
    return snapshot, knobs, metadata


def _snapshot_strategy(snapshot, task_id: Any, fallback: str) -> str:
    """The native deterministic selector, applied to the same reset snapshot."""
    try:
        mix = snapshot.get("user_strategy_mix", {}) or {}
        if "tau2" not in mix:
            return fallback
        probability = float(mix["tau2"])
    except (TypeError, ValueError, AttributeError):
        return fallback
    if probability <= 0:
        return "llm"
    if probability >= 1:
        return "flow"
    digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()
    return "flow" if int(digest[:8], 16) / 0xFFFFFFFF < probability else "llm"


def _record_user_uptake(env, user, metadata):
    directory = os.environ.get("EAU_USER_SIM_UPTAKE_DIR")
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)
    supplied = dict(getattr(user, "knobs", {}) or {})
    effective = {
        key: supplied.get(key) if supplied.get(key) is not None else default
        for key, default in _FLOW_RUNTIME_DEFAULTS.items()
    } if env.user_strategy == "flow" else {}
    if metadata.get("usersim_bundle_sha256"):
        # The complete simulator owns its knob semantics, including new keys.
        effective = copy.deepcopy(supplied)
    record = {
        **metadata,
        "time": datetime.now(timezone.utc).isoformat(),
        "run": metadata.get("run") or os.environ.get("EAU_STEP_REVIEW_RUN"),
        "episode_id": env.episode_id,
        "task_id": str(env.task.id), "domain": env.config.domain,
        "user_strategy": env.user_strategy,
        "flow_knobs": effective, "supplied_flow_knobs": supplied,
        "pid": os.getpid(),
        "rank_environment": {key: os.environ[key] for key in (
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "RAY_WORKER_ID", "RAY_NODE_ID", "RAY_RANK"
        ) if key in os.environ},
    }
    payload = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(os.path.join(directory, f"{os.getpid()}.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        # A Ray worker can reset multiple envs concurrently. Hold the append
        # lock for the complete record, including a possible partial write.
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("User-simulator uptake append made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _difficulty(field: str, default):
    try:
        _add_userrl_gyms_to_path()
        from symbolic_layer import difficulty  # type: ignore
        return difficulty.get_overrides("tau2").get(field, default)
    except Exception:  # noqa: BLE001
        return default


def _tau2_process_weights() -> Dict[str, float]:
    """Hot-tunable process-reward weights for the tau2 gym.

    Empty (== all 0.0) reproduces the pre-existing terminal-only reward byte for
    byte, so landing this code changes nothing until a ``"tau2"`` section is
    written into process_reward.json. The loader there is a stat-on-access
    _HotJson (USERRL_META_ARM=online), so weight edits hot-reload mid-run -- the
    meta-agent can retune the ladder at a checkpoint without a relaunch.
    """
    if os.environ.get("EAU_TAU2_REWARD_MODE") == "terminal_only":
        return {}
    try:
        _add_userrl_gyms_to_path()
        from symbolic_layer.process_reward import weights as _w  # type: ignore
        return _w("tau2") or {}
    except Exception:  # noqa: BLE001
        return {}


# Argument keys that carry a domain identifier across tau2's three domains
# (retail / airline / telecom). "user_id" is priced separately from the rest so
# the auth->lookup chain's two milestones can be rewarded independently.
_TAU2_ID_KEYS = (
    "user_id", "order_id", "item_ids", "new_item_ids", "payment_method_id",
    "product_id", "reservation_id", "flight_number", "customer_id", "line_id",
)

# Tokens that count as a harvestable identifier when they appear in a tool
# RESULT: retail order (#W...), 10+-digit item ids, payment ids, snake-case user
# ids, and airline's 6-char booking codes. Deliberately harvested from tool
# results only -- an id the agent typed that never came back from a tool is
# exactly the hallucination (#W0000000) the grounding gate must NOT reward.
_TAU2_ID_TOKEN_RES = (
    re.compile(r"#W\d{6,}"),
    re.compile(r"\b\d{10,}\b"),
    re.compile(r"\b(?:gift_card|credit_card|paypal)_\d+\b"),
    re.compile(r"\b[a-z]+_[a-z]+_\d{2,4}\b"),
    re.compile(r"\b[A-Z0-9]{6}\b"),
)
_TAU2_REPEAT_MULT_CAP = 4


# ----------------------------------------------------------------------------
# action parsing
# ----------------------------------------------------------------------------

_CHOICES = ("message", "tool_call", "done")


def parse_action_string(action: str) -> Tuple[str, str]:
    """Return ``(choice, content)`` from whatever the tool layer handed us.

    Three shapes reach here in practice: the JSON the interact tool emits, the
    ``[message] ...`` bracket form the sibling gyms use, and bare text from a
    model that ignored both. Bare text is treated as a message rather than an
    error -- refusing it would train the policy on a formatting game instead of
    on the task.
    """
    raw = str(action or "").strip()
    if not raw:
        return "message", ""

    # Qwen3 (the policy model) prefixes EVERY turn with a <think>...</think>
    # reasoning block and emits tool calls in the Hermes/Qwen native wrapper
    # <tool_call>{json}</tool_call>. Neither shape was handled here: the leading
    # <think> meant raw never .startswith("{"), the functional-form anchor below
    # never matched (it starts with "<"), so the whole turn -- tool call included
    # -- fell through to "message". find_user_id_by_name_zip and every other
    # domain call was therefore delivered to the USER as chat text, the tool
    # never executed, no tool result came back, and success sat pinned at 0 for
    # the entire run while the agent looped "I need your user id" forever (the
    # same misroute the JSON/functional branches below were added to stop, just
    # for the wrapper this policy actually emits). Strip the think block first,
    # then honour the <tool_call> wrapper before the older shape checks.
    raw = re.sub(r"^\s*<think>.*?</think>\s*", "", raw, flags=re.S).strip()
    if not raw:
        return "message", ""
    tc = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.S)
    if tc and tc.group(1).strip():
        return "tool_call", tc.group(1).strip()

    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            choice = str(obj.get("choice", "")).strip().lower()
            if choice in _CHOICES:
                return choice, str(obj.get("content", "") or "")
            # A model that ignores the interact_with_env wrapper and emits a
            # domain tool call directly -- {"name": "...", "arguments": {...}} --
            # used to fall through to "message", so the call TEXT was delivered to
            # the user sim, the domain tool never executed, and success sat pinned
            # at 0 with no gradient toward the wrapper. Route that shape as a
            # tool_call; parse_tool_call consumes the identical JSON. interact_with_env
            # itself always arrives with a "choice" key, so it is handled above and
            # never reaches here.
            if isinstance(obj, dict) and (
                obj.get("name") or obj.get("tool") or obj.get("function")
            ):
                return "tool_call", raw
        except Exception:  # noqa: BLE001
            pass

    m = re.match(r"^\[(\w+)\]\s*(.*)$", raw, re.S)
    if m and m.group(1).lower() in _CHOICES:
        return m.group(1).lower(), m.group(2)

    lowered = raw.lower()
    if lowered in ("done", "[done]", "finish", "###stop###"):
        return "done", ""
    # Bare functional call form `name(arg="value")` -- the exact shape the tool
    # reference tells the agent to use -- must route as a tool_call too, or the
    # same misroute strands models that skip the JSON wrapper. The regex is a
    # full-string anchor (identifier + one balanced paren group), so ordinary
    # prose messages, which do not match it, are unaffected and still route to
    # the user. interact_with_env is excluded so a stray bare call to the wrapper
    # tool does not become a tau2 "tool not found".
    fm = re.match(r"^([A-Za-z_]\w*)\s*\(.*\)\s*$", raw, re.S)
    if fm and fm.group(1) != "interact_with_env":
        return "tool_call", raw
    return "message", raw


def parse_tool_call(content: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Parse ``{"name":..,"arguments":{..}}`` or ``name(a=1, b="x")``."""
    raw = str(content or "").strip()
    if not raw:
        return None, {}
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            name = obj.get("name") or obj.get("tool") or obj.get("function")
            args = obj.get("arguments") or obj.get("args") or obj.get("kwargs") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            if name:
                return str(name), dict(args if isinstance(args, dict) else {})
        except Exception:  # noqa: BLE001
            pass
    m = re.match(r"^([A-Za-z_]\w*)\s*\((.*)\)\s*$", raw, re.S)
    if not m:
        return None, {}
    name, body = m.group(1), m.group(2).strip()
    args: Dict[str, Any] = {}
    if body:
        try:
            args = json.loads("{" + re.sub(r"(\w+)\s*=", r'"\1":', body) + "}")
        except Exception:  # noqa: BLE001
            for kv in re.finditer(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,]+)", body):
                args[kv.group(1)] = kv.group(2).strip().strip("'\"")
    return name, args


class Tau2Env(gym.Env if gym is not None else object):
    """Gym wrapper over one tau2-bench task.

    One episode = one customer-service conversation. The agent alternates
    between talking to the user and calling domain tools; reward comes from
    tau2's evaluator at the end.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[Tau2GymConfig] = None):
        self.config = config or Tau2GymConfig()
        _ensure_tau2_importable(self.config.tau2_data_dir)

        from tau2.registry import registry  # noqa: WPS433

        self._registry = registry
        self._tasks = self._load_split_tasks()
        if not self._tasks:
            raise ValueError(
                f"no tasks for domain={self.config.domain} split={self.config.task_split}"
            )

        self._task_cursor = 0
        self.env = None
        self.task = None
        self.user = None
        self.user_strategy = "flow"
        self.episode_id = None
        self._episode_provenance: Dict[str, Any] = {}
        self.messages: List[Dict[str, Any]] = []
        self.step_count = 0
        self.episode_complete = False
        self._final_reward = 0.0
        self._process_reward_total = 0.0
        self._training_reward_total = 0.0
        self._environment_reward_delta_total = 0.0
        self._environment_mechanism = None
        self._environment_config = {}
        self._environment_metadata = None
        self._episode_process_weights = None
        self._episode_max_steps = None
        self._start_time = None
        # dense process-reward state (see _tau2_milestone_reward); reset per
        # episode. Init here too so a stray step() before reset() can't crash.
        self._grounded_ids: set = set()
        self._grounded_blob = ""
        self._probe_counts: Dict[str, int] = {}
        self._failed_calls: set = set()
        self._turn_reward = 0.0

    # ------------------------------------------------------------------
    # tasks
    # ------------------------------------------------------------------

    def _load_split_tasks(self) -> List[Any]:
        """Tasks for the configured split, in the benchmark's own order.

        tau2 ships ``split_tasks.json`` next to each domain; using it (rather
        than slicing ``tasks.json``) is what keeps this gym's train/test
        boundary identical to the benchmark's and to the repo's parquet.
        """
        tasks = self._registry.get_tasks_loader(self.config.domain)()
        by_id = {str(t.id): t for t in tasks}
        split = (self.config.task_split or "train").lower()
        if split == "base":
            return list(tasks)
        data_dir = os.environ.get("TAU2_DATA_DIR", self.config.tau2_data_dir or "")
        path = os.path.join(
            data_dir, "tau2", "domains", self.config.domain, "split_tasks.json"
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                ids = json.load(f).get(split) or []
            picked = [by_id[str(i)] for i in ids if str(i) in by_id]
            if picked:
                return picked
        except Exception:  # noqa: BLE001
            pass
        return list(tasks)

    def _select_task(self, options: Optional[Dict[str, Any]]) -> Any:
        wanted = None
        if options:
            wanted = options.get("task_id") or options.get("id")
        if wanted is None and self.config.data_source:
            src = self.config.data_source
            wanted = src if isinstance(src, str) else src[0]
        if wanted is not None:
            for t in self._tasks:
                if str(t.id) == str(wanted):
                    return t
        if self.config.data_mode == "random":
            import random

            return random.choice(self._tasks)
        task = self._tasks[self._task_cursor % len(self._tasks)]
        self._task_cursor += 1
        return task

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        from tau2.data_model.message import UserMessage  # noqa: WPS433

        self.task = self._select_task(options)
        self.env = self._registry.get_env_constructor(self.config.domain)()
        # Task-specific initial DB state, when the task ships one. Skipping this
        # would score every task against the domain's pristine database.
        init = getattr(self.task, "initial_state", None)
        if init is not None and hasattr(self.env, "set_state"):
            try:
                self.env.set_state(
                    initialization_data=getattr(init, "initialization_data", None),
                    initialization_actions=getattr(init, "initialization_actions", None),
                    message_history=getattr(init, "message_history", None) or [],
                )
            except Exception:  # noqa: BLE001
                if (os.environ.get("EAU_USER_SIM_MODE") == "native"
                        or os.environ.get("EAU_USER_SIM_BOOTSTRAP") == "coser_scratch_v1"):
                    raise
                pass

        self._review_flow_knobs = None
        self._review_policy_metadata = None
        self._usersim_factory = None
        self._environment_mechanism = None
        self._environment_config = {}
        self._environment_metadata = None
        self._episode_process_weights = None
        self._episode_max_steps = None
        self.episode_id = uuid.uuid4().hex
        fallback = os.environ.get("USERRL_USER_STRATEGY", "flow")
        bundle_pointer = os.environ.get("EAU_USER_SIM_BUNDLE")
        if os.environ.get("EAU_USER_SIM_MODE") == "native":
            # No policy mix or inherited mutable bundle may replace the stock user.
            if bundle_pointer or os.environ.get("EAU_STEP_REVIEW_STATE"):
                raise ValueError("Native baseline cannot use a user bundle or step reviewer")
            self.user_strategy = "native"
        elif bundle_pointer:
            from .usersim_runtime import load_usersim_bundle  # noqa: WPS433

            self._usersim_factory, snapshot, self._review_policy_metadata = load_usersim_bundle(
                bundle_pointer, expected_run=os.environ.get("EAU_STEP_REVIEW_RUN") or None,
            )
            self._review_flow_knobs = copy.deepcopy(snapshot.get("tau2_flow", {}))
            # A bundle is the versioned flow implementation; a strategy mix in
            # its full policy must not silently redirect it to the LLM user.
            self.user_strategy = "flow"
        elif os.environ.get("EAU_USER_SIM_UPTAKE_DIR"):
            snapshot, self._review_flow_knobs, self._review_policy_metadata = _review_policy_snapshot()
            self.user_strategy = _snapshot_strategy(snapshot, str(self.task.id), fallback)
        else:
            self.user_strategy = _resolve_strategy(str(self.task.id), fallback)
        self._episode_provenance = {"episode_id": self.episode_id}
        if self._review_policy_metadata is not None:
            self._episode_provenance.update({
                key: value for key, value in self._review_policy_metadata.items()
                if key in {
                    "usersim_bundle_version", "usersim_bundle_sha256",
                    "usersim_code_sha256", "usersim_config_sha256",
                    "code_sha256", "config_sha256", "config_version",
                }
            })
        environment_pointer = os.environ.get('EAU_ENVIRONMENT_BUNDLE')
        if environment_pointer:
            if self.user_strategy == 'native':
                raise ValueError('Native baseline cannot use a training environment bundle')
            from .environment_runtime import load_environment_bundle

            mechanism, self._environment_config, self._environment_metadata = load_environment_bundle(
                environment_pointer, expected_run=os.environ.get('EAU_STEP_REVIEW_RUN') or None)
            self._episode_provenance.update({key: value for key, value in self._environment_metadata.items()
                if key.startswith('environment_') and key != 'environment_files'})
            self._environment_mechanism = mechanism(copy.deepcopy(self._environment_config))
        self.user = None
        self.messages = []
        self.step_count = 0
        self.episode_complete = False
        self._final_reward = 0.0
        self._scoring_error = None
        self._termination_reason = "agent_stop"
        self._process_reward_total = 0.0
        self._training_reward_total = 0.0
        self._environment_reward_delta_total = 0.0
        self._start_time = datetime.now(timezone.utc)
        # Fresh grounding memory per episode: an id is only "grounded" if it came
        # back from a successful tool result IN THIS conversation.
        self._grounded_ids = set()
        self._grounded_blob = ""
        self._probe_counts = {}
        self._failed_calls = set()
        self._turn_reward = 0.0

        if self._environment_mechanism is not None:
            self._call_environment_hook('on_reset')
            self._snapshot_environment_controls()
        self.user = self._build_user()
        if self._environment_metadata is not None:
            from .environment_runtime import record_environment_uptake

            record_environment_uptake(self, self._environment_metadata)

        opening = self.user.opening() if hasattr(self.user, "opening") else ""
        if opening:
            self.messages.append({"role": "user", "content": opening})

        # The agent's only declared function is the trainer's interact_with_env
        # wrapper, so the domain's tools are invisible to it unless they are
        # written into what it reads. Without this it invented plausible names
        # ("check_network_status") and every call came back "Tool not found",
        # which is indistinguishable from a policy failure in the reward.
        tool_ref = self._tool_reference()
        first_obs = f"{tool_ref}\n\n{opening}" if tool_ref else opening

        # "feedback" is the key the trainer reads (env_manager._create_tau2gym_
        # environment does observation.get("feedback", str(observation)), and the
        # ColBench env uses the same key). Returning only "observation" made the
        # fallback fire and handed the model the whole dict's repr -- policy text,
        # tool list and all -- as the user's opening line.
        obs = {
            "feedback": first_obs,
            "observation": first_obs,
            "policy": self._policy_text(),
            "tools": [t.name for t in self.env.get_tools()],
            "domain": self.config.domain,
            "task_id": str(self.task.id),
            **self._episode_provenance,
        }
        return obs, {"task_id": str(self.task.id), "user_strategy": self.user_strategy,
                     **self._episode_provenance}

    def _build_user(self):
        bootstrap = os.environ.get("EAU_USER_SIM_BOOTSTRAP")
        if bootstrap:
            factory = getattr(self, "_usersim_factory", None)
            if (bootstrap != "coser_scratch_v1" or self.user_strategy != "flow"
                    or getattr(factory, "tau2_public_user_api", None) != 1):
                raise ValueError("Scratch CoSER requires a verified public-scenario user factory")
            initial = getattr(self.task, "initial_state", None)
            if getattr(initial, "message_history", None):
                raise ValueError("Scratch CoSER requires tasks without initial message history")
            knobs = self._review_flow_knobs
            if not isinstance(knobs, dict):
                raise ValueError("Scratch user knobs must come from its verified new bundle")
            try:
                tools = self.env.get_user_tools(include=self.task.user_tools) or None
            except Exception:
                tools = None
            user = factory(self.task.user_scenario, knobs, user_tools=tools,
                           environment=self.env, record=lambda message: self.messages.append(message),
                           task_id=str(self.task.id))
            if self._review_policy_metadata is not None:
                _record_user_uptake(self, user, self._review_policy_metadata)
            return user
        if self.user_strategy == "native":
            from .native_user import Tau2NativeUser

            initial = getattr(self.task, "initial_state", None)
            if getattr(initial, "message_history", None):
                raise ValueError("Native training adapter requires tasks without initial message history")
            try:
                tools = self.env.get_user_tools(include=self.task.user_tools) or None
            except Exception:
                tools = None
            return Tau2NativeUser(
                self.task.user_scenario, tools, self.env,
                lambda message: self.messages.append(message),
            )
        instructions = getattr(
            getattr(self.task, "user_scenario", None), "instructions", None
        )
        criteria = getattr(self.task, "evaluation_criteria", None)
        actions = getattr(criteria, "actions", None) or []
        if self.user_strategy == "flow":
            # Telecom's user owns 30 device tools; retail/airline raise
            # "User tools not available", which is a normal outcome, not an error.
            try:
                user_tools = self.env.get_user_tools()
            except Exception:  # noqa: BLE001
                user_tools = []
            knobs = self._review_flow_knobs if self._review_flow_knobs is not None else _tau2_flow_knobs()
            factory = self._usersim_factory
            if factory is None:
                from .flow_user import Tau2FlowUser  # noqa: WPS433

                factory = Tau2FlowUser
            u = factory(instructions, actions, knobs, user_tools=user_tools)
            if user_tools:
                u.tool_runner = self._run_user_tool
            if self._review_policy_metadata is not None:
                _record_user_uptake(self, u, self._review_policy_metadata)
            return u
        from .llm_user import Tau2LLMUser  # noqa: WPS433

        user = Tau2LLMUser(instructions, self.config)
        if self._review_policy_metadata is not None:
            _record_user_uptake(self, user, self._review_policy_metadata)
        return user

    def _run_user_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Execute one user-side tool through tau2's own path, and record it.

        requestor="user" is what routes it to the user environment; the returned
        content is already phrased for the user to say aloud ("Your SIM card is
        active and working."), so nothing here needs to format it.

        The call is written into ``self.messages`` as a user-requestor tool call
        plus its response. Without this the device action happened live but left
        no ToolCall in the scored trajectory, so tau2's evaluator -- which credits
        an expected action only when a matching ToolCall is present -- saw zero
        user actions and scored every telecom task with a user-side golden action
        (66 of 74 in the training split) a flat 0. Recording it here is what makes
        those tasks winnable under both ACTION and ENV evaluation.
        """
        from tau2.data_model.message import ToolCall  # noqa: WPS433

        args = dict(arguments or {})
        call_id = str(uuid.uuid4())
        call = ToolCall(id=call_id, name=name, arguments=args, requestor="user")
        try:
            msg = self.env.get_response(call)
            out = str(getattr(msg, "content", "") or "")
            err = bool(getattr(msg, "error", False))
        except Exception as e:  # noqa: BLE001
            out, err = f"Error: {e}", True
        self.messages.append(
            {"role": "user", "content": None, "requestor": "user",
             "tool_calls": [{"id": call_id, "name": name, "arguments": args}]}
        )
        self.messages.append(
            {"role": "tool", "id": call_id, "name": name, "requestor": "user",
             "content": out, "error": err}
        )
        # A device action can reveal a real identifier (e.g. a line/customer id)
        # that a later AGENT call may legitimately reuse, so harvest it -- but do
        # NOT pay a milestone here: the auth->lookup ladder rewards agent-side
        # progress, and the user performing a device check is the user's move.
        if not err:
            self._harvest_ids(out)
        # A user-requestor golden action advances the task exactly like an agent
        # one, so completion must be marked here too, or the flow user would
        # guide forever towards device work already done.
        if hasattr(self.user, "note_tool_call"):
            self.user.note_tool_call(name, args)
        return out

    def _tool_reference(self) -> str:
        """One line per domain tool: name(params). Names and parameter names
        only -- the descriptions are long enough to crowd out the task."""
        try:
            lines = []
            for t in self.env.get_tools():
                # Read openai_schema, not `params`: on tau2's Tool the latter is
                # a *class*, which is truthy, so a `params or {}` fallback never
                # fired and every signature came out empty.
                schema = getattr(t, "openai_schema", None) or {}
                props = (((schema.get("function") or {}).get("parameters") or {})
                         .get("properties") or {})
                required = set((((schema.get("function") or {}).get("parameters")
                                 or {}).get("required") or []))
                names = ", ".join(
                    n if n in required else f"{n}?" for n in list(props)[:8])
                lines.append(f"  {t.name}({names})")
            if not lines:
                return ""
            ref = ("Available tools (call with choice=\"tool_call\" and content "
                   "like name(arg=\"value\")):\n" + "\n".join(lines))

            # Telecom's policy text describes the device checks the *user*
            # performs, and the agent read those names and tried to call them
            # itself: 7189 "Tool not found" in one run, every one of them a
            # user-side tool. Naming them here, on the other side of an explicit
            # line, is what tells the agent to ask instead of call.
            try:
                user_tools = self.env.get_user_tools()
            except Exception:  # noqa: BLE001
                user_tools = []
            if user_tools:
                names = ", ".join(t.name for t in user_tools[:30])
                ref += ("\n\nThe following are actions on the USER's own device. "
                        "You cannot call them. Ask the user to do them in plain "
                        "language (choice=\"message\"), e.g. \"could you check "
                        "your SIM card status?\":\n  " + names)
            return ref
        except Exception:  # noqa: BLE001
            return ""

    def _policy_text(self) -> str:
        try:
            return self.env.get_policy()
        except Exception:  # noqa: BLE001
            return ""

    def _max_steps(self) -> int:
        if getattr(self, '_episode_max_steps', None) is not None:
            return self._episode_max_steps
        return int(_difficulty("max_steps", self.config.max_steps))

    def _snapshot_environment_controls(self):
        """Pin legacy reward/difficulty controls for this environment version."""
        settings = self._environment_config
        configured = settings.get('process_reward')
        if configured is None:
            configured = _tau2_process_weights()
        elif isinstance(configured, dict) and 'tau2' in configured:
            configured = configured['tau2']
        if not isinstance(configured, dict):
            raise ValueError('Environment process_reward must be a weight mapping')
        weights = {}
        for key, value in configured.items():
            if (not isinstance(key, str) or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError('Environment process reward weights must be finite numbers')
            weights[key] = float(value)
        self._episode_process_weights = copy.deepcopy(weights)
        difficulty = settings.get('difficulty', {})
        if not isinstance(difficulty, dict):
            raise ValueError('Environment difficulty must be a JSON object')
        limit = settings.get('max_steps', difficulty.get('max_steps',
            _difficulty('max_steps', self.config.max_steps)))
        if isinstance(limit, bool):
            raise ValueError('Environment max_steps must be a positive integer')
        parsed = int(limit)
        if parsed != limit or parsed <= 0:
            raise ValueError('Environment max_steps must be a positive integer')
        self._episode_max_steps = parsed

    def _environment_protected_state(self):
        """Protect fixed users, scoring and recorded actions across a hook.

        Environment code is trusted Python, not a security sandbox. These
        checks catch accidental writes outside the declared mutable boundary;
        the native domain instance in env.env remains editable for dynamics.
        """
        user = self.user
        task = self.task.model_dump(mode='python') if hasattr(self.task, 'model_dump') else vars(self.task)
        scoring_fields = ('domain', 'task_split', 'solo_mode', 'reward_scale',
                          'use_action_evaluation', 'use_nl_assertions', 'use_env_assertions',
                          'tau2_data_dir')
        return {'user_identity': id(user), 'user_type': type(user),
                'user_knobs': copy.deepcopy(getattr(user, 'knobs', None)),
                'user_methods': tuple(getattr(user, key, None) for key in
                    ('opening', 'reply', 'note_tool_call')),
                'usersim_factory': self._usersim_factory,
                'usersim_config': copy.deepcopy(self._review_flow_knobs),
                'usersim_metadata': copy.deepcopy(self._review_policy_metadata),
                'environment_config': copy.deepcopy(self._environment_config),
                'environment_metadata': copy.deepcopy(self._environment_metadata),
                'episode_weights': copy.deepcopy(self._episode_process_weights),
                'episode_max_steps': self._episode_max_steps,
                'task_identity': id(self.task), 'task': copy.deepcopy(task),
                'messages': copy.deepcopy(self.messages),
                'episode_id': self.episode_id,
                'step_count': self.step_count, 'episode_complete': self.episode_complete,
                'termination_reason': self._termination_reason,
                'provenance': copy.deepcopy(self._episode_provenance),
                'native_reward': self._final_reward, 'scoring_error': self._scoring_error,
                'process_total': self._process_reward_total,
                'training_total': self._training_reward_total,
                'delta_total': self._environment_reward_delta_total,
                'score_method': self._score, 'score_type_method': self._evaluation_type,
                'audit_method': self._record_terminal_reward_audit,
                'scoring_config': {key: copy.deepcopy(getattr(self.config, key, None)) for key in scoring_fields}}

    def _call_environment_hook(self, name, *arguments):
        mechanism = getattr(self, '_environment_mechanism', None)
        if mechanism is None:
            return None
        before = self._environment_protected_state()
        try:
            result = getattr(mechanism, name)(self, *arguments)
        finally:
            # Even a failed hook cannot contaminate the independent score
            # consumed by rollout finalization or the native audit channel.
            changed_native = (self._final_reward != before['native_reward']
                              or self._scoring_error != before['scoring_error'])
            self._final_reward = before['native_reward']
            self._scoring_error = before['scoring_error']
        if changed_native or self._environment_protected_state() != before:
            raise ValueError(f'Environment {name} modified frozen user, native scoring or recorded trajectory state')
        if name == 'on_reset' and result is not None:
            raise ValueError('Environment on_reset must return None')
        return result

    def step(self, action_input: str):
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")

        self.step_count += 1
        # _do_tool_call / _run_user_tool add milestone reward here as they fire.
        self._turn_reward = 0.0
        choice, content = parse_action_string(action_input)
        if getattr(self, '_environment_mechanism', None) is not None:
            selected = self._call_environment_hook('before_action', choice, content)
            if (not isinstance(selected, tuple) or len(selected) != 2
                    or selected[0] not in _CHOICES or not isinstance(selected[1], str)):
                raise ValueError('Environment before_action must return a valid (choice, content) tuple')
            choice, content = selected

        terminated = False
        if choice == "tool_call":
            feedback = self._do_tool_call(content)
        elif choice == "done":
            terminated = True
            feedback = "Conversation ended."
        else:
            feedback, terminated = self._do_message(content)

        truncated = (not terminated) and self.step_count >= self._max_steps()
        # Dense per-turn reward always counts; the terminal task score is added
        # ON TOP on the last turn. A [done] turn makes no tool call, so
        # _turn_reward is 0 there and terminal reward == _score() exactly as
        # before -- gold-replay [done] still scores 1.0, doing-nothing still 0.
        reward = (0.0 if os.environ.get("EAU_TAU2_REWARD_MODE") == "terminal_only"
                  else self._turn_reward)
        self._process_reward_total = getattr(self, '_process_reward_total', 0.0) + reward
        if terminated or truncated:
            if (self.user_strategy == "native"
                    or os.environ.get("EAU_TAU2_REWARD_MODE") in {"terminal_only", "process"}):
                self._termination_reason = ("max_steps" if truncated else
                                            "agent_stop" if choice == "done" else "user_stop")
            reward = reward + self._score()
            self.episode_complete = True

        base_terminal = terminated or truncated
        base_reward = reward

        obs = {"feedback": feedback, "observation": feedback,
               "step": self.step_count, **self._episode_provenance}
        info = {
            "task_id": str(self.task.id),
            "user_strategy": self.user_strategy,
            "turns": self.step_count,
            "reward": reward,
            **self._episode_provenance,
        }
        if getattr(self, '_environment_mechanism', None) is not None:
            result = self._call_environment_hook('after_action', obs, reward, terminated, truncated, info)
            if (not isinstance(result, tuple) or len(result) != 5
                    or not isinstance(result[0], dict) or isinstance(result[1], bool)
                    or not isinstance(result[1], (int, float)) or not math.isfinite(result[1])
                    or type(result[2]) is not bool or type(result[3]) is not bool
                    or not isinstance(result[4], dict)):
                raise ValueError('Environment after_action must return finite reward and a valid five-tuple')
            obs, reward, terminated, truncated, info = result
            reward = float(reward)
            # A new environment stop still receives the independent native
            # score audit; the returned training reward belongs to the hook.
            if (terminated or truncated) and not base_terminal:
                self._termination_reason = ('max_steps' if truncated else
                    'agent_stop' if choice == 'done' else 'user_stop')
                self._score()
            self.episode_complete = terminated or truncated
            obs.update(self._episode_provenance)
            info.update(self._episode_provenance, task_id=str(self.task.id),
                        user_strategy=self.user_strategy, turns=self.step_count, reward=reward)
        self._training_reward_total = getattr(self, '_training_reward_total', 0.0) + reward
        self._environment_reward_delta_total = getattr(self, '_environment_reward_delta_total', 0.0) + reward - base_reward
        if (terminated or truncated) and os.environ.get('EAU_TERMINAL_REWARD_AUDIT_DIR'):
            info['terminal_action_reward'] = self._final_reward
            info['process_reward_total'] = self._process_reward_total
            self._record_terminal_reward_audit()
        return obs, reward, terminated, truncated, info

    def _record_terminal_reward_audit(self):
        """Keep task completion separate from combined training reward."""
        directory = os.environ.get('EAU_TERMINAL_REWARD_AUDIT_DIR')
        if not directory:
            return
        os.makedirs(directory, exist_ok=True)
        record = {**self._episode_provenance,
                  'run': os.environ.get('EAU_STEP_REVIEW_RUN'),
                  'task_id': str(self.task.id), 'domain': self.config.domain,
                  'terminal_action_reward': self._final_reward,
                  'process_reward_total': self._process_reward_total,
                  'training_reward_total': getattr(self, '_training_reward_total', self._process_reward_total + self._final_reward),
                  'environment_reward_delta_total': getattr(self, '_environment_reward_delta_total', 0.0),
                  'termination_reason': self._termination_reason,
                  'scoring_error_type': getattr(self, '_scoring_error', None)}
        descriptor = os.open(os.path.join(directory, f'{os.getpid()}.jsonl'),
                             os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, 'a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()

    # ------------------------------------------------------------------
    # the two things an agent can do
    # ------------------------------------------------------------------

    def _do_tool_call(self, content: str) -> str:
        from tau2.data_model.message import ToolCall  # noqa: WPS433

        name, args = parse_tool_call(content)
        if not name:
            return f"Could not parse a tool call from: {content[:200]}"

        # Go through tau2's own get_response rather than use_tool + our own
        # json.dumps: the evaluator re-runs the trajectory and compares each
        # recorded tool output against a fresh one *as text*. Serialising it
        # ourselves produced JSON where tau2 produces a pydantic repr, so every
        # otherwise-perfect episode failed the replay and scored 0. Using the
        # benchmark's own call path makes the two identical by construction.
        call_id = str(uuid.uuid4())
        call = ToolCall(id=call_id, name=name, arguments=args)
        self.messages.append(
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": call_id, "name": name, "arguments": args}]}
        )
        try:
            tool_msg = self.env.get_response(call)
            out = tool_msg.content or ""
            err = bool(getattr(tool_msg, "error", False))
        except Exception as e:  # noqa: BLE001
            out, err = f"Error: {e}", True
        self.messages.append(
            {"role": "tool", "id": call_id, "name": name,
             "content": out, "error": err}
        )
        # 先计价、后采集: price this call against ids grounded by PRIOR results,
        # then harvest THIS result -- so a call can never ground itself.
        self._turn_reward += self._tau2_milestone_reward(name, args, out, err)
        if not err:
            self._harvest_ids(out)
        # Progress is measured by tool calls, not by what the user was told, so
        # the flow user only learns the task advanced from here.
        if hasattr(self.user, "note_tool_call"):
            self.user.note_tool_call(name, args)
        return out

    # ------------------------------------------------------------------
    # dense process reward (episode-once grounding ladder, keyed "tau2")
    # ------------------------------------------------------------------

    def _harvest_ids(self, text: str) -> None:
        """Record identifiers that appeared in a SUCCESSFUL tool result.

        Both the raw result text (for substring grounding of domain ids the
        regexes below don't enumerate, e.g. airline booking codes) and the
        regex-extracted tokens are kept. An id must have come back from a tool
        to become grounded -- one the agent merely typed never does.
        """
        t = str(text or "")
        if not t:
            return
        self._grounded_blob += "\n" + t
        for rx in _TAU2_ID_TOKEN_RES:
            for tok in rx.findall(t):
                self._grounded_ids.add(str(tok))

    def _grounded_arg_kinds(self, args: Optional[Dict[str, Any]]) -> set:
        """Which grounded-id kinds this call reuses: subset of {user_id, other}.

        An arg is grounded if its value was harvested from a prior successful
        result (exact token) or appears verbatim in prior result text (>=4 chars,
        catches domain ids the token regexes miss). "user_id" is its own kind so
        the authenticate milestone can be priced apart from object lookups.
        """
        kinds: set = set()
        for k in _TAU2_ID_KEYS:
            if not args or k not in args:
                continue
            v = args[k]
            vals = [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
            for val in vals:
                if val in self._grounded_ids or (
                        len(val) >= 4 and val in self._grounded_blob):
                    kinds.add("user_id" if k == "user_id" else "other")
        return kinds

    def _guidance_reward(self, key: str, weights: Dict[str, float]) -> float:
        """Episode-once bonus: a milestone key pays its weight the FIRST time it
        fires and 0 thereafter, so it can't be farmed by repetition. Weight 0
        (the default) disables the key entirely."""
        w = float(weights.get(key, 0.0) or 0.0)
        if w == 0.0 or self._probe_counts.get(key, 0) > 0:
            return 0.0
        self._probe_counts[key] = 1
        return w

    def _tau2_milestone_reward(self, name: str, args: Optional[Dict[str, Any]],
                               out: str, err: bool) -> float:
        """Per-turn process reward for one AGENT tool call. Sum of episode-once
        bonuses stays well under the terminal 1.0. Empty weights => 0.0 => the
        env behaves exactly as before this landed."""
        weights = getattr(self, '_episode_process_weights', None)
        if weights is None:
            weights = _tau2_process_weights()
        if not weights:
            return 0.0
        total = 0.0
        sig = f"{name}:{json.dumps(args or {}, sort_keys=True, default=str)}"
        if not err:
            total += self._guidance_reward("first_tool_ok", weights)
            kinds = self._grounded_arg_kinds(args)
            if "user_id" in kinds:
                total += self._guidance_reward("first_grounded_user_id", weights)
            if "other" in kinds:
                total += self._guidance_reward("first_grounded_object_id", weights)
            total += float(weights.get("tool_call_ok", 0.0) or 0.0)
        else:
            total += float(weights.get("tool_call_error", 0.0) or 0.0)
            rw = float(weights.get("repeat_failed_call", 0.0) or 0.0)
            if rw and sig in self._failed_calls:
                n = min(self._probe_counts.get("repeat_failed_call", 0) + 1,
                        _TAU2_REPEAT_MULT_CAP)
                self._probe_counts["repeat_failed_call"] = \
                    self._probe_counts.get("repeat_failed_call", 0) + 1
                total += rw * n
            self._failed_calls.add(sig)
        return total

    def _do_message(self, content: str) -> Tuple[str, bool]:
        self.messages.append({"role": "assistant", "content": content})
        reply = self.user.reply(content)
        if reply is None:
            return "The user is satisfied and has ended the conversation.", True
        self.messages.append({"role": "user", "content": reply})
        return reply, False

    # ------------------------------------------------------------------
    # scoring -- delegated to tau2's own evaluator
    # ------------------------------------------------------------------

    def _score(self) -> float:
        self._scoring_error = None
        try:
            from tau2.data_model.simulation import SimulationRun, TerminationReason
            from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
            from tau2.data_model.message import (
                AssistantMessage, ToolCall, ToolMessage, UserMessage,
            )
        except Exception as e:  # noqa: BLE001
            self._scoring_error = type(e).__name__
            if self.user_strategy == "native" or os.environ.get("EAU_STRICT_SCORING") == "1":
                raise
            return 0.0

        try:
            typed: List[Any] = []
            for m in self.messages:
                role = m.get("role")
                if role == "user":
                    calls = m.get("tool_calls")
                    if calls:
                        # A user-requestor device action. Recorded as a
                        # UserMessage carrying the ToolCall so the evaluator's
                        # extract_tool_calls (Assistant OR User) finds it.
                        typed.append(
                            UserMessage(
                                role="user",
                                content=m.get("content"),
                                tool_calls=[
                                    ToolCall(id=c.get("id") or str(uuid.uuid4()),
                                             name=c["name"],
                                             arguments=c.get("arguments") or {},
                                             requestor="user")
                                    for c in calls
                                ] or None,
                            )
                        )
                    else:
                        typed.append(UserMessage(role="user", content=m.get("content") or ""))
                elif role == "tool":
                    typed.append(
                        ToolMessage(
                            id=m.get("id") or str(uuid.uuid4()), role="tool",
                            content=m.get("content") or "",
                            requestor=m.get("requestor", "assistant"),
                            error=bool(m.get("error", False)),
                        )
                    )
                elif role == "assistant":
                    calls = m.get("tool_calls")
                    typed.append(
                        AssistantMessage(
                            role="assistant",
                            content=m.get("content"),
                            tool_calls=[
                                ToolCall(id=c.get("id") or str(uuid.uuid4()),
                                         name=c["name"],
                                         arguments=c.get("arguments") or {})
                                for c in (calls or [])
                            ] or None,
                        )
                    )
            now = datetime.now(timezone.utc)
            sim = SimulationRun(
                id=str(uuid.uuid4()),
                task_id=str(self.task.id),
                start_time=(self._start_time or now).isoformat(),
                end_time=now.isoformat(),
                duration=(now - (self._start_time or now)).total_seconds(),
                termination_reason=TerminationReason(
                    getattr(self, "_termination_reason", "agent_stop")
                ),
                messages=typed,
            )
            info = evaluate_simulation(
                simulation=sim,
                task=self.task,
                evaluation_type=self._evaluation_type(EvaluationType),
                solo_mode=bool(self.config.solo_mode),
                domain=self.config.domain,
            )
            self._final_reward = float(getattr(info, "reward", 0.0) or 0.0)
        except Exception as e:  # noqa: BLE001
            self._scoring_error = type(e).__name__
            if self.user_strategy == "native" or os.environ.get("EAU_STRICT_SCORING") == "1":
                raise
            if self.config.verbose:
                print(f"[Tau2Env] evaluation failed: {e}")
            self._final_reward = 0.0
        return self._final_reward * float(self.config.reward_scale)

    def _evaluation_type(self, EvaluationType):
        """Which tau2 criteria form the reward.

        This is a real decision, not a passthrough. ``ALL`` is the
        benchmark-faithful choice, but tau2's reward is the *product* of the
        components in a task's ``reward_basis``, and retail's basis includes
        NL_ASSERTION -- an LLM judge. That puts an API call inside every
        rollout's reward, and when the key is missing the judge does not raise,
        it scores 0, so a perfectly-solved episode is silently graded a failure.
        Training on that would teach the policy from noise.

        So ``use_nl_assertions=False`` (the default for training) grades on
        environment/DB state instead, which is deterministic and local. Set it
        True for benchmark-faithful evaluation, and make sure a judge model is
        reachable when you do.
        """
        # Env var wins, because the trainer builds its config with
        # tau2gym.get_default_config() -- whose use_nl_assertions defaults to
        # True -- so the config object alone cannot keep the LLM judge out of a
        # training run. It did not: every rollout hit tau2's own llm_utils and
        # failed on the model string, and nothing reached step 1.
        mode = os.environ.get("TAU2_EVAL_TYPE", "action").strip().lower()
        if mode == "all":
            return EvaluationType.ALL
        if mode == "env":
            return EvaluationType.ENV
        if mode == "action":
            return EvaluationType.ACTION
        if getattr(self.config, "use_nl_assertions", False):
            return EvaluationType.ALL
        if not getattr(self.config, "use_env_assertions", True):
            return EvaluationType.ACTION
        # ENV covers DB checks and env assertions -- the basis for retail
        # (DB) and telecom (ENV_ASSERTION) alike, with no LLM in the path.
        return EvaluationType.ENV

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # trainer-side contract
    # ------------------------------------------------------------------
    # Two names the training path reaches for that the offline tests never
    # exercised, because they only exist once verl's tool layer is in the loop:
    #
    #   .tau2_env   verl/tools/tau2_tool_wrapper.py asserts on it before every
    #               tool call and treats None as "environment not initialised".
    #   step_async  interact_tool awaits it; EnvWrapper only falls back to the
    #               sync step() when the attribute is absent on the wrapper,
    #               not when it is absent on the env underneath.

    @property
    def tau2_env(self):
        """The underlying tau2-bench Environment, under the name the tool layer
        expects. Kept as a property rather than a second attribute so the two
        can never drift apart."""
        return self.env

    async def step_async(self, action_input: str):
        """Async form of step(). The work is CPU- and subprocess-bound, so there
        is nothing to await -- this exists to satisfy the await in the tool
        layer, not to add concurrency."""
        return self.step(action_input)

    def render(self, mode="human"):
        for m in self.messages:
            print(f"[{m.get('role')}] {str(m.get('content'))[:200]}")

    def close(self):
        self.env = None
