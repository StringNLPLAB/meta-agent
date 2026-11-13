"""Hot-reloadable per-gym difficulty overrides.

Every gym is built as ``get_default_config()`` then mutated by
``env_manager._apply_llm_env_overrides``, which until now injected only the LLM
settings (api key, base url, timeout, max_tokens, temperature). The difficulty
fields -- how often the search backend fails on purpose, how many turns the user
waits before volunteering a preference, how many decoy options exist -- sat in
the dataclass defaults and could only be changed by restarting.

This layer makes them writable mid-run so the meta-agent can raise difficulty on
a gym once the policy is comfortably solving it, without waiting for the next
run.

Two things to keep in mind when using it, both learned the expensive way:

* Raising difficulty **lowers the training score**. A band recorded at one
  difficulty is not comparable to a band at another, so the level has to be
  logged next to every reading or the curve stops meaning anything (this is L12
  with a bigger lever).
* **Eval difficulty is fixed.** Training harder than you evaluate is a
  train/eval mismatch, the same shape as flow-vs-llm and the greedy user
  simulator -- both of which cost a run before anyone noticed.

Config path: ``USERRL_DIFFICULTY_CONFIG``, default
``symbolic_layer/difficulty.json``. Absent file or absent gym = untouched
defaults.
"""

import json
import os
import threading
from typing import Any, Dict

_DEFAULT_PATH = os.environ.get(
    "USERRL_DIFFICULTY_CONFIG",
    os.path.join(os.path.dirname(__file__), "difficulty.json"),
)

# Whitelist: gym -> {field: (lo, hi, "direction that makes it harder")}. A field
# absent here can never be written, however the config file is edited.
KNOWN_FIELDS: Dict[str, Dict[str, Any]] = {
    "travel": {
        "search_failure_interval": (1, 50, "lower"),
        "elicitation_interval": (1, 20, "higher"),
        "noise_choice_number": (0, 30, "higher"),
        "wrong_choice_number": (1, 50, "lower"),
        "max_steps": (5, 40, "lower"),
    },
    "function": {
        "max_repeated_probes": (1, 20, "lower"),
        "max_steps": (5, 40, "lower"),
    },
    "turtle": {
        "success_threshold": (0.5, 1.0, "higher"),
        "max_steps": (5, 40, "lower"),
    },
    "intention": {
        "multi_detail_penalty": (0.0, 1.0, "higher"),
        "max_steps": (5, 40, "lower"),
    },
    # ColBench: the turn budget is the interview length, and the character cap
    # is how much the simulated human discloses per answer. Both throttle how
    # much hidden information the policy can buy before it has to commit.
    "colbench": {
        "max_steps": (3, 15, "lower"),
        "human_response_char_limit": (80, 800, "lower"),
        "min_questions": (0, 3, "higher"),
    },
    "persuade": {"max_steps": (5, 40, "lower")},
    "telepathy": {"max_steps": (5, 40, "lower")},
    "tau": {"max_steps": (5, 40, "lower")},
    # tau2 gets its own key rather than sharing "tau": the two gyms can be
    # configured in the same run, and a shared key would make a change aimed at
    # one silently move the other -- exactly the confound these knobs exist to
    # avoid. tau2 conversations are longer than tau1's (train.sh allows 50
    # turns), so the ceiling is higher.
    "tau2": {"max_steps": (5, 60, "lower")},
}

# env_manager's env_name -> the gym key used here and in the metrics.
ENV_NAME_TO_GYM = {
    "TravelGym": "travel",
    "ColBenchCodeEnv": "colbench",
    "FunctionGym": "function",
    "TurtleGym": "turtle",
    "IntentionGym": "intention",
    "PersuadeGym": "persuade",
    "TelepathyGym": "telepathy",
    "TauGym": "tau",
    "Tau2Gym": "tau2",
}


# ---------------------------------------------------------------------------
# USERRL_META_ARM -- which experimental arm this run belongs to.
#
# Reserved for the planned three-arm comparison (see
# InfoPO/reports/TODO_arms.md). The switch lives HERE, at the loader, rather
# than in the meta-agent's discipline, because "I will remember not to write
# the hot layer during this run" is not a control -- it is an intention, and
# this session already recorded what happens to controls that depend on
# somebody remembering.
#
#   "online" (default)  hot layers are re-read on every access. Current design:
#                       the meta-agent can change the environment MID-run.
#   "frozen"            hot layers are read ONCE, at first access, and never
#                       re-stat'ed. The arm may still configure everything
#                       BEFORE the run; mid-run writes are silently inert.
#   "off"               hot layers are never read at all; built-in defaults
#                       only. The closest thing to plain PPO on this stack.
#
# Read once at import: an arm that could change mid-run would defeat its own
# purpose.
_META_ARM = os.environ.get("USERRL_META_ARM", "online").strip().lower()


class _HotConfig:
    """Stat-on-access hot reload; keeps the last good value on any error."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._cfg: Dict[str, Any] = {}
        self._stamp = None

    def get(self) -> Dict[str, Any]:
        if _META_ARM == "off":
            return dict(_DEFAULTS)
        if _META_ARM == "frozen" and self._cfg is not None:
            return self._cfg
        with self._lock:
            try:
                st = os.stat(self._path)
                stamp = (st.st_mtime_ns, st.st_size)
            except OSError:
                return self._cfg
            if stamp == self._stamp:
                return self._cfg
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    new = json.load(f)
                if not isinstance(new, dict):
                    raise ValueError("difficulty config root must be an object")
            except Exception as e:  # noqa: BLE001
                print(f"[difficulty] ignoring unreadable {self._path}: {e}")
                return self._cfg
            self._cfg = new
            self._stamp = stamp
            print(f"[difficulty] loaded v{new.get('version')}: "
                  f"{ {k: v for k, v in new.items() if k != 'version' and not k.startswith('_')} }")
            return new


_CONFIG = _HotConfig(_DEFAULT_PATH)


def get_overrides(gym: str) -> Dict[str, Any]:
    """Validated difficulty overrides for one gym; {} when nothing applies.

    Re-validates against the whitelist on every read rather than trusting the
    file: the gate is the front door, this is the lock.
    """
    try:
        cfg = _CONFIG.get()
        raw = (cfg.get("gyms", {}) or {}).get(gym) or {}
        if not raw:
            return {}
    except Exception:  # noqa: BLE001
        return {}
    allowed = KNOWN_FIELDS.get(gym, {})
    out: Dict[str, Any] = {}
    for field, value in raw.items():
        spec = allowed.get(field)
        if spec is None:
            print(f"[difficulty] ignoring unknown field {gym}.{field}")
            continue
        lo, hi, _ = spec
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            print(f"[difficulty] ignoring non-numeric {gym}.{field}={value!r}")
            continue
        if not (lo <= value <= hi):
            print(f"[difficulty] ignoring out-of-range {gym}.{field}={value} (allowed [{lo},{hi}])")
            continue
        out[field] = type(lo)(value) if isinstance(lo, int) else float(value)
    return out


def apply_to_config(env_config: Any, env_name: str) -> Dict[str, Any]:
    """Apply overrides onto a gym config object. Returns what was applied."""
    gym = ENV_NAME_TO_GYM.get(env_name)
    if gym is None:
        return {}
    applied = {}
    for field, value in get_overrides(gym).items():
        if hasattr(env_config, field):
            setattr(env_config, field, value)
            applied[field] = value
    return applied
