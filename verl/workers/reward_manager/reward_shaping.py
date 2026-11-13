"""Bounded, hot-reloadable turn-level reward shaping for the interact gyms.

This is the single choke point where the per-turn env rewards assembled by the
reward manager can be rescaled / clipped from a JSON config, so the meta-agent
(UserDirector) can tune the dense reward signal within a bounded, validated
scope. The transform is *monotone* (non-negative scale + clip), so it never
changes the within-gym ordering of rewards and cannot introduce a new
reward-hacking channel.

Disabled by default (``enabled: false``) -> zero behavior change for any run
that does not explicitly opt in.

Config path: ``USERRL_REWARD_SHAPING_CONFIG`` env var, defaulting to
``meta_agent/reward_shaping.json`` under the InfoPO project root.
"""

import json
import os
import threading

_DEFAULT_PATH = os.environ.get(
    "USERRL_REWARD_SHAPING_CONFIG",
    os.path.join(os.getcwd(), "meta_agent", "reward_shaping.json"),
)

# data_source (non_tensor_batch key) -> short gym key used in the config.
_DATA_SOURCE_TO_GYM = {
    "interact_travelgym": "travel",
    "interact_taugym": "tau",
    "interact_turtlegym": "turtle",
    "interact_telepathygym": "telepathy",
    "interact_intention": "intention",
    "interact_persuade": "persuade",
    "interact_function": "function",
    "interact_searchgym": "search",
    "interact_colbench_code": "colbench",
    # tau2 ships one data_source per domain. All three map to the same gym key
    # so the flow/llm arms aggregate across domains -- without this the
    # per-strategy buckets were never filled and "is it the LLM user or the
    # symbolic one?" had no metric that could answer it.
    "interact_tau2_retail": "tau2",
    "interact_tau2_airline": "tau2",
    "interact_tau2_telecom": "tau2",
}

# Known gym keys (whitelist for the gate / config).
KNOWN_GYMS = [
    "travel", "tau", "turtle", "telepathy", "intention", "persuade", "function", "search",
    "colbench", "tau2",
]
# Known per-turn event (choice) types.
KNOWN_EVENTS = ["search", "action", "answer"]

_DEFAULT_CONFIG = {
    "version": 1,
    "enabled": False,
    "default_scale": 1.0,
    "per_gym_scale": {},      # gym -> scale in [0, 2]
    "event_weight": {},       # choice -> weight in [0, 2]
    "turn_reward_clip": [0.0, 1.0],  # [lo, hi], 0 <= lo < hi <= 2
}


def default_config() -> dict:
    return dict(_DEFAULT_CONFIG)


def gym_of(data_source) -> str:
    """Normalize a data_source string to its short gym key."""
    if not isinstance(data_source, str):
        return ""
    return _DATA_SOURCE_TO_GYM.get(data_source, data_source)


_META_ARM = os.environ.get("USERRL_META_ARM", "online").strip().lower()


class HotConfig:
    """Stat-on-access hot-reloadable JSON config (same discipline as the
    symbolic_layer ConfigLoader: atomic swap, keep last-good on failure)."""

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._cfg = None
        self._stamp = None

    def _stamp_now(self):
        st = os.stat(self._path)
        return (st.st_mtime_ns, st.st_size)

    def get(self) -> dict:
        if _META_ARM == "off":
            return default_config()
        if _META_ARM == "frozen" and self._cfg is not None:
            return self._cfg
        with self._lock:
            try:
                stamp = self._stamp_now()
            except OSError:
                # missing/unreadable file -> disabled default (or last good)
                return self._cfg if self._cfg is not None else default_config()
            if stamp == self._stamp and self._cfg is not None:
                return self._cfg
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    new = json.load(f)
                if not isinstance(new, dict):
                    raise ValueError("reward shaping config root must be an object")
            except Exception:
                return self._cfg if self._cfg is not None else default_config()
            self._cfg = new
            self._stamp = stamp
            return self._cfg


_CONFIG = HotConfig()


def get_config() -> dict:
    """Current shaping config (hot-reloaded). Disabled-default if absent."""
    return _CONFIG.get()


def shape_turn_reward(reward, choice, data_source, cfg) -> float:
    """Shape one turn's reward. Monotone: non-negative scale + clip."""
    if not cfg.get("enabled", False):
        return float(reward)
    gym = gym_of(data_source)
    scale = float(cfg.get("per_gym_scale", {}).get(gym, cfg.get("default_scale", 1.0)))
    eweight = float(cfg.get("event_weight", {}).get(choice, 1.0))
    r = float(reward) * scale * eweight
    clip = cfg.get("turn_reward_clip", [0.0, 1.0])
    lo, hi = float(clip[0]), float(clip[1])
    return min(max(r, lo), hi)


def apply_reward_shaping(turn_rewards, turn_choices, data_source, cfg=None):
    """Shape a list of per-turn rewards. No-op (returns a copy) when disabled."""
    if cfg is None:
        cfg = get_config()
    if not cfg.get("enabled", False):
        return [float(r) for r in turn_rewards]
    choices = list(turn_choices) + [None] * (len(turn_rewards) - len(turn_choices))
    return [
        shape_turn_reward(r, c, data_source, cfg)
        for r, c in zip(turn_rewards, choices)
    ]
