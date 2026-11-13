"""Hot-reloadable process-reward weights shared by the gyms.

Same discipline as ``env_policy`` / ``difficulty``: the *predicate logic* lives
in the gym and is never edited at runtime; only the *weight* is hot-tunable,
and every weight defaults to 0.0 so an empty config reproduces the previous
behaviour exactly.

``repeat_answer`` (UG24, from trajectory reading)
------------------------------------------------
Re-submitting an answer used to be free in every gym, so the policy learned to
hammer the answer channel until the turn budget ran out. Measured on UG19's
dumps, the reward a submission actually returns by attempt index:

    turtle    1st 0.642  2nd 0.148  3rd 0.055  4th 0.038  5th 0.056  6th+ 0.000 (n=81)
    function  1st 0.114  2nd 0.108  3rd 0.034  4th 0.027  5th 0.023  6th+ 0.011 (n=190)

From the third attempt on the channel is essentially dead, yet it still costs a
turn *and* an LLM judge call. ``repeat_answer`` prices attempts beyond
``FREE_ANSWER_ATTEMPTS`` so those turns are no longer free.

This is deliberately *not* ``step_penalty`` / ``max_steps``: both of those charge
questions and answers alike, and cutting them removes legitimate exploration.
"""

import json
import os
import threading
from typing import Any, Dict

_PATH = os.environ.get("USERRL_PROCESS_REWARD_CONFIG", os.path.join(os.path.dirname(__file__), "process_reward.json"))

# The first two submissions are the ones that still pay (see the table above),
# so they stay free. Fixed rather than hot-tunable: the gate validates weights
# in [-0.5, 0.5] and an attempt *count* would not fit that invariant.
FREE_ANSWER_ATTEMPTS = 2


# USERRL_META_ARM —— 本 run 属于哪个对照臂（与 env_policy.py / difficulty.py /
# config_loader.py 同一语义，见 reports/TODO_arms.md）：
#   online（默认）：每次访问 stat + 必要时重读。与改动前逐字节等价。
#   frozen        ：首次访问读一次，此后不再 stat。中途写入静默失效。
#   off           ：完全不读 JSON，返回空字典 —— tau/turtle/persuade/function
#                   的全部过程奖励权重回落到 0，即 gym 原生奖励。
# 在 import 时读一次而不是每次读：一个能中途改变的臂会毁掉它自己的目的。
_META_ARM = os.environ.get("USERRL_META_ARM", "online").strip().lower()


class _HotJson:
    """Stat-on-access hot reload; keeps the last good value on any error."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._cfg: Dict[str, Any] = {}
        self._stamp = None

    def get(self) -> Dict[str, Any]:
        if _META_ARM == "off":
            return {}
        if _META_ARM == "frozen" and self._stamp is not None:
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
                    raise ValueError("process_reward config root must be an object")
            except Exception:  # noqa: BLE001
                return self._cfg
            self._cfg, self._stamp = new, stamp
            return new


_CFG = _HotJson(_PATH)


def weights(gym_key: str) -> Dict[str, float]:
    section = _CFG.get().get(gym_key) or {}
    out: Dict[str, float] = {}
    for k, v in section.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def repeat_answer_penalty(gym_key: str, attempt_index: int) -> float:
    """Non-negative amount to subtract for the ``attempt_index``-th (1-based)
    answer submission of an episode. 0.0 unless a weight is configured."""
    if attempt_index <= FREE_ANSWER_ATTEMPTS:
        return 0.0
    w = weights(gym_key).get("repeat_answer", 0.0)
    return -w if w < 0 else 0.0
