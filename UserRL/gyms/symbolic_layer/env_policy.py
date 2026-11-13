"""Hot-reloadable environment-policy knobs.

These are the environment *semantics* the meta-agent may retune mid-run, as
opposed to the reward scaling in ``reward_shaping.json`` or the predicate
weights in ``process_reward.json``. Same discipline as those: stat-on-access,
last-good kept on any error, and a disabled-by-default shape so a missing file
means "behave exactly as before".

Config path: ``USERRL_ENV_POLICY_CONFIG`` env var, defaulting to
``symbolic_layer/env_policy.json``.

Knobs
-----
``persuade.stance_ratchet`` (bool, default False)
    Forbid the persuadee's stance from sliding back toward agreement. The
    reward pays for forward movement and charges nothing for backing up, so a
    judge whose stance wavers hands out the same progress repeatedly: 31% of
    CoSER-judged episodes scored above 1.0, the value of pushing the full
    7-level scale exactly once. A ratchet removes that without making the
    reward sparser (unlike potential-based shaping, which would).

``user_persona`` (dict gym -> {enabled, personas:[{name, weight, text}]}, default {})
    A character sketch appended to the user simulator's system prompt. Every
    simulated user currently opens with "You are a helpful user ..." -- one
    persona for the whole dataset, so the policy only ever practises against a
    cooperative, indirect interlocutor. Drawing a persona per task widens that
    without touching the task content or the reward.

    Applies ONLY to the LLM-backed reply paths. TravelGym answers two branches
    from hard-coded strings (the "I do not have specific preference" and "your
    question is too vague" replies); no prompt knob reaches those.

    The draw is seeded by the task id, so a task always gets the same persona.
    Absent/disabled = the prompt is untouched.

``colbench_flow`` (dict, see _COLBENCH_FLOW_DEFAULTS)
    Knobs for ColBench's symbolic user. ``facts_per_reply`` is how many
    requirements one answer discloses; ``match_threshold`` is how well the
    question must match before anything is disclosed; ``volunteer_interval`` is
    how many off-target turns before the user offers something unprompted (0 =
    never); ``max_reveals`` caps total disclosures per episode (0 = no cap);
    ``reveal_return_shape`` controls whether the free "it returns a dict with
    keys X, Y" fact is available at all.

``user_strategy_mix`` (dict gym -> float in [0,1], default {})
    Per-gym probability of drawing the *symbolic* ``flow`` user simulator
    instead of the plain ``llm`` one. Training entirely on ``flow`` teaches the
    policy to satisfy a keyword matcher rather than to ask naturally: on
    travel22 the flow user credits 2.46 preference hits per episode where the
    llm user credits 0.86. Mixing forces both to be satisfied. 0.0 = always
    llm, 1.0 = always flow (the pre-existing behaviour), absent = fall back to
    USERRL_USER_STRATEGY.

    The draw is seeded by the task id, so a given task always lands on the same
    simulator -- mixed across the dataset, reproducible within it.
"""

import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional

_DEFAULT_PATH = os.environ.get(
    "USERRL_ENV_POLICY_CONFIG",
    os.path.join(os.path.dirname(__file__), "env_policy.json"),
)

# ColBench's flow user is fully symbolic -- it reads facts off the reference
# solution and renders them from templates -- so every one of these is a direct,
# monotone lever on how much the agent can learn per question. The defaults
# reproduce a cooperative user: one fact per answer, answered whenever the
# question plausibly refers to it, and a nudge after two off-target turns.
_COLBENCH_FLOW_DEFAULTS: Dict[str, Any] = {
    "facts_per_reply": 1,
    "match_threshold": 0.34,
    "volunteer_interval": 2,
    "max_reveals": 0,
    "reveal_return_shape": True,
}

# Cross-gym price on re-asking. penalty=0 reproduces pre-UG20 behaviour.
# similarity is the difflib ratio above which two [action] messages count as the
# same question; window is how many previous questions are compared.
_REPETITION_DEFAULTS: Dict[str, Any] = {
    "penalty": 0.02,
    "similarity": 0.85,
    "window": 4,
    # The penalty was calibrated on natural-language re-asking (UG19 question
    # texts averaged 47-225 characters). FunctionGym's [action] is an eight-char
    # probe like "7 8 9 10", where changing ONE input -- the single move that
    # buys information -- scores 0.875 character similarity and was being
    # charged as a repeat, turning a correct probe's 0.0152 into -0.0048.
    # min_chars keeps the rule on the distribution it was measured on; the
    # exclude list is the explicit escape hatch for gyms whose action channel is
    # not prose at all.
    "min_chars": 40,
    "exclude_gyms": ["FunctionGym"],
}

# tau2's flow user reads the benchmark's own structured user brief
# (known_info / unknown_info / reason_for_call), so these knobs control
# disclosure without touching what the task *is*. Field names deliberately
# mirror _COLBENCH_FLOW_DEFAULTS: the same experiment run on both gyms should
# be describable with the same words. ``guide_interval`` is tau2's analogue of
# colbench's ``volunteer_interval``; ``reveal_identifiers`` decides whether
# exact order/reservation/phone ids are handed over or must be looked up.
_TAU2_FLOW_DEFAULTS: Dict[str, Any] = {
    "facts_per_reply": 1,
    "match_threshold": 0.34,
    "guide_interval": 2,
    "max_reveals": 0,
    "reveal_identifiers": True,
    "tool_threshold": 0.30,
}


_DEFAULTS: Dict[str, Any] = {
    "version": 0,
    "persuade": {"stance_ratchet": False, "max_stance_step": 1},
    "turtle": {"require_explicit_coverage": True, "coverage_min_shared_words": 1},
    "repetition": dict(_REPETITION_DEFAULTS),
    "user_strategy_mix": {},
    "user_persona": {},
    "colbench_flow": dict(_COLBENCH_FLOW_DEFAULTS),
    "tau2_flow": dict(_TAU2_FLOW_DEFAULTS),
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
        self._cfg: Optional[Dict[str, Any]] = None
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
                return self._cfg if self._cfg is not None else dict(_DEFAULTS)
            if stamp == self._stamp and self._cfg is not None:
                return self._cfg
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    new = json.load(f)
                if not isinstance(new, dict):
                    raise ValueError("env policy config root must be an object")
            except Exception as e:  # noqa: BLE001
                print(f"[env_policy] ignoring unreadable {self._path}: {e}")
                return self._cfg if self._cfg is not None else dict(_DEFAULTS)
            merged = dict(_DEFAULTS)
            merged.update(new)
            self._cfg = merged
            self._stamp = stamp
            print(f"[env_policy] loaded v{merged.get('version')}: "
                  f"stance_ratchet={merged.get('persuade', {}).get('stance_ratchet')} "
                  f"user_strategy_mix={merged.get('user_strategy_mix')}")
            return merged


_CONFIG = _HotConfig(_DEFAULT_PATH)


def get_env_policy() -> Dict[str, Any]:
    return _CONFIG.get()


def get_colbench_flow() -> Dict[str, Any]:
    """Flow-user knobs, with every missing field filled from the defaults.

    Merging against the defaults matters: a patch that sets one field must not
    leave the others undefined, or the FlowUser would silently fall back to its
    own in-code defaults and the config would stop describing the environment.
    """
    out = dict(_COLBENCH_FLOW_DEFAULTS)
    try:
        cfg = get_env_policy().get("colbench_flow") or {}
        if isinstance(cfg, dict):
            out.update({k: v for k, v in cfg.items() if k in _COLBENCH_FLOW_DEFAULTS})
    except Exception:  # noqa: BLE001
        pass
    return out


def get_tau2_flow() -> Dict[str, Any]:
    """Flow-user knobs for tau2, missing fields filled from the defaults.

    Same reason as get_colbench_flow: a patch that sets one field must not leave
    the rest undefined, or the user simulator falls back to its own in-code
    defaults and the config stops describing the environment.
    """
    out = dict(_TAU2_FLOW_DEFAULTS)
    try:
        cfg = get_env_policy().get("tau2_flow") or {}
        if isinstance(cfg, dict):
            out.update({k: v for k, v in cfg.items() if k in _TAU2_FLOW_DEFAULTS})
    except Exception:  # noqa: BLE001
        pass
    return out


def stance_ratchet_enabled() -> bool:
    try:
        return bool(get_env_policy().get("persuade", {}).get("stance_ratchet", False))
    except Exception:  # noqa: BLE001
        return False


def persuade_max_stance_step() -> int:
    """Max stance levels the persuadee may move in ONE exchange (default 1).

    The 7-level scale starts at "Strongly Agree" and the reward is
    ``levels_changed / 6``.  Measured on 24 episodes, the judge moved **5 of 6
    levels on the very first argument** in 17 of them (23/24 moved >=4), so the
    first turn carried 88% of the episode reward and the remaining ~15 turns
    carried 12%.  A cap makes the multi-turn structure mean something; 0 or a
    negative value disables it.
    """
    try:
        v = int(get_env_policy().get("persuade", {}).get("max_stance_step", 1))
        return v
    except Exception:  # noqa: BLE001
        return 1


def turtle_require_explicit_coverage() -> bool:
    """Make the TurtleGym judge score each criterion against the USER'S text.

    Measured over 163 criteria the judge marked "Covered" across three runs,
    **17% shared not one distinctive word with the answer** and 34% shared <=20%
    -- e.g. "The farmer uses the phrase 'This one isn't ripe' while tapping on
    the heads" scored as covered by a story that mentions no farmer, no ripeness,
    no tapping and no heads.  The judge was reasoning holistically ("is this
    story roughly the ground truth?") while being asked for a per-criterion
    score, so a vague answer collected the same reward as a precise one.
    """
    try:
        return bool(get_env_policy().get("turtle", {}).get("require_explicit_coverage", True))
    except Exception:  # noqa: BLE001
        return True



def repetition_knobs() -> Dict[str, Any]:
    """Hot knobs for the cross-gym repeated-question price.

    Read by ``verl/tools/env_manager.py::RepetitionPricingWrapper``. Repeating a
    question was free in UG19 and the policy leaned on it hardest exactly where it
    was least useful -- 46% near-duplicate questions on IntentionGym, 20% on
    TelepathyGym, against 10-17% on the trained gyms.
    """
    cfg = dict(_REPETITION_DEFAULTS)
    try:
        cfg.update(get_env_policy().get("repetition") or {})
    except Exception:  # noqa: BLE001
        pass
    return cfg


def turtle_coverage_min_shared_words() -> int:
    """How many content words a criterion must share with the user's answer.

    0 disables the check (pre-UG20 behaviour: the judge's score stands as given).
    1 -- the UG20 default -- means a criterion scored above 0 must share at least
    one five-plus-character content word with what the user actually wrote. See
    ``TurtleGym/env/prompts.py::_coverage_verified`` for why the prompt-level rule
    was not enough. Hot-reloadable so the strictness can be backed off mid-run if
    it turns out to zero genuine paraphrases.
    """
    cfg = get_env_policy().get("turtle") or {}
    try:
        return max(0, int(cfg.get("coverage_min_shared_words", 1)))
    except Exception:  # noqa: BLE001
        return 1

def resolve_persona(gym: str, task_id: Any) -> str:
    """Persona text for one episode, or "" when the knob is off.

    Deterministic in ``task_id`` like resolve_user_strategy, so the persona is a
    property of the task rather than a source of run-to-run variance.
    """
    try:
        cfg = (get_env_policy().get("user_persona", {}) or {}).get(gym) or {}
        if not cfg.get("enabled", False):
            return ""
        personas = [p for p in (cfg.get("personas") or []) if p.get("text")]
        if not personas:
            return ""
    except Exception:  # noqa: BLE001
        return ""
    weights = [max(float(p.get("weight", 1.0)), 0.0) for p in personas]
    total = sum(weights)
    if total <= 0:
        return ""
    digest = hashlib.sha256(f"persona:{gym}:{task_id}".encode("utf-8")).hexdigest()
    draw = (int(digest[:8], 16) / 0xFFFFFFFF) * total
    acc = 0.0
    for p, w in zip(personas, weights):
        acc += w
        if draw <= acc:
            return str(p["text"]).strip()
    return str(personas[-1]["text"]).strip()


def apply_persona(system_prompt: str, gym: str, task_id: Any) -> str:
    """Append the drawn persona to a user-simulator system prompt."""
    text = resolve_persona(gym, task_id)
    if not text:
        return system_prompt
    return f"{system_prompt}\n\n## **Your Character**\n{text}"


def content_task_id(text: Any) -> str:
    """Stable task id derived from the task statement.

    Some datasets ship no per-task id in ``create_kwargs`` -- ColBench is one.
    Passing None into resolve_user_strategy hashes a *constant*, so every
    episode of the run draws the same simulator and a mix of 0.5 silently
    becomes all-llm rather than half. The problem statement is the one field
    both the environment and the reward manager can see, so deriving the id
    from it keeps the two sides agreeing on which simulator an episode faced.
    """
    return "sha:" + hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def resolve_user_strategy(gym: str, task_id: Any, fallback: str) -> str:
    """Pick ``flow`` or ``llm`` for one episode.

    Deterministic in ``task_id`` so a task always draws the same simulator --
    the mix is across the dataset, never a source of run-to-run variance.
    """
    try:
        mix = get_env_policy().get("user_strategy_mix", {}) or {}
        if gym not in mix:
            return fallback
        p = float(mix[gym])
    except Exception:  # noqa: BLE001
        return fallback
    if p <= 0.0:
        return "llm"
    if p >= 1.0:
        return "flow"
    digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()
    draw = int(digest[:8], 16) / 0xFFFFFFFF
    return "flow" if draw < p else "llm"
