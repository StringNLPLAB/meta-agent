"""Training-only ColBench drafts, update snapshots and episode uptake.

The editor writes ordinary files. The trainer publishes a checked snapshot at
an update boundary; each new episode retains that snapshot for its lifetime.
Python validation checks syntax and the hook ABI, not semantic correctness or
filesystem confinement. Deploy the editor in a separate container for isolation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import types

from meta_agent.tau2_runtime import _read_drafts, write, refresh_context as _refresh_context

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = '''"""Editable ColBench training dynamics and reward hooks."""
class EnvironmentMechanism:
    def __init__(self, settings): self.settings = settings
    def on_reset(self, env): return None
    def before_action(self, env, action): return action
    def after_action(self, env, observation, reward, terminated, truncated, info):
        return observation, reward, terminated, truncated, info
    def terminal_reward(self, env, native_reward): return native_reward
'''
REQUIRED = {"user/flow_user.py", "user/env_policy.json",
            "environment/environment.py", "environment/settings.json",
            "agent/entropy_schedule.json", "agent/clip_schedule.json"}
_CACHE = {}


def initialize(run, *, experience=None):
    if experience and Path(experience).suffix not in {".md", ".txt"}:
        raise ValueError("Experience must be a selected prose document")
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    for name in ("drafts/user", "drafts/environment", "drafts/agent", "live", "active",
                 "logs", "training_rollouts", "training_samples", "uptake"):
        (run / name).mkdir(parents=True)
    (run / "drafts/user/flow_user.py").write_bytes(
        (ROOT / "gyms/ColBenchGym/colbenchgym/env/flow_user.py").read_bytes())
    write(run / "drafts/user/env_policy.json", {
        "version": 0, "user_strategy_mix": {"colbench": 0.5},
        "colbench_flow": {"facts_per_reply": 1, "match_threshold": 0.34,
                          "volunteer_interval": 2, "max_reveals": 0, "reveal_return_shape": True},
        "repetition": {"penalty": 0.0}})
    (run / "drafts/environment/environment.py").write_text(ENVIRONMENT)
    write(run / "drafts/environment/settings.json", dict(max_steps=10,
        human_response_char_limit=400, min_questions=0, reward_scale=1.0, step_penalty=0.0))
    for name in ("entropy_schedule", "clip_schedule"):
        write(run / f"drafts/agent/{name}.json", dict(enable=False))
    (run / "experience.md").write_text(Path(experience).read_text() if experience else
        "# Experience\n\nFresh policy, configuration and editor sessions. No prior executable state inherited.\n")
    for name in ("logs/training.log", "metrics.jsonl", "trajectory_excerpt.txt"):
        (run / name).touch()
    write(run / "active/disabled.json", dict(enable=False))
    pin_update(run, step=0)
    refresh_context(run)


def _finite(value, lower, upper):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and lower <= value <= upper)


def _validate(files):
    if not REQUIRED <= files.keys():
        raise ValueError("Required ColBench draft interface is missing")
    policy = json.loads(files["user/env_policy.json"])
    if not _finite(policy.get("user_strategy_mix", {}).get("colbench"), 0, 1):
        raise ValueError("ColBench flow probability must be in [0,1]")
    flow = policy.get("colbench_flow", {})
    for key, low, high in (("facts_per_reply", 1, 8), ("volunteer_interval", 0, 100), ("max_reveals", 0, 100)):
        if key in flow and (type(flow[key]) is not int or not low <= flow[key] <= high):
            raise ValueError("Invalid flow integer: " + key)
    if "match_threshold" in flow and not _finite(flow["match_threshold"], 0, 1):
        raise ValueError("Invalid flow match threshold")
    if "reveal_return_shape" in flow and type(flow["reveal_return_shape"]) is not bool:
        raise ValueError("reveal_return_shape must be boolean")
    settings = json.loads(files["environment/settings.json"])
    for key, low, high in (("max_steps", 1, 10), ("human_response_char_limit", 1, 2000), ("min_questions", 0, 9)):
        if type(settings.get(key)) is not int or not low <= settings[key] <= high:
            raise ValueError("Invalid environment integer: " + key)
    if settings["min_questions"] >= settings["max_steps"]:
        raise ValueError("The turn budget must allow an answer after required questions")
    for key, low, high in (("reward_scale", 0.01, 10), ("step_penalty", 0, 1)):
        if not _finite(settings.get(key), low, high):
            raise ValueError("Invalid reward scalar: " + key)
    for name, low, high in (("entropy_schedule", 0, .05), ("clip_schedule", .01, .5)):
        config = json.loads(files[f"agent/{name}.json"])
        if type(config.get("enable")) is not bool:
            raise ValueError("A schedule needs a boolean enable field")
        for key in ("value", "override", "init", "final"):
            if key in config and not _finite(config[key], low, high):
                raise ValueError("Invalid schedule scalar: " + key)
        if "steps" in config and (type(config["steps"]) is not int or config["steps"] < 1):
            raise ValueError("Schedule steps must be a positive integer")
        if "max_delta_per_step" in config and not _finite(config["max_delta_per_step"], 0, .5):
            raise ValueError("Invalid schedule ramp")
    return policy, settings


def _factories(files, digest):
    if digest not in _CACHE:
        loaded = []
        for name, classname, methods in (
                ("user/flow_user.py", "FlowUser", ("reply",)),
                ("environment/environment.py", "EnvironmentMechanism",
                 ("on_reset", "before_action", "after_action", "terminal_reward"))):
            module = types.ModuleType("_colbench_" + digest + classname)
            module.__file__ = name
            exec(compile(files[name], name, "exec"), module.__dict__)
            factory = getattr(module, classname, None)
            if not callable(factory) or any(not callable(getattr(factory, method, None)) for method in methods):
                raise ValueError("Editable module has an incompatible ABI: " + name)
            loaded.append(factory)
        _CACHE[digest] = tuple(loaded)
    return _CACHE[digest]


def pin_update(run, *, step):
    run = Path(run).resolve(strict=True)
    pointer = run / "live/snapshot.json"
    previous = json.loads(pointer.read_text()) if pointer.exists() else None
    try:
        files = _read_drafts(run / "drafts")
        if files != _read_drafts(run / "drafts"):
            raise ValueError("Draft changed during snapshot")
        policy, _ = _validate(files)
        hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        if previous and previous["sha256"] == digest:
            return previous
        _factories(files, digest)
    except Exception as error:
        with (run / "rejected_drafts.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, error_type=type(error).__name__, error=str(error)[:1000])) + "\n")
        if previous:
            return previous
        raise
    # Fail closed on a filesystem failure after validation. Do not continue with
    # a mixture of old and new active scalars after an incomplete publication.
    version = previous["version"] + 1 if previous else 0
    while (run / f"snapshots/v{version:06d}").exists():
        version += 1
    snapshot = run / f"snapshots/v{version:06d}"
    snapshot.mkdir(parents=True)
    for name, raw in files.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    policy["version"] = version
    write(run / "active/env_policy.json", policy)
    for name in ("entropy_schedule", "clip_schedule"):
        write(run / f"active/{name}.json", json.loads(files[f"agent/{name}.json"]))
    record = dict(step=step, version=version, sha256=digest, snapshot=str(snapshot), files=hashes,
                  worker_uptake="See uptake/consumption.jsonl; publication is not evidence of a gain")
    write(pointer, record)
    with (run / "pinned_updates.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def load_episode(run):
    run = Path(run).resolve(strict=True)
    receipt = json.loads((run / "live/snapshot.json").read_text())
    snapshot = Path(receipt["snapshot"]).resolve(strict=True)
    if not snapshot.is_relative_to(run / "snapshots"):
        raise ValueError("Episode snapshot is outside the run")
    files = _read_drafts(snapshot)
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if hashes != receipt["files"] or digest != receipt["sha256"]:
        raise ValueError("Published ColBench snapshot changed")
    policy, settings = _validate(files)
    flow, environment = _factories(files, digest)
    return flow, environment(settings), policy, settings, receipt


def refresh_context(run, metrics=None, step=None):
    _refresh_context(run, metrics, step)
    path = Path(run) / "audit_context.json"
    context = json.loads(path.read_text())
    context["instructions"] = (
        "Review training logs, trajectories, metrics and experience only. Each invocation is a fresh session. "
        "Edit drafts directly and atomically; explain changes in natural language. "
        "User: user/flow_user.py and env_policy.json (flow probability and disclosure knobs). "
        "Environment: environment/environment.py and settings.json (dynamics, reward scale and step penalty). "
        "Agent: agent/entropy_schedule.json and clip_schedule.json. "
        "Checked snapshots take effect at a future update boundary and are retained for each episode. "
        "Record proposed mechanisms and observed outcomes separately; a published edit is not a demonstrated gain. "
        "Update experience.md with concise training-only lessons. Do not access held-out tasks or evaluation results. "
        "Previous rounds transfer prose experience, never executable configuration, weights or session state.")
    write(path, context)
