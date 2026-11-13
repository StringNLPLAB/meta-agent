"""Update-boundary snapshots for the standalone Tau2 experiment.

Editors write drafts. Training consumers read immutable Python bundles and
active scalar files published only before a rollout batch. Integrity checks
are not a correctness proof or an OS sandbox for editable Python.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path

from meta_agent.audit_evidence import build_training_context
from meta_agent.nonblocking_audit import _atomic_json, training_metrics
from meta_agent.tau_coser_bootstrap import bootstrap_bundle

WEIGHTS = {"first_tool_ok": 0.02, "first_grounded_user_id": 0.07, "first_grounded_object_id": 0.09,
           "tool_call_ok": 0.0, "tool_call_error": 0.0, "repeat_failed_call": 0.0}
ENVIRONMENT = '''"""Training-only dynamics hooks; native evaluation is external."""
class EnvironmentMechanism:
    def __init__(self, settings): self.settings = settings
    def on_reset(self, env): return None
    def before_action(self, env, choice, content): return choice, content
    def after_action(self, env, observation, reward, terminated, truncated, info):
        return observation, reward, terminated, truncated, info
'''


def write(path, value):
    _atomic_json(Path(path), value)


def initialize(run, *, native_reward=False, experience=None):
    if experience and Path(experience).suffix not in {".md", ".txt"}:
        raise ValueError("Experience must be a selected prose document")
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    for name in ("drafts/user", "drafts/environment", "drafts/agent", "live", "active", "logs",
                 "training_rollouts", "training_samples", "uptake", "terminal_action_scores"):
        (run / name).mkdir(parents=True)
    for name, raw in bootstrap_bundle().items():
        (run / "drafts/user" / name).write_bytes(raw)
    (run / "drafts/environment/environment.py").write_text(ENVIRONMENT)
    write(run / "drafts/environment/environment_config.json",
          dict(version=0, process_reward={} if native_reward else WEIGHTS, max_steps=50))
    for name in ("entropy_schedule", "clip_schedule"):
        write(run / f"drafts/agent/{name}.json", dict(enable=False))
    (run / "experience.md").write_text(Path(experience).read_text() if experience else
        "# Experience\n\nFresh run. No previous executable configuration, checkpoint or agent session was inherited.\n")
    (run / "logs/training.log").touch()
    (run / "metrics.jsonl").touch()
    (run / "trajectory_excerpt.txt").touch()
    refresh_context(run)
    pin_update(run, step=0)


def _read_drafts(root):
    values = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not (p.is_dir() or p.is_file()):
            raise ValueError("Drafts must contain regular files/directories")
        if not p.is_file():
            continue
        if p.suffix in {".pending", ".tmp"} or p.name.endswith("~"):
            continue
        name = p.relative_to(root).as_posix()
        if p.suffix not in {".py", ".json", ".txt", ".jinja"} or p.stat().st_size > 1024 * 1024:
            raise ValueError("Unsupported or oversized draft: " + name)
        raw = p.read_bytes()
        if p.suffix == ".py":
            ast.parse(raw, filename=name)
        elif p.suffix == ".json":
            json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        values[name] = raw
        if len(values) > 128 or sum(map(len, values.values())) > 8 * 1024 * 1024:
            raise ValueError("Draft snapshot exceeds its file/byte budget")
    return values


def _bundle(run, group, version, values):
    dirname, entry = ("versions", "flow_user.py") if group == "user" else ("environment_versions", "environment.py")
    destination = run / dirname / "versions" / f"v{version:06d}"
    destination.mkdir(parents=True)
    files = {}
    for name, raw in values.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files[name] = hashlib.sha256(raw).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(schema_version=1, run=run.name, version=version, snapshot=str(destination), files=files,
                bundle_sha256=digest, entrypoint=entry)


def pin_update(run, *, step):
    run = Path(run).resolve(strict=True)
    previous_path = run / "active/current.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    publishing = False
    # Two equal bounded reads catch concurrent multi-file saves; rejected drafts
    # leave the previous complete update configuration active.
    try:
        first = _read_drafts(run / "drafts")
        if first != _read_drafts(run / "drafts"):
            raise ValueError("Draft changed during the snapshot")
        required = {"user/flow_user.py", "user/flow.py", "user/env_policy.json",
                    "environment/environment.py", "environment/environment_config.json",
                    "agent/entropy_schedule.json", "agent/clip_schedule.json"}
        if not required <= first.keys():
            raise ValueError("A required live interface file is missing")
        policy = json.loads(first["user/env_policy.json"])
        probability = policy.get("tau2_flow", {}).get("flow_probability", 0)
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError("flow_probability must be in [0,1]")
        settings = json.loads(first["environment/environment_config.json"])
        weights = settings.get("process_reward", {})
        if not isinstance(weights, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or abs(v) > 0.5 for v in weights.values()):
            raise ValueError("Process weights must be finite in [-0.5,0.5]")
        if type(settings.get("max_steps")) is not int or not 1 <= settings["max_steps"] <= 50:
            raise ValueError("Training max_steps must be an integer in [1,50]")
        for name in ("entropy_schedule", "clip_schedule"):
            scalar = json.loads(first[f"agent/{name}.json"])
            if not isinstance(scalar, dict) or type(scalar.get("enable")) is not bool:
                raise ValueError("A schedule needs a boolean enable field")
            for field in ("value", "override", "init", "final", "max_delta_per_step"):
                if field in scalar and (isinstance(scalar[field], bool) or not isinstance(scalar[field], (int, float))
                                        or not math.isfinite(scalar[field])):
                    raise ValueError("Schedule scalars must be finite")
        digest = hashlib.sha256(b"".join(name.encode() + b"\0" + raw for name, raw in sorted(first.items()))).hexdigest()
        if previous and previous["draft_sha256"] == digest:
            return previous
        version = previous["version"] + 1 if previous else 0
        for scope in ("versions/versions", "environment_versions/versions"):
            versions = [int(p.name[1:]) for p in (run / scope).glob("v[0-9][0-9][0-9][0-9][0-9][0-9]")]
            if versions:
                version = max(version, max(versions) + 1)
        users = {k.removeprefix("user/"): v for k, v in first.items() if k.startswith("user/")}
        # The public user loader requires policy.version == bundle.version.
        policy["version"] = version
        users["env_policy.json"] = json.dumps(policy).encode()
        environments = {k.removeprefix("environment/"): v for k, v in first.items() if k.startswith("environment/")}
        settings["version"] = version
        environments["environment_config.json"] = json.dumps(settings).encode()
        user = _bundle(run, "user", version, users)
        environment = _bundle(run, "environment", version, environments)
        # Import and ABI-check captured modules before any live pointer changes.
        from tau2gym.env.usersim_runtime import _factory as user_factory
        from tau2gym.env.environment_runtime import _factory as environment_factory
        user_factory(user["bundle_sha256"], Path(user["snapshot"]), users)
        environment_factory(environment["bundle_sha256"], Path(environment["snapshot"]), environments)
        publishing = True
        write(run / "live/usersim_bundle.json", user)
        write(run / "live/environment_bundle.json", environment)
        for name in ("entropy_schedule", "clip_schedule"):
            write(run / f"active/{name}.json", json.loads(first[f"agent/{name}.json"]))
        write(run / "active/env_policy.json", policy)
        write(run / "active/process_reward.json", dict(version=version, tau2=weights))
        receipt = dict(step=step, version=version, draft_sha256=digest,
                       user_bundle_sha256=user["bundle_sha256"], environment_bundle_sha256=environment["bundle_sha256"],
                       worker_uptake="See per-episode uptake records; publication alone is not uptake")
        write(previous_path, receipt)
        with (run / "pinned_updates.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt) + "\n")
        return receipt
    except Exception as error:
        with (run / "rejected_drafts.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, error_type=type(error).__name__, error=str(error)[:1000])) + "\n")
        if previous and not publishing:
            return previous
        if previous:
            raise RuntimeError("Publication failed after live writes began; inspect rejected_drafts before continuing") from error
        raise


def refresh_context(run, metrics=None, step=None):
    run = Path(run)
    if metrics is not None:
        with (run / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, metrics=training_metrics(metrics)), allow_nan=False) + "\n")
    samples = sorted((run / "training_rollouts").glob("*.jsonl"), key=lambda p: p.stat().st_mtime_ns)
    if samples:
        with samples[-1].open("rb") as stream:
            size = samples[-1].stat().st_size
            stream.seek(max(0, size - 65536))
            excerpt = stream.read(65536)
        (run / "trajectory_excerpt.txt").write_bytes(excerpt)
    context = build_training_context(root=run, run=run.name, logs=["logs/training.log"],
        trajectories=["trajectory_excerpt.txt"], metrics=["metrics.jsonl"], experience=["experience.md"])
    context["editable_roots"] = [str(run / "drafts")]
    context["instructions"] = ("Use training evidence only. Edit drafts directly; write atomically. Explain changes in natural language. "
        "Each invocation is a fresh process/session; do not resume a previous session. Previous run knowledge is prose only. "
        "Interfaces: user/*.py and env_policy.json; environment/environment.py and environment_config.json; agent/*_schedule.json. "
        "Changes are syntax/schema checked and pinned before a future rollout batch. Native held-out evaluation is separate. "
        "Do not infer causal learning gains from successful publication. Update experience.md with concise training-only lessons.")
    write(run / "audit_context.json", context)
