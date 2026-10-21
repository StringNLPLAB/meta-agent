#!/usr/bin/env python3
"""Package recorded supervision requests for Harbor's viewer, without running a job.

The runtime exporter has no Harbor dependency. The JSON shapes here target the
public JobConfig/JobResult/TrialConfig/TrialResult models in Harbor 0.24.0.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid


MARKER = "meta-agent-bundle.json"
FORMAT = "meta-agent-harbor-bundle-v1"
SOURCE = "imported-supervision-trace"
NAMESPACE = uuid.UUID("f9e046bd-3a05-4a60-8f75-21c56ee8263b")
ARTIFACTS = (
    "request.json", "export.json", "response.json", "prompt.txt",
    "agent.stdout", "agent.stderr", "operation.json", "receipt.json",
    "executor.stdout", "executor.stderr", "native/capture.json",
)
NOTICE = (
    "Imported supervision trace. This directory is for Harbor viewing only; "
    "it is not a runnable Harbor task or job. Completed means the recorded "
    "meta-agent pass completed, not that training succeeded or an operation "
    "was applied. No verifier ran and no rewards were inferred."
)


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _no_symlinks(path):
    path = Path(os.path.abspath(path))
    for parent in (*reversed(path.parents), path):
        if parent.is_symlink():
            raise ValueError(f"Symbolic links are not accepted: {parent}")
    return path


def _read(path, required=False):
    _no_symlinks(path)
    if not path.exists():
        if required:
            raise ValueError(f"Required source file is missing: {path}")
        return None
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return path.read_bytes()


def _object(path, required=False):
    data = _read(path, required)
    if data is None:
        return {}
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _time(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if result.tzinfo is None:
                return None
            return result.astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
    return None


def _outcome(request, export):
    value = request.get("outcome") or export.get("status") or request.get("stage")
    if isinstance(value, dict):
        value = value.get("status")
    return str(value or "unknown")


def _relative(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or any(p in (".", "..") for p in path.parts):
        raise ValueError(f"Invalid generated file path: {value}")
    return path


def _prior(output, source_identity):
    if not output.exists():
        return {}
    if not output.is_dir():
        raise ValueError(f"Output must be a directory: {output}")
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symbolic link in existing bundle: {path}")
    if not any(output.iterdir()):
        return {}
    marker = _object(output / MARKER, required=True)
    if marker.get("format") != FORMAT or marker.get("source_identity") != source_identity:
        raise ValueError("Refusing to overwrite an unrecognized bundle or a different source run")
    generated = marker.get("generated_sha256")
    if not isinstance(generated, dict):
        raise ValueError("Bundle marker has no generated file inventory")
    for relative, digest in generated.items():
        path = output / _relative(relative)
        data = _read(path)
        if data is not None and _sha(data) != digest:
            raise ValueError(f"Refusing to overwrite a modified generated file: {path}")
    return generated


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".bundle-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def write_bundle(run_dir: Path, output_dir: Path) -> dict:
    """Write one imported Harbor viewer job and return its provenance manifest.

    Only allowlisted request-local artifacts are copied. Existing generated
    files may be refreshed when their hashes still match the prior inventory;
    unrelated files are preserved and never overwritten.
    """
    run_dir = _no_symlinks(run_dir)
    output_dir = _no_symlinks(output_dir)
    if run_dir == output_dir or run_dir in output_dir.parents or output_dir in run_dir.parents:
        raise ValueError("Bundle output must be outside the source run directory and its ancestors")
    state = _object(run_dir / "state.json", required=True)
    run_index = _object(run_dir / "trajectories.json")
    indexed = {str(item.get("request_id")): item for item in run_index.get("requests", [])
               if isinstance(item, dict) and item.get("request_id")}
    requests_dir = _no_symlinks(run_dir / "requests")
    source_identity = _sha(str(run_dir).encode())
    previous = _prior(output_dir, source_identity)
    imported_at = datetime.now(timezone.utc).isoformat()
    run = str(state.get("binding", {}).get("run") or state.get("run")
              or state.get("identity", {}).get("run") or run_dir.name)
    job_id = str(uuid.uuid5(NAMESPACE, source_identity))
    files = {}
    trials = []
    summaries = []
    warnings = []
    agents = {}
    environment = {"import_path": "meta_agent.import_only:RecordedEnvironment",
                   "kwargs": {"import_only": True, "not_runnable": True}}
    for directory in sorted(requests_dir.iterdir()) if requests_dir.exists() else []:
        if directory.is_symlink():
            warnings.append(f"Skipped symbolic link request directory: {directory.name}")
            continue
        if not directory.is_dir():
            continue
        request_data = _read(directory / "request.json", required=True)
        request = json.loads(request_data)
        if not isinstance(request, dict):
            raise ValueError(f"Expected a request object: {directory / 'request.json'}")
        export = _object(directory / "export.json")
        request_id = str(request.get("id") or request.get("request_id") or directory.name)
        # Backfills can establish an old request's completion from its durable
        # event log without rewriting the original request.json.
        request = {**indexed.get(request_id, {}), **request}
        suffix = _sha(directory.name.encode())[:12]
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", request_id)[:64] or "request"
        trial_name = f"request-{safe_id}-{suffix}"
        trial_dir = output_dir / trial_name
        prefix = trial_name + "/"
        trajectory_data = _read(directory / "trajectory.json")
        trajectory = json.loads(trajectory_data) if trajectory_data else {}
        if not isinstance(trajectory, dict):
            raise ValueError(f"Expected a trajectory object: {directory / 'trajectory.json'}")
        reported_agent = trajectory.get("agent") or {}
        model_name = reported_agent.get("model_name")
        model_name = model_name if isinstance(model_name, str) and model_name else None
        phase = str(request.get("phase") or "unknown")
        step = request.get("decision_step", request.get("step"))
        outcome = _outcome(request, export)
        completed = outcome in ("completed", "review_complete", "complete", "success", "succeeded")
        error = request.get("error") or request.get("last_error") or export.get("error")
        failed = outcome in ("failed", "error", "interrupted", "timeout", "cancelled")
        started_at = _time(request.get("started_at") or request.get("created_at"))
        finished_at = _time(request.get("finished_at") or request.get("completed_at"))
        metadata = {"import_only": True, "notice": NOTICE, "run": run,
                    "request_id": request_id, "phase": phase, "training_step": step,
                    "outcome": outcome, "request_pass_completed": completed,
                    "trajectory_available": bool(trajectory_data), "export_capture": export.get("capture"),
                    "partial": bool(export.get("coverage", {}).get("partial", not completed))}
        agent = {"name": "meta-agent-supervisor", "kwargs": {"import_only": True, "notice": NOTICE}}
        if isinstance(model_name, str) and model_name:
            agent["model_name"] = model_name
        agents[json.dumps(agent, sort_keys=True)] = agent
        task = {"path": str(trial_dir / "artifacts/request.json"), "source": SOURCE}
        config = {"task": task, "trial_name": trial_name, "trials_dir": str(output_dir),
                  "agent": agent, "environment": environment, "verifier": {"disable": True},
                  "job_id": job_id, "extra_instructions": [NOTICE]}
        agent_info = {"name": "meta-agent-supervisor", "version": str(reported_agent.get("version") or "unknown")}
        if model_name:
            agent_info["model_info"] = {"name": model_name}
        exception = None
        if failed or error:
            detail = error if isinstance(error, str) else json.dumps(error, ensure_ascii=False) if error else outcome
            exception = {"exception_type": "ImportedSupervisionError", "exception_message": detail,
                         "exception_traceback": "Imported trace; no exception traceback was captured by the bundle writer.",
                         "occurred_at": finished_at or imported_at}
            if not finished_at:
                exception["exception_message"] += "; occurred_at is import time; original error time unavailable"
        result = {"id": str(uuid.uuid5(NAMESPACE, source_identity + "/" + directory.name)),
                  "task_name": f"meta-agent-{phase}-step-{step if step is not None else 'unknown'}",
                  "trial_name": trial_name, "trial_uri": trial_dir.as_uri(),
                  "task_id": {"path": task["path"]}, "source": SOURCE,
                  "task_checksum": _sha(request_data),
                  "config": config, "agent_info": agent_info, "agent_result": {"metadata": metadata},
                  "verifier_result": None, "exception_info": exception,
                  "started_at": started_at, "finished_at": finished_at,
                  "agent_execution": {"started_at": started_at, "finished_at": finished_at}}
        metrics = trajectory.get("final_metrics") or {}
        for source, target in (("total_prompt_tokens", "n_input_tokens"), ("total_cached_tokens", "n_cache_tokens"),
                               ("total_completion_tokens", "n_output_tokens"), ("total_cost_usd", "cost_usd")):
            value = metrics.get(source)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                result["agent_result"][target] = value
        files[prefix + "config.json"] = _json_bytes(config)
        files[prefix + "result.json"] = _json_bytes(result)
        files[prefix + "analysis.md"] = (NOTICE + "\n\n" + json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode()
        if trajectory_data:
            files[prefix + "agent/trajectory.json"] = trajectory_data
        allowlist = list(ARTIFACTS)
        native = (export.get("source") or {}).get("path")
        if isinstance(native, str) and re.fullmatch(r"native/sessions/[a-f0-9]{64}\.jsonl", native):
            allowlist.append(native)
        for relative in allowlist:
            try:
                content = request_data if relative == "request.json" else _read(directory / relative)
            except ValueError as exc:
                if "Symbolic links" not in str(exc):
                    raise
                warnings.append(f"Skipped symbolic link artifact: {directory.name}/{relative}")
                continue
            if content is not None:
                files[prefix + "artifacts/" + relative] = content
        if prefix + "artifacts/agent.stdout" in files:
            files[prefix + "agent/command-0/stdout.txt"] = files[prefix + "artifacts/agent.stdout"]
        if prefix + "artifacts/agent.stderr" in files:
            files[prefix + "agent/command-0/stderr.txt"] = files[prefix + "artifacts/agent.stderr"]
        trials.append(result)
        summaries.append({"trial_name": trial_name, **metadata})
    config = {"job_name": output_dir.name, "jobs_dir": str(output_dir.parent), "n_concurrent_trials": 1,
              "agents": list(agents.values()) or [{"name": "meta-agent-supervisor", "kwargs": {"import_only": True}}],
              "environment": environment, "verifier": {"disable": True},
              "tasks": [trial["config"]["task"] for trial in trials], "extra_instructions": [NOTICE]}
    completed_count = sum(item["request_pass_completed"] or trial["exception_info"] is not None
                          for item, trial in zip(summaries, trials))
    stats = {"n_completed_trials": completed_count,
             "n_errored_trials": sum(trial["exception_info"] is not None for trial in trials),
             "n_pending_trials": len(trials) - completed_count, "n_running_trials": 0}
    for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"):
        values = [trial["agent_result"][key] for trial in trials if key in trial["agent_result"]]
        if values:
            stats[key] = sum(values)
    starts = [trial["started_at"] for trial in trials if trial["started_at"]]
    result = {"id": job_id, "started_at": min(starts) if starts else imported_at,
              "updated_at": imported_at, "finished_at": None, "n_total_trials": len(trials),
              "stats": stats, "trial_results": trials}
    files["config.json"] = _json_bytes(config)
    files["result.json"] = _json_bytes(result)
    files["README.txt"] = (NOTICE + "\n\nView with: harbor view " + str(output_dir.parent) + "\n").encode()
    # Check all destinations before the first mutation, including unknown user files.
    for relative in files:
        target = _no_symlinks(output_dir / _relative(relative))
        if target.exists() and relative not in previous:
            raise ValueError(f"Refusing to overwrite an unowned bundle file: {target}")
    marker = {"format": FORMAT, "harbor_version": "0.24.0", "imported_at": imported_at,
              "source_identity": source_identity, "run": run, "notice": NOTICE,
              "source_supervision_status": state.get("status"), "warnings": warnings, "trials": summaries,
              "generated_sha256": {name: _sha(data) for name, data in files.items()}}
    for relative, data in files.items():
        _write(output_dir / relative, data)
    for relative in previous.keys() - files.keys():
        stale = output_dir / _relative(relative)
        if stale.exists():
            stale.unlink()
    _write(output_dir / MARKER, _json_bytes(marker))
    return marker
