"""Export saved meta-agent requests without invoking an agent or changing training."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile


INDEX_FORMAT = "meta-agent-atif-index"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink export destination: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                         encoding="utf-8", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(directory):
    lock_path = directory / "runner.lock"
    if lock_path.is_symlink():
        raise ValueError("Refusing a symlink run lock")
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("A supervisor owns this run; use its automatic export or export after it stops") from None
        yield


def load_run(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("--run-dir must be an existing supervision run directory")
    state_path = directory / "state.json"
    if state_path.is_symlink():
        raise ValueError("Refusing a symlink supervision state")
    state = read_json(state_path)
    run = state.get("binding", {}).get("run")
    if not isinstance(run, str) or not run:
        raise ValueError("Supervision state must identify binding.run")
    return directory, state, run


def requests(directory):
    parent = directory / "requests"
    if parent.is_symlink():
        raise ValueError("Refusing a symlink requests directory")
    result = []
    for child in sorted(parent.iterdir()) if parent.exists() else []:
        if child.is_symlink():
            raise ValueError(f"Refusing a symlink request directory: {child.name}")
        if not child.is_dir():
            continue
        path = child / "request.json"
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink request record: {child.name}")
        if not path.exists():
            continue
        request = read_json(path)
        if not isinstance(request, dict) or request.get("id") != child.name:
            raise ValueError(f"Request identity does not match its directory: {child.name}")
        result.append((child, request))
    return sorted(result, key=lambda item: (item[1].get("started_at", 0), item[0].name))


def completed_events(directory):
    path = directory / "events.jsonl"
    if path.is_symlink():
        raise ValueError("Refusing a symlink event journal")
    completed = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # A crash can leave a final partial event line.
            if isinstance(event, dict) and event.get("event") == "review_complete" and event.get("request_id"):
                completed[event["request_id"]] = event
    return completed


def request_outcome(request, state, completed):
    if request.get("outcome") in {"completed", "failed", "interrupted"}:
        return request["outcome"]
    if request["id"] in completed:
        return "completed"
    active = state.get("active_request") or {}
    if active.get("id") == request["id"]:
        if state.get("status") == "interrupted":
            return "interrupted"
        if state.get("status") == "needs_reconciliation":
            return "failed"
    return "incomplete"


def write_index(directory, *, run=None):
    directory, state, actual_run = load_run(directory)
    if run is not None and run != actual_run:
        raise ValueError("Requested run differs from the saved supervision identity")
    completed = completed_events(directory)
    entries = []
    for child, request in requests(directory):
        if request.get("run", actual_run) != actual_run:
            raise ValueError(f"Request belongs to another run: {child.name}")
        manifest_path = child / "export.json"
        if manifest_path.is_symlink():
            raise ValueError("Refusing a symlink trajectory manifest")
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        trajectory = child / "trajectory.json"
        if trajectory.is_symlink():
            raise ValueError("Refusing a symlink trajectory")
        relative = child.relative_to(directory).as_posix()
        entries.append({
            "request_id": request["id"], "phase": request.get("phase"), "step": request.get("step"),
            "decision_step": request.get("decision_step"), "started_at": request.get("started_at"),
            "previous_request_id": (request.get("previous_request") or {}).get("id"),
            "finished_at": request.get("finished_at", completed.get(request["id"], {}).get("time")),
            "outcome": request_outcome(request, state, completed),
            "request_path": f"{relative}/request.json",
            "trajectory_path": f"{relative}/trajectory.json" if trajectory.exists() else None,
            "export_path": f"{relative}/export.json" if manifest_path.exists() else None,
            "capture": manifest.get("capture"), "coverage": manifest.get("coverage"),
            "warnings": manifest.get("warnings", []),
        })
    index = {"schema_version": 1, "format": INDEX_FORMAT, "run": actual_run,
             "trajectory_kind": "meta_agent_supervision", "requests": entries}
    path = directory / "trajectories.json"
    if path.exists() and (path.is_symlink() or read_json(path).get("format") != INDEX_FORMAT):
        raise ValueError("Existing trajectories.json is not an exporter-owned index")
    atomic_json(path, index)
    return index


def export_run(directory, *, sessions_dir=None, harbor_out=None):
    from export_atif import export_request
    directory, state, run = load_run(directory)
    if sessions_dir is None:
        sessions_dir = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser() / "sessions"
    with run_lock(directory):
        completed = completed_events(directory)
        errors = []
        exported = 0
        for child, request in requests(directory):
            if request.get("run", run) != run:
                raise ValueError(f"Request belongs to another run: {child.name}")
            try:
                export_request(child, run=run, sessions_dir=sessions_dir,
                               status=request_outcome(request, state, completed))
                exported += 1
            except (OSError, ValueError, TypeError) as error:
                errors.append({"request_id": request["id"], "error": f"{type(error).__name__}: {error}"})
        index = write_index(directory, run=run)
        summary = {"run": run, "exported": exported, "index": str(directory / "trajectories.json"),
                   "errors": errors, "request_count": len(index["requests"])}
        if harbor_out is not None:
            from harbor_bundle import write_bundle
            summary["harbor"] = write_bundle(directory, Path(harbor_out))
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Saved reports/supervision/<RUN> directory")
    parser.add_argument("--codex-sessions", help="Optional native sessions source; only exact thread IDs are eligible")
    parser.add_argument("--harbor-out", help="Optional imported Harbor viewer job directory, outside the source run")
    args = parser.parse_args()
    try:
        summary = export_run(args.run_dir, sessions_dir=args.codex_sessions, harbor_out=args.harbor_out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2 if summary["errors"] else 0
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"[EXPORT] {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
