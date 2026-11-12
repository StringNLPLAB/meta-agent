"""Single-flight, process-isolated audits at completed training boundaries.

Reviewers only propose.  A separate publisher command owns the domain gate and
must check ``request.state_hashes`` and ``publication_allowed`` immediately before
its atomic commit.  This coordinator holds ``commit_lock_path`` while the
publisher runs; publishers must not acquire that same lock a second time.

``on_step`` and ``before_step`` never wait for a model or publisher process.
Scheduled requests coalesce to the latest due step while one process is active.
The final rollout closes publication; ``close`` performs a bounded read-only
drain, then kills the child process group.  Every invocation is a fresh process.
The legacy synchronous tau2 review protocol is intentionally unchanged.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any
import uuid


_EVAL_KEY = re.compile(r"^(?:val(?:idation)?|test(?:ing)?|eval(?:uation)?)(?:[/_.-]|$)", re.I)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def training_metrics(metrics: dict) -> dict:
    """Retain finite JSON scalars from training, excluding evaluator namespaces."""
    result = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or _EVAL_KEY.match(key):
            continue
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, TypeError):
                continue
        if value is None or isinstance(value, (bool, str, int)):
            result[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[key] = value
    return result


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    values = {}
    for path in paths:
        if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
            raise ValueError(f"Audit mutable path must be an absolute regular file without symlinks: {path}")
        values[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def _command(value: list[str], *, required: bool = True) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError("Audit command must be a JSON array of nonempty argv strings")
    if required and not value:
        raise ValueError("Audit command cannot be empty")
    return value


def process_identity(pid: int) -> dict:
    """Linux PID identity for conservative cleanup after an owner has exited."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    start_ticks = int(stat.rsplit(")", 1)[1].split()[19])
    return dict(pid=pid, pid_start_ticks=start_ticks,
                cmdline_sha256=hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest())


class NonBlockingAudit:
    """One run's asynchronous reviewer and gated publisher.

    ``failure_policy=continue_unchanged`` records model/gate failures without
    blocking the policy.  Missing or invalid publication receipts after a state
    change always require reconciliation, regardless of that policy.
    """

    def __init__(self, *, run: str, directory: Path, command: list[str],
                 context_path: Path, mutable_paths: list[Path],
                 publish_command: list[str] | None = None, every: int = 20,
                 stop: int = 624, first: int = 1, timeout: float = 1800, failure_policy: str = "raise"):
        if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run):
            raise ValueError("Audit run must be a nonempty safe identifier")
        if type(every) is not int or every < 1 or type(stop) is not int or stop < 1:
            raise ValueError("Audit cadence and stop must be positive integers")
        if type(first) is not int or first < 1 or first > stop:
            raise ValueError("First audit step must be within the training interval")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Audit timeout must be finite and positive")
        if failure_policy not in {"raise", "continue_unchanged"}:
            raise ValueError("Unsupported asynchronous audit failure policy")
        self.run = run
        self.directory = Path(directory)
        self.context_path = Path(context_path)
        if not self.directory.is_absolute() or not self.context_path.is_absolute():
            raise ValueError("Audit directory and context manifest must be absolute")
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.resolve() != self.directory or self.context_path.resolve(strict=True) != self.context_path:
            raise ValueError("Audit directory and context manifest must not follow symlinks")
        self.command = _command(command)
        self.publish_command = _command(publish_command or [], required=False)
        self.mutable_paths = [Path(p) for p in mutable_paths]
        _file_hashes(self.mutable_paths)
        if self.publish_command and not self.mutable_paths:
            raise ValueError("A publisher requires explicitly bound mutable files")
        self.every, self.stop, self.first, self.timeout = every, stop, first, timeout
        self.failure_policy = failure_policy
        self.state_path = self.directory / "state.json"
        self.metrics_path = self.directory / "metrics.jsonl"
        self.events_path = self.directory / "events.jsonl"
        self._owner = (self.directory / "owner.lock").open("a+")
        try:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._owner.close()
            raise RuntimeError("Another audit coordinator owns this run directory") from error
        # Reusing a directory could resurrect an old-round response or duplicate
        # a request after a controller crash. Resume requires explicit recovery.
        if self.state_path.exists():
            self._owner.close()
            raise RuntimeError("Audit directory already has durable state; use a fresh run or reconcile it")
        self._active = None
        self._pending = None
        self._closing = False
        self._closed = False
        self._commit_lock = None
        self._fatal = None
        self._handles = []
        self.state = dict(schema_version=1, run=run, status="watching", latest_step=0,
                          publication_allowed=True, active_request=None, pending_step=None,
                          submitted=0, completed=0, failures=0, coalesced=0, stale=0,
                          cancelled=0, final_review_outcome=None, last_error=None)
        owner = process_identity(os.getpid())
        self.state.update(owner_pid=owner["pid"], owner_start_ticks=owner["pid_start_ticks"],
                          owner_cmdline_sha256=owner["cmdline_sha256"])
        self._save()

    def _save(self) -> None:
        _atomic_json(self.state_path, self.state)

    def _event(self, kind: str, **fields) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(time=time.time(), run=self.run, event=kind, **fields),
                                    allow_nan=False, sort_keys=True) + "\n")
            handle.flush()

    def _release_child(self) -> None:
        for handle in self._handles:
            handle.close()
        self._handles = []
        if self._commit_lock is not None:
            fcntl.flock(self._commit_lock.fileno(), fcntl.LOCK_UN)
            self._commit_lock.close()
            self._commit_lock = None
        self._active = None
        self.state["active_request"] = None

    def _kill_child(self) -> None:
        # A reviewer may exit while a same-group helper remains alive. Kill the
        # group even after its leader exits before consuming a final response.
        if self._active and "process" in self._active:
            try:
                os.killpg(self._active["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _fail(self, message: str, *, fatal: bool = False) -> None:
        self.state["failures"] += 1
        self.state["last_error"] = message
        self._event("audit_failed", error=message, requires_reconciliation=fatal)
        print(f"[ASYNC_AUDIT] run={self.run} failed: {message}", flush=True)
        if fatal or self.failure_policy == "raise":
            self.state["status"] = "needs_reconciliation" if fatal else "failed"
            self.state["publication_allowed"] = False
            self._closing = True
            self._fatal = message

    def _spawn(self, command: list[str], phase: str) -> None:
        active = self._active
        paths = dict(request=str(active["directory"] / "request.json"),
                     response=str(active["directory"] / "response.json"),
                     publication=str(active["directory"] / "publication.json"))
        argv = []
        for part in command:
            for key, value in paths.items():
                part = part.replace("{" + key + "}", value)
            argv.append(part)
        stdout = (active["directory"] / f"{phase}.stdout").open("wb")
        stderr = (active["directory"] / f"{phase}.stderr").open("wb")
        self._handles = [stdout, stderr]
        try:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout,
                                     stderr=stderr, start_new_session=True)
        except Exception:
            for handle in self._handles:
                handle.close()
            self._handles = []
            raise
        active.update(process=child, phase=phase, deadline=time.monotonic() + self.timeout)
        self.state["active_request"] = dict(id=active["request"]["id"], step=active["request"]["step"],
                                            phase=phase, **process_identity(child.pid))
        self._event("process_started", request_id=active["request"]["id"], phase=phase, pid=child.pid)

    def _start_pending(self) -> None:
        if self._active or self._pending is None or self._fatal:
            return
        pending, self._pending = self._pending, None
        self.state["pending_step"] = None
        if self._closing and not pending["is_final"]:
            self._event("pending_discarded_at_close", step=pending["step"])
            return
        request_id = f"audit_{pending['step']:06d}_{uuid.uuid4().hex[:12]}"
        directory = self.directory / "requests" / request_id
        directory.mkdir(parents=True)
        context_raw = self.context_path.read_bytes()
        context = json.loads(context_raw)
        if not isinstance(context, dict) or context.get("run") != self.run:
            raise RuntimeError("Audit context manifest has a mismatched run binding")
        snapshot = directory / "context.json"
        snapshot.write_bytes(context_raw)
        request = dict(schema_version=1, id=request_id, run=self.run, **pending,
                       created_at=time.time(), state_hashes=_file_hashes(self.mutable_paths),
                       context_path=str(snapshot), context_source_path=str(self.context_path),
                       context_sha256=hashlib.sha256(context_raw).hexdigest(),
                       metrics_history_path=str(self.metrics_path),
                       coordinator_state_path=str(self.state_path),
                       commit_lock_path=str(self.directory / "commit.lock"))
        _atomic_json(directory / "request.json", request)
        self._active = dict(directory=directory, request=request)
        self.state["submitted"] += 1
        try:
            self._spawn(self.command, "review")
        except Exception as error:
            self._release_child()
            self._fail(f"Could not start reviewer: {type(error).__name__}: {error}")

    def _read_response(self, active) -> dict:
        response = json.loads((active["directory"] / "response.json").read_text())
        request = active["request"]
        if (not isinstance(response, dict) or response.get("schema_version") != 1
                or response.get("request_id") != request["id"] or response.get("run") != self.run
                or type(response.get("step")) is not int or response["step"] != request["step"]
                or response.get("decision") not in {"observe", "apply"}):
            raise ValueError("Reviewer response has an invalid envelope or stale run/request binding")
        if response["decision"] == "observe" and response.get("proposal") is not None:
            raise ValueError("An observe response cannot carry a proposal")
        if response["decision"] == "apply" and not isinstance(response.get("proposal"), dict):
            raise ValueError("An apply response requires a proposal object")
        return response

    def _finish_review(self, active, code: int) -> None:
        self._kill_child()
        if code:
            self._release_child()
            self._fail(f"Reviewer exited {code}; inspect {active['directory']}")
            return
        try:
            response = self._read_response(active)
        except (OSError, ValueError, TypeError) as error:
            self._release_child()
            self._fail(f"Invalid reviewer response: {error}")
            return
        if response["decision"] == "observe":
            self.state["completed"] += 1
            if active["request"]["is_final"]:
                self.state["final_review_outcome"] = "observed"
            self._event("review_observed", request_id=active["request"]["id"], step=active["request"]["step"])
            self._release_child()
            return
        if self._closing or active["request"]["is_final"]:
            self._event("proposal_discarded_at_close", request_id=active["request"]["id"])
            self.state["completed"] += 1
            if active["request"]["is_final"]:
                self.state["final_review_outcome"] = "apply_discarded_read_only"
            self._release_child()
            return
        if not self.publish_command:
            self._release_child()
            self._fail("Apply proposed without an explicitly configured publisher")
            return
        self._commit_lock = (self.directory / "commit.lock").open("a+")
        try:
            fcntl.flock(self._commit_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._release_child()
            self._fail("Commit lock unexpectedly held; refusing an uncoordinated writer", fatal=True)
            return
        if _file_hashes(self.mutable_paths) != active["request"]["state_hashes"]:
            self.state["stale"] += 1
            self._event("stale_proposal", request_id=active["request"]["id"])
            self._release_child()
            return
        for handle in self._handles:
            handle.close()
        self._handles = []
        try:
            self._spawn(self.publish_command, "publish")
        except Exception as error:
            self._release_child()
            self._fail(f"Could not start publisher: {type(error).__name__}: {error}")

    def _finish_publish(self, active, code: int) -> None:
        self._kill_child()
        before = active["request"]["state_hashes"]
        after = _file_hashes(self.mutable_paths)
        if active.get("cancelled_at_close") and code and after == before:
            self._event("publisher_cancelled_unchanged", request_id=active["request"]["id"])
            self._release_child()
            return
        try:
            receipt = json.loads((active["directory"] / "publication.json").read_text())
            if (not isinstance(receipt, dict) or receipt.get("schema_version") != 1
                    or receipt.get("request_id") != active["request"]["id"] or receipt.get("run") != self.run
                    or receipt.get("status") not in {"applied", "rejected", "stale"}
                    or receipt.get("before_state_hashes") != before
                    or receipt.get("after_state_hashes") != after):
                raise ValueError("Publisher receipt mismatch")
            unchanged_base = receipt.get("observed_before_state_hashes", before)
            if receipt["status"] in {"rejected", "stale"} and after != unchanged_base:
                raise ValueError("Rejected publisher changed live state")
            if code and not active.get("cancelled_at_close"):
                raise ValueError(f"Publisher exited {code}")
        except (OSError, ValueError, TypeError) as error:
            self._release_child()
            self._fail(f"Invalid publisher result: {error}", fatal=after != before)
            return
        self._event("publication_finished", request_id=active["request"]["id"], status=receipt["status"])
        self.state["completed"] += 1
        if receipt["status"] == "stale":
            self.state["stale"] += 1
        self._release_child()

    def poll(self, *, start_pending: bool = True) -> None:
        """Consume already-finished work and spawn pending work without waiting."""
        active = self._active
        if active is not None:
            code = active["process"].poll()
            if code is None and time.monotonic() >= active["deadline"]:
                self._kill_child()
                active["timed_out"] = True
                self._event("process_timeout", request_id=active["request"]["id"], phase=active["phase"])
                code = active["process"].poll()
            if code is not None:
                if active["phase"] == "publish":
                    self._finish_publish(active, code)
                else:
                    self._finish_review(active, code)
        if start_pending:
            self._start_pending()
        self._save()
        if self._fatal:
            raise RuntimeError(self._fatal)

    def before_step(self, step: int, *, is_final: bool = False) -> None:
        if self._closed:
            raise RuntimeError("Audit coordinator is closed")
        if is_final or step >= self.stop:
            self._close_publication()
        self.poll()

    def _close_publication(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.state["publication_allowed"] = False
        self.state["status"] = "closing"
        if self._pending and not self._pending["is_final"]:
            self._event("pending_discarded_at_close", step=self._pending["step"])
            self._pending = None
            self.state["pending_step"] = None
        self._save()  # Publisher checks this immediately before its atomic write.
        if self._active and self._active["phase"] == "publish":
            self._active["cancelled_at_close"] = True
            self._kill_child()
            self._event("publisher_cancelled_at_close", request_id=self._active["request"]["id"])

    def on_step(self, step: int, metrics: dict, *, is_final: bool = False) -> None:
        if self._closed or type(step) is not int or step <= self.state["latest_step"]:
            raise RuntimeError("Audit completed steps must strictly increase in an open coordinator")
        is_final = bool(is_final or step >= self.stop)
        if is_final:
            self._close_publication()
        values = training_metrics(metrics)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(step=step, metrics=values), allow_nan=False, sort_keys=True) + "\n")
            handle.flush()
        self.state["latest_step"] = step
        self.poll(start_pending=False)
        if is_final or step == self.first or step % self.every == 0:
            if self._pending is not None:
                self.state["coalesced"] += 1
                self._event("request_coalesced", skipped_step=self._pending["step"], replacement_step=step)
            self._pending = dict(step=step, metrics=values, is_final=is_final)
            self.state["pending_step"] = step
            self._event("request_due", step=step, is_final=is_final, busy=self._active is not None)
        self._start_pending()
        self._save()

    def close(self, *, timeout: float = 30) -> None:
        """Drain read-only work for at most timeout seconds, then kill and reap."""
        if self._closed:
            return
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Audit drain timeout must be finite and nonnegative")
        self._close_publication()
        deadline = time.monotonic() + timeout
        try:
            while self._active is not None or self._pending is not None:
                self.poll()
                if self._active is None and self._pending is None:
                    break
                if time.monotonic() >= deadline:
                    self._event("drain_timeout", timeout=timeout)
                    self._kill_child()
                    # SIGKILL ends the full process group; do not permit a late
                    # publisher to survive a round transition.
                    if self._active is not None:
                        active = self._active
                        code = active["process"].wait(timeout=5)
                        self._event("request_cancelled", request_id=active["request"]["id"],
                                    phase=active["phase"])
                        self.state["cancelled"] += 1
                        if active["request"]["is_final"]:
                            self.state["final_review_outcome"] = "cancelled"
                        if active["phase"] == "publish":
                            self._finish_publish(active, code)
                        else:
                            self._release_child()
                    self._pending = None
                    self.state["pending_step"] = None
                    break
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        finally:
            if self._active is not None:
                self._kill_child()
                self._active["process"].wait(timeout=5)
                self._release_child()
            self._closed = True
            if not self._fatal:
                self.state["status"] = "completed_with_errors" if self.state["failures"] else "completed"
            self.state["publication_allowed"] = False
            self._save()
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_UN)
            self._owner.close()
