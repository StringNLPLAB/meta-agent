"""Credential-free, process-safe accounting for external training-user calls."""
import fcntl
import json
import os
from pathlib import Path
import time


def check_training_user_health(destination, *, run):
    """Stop on permanent API errors or an exhausted environment retry loop.

    The reply recorder still never changes environment behavior. The campaign
    supervisor checks its separate, credential-free ledger so it cannot silently
    continue training with the environment's ``No response.`` fallback.
    """
    path = Path(destination)
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            start = max(0, size - 1024 * 1024)
            stream.seek(start)
            if start:
                stream.readline()  # Discard a possibly partial first record.
            data = stream.read()
    except FileNotFoundError:
        return
    # A concurrent writer can leave its final line incomplete for this check.
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if (not isinstance(row, dict) or row.get("run") != run
                or row.get("operation") != "colbench_training_user"
                or row.get("status") != "failed"):
            continue
        if type(row.get("http_status")) is int and row["http_status"] in (401, 402, 403):
            raise RuntimeError(
                f"Training user API returned HTTP {row['http_status']}; "
                "stopping invalid simulator trial. Check provider authorization, balance and limits.")
        if type(row.get("attempt")) is int and row["attempt"] >= 3:
            raise RuntimeError(
                "Training user API failed all three environment attempts; "
                "stopping invalid simulator trial instead of admitting fallback replies.")


def record_call(completion=None, *, model_requested, task_id, attempt, error=None):
    destination = os.environ.get("EAU_TRAINING_API_USAGE_FILE")
    if not destination:
        return
    row = {"time": time.time(), "run": os.environ.get("EAU_ASYNC_AUDIT_RUN"),
           "operation": "colbench_training_user", "model_requested": str(model_requested),
           "task_id": str(task_id), "attempt": attempt, "status": "failed" if error else "completed"}
    if completion is not None:
        row["returned_model"] = str(getattr(completion, "model", ""))
        usage = getattr(completion, "usage", None)
        row["usage"] = {key: value for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if type(value := getattr(usage, key, None)) is int and value >= 0}
        row["usage_available"] = bool(row["usage"])
    if error is not None:
        row["error_type"] = type(error).__name__
        status = getattr(error, "status_code", None)
        if type(status) is int:
            row["http_status"] = status
    # Do not serialize request/response text, headers, exception messages or env.
    try:
        path = Path(destination)
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError("API accounting path must be canonical and absolute")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    except (OSError, ValueError) as failure:
        # Accounting must never change a user reply or create an API retry.
        print(f"[TRAINING_API_USAGE_CAPTURE_ERROR] {type(failure).__name__}", flush=True)
