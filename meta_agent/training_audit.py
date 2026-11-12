"""Environment-configured opt-in adapter for the non-blocking audit coordinator.

EAU_ASYNC_AUDIT_COMMAND: JSON reviewer argv, placeholders {request}/{response}.
EAU_ASYNC_AUDIT_PUBLISH_COMMAND: JSON gated-publisher argv; adds {publication}.
EAU_ASYNC_AUDIT_DIR / EAU_ASYNC_AUDIT_RUN: fresh absolute run directory / ID.
EAU_AUDIT_CONTEXT_PATH: absolute training-only context JSON with matching run.
EAU_ASYNC_AUDIT_MUTABLE_PATHS: JSON array of absolute gated mutable file paths.
EAU_ASYNC_AUDIT_EVERY / STOP / TIMEOUT / DRAIN_TIMEOUT: 20 / 624 / 1800 / 30.
EAU_ASYNC_AUDIT_FIRST: 1 (first completed batch is reviewed before warmup ends).
EAU_ASYNC_AUDIT_FAILURE_POLICY: raise (default) or continue_unchanged.
EAU_ASYNC_AUDIT_DIRECT_ROOTS: optional JSON array of editable directories.

This mode must not be combined with an independently configured blocking review barrier.

With DIRECT_ROOTS, the command is a direct editor with natural-language output.
The integrating project supplies its backend, access boundary and live consumers.
"""

import atexit
import json
import os
from pathlib import Path


_coordinator = None






def _get():
    global _coordinator
    if not os.environ.get("EAU_ASYNC_AUDIT_COMMAND"):
        return None
    if os.environ.get("EAU_STEP_REVIEW_STATE"):
        raise RuntimeError("Asynchronous audit cannot be combined with the strict step-review barrier")
    if _coordinator is None:
        from meta_agent.nonblocking_audit import NonBlockingAudit
        direct = os.environ.get('EAU_ASYNC_AUDIT_DIRECT_ROOTS')
        if direct:
            from meta_agent.direct_file_audit import DirectFileAudit
            coordinator_class = DirectFileAudit
            extra = dict(editable_roots=json.loads(direct))
        else:
            coordinator_class = NonBlockingAudit
            extra = {}
        _coordinator = coordinator_class(
            run=os.environ["EAU_ASYNC_AUDIT_RUN"],
            directory=Path(os.environ["EAU_ASYNC_AUDIT_DIR"]),
            context_path=Path(os.environ["EAU_AUDIT_CONTEXT_PATH"]),
            command=json.loads(os.environ["EAU_ASYNC_AUDIT_COMMAND"]),
            publish_command=json.loads(os.environ.get("EAU_ASYNC_AUDIT_PUBLISH_COMMAND", "[]")),
            mutable_paths=[Path(p) for p in json.loads(os.environ.get("EAU_ASYNC_AUDIT_MUTABLE_PATHS", "[]"))],
            every=int(os.environ.get("EAU_ASYNC_AUDIT_EVERY", "20")),
            stop=int(os.environ.get("EAU_ASYNC_AUDIT_STOP", "624")),
            first=int(os.environ.get("EAU_ASYNC_AUDIT_FIRST", "1")),
            timeout=float(os.environ.get("EAU_ASYNC_AUDIT_TIMEOUT", "1800")),
            failure_policy=os.environ.get("EAU_ASYNC_AUDIT_FAILURE_POLICY", "raise"), **extra)
        # On exceptional interpreter exit cancel promptly; the normal end of a
        # round calls close_training_audit with its explicit bounded drain.
        atexit.register(lambda: _coordinator.close(timeout=0) if _coordinator else None)
    return _coordinator


def before_training_step(step, *, is_final=False):
    audit = _get()
    if audit is not None:
        audit.before_step(step, is_final=is_final)


def audit_training_step(step, metrics, *, is_final=False):
    audit = _get()
    if audit is not None:
        audit.on_step(step, metrics, is_final=is_final)


def close_training_audit():
    audit = _get()
    if audit is not None:
        audit.close(timeout=float(os.environ.get("EAU_ASYNC_AUDIT_DRAIN_TIMEOUT", "30")))
