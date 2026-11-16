"""Small process-safe readback receipts for the ColBench trial only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

_SEEN = set()


def record(kind, values, *, once=False):
    directory = os.environ.get("EAU_COLBENCH_UPTAKE_DIR")
    if not directory:
        return
    fingerprint = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()
    identity = (os.getpid(), kind, fingerprint)
    if once and identity in _SEEN:
        return
    _SEEN.add(identity)
    row = {"run": os.environ.get("EAU_ASYNC_AUDIT_RUN", Path(directory).parent.name), "kind": kind,
           "pid": os.getpid(), "time": time.time(), "values": values,
           "values_sha256": fingerprint, "scope": "consumer_readback"}
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(row, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(target / "consumption.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)
