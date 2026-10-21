"""Resolve adapter paths, run aliases and an explicitly chosen review schedule."""
import glob
import json
import os
from pathlib import Path
import re

RUN_RE = re.compile(r"[A-Za-z0-9_-]+")


def root_path():
    value = os.environ.get("EAU_ROOT")
    if not value:
        raise RuntimeError("Set EAU_ROOT to the training checkout before using this helper")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"EAU_ROOT is not a directory: {path}")
    return str(path)


def gym_path():
    """The handover may load gyms from a sibling checkout rather than EAU_ROOT."""
    return str(Path(os.environ.get("EAU_GYM_ROOT", str(Path(root_path()) / "gyms")))
               .expanduser().resolve())


def _alias_map():
    path = Path(root_path()) / "reports" / "run_aliases.json"
    try:
        aliases = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(aliases, dict):
        raise ValueError(f"Run aliases must be an object: {path}")
    aliases = {k: v for k, v in aliases.items() if not k.startswith("_")}
    if any(not RUN_RE.fullmatch(k) or not isinstance(v, str)
           or not RUN_RE.fullmatch(v) for k, v in aliases.items()):
        raise ValueError(f"Invalid run label in {path}")
    return aliases


def _canonical(run, aliases):
    if not isinstance(run, str) or not RUN_RE.fullmatch(run):
        raise ValueError("An explicit run label using letters, digits, '_' or '-' is required")
    seen = set()
    while run in aliases and aliases[run] != run:
        if run in seen:
            raise ValueError(f"Cyclic run aliases at {run}")
        seen.add(run)
        run = aliases[run]
    return run


def canonical_run(run):
    return _canonical(run, _alias_map())


def run_names(run):
    aliases = _alias_map()
    canonical = _canonical(run, aliases)
    return {canonical, run} | {k for k in aliases if _canonical(k, aliases) == canonical}


def run_for_log(log_path, run=None):
    match = re.fullmatch(r"train_([A-Za-z0-9_]+?)_(\d{8})_(\d{6})\.log",
                         Path(log_path).name)
    if not match:
        raise ValueError("Expected a train_<name>_<YYYYMMDD>_<HHMMSS>.log filename")
    inferred = canonical_run("".join(match.groups()))
    if run is not None and canonical_run(run) != inferred:
        raise ValueError(f"Run {run!r} does not match log identity {inferred!r}; "
                         "register an explicit run_aliases.json mapping if they are the same run")
    return inferred


def select_log(log_path=None, run=None, pattern=None):
    """Never let another run's newest mtime silently select or switch a run."""
    if log_path:
        path = Path(log_path).expanduser()
        if not path.is_absolute():
            path = Path(root_path()) / path
        if not path.is_file():
            raise ValueError(f"Training log does not exist: {path}")
        run_for_log(path, run)
        return str(path.resolve())
    paths = glob.glob(pattern or str(Path(root_path()) / "logs" / "train_*.log"))
    if run is not None:
        paths = [p for p in paths if run_for_log(p) == canonical_run(run)]
    if len(paths) > 1:
        raise ValueError("Multiple matching training logs; select the current run with --log")
    if not paths:
        return None
    run_for_log(paths[0], run)
    return str(Path(paths[0]).resolve())


def review_interval(value=None):
    if value is None:
        current = os.environ.get("EAU_REVIEW_INTERVAL")
        legacy = os.environ.get("EAU_MANDATE_INTERVAL")
        if current and legacy and int(current) != int(legacy):
            raise ValueError("Conflicting EAU_REVIEW_INTERVAL and EAU_MANDATE_INTERVAL")
        value = current or legacy
    if value is None:
        raise ValueError("Choose --interval or EAU_REVIEW_INTERVAL for this run (no fixed default)")
    value = int(value)
    if value <= 0:
        raise ValueError("Review interval must be positive")
    return value


def first_review_step(interval, resume_step=0, first=None):
    first = int(first if first is not None else os.environ.get("EAU_FIRST_REVIEW", interval))
    if first <= 0:
        raise ValueError("First review step must be positive")
    if first <= resume_step:
        first += ((resume_step - first) // interval + 1) * interval
    return first
