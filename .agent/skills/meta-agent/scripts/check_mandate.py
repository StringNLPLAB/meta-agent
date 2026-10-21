"""Check review completeness, separately from application success and outcomes.

The historical filename/state file and eight-column log remain compatible.
Config versions are diagnostics only; they never prove uptake or require a write.
New reviews use tagged fields in existing cells. Untagged history is reported as
legacy/unverified rather than being rewritten or upgraded to runtime evidence.
"""
import argparse
import html
import json
import os
import re
import sys
from typing import Optional

from eau_paths import canonical_run, gym_path, root_path, run_for_log, run_names, select_log

ROOT = root_path()
CHANGE_LOG = os.path.join(ROOT, "reports", "change_log.md")
STATE = os.path.join(ROOT, "reports", ".mandate_state.json")
WATCHED = {
    "entropy_schedule": os.path.join(ROOT, "entropy_schedule.json"),
    "env_policy": os.path.join(gym_path(), "symbolic_layer", "env_policy.json"),
    "reward_shaping": os.path.join(ROOT, "meta_agent", "reward_shaping.json"),
    "process_reward": os.path.join(gym_path(), "symbolic_layer", "process_reward.json"),
    "difficulty": os.path.join(gym_path(), "symbolic_layer", "difficulty.json"),
    "clip_schedule": os.path.join(ROOT, "clip_schedule.json"),
}


def _versions():
    out = {}
    for name, path in WATCHED.items():
        try:
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle)
            out[name] = config.get("version") if isinstance(config, dict) else None
        except (OSError, ValueError):
            out[name] = None
    return out


def _load_state():
    try:
        with open(STATE, encoding="utf-8") as handle:
            state = json.load(handle)
    except FileNotFoundError:
        return {}
    if not isinstance(state, dict):
        raise ValueError(f"Invalid checker state: {STATE}")
    return state


def _save_state(state):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, STATE)


def table_cells(line):
    """Split an outer-pipe Markdown row, respecting escaped pipes."""
    line = line.strip()
    if not line.startswith("|"):
        return []
    cells, cell, slashes = [], [], 0
    for char in line:
        if char == "|" and slashes % 2 == 0:
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append(char)
        slashes = slashes + 1 if char == "\\" else 0
    cells.append("".join(cell).strip())
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return cells


def _present(value):
    return bool(str(value).strip()) and str(value).strip().lower() not in {"—", "-", "none", "n/a"}


def _review_rows(run: Optional[str] = None, path=None):
    names = run_names(run) if run is not None else None
    try:
        with open(path or CHANGE_LOG, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for number, line in enumerate(lines, 1):
        cells = table_cells(line)
        if len(cells) < 2 or not re.fullmatch(r"[A-Za-z0-9_-]+", cells[0]) or not cells[1].isdigit():
            continue
        if names is not None and cells[0] not in names:
            continue
        row = {"run": cells[0], "step": cells[1], "line": number, "cells": cells, "raw": line}
        for index, name in ((3, "knob"), (5, "evidence"), (6, "prediction"), (7, "verification")):
            row[name] = cells[index] if len(cells) > index else ""
        row["has_trajectory"] = any(m in " ".join(cells) for m in ("trajectory:", "TRAJ:"))
        rows.append(row)
    return rows


def _prediction_rows(run: Optional[str] = None, path=None):
    """Compatibility API for old consumers; current reviews need no invented prediction."""
    return [r for r in _review_rows(run, path) if len(r["cells"]) == 8 and _present(r["prediction"])]


_FIELDS = {
    "decision", "reason", "next", "target", "old", "new", "mechanism", "window", "samples",
    "application", "attempt", "boundary", "scope", "confirmed", "uptake", "question", "method",
    "limitation", "plan", "recovery",
}
_DECISIONS = {"apply", "instrument", "observe", "defer", "restart_plan", "leave_unchanged"}
_APPLICATIONS = {"pending", "partial", "failed", "fully_applied", "control_reverted"}


def review_fields(row):
    fields, duplicates = {}, []
    for cell in row["cells"][2:]:
        for match in re.finditer(r"(?:^|;\s*)([a-z][a-z_]*):\s*([^;]*)", html.unescape(cell)):
            key, value = match.groups()
            if key not in _FIELDS:
                continue
            if key in fields:
                duplicates.append(key)
            fields[key] = value.strip()
    return fields, duplicates


def is_missed(row):
    if len(row["cells"]) != 8:
        return False
    text = row["knob"].strip().lower()
    return bool(re.match(r"^(?:missed$|missed[ -](?:review|checkpoint)\b|(?:review|checkpoint)[ -]missed\b)", text))


def validate_review(row):
    """Return structural errors; this does not authenticate observations or causality."""
    if len(row["cells"]) != 8:
        return [f"expected 8 columns, found {len(row['cells'])}"]
    if is_missed(row):
        return ["acknowledged gap, not a completed review"]
    fields, duplicates = review_fields(row)
    if "decision" not in fields and "application" not in fields:
        row["schema"] = "legacy_unverified"
        return ([] if _present(row["prediction"]) and _present(row["evidence"])
                else ["legacy row needs evidence and prediction; use tagged fields for new reviews"])
    row["schema"] = "review-v2"
    errors = [f"duplicate field: {key}" for key in duplicates]
    decision = fields.get("decision")
    if decision not in _DECISIONS:
        errors.append(f"unknown or missing decision: {decision!r}")
    required = ["reason", "next"]
    if not _present(row["evidence"]):
        errors.append("missing evidence (or explicit reason evidence is unavailable)")
    if decision == "apply":
        required += ["target", "old", "new", "mechanism", "window", "application", "attempt", "recovery"]
        if not _present(row["prediction"]):
            errors.append("apply requires a preregistered prediction and criterion")
    elif decision == "instrument":
        required += ["question", "method"]
    elif decision == "observe":
        if not any(_present(fields.get(key, "")) for key in ("window", "samples")):
            errors.append("observe requires a pending window or sample requirement")
    elif decision in {"defer", "restart_plan"}:
        required += ["limitation", "plan"]
        if decision == "restart_plan":
            required.append("recovery")
    if "application" in fields:
        status = fields["application"]
        if status not in _APPLICATIONS:
            errors.append(f"unknown application status: {status!r}")
        if decision != "apply":
            errors.append("application status belongs to an apply review")
        if status == "fully_applied":
            required += ["boundary", "scope", "confirmed", "uptake"]
            unknown = {"unknown", "pending", "unavailable", "not checked", "not observed"}
            for key in ("boundary", "scope", "confirmed", "uptake"):
                if fields.get(key, "").lower() in unknown:
                    errors.append(f"fully_applied cannot use {key}={fields[key]!r}")
            scope = {x.strip() for x in fields.get("scope", "").split(",") if x.strip()}
            confirmed = {x.strip() for x in fields.get("confirmed", "").split(",") if x.strip()}
            if not scope or confirmed != scope:
                errors.append("fully_applied requires confirmation of the complete declared consumer scope")
    errors += [f"missing field: {key}" for key in dict.fromkeys(required)
               if not _present(fields.get(key, ""))]
    return errors


def review_coverage(run, path=None):
    """A complete row cannot hide an incomplete application at the same review."""
    grouped, missed = {}, set()
    for row in _review_rows(run, path):
        number = int(row["step"])
        if is_missed(row):
            missed.add(number)
        else:
            grouped.setdefault(number, []).append(row)
    handled, invalid = set(), {}
    for number, rows in grouped.items():
        errors = [f"line {row['line']}: {error}" for row in rows for error in validate_review(row)]
        if number in missed:
            errors.append("review conflicts with an acknowledged gap; do not backfill")
        if errors:
            invalid[number] = errors
        else:
            handled.add(number)
    # A contradictory miss marker cannot conceal an incomplete current review.
    missed -= set(grouped)
    return handled, missed, invalid


def check(step: int, run: Optional[str] = None, *, log_path=None, baseline=False, read_only=False):
    if step < 0:
        raise ValueError("Step must be non-negative")
    if log_path:
        log_path = select_log(log_path, run)
        run = run_for_log(log_path, run)
    else:
        run = canonical_run(run)
    state = _load_state()
    runs = dict(state.get("runs") or {})
    # Preserve old unscoped state without assigning it to an unrelated current run.
    if not runs and (state.get("snapshots") or state.get("last_checkpoint_step") is not None):
        runs["_legacy"] = state.get("snapshots") or [{
            "step": state["last_checkpoint_step"], "versions": state.get("versions") or {}}]
    names = run_names(run)
    snapshot_by_step = {}
    for name in sorted(names - {run}) + [run]:
        for snapshot in runs.get(name, []):
            snapshot_by_step[snapshot["step"]] = snapshot
    snapshots = list(snapshot_by_step.values())
    earlier = [s for s in snapshots if isinstance(s.get("step"), int) and s["step"] < step]
    base = max(earlier, key=lambda s: s["step"]) if earlier else None
    now, before = _versions(), (base.get("versions") or {}) if base else {}
    changed = [name for name, version in now.items() if version != before.get(name)]
    bumped = [name for name in changed if isinstance(now[name], (int, float))
              and (before.get(name) is None or (isinstance(before[name], (int, float))
                                              and now[name] > before[name]))]
    rows = [r for r in _review_rows(run) if int(r["step"]) == step]
    violations = []
    if not baseline:
        if not rows:
            violations.append(f"No review record for run {run} at step {step}; do not backfill missed reviews")
        for row in rows:
            violations.extend(f"line {row['line']}: {error}" for error in validate_review(row))
    legacy = [r["line"] for r in rows if r.get("schema") == "legacy_unverified"]
    result = {
        "step": step, "run": run, "previous_checkpoint": base["step"] if base else None,
        "first_run": base is None, "baseline_only": baseline, "ok": not violations,
        "configs_bumped": bumped, "configs_changed": changed, "new_review_rows": rows,
        "new_prediction_rows": [r for r in rows if _present(r["prediction"])],
        "legacy_rows": legacy, "violations": violations,
        "warnings": (["Legacy history is unverified; no runtime uptake or outcome is certified"] if legacy else []),
        "runtime_verified": False,
    }
    if not read_only:
        snapshots = [s for s in snapshots if s.get("step") != step]
        snapshots.append({"step": step, "versions": now, "review_ok": not violations and not baseline})
        runs[run] = sorted(snapshots, key=lambda s: s["step"])[-40:]
        state.update({"runs": runs, "last_checkpoint_step": step, "versions": now})
        _save_state(state)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--run", help="Explicit run label, or derive it from --log and registered aliases")
    parser.add_argument("--log", help="Exact training log; verifies the run identity when supplied")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="Snapshot only; does not complete a review")
    parser.add_argument("--read-only", action="store_true", help="Do not update checker snapshots")
    args = parser.parse_args()
    try:
        result = check(args.step, args.run, log_path=args.log, baseline=args.baseline, read_only=args.read_only)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"[REVIEW] {error}\n")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["baseline_only"]:
        print(f"[REVIEW] {result['run']} step {args.step}: baseline only; no review completed")
    else:
        status = "complete (record only, not runtime verification)" if result["ok"] else "INCOMPLETE"
        print(f"[REVIEW] {result['run']} step {args.step}: {status}")
        for message in result["violations"] + result["warnings"]:
            print(f"  - {message}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
