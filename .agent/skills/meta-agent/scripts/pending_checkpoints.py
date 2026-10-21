"""List incomplete scheduled reviews for an explicitly identified EAU run."""
import argparse
import os
import re
import sys

from eau_paths import first_review_step, review_interval, run_for_log, select_log
from check_mandate import _review_rows, review_coverage


def expected_total_steps(log_path):
    with open(log_path, errors="replace") as handle:
        for line in handle:
            match = re.search(r"Training Progress:.*?\|\s*\d+/(\d+)", line)
            if match:
                return int(match.group(1))
    return None


def resume_start_step(log_path):
    with open(log_path, errors="replace") as handle:
        for line in handle:
            match = re.search(r"Setting global step to (\d+)", line)
            if match:
                return int(match.group(1))
    return 0


def current_run_and_step(log_path=None, run=None):
    path = select_log(log_path, run)
    if path is None:
        return None, None
    step = None
    with open(path, errors="replace") as handle:
        for line in handle:
            match = re.search(r"step:(\d+) - global_seqlen", line)
            if match:
                step = int(match.group(1))
    return run_for_log(path, run), step


def pending(run, step, *, log_path=None, interval=None, first_review=None):
    if run is None or step is None:
        return [], set()
    path = select_log(log_path, run)
    if path is None:
        raise ValueError(f"No log identifies run {run}")
    cadence = review_interval(interval)
    start = resume_start_step(path)
    first = first_review_step(cadence, start, first_review)
    handled, missed, _ = review_coverage(run)
    owed = [number for number in range(first, step + 1, cadence)
            if number not in handled and number not in missed]
    return owed, {number for number in missed if start < number <= step}


def _trainer_alive(pid):
    """Inspect only the PID explicitly bound to this run by the caller."""
    if pid is None:
        return None
    if pid <= 0:
        raise ValueError("Trainer PID must be positive")
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            command = handle.read().replace(b"\0", b" ")
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return b"verl.trainer.main_ppo" in command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", help="Exact log; required when automatic selection is ambiguous")
    parser.add_argument("--run", help="Must match --log through the registered aliases")
    parser.add_argument("--interval", type=int, help="Review interval; otherwise use EAU_REVIEW_INTERVAL")
    parser.add_argument("--first-review", type=int, help="First review boundary before resume adjustment")
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--trainer-pid", type=int, help="Caller-verified local trainer PID for this exact run")
    state.add_argument("--finished", action="store_true", help="Operator-confirmed end; report gaps and TEARDOWN")
    args = parser.parse_args()
    try:
        cadence = review_interval(args.interval)
        path = select_log(args.log, args.run)
        if path is None:
            print("[PENDING] no training log found; no run or supervision established")
            return 0
        run, step = current_run_and_step(path, args.run)
        if step is None:
            print(f"[PENDING] {run}: no completed step observed; run state unknown")
            return 0
        owed, missed = pending(run, step, log_path=path, interval=cadence, first_review=args.first_review)
        _, _, invalid = review_coverage(run)
        alive = _trainer_alive(args.trainer_pid)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"[PENDING] {error}\n")
    label = "finished (declared)" if args.finished else {
        True: "running (caller-bound PID)", False: "stopped (caller-bound PID)", None: "liveness unknown"
    }[alive]
    print(f"[PENDING] {run} at step {step} ({label}); review interval={cadence}")
    if missed:
        print("[PENDING] acknowledged missed reviews: " + ", ".join(map(str, sorted(missed))))
    legacy = [r for r in _review_rows(run) if int(r["step"]) <= step
              and r.get("cells") and not any("decision:" in c for c in r["cells"])]
    if legacy:
        print("[PENDING] legacy history retained; it does not certify runtime uptake")
    if owed:
        print("[PENDING] incomplete review steps: " + ", ".join(map(str, owed)))
        for number in owed:
            for error in invalid.get(number, []):
                print(f"[PENDING] step {number}: {error}")
    else:
        print("[PENDING] no reviews owed")
    if args.finished or alive is False:
        print("[PENDING] TEARDOWN: record remaining gaps without retro-filling; retire this run's "
              "watcher/sweep using their recorded IDs")
        return 0
    if owed:
        print("[PENDING] At the current review, reassess hypotheses and permitted evidence; choose apply, "
              "instrument, observe, defer, restart_plan or leave_unchanged. Record reason and next condition. "
              "A version bump is not required. Acknowledge older gaps instead of backfilling.")
    return 1 if owed else 0


if __name__ == "__main__":
    sys.exit(main())
