
"""Watch EAU training metrics and check checkpoint records; signaling is opt-in."""
from eau_paths import (root_path, select_log, run_for_log, review_interval, first_review_step)
from pending_checkpoints import resume_start_step

import argparse
import signal
import collections
import glob
import math
import os
import re
import subprocess
import sys
import time

ROOT = (root_path() + '')
CHECKER = os.path.join(os.path.dirname(__file__), "check_mandate.py")

_UC_THRESHOLD = float(os.environ.get("USERRL_AUTOKILL_UNCLOSED_THR", "0.10"))

_UC_MIN_GYMS = int(os.environ.get("USERRL_AUTOKILL_UNCLOSED_MIN_GYMS", "3"))

_SKIP_STEPS = int(os.environ.get("USERRL_AUTOKILL_SKIP_STEPS", "3"))
PY = sys.executable

STEP_RE = re.compile(r"step:(\d+) - (.*)")
TOOLFAIL_RE = re.compile(r"TOOL_CALL_FAILED|Failed to parse fc related info")

CRASH_RE = re.compile(
    r"Traceback \(most recent call|CUDA out of memory|OutOfMemoryError|"
    r"RuntimeError:|AssertionError|Killed|Segmentation fault|ray\.exceptions",
)

FIELDS = [
    ("ent", "actor/entropy", "{:.3f}"),
    ("rew", "task_reward/mean", "{:.3f}"),
    ("sr", "task_reward/success_rate", "{:.3f}"),
    ("az", "debug/groups_all_zero", "{:.0f}"),
    ("entc", "actor/entropy_coeff", "{:.4f}"),
    ("gn", "actor/grad_norm", "{:.2f}"),

    ("len", "response_length/mean", "{:.0f}"),
    ("trunc", "response_length/clip_ratio", "{:.3f}"),
]
GYMS = ["travel", "persuade", "turtle", "function", "tau"]

def emit(line):
    sys.stdout.write(line.rstrip() + "\n")
    sys.stdout.flush()

def parse(blob):
    out = {}
    for tok in blob.split(" - "):
        k, _, v = tok.partition(":")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out

def _find_trainer_pids():
    """PIDs of the verl trainer, excluding this process and its ancestors.

    Enumerating /proc rather than shelling out to pgrep/ps: a `grep -f` pattern
    matches the watcher's OWN command line, which has already cost this
    engagement seven self-matches (and once killed the shell that ran it).
    """
    me = os.getpid()
    mine = {me}
    p = me
    for _ in range(8):
        try:
            ppid = int(open(f"/proc/{p}/stat").read().split(") ", 1)[1].split()[1])
            mine.add(ppid)
            p = ppid
        except Exception:  
            break
    out = []
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(d.rsplit("/", 1)[-1])
        if pid in mine:
            continue
        try:
            cmd = open(d + "/cmdline", "rb").read().replace(b"\0", b" ").decode("utf8", "replace")
        except Exception:  
            continue
        if "verl.trainer.main_ppo" in cmd:
            out.append(pid)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    ap.add_argument("--run", help="Run identity, checked against the log and registered aliases")
    ap.add_argument("--interval", type=int, help="Review interval for this run")
    ap.add_argument("--first-review", type=int, help="First review boundary before resume adjustment")
    ap.add_argument("--glob", default=os.path.join(ROOT, "logs", "train_*.log"))

    ap.add_argument("--autokill", action="store_true",
                    help="SIGTERM the trainer when the collapse criterion fires")
    ap.add_argument("--replay", action="store_true",
                    help="Read the log from the START and exit at EOF instead of "
                         "following it, and never signal any process. This is how "
                         "an autokill criterion gets verified against a historical "
                         "log THROUGH THE REAL CODE PATH -- criterion A was only "
                         "ever backtested by a separate script that reimplemented "
                         "it, and a reimplementation can agree with the log while "
                         "disagreeing with the code that actually runs.")
    args = ap.parse_args()

    try:
        interval = review_interval(args.interval)
        path = select_log(args.log, args.run, args.glob)
        while path is None and not args.replay:
            time.sleep(20)
            path = select_log(args.log, args.run, args.glob)
        if path is None:
            raise ValueError("No log to replay")
        run = run_for_log(path, args.run)
        first_due = first_review_step(interval, resume_start_step(path), args.first_review)
    except (OSError, ValueError, RuntimeError) as error:
        ap.exit(2, f"[WATCH] {error}\n")
    emit(f"[WATCH] following {os.path.basename(path)} as {run}; interval={interval}")

    BAND = interval
    hist = collections.deque(maxlen=BAND)

    toolfail = [0]          
    last_toolfail = [0]

    seen_step = -1
    zero_sr_run = [0]       
    len_hist = []           
    uc_hist = []            
    skip_hist = []          
    killed = [False]
    f = open(path, "r", errors="replace")
    if not getattr(args, "replay", False):
        f.seek(0, os.SEEK_END)
    while True:
        line = f.readline()
        if not line and getattr(args, "replay", False):
            emit("[REPLAY] end of log reached")
            return
        if not line:
            time.sleep(2)
            continue

        if TOOLFAIL_RE.search(line):
            toolfail[0] += 1
            continue
        if CRASH_RE.search(line):
            emit(f"[CRASH] {line.strip()[:300]}")
            continue

        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        if step == seen_step:
            continue
        seen_step = step
        d = parse(m.group(2))

        hist.append((d.get("actor/entropy"), d.get("task_reward/mean")))

        L = d.get("response_length/mean")
        if isinstance(L, float):
            len_hist.append(L)
        sr = d.get("task_reward/success_rate")
        if isinstance(sr, float):
            zero_sr_run[0] = zero_sr_run[0] + 1 if sr == 0.0 else 0
        uc = d.get("task_reward/unclosed_call_frac")
        if isinstance(uc, float):
            uc_hist.append(uc)

        osk = d.get("actor/optim_skipped")
        gn, kl = d.get("actor/grad_norm"), d.get("actor/kl_loss")
        if isinstance(osk, float) and not math.isnan(osk):
            skip_hist.append(osk > 0.5)
        elif isinstance(gn, float):

            skip_hist.append(math.isnan(gn) and isinstance(kl, float) and not math.isnan(kl))

        def _fire(reason, detail):
            killed[0] = True
            if getattr(args, "replay", False):
                emit(f"[AUTOKILL-DRYRUN] step {step}: {reason} (replay: no signal sent)")
                return
            emit(f"[AUTOKILL] step {step}: {reason} -- stopping the trainer.")
            victims = _find_trainer_pids()
            for pid in victims:
                try:
                    os.kill(pid, signal.SIGTERM)
                    emit(f"[AUTOKILL] SIGTERM -> pid {pid}")
                except Exception as e:  
                    emit(f"[AUTOKILL] could not signal {pid}: {e}")
            if not victims:
                emit("[AUTOKILL] no verl.trainer.main_ppo process found; nothing to stop")
            with open(os.path.join(ROOT, "reports", "autokill.log"), "a") as fh:
                fh.write(f"{time.strftime('%F %T')}  {os.path.basename(path)} step {step} "
                         f"{detail} pids={victims}\n")

        if (args.autokill and not killed[0] and step >= 25
                and zero_sr_run[0] >= 3 and len(len_hist) >= 3):
            best = max(sum(len_hist[max(0, j - 2):j + 1]) / len(len_hist[max(0, j - 2):j + 1])
                       for j in range(len(len_hist)))
            if len_hist[-1] < 0.6 * best:
                _fire(f"success_rate=0 for {zero_sr_run[0]} consecutive steps and resp_len "
                      f"{len_hist[-1]:.0f} < 60% of this run's best 3-step band ({0.6 * best:.0f})",
                      f"critA sr=0x{zero_sr_run[0]} len={len_hist[-1]:.0f} thr={0.6*best:.0f}")

        if (args.autokill and not killed[0] and step >= 25 and len(uc_hist) >= 3):
            win = uc_hist[-3:]

            n_gyms_over = sum(
                1 for k, v in d.items()
                if k.startswith("task_reward/by_gym/")
                and k.endswith("/unclosed_call_frac")
                and isinstance(v, float) and v > _UC_THRESHOLD
            )
            if sum(1 for v in win if v > _UC_THRESHOLD) >= 2 and n_gyms_over >= _UC_MIN_GYMS:
                _fire(f"unclosed_call_frac > {_UC_THRESHOLD} on {sum(1 for v in win if v > _UC_THRESHOLD)} "
                      f"of the last 3 steps ({', '.join(f'{v:.3f}' for v in win)}) "
                      f"AND {n_gyms_over} gyms individually over it; legacy reference range was 0.003-0.014",
                      f"critB uc={win[-1]:.3f} thr={_UC_THRESHOLD} gyms={n_gyms_over}")

        if (args.autokill and not killed[0] and step >= 25
                and len(skip_hist) >= _SKIP_STEPS and all(skip_hist[-_SKIP_STEPS:])):
            _fire(f"legacy skipped-update criterion on {_SKIP_STEPS} consecutive steps; "
                  "inspect optimizer/update provenance to establish affected learning",
                  f"critC skipped={_SKIP_STEPS} step={step}")

        if step >= first_due and (step - first_due) % interval == 0:
            parts = []
            for label, key, fmt in FIELDS:
                if key in d:
                    parts.append(f"{label}=" + fmt.format(d[key]))
            for g in GYMS:
                k = f"task_reward/by_gym/{g}/mean"
                if k in d:
                    parts.append(f"{g}={d[k]:.3f}")
            ents = [e for e, _ in hist if isinstance(e, float)]
            rews = [r for _, r in hist if isinstance(r, float)]
            if ents:
                parts.append(f"|{len(ents)}-step mean ent={sum(ents)/len(ents):.3f}")
            if rews:
                parts.append(f"rew={sum(rews)/len(rews):.3f}")
            d_tf = toolfail[0] - last_toolfail[0]
            last_toolfail[0] = toolfail[0]
            parts.append(f"toolfail+{d_tf}")
            emit(f"step {step:4d} " + " ".join(parts))

        if step >= first_due and (step - first_due) % interval == 0:
            try:

                cmd = [PY, CHECKER, "--step", str(step), "--run", run,
                       "--log", path, "--read-only"]
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=20,
                )
                for ln in ((r.stdout or "") + (r.stderr or "")).splitlines():
                    emit(ln)
                if r.returncode != 0:
                    emit(f"[REVIEW] ^^ exit={r.returncode} — review record incomplete; no forced write.")
            except Exception as e:  
                emit(f"[REVIEW] checker failed to run: {e}")

if __name__ == "__main__":
    main()
