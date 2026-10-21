
"""Capture and compare EAU run configuration and code fingerprints."""
from eau_paths import root_path, gym_path, canonical_run, select_log
import argparse, hashlib, json, os, re, subprocess, sys, time

ROOT = (root_path() + '')
GYMS = gym_path()
OUT = os.path.join(ROOT, "reports", "manifests")

HOT_JSON = {
    "reward_shaping":  f"{ROOT}/meta_agent/reward_shaping.json",
    "process_reward":  f"{GYMS}/symbolic_layer/process_reward.json",
    "env_policy":      f"{GYMS}/symbolic_layer/env_policy.json",
    "difficulty":      f"{GYMS}/symbolic_layer/difficulty.json",
    "symbolic_config": f"{GYMS}/symbolic_layer/config.json",
    "entropy_schedule": f"{ROOT}/entropy_schedule.json",
    "clip_schedule":   f"{ROOT}/clip_schedule.json",
}

GYM_CONFIGS = {
    "travelgym":  f"{GYMS}/TravelGym/travelgym/config.py",
    "functiongym": f"{GYMS}/FunctionGym/functiongym/config.py",
    "turtlegym":  f"{GYMS}/TurtleGym/turtlegym/config.py",
    "persuadegym": f"{GYMS}/PersuadeGym/persuadegym/config.py",
}
GYM_FIELDS = re.compile(
    r"^\s*(search_correct_reward|preference_correct_reward|correct_answer_reward|"
    r"incorrect_answer_reward|step_penalty|success_threshold|judge_temperature|"
    r"one_choice_per_aspect|search_failure_interval|max_steps)\s*:\s*[\w\[\]]+\s*=\s*(.+?)\s*$",
    re.M)

CODE_PATHS = [
    f"{ROOT}/verl/tools/env_manager.py",
    f"{ROOT}/verl/tools/interact_tool.py",
    f"{ROOT}/verl/workers/reward_manager/reward_shaping.py",

    f"{ROOT}/verl/workers/reward_manager/naive.py",
    f"{ROOT}/verl/workers/rollout/sglang_rollout/sglang_rollout_customized.py",
    f"{ROOT}/verl/trainer/ppo/ray_trainer.py",
    f"{ROOT}/verl/trainer/ppo/core_algos.py",

    f"{ROOT}/verl/workers/actor/dp_actor.py",
    f"{ROOT}/tau-bench/tau_bench/envs/user.py",
    f"{GYMS}/symbolic_layer/env_policy.py",
    f"{GYMS}/symbolic_layer/tau_rules.py",
    f"{GYMS}/TurtleGym/turtlegym/env/prompts.py",
    f"{GYMS}/FunctionGym/functiongym/env/function_env.py",
    f"{GYMS}/PersuadeGym/persuadegym/env/prompts.py",
    f"{GYMS}/TravelGym/travelgym/env/travel_env.py",

    f"{GYMS}/TurtleGym/turtlegym/env/story_env.py",
    f"{GYMS}/PersuadeGym/persuadegym/env/persuade_env.py",
    f"{GYMS}/TelepathyGym/telepathygym/env/telepathy_env.py",
    f"{GYMS}/IntentionGym/intentiongym/env/intention_env.py",
    f"{GYMS}/symbolic_layer/process_reward.py",
    f"{GYMS}/symbolic_layer/difficulty.py",
]

ENV_PREFIXES = ("TRAVELGYM_", "USERRL_", "TAU_", "USER_", "KL_LOSS_COEF", "ENTROPY_COEFF",
                "CLIP_HIGH_FINAL", "CLIP_ZERO_OOR", "EXP", "MODEL_PATH", "EXTRA_ARGS",
                "ROLLOUT_GPU_MEMORY_UTILIZATION")

CLI_KEYS = [

    "trainer.log_train_trajectories","kl_loss_coef", "entropy_coeff", "critic_warmup", "total_epochs",
            "resume_mode", "resume_from_path", "save_freq", "train_batch_size",
            "format_fatal_penalty", "format_no_tool_call_penalty",
            "format_multi_tool_penalty", "format_repair_penalty",
            "tool_call_repair", "format_error_feedback", "clip_ratio_high_schedule.final",
            "response_length_one_turn", "max_turns", "optim.lr", "adv_estimator",
            "turn_reward_mode", "gamma", "lam", "rollout.n", "data.train_files",
            "data.val_files", "test_freq"]

def _sha(path, limit=None):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
                if limit and f.tell() > limit:
                    break
        return h.hexdigest()[:16]
    except Exception:
        return None

def _training_pid(pid=None):
    if pid is None:
        return None
    if pid <= 0:
        raise ValueError("Trainer PID must be positive")
    with open(f"/proc/{pid}/cmdline", "rb") as handle:
        command = handle.read().replace(b"\0", b" ")
    if b"verl.trainer.main_ppo" not in command:
        raise ValueError("Caller-bound PID is not a verl trainer")
    return pid

def _proc_env(pid):
    """Environment variables of the LIVE process. The resolved config dump does
    NOT contain these, which is exactly how TRAVELGYM_SEARCH_CORRECT_REWARD went
    unnoticed across two runs."""
    out = {}
    if not pid:
        return out
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            for kv in f.read().split(b"\0"):
                if not kv:
                    continue
                k, _, v = kv.decode(errors="replace").partition("=")
                if any(k.startswith(p) for p in ENV_PREFIXES):
                    out[k] = v
    except Exception:
        pass
    return out

def _cli_from_log(log):
    """Last value wins, the way hydra resolves it."""
    out = {}
    if not log or not os.path.exists(log):
        return out
    try:
        with open(log, errors="ignore") as f:
            for line in f:
                if "main_ppo" not in line:
                    continue
                for k in CLI_KEYS:
                    for m in re.finditer(re.escape(k) + r"=([^\s]+)", line):
                        out[k] = m.group(1)
                break
    except Exception:
        pass
    return out

def _resolved_from_log(log):
    """A few keys read back out of the trainer's own resolved dump, so the manifest
    records what the process USED rather than what the command line asked for."""
    out = {}
    if not log or not os.path.exists(log):
        return out
    want = ["log_train_trajectories",
            "kl_loss_coef", "entropy_coeff", "critic_warmup", "total_epochs",
            "resume_mode", "gamma", "lam", "clip_ratio_high", "clip_ratio_low",
            "response_length_one_turn", "max_turns", "warmup_style",
            "lr_warmup_steps_ratio", "test_freq", "save_freq"]
    try:
        with open(log, errors="ignore") as f:
            txt = f.read(4_000_000)
        for k in want:
            m = re.search(r"'" + re.escape(k) + r"': ([^,}\n]+)", txt)
            if m:
                out[k] = m.group(1).strip()
        m = re.search(r"actor/kl_coef:([0-9.eE+-]+)", txt)
        if m:
            out["actor/kl_coef (metric)"] = m.group(1)
    except Exception:
        pass
    return out

def capture(run, log=None, note="", trainer_pid=None):
    run = canonical_run(run)
    if log is not None:
        log = select_log(log, run)
    pid = _training_pid(trainer_pid)
    man = {
        "run": run,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": log,
        "training_pid": pid,
        "env_source": "caller-bound PID" if pid else "unobserved (no caller-bound PID)",
        "snapshot_scope": "checkout files; loaded revisions and runtime uptake require separate evidence",
        "note": note,
        "hot_json": {},
        "gym_config_defaults": {},
        "code_sha": {},
        "env": _proc_env(pid),
        "cli_overrides": _cli_from_log(log),
        "resolved": _resolved_from_log(log),
        "data": {},
    }
    for name, p in HOT_JSON.items():
        try:
            with open(p) as f:
                man["hot_json"][name] = json.load(f)
        except Exception as e:
            man["hot_json"][name] = f"<unreadable: {e}>"
    for name, p in GYM_CONFIGS.items():
        try:
            with open(p) as f:
                txt = f.read()
            man["gym_config_defaults"][name] = {
                "sha": _sha(p),
                "fields": {m.group(1): m.group(2) for m in GYM_FIELDS.finditer(txt)},
            }
        except Exception as e:
            man["gym_config_defaults"][name] = f"<unreadable: {e}>"
    for p in CODE_PATHS:
        man["code_sha"][p.replace(ROOT + "/", "").replace(GYMS + "/", "gyms/")] = _sha(p)
    for key in ("data.train_files", "data.val_files"):
        p = man["cli_overrides"].get(key)
        if p and os.path.exists(p):
            st = os.stat(p)
            man["data"][key] = {"path": p, "size": st.st_size,
                                "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                                       time.localtime(st.st_mtime))}
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"{run}.json")
    with open(dst, "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1, sort_keys=True)
    return dst

def _flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        kk = f"{pre}.{k}" if pre else str(k)
        if isinstance(v, dict):
            out.update(_flat(v, kk))
        else:
            out[kk] = v
    return out

SKIP = ("captured_at", "log", "training_pid", "run", "note", "env.EXP", "env.EXTRA_ARGS")

def diff(a, b):
    pa, pb = (os.path.join(OUT, f"{x}.json") for x in (a, b))
    A, B = (json.load(open(p)) for p in (pa, pb))
    fa, fb = _flat(A), _flat(B)
    keys = sorted(set(fa) | set(fb))
    rows = [(k, fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in keys
            if fa.get(k, "<absent>") != fb.get(k, "<absent>")
            and not any(k.startswith(s) for s in SKIP)]
    print(f"=== {a} vs {b}: {len(rows)} differing keys ===")
    for k, x, y in rows:
        print(f"  {k}\n      {a} = {str(x)[:80]}\n      {b} = {str(y)[:80]}")
    if not rows:
        print("  (identical on every captured key)")
    return rows

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--run", required=True)
    c.add_argument("--log"); c.add_argument("--note", default="")
    c.add_argument("--trainer-pid", type=int, help="Caller-verified local PID for this exact run")
    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "capture":
        print("wrote", capture(args.run, args.log, args.note, args.trainer_pid))
    elif args.cmd == "diff":
        diff(args.a, args.b)
    else:
        os.makedirs(OUT, exist_ok=True)
        for f in sorted(os.listdir(OUT)):
            print(" ", f)

if __name__ == "__main__":
    main()
