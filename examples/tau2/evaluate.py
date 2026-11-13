"""Frozen native ALL evaluation: 100 held-out tasks, four trials each."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tau2.prepare import verify, write


def summarize(rows, trials=4):
    grouped = {}
    for row in rows:
        key = (row["domain"], str(row["task_id"]))
        trial = row["trial"]
        if type(trial) is not int or not 0 <= trial < trials or trial in grouped.setdefault(key, {}):
            raise ValueError("Duplicate or invalid trial")
        reward = row["reward"]
        if type(reward) not in (int, float) or reward not in (0, 1):
            raise ValueError("Native ALL outcomes must be binary")
        grouped[key][trial] = reward
    if not grouped or any(len(v) != trials for v in grouped.values()):
        raise ValueError("Every selected task must have exactly four successful evaluator calls")
    successes = [sum(v.values()) for v in grouped.values()]
    n = len(successes)
    return dict(tasks=n, trials=trials, evaluated_episodes=n * trials,
                success_rate=sum(successes) / (n * trials),
                pass_at_k={str(k): sum(1 - (math.comb(trials - s, k) / math.comb(trials, k)
                    if trials - s >= k else 0) for s in successes) / n for k in range(1, trials + 1)},
                pass_power_k={str(k): sum(math.comb(s, k) / math.comb(trials, k)
                    if s >= k else 0 for s in successes) / n for k in range(1, trials + 1)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Separate workspace prepared with --split test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--policy-model", default="tau2-policy")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Continue only missing trials of the identical evaluation contract")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    workspace, output = Path(args.workspace).resolve(strict=True), Path(args.output).resolve()
    manifest = verify(workspace, split="test")
    if not 1 <= args.concurrency <= 32:
        raise ValueError("concurrency must be in [1,32]")
    contract = dict(schema_version=1, evaluation_type="ALL", tasks=100, trials=4, seed=42,
        user_model=args.judge_model, judge_model=args.judge_model, user_temperature=0.7, judge_temperature=0.7,
        policy_temperature=0.7, policy_max_tokens=8192, max_steps=50,
        policy_model=args.policy_model, policy_url=args.policy_url,
        benchmark_manifest_sha256=hashlib.sha256((workspace / "manifest.json").read_bytes()).hexdigest())
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, model_calls=0, contract=contract), indent=2))
        return
    key, base = os.environ.get("EVAL_API_KEY"), os.environ.get("EVAL_BASE_URL")
    if not key or not base:
        raise ValueError("Export EVAL_API_KEY and EVAL_BASE_URL for the frozen evaluation user and judge")
    if output.exists():
        if not args.resume or json.loads((output / "contract.json").read_text()) != contract:
            raise ValueError("Existing evaluation requires --resume and an identical contract")
    else:
        output.mkdir(parents=True)
        write(output / "contract.json", contract)
    os.environ.update(TAU2_DATA_DIR=str(workspace / "benchmark/data"), PYTHON_DOTENV_DISABLED="1",
                      LITELLM_LOCAL_MODEL_COST_MAP="True", LOGURU_LEVEL="ERROR")
    sys.path.insert(0, str(workspace / "benchmark/src"))
    # These constants are copied by evaluator modules at import time.
    import tau2.config as native_config
    native_config.DEFAULT_LLM_NL_ASSERTIONS = args.judge_model
    native_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(temperature=0.7, api_base=base, api_key=key,
                                                      timeout=120, num_retries=0)
    import tau2.evaluator.evaluator_nl_assertions as native_judge
    native_judge.DEFAULT_LLM_NL_ASSERTIONS = native_config.DEFAULT_LLM_NL_ASSERTIONS
    native_judge.DEFAULT_LLM_NL_ASSERTIONS_ARGS = native_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS
    from tau2.run import run_task
    from tau2.registry import registry
    from tau2.evaluator.evaluator import EvaluationType
    cases = []
    for domain, count in manifest["counts"].items():
        tasks = registry.get_tasks_loader(domain)()
        if len(tasks) != count:
            raise ValueError("Native held-out task count differs")
        cases.extend((domain, task, trial) for task in tasks for trial in range(4))
    completed = {}
    for path in output.glob("episodes/*/*.json"):
        row = json.loads(path.read_text())
        completed[(row["domain"], str(row["task_id"]), row["trial"])] = row

    def episode(case):
        domain, task, trial = case
        result = run_task(domain=domain, task=task, agent="llm_agent", user="user_simulator",
            llm_agent="openai/" + args.policy_model, llm_user=args.judge_model,
            llm_args_agent=dict(api_base=args.policy_url, api_key=os.environ.get("POLICY_API_KEY", "EMPTY"),
                               temperature=0.7, max_tokens=8192, timeout=120, num_retries=0),
            llm_args_user=dict(api_base=base, api_key=key, temperature=0.7, timeout=120, num_retries=0),
            max_steps=50, evaluation_type=EvaluationType.ALL, seed=42 + trial)
        if result.reward_info is None:
            raise ValueError("Native evaluator returned no reward")
        row = dict(domain=domain, task_id=str(task.id), trial=trial, reward=result.reward_info.reward,
                   termination_reason=result.termination_reason.value,
                   simulation=result.model_dump(mode="json"))
        write(output / "episodes" / domain / f"{task.id}__{trial}.json", row)
        return row

    pending = [c for c in cases if (c[0], str(c[1].id), c[2]) not in completed]
    failures = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(episode, c): c for c in pending}
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
                completed[(row["domain"], row["task_id"], row["trial"])] = row
                print(json.dumps(dict(completed=len(completed), total=400)), flush=True)
            except Exception as error:
                # Do not turn API failures into unsuccessful policy outcomes or print credentials.
                failures.append(dict(domain=case[0], task_id=str(case[1].id), trial=case[2], error_type=type(error).__name__))
    write(output / "incomplete.json", failures)
    expected = {(d, str(t.id), i) for d, t, i in cases}
    if failures or set(completed) != expected:
        raise RuntimeError("Evaluation incomplete; repair the service and rerun with --resume")
    result = summarize(list(completed.values()))
    write(output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
