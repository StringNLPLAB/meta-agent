"""Fresh ColBench rounds with prose experience; no inherited executable state."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from meta_agent.colbench_runtime import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=0)
    parser.add_argument("--gpus", type=int, choices=(2, 4), default=2)
    parser.add_argument("--user-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--user-model", default="coser-usersim")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--full-checkpoints", action="store_true")
    parser.add_argument("--audit-command")
    parser.add_argument("--audit-every", type=int, default=20)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() or not 1 <= args.rounds <= 20:
        raise ValueError("Use a fresh campaign directory and1–20 rounds")
    output.mkdir(parents=True)
    experience = None
    write(output / "campaign.json", dict(rounds=args.rounds, seed=args.seed, seed_stride=args.seed_stride,
        original_policy=str(Path(args.policy).resolve()), transfer="Selected training-only prose experience",
        statistical_scope="Adaptive round chain; rounds are not independent training seeds"))
    for index in range(args.rounds):
        run = output / f"round_{index + 1}"
        command = [sys.executable, str(ROOT / "examples/colbench/train.py"), "--workspace", args.workspace,
            "--output", str(run), "--policy", args.policy, "--seed", str(args.seed + index * args.seed_stride),
            "--gpus", str(args.gpus), "--user-url", args.user_url, "--user-model", args.user_model,
            "--audit-every", str(args.audit_every)]
        if args.audit_command:
            command += ["--audit-command", args.audit_command]
        if args.steps is not None:
            command += ["--steps", str(args.steps)]
        if args.full_checkpoints:
            command += ["--full-checkpoints"]
        if experience:
            command += ["--experience", str(experience)]
        subprocess.run(command, cwd=ROOT, check=True)
        lessons = (run / "experience.md").read_text()
        history = "\n\n## Completed training round " + str(index + 1) + "\n\n"
        for name in ("contract.json", "pinned_updates.jsonl", "metrics.jsonl", "completed.json"):
            text = (run / name).read_text()
            history += f"### {name}\n\n```json\n{text[-12000:]}\n```\n\n"
        experience = output / f"experience_after_round_{index + 1}.md"
        experience.write_text("# Prior training experience\n\n" + lessons[:16000] + history[-32000:])
        write(output / "status.json", dict(completed_round=index + 1, rounds=args.rounds,
              experience=str(experience), evaluation_feedback_used=False))
        print(json.dumps(dict(completed_round=index + 1, experience=str(experience))), flush=True)


if __name__ == "__main__":
    main()
