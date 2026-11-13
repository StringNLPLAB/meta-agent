"""Sequential independent Tau2 rounds; transfer only a training experience document."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--seed-stride", type=int, default=0, help="0 holds the policy RNG seed fixed across adaptive rounds")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--coser-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--audit-command")
    parser.add_argument("--native-reward", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() or not 1 <= args.rounds <= 20:
        raise ValueError("Use a fresh campaign output and 1–20 rounds")
    output.mkdir(parents=True)
    experience = None
    for index in range(args.rounds):
        run = output / f"round_{index + 1}"
        command = [sys.executable, str(ROOT / "examples/tau2/train.py"), "--workspace", args.workspace,
                   "--output", str(run), "--policy", args.policy, "--seed", str(args.seed + index * args.seed_stride),
                   "--gpus", str(args.gpus), "--coser-url", args.coser_url]
        if args.audit_command:
            command += ["--audit-command", args.audit_command]
        if args.native_reward:
            command += ["--native-reward"]
        if experience:
            command += ["--experience", str(experience)]
        subprocess.run(command, cwd=ROOT, check=True)
        # The next round never loads this round's executable drafts or weights.
        lessons = (run / "experience.md").read_text()
        history = ""
        history += "\n\n## Completed training round " + str(index + 1) + "\n\n"
        for name in ("contract.json", "pinned_updates.jsonl", "metrics.jsonl"):
            text = (run / name).read_text()
            history += f"### {name}\n\n```json\n{text[-12000:]}\n```\n\n"
        experience = output / f"experience_after_round_{index + 1}.md"
        # Put the most recent lessons first; bounded audit packets read the head.
        experience.write_text("# Prior training experience\n\n" + lessons[:16000] + "\n" + history[-32000:])
        print(json.dumps(dict(completed_round=index + 1, experience=str(experience))), flush=True)


if __name__ == "__main__":
    main()
