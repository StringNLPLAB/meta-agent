"""Launch with only a verified train workspace, original policy and fresh output."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.colbench.prepare import verify


def docker_command(*, image, workspace, policy, output, devices, seed=42,
                   user_url="http://127.0.0.1:18081/v1", user_model="coser-usersim", audit_command=None):
    devices = devices.split(",")
    if len(devices) not in (2, 4) or len(set(devices)) != len(devices) or any(not x.isdigit() for x in devices):
        raise ValueError("Assign exactly two or four distinct numeric policy GPUs")
    inputs = [Path(x).resolve() for x in (workspace, policy, output)]
    if any("," in str(x) for x in inputs):
        raise ValueError("Docker bind paths cannot contain commas")
    command = ["docker", "run", "--rm", "--init", "--gpus", '"device=' + ",".join(devices) + '"',
        "--network", "host", "--shm-size", "16g", "--pids-limit", "1024",
        "--read-only", "--tmpfs", "/tmp:rw,exec,nosuid", "--tmpfs", "/root:rw,nosuid",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--mount", f"type=bind,src={inputs[0]},dst=/training,readonly",
        "--mount", f"type=bind,src={inputs[1]},dst=/policy,readonly",
        "--mount", f"type=bind,src={inputs[2]},dst=/runs", "--env", "COLBENCH_USER_API_KEY"]
    if audit_command:
        path = Path(audit_command).resolve()
        if "," in str(path):
            raise ValueError("Docker bind paths cannot contain commas")
        command += ["--mount", f"type=bind,src={path},dst=/audit-command.json,readonly",
                    "--env", "META_AGENT_API_KEY", "--env", "META_AGENT_BASE_URL"]
    command += [image, "python", "examples/colbench/train.py", "--workspace", "/training",
        "--policy", "/policy", "--output", "/runs/experiment", "--seed", str(seed), "--gpus", str(len(devices)),
        "--user-url", user_url, "--user-model", user_model]
    if audit_command:
        command += ["--audit-command", "/audit-command.json"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "workspace", "policy", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--user-model", default="coser-usersim")
    parser.add_argument("--audit-command")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    verify(args.workspace)
    output = Path(args.output)
    if output.exists() or not (Path(args.policy) / "config.json").is_file():
        raise ValueError("Use fresh output and an existing original policy directory")
    command = docker_command(**{k: v for k, v in vars(args).items() if k != "dry_run"})
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, command=command, evaluation_volumes=0,
                              network_access="User/meta-agent services remain reachable; public downloads are not blocked"), indent=2))
        return
    output.mkdir(parents=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
