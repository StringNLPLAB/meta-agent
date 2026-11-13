"""Launch with explicit train/model/output mounts and no evaluation volume.

Host networking permits a separately served CoSER/meta-agent. It does not
prevent downloading public benchmark data over the network.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.tau2.prepare import verify


def docker_command(*, image, workspace, policy, output, devices, seed=43, native_reward=False, audit_command=None):
    workspace, policy, output = Path(workspace).resolve(), Path(policy).resolve(), Path(output).resolve()
    if (not devices or any(not token.isdigit() for token in devices.split(","))
            or len(set(devices.split(","))) != len(devices.split(","))):
        raise ValueError("devices must name distinct numerical GPU IDs")
    command = ["docker", "run", "--rm", "--init", "--gpus", '"device=' + devices + '"',
               "--network", "host", "--shm-size", "16g", "--pids-limit", "1024",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--mount", f"type=bind,src={workspace},dst=/training,readonly",
               "--mount", f"type=bind,src={policy},dst=/policy,readonly",
               "--mount", f"type=bind,src={output},dst=/runs"]
    if audit_command:
        command += ["--mount", f"type=bind,src={Path(audit_command).resolve()},dst=/audit-command.json,readonly"]
        # Docker inherits named variables without putting secret values in argv.
        command += ["--env", "META_AGENT_API_KEY", "--env", "META_AGENT_BASE_URL"]
    command += [image, "python", "examples/tau2/train.py", "--workspace", "/training",
                "--policy", "/policy", "--output", "/runs/experiment", "--seed", str(seed),
                "--gpus", str(len(devices.split(",")))]
    if native_reward:
        command += ["--native-reward"]
    if audit_command:
        command += ["--audit-command", "/audit-command.json"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Prebuilt image with the documented GPU dependencies")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True, help="Fresh host directory; run appears under experiment/")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--native-reward", action="store_true")
    parser.add_argument("--audit-command", help="Optional mounted argv file; its paths must refer to the image filesystem")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    verify(args.workspace, split="train")
    if Path(args.output).exists() or not (Path(args.policy) / "config.json").is_file():
        raise ValueError("Use a fresh output directory and an existing policy directory")
    command = docker_command(image=args.image, workspace=args.workspace, policy=args.policy,
        output=args.output, devices=args.devices, seed=args.seed, native_reward=args.native_reward,
        audit_command=args.audit_command)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, command=command, host_evaluation_mounts=0,
                              network_data_downloads_blocked=False), indent=2))
        return
    Path(args.output).mkdir(parents=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
