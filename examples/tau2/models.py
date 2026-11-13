"""Download public model weights outside Git and verify the declared identities."""
import argparse
import hashlib
import json
from pathlib import Path

IDENTITY = Path(__file__).with_name("model_identity.json")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_policy(directory):
    expected = json.loads(IDENTITY.read_text())["policy_files"]
    for name, row in expected.items():
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Original Qwen3-4B file identity differs: " + name)
    return {name: row["sha256"] for name, row in expected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--role", choices=("policy", "user", "both"), default="both")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    identity = json.loads(IDENTITY.read_text())
    for role in ("policy", "user"):
        if args.role not in (role, "both"):
            continue
        repo = identity[role + "_repository"]
        revision = identity["user_revision"] if role == "user" else HfApi().model_info(repo).sha
        destination = Path(args.directory).resolve() / role
        snapshot_download(repo, revision=revision, local_dir=destination,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*"])
        if role == "policy":
            verify_policy(destination)
        else:
            expected = "a6f261303b790f7e300534ad56e0b4bc4d8228af052e30288ea09663704fc9c2"
            if sha(destination / "config.json") != expected:
                raise ValueError("CoSER config identity differs")
        print(json.dumps(dict(role=role, repository=repo, revision=revision, directory=str(destination), verified=True)))


if __name__ == "__main__":
    main()
