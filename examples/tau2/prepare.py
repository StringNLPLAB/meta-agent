"""Prepare a detached, hash-bound Tau2 train OR evaluation workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REVISION = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
UPSTREAM = "https://github.com/sierra-research/tau2-bench.git"
TRAIN_SHA = "621ef8b70d53e1a52ea6c8198d9ef02fdaf159bb51041e7c184e072a691ad095"
COUNTS = {"train": {"retail": 74, "airline": 30, "telecom": 74},
          "test": {"retail": 40, "airline": 20, "telecom": 40}}
PINS = {
    "retail/tasks.json": "8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8",
    "retail/split_tasks.json": "ed0580ec52575b63fbf76568af42490da6ee7783ecb4aa81af46961291358f20",
    "airline/tasks.json": "ccd8ba737b4cc371415af70151187788f728d6108d0916e73bb4317b40542052",
    "airline/split_tasks.json": "b22ced4d9a9850ac9aea31c53bdcb6d6009058140bd9acc7db37c1d36222ba8b",
    "telecom/tasks.json": "37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4",
    "telecom/split_tasks.json": "605b488bb9a6acb3c7f4505240a855fdc8681d09aadb16a8f38b2efcfc5c3aec",
}
SHARED = {"retail": ("db.json", "policy.md"), "airline": ("db.json", "policy.md"),
          "telecom": ("db.toml", "user_db.toml", "main_policy.md", "tech_support_manual.md", "tech_support_workflow.md")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def verify(workspace, *, split):
    workspace = Path(workspace).resolve(strict=True)
    manifest = json.loads((workspace / "manifest.json").read_text())
    if manifest["split"] != split or manifest["benchmark_revision"] != REVISION:
        raise ValueError("Wrong workspace split or benchmark revision")
    actual = {p.relative_to(workspace).as_posix(): sha(p)
              for scope in ("benchmark", "data") for p in (workspace / scope).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual != manifest["files"] or any(p.is_symlink() for p in workspace.rglob("*")):
        raise ValueError("Prepared data/source census changed")
    return manifest


def materialize(source, destination, split):
    source, destination = Path(source).resolve(strict=True), Path(destination).absolute()
    if destination.exists() or destination.resolve() != destination or destination.is_relative_to(source):
        raise ValueError("Use a fresh canonical destination outside the source checkout")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("The upstream checkout must be the pinned v1.0.1 commit")
    # A dirty source checkout must not silently acquire the pinned commit's identity.
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if status:
        raise ValueError("The pinned upstream checkout has tracked modifications")
    for name, digest in PINS.items():
        if sha(source / "data/tau2/domains" / name) != digest:
            raise ValueError("Original Tau2 task/split hash differs: " + name)
    benchmark = destination / "benchmark"
    benchmark.mkdir(parents=True)
    shutil.copytree(source / "src", benchmark / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(source / name, benchmark / name)
    for domain, count in COUNTS[split].items():
        original = source / "data/tau2/domains" / domain
        target = benchmark / "data/tau2/domains" / domain
        splits = json.loads((original / "split_tasks.json").read_text())
        ids = list(map(str, splits[split]))
        tasks = [t for t in json.loads((original / "tasks.json").read_text()) if str(t["id"]) in ids]
        if len(ids) != count or len(tasks) != count or len(set(ids)) != count:
            raise ValueError("Unexpected task counts or duplicate IDs")
        write(target / "tasks.json", tasks)
        write(target / "split_tasks.json", {split: ids, "base": ids})
        for name in SHARED[domain]:
            shutil.copyfile(original / name, target / name)
    target = benchmark / "data/tau2/user_simulator"
    target.mkdir(parents=True)
    for name in ("simulation_guidelines.md", "simulation_guidelines_tools.md"):
        shutil.copyfile(source / "data/tau2/user_simulator" / name, target / name)
    (destination / "data").mkdir()
    if split == "train":
        original = ROOT / "examples/tau2/data/train.parquet"
        if sha(original) != TRAIN_SHA:
            raise ValueError("Bundled public InfoPO training parquet changed")
        shutil.copyfile(original, destination / "data/train.parquet")
        import pyarrow.parquet as pq
        rows = pq.read_table(original, columns=["domain", "reward_model"]).to_pylist()
        for domain, count in COUNTS[split].items():
            ids = [str(r["reward_model"]["id"]) for r in rows if r["domain"] == domain]
            expected = json.loads((benchmark / "data/tau2/domains" / domain / "split_tasks.json").read_text())[split]
            if len(ids) != count or set(ids) != set(expected):
                raise ValueError("Training parquet disagrees with the native train split")
    files = {p.relative_to(destination).as_posix(): sha(p)
             for scope in ("benchmark", "data") for p in (destination / scope).rglob("*") if p.is_file()}
    manifest = dict(schema_version=1, split=split, benchmark_revision=REVISION, upstream=UPSTREAM,
                    counts=COUNTS[split], files=files, source_task_sha256=PINS,
                    training_parquet_sha256=TRAIN_SHA if split == "train" else None,
                    security_scope="Detached selected split; no upstream git history or other task split copied. Shared native databases remain intact. Process/network isolation is a separate deployment setting.")
    write(destination / "manifest.json", manifest)
    return verify(destination, split=split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--split", choices=COUNTS, default="train")
    parser.add_argument("--source", help="Optional clean, pinned offline upstream checkout")
    args = parser.parse_args()
    if args.source:
        result = materialize(args.source, args.destination, args.split)
    else:
        # Full source and Git metadata exist only in this temporary bootstrap directory.
        with tempfile.TemporaryDirectory(prefix="tau2-bootstrap-") as temporary:
            source = Path(temporary) / "source"
            subprocess.run(["git", "clone", "--depth", "1", "--branch", "v1.0.1", UPSTREAM, str(source)], check=True)
            result = materialize(source, args.destination, args.split)
    print(json.dumps({"workspace": str(Path(args.destination).resolve()), "split": args.split,
                      "tasks": sum(result["counts"].values()), "files": len(result["files"])}))


if __name__ == "__main__":
    main()
