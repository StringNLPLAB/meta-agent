"""Prepare the pinned public ColBench backend TRAIN split only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.data_preprocess.colbench_code_multiturn import (
    OFFICIAL_REPO, OFFICIAL_REVISION, OFFICIAL_TRAIN_SHA256, preprocess)
from meta_agent.tau2_runtime import write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists() or destination.resolve() != destination:
        raise ValueError("Use a fresh canonical training workspace")
    # Check before creating any output or parsing privileged training fields.
    if sha(source) != OFFICIAL_TRAIN_SHA256:
        raise ValueError("Source is not the pinned official ColBench training split")
    converted = preprocess(source, destination / "data")
    if converted["source_rows"] != 10000:
        raise ValueError("Expected all 10,000 original backend training tasks")
    files = {p.relative_to(destination).as_posix(): sha(p)
             for p in (destination / "data").iterdir() if p.is_file()}
    manifest = dict(schema_version=1, benchmark="ColBench backend", split="train",
        dataset=OFFICIAL_REPO, revision=OFFICIAL_REVISION, source_sha256=OFFICIAL_TRAIN_SHA256,
        rows=10000, files=files, test_downloaded=False,
        security_scope="Only train rows and a one-row TRAIN loader placeholder; no held-out tasks or Git history. OS isolation is separate.")
    write(destination / "manifest.json", manifest)
    return verify(destination)


def verify(workspace):
    workspace = Path(workspace).resolve(strict=True)
    manifest = json.loads((workspace / "manifest.json").read_text())
    if (manifest.get("split") != "train" or manifest.get("dataset") != OFFICIAL_REPO
            or manifest.get("revision") != OFFICIAL_REVISION or manifest.get("rows") != 10000
            or manifest.get("source_sha256") != OFFICIAL_TRAIN_SHA256
            or manifest.get("test_downloaded") is not False):
        raise ValueError("Wrong ColBench dataset identity or split")
    actual = {}
    for p in workspace.rglob("*"):
        if p.is_symlink():
            raise ValueError("Training workspace cannot contain symlinks")
        if p.is_file() and p != workspace / "manifest.json":
            actual[p.relative_to(workspace).as_posix()] = sha(p)
    expected = {"data/train.parquet", "data/train_manifest.json", "data/unused_train_loader.parquet"}
    if set(actual) != expected or actual != manifest["files"]:
        raise ValueError("Prepared training workspace changed or contains undeclared files")
    import pyarrow.parquet as pq
    if pq.ParquetFile(workspace / "data/train.parquet").metadata.num_rows != 10000:
        raise ValueError("Prepared train row count differs")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--source", help="Optional local pinned backend_tasks/train.jsonl")
    args = parser.parse_args()
    if args.source:
        result = materialize(args.source, args.destination)
    else:
        url = f"https://huggingface.co/datasets/{OFFICIAL_REPO}/resolve/{OFFICIAL_REVISION}/backend_tasks/train.jsonl"
        with tempfile.TemporaryDirectory(prefix="colbench-train-download-") as temporary:
            source = Path(temporary) / "train.jsonl"
            with urllib.request.urlopen(url, timeout=120) as response, source.open("wb") as output:
                import shutil
                shutil.copyfileobj(response, output)
            result = materialize(source, args.destination)
    print(json.dumps(dict(workspace=str(Path(args.destination).resolve()),
                          split="train", rows=result["rows"], test_downloaded=False)))


if __name__ == "__main__":
    main()
