"""Deterministic train-only ColBench conversion for the Flint trial.

The upstream README references a converter absent from this checkout. This
converter preserves official JSONL order and benchmark task contents. Reference
solutions and test cases enter environment kwargs, never the policy prompt.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = "facebook/collaborative_agent_bench"
OFFICIAL_REVISION = "cf3526da25989b53f105fe9b74c1174a3e19c548"
OFFICIAL_TRAIN_SHA256 = "11e480b7a82a010b13e2e3e69b997865e5bf7a87e06b498858188a105477a95f"


def normalize_test_cases(raw):
    """Mirror ColBenchCodeEnv._normalize_test_cases without importing the gym."""
    if isinstance(raw, dict):
        def order(item):
            name = str(item[0])
            digits = "".join(c for c in name if c.isdigit())
            return (int(digits) if digits else 0, name)
        return [value.strip() for _, value in sorted(raw.items(), key=order)
                if isinstance(value, str) and value.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(value).strip() for value in raw if str(value).strip()]
    raise ValueError("A ColBench training task needs structured test cases")


def make_training_row(task, index, system_prompt):
    description = task.get("problem_description")
    reference = task.get("ground_truth")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Training task {index} has no problem description")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Training task {index} has no reference program")
    cases = normalize_test_cases(task.get("test_cases"))
    if not cases:
        raise ValueError(f"Training task {index} has no usable test cases")
    task_id = f"colbench_backend_train_{index:05d}"
    return {
        "data_source": "interact_colbench_code",
        "prompt": [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": description}],
        "ability": "interaction",
        "reward_model": {"style": "rule", "ground_truth": reference,
                         "env_name": "ColBenchCodeEnv", "id": task_id},
        "extra_info": {
            "split": "train", "index": index, "need_tools_kwargs": True,
            "tools_kwargs": {"interact_with_env": {"create_kwargs": {
                "env_name": "ColBenchCodeEnv", "id": task_id,
                "task": {"problem_description": description, "ground_truth": reference,
                         "test_cases": cases}}}}},
    }


def preprocess(train_data, output_dir, *, expected_source_sha256=OFFICIAL_TRAIN_SHA256,
               tokenizer_path=None, max_prompt_length=1152):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source, output = Path(train_data), Path(output_dir)
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if expected_source_sha256 and source_sha != expected_source_sha256:
        raise ValueError("ColBench source differs from the preregistered official train split")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "train.parquet"
    manifest_path = output / "train_manifest.json"
    if target.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing prepared training split")
    prompt_path = ROOT / "gyms/ColBenchGym/colbenchgym/prompts/llm_agent_code_prompt.txt"
    prompt = prompt_path.read_text().strip()
    tasks = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    rows = [make_training_row(task, index, prompt) for index, task in enumerate(tasks)]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, target, compression="zstd")
    # The trainer constructs a validation loader even when all evaluation is
    # disabled. This explicitly labelled placeholder contains one TRAIN row.
    placeholder = output / "unused_train_loader.parquet"
    pq.write_table(table.slice(0, 1), placeholder, compression="zstd")
    lengths = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True,
                                                   trust_remote_code=False)
        lengths = [len(tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                      add_generation_prompt=True, enable_thinking=False)) for row in rows]
    filtered = sum(length <= max_prompt_length for length in lengths) if lengths is not None else None
    manifest = {
        "dataset": OFFICIAL_REPO, "revision": OFFICIAL_REVISION,
        "source_split": "backend_tasks/train.jsonl", "source_sha256": source_sha,
        "source_rows": len(rows), "row_order": "official source order; no sampling or shuffle",
        "policy_prompt": "fixed agent instructions plus problem_description only",
        "privileged_training_fields": "ground_truth and native test cases in environment kwargs only",
        "test_cases_storage": "ordered list; identical native ordering, avoids heterogeneous Arrow struct keys",
        "agent_prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "train_parquet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "train_parquet_bytes": target.stat().st_size,
        "columns": table.column_names, "test_split_read": False,
        "unused_train_loader": str(placeholder),
        "unused_loader_requirement": "trainer.val_before_train=False and trainer.test_freq=-1; not held-out evaluation",
        "max_prompt_length": max_prompt_length,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "prompt_length_filter_survivors": filtered,
        "nominal_updates_batch32_two_epochs": (len(rows) // 32) * 2,
        "filtered_updates_batch32_two_epochs": (filtered // 32) * 2 if filtered is not None else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--max-prompt-length", type=int, default=1152)
    args = parser.parse_args()
    print(json.dumps(preprocess(args.train_data, args.output_dir,
                               tokenizer_path=args.tokenizer_path,
                               max_prompt_length=args.max_prompt_length), indent=2))


if __name__ == "__main__":
    main()
