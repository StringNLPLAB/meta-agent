# τ³-bench training with the InfoPO verl fork

This recipe brings over the handover's actual distributed training path: `verl.trainer.main_ppo` → Ray/FSDP actor → synchronous SGLang multi-turn rollout → native τ³-bench user/environment → Info-GRPO advantage computation → optimizer update and checkpoint. It is separate from the small [CPU/minimal trainer](../tau3_minimal/README.md).

The outer coding agent can resolve all paths, model choices, GPU and step budgets, and review cadence from a natural-language request. It prepares a recipe, checks it, and starts the [owned runtime](../../runtime/README.md). The commands below document that agent-facing interface; the experiment owner does not need to start supervision in a separate shell.

## Dependencies and external resources

Use **Python 3.12** in a fresh virtual environment. Although the upstream benchmark declares 3.12–3.13, this inspected checkout imports `audioop`, which Python 3.13 removed; the migrated and resolved training profile targets 3.12. The external τ³-bench checkout must expose package `tau2` version **1.0.1**, contain its official `data/`, and be installed with the `[gym]` extra. The benchmark retained the `tau2` package/directory names when it introduced τ³. Its code/data is not copied into this repository.

Install the compatible CUDA/SGLang dependency profile using the [training installation instructions](../../requirements/README.md), then install the benchmark and the thin Gym bridge in that same interpreter:

```bash
bash requirements/install_training.sh \
  --python /absolute/path/to/venv/bin/python \
  --benchmark-root /absolute/path/to/tau2-bench
```

The profile includes FlashAttention 2 and SGLang 0.4.7; compiling its CUDA extensions needs an appropriate CUDA toolkit. Use the profile's Torch/Transformers versions together. An existing environment with a different SGLang release is not a compatible substitute. Installation is explicit; launching an experiment never installs packages, downloads policy weights, or starts inference servers.

Supply a local Hugging Face causal policy directory, sufficient CUDA memory for its context/batch, and a separately running OpenAI-compatible user simulator. An OpenCode coding model endpoint is a third, separate resource. The recipe defaults to one GPU, two prompts × two rollouts, eight turns, and ten optimizer steps; this is a bounded configuration, not a promise that an arbitrary model fits on any GPU. Reduce context/turn budgets or choose suitable hardware after inspecting the model. A user-model credential is passed by environment variable name only; no `.secrets`, `.env`, or embedded credential fallback is read.

## Prepare, inspect, and run

The outer agent generates real Parquet from the official train split. Preparation loads native tasks, public domain policy and tool schemas without calling the user model:

```bash
/absolute/path/to/venv/bin/python examples/tau3_verl/prepare.py \
  --benchmark-root /absolute/path/to/tau2-bench \
  --model-path /absolute/path/to/local-policy \
  --user-model openai/local-user \
  --user-base-url http://127.0.0.1:31000/v1 \
  --domain retail --reward-mode action \
  --run TAU3_VERL_001 --steps 10 --gpus 1 \
  --runtime-output /absolute/path/to/tau3-launch.json \
  --backend opencode --meta-model flint \
  --meta-base-url http://127.0.0.1:8000/v1 \
  --every-steps 1 --every-seconds 600 --max-calls 30
```

For authenticated endpoints, add `--user-api-key-env USER_MODEL_API_KEY` and/or `--meta-api-key-env META_MODEL_API_KEY` after setting those variables in the launcher environment. The prepared recipe and runtime JSON contain the variable names.

Inspect the composed Hydra configuration and literal training command without starting CUDA, training, inference, or model downloads:

```bash
/absolute/path/to/venv/bin/python examples/tau3_verl/launch.py \
  --recipe reports/tau3_verl/TAU3_VERL_001/recipe.json --dry-run
```

`--dry-run` validates artifact identities and composes the real inherited `ppo_trainer` configuration. It reports missing dependencies and installed versions, but does not certify GPU/model/server readiness. The actual launch additionally requires the dependencies, compatible SGLang, and the configured CUDA devices. Start training and periodic supervision through one owner:

```bash
python3 runtime/run_experiment.py --config /absolute/path/to/tau3-launch.json --detach
```

If supervision is deliberately unnecessary, the same validated recipe starts the actual trainer directly:

```bash
/absolute/path/to/venv/bin/python examples/tau3_verl/launch.py \
  --recipe reports/tau3_verl/TAU3_VERL_001/recipe.json
```

`prepare.py --backend codex` generates a Codex runtime configuration instead. JSON and Hydra settings remain inspectable implementation artifacts; requests for resource choices and parameter changes can be expressed to the outer agent in natural language.

## Data, rewards, and evidence

Preparation fails if the native loader does not return exactly the declared official train IDs. A deterministic 90/10 split of those IDs produces disjoint `train.parquet` and `dev.parquet`; the latter satisfies verl's validation-dataloader contract. Validation is disabled by default (`val_before_train=false`, `test_freq=-1`). No official test/base task outcomes are used for steering or checkpoint selection. The manifest records the source/data hashes, task IDs, partition, and explicit reward proxy. Hidden task criteria and private user instructions stay inside native Gym/UserSimulator/evaluator and never become policy inputs or dataset prompts. Public tools and domain policy retain their native behavior.

ACTION or ENV is an explicit **training reward proxy**, not the official τ³ benchmark score. Neither invokes the default ALL evaluation's external LLM judge. The bridge uses the native user simulator, rejects non-train tasks, propagates user/service/evaluator failures, and treats controller truncation as zero. It replaces the handover's custom flow user and its silent error-to-zero behavior. Reaching a planned episode cutoff is recorded distinctly from a service failure.

The bounded plan keeps the final planned checkpoint without held-out selection. Real training diagnostic trajectories are written under `reports/tau3_verl/<RUN>/checkpoints/debug_trajectories/`, with origin and sampling-bias metadata. These outcome-stratified samples help inspect behavior; they do not estimate population success rates. Supervision reads them and `reports/runtime/<RUN>/trainer.stdout` at the configured step/time boundaries. Runtime and ATIF outcome/trajectory records follow the existing ownership and export rules.

Dataset files, model weights, generated logs, and old experiment results are not committed. Reusing an existing output or silently resuming an interrupted experiment is rejected; prepare a new run only after checking its lineage.

## Validation scope

CPU tests exercise actual Hydra inheritance, native tool/dataset contracts, real Parquet serialization, strict train partitioning, lifecycle config, and failure/cutoff behavior:

```bash
python3 -m unittest discover -s tests -p 'test_tau3_verl.py' -v
```

Native preparation can also be checked against the external benchmark without a model server. Running the distributed GPU training itself requires the documented dependencies, local weights, available GPUs, and a reachable user simulator; CPU preflight does not establish that a GPU optimizer step succeeded.
