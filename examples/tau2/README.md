# Standalone Tau2 experiments

This branch contains the customized verl/InfoPO engine, the Tau2 training
adapter, the original public 178-row training parquet, non-blocking audit hooks,
CoSER serving template, and launch/evaluation commands. It runs on ordinary
local Linux CUDA machines without a platform SDK or another project checkout.
Weights and the native benchmark are downloaded separately. Commands below run
from this repository's root.

## Protocol

| Component | Default |
|---|---|
| Policy | Original Qwen/Qwen3-4B; byte identities in `model_identity.json` |
| Training user | Neph0s/CoSER-Llama-3.1-8B, revision `80d5c88072599026bd0b0e2eb4369a87e6a5c960`; temperature 0.7, 512 tokens |
| Starting flow | Empty implementation; flow probability 0 |
| Algorithm | InfoPO / info_grpo, teacher-forced HPS, gamma 0.8, intrinsic weight 0.1, gate temperature 0.05 |
| Training budget | 22 updates × 16 prompts × 5 online rollouts = 1,760 episodes per run |
| Optimizer | LR 1e-6; one PPO epoch; clip 0.2; entropy 0; no KL loss |
| Policy sampling | Non-thinking; temperature 0.6; 1,024 tokens/turn; 50 turns; prompt 8,192 + response 32,768 tokens |
| Training reward | Native ACTION plus first successful tool 0.02, first grounded user ID 0.07, first grounded object ID 0.09; all other milestone terms zero |
| Alternative starting reward | `--native-reward`: native ACTION only, all intermediate weights zero |
| Train split | 178 tasks: retail 74, airline 30, telecom 74 |
| Final selection | Last update 22; no validation or held-out feedback in training |
| Evaluation | Separate 100 tasks: retail 40, airline 20, telecom 40; 4 trials/task; native ALL |
| Evaluation models | gpt-4o-mini user and NL-assertion judge, both temperature 0.7; policy temperature 0.7, 8,192 tokens, 50 steps |

The live SGLang scheduler input limit is checked before every generation; its reserved tokens are respected without truncating history or enlarging the configured context.

The explicit non-thinking switch takes precedence over model-name detection,
including weights downloaded into a directory named `base`. Every actual policy
generation prompt must include the closed empty thinking suffix. Output
deviations are counted and the raw on-policy samples are retained without
stripping, filtering or resampling. A marker in a completion does not terminate
training. Per-worker receipts under `nonthinking_behavior/` are updated
periodically and flushed when saving checkpoints.

ACTION training reward and native ALL task success are different quantities.
The B0 milestone profile is explicit; do not describe it as native-only Tau2.
`--native-reward` removes these milestones while retaining the same adapter and
format handling. Transport failures stop an attempt rather than producing
fabricated training/evaluation outcomes. Invalid drafts are rejected while the
previous complete configuration stays active. Empty non-tool CoSER
outputs terminate the episode; they are not retried into a different sample.

## Install

Use Python 3.12, NVIDIA CUDA 12 and a compatible driver/compiler. The documented
GPU profile uses Torch 2.7.1, Transformers 4.52.3 and SGLang 0.4.7. Its package
constraints are separate from newer vLLM profiles.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools>=77,<81' wheel packaging ninja 'numpy<2'
python -m pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1
MAX_JOBS=4 python -m pip install --no-build-isolation -r requirements/tau2.txt
```

FlashAttention is a compiled dependency. Its installation needs the matching
CUDA toolchain; the CPU tests do not validate that build. For CPU-only
benchmark/configuration checks, install `requirements/tau2-cpu.txt` instead.

## Prepare training data and models

The benchmark is pinned to official v1.0.1 commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992`. Six original task/split SHA256 values
are checked. Preparation copies native source, unchanged shared databases and
policies, and **only the selected task split** into a fresh directory. It does
not copy upstream Git history or other task variants. The temporary download
is removed. `--source` accepts a clean checkout of that exact commit offline.

```bash
python examples/tau2/prepare.py --destination /scratch/tau2-train --split train
python examples/tau2/models.py --directory /scratch/tau2-models
```

The bundled `data/train.parquet` is byte-identical to the public
[InfoPO training parquet](https://github.com/kfq20/InfoPO/blob/d230e78afb3890eb1c7eec2f27f75bcc6dceed8d/data/tau2_multiturn/train.parquet),
SHA256 `621ef8b70d53e1a52ea6c8198d9ef02fdaf159bb51041e7c184e072a691ad095`.
Preparation verifies that every parquet row belongs to the official train IDs.
No evaluation parquet or held-out task labels are committed in this branch.
The model download helper records resolved revisions and verifies original
policy file hashes. Existing weights can be used directly if those hashes
match. No weights or API keys belong in Git.

## Serve CoSER, validate, and train

The default allocation is **three GPUs total**: two policy GPUs and one CoSER
GPU. Capacity depends on the GPU memory and sequence lengths; this recipe is
intended for large-memory accelerators, not a promise that any three cards fit.
The user service is explicit and can also live on another machine.

```bash
# Terminal 1: keep this service running. The template preserves native user tools.
CUDA_VISIBLE_DEVICES=2 python examples/tau2/serve.py \
  --role user --model /scratch/tau2-models/user --port 18081 \
  --receipt /scratch/services/coser.json

# Terminal 2: compose the actual configuration with zero model calls or updates.
python examples/tau2/train.py --workspace /scratch/tau2-train \
  --policy /scratch/tau2-models/policy --output /scratch/runs/seed43 --dry-run

CUDA_VISIBLE_DEVICES=0,1 python examples/tau2/train.py \
  --workspace /scratch/tau2-train --policy /scratch/tau2-models/policy \
  --output /scratch/runs/seed43 --seed 43 --gpus 2
```

Each attempt requires a new output directory and original policy weights.
`logs/training.log`, `resolved_config.json`, `contract.json`, training trajectory
dumps, per-episode uptake, snapshot hashes, final checkpoint and a cost receipt
stay in that directory. A successful process exit alone is insufficient: the
launcher also requires `checkpoints/global_step_22/actor`.
Cost receipts cover the launched trainer attempt; record allocation idle time,
CoSER serving time, restarts and meta-agent API costs separately for a paper's
cumulative budget comparison. No equal-total-compute claim follows from equal
per-run update counts.

## Non-blocking meta-agent and hot interfaces

Omitting `--audit-command` runs the fixed baseline. To enable editing, provide
a JSON argv file for a fresh-process coding backend. For example, configure
your Pi provider/model through its supported environment or explicit CLI
options, then save a backend command such as:

```json
["pi", "--print", "{prompt}"]
```

Save an audit command JSON with absolute paths to your Python, checkout, and
backend command file:

```json
["/path/to/python", "/path/to/meta-agent/examples/tau2/agent.py",
 "--request", "{request}", "--backend-command", "/path/to/backend.json"]
```

Pass `--audit-command /path/to/audit.json --audit-every 4` to `train.py`.
The external coding CLI/model is installed and configured separately; the
trainer does not silently choose or download a meta-agent. Direct commands
that already implement the request/context contract can replace `agent.py`.
The supplied adapter starts a new process with fresh Pi/OpenCode state
directories and passes no resume/session ID. It expects natural-language
completion and direct file edits, not a JSON proposal or a proxy executor.
For a custom Flint provider, add `--pi-models /path/to/models.json` to the
audit command and select its provider/model in the backend argv. This copies
only that explicit provider configuration into the fresh Pi directory; old
sessions and conversation history are never copied. Keep credentials in
environment variables rather than committing them in the provider file.

| Channel | Editable files under the run directory | Consumer |
|---|---|---|
| User | `drafts/user/flow.py`, `flow_user.py`, `env_policy.json` | Fresh CoSER bootstrap; increase `tau2_flow.flow_probability` to enable authored flow |
| Environment/reward | `drafts/environment/environment.py`, `environment_config.json` | `EnvironmentMechanism.on_reset`, `before_action`, `after_action`; finite reward weights and max steps |
| Agent/RL | `drafts/agent/entropy_schedule.json`, `clip_schedule.json` | Existing bounded, rate-limited entropy and clip consumers |

An entropy override is `{"enable": true, "value": 0.005}`. A clip override is
`{"enable": true, "override": 0.25}`. Disabled files keep static defaults.
Initial environment hooks are identity functions and initial flow is empty.
The agent receives bounded snapshots of training logs, trajectory excerpts,
training metrics and `experience.md`. It may record lessons in that prose
document. It receives no evaluation results through this interface.

Audits run in a background process. Triggers coalesce while an audit is busy;
they do not queue a blocking audit for every update. Syntax/schema and module
ABI checks run at the next update boundary. Published Python bundles are
immutable per episode, and active scalar files change before a rollout batch.
A bad draft is recorded and retains the previous configuration; an I/O failure
after publication starts aborts the attempt for reconciliation. These checks establish
integrity and interface validity, not a proof that a proposed change improves
learning. Episode uptake is recorded separately from observed file writes.
The direct editor is stopped/reaped before the final update; a new writer is
not dispatched after training ends.

For five rounds with fresh policy, user/environment defaults and sessions:

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/tau2/campaign.py \
  --workspace /scratch/tau2-train --policy /scratch/tau2-models/policy \
  --output /scratch/campaign --rounds 5 --seed 43 --native-reward \
  --audit-command /path/to/audit.json
```

Only a generated training experience document passes between rounds. Previous
weights, drafts, optimizer state and agent sessions are never resumed. Seed 43
is held fixed by default; `--seed-stride 1` instead runs seeds 43–47. Independent
baseline seeds can be run without an audit command. These are distinct
experimental designs and should be labeled accordingly.

## Evaluation boundary

The detached train workspace contains no held-out task objects or ID lists;
native shared databases and public domain policies remain unchanged. This is
a **data-minimization property, not an OS sandbox**. A host process with broad
filesystem/network permissions can still read another checkout, historical
archive or public benchmark download. Do not claim eval invisibility from this
Python launcher alone.

For filesystem isolation, run the trainer and coding CLI inside a container or
Harbor environment that mounts only this checkout, the detached train
workspace, model directories and a fresh run directory. Mount model/code/data
inputs read-only and the run directory writable. Do not mount the Docker
socket, a host home directory, bootstrap checkout, evaluation volume, API
credentials for evaluation, or the host PID namespace. Install dependencies
in the image before launch. The evaluation command runs in a separate process
or container with its own data and credentials. Internet access must also be
restricted if the threat model includes fetching public held-out data again.
This branch does not label an unrestricted host run as Harbor-isolated.

The supplied Docker launcher makes the filesystem mount list explicit:

```bash
docker build -f examples/tau2/Dockerfile -t tau2-training:local .
python examples/tau2/container.py --image tau2-training:local \
  --workspace /scratch/tau2-train --policy /scratch/tau2-models/policy \
  --output /scratch/container-runs/seed43 --devices 0,1
```

It mounts the detached train workspace and policy read-only, plus a fresh
output directory. Code/dependencies are baked into the image; there is no host
home, full benchmark checkout, eval volume, Docker socket or host PID mount.
The CoSER service remains on the host. `--dry-run` prints the exact mount/launch
arguments without starting Docker. Host networking is explicit, so this is
filesystem isolation and does **not** block network re-downloads. The Docker
image build/GPU launch must be validated on the target CUDA machine; the CPU
test checks the actual launcher argument construction.

For audits, extend the image with your coding CLI and a backend argv file,
then pass `--audit-command` with a JSON file whose command uses image paths
such as `/opt/tau2/bin/python` and `/code/examples/tau2/agent.py`. Optional
`META_AGENT_API_KEY` and `META_AGENT_BASE_URL` are inherited by name, never
embedded in Docker argv, and are applied only by the coding CLI adapter.

## Merge and evaluate the final policy

Run preparation/evaluation outside the training container and never expose its
output or credentials to the coding agent:

```bash
python examples/tau2/prepare.py --destination /scratch/tau2-heldout --split test
python eval/merge.py merge --backend fsdp \
  --local_dir /scratch/runs/seed43/checkpoints/global_step_22/actor \
  --target_dir /scratch/policies/seed43

# Terminal 1, after training has finished; use one policy GPU.
CUDA_VISIBLE_DEVICES=0 python examples/tau2/serve.py \
  --role policy --model /scratch/policies/seed43 --port 18080 \
  --receipt /scratch/services/seed43-policy.json

# Terminal 2: export EVAL_BASE_URL and EVAL_API_KEY through your secret manager.
python examples/tau2/evaluate.py --workspace /scratch/tau2-heldout \
  --output /scratch/evaluations/seed43
```

The supplied policy server fixes `enable_thinking=false` in its external
Qwen3 template. `--thinking` enables the separate thinking comparison. An
arbitrary external policy endpoint must independently honor its declared
thinking setting. Model/user/judge temperature is 0.7 during evaluation.
Native ALL evaluation code is unchanged; the NL-assertion judge's model and
transport are explicitly configured. There is no training reward shaping in
this evaluator. API failures leave the run incomplete rather than becoming
policy failures. After repair, `--resume` continues missing trials only when
the stored evaluation contract matches exactly.

`summary.json` reports success rate and both estimators for k=1..4. For n=4
trials and c successes on a task, pass@k is `1-C(n-c,k)/C(n,k)` and pass^k is
`C(c,k)/C(n,k)`, averaged over tasks. The latter estimates joint success of k
distinct trials; it is not obtained by raising the overall success rate to k.

## Checks and provenance

```bash
python -m unittest discover -s tests
TAU2_TRAIN_WORKSPACE=/scratch/tau2-train python -m unittest discover \
  -s tests -p test_tau2_experiment.py -v
python examples/tau2/evaluate.py --workspace /scratch/tau2-heldout \
  --output /scratch/evaluations/dry-run --dry-run
```

The native CPU integration uses real task/split loading, native domain tools,
bundle loading and reward accounting, with model I/O scripted. It performs no
GPU optimization or benchmark performance evaluation. A successful dry run is
configuration validation, not evidence that CUDA training has completed.
Source provenance and local engine patches are in `provenance.json` and
`verl/vendor_provenance.json`; inherited licenses are retained in `NOTICE`.
