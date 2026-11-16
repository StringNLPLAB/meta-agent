# Standalone ColBench training and non-blocking audits

This package ports the existing ColBench **backend-programming** integration
into this repository. It includes the environment, flow user, pinned training
data preparation, a direct PPO/GAE launcher, optional background coding-agent
reviews, and fresh-round experience transfer. It needs no platform SDK or
another private checkout. Actual training requires NVIDIA GPUs and the pinned
Python 3.12 SGLang profile; the user and meta-agent are separately served models.

## Install and prepare

Use the same GPU dependency installation as the
[Tau2 package](../tau2/README.md), then install the ColBench requirements:

```bash
python -m pip install -r requirements/colbench-cpu.txt -e gyms/ColBenchGym
python examples/colbench/prepare.py --destination /training/colbench
```

For offline preparation, add `--source /downloads/backend_train.jsonl`.
The converter checks SHA256
`11e480b7a82a010b13e2e3e69b997865e5bf7a87e06b498858188a105477a95f`
against the public `facebook/collaborative_agent_bench` revision
`cf3526da25989b53f105fe9b74c1174a3e19c548`, preserving all 10,000 training rows
and their native order. No test split is downloaded. Reference programs and
test cases are environment inputs, not policy prompt content. The unused
validation loader contains **one training row**; validation is disabled.
Generated data, weights and experiment outputs remain outside Git.

Download and verify the original policy and CoSER with the shared model helper:

```bash
python examples/tau2/models.py --directory /models --role both
CUDA_VISIBLE_DEVICES=2 python examples/tau2/serve.py \
  --role user --model /models/user --port 18081 --receipt /runs/coser-service.json
```

The user server uses the CoSER serving template documented in the Tau2 guide.
An existing compatible server can instead be supplied with `--user-url` and
`--user-model`, including a Qwen2.5-7B user. Record that change in comparisons.
For an authenticated user service, set `COLBENCH_USER_API_KEY` in the environment.

## Inspect and run the fixed recipe

```bash
python examples/colbench/train.py \
  --workspace /training/colbench --policy /models/policy \
  --output /runs/colbench-check --gpus 2 --dry-run

CUDA_VISIBLE_DEVICES=0,1 python examples/colbench/train.py \
  --workspace /training/colbench --policy /models/policy \
  --output /runs/colbench-fixed --gpus 2
```

The dry run resolves the real Hydra configuration with no generation, API
calls, GPU allocation or optimizer updates. The default recipe preserves the
existing supplemental ColBench PPO profile:

| Component | Initial setting |
|---|---|
| Policy | Verified original Qwen3-4B; non-thinking input; temperature 0.6 |
| Optimization | PPO with multiturn GAE/SAO; gamma 1; lambda 1; actor LR 1e-6 |
| Budget | Two epochs; batch 32; one rollout per prompt; at most 624 updates |
| Critic | Original base;20 warmup updates;2 critic epochs per update |
| KL/entropy | Actor KL coefficient 0.001; entropy 0 |
| Clip | Upper clip 0.2→0.28 over100 updates; zero out of range |
| User | Task-ID-deterministic 50% flow /50% configured LLM; LLM temperature 0.7 |
| Environment | Ten turns; 400-character LLM replies; code disclosure guard |
| Initial reward | Native fraction of passing test cases; scale 1; step penalty 0 |
| Sequence | 1152 prompt + 10240 response tokens; 768 tokens per turn |
| Checkpoints | Every 200 updates and final; model only by default |
| Selection | Final completed update; no validation or best-checkpoint selection |

Prompt filtering can reduce the actual update count below 624. `--steps N`
declares a shorter budget; the first 20 updates warm the critic, so use a larger
budget to train the actor. `--full-checkpoints` adds optimizer, scheduler/RNG
and dataloader state; the default model-only profile is for final policy use.
Both actor and critic start afresh. This is the existing PPO/GAE port, **not a
claim to implement the official SWEET-RL algorithm** or to reproduce a paper
score without a GPU run and frozen external evaluation.

## Natural-language editor and five fresh rounds

Use a coding-agent CLI against your Flint endpoint. The shared adapter starts
a fresh process with an empty session directory for every review. For example,
save a backend argv array in `/config/backend.json`:

```json
["pi", "--print", "--provider", "flint", "--model", "flint", "{prompt}"]
```

Configure Pi's `flint` provider in `/config/models.json` with your served model
ID and API endpoint. Supply credentials through the CLI's supported environment
or provider configuration. The adapter copies only the selected provider JSON,
never a previous conversation directory. Pi/OpenCode installation and provider
configuration are external; no closed or open model is called by default.
Then save `/config/audit-command.json` (replace `/code` with this checkout):

```json
["python", "/code/examples/colbench/agent.py", "--request", "{request}", "--backend-command", "/config/backend.json", "--pi-models", "/config/models.json"]
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/colbench/campaign.py \
  --workspace /training/colbench --policy /models/policy \
  --output /runs/colbench-chain --rounds 5 --gpus 2 \
  --audit-command /config/audit-command.json --audit-every 20
```

Reviews run in the background after the first update and at the chosen
interval. Busy requests coalesce. The last update stops writers before its
snapshot. Each request sees selected training logs, trajectories, metrics and
`experience.md`, rather than a continuing conversation. The coding agent edits
files directly and explains its work in natural language; there is no required
JSON proposal format or model proxy.

| Component | Editable interface | Consumer |
|---|---|---|
| User | `drafts/user/flow_user.py`, `env_policy.json` | New episodes load the published flow implementation and mixture/disclosure settings |
| Environment/reward | `drafts/environment/environment.py`, `settings.json` | Reset/action/reward hooks, turn/reply limits and reward scalars |
| Agent/RL | `drafts/agent/entropy_schedule.json`, `clip_schedule.json` | Existing trainer schedule consumers |

For example, `{"enable":true,"value":0.004}` sets an entropy target; the
existing trainer limits its ramp. Clip controls use `override` or
`init`/`final`/`steps`. Drafts undergo bounded file, syntax and scalar checks.
An invalid draft retains the last good snapshot. Changes publish only before
a rollout update, and each episode keeps one immutable user/environment
snapshot. Worker receipts under `uptake/` identify the snapshot actually used;
`pinned_updates.jsonl` records publication, not demonstrated learning benefit.

Every round restarts the original policy, critic, optimizer and draft defaults.
Only an explicit training-derived Markdown document crosses round boundaries;
the meta-agent must choose whether to reapply an earlier idea. By default the
seed stays fixed; `--seed-stride` changes it. These adaptive rounds are not
independent end-to-end training seeds. Actual elapsed policy allocation is
recorded per attempt; user/model-service and meta-agent costs are separate.

## Training/evaluation access boundary

The prepared workspace contains only training data. Training disables native
validation and gives the editor selected training evidence. Evaluation tasks,
evaluation results and prior policies are not mounted by the supplied container
launcher:

```bash
docker build -f examples/colbench/Dockerfile -t eau-colbench .
python examples/colbench/container.py --image eau-colbench \
  --workspace /training/colbench --policy /models/policy \
  --output /runs/fresh-colbench-container --devices 0,1 --dry-run
```

Remove `--dry-run` to launch. The image filesystem and training/model mounts
are read-only; temporary caches and the fresh output are writable. A coding-agent CLI and
its explicit provider configuration must be installed in the image if enabling
audits there. Host networking reaches user/meta-agent services; it does not
block downloading public data. Local execution also grants the editor the
host user's filesystem access. Therefore the statement supported by this
package is “no held-out tasks supplied or mounted,” rather than an assertion
that prompt instructions enforce a sandbox. Enforce network and service
allowlists externally when that stronger boundary is required. The package
does not invoke Harbor or claim a Harbor boundary.

Run final-policy evaluation in a separate process/container with a fixed
external ColBench evaluation implementation. Training hooks and flow settings
must not be inherited into that evaluator. The official SWEET-RL source can be
obtained separately at the recorded revision:

```bash
git clone https://github.com/facebookresearch/sweet_rl.git /benchmarks/sweet_rl
git -C /benchmarks/sweet_rl checkout 38daf2a58497e898a65ef530a7b1c13ff4b0f128
```

The checkout and held-out data are not dependencies of training and should
remain outside its mounts. No experiment results or checkpoints are included.

## CPU validation

```bash
python -m unittest discover -s tests -p 'test_colbench*.py' -v
python -m unittest discover -s tests -v
```

The tests exercise actual environment scoring, update-pinned edits, rejected
drafts, fresh-round defaults, non-blocking dispatch, data conversion, mount
boundaries and real Hydra composition. They do not establish GPU throughput,
end-to-end training quality or a new evaluated model score.
