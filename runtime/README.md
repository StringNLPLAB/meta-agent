# Owning a training experiment

`run_experiment.py` owns one trainer and the existing meta-agent supervisor on a POSIX host. The outer coding agent generates its configuration from the experiment request, then launches it. The experiment owner can describe paths, resources, cadence, and model choices in natural language.

The lifecycle is:

```text
read-only discovery → launch trainer + supervisor → periodic evidence reviews
                   → trainer exit receipt → final review → recorded outcome
```

The supervisor polls the configured log and clock. When a step or time boundary is due, it starts Codex or OpenCode with the skill, contract, response schema, and permitted evidence paths. The coding agent reads the current log and training trajectories during that request. The model server does not schedule these calls. Reviews can propose changes; application still requires the existing authorized project gate and uptake checks.

## Launch interface

From the repository, the outer agent invokes:

```bash
python3 runtime/run_experiment.py --config /absolute/path/to/launch.json --detach
```

Omit `--detach` to keep the owner in the foreground. Detachment creates a new session with redirected input/output, checks the child's PID and process start token, and returns its state path. A `started` response means the owner is live; discovery and training may still fail afterward. Inspect `reports/runtime/<RUN>/state.json` and the bound log to confirm progress. No OS service, reboot recovery, or automatic trainer restart is installed. Use the host's job service if its policy kills sessions when a shell or agent session ends.

For the included τ³-bench example, [prepare.py](../examples/tau3_minimal/prepare.py) generates the configuration. For other trainers, inspect their real entry point and evidence outputs before constructing it. Its format is version 1:

```json
{
  "version": 1,
  "root": "/absolute/path/to/training-checkout",
  "run": "EXPERIMENT_001",
  "trainer_command": ["python3", "-u", "train.py", "--steps", "200"],
  "contract": {
    "objective": "Improve this training objective within the declared budget",
    "budget": "200 optimizer steps, 4 hours of training, at most 30 agent calls",
    "guardrails": ["Preserve the fixed evaluation protocol"],
    "allowed_feedback": ["Current training logs and training diagnostic samples"],
    "excluded_feedback": ["Independent test data, scores, trajectories and summaries"],
    "evaluation_protocol": "Independent final evaluation; no test-based steering",
    "checkpoint_selection": "Final checkpoint from this run",
    "allowed_operations": []
  },
  "supervisor_args": [
    "--log", "reports/runtime/EXPERIMENT_001/trainer.stdout",
    "--trajectories", "reports/training_samples/EXPERIMENT_001",
    "--backend", "opencode",
    "--opencode-base-url", "http://localhost:30000/v1",
    "--agent-model", "flint",
    "--every-steps", "20", "--every-seconds", "600",
    "--max-calls", "30", "--poll-seconds", "1",
    "--export-atif"
  ],
  "trainer_timeout_seconds": 14400,
  "poll_seconds": 0.2,
  "final_timeout_seconds": 660,
  "failure_policy": "stop_owned_trainer"
}
```

This is an illustrative trainer command, not a bundled `train.py` interface. Commands are literal argument arrays, executed without a shell in `root`. Relative `root` values resolve against the launch configuration's directory. A recipe needing shell syntax can use a reviewed script as its entry point. `trainer_timeout_seconds` is required and positive; it bounds the trainer after launch. `final_timeout_seconds` bounds waiting for supervision after the trainer exits; choose it to cover the configured agent timeout and export work. Review calls, including discovery, reconciliation, and final review, share `--max-calls`. Allocate discovery and final-review overhead when resolving an overall wall-clock budget.

The runtime supplies `--root`, `--run`, `--contract`, and `--completion-file`. Those flags, `--once`, and `--adopt-settings` are rejected in `supervisor_args`. The supervisor's other flags and contract rules remain documented in [supervision.md](../.agent/skills/meta-agent/references/supervision.md). For Codex, omit the OpenCode backend/model/endpoint flags or replace them with the desired Codex settings.

Configure the trainer's own finite step/epoch recipe. Omit supervisor `--stop-step`: final review should follow actual trainer exit. Reaching a supervision stop boundary while training is still active ends supervision early and causes the runtime to stop its owned trainer.

Only `stop_owned_trainer` is currently supported as a failure policy. Credentials remain inherited environment variables; pass names such as `--opencode-api-key-env LOCAL_MODEL_API_KEY`, never key values in a saved configuration. Training policy, user simulator, and review model are separate resources. This runtime starts neither inference servers nor model downloads.

## Records and outcomes

Each fresh run creates `reports/runtime/<RUN>/` with normalized `launch.json`, `contract.json`, `state.json`, child output files, process identities, and a trainer `completion.json` receipt. Detached owner output is captured as `runtime.stdout` and `runtime.stderr`. The actual trainer exit code is retained separately from its managed wrapper's exit code. Review requests, responses, decisions, and ATIF files remain under `reports/supervision/<RUN>/`; training trajectories stay in the configured training evidence directory.

The runtime starts training only after discovery reports both `discovery_complete=true` and `status=watching`. Overall `completed` requires a successful trainer exit and a successful final supervisor state. A successful final review cannot turn failed training into a successful experiment. Budget exhaustion, agent handoff, idle handoff, missing completion receipt, and timeouts remain visible failure or handoff states. A zero supervisor process exit code alone does not establish success.

Existing runtime directories, supervision runs, or the planned training log are rejected. There is no implicit resume, retry of a gate operation, or restart of training. Investigate interrupted state and lineage before explicitly preparing a new run. The lower-level supervisor retains its documented reconciliation path for separately managed jobs.

SIGINT, SIGTERM, timeout, and early supervision exit clean up only process groups created by this invocation. Unrelated trainers and model servers are outside its ownership. An uncatchable owner kill or host failure can leave state stale; verify recorded process identities and locks before recovery. Processes that deliberately escape their owned group require stronger host containment.

## Validation

```bash
python3 -m unittest discover -s tests -v
```

Runtime tests launch temporary synthetic trainers and protocol fixtures. The separate minimal-training smoke checks use CPU PyTorch and real parameter updates. Set `META_AGENT_OPENCODE_BIN` to an OpenCode 1.19.3 executable to include a real-CLI integration with a localhost mock model API. These checks verify process ordering, evidence reads, cadence, final review, and export; they do not establish model quality or τ³-bench performance.
