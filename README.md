# Meta-Agent Skill

A general skill for supervising long-running model training. It combines the EAU reference, prior experiment experience, and current behavior to discover mechanism hypotheses, maps them to actual code and runtime controls, then tests worthwhile changes and verifies uptake and outcomes.

The workflow applies across supervised, reinforcement, and agent training. Each review can apply a justified change, collect evidence, observe a pending window, or plan a restart-required intervention. The experiment contract defines feedback boundaries; EAU reproduction uses training-side feedback only.

Load [.agent/skills/meta-agent/SKILL.md](.agent/skills/meta-agent/SKILL.md) in your agent, or copy `.agent/skills/meta-agent/` into its skill directory. To run initial component discovery and subsequent agent reviews automatically, start the bundled runner as below.

## Start supervision

Use Linux with Python 3 and an installed, authenticated Codex CLI. The runner uses the CLI's configured model and loads the skill into each read-only agent call. Another compatible agent can be supplied with `--agent-command`; see the [backend instructions](.agent/skills/meta-agent/references/supervision.md#agent-backend-and-returned-review).

1. Choose the training checkout, a unique run ID, the exact planned/current training log, and the training trajectory directory. Logs and trajectories may be absent before training.
2. Create `reports/meta_agent_contract.json` inside that training checkout using the [contract example](.agent/skills/meta-agent/references/supervision.md#establish-inputs-and-run-discovery-first). Set the objective, budget, guardrails, allowed/excluded feedback, evaluation protocol, and checkpoint-selection rule. Leave `allowed_operations` empty for proposal-only supervision.
3. From this repository's root, replace the example training path, log/trajectory paths, run ID, and budgets below, then run discovery:

```bash
SKILL_DIR="$PWD/.agent/skills/meta-agent"
TRAIN_ROOT=/absolute/path/to/training-checkout
RUN=my_run

SUPERVISE=(python3 "$SKILL_DIR/scripts/supervise.py"
  --root "$TRAIN_ROOT" --run "$RUN"
  --log logs/train.log
  --trajectories trajectories
  --contract reports/meta_agent_contract.json
  --every-steps 20 --every-seconds 600
  --max-calls 30 --stop-step 200 --idle-timeout 1800)

# Before launching training: one discovery pass.
"${SUPERVISE[@]}" --once
```

Inspect `$TRAIN_ROOT/reports/intervention_map_<RUN>.md` and confirm `discovery_complete` is `true` in `$TRAIN_ROOT/reports/supervision/<RUN>/state.json`. The map identifies candidate components, mechanisms, applicability, runtime feasibility, and unknowns. If discovery failed, inspect that request's stdout/stderr and rerun the same command; it repeats read-only discovery with the prior artifacts.

Launch training separately after discovery, then start continuing supervision in the **same Bash session**:

```bash
"${SUPERVISE[@]}"
```

This foreground process invokes the agent to read the bound log and raw trajectories, reassess pending predictions, and propose changes. Keep it running in a terminal/session managed by your host. To resume in another shell, recreate the same variables and `SUPERVISE` array, then run it again; saved state is reused. Loading the skill alone does not start a background process, and the runner does not launch training or install a service.

The values above are examples. Choose either trigger or both; if both are configured, either becoming due triggers one serial review:

| Argument | Behavior |
|---|---|
| `--every-steps N` | Review at training-step boundaries. The default log format is `step:<N> - ...`; use `--step-regex` for another format. |
| `--every-seconds S` | Review after this interval since the last completed review, even without a new step. |
| `--once` | Perform at most one due pass and exit. An existing host scheduler can invoke this repeatedly. |
| `--max-calls N` | Total persisted agent-call budget, including discovery, failed calls, reconciliation, and final review. Required for continuous mode. |
| `--stop-step N` | Request a read-only final review when this step is observed; training is left running. |
| `--idle-timeout S` | Hand off supervision when the bound log stops advancing for this long. |

Default `--mode propose` writes review records without changing training/configuration. For execution, configure a project gate with `--mode apply`, `--executor-command`, and matching `contract.allowed_operations`; follow the [gate interface](.agent/skills/meta-agent/references/supervision.md#optional-execution-through-a-project-gate). Discovery remains read-only. After an uncertain execution, resumption reconciles actual state without automatically retrying the operation.

Reviews default to `reports/change_log.md`; maps, experience, and supervision state stay under the training checkout's `reports/`. Each `reports/supervision/<RUN>/requests/<ID>/` retains the prompt, response, stdout/stderr, and any operation receipt. See the [full runner instructions](.agent/skills/meta-agent/references/supervision.md) for log/run aliases, timeouts, explicit settings changes, and recovery.

The [intervention discovery guide](.agent/skills/meta-agent/references/intervention-discovery.md) explains candidate selection. The optional [EAU / verl adapter](.agent/skills/meta-agent/references/eau-adapter.md) documents project-specific helpers and layouts; its watchers check records without invoking the reasoning agent.

## Validation

Run offline helper regression checks with `python3 -m unittest discover -s .agent/skills/meta-agent/scripts/tests -v`. Fixtures use temporary training roots and do not connect to training processes.

Originally developed for **EAU: Environment-Agent-User Co-evolution for Training User-Centric Agents**. This branch contains the skill and supervision helpers; training frameworks, weights, and run outputs remain outside the package.
