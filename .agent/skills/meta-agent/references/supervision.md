# Running discovery and periodic agent reviews

Use `scripts/supervise.py` when the task calls for continuing supervision. It performs initial discovery, then invokes an agent at time/step boundaries to read training logs and raw trajectories, reassess hypotheses, and propose changes. Unlike `run_watch.py` and `checkpoint_watchdog.sh`, it actually calls the reasoning agent. The runner is a foreground process with persisted state, not an installed OS service; its host must keep it alive or invoke `--once` through an existing scheduler.

## Establish inputs and run discovery first

Provide an exact training root, run identity, planned/current log path, training trajectory directory, and a JSON experiment contract. The log and trajectory directory may be absent before training. With an EAU `train_<name>_<YYYYMMDD>_<HHMMSS>.log` filename, the supplied run must match its inferred identity or a registered `reports/run_aliases.json` mapping. Other log filenames use the explicitly bound run/path pair. The runner never picks another run by mtime.

Example contract, stored in the **training checkout**, with values chosen for that experiment:

```json
{
  "objective": "Improve full-task completion while preserving useful clarification",
  "budget": "200 optimizer steps and at most 30 agent calls, including discovery and final review",
  "guardrails": ["Preserve task semantics and legitimate information gathering"],
  "allowed_feedback": ["This run's training log, training trajectories, code and training-only experience"],
  "excluded_feedback": ["Independent evaluation/test instances, trajectories, scores and summaries"],
  "evaluation_protocol": "Fixed task distribution, user settings, interaction limits, judge and scoring version; final evaluation is reporting-only",
  "checkpoint_selection": "Final run and final checkpoint; no final-test-based selection",
  "allowed_operations": [],
  "records": {
    "change_log": "reports/change_log.md",
    "intervention_map": "reports/intervention_map_RUN.md",
    "experience": "reports/meta_agent_experience.md"
  }
}
```

Reuse existing record paths when available; the map defaults to `reports/intervention_map_<RUN>.md`, and the other two paths above are defaults. All writable records must remain under the training root's `reports/`. Permission to read a source is governed by the contract, not inferred from its presence on disk. The contract's descriptions guide the agent; they are not filesystem access-control rules. Configure stronger source isolation through the host when required.

Run discovery **before launching training**. The example uses Bash arguments shared between preflight and continuing supervision:

```bash
SUPERVISE=(python3 "$SKILL_DIR/scripts/supervise.py"
  --root "$TRAIN_ROOT" --run RUN
  --log logs/train.log
  --trajectories checkpoints/EXP/debug_trajectories
  --contract reports/meta_agent_contract.json
  --every-steps 20 --every-seconds 600
  --max-calls 30 --stop-step 200 --idle-timeout 1800)

"${SUPERVISE[@]}" --once
# Inspect discovery findings and reports/supervision/RUN/state.json.
# Launch the separately authorized training job only after discovery completes.
"${SUPERVISE[@]}"
```

The first call must return an intervention map and identify paper, code and experience sources or explicit availability gaps. It cannot execute an intervention. The next call resumes that discovery state. If training was already running when supervision began, this is a takeover assessment at the actual observed step; it does not invent a preflight or prior reviews.

These numbers are examples, not defaults for another recipe. Use `--every-steps N`, `--every-seconds S`, or both (either being due triggers one serial review). `--first-review S` shifts the first step boundary; boundaries at/before the selected log's resume anchor are excluded. Time intervals start after the last completed review. `--agent-timeout` defaults to 600 seconds and `--executor-timeout` to 120; budget initial paper/code discovery separately from review cadence, and choose timeouts that fit its scope. `--poll-seconds` defaults to 5, with a maximum of 60. The completed-line step regex defaults to `step:(\d+) - `; adapt `--step-regex` with exactly one capturing group for other trainers.

`--once` performs at most one due discovery/review/reconciliation/final pass and can be called by a host scheduler. It does not call the agent again if neither trigger is due. A continuous runner requires an explicit `--max-calls` budget; it counts all dispatched calls, including failed calls, discovery and reconciliation. Increase this limit explicitly when more supervision budget is authorized. `--stop-step` or a separately produced `--completion-file` triggers a read-only final review. Idle timeout takes precedence over another scheduled review, so timer reviews cannot keep an idle run supervised indefinitely. Idle timeout, exhausted budget, or an explained agent handoff stops **supervision**, never training, and preserves unresolved state.

## Agent backend and returned review

The default backend is an installed, authenticated Codex CLI, using its configured model:

```text
codex exec --sandbox read-only --skip-git-repo-check --output-schema SCHEMA --output-last-message RESPONSE --json -
```

The actual skill text is embedded in every request, with an absolute skill root; initial discovery also receives the discovery guide. The agent reads relevant linked references, existing records and permitted code/log/trajectory files. This explicit loading works even when the skill is outside the training checkout's auto-discovery paths. It receives one pass and must not launch another supervisor. The runner owns record publication and scheduling.

Codex non-interactive structured responses and sandbox flags follow the [official documentation](https://developers.openai.com/codex/noninteractive), also checked against the installed CLI's `exec --help`. Existing CLI authentication and model configuration are reused; no key is copied into the contract. Filesystem sandboxing does not authorize external actions through configured tools. The task/feedback contract still governs behavior.

For another compatible agent, pass `--agent-command '["/absolute/path/to/agent", "literal-argument"]'`. It receives the prompt on stdin and returns one JSON object on stdout, or writes that object to `$META_AGENT_RESPONSE`. `$META_AGENT_REQUEST` and `$META_AGENT_PHASE` identify the current request. Commands are literal argument arrays, never evaluated as shell text. A custom backend must enforce its own read-only/tool restrictions; the runner does not sandbox arbitrary binaries.

The per-request `response.schema.json` defines the response: phase, summary, complete revised intervention map (or empty to retain it after discovery), cited source arrays, seven review fields, optional operation payload, pending prediction descriptions, optional evidence-labeled experience update, next condition and continue/handoff decision. `review.change`, `prediction` and `verification` carry the existing adapter's tagged fields; the controller prefixes the structured decision and adds its request ID/time. A pending window, instrument plan, justified no-op or deferred restart is valid. Every review reports log and raw trajectory evidence, or explicit gaps, and keeps hypothesis verdict separate from application status.

The runner validates record structure using `check_mandate.validate_review`, publishes the existing eight-column table, updates the selected map, and appends warranted experience. A schema-valid response is not proof that every observation is true. Agent events and actual source references remain reviewable. Selected trajectory files are diagnostic samples, not population-rate denominators.

## Optional execution through a project gate

Default `--mode propose` performs no training/config writes. To enable bounded operations, explicitly select `--mode apply`, declare operation identifiers in `contract.allowed_operations`, and provide `--executor-command` as a JSON argv array. The reasoning agent remains read-only. Only a regular review with `decision=apply` can invoke this gate; discovery, reconciliation and final review cannot.

The agent supplies `action.operation` and `action.arguments_json` (a JSON object encoded as a string), plus a complete pending application review with old/new state, mechanism, prediction/window and recovery. The controller durably records that prediction **before** calling the configured executable. The executor receives JSON on stdin:

```json
{
  "request_id": "unique-stable-request-id",
  "run": "RUN",
  "step": 40,
  "observation_step": 35,
  "operation": "set_repeat_budget",
  "arguments": {"expected_version": 1, "value": 2},
  "prediction": "The preregistered mechanism criterion and window",
  "review": {"...": "the complete proposed review"},
  "contract": {"...": "the experiment contract"}
}
```

This is an interface illustration, not a runnable training patch. `observation_step` is the review-start cutoff; `step` is re-read before gate dispatch. The record also captures `decision_step` after analysis. Prefer exposure windows relative to verified uptake. If the configured end boundary arrives during analysis, the runner records the operation as not sent and proceeds toward final review. A project executor maps allowed operations to existing validated controls, enforces ranges, expected old state/version and idempotency by request ID, and verifies the supported consistency boundary. It must be a bounded synchronous control operation, not a launcher for detached training/services. The runner adds no training algorithm, control reader, distributed transaction or checkpoint-restoration implementation. New code/restart mechanisms remain candidates until a compatible authorized gate exists.

The executor returns a JSON receipt on stdout, with these string fields:

```json
{
  "application": "fully_applied",
  "attempt": "native gate receipt and request identity",
  "boundary": "the actual effective episode/update boundary",
  "scope": "worker0,worker1",
  "confirmed": "worker0,worker1",
  "uptake": "traceable consumed value/version evidence for the declared scope",
  "recovery": "control/checkpoint recovery status and inherited lineage"
}
```

Use `pending`, `partial` or `failed` when warranted; they do not require fictitious uptake and can complete a review. All seven fields must be present as strings; unconfirmed boundary/scope/uptake fields may be empty for those statuses. A nonzero executor exit can accompany an honest failed/partial/pending receipt. A contradictory success receipt, missing fields, absent uptake for fully-applied, or incomplete consumer scope stops the runner for reconciliation. Exit zero and a writable file alone never certify fully-applied. Keep machine field values free of semicolons (the tagged table delimiter). The proposal and application receipt remain separate rows; planned recovery and actual recovery are distinguished. The controller records `runtime_verified=false`: it checks supplied evidence structure, not independent worker authenticity or causal benefit.

## Resumption, interruption and ownership

State lives in `reports/supervision/<RUN>/state.json`, with pending predictions, review position/time, next time/step trigger, foreground owner and active request. Normal exits explicitly mark the foreground owner stopped; `--once` never claims a future invocation is arranged. Abrupt-kill ownership metadata can be stale, so use the run lock and process identity during takeover. `events.jsonl` records dispatch/completion/failure. Each `requests/<ID>/` retains its prompt, schema, response, stdout/stderr, and any operation/receipt. Reuse these instead of creating another hypothesis registry. Global record locking and atomic table replacement preserve existing history across controller interruptions; per-run locks prevent concurrent supervisors.

The log cursor persists offsets, incomplete lines, inode/prefix identity and resume provenance; it reads appended complete lines rather than rereading the entire log at every poll. Log replacement/truncation, step regression and lines over 1 MiB stop supervision for inspection. Late polls review the actual observed step once and acknowledge crossed step boundaries as gaps, without fabricated historical predictions. Incomplete existing records are not silently marked handled or overwritten.

After a failed/timed-out/interrupted request, the next invocation performs a read-only reconciliation (or repeats initial read-only discovery if it never completed), with all prior artifacts. It does not rerun the uncertain operation. A live old child holds the inherited run lock even if the controller died; a second runner cannot overlap it. Timeouts/Ctrl-C/SIGTERM clean up only the process group created for that agent/gate call, never search for or signal a trainer. Project request preconditions/idempotency remain necessary: process failure cannot erase a write that already happened, and this is not an exactly-once distributed transaction.

Root/run/log/trajectory bindings cannot be reused for another run. Contract, cadence, backend/mode and skill changes are detected; use `--adopt-settings` explicitly to retain the prior binding in history and redo read-only discovery before proceeding. Do not leave that flag in a recurring invocation. Increasing the call budget does not itself rewrite the contract. If a prediction/window or effective configuration is unknown, preserve that uncertainty during reassessment.

A successful final pass records the handoff under the declared checkpoint/evaluation rules; it does not run disallowed evaluation or terminate training. `budget_exhausted`, `idle_handoff`, `agent_handoff` and `needs_reconciliation` are distinct from `completed`. Inspect the saved state and arrange an actual responsible owner whenever supervision stops before the experiment is finished.
