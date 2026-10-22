# Running discovery and periodic agent reviews

This is an implementation reference for the outer coding agent, whether Codex or OpenCode, not a manual setup checklist for the experiment owner. Accept the training code location and natural-language goals, resources, budget, review cadence, and allowed changes; derive the JSON contract and CLI arguments from that request and the existing project. Reuse an established recipe and sensible project settings rather than asking the user to fill in flags or JSON. Ask only about a necessary objective, budget, permission, data source, or credential that cannot be resolved from available context.

The outer agent owns the full authorized workflow: complete one-time reference setup if needed; reuse its summary and applicable experience; inspect current code; establish the contract and run bindings; complete per-run read-only discovery; launch the existing training recipe; arrange an actual continuing supervisor with trajectory export; verify training health, owner and next wake-up; then finish or hand off under the declared budget. For an already-running job, bind to it and assess the current state without relaunching it. Loading the skill is not itself a request for an unbounded experiment.

This workflow covers pretraining, self-supervised learning, supervised learning and SFT, reinforcement learning, and agent training. Discover the components the project actually exposes: objectives and losses, data sampling and preprocessing, optimizer schedules, batching, resource use and throughput, and training measurements. Reward, rollout, environment, and user-simulator controls are additional project-specific candidates, not required inputs. Inspect and verify the actual framework's launch interface, progress reporting, diagnostic sources, control readers, and consistency boundaries; general scope does not establish a tested integration for an uninspected framework.

Use `scripts/supervise.py` for discovery and subsequent time/step-triggered reviews. Unlike `run_watch.py` and `checkpoint_watchdog.sh`, it actually calls the reasoning agent to read training logs and available diagnostic samples, reassess hypotheses, and propose changes. For new local experiments from the full repository, the [experiment runtime](../../../../runtime/README.md) owns this foreground worker together with the trainer: it performs discovery, starts both processes, enforces a trainer timeout, and waits for final supervision. The runtime can detach with a verified live owner; it does not install an OS service. For standalone skill installations, existing host jobs, or takeover of running training, place the worker under a supported host owner or arrange repeated `--once` calls. Record actual identities and the next wake-up, and verify their lifetime on the selected host.

The runtime generates its own completion marker when the owned trainer exits. Do not use `--stop-step` to finish supervision before that trainer exits: the runtime treats early supervision exit as a failure and stops its owned trainer. It does not silently resume an old run or expand a review budget. See the runtime reference for configuration and lifecycle details; the lower-level examples below apply when another host owner already manages training.

## One-time reference setup

During setup, the outer coding agent reads the bundled `references/eau_manuscript_ref.pdf`, **EAU: Environment-Agent-User Co-evolution for Training User-Centric Agents**, and saves a reusable summary in the existing experience record or intervention map. Record the paper version/source identity, key mechanisms, applicability conditions, and limitations. Reuse those records rather than creating another registry, and pass the saved summary's location to subsequent run discovery.

EAU is a method reference, not a universal training recipe or a requirement to introduce RL components. Retain only mechanisms that fit the inspected project and use its own objectives, feedback boundaries, and supported controls. The EAU / verl adapter documents one optional project setting.

Later experiments, periodic reviews and modifications reuse this cached mechanism summary, applicable experience, and the current run's code and evidence. They do not need to read the paper again. Repeat setup only if the paper version changes or the summary is missing; a new run or training phase alone is not a reason to reread it. This does not replace per-run component discovery: candidates still need to be mapped to the current recipe, resources, code paths, and runtime controls. The outer agent handles setup automatically within the authorized workflow.

## Establish inputs and run discovery first

Resolve an exact training root, unique run identity, planned/current log path, diagnostic-sample directory, and experiment contract from the user's request and the inspected recipe. Save the contract as JSON for the runner; the user need not create it. The log and sample directory may be absent before training. The compatibility argument `--trajectories` names this sample directory: for non-interactive training it can contain sampled examples, predictions, or per-batch diagnostics; for interactive training it can contain trajectories. If samples are unavailable, report that evidence gap and, within the authorized scope, arrange useful sampling or instrumentation. Do not manufacture rollouts or require a nonexistent interactive agent.

With an EAU `train_<name>_<YYYYMMDD>_<HHMMSS>.log` filename, the supplied run must match its inferred identity or a registered `reports/run_aliases.json` mapping. Other log filenames use the explicitly bound run/path pair. The runner never picks another run by mtime.

Example contract, stored in the **training checkout**, with values chosen for that experiment:

```json
{
  "objective": "Improve the declared training objective within the compute budget",
  "budget": "200 optimizer steps and at most 30 agent calls, including discovery and final review",
  "guardrails": ["Preserve data semantics and the declared evaluation protocol"],
  "allowed_feedback": ["This run's training log, permitted diagnostic samples, code and training-only experience"],
  "excluded_feedback": ["Independent evaluation/test instances, trajectories, scores and summaries"],
  "evaluation_protocol": "Fixed evaluation data, preprocessing, metrics and scoring version; final evaluation is reporting-only",
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

Run discovery **before launching training**. The following Bash example shows the invocation the outer agent derives; all values are illustrative. Persist the resolved arguments with the run's launch metadata so a new shell or host job can reconstruct them:

```bash
SKILL_DIR=/absolute/path/to/meta-agent/.agent/skills/meta-agent
TRAIN_ROOT=/absolute/path/to/training-checkout
RUN=RUN

SUPERVISE=(python3 "$SKILL_DIR/scripts/supervise.py"
  --root "$TRAIN_ROOT" --run "$RUN"
  --log logs/train.log
  --trajectories reports/training_samples/RUN
  --contract reports/meta_agent_contract.json
  --every-steps 20 --every-seconds 600
  --max-calls 30 --stop-step 200 --idle-timeout 1800
  --export-atif)

"${SUPERVISE[@]}" --once
```

Confirm `discovery_complete=true` in `reports/supervision/<RUN>/state.json` and inspect `reports/intervention_map_<RUN>.md` (or the contract's selected map path). The first call must return an intervention map and identify the cached setup summary, code and experience sources or explicit availability gaps. `sources.paper` may cite that saved setup memo or a versioned mechanism guide, with the paper/version identity; it must not claim the current call read the PDF when it reused a summary. It cannot execute an intervention. If it fails, inspect the saved request and retry the same read-only discovery with its prior artifacts.

After discovery, launch the authorized training recipe through the project's supported job system. Submit the resolved supervision command **without `--once`** to the verified host owner, or schedule repeated `--once` invocations. The next invocation resumes the saved discovery state; it does not depend on keeping the example's shell variables or Bash session alive. Verify the exact training log is advancing, the supervisor's recorded process/task is live, export is enabled, and the next review is actually arranged. Store the training job and supervisor identities together with their ownership and lifetime. If training was already running when supervision began, discovery is a takeover assessment at the actual observed step; do not invent a preflight or prior reviews.

The outer agent also owns the training recipe's resource/stop budget and checkpoint completion; `--stop-step` only ends supervision after a final review. Configure training's own bounded recipe or host job separately within the same automated launch workflow. Do not infer that the runner will stop a trainer.

These numbers are examples, not defaults for another recipe. Use `--every-steps N`, `--every-seconds S`, or both (either being due triggers one serial review). `--first-review S` shifts the first step boundary; boundaries at/before the selected log's resume anchor are excluded. Time intervals start after the last completed review. `--agent-timeout` defaults to 600 seconds and `--executor-timeout` to 120; budget one-time setup and per-run component discovery separately from review cadence, and choose timeouts that fit their scope. `--poll-seconds` defaults to 5, with a maximum of 60. The completed-line step regex defaults to `step:(\d+) - `, an EAU-style log pattern. For another trainer, the outer agent inspects actual completed log lines and derives `--step-regex` with exactly one capturing group for a nondecreasing integer progress counter, such as global optimizer updates or completed epochs. Bind cadence and stop limits to that counter's units; counters that reset each epoch need a stable global representation. The user need not select these flags. If no suitable counter is exposed, the outer agent reports the gap and arranges authorized progress instrumentation or uses time-based reviews without claiming step coverage.

Idle timeout measures advancement of the parsed progress counter; appended loss lines or other log bytes do not reset it. Prefer adapting or instrumenting a monotonic global counter before using step-based controls. If a usable counter is unavailable, omit `--every-steps`, `--stop-step`, and `--idle-timeout`; use `--every-seconds` with a separately produced `--completion-file` and an explicit call budget, while the outer agent arranges training health checks and the host's time/resource budget. Do not apply the example's step-idle timeout to a timer-only run without a progress counter.

`--once` performs at most one due discovery/review/reconciliation/final pass and can be called by a host scheduler. It does not call the agent again if neither trigger is due. A continuous runner requires an explicit `--max-calls` budget; it counts all dispatched calls, including failed calls, discovery and reconciliation. Increase this limit explicitly when more supervision budget is authorized. `--stop-step` or a separately produced `--completion-file` triggers a read-only final review. A configured idle timeout takes precedence over another scheduled review, so timer reviews cannot keep a run past that timeout. Idle timeout, exhausted budget, or an explained agent handoff stops **supervision**, never training, and preserves unresolved state.

## How a review invokes the coding agent

The supervisor creates the trigger; the coding agent does not watch files between calls:

1. The outer agent launches the trainer and places `supervise.py` under the verified host owner, or arranges scheduler-owned `--once` invocations.
2. The runner reads completed lines from the bound log and checks its persisted step/time cadence. A due boundary dispatches one review; discovery, interrupted-call reconciliation, and completion/final review have their own existing dispatch conditions.
3. Before invoking the backend, it saves a request with the actual skill text, experiment contract, current phase, observation cutoff, evidence paths, and response schema. It then launches the selected CLI and sends the request on stdin.
4. The coding agent reads permitted code, records, logs, and diagnostic samples with its configured tools, then returns one structured decision. The controller validates and publishes it and can invoke an authorized project gate. A later review assesses uptake and outcomes.

Reading a `SKILL.md` supplies instructions; the explicit backend call makes the agent act on them. A continuously running supervisor polls at `--poll-seconds`; a scheduler can instead invoke `--once`, which exits without calling the agent when nothing is due. Neither arrangement requires a permanently open coding-agent chat session.

[OPAL](https://github.com/xingjian-bai/opal) places this trigger inside its optimizer wrapper: `opt.step()` checks the training update, scheduled decisions, non-finite losses, and an analysis program's optional `wake` result. It queues a request to an agent worker, which calls the coding-agent backend. Its background worker can run while training continues or while training waits, depending on configuration. This runner uses external progress-log/time polling; it does not implement OPAL's in-process `analysis.wake` hook or install an optimizer wrapper. Additional event sources need an explicit adapter rather than assuming the agent notices a changed file.

## Agent backend and returned review

The default `--backend codex` uses an installed, authenticated Codex CLI and its configured model. `--agent-model NAME` explicitly selects a model when requested:

```text
codex exec --sandbox read-only --skip-git-repo-check --output-schema SCHEMA --output-last-message RESPONSE --json -
```

The actual skill text is embedded in every request, with an absolute skill root; initial discovery also receives the discovery guide. The agent reads the cached setup summary, relevant mechanism guides, existing records and permitted code/log/trajectory files, without reopening the paper each pass. This explicit loading works even when the skill is outside the training checkout's auto-discovery paths. It receives one pass and must not launch another supervisor. The runner owns record publication and scheduling.

Codex non-interactive structured responses and sandbox flags follow the [official documentation](https://developers.openai.com/codex/noninteractive), also checked against the installed CLI's `exec --help`. Existing CLI authentication and model configuration are reused; no key is copied into the contract. Filesystem sandboxing does not authorize external actions through configured tools. The task/feedback contract still governs behavior.

Select `--backend opencode` for a local OpenAI-compatible endpoint, with `--opencode-base-url` and the served model ID in `--agent-model`. The adapter targets OpenCode 1.19.3, supplies isolated per-request configuration, uses a fresh session and read-only tools, and parses the final JSON response before the same controller validation and project gate. It does not fall back to a cloud provider. Provider credentials are referenced by environment-variable name with `--opencode-api-key-env`, never placed in the contract or saved prompt. Optional context/output limits and a CLI-path override are documented in the [OpenCode reference](opencode.md). OpenCode 2.x configuration is outside this adapter's supported interface.

For another compatible agent, pass `--agent-command '["/absolute/path/to/agent", "literal-argument"]'`. It receives the prompt on stdin and returns one JSON object on stdout, or writes that object to `$META_AGENT_RESPONSE`. `$META_AGENT_REQUEST` and `$META_AGENT_PHASE` identify the current request. Commands are literal argument arrays, never evaluated as shell text. A custom backend must enforce its own read-only/tool restrictions; the runner does not sandbox arbitrary binaries. Do not combine this custom-command interface with `--backend opencode`; use `--opencode-command` to select a different OpenCode executable.

The per-request `response.schema.json` defines the response: phase, summary, complete revised intervention map (or empty to retain it after discovery), cited source arrays, seven review fields, optional operation payload, pending prediction descriptions, optional evidence-labeled experience update, next condition and continue/handoff decision. `review.change`, `prediction` and `verification` carry the existing adapter's tagged fields; the controller prefixes the structured decision and adds its request ID/time. A pending window, instrument plan, justified no-op or deferred restart is valid. Every review reports log and diagnostic-sample evidence, or explicit gaps, and keeps hypothesis verdict separate from application status.

The schema retains compatibility names rather than requiring RL concepts. `sources.trajectories` can cite examples, predictions, per-batch diagnostics, interaction trajectories, or their explicit availability gaps. `review.tier` can name a relevant component category such as data, objective, optimization, or instrumentation; `review.knob` identifies the actual candidate target. These fields do not require an environment, a reward function, or action trajectories. Missing interaction trajectories alone do not make a non-interactive training task unsupported.

The runner validates record structure using `check_mandate.validate_review`, publishes the existing eight-column table, updates the selected map, and appends warranted experience. A schema-valid response is not proof that every observation is true. Agent events and actual source references remain reviewable. Selected diagnostic samples are not population-rate denominators.

## Optional execution through a project gate

Default `--mode propose` performs no training/config writes. To enable bounded operations, explicitly select `--mode apply`, declare operation identifiers in `contract.allowed_operations`, and provide `--executor-command` as a JSON argv array. The reasoning agent remains read-only. Only a regular review with `decision=apply` can invoke this gate; discovery, reconciliation and final review cannot.

The outer coding agent translates the user's natural-language authorization into this configuration and reuses that scope for subsequent reviews. The gate is automatic programmatic validation/execution, not a request for per-review user confirmation. When needed, the outer agent may implement a compatible gate for already-authorized operations after inspecting the actual readers and consistency boundaries. Ask only when a necessary operation or restart falls outside the existing scope or a required input cannot be resolved. Gate availability never proves that arbitrary source edits affect resident workers; preserve runtime feasibility and the authorized restart plan.

The agent supplies `action.operation` and `action.arguments_json` (a JSON object encoded as a string), plus a complete pending application review with old/new state, mechanism, prediction/window and recovery. The controller durably records that prediction **before** calling the configured executable. The executor receives JSON on stdin:

```json
{
  "request_id": "unique-stable-request-id",
  "run": "RUN",
  "step": 40,
  "observation_step": 35,
  "operation": "set_learning_rate",
  "arguments": {"expected_version": 1, "value": 0.0001},
  "prediction": "The preregistered mechanism criterion and window",
  "review": {"...": "the complete proposed review"},
  "contract": {"...": "the experiment contract"}
}
```

This is an interface illustration, not a supplied learning-rate adapter or a runnable training patch. `observation_step` is the review-start cutoff; `step` is re-read before gate dispatch. The record also captures `decision_step` after analysis. Prefer exposure windows relative to verified uptake. If the configured end boundary arrives during analysis, the runner records the operation as not sent and proceeds toward final review. A project executor maps allowed operations to existing validated controls, enforces ranges, expected old state/version and idempotency by request ID, and verifies the supported consistency boundary. It must be a bounded synchronous control operation, not a launcher for detached training/services. The runner adds no training algorithm, control reader, distributed transaction or checkpoint-restoration implementation. New code/restart mechanisms remain candidates until a compatible authorized gate exists.

The executor returns a JSON receipt on stdout, with these string fields:

```json
{
  "application": "fully_applied",
  "attempt": "native gate receipt and request identity",
  "boundary": "the actual effective update, batch, epoch or episode boundary",
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

## Export meta-agent trajectories

Add `--export-atif` to the runner to export each discovery, review, reconciliation and final agent call, including failed or interrupted calls. This is an observation option: enabling or disabling it does not require `--adopt-settings`, invoke an additional agent, or change the experiment contract. `--once` still performs at most one due pass and installs no service.

This records the **meta-agent's own** visible messages, source-reading tool calls, tool results and structured decisions. The runner's compatibility argument `--trajectories` remains the directory of **training diagnostic samples or interaction trajectories that the meta-agent reads**. The two have different roles.

Exports use the [Harbor Agent Trajectory Interchange Format (ATIF)](https://docs.harborframework.com/agents/atif), version 1.7, using only Python's standard library. Each request directory gains:

- `trajectory.json`: the exported ATIF trajectory for this pass.
- `export.json`: source provenance, hashes, coverage and any export gaps or errors.
- A visible-event session archive when a matching native Codex session is available. `encrypted_content` fields are excluded from this archive.

`reports/supervision/<RUN>/trajectories.json` indexes the request exports. Existing prompt/response/stdout/stderr and operation receipts remain the source records. Export errors retain those records and produce a warning; they do not trigger agent or training-operation retries.

For the default Codex backend, the exporter uses the invocation's `thread_id` to find the exact session under the current `CODEX_HOME`'s `sessions/`, or `~/.codex/sessions/` when `CODEX_HOME` is unset. To use another location, add `--codex-sessions /absolute/path/to/sessions`. This selects where the exporter reads sessions; it does not rewrite Codex configuration, authentication or environment. The manifest records source and archived hashes so the filtered archive remains distinguishable from the native source.

If the native Codex session is missing, the exporter falls back to available CLI JSON stdout. For OpenCode, it exports the captured per-request JSON event stream, including available text and tool events; it does not look for a Codex session. Coverage depends on the events the selected OpenCode version actually emits. For custom backends, a prompt/response-only record is valid but has partial coverage. Inspect `export.json` before treating an export as a complete tool trace. The exporter records only supplied visible events and usage; it does not reconstruct hidden reasoning or invent token/cost figures. A gate invocation appears as a separate controller actor step with `llm_call_count=0`, tied to the recorded operation and receipt. It is not represented as another agent decision or proof of worker uptake.

To export an existing run without new agent calls:

```bash
python3 "$SKILL_DIR/scripts/export_run.py" \
  --run-dir "$TRAIN_ROOT/reports/supervision/$RUN"
```

To also create a Harbor job/trial layout for the viewer:

```bash
python3 "$SKILL_DIR/scripts/export_run.py" \
  --run-dir "$TRAIN_ROOT/reports/supervision/$RUN" \
  --harbor-out "/tmp/meta-agent-jobs/$RUN"

# Run only if Harbor is installed in this environment.
harbor view /tmp/meta-agent-jobs
```

The batch exporter uses the current Codex sessions directory by default and accepts the same `--codex-sessions` override. It refuses to run while a foreground supervisor owns the run lock; use automatic export during supervision, or batch-export after the runner stops. Existing completed passes are identified from their request outcome or durable completion event; an available response alone does not establish completion.

Viewer bundles target Harbor 0.24.0 and contain a disabled verifier, explicit import-only metadata, and no inferred rewards. Use a separate output directory; re-export preserves unrelated files and refuses to overwrite edited generated files. The bundle is for inspection and is not a runnable Harbor job. Harbor installation is needed for its viewer or optional validation, not ATIF generation. To validate one exported trajectory with an installed Harbor package, use `python -m harbor.utils.trajectory_validator /path/to/trajectory.json`.

Scheduling, feedback boundaries, recovery and the explicit project gate remain owned by the supervisor; this export layer does not replace it with a Harbor custom agent.
