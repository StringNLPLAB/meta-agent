# EAU / verl adapter

Use this reference only for compatible EAU/verl layouts. It describes bundled helper behavior and a code inspection of `InfoPO_handover_20260914`; it does not authorize launching training, installing supervision, or changing a recipe. Run records belong in the training checkout, never the skill installation.

## Verified layout and version boundaries

The inspected handover has sibling `InfoPO/` and `UserRL/` roots. Relevant evidence is in `InfoPO/HANDOVER.md`, `meta_agent/knob_inventory.md`, `reports/{change_log,meta_agent_experience,next_run_config}.md`, trainer/reward code under `InfoPO/verl/`, and gym readers under `UserRL/gyms/`. ColBench's `flow_user.py` and `code_env.py` are under `InfoPO/gyms/ColBenchGym/colbenchgym/env/`. Verify the actually imported copies before interpreting a live run; duplicate source trees and old absolute `/workspace/cm_workspace/...` defaults are present.

Bundled helpers use:

- `EAU_ROOT`: required training checkout, containing `logs/`, `reports/`, and trainer code.
- `EAU_GYM_ROOT`: optional actual gym source root, default `$EAU_ROOT/gyms`. For the handover's external symbolic readers, select the sibling `UserRL/gyms` after verifying imports. This affects helper snapshots, not training imports or the readers' own `USERRL_*_CONFIG` paths.
- Logs: `logs/train_<name>_<YYYYMMDD>_<HHMMSS>.log`, step lines containing `step:N - global_seqlen`.
- Records: `reports/change_log.md`, `reports/.mandate_state.json`, `reports/manifests/`, and `reports/run_aliases.json`.
- Diagnostic snapshots: `$EAU_ROOT/{entropy_schedule,clip_schedule}.json`, `$EAU_ROOT/meta_agent/reward_shaping.json`, `$EAU_GYM_ROOT/symbolic_layer/{env_policy,process_reward,difficulty}.json`. Missing versions are absent diagnostics, never a failed review or proof of equality.

The handover's old helpers instead use `InfoPO/meta_agent/run_aliases.json`, including historical short labels. The bundled helpers consistently use `reports/run_aliases.json` and infer `<name><YYYYMMDD><HHMMSS>` from the full filename. Register verified mappings for the exact inferred labels; do not guess truncated aliases or rewrite old records. Invalid/cyclic mappings and conflicting explicit run/log identities fail clearly.

The bundled 2026-10-12 manuscript and 2026-09-14 handover are different snapshots. For example, Appendix A.3 / Table 8 restricts selector support and the predicate library in its documented implementation; the inspected `meta_agent/env_policy_gate.py:KNOWN_MIX_GYMS` includes `tau`, and `UserRL/gyms/symbolic_layer/process_reward.py` plus the gym call sites include additional predicates such as `repeat_answer`. Neither inventory replaces inspecting the launched code and actual strategy exposure. Historical operator-only restrictions or later permission changes do not transfer to a new task.

## Paper-guided mechanism mapped to code

Appendix A.1–A.2 motivates checking repeated disclosure and fact-type filtering; it does not claim a repair succeeded. In the inspected ColBench code:

- `FlowUser.reply()` in `flow_user.py` checks the overall `max_reveals` cap at entry. Within its loop, an already-revealed fact is appended without incrementing `reveal_count`; repeating facts can therefore evade accounting while the count remains below the cap. This is not a claim that repeats bypass an already-exhausted cap.
- `reveal_return_shape` excludes `Fact.kind == "shape"`; it does not target `kind == "return"`. Diagnose actual fact types before choosing a filter.
- `ColBenchCodeEnv._flow_reply()` in `code_env.py` refreshes `self._flow.knobs = get_colbench_flow()` before each reply. In online mode, the existing reader can affect turns of an episode already in flight; it does not provide episode-wide configuration consistency.
- `UserRL/gyms/symbolic_layer/env_policy.py:_HotConfig.get()` caches by `(mtime_ns, size)` and keeps the last good value on read/parse errors. Its load log is local evidence, not an acknowledgment from every worker.

A candidate can therefore be a repeat-disclosure counter/budget with a per-turn fact-state probe, or a fact-type-aware disclosure policy. Check current trajectories, allowed disclosure, existing refuted hypotheses, and task performance before applying either. New Python logic requires a restart unless this runtime demonstrates another supported path; the current numeric knobs do not implement those repairs. Missing trajectories warrant instrumentation with applicability unknown, not a diagnosis of current leakage.

Other verified paths: `InfoPO/verl/trainer/ppo/ray_trainer.py:_scheduled_entropy_coeff()` and `_scheduled_clip_high()` resolve values before `update_actor` and place them in `batch.meta_info`; `InfoPO/verl/workers/actor/dp_actor.py` consumes those values and reports `actor/entropy_coeff` and `actor/clip_ratio_high`. Inspect actor warmup and actual batch/update provenance; no fixed warmup length is assumed. Gradient clipping and skipped/applied optimizer updates are separate signals.

The symbolic readers and trainer loaders inspect `USERRL_META_ARM` at import. `online` rechecks files on access, `frozen` caches the first successful read, and `off` uses defaults or the Hydra/static path. Editing files in frozen/off arms is not evidence of changed runtime behavior. No bundled helper adds readers, coordinates workers, authenticates acknowledgments, drains queues, or restores training state.

## Review schedule, identity and actual supervision

Choose `--interval N` or set `EAU_REVIEW_INTERVAL=N` for this run. There is no built-in 20-step cadence. `EAU_MANDATE_INTERVAL` remains a legacy spelling; conflicting environment values fail. `--first-review S` (or `EAU_FIRST_REVIEW`) sets the first boundary, then the interval repeats. Both watcher and pending checker skip boundaries at/before the selected log's `Setting global step to N` resume anchor. Review cadence is not the prediction-verification window.

Use the exact `--log` and, when useful, `--run` with a registered alias. With several logs, automatic selection fails instead of following whichever run most recently wrote a log. A watcher stays on its selected run and never switches to another run by mtime.

When supervision is authorized and compatible, example invocations are:

```bash
python3 "$SKILL_DIR/scripts/run_watch.py" --log "$TRAIN_LOG" --run "$RUN" --interval "$REVIEW_INTERVAL"
python3 "$SKILL_DIR/scripts/pending_checkpoints.py" --log "$TRAIN_LOG" --run "$RUN" --interval "$REVIEW_INTERVAL"
bash "$SKILL_DIR/scripts/checkpoint_watchdog.sh" --log "$TRAIN_LOG" --run "$RUN" --interval "$REVIEW_INTERVAL"
```

For actual recurring agent reasoning, use the [supervision runner](supervision.md); it also supports the EAU step format and exact run/alias binding. The optional watchers below remain detection tools.

The watcher prints metrics and runs the checker read-only; it neither performs a review nor schedules an agent. The watchdog records pending-review transitions every `WATCHDOG_INTERVAL` seconds (legacy default 600) and exits on TEARDOWN. It does not provide persistent agent reasoning. Arrange a host-supported wake-up only if requested/authorized, and record owner, lifetime, next wake-up, task IDs/PIDs, and supervision gaps. No particular Monitor/Cron API is assumed.

The handover separately contains an autonomous `InfoPO/meta_agent/loop.py`, launched by `InfoPO/scripts/train_director.sh`. Its defaults poll the log every 60 seconds and trigger after 5 new steps (also on its first observed step): `observer.collect` → `analyst.analyze` / `reward_agent.run_reward_agent` → gate validation → optional atomic symbolic/reward-config writes. These are legacy project defaults, not this skill's cadence. The analyst uses its own fixed prompt and GLM model call; this does not load the revised skill. It records JSONL actions and a guard control file, not the tagged review/uptake contract. The last analyzed step lives only in process memory; guard recommendations do not themselves restore checkpoints. Even `--dry-run` can call the model and write logs/control status. This legacy external loop is not bundled or started. The bundled `supervise.py` provides the revised skill/contract integration separately, using the declared agent backend and optional project gate; the watchers alone still do not provide autonomous reviews.

Pending exits 1 for incomplete reviews, 0 when none are owed or the selected run is declared ended, and 2 for argument/identity/schedule errors. It never borrows another run's resume anchor or process liveness. Without a caller-verified `--trainer-pid`, liveness remains unknown; `--finished` explicitly declares an ended run and requests TEARDOWN. A vanished/nontrainer caller-bound PID also requests teardown. Verify PID ownership and reuse; remote/distributed liveness needs project support. Historical gaps remain gaps, even when acknowledged.

`run_watch.py --autokill` retains three **legacy recipe heuristics** (success/length collapse, unclosed-call rates across gyms, repeated skipped updates). They are uncalibrated in another setting, and its signaling selector can match multiple local verl trainers. Leave signaling disabled unless criteria, exact process scope and current termination authorization have been established; this adapter does not fix that selector. Token-entropy growth alone is not a collapse verdict. Offline replay never signals or writes checker state:

```bash
python3 "$SKILL_DIR/scripts/run_watch.py" --log "$TRAIN_LOG" --run "$RUN" --interval "$REVIEW_INTERVAL" --replay
python3 "$SKILL_DIR/scripts/read_trajectory.py" /path/to/dump.json --index 0
```

## Eight-column reviews and compatibility

Reuse the existing table:

```markdown
| run | step | tier | knob | change | evidence | prediction | verification |
|---|---|---|---|---|---|---|---|
```

New reviews put lowercase `field: value` tags, separated by semicolons, in existing cells. Do not duplicate tags; their values end at the next semicolon. There is no new registry. Every new row uses `decision: ...`, `reason: ...`, `next: ...`, and an evidence cell with the actual source/observation or an explicit absence and reason. `next` specifies the next decision condition; prediction verdicts remain separate text in the prediction/verification cells.

| Decision | Additional fields checked |
|---|---|
| `apply` | `target`, `old`, `new`, `mechanism`, `window`, `attempt`, `recovery`, `application`, and a non-placeholder prediction with a criterion |
| `instrument` | `question`, `method`; the common `next` field states the collection/decision trigger |
| `observe` | `window` or `samples` describing the pending exposure/sample requirement |
| `defer` | `limitation`, `plan` preserving the mechanism and proposed implementation |
| `restart_plan` | `limitation`, `plan`, `recovery` describing compatible restoration or fresh initialization |
| `leave_unchanged` | Common reason and reassessment trigger; no fabricated prediction required |

For an apply review, `application` is `pending`, `partial`, `failed`, `fully_applied`, or `control_reverted`. The `attempt` describes the actual request/receipt/error or why no request was sent. A pending/partial/failed application may have a complete review without uptake; this never implies successful application or requires an automatic retry. `control_reverted` records only the control recovery; describe inherited learning and lineage in `recovery`.

`fully_applied` additionally requires `boundary`, `uptake`, `scope`, and `confirmed`. The last two list the expected and acknowledged consumers, using comma-separated stable names; the sets must match. Unknown/pending observations do not qualify. Record the consumed value/version and traceable observation, not just a write receipt. These are structural checks of a claim: even a passing checker does not authenticate runtime evidence, queue provenance, a prediction verdict, or task benefit.

Example of a legal prediction-free observe review using the Python API:

```python
import sys
sys.path.insert(0, "/absolute/path/to/skill/scripts")
from append_row import append_row

append_row(
    run="RUN", step=40, tier="user", knob="disclosure",
    decision="observe",
    change="reason: H1 needs more admitted episodes; window: clean exposure through step 60; next: review at step 60 or 200 admitted episodes",
    evidence="trajectory: sample 37 repeats a known fact; population sample collection is incomplete",
    prediction="H1 pending; retain its registered mechanism and task criteria",
    verification="not mature",
)
```

Use actual evidence and the actual step. The append helper accepts a custom `path`, verifies that path, and escapes pipes as HTML entities while preserving displayed text. The parser also accepts escaped Markdown `\|`. Hand-written unescaped pipes can still break a row.

```bash
python3 "$SKILL_DIR/scripts/check_table.py"
python3 "$SKILL_DIR/scripts/check_mandate.py" --step "$STEP" --run "$RUN" --log "$TRAIN_LOG"
```

`check_mandate.py` keeps its historical filename and `.mandate_state.json` snapshots. Config version changes are diagnostic only; the old recent-log/restart-mtime shortcut is removed. First invocation does not excuse a missing review. `--baseline` snapshots only and never completes a review; `--read-only` performs no state writes. Rechecking a step uses the same earlier snapshot and revalidates its records. Missing or malformed reviews, wrong identities, and incomplete fully-applied claims still fail. One complete row cannot conceal an invalid application row at the same step.

Untagged eight-column history with evidence and a non-placeholder prediction counts as legacy coverage, explicitly unverified; it is not migrated or certified as uptake. New reviews should use the tags above. A structurally valid row whose knob is `missed` or begins `missed review` or `missed checkpoint` acknowledges a gap; pending reports it separately instead of inventing work. Malformed gap rows and conflicts between gap markers and review claims remain incomplete; do not backfill. Historical sweep prompts in the handover are archived run instructions: their forced writes and fixed cadences are not the current contract. No sweep prompt is bundled here, and those records are not rewritten.

## Measurement, manifests, retention and handoff

`HANDOVER.md` §5 identifies reward-stratified debug dumps and a `success_rate` computed as `n_positive / n`, with `n_positive += (final_reward > 0)`, in `InfoPO/verl/workers/reward_manager/naive.py`. Process rewards can inflate that rate; it is not automatically task completion or invariant to every reward change. Separate the benchmark's completion criterion, per-turn reward, flow/LLM subgroups and denominators. This corrects assumptions in older experience and retention scripts. The paper's final evaluation remains fixed; use only protocol-permitted measurements for steering.

Capture configuration/code snapshots with the selected identity:

```bash
python3 "$SKILL_DIR/scripts/run_manifest.py" capture --run "$RUN" --log "$TRAIN_LOG"
python3 "$SKILL_DIR/scripts/run_manifest.py" diff BASELINE_RUN RUN
```

Manifest capture does not select a generic trainer process. Add a caller-verified `--trainer-pid` only when live process environment capture is needed; otherwise the manifest labels environment state unobserved. File hashes/config snapshots are not the loaded revision or effective worker state. The code/path/variable list is project-specific and incomplete; extend it for an actual new control, not this skill update.

The handover has `meta_agent/ckpt_keep_best.sh` and destructive `ckpt_janitor.sh`; neither is bundled, invoked, or installed by this skill. The former includes a fixed 20-step score band and shard-count/size assumptions; its metric needs checking against the contract and shaping semantics. The latter removes optimizer state and changes resume capability. Do not transplant their defaults. Determine expected actor/critic/optimizer/scheduler/data state and world size from the actual trainer; stage and verify complete backups before replacements. Retention candidates must respect the declared checkpoint-selection protocol.

At end/handoff, record selected checkpoint, final evaluation/version or pending evaluation owner, actual budget, effective overrides and intervention lineage, unresolved predictions, missed reviews, reconstructable revisions/patches, and next owner/wake-up. Retire only this run's owned watcher/sweep/watchdog/retention tasks by their recorded IDs. The foreground `supervise.py` runner persists and resumes supervision state, but no OS service, all-worker acknowledgment, queue tracker, checkpoint restoration, or live-training verification is supplied.
