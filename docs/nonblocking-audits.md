# Non-blocking training audits

The optional `meta_agent` Python modules provide the background audit coordinator
used by the training integration. The package uses the Python standard library
and Linux process identities (`/proc`); it has no training-framework or hosted
service dependency. The existing skill supervisor remains available.

## Execution and context

Call `before_step` before collecting a rollout or performing the next update,
and `on_step` after completing an update. Both poll completed work and dispatch
subprocesses without waiting for model inference or a publisher. The first
completed update and the configured cadence can trigger a review. While one
review is active, later due steps coalesce to the latest pending request rather
than starting concurrent agents.

Each request launches a new process and saves its own request, context snapshot,
response, stdout and stderr. A process boundary does not itself reset a CLI's
saved conversations: configure the backend to start a new session without a
resume flag. Supply earlier knowledge through selected experience documents,
current training metrics, logs and trajectories. Avoid automatically copying
the preceding run's live edits, policy, optimizer state or agent session into
an independently initialized run.

The coordinator is model-independent. A project-owned reviewer command may
invoke its chosen coding-agent backend and translate its response to the small
review envelope below. No specific model, account, API endpoint or credential
is included in this package.

## Minimal integration

Put this repository on the training process's Python import path. Create a
context JSON file with at least `{"run": "experiment_r1"}` before construction.
All paths passed to the coordinator must be absolute canonical paths.

```python
from pathlib import Path
import sys
from meta_agent.nonblocking_audit import NonBlockingAudit

run_dir = Path("/absolute/path/to/current-training-run")
audit = NonBlockingAudit(
    run="experiment_r1",
    directory=run_dir / "audits",       # Fresh directory for this run.
    context_path=run_dir / "context.json",
    mutable_paths=[run_dir / "live_control.json"],
    command=[sys.executable, "/absolute/path/to/reviewer.py",
             "--request", "{request}", "--response", "{response}"],
    publish_command=[sys.executable, "/absolute/path/to/project_gate.py",
                     "--request", "{request}", "--response", "{response}",
                     "--publication", "{publication}"],
    first=1, every=3, stop=22, timeout=1800,
)
try:
    for step in range(1, 23):
        audit.before_step(step, is_final=step == 22)
        # Pin the live controls for this whole update before starting work.
        metrics = train_one_update(step)
        audit.on_step(step, metrics, is_final=step == 22)
finally:
    audit.close(timeout=30)
```

Mark the final update **before** beginning it. This closes publication, cancels
an active publisher, and permits only a read-only final review. `close` drains
read-only work for its bounded timeout and then terminates the owned process
group. A request may time out or be cancelled at finalization; inspect its
record rather than assuming every scheduled review completed.

`before_step` and `on_step` are non-blocking with respect to model and publisher
work. They still perform local bookkeeping and hash explicitly declared mutable
files, so use small control files. The final `close` call intentionally permits
a bounded wait. Run the trainer under a persistent process owner for unattended
experiments; this library does not keep a stopped notebook or host alive.

## Training evidence and experience documents

`build_training_context` takes explicit relative artifact paths under a training
root. It reads bounded windows and records their byte ranges and SHA256 hashes.
Metric rows omit evaluation namespaces and non-finite scalars. Experience inputs
must be Markdown or text documents; they are included as prose, not executed.

```python
import json
from meta_agent.audit_evidence import build_training_context

packet = build_training_context(
    root=run_dir,
    run="experiment_r1",
    logs=["train.log"],
    trajectories=["training_rollouts/3.jsonl"],
    metrics=["training_metrics.jsonl"],
    experience=["experience/completed_previous_run.md"],
)
temporary = run_dir / "context.next.json"
temporary.write_text(json.dumps(packet))
temporary.replace(run_dir / "context.json")
```

Refresh the source context when new evidence is available. Already launched
requests retain their original context snapshot. Selection is explicit, not a
recursive search through a training checkout. Byte windows may be partial;
their hashes describe those windows rather than the entire changing file.
Filter or redact confidential fields before selecting inputs.

These checks establish an input contract. They do not prove that log contents
are evaluation-free, and they do not sandbox the reviewer. If the coding agent
must not access heldout data, enforce that restriction with a separate curated
workspace and filesystem/process boundary. Do not mount heldout tasks or results
in that workspace.

## Reviewer and publisher contract

The reviewer receives `{request}` and writes `{response}`. Its response is:

```json
{
  "schema_version": 1,
  "request_id": "COPY_FROM_REQUEST_ID",
  "run": "COPY_FROM_REQUEST_RUN",
  "step": 3,
  "decision": "apply",
  "proposal": {"learning_rate": 0.0008}
}
```

Use `"decision": "observe", "proposal": null` when no change is justified.
The run, request ID and completed step must match the actual request.

A separate project gate validates allowed operations and bounds. The
coordinator holds `commit.lock` while that gate runs; the gate must not acquire
that same lock again. Immediately before an atomic write, the gate must check
the current `publication_allowed` state, run identity and request's live-file
hashes. Stale proposals are rejected, not silently rebased. The gate writes:

```json
{
  "schema_version": 1,
  "request_id": "COPY_FROM_REQUEST_ID",
  "run": "COPY_FROM_REQUEST_RUN",
  "status": "applied",
  "before_state_hashes": {"ABSOLUTE_CONTROL_PATH": "OLD_SHA256"},
  "observed_before_state_hashes": {"ABSOLUTE_CONTROL_PATH": "OLD_SHA256"},
  "after_state_hashes": {"ABSOLUTE_CONTROL_PATH": "NEW_SHA256"}
}
```

Allowed statuses are `applied`, `rejected` and `stale`. A changed live file with
an absent or invalid receipt requires reconciliation, even under
`failure_policy="continue_unchanged"`. The reviewer should propose through this
gate rather than write live files itself. Direct editing needs a project's own
bounded editor and consumer-pinning adapter; those experimental integrations are
not silently enabled by this common reviewer/publisher interface.

An applied file is not evidence that a running worker consumed it or that it
improved learning. Record worker uptake and subsequent training outcomes
separately. Freeze independent evaluation settings and do not feed heldout
scores back into adaptive training.

## Direct file editing with natural-language output

`DirectFileAudit` provides the direct-editor mode used when a coding agent
already operates on the allowed hot-interface files. It reuses the same
background process ownership, fresh request/context snapshots and coalesced
cadence. It does not require a JSON final response or invoke a proxy publisher.
The agent's ordinary stdout and stderr are retained verbatim.

```python
from meta_agent.direct_file_audit import DirectFileAudit

audit = DirectFileAudit(
    run="experiment_r1",
    directory=run_dir / "audits",
    context_path=run_dir / "context.json",
    editable_roots=[run_dir / "live"],
    command=["/absolute/path/to/fresh-agent-wrapper", "--request", "{request}"],
    first=1, every=3, stop=22,
)
# Use the same before_step / on_step / finally: close integration shown above.
```

The wrapper reads the JSON request and its immutable training context, starts
the configured coding agent in a fresh session, and supplies the declared
editable roots. Actual root paths are included in each request. The project
must restrict filesystem/process access and validate live schemas and allowed
operations. Merely listing a root does not prevent a CLI from writing elsewhere.
Keep trusted audit state and context outside those editable roots.

The observer inventories files through directory descriptors, records bounded
SHA256 change sets, and never follows live symlinks to read external contents.
`direct_edits.jsonl` records what changed; `direct_receipt.json` records process
completion or cancellation. A successful process exit does not verify worker
uptake or learning benefit. A failed editor that changed files requires
reconciliation; `continue_unchanged` cannot conceal those writes.

Before the final update, the direct mode kills and reaps its owned editor
process group. This cancellation may wait up to five seconds for OS cleanup,
but never waits for model completion. No final direct-editor request is
launched. Subsequent final diagnostics need an explicitly read-only backend.
Consumers must still pin each update's effective controls. Earlier completed
edits remain live; independent runs should explicitly initialize their own
components and import prior knowledge only through selected documents.

## Environment adapter

`meta_agent.training_audit` exposes `before_training_step`,
`audit_training_step` and `close_training_audit`. With no
`EAU_ASYNC_AUDIT_COMMAND` they are no-ops. With it configured, they build the
same coordinator. The configuration is:

| Variable | Meaning | Default |
| --- | --- | --- |
| `EAU_ASYNC_AUDIT_COMMAND` | Reviewer argv encoded as a JSON array | Disabled |
| `EAU_ASYNC_AUDIT_PUBLISH_COMMAND` | Project-gate argv as a JSON array | None |
| `EAU_ASYNC_AUDIT_RUN` | Unique run ID | Required |
| `EAU_ASYNC_AUDIT_DIR` | Fresh absolute audit directory | Required |
| `EAU_AUDIT_CONTEXT_PATH` | Absolute context JSON with the same run ID | Required |
| `EAU_ASYNC_AUDIT_MUTABLE_PATHS` | JSON array of absolute control-file paths | `[]` |
| `EAU_ASYNC_AUDIT_FIRST` | First completed update to review | `1` |
| `EAU_ASYNC_AUDIT_EVERY` | Review cadence in completed updates | `20` |
| `EAU_ASYNC_AUDIT_STOP` | Final training update | `624` |
| `EAU_ASYNC_AUDIT_TIMEOUT` | Timeout per reviewer/publisher process, seconds | `1800` |
| `EAU_ASYNC_AUDIT_DRAIN_TIMEOUT` | Bounded read-only final drain, seconds | `30` |
| `EAU_ASYNC_AUDIT_FAILURE_POLICY` | `raise` or `continue_unchanged` | `raise` |
| `EAU_ASYNC_AUDIT_DIRECT_ROOTS` | JSON array of editable directory roots; selects direct-editor mode | Disabled |

Configure one coordinator per trainer process and use a fresh process/run for a
new independent experiment. Do not combine this adapter with the blocking
`EAU_STEP_REVIEW_STATE` barrier.
For direct mode, leave `EAU_ASYNC_AUDIT_PUBLISH_COMMAND` and
`EAU_ASYNC_AUDIT_MUTABLE_PATHS` empty and declare `EAU_ASYNC_AUDIT_DIRECT_ROOTS`
instead. The command may finish in natural language.

## Offline validation

From the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 examples/nonblocking/train_demo.py
```

The demonstration uses synthetic training metrics, a slow deterministic
reviewer fixture, an atomic scalar gate and temporary files. It makes no model
or network calls and is not a benchmark result. It illustrates a trainer
continuing while a review runs, coalesced requests, reloaded controls and
read-only finalization.
