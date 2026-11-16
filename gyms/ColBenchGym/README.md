# ColBench backend-programming environment

`colbenchgym.ColBenchCodeEnv` supplies the local backend-programming port used
by the bundled verl trainer: direct text questions, an LLM or symbolic flow
user, answer extraction and subprocess-based test-case scoring. It is a
training integration, not a complete frontend/browser implementation.

See [the experiment guide](../../examples/colbench/README.md) for the pinned
public dataset, actual PPO recipe, non-blocking editor, five-round launcher and
evaluation boundary. Editable training behavior is distinct from a frozen
external benchmark evaluation.

Generated and reference programs run in subprocesses with timeouts. Those
subprocesses do not constitute a secure sandbox for arbitrary programs; use
the documented container deployment on an isolated training host.
