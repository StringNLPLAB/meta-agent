---
name: meta-agent
description: Set up, launch, supervise, and improve model-training experiments, including pretraining, supervised fine-tuning, and reinforcement learning, from code and natural-language requirements. Discover candidate components, periodically review logs and training samples, test interventions, verify uptake and outcomes, and plan intra-run or restart-required changes across training frameworks.
---

# Meta-Agent

Improve training through evidence-driven investigation and intervention. Understand the objective and allowed feedback, generate mechanism hypotheses, map them to the actual implementation, and decide what is feasible and worth testing. Existing controls are execution paths, not the limit of the search.

Apply this workflow to pretraining, self-supervised learning, supervised learning/fine-tuning, RL, and agent training. Derive the objective, evidence, candidate components, and review unit from the actual training recipe. Data quality and sampling, preprocessing, loss construction, optimization, batching, resource use, and instrumentation are possible surfaces where present. Rewards, rollouts, environments, and user simulators apply to recipes that contain them. EAU provides a methodological reference; its training algorithm and component layout are not prerequisites.

## Own experiment setup, launch and supervision

When this skill is loaded **inside a runner request**, complete only the requested discovery/review/reconciliation/final pass and return its structured response. Do not launch training, recursively launch a supervisor, or write the controller's records. The top-level experiment agent owns orchestration; the runner's reasoning calls remain read-only.

For a top-level request to run or improve an experiment, take responsibility for the whole workflow. Accept the skill location, training-code location, and ordinary language about objectives, resources, budgets, review timing and permitted changes. Inspect the existing launcher, recipes, data paths, configuration and run records; translate these requirements into concrete arguments and an experiment contract yourself. Reuse known session constraints and sensible project defaults. Ask only for essential constraints that cannot be established from the request or repository; do not require the user to write JSON, choose CLI flags, start training, or keep a shell session open.

Before each experiment, establish the contract below, reuse the cached EAU setup summary and read relevant current experience, inspect the code and control readers, and produce the intervention map. Paper reading belongs to setup, not the normal experiment loop. Identify candidate components and unknowns before proposing or applying changes. For an already-running job, perform this initial assessment before steering it and identify the takeover step without inventing a past preflight.

Use the [runner instructions](references/supervision.md) as implementation guidance for the agent. Resolve exact run/log/sample paths, a monotonic progress counter or time-based schedule, and a bounded supervision-call budget, write the contract, and run the `--once` discovery preflight. For noninteractive training, the runner's `--trajectories` input can hold examples, predictions, or batch diagnostics. After discovery completes, launch the project's training entry point with the resolved configuration within the authorized experiment scope. Record the actual launch command, code/configuration provenance, trainer identity, logs and checkpoint location. Verify that training started successfully.

Start continuing supervision through the host's supported process manager or scheduler and record its actual process/task identity and next review trigger. The bundled `scripts/supervise.py` invokes a skill-loaded agent at configured time/step boundaries; it is a foreground worker, so the top-level agent must arrange its lifetime. Verify that the trainer and supervisor remain alive and bound to the intended run. A transient command/session identifier alone does not prove supervision will survive the current turn; establish a real owner or report the specific host limitation. A watcher that only checks records does not substitute for this agent loop.

Enable `--export-atif` when recording the meta-agent's own trajectory, and retain pending predictions through completion or handoff. Default execution is proposal-only. When the task authorizes interventions, configure a compatible project gate and declared operation scope yourself; keep runtime validation and uptake checks intact. Continue routine launch and supervision work under the user's existing authorization rather than asking the user to execute generated commands.

## Establish the experiment contract

Before selecting interventions, record:

- The primary objective, decision horizon and remaining budget (including supervision and attempted runs), unacceptable regressions, and checkpoint-selection rule.
- A unique run identity and baseline: launched code/data revisions, resolved configuration, environment overrides, inherited model/optimizer/data state, current phase, and recent interventions.
- The feedback allowed to steer this experiment, and the information reserved for independent evaluation and reporting. The existence of a test set does not authorize reading its feedback.
- A fixed, versioned evaluation dataset/distribution, preprocessing, metrics and scoring semantics; include generation settings, environment/user configuration, interaction limits, or judge/rubric where applicable. Bind every result to its exact checkpoint and evaluator version. After evaluator repairs, rescore both sides when raw data permits; otherwise identify the incomparable portions.
- Review timing based on update boundaries, propagation and learning delays, noise, and budget. Declare whether supervision is one-shot, session-bound, or owned by a persistent runner, with the actual next wake-up mechanism and responsible party. A report or watcher alone does not arrange a future agent review.
- The project's existing intervention map, change log, experience, inventory, and next-run plan; otherwise keep records under `reports/` in the training checkout, outside the skill installation.

When reproducing EAU, follow the bundled paper's training-side-only protocol (§§4.6 and 5.1): evaluation/test instances, trajectories, scores, and derived summaries cannot guide interventions, accept changes, enter the experience pool, or steer later runs. It reports the final run and final checkpoint without test-based selection. If a different experiment explicitly permits development-set steering, label it as development evidence, document adaptive reuse, and reserve independent final validation. Keep selection and feedback access consistent with the declared protocol.

## Discover hypotheses and execution paths

During initial setup, read the bundled [EAU reference](references/eau_manuscript_ref.pdf) once, focusing on its method, intervention examples, and failure analyses. Save the source version, candidate mechanisms, applicability conditions, and limitations in a compact summary within the existing project experience or intervention map. Later experiments and periodic reviews reuse that summary and current supported, refuted, and unresolved experience; they do not routinely reopen the paper. Refresh setup if the reference version changes or its saved summary is unavailable. Cite the summary actually consulted rather than claiming a new direct paper reading.

Follow [intervention discovery](references/intervention-discovery.md) for the ordered investigation, a versioned paper mechanism map, implementation tracing, measurement, and recovery. Generate each experiment's candidates from the current objective, code, experience and observations, using applicable cached mechanisms. The paper's examples are not a component checklist: mark absent components inapplicable and retain useful candidates from the current project. Distinguish paper reports, transfer hypotheses, current evidence, and verified outcomes; paper reports do not establish defects in this run. Reassess applicability as conditions change using the saved mapping. Inspect representative raw samples alongside population metrics and subgroup distributions, including opportunities in healthy training: redundant data, coverage gaps, saturated tasks, weak learning signal, ineffective updates, and resource bottlenecks. For interactive recipes, also inspect redundant interactions and narrow behavior.

Extend the existing intervention map rather than creating a parallel registry:

| Field | Required finding |
|---|---|
| Hypothesis basis | Specific paper section, relevant past experience, current sample/metric, or code finding |
| Applicability | Conditions needed for transfer; matches, mismatches, and unknowns for this task/model/phase |
| Mechanism and target | Behavior or bottleneck to change, and why this module governs it |
| Runtime feasibility | Hot, boundary-applied, restart-only, frozen, or unknown, with implementation evidence |
| Uptake evidence | Source, reader, read timing, caches, process scope, consistency boundary, and observable use |
| Value | Relation to the objective, expected benefit, implementation/measurement cost, and remaining exposure/window |
| Execution and recovery | Allowed scope/range, interface, coupled state, failure handling, recovery, and current authorization |
| Decision | Apply, instrument, observe, defer, leave unchanged, or prepare a restart, with the next decision condition |

Keep four judgments distinct: a source motivates investigating it; current evidence supports changing it; the running system can apply it; the remaining budget makes it worth testing. Preserve promising mechanisms that need instrumentation, a new control, code changes, or a restart. If adding a control, specify its reader, validation, synchronization, uptake signal, and rollback; never assume editing Python updates resident workers.

Use native project tools. The optional [EAU / verl adapter](references/eau-adapter.md) documents verified code paths, helper interfaces, review records, and limitations; its project conventions are not universal requirements.

## Complete each scheduled review

1. Reassess active hypotheses using only permitted feedback. Keep predictions pending until their registered exposure window and sample requirement mature. At the end, classify supported, refuted (actual counterevidence), or unresolved; missing or weak evidence is not refutation. Record local mechanism predictions separately from final task benefit: one can hold without the other.
2. Read relevant experience before generating new hypotheses. Inspect available examples, targets, predictions, per-sample/batch diagnostics and subgroup/population evidence; include action/observation trajectories for interactive recipes. If new evidence is unavailable, record that limitation and a useful next action; do not invent an observation or repeat a refuted test without changed conditions.
3. Rank candidates by applicability, expected value, implementation cost, and time to a useful answer. Choose the smallest mechanism-level scope that tests the hypothesis, with an amplitude large enough for an informative comparison. Prefer one mechanism at a time; label inseparable bundles.
4. Choose and record the next useful action: apply a justified intervention, instrument a missing signal, continue a pending window, defer with a concrete implementation/recovery plan, or leave controls unchanged with a reassessment trigger. Every review needs a reason and next decision condition; it need not change a configuration, add a knob, or yield a general principle.
5. Before applying, register baseline, mechanism and task predictions, criterion, exposure/window, uptake test, recovery and interruption conditions. Check expected old values/version, use native validated and atomic/idempotent operations when supported, and preserve unrelated settings. After a timeout inspect actual state before retrying. Partial acknowledgments remain partial.
6. Verify who actually consumed the change and at which boundary. Writable files do not prove uptake; uptake does not prove benefit. Track in-flight data and updates before declaring a clean comparison window. Record application status separately from prediction verdict, including pending, partial, failed, and control-reverted states.
7. Save the review position, active configuration/intervention state, pending predictions, next condition and actual wake-up ownership so a successor avoids duplicate actions. Record missed reviews as gaps, never retrospective reviews. Update experience with evidence levels and applicability rather than universalizing a single result.

Use the project's record format. Include run and actual step/time, decision, target, old/new state for applications, source/sample evidence or its absence, mechanism, prediction/window, application attempt, effective boundary, uptake, outcome, and recovery/lineage. Use the adapter's eight-column tagged records only with its checkers. A complete record of a failed or pending application is a complete review, not a successful intervention or a reason to retry automatically.

## Measurement and recovery

Compare with the same detector and scoring semantics; check precision, dynamic range, coverage, missingness, denominators, noise, autocorrelation, subgroup composition, and delay. Use independent corroboration permitted by the feedback contract. Selected dumps diagnose behavior; they do not establish population rates. A nonstationary before/after change is not automatically causal. See the discovery guide for practical criteria and common misinterpretations.

Determine whether a defect affects observation only, future data, or already-applied updates. A measurement repair does not automatically require a restart. Repair incorrect targets, loss/feedback semantics, and data or applicable environment channels without merely making metrics look healthy; assess useful learning signal separately. Choose among continuing, reverting a control, restoring a compatible checkpoint, and starting fresh according to contamination, state compatibility, the objective, and budget. Reverting a setting leaves prior learning and data exposure in the lineage.

Respect current task authorization and operator-only controls; past project permissions do not authorize future actions. Before stopping processes or cleaning checkpoints, verify ownership and required checkpoint state. Report limitations when supervision ends, versions cannot be observed, or only some consumers confirm uptake.

At completion or handoff, record the selected checkpoint and selection rule, final evaluation/version (or its pending owner), consumed budget, effective configuration and intervention history, unresolved predictions, and code/data versions plus patches needed to reconstruct the experiment. Stop only the helpers owned by this supervision task, using their recorded IDs.
