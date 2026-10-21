# Discovering worthwhile interventions

Read this at the initial assessment and when new evidence changes the hypothesis or runtime surface. Contents: candidate discovery; paper mechanism map; implementation and uptake; measurement; recovery and experience.

## Generate candidates before choosing controls

1. Establish the primary objective, decision horizon, remaining budget, allowed feedback, and unacceptable regressions from the experiment contract.
2. Reuse the EAU mechanism/applicability summary saved during setup, or the versioned mechanism map below, then read relevant supported, refuted, and unresolved project hypotheses. The full EAU paper is read during setup only; later experiments and reviews use the cached knowledge. Check whether task, model, phase, reward semantics, and implementation conditions transfer. Do not import independent test feedback through old experience under a training-only protocol.
3. Read representative raw samples/trajectories together with population metrics, subgroup distributions, and implementation details. Look for counterexamples as well as apparent successes; reward-stratified dumps cannot estimate rates.
4. Identify both broken behavior and headroom in healthy training: repeated or low-information interactions, saturation of easy tasks, insufficient feedback, narrow policies, or an experience distribution poorly matched to the current policy.
5. Form the mechanism hypothesis first, including a competing explanation and a distinguishing observation; then choose a module or control. A metric moving in the desired direction can be caused by repetition, leakage, or reward rescaling.
6. Trace source, reader, caching, propagation, effective boundary, consumer scope, and verification signals in the launched implementation. Preserve candidates whose implementation is currently unavailable.
7. Rank a short list by expected value, applicability evidence, cost, exposure and verification time. Record the decision and the condition that would change it in the existing intervention map.

**Existing knobs describe available execution paths; they do not bound the search. Keep promising mechanisms visible when they require new instrumentation, a new runtime control, a code change, or a restart.** An unrelated hot write does not advance a hypothesis whose real target is restart-only.

## EAU as a source of hypotheses

The bundled [EAU manuscript](eau_manuscript_ref.pdf) is an 18-page preprint, PDF creation metadata `2026-10-12T10:00:18Z`, SHA-256 `a7128aa6ef3f6ff2203d3a4662499e5ef18d281a3e5d13997d7469debde4d323`. No explicit revision identifier is printed. These locations refer to this local version, not another draft or the older handover code.

During initial setup, read §§4.1–4.6 (pp. 4–7), §5.1 (pp. 7–8), §5.4 (p. 9), and Appendix A.1–A.3 / Tables 6–8 (pp. 14–16). Retain a compact mechanism/applicability mapping with this source identifier in the existing experience or intervention map. Subsequent experiments and periodic reviews reuse that mapping; starting a new run does not require another full-paper read. Refresh setup when the reference version changes or the saved mapping is unavailable. The table below is a bundled versioned summary that can be reused without reopening the PDF; identify whether a finding comes from that summary, saved setup notes, or a direct setup reading.

| Paper report and location | Candidate mechanism / symptom to investigate | Applicability, limits, and distinguishing evidence |
|---|---|---|
| §4.2, Fig. 4: explicit disclosure rules can answer unknown facts without hallucination | Inspect unsupported answers, premature disclosure, intent drift; consider stateful rules or controllable simulator prompts | Requires a user/information provider and a specification of allowed disclosure. Inspect actual dialogues before claiming leakage; symbolic rules can themselves leak or omit states. |
| §4.2, §5.4 Table 3: symbolic/LLM mixture; symbolic-only scores worse in that ablation | Investigate diversity versus reliability and experience exposure by simulator strategy | Confirm that the selector reaches each task and measure actual mixture. Appendix A keeps the coefficient fixed at 0.5 in its analyzed ColBench runs; it does not demonstrate benefit from dynamic mixture tuning. |
| §4.3, Fig. 5: intermediate progress rewards and interaction constraints | Investigate sparse feedback, unreachable progress, reward bypasses and termination incentives | Identify useful task progress and effective advantages. Do not assume another positive reward term or a shorter episode improves the objective. |
| §4.4: configurable optimization dynamics with a fixed underlying training algorithm | Investigate exploration, skipped updates, and ineffective optimization | Trace the optimizer's real consumers and state. Token entropy is not behavioral diversity; an exposed hyperparameter need not be hot. |
| §4.5–4.6, Fig. 1: intra/inter-run changes and reuse of experience; editable knob list | Retain upstream code repairs, new controls, and restart candidates alongside current knobs | Confirm loaded code and recovery state. The framework's support for hot swaps does not prove a particular module can change live. |
| Appendix A.2 Table 7: `min_questions` counter omitted combined question/answer turns; later threshold increases differed in effect | Investigate deadlock and the state transition that recognizes a question | Separate fixing counter semantics from optimizing its threshold; a bug repair is not evidence that a larger threshold is better. |
| Appendix A.2: `facts_per_reply=3` exposed reference content; volunteering improved flow score while question rate fell | Investigate information quality, not just score or turn count | Compare replies with permitted facts, new information per turn, and question diversity. Do not copy the paper's counts or thresholds as current-run facts. |
| Appendix A.1–A.2: `max_reveals` omitted repeat disclosures; `reveal_return_shape` filtered `shape`, while the diagnosed leak involved `return` | Inspect repeat-state accounting and fact-type coverage; consider a new repeat budget or type-aware filter | These are documented interface limitations, not successful repairs. Confirm reachability in the current code and observations; retain an implementation/restart candidate if no existing control governs the mechanism. |
| Appendix A.2: local harms, retracted attribution, inconclusive threshold changes; entropy rose while diversity fell and lag exceeded the window | Register mechanism-specific outcomes, delays, reversals or replication where useful | Local observations are not controlled contributions to the final result. Non-effects below the noise floor remain unresolved; reversal also has carryover from prior learning. |

For example, without disclosure trajectories, propose sampling replies with fact-type and prior-disclosure state; mark leakage applicability unknown. If repeat disclosure is observed and the reader only counts new facts, preserve a repeat-accounting code/control candidate. Do not adjust unrelated entropy to meet a review deadline. The [adapter](eau-adapter.md) maps this hypothesis to the handover implementation and records version mismatches.

EAU's user, environment, reward, and optimization directions are functional prompts. A system without a user simulator has no user-simulator intervention: map to data/feedback generation where appropriate, or mark it inapplicable.

## Trace the actual implementation

Follow launch → resolved config → trainer/worker lifecycle → source → reader → propagation → effective boundary → observable use. Inspect the launched revision, imported module paths, runtime APIs, callbacks, schedulers, dataset iterators, prefetched buffers, and distributed broadcasts. A mutable file, CLI flag, or Python object alone is not a live control.

| Surface | Mechanisms and implementation questions |
|---|---|
| Data and curriculum | Mixture, difficulty, filtering, weights; does the active sampler read updates, and which cached/prefetched examples remain? |
| Objective and feedback | Loss/reward terms, judge policy, constraints; where are they computed, do their semantics change, and are incentives already broken? |
| Optimization | Learning rate, clipping, regularization, update ratios; which optimizer consumes the value, can schedules overwrite it, and is state migration needed? |
| Interaction and generation | Prompts, tools, dynamics, decoding, exploration; which workers cache them, and which task/experience semantics change? |
| Compute allocation | Rollouts, token/task budgets, evaluation frequency; does reshaping state require restart, and what useful exposure is lost? |
| Observation | Logging, unbiased samples, independent permitted signals; will collection resolve a specific uncertainty without biased denominators? |

For supervised training, inspect examples, predictions, losses and data slices. For RL/agent training, also inspect actions, observations, per-turn feedback and exposure. Instrument a missing distinguishing signal before treating a plausible mechanism as established.

Classify runtime feasibility from evidence:

| Class | Meaning |
|---|---|
| Hot | Supported update reaches resident consumers without restarting |
| Boundary-applied | Applied at an identified episode, batch, update, epoch, or evaluation boundary |
| Restart-only | Requires rebuilt workers/iterators/graphs or compatible restored state |
| Frozen | Fixed by protocol, runtime arm, or current operator constraint |
| Unknown | Read path, scope, or consistency has not been established |

A new control needs a reader, validation, synchronization boundary, observable uptake and recovery path. Adding that reader can itself require a restart. Use only capabilities actually supported; record gaps rather than building a new control framework by default.

### Apply and verify consistently

Choose the unit that must share one configuration: episode, comparison group, gradient-accumulation update, or evaluation batch. Record old value/version preconditions and expected consumers. Prefer native atomic validation and idempotent request IDs when available. A local atomic file replace does not synchronize workers. On timeout inspect actual state before retrying; without acknowledgments or reliable version observations, record pending/partial/unknown rather than fully applied.

Track in-flight rollouts, prefetched batches, reward requests, replay/buffers and asynchronous evaluations. State whether old data is drained, retained and labeled, discarded through a supported path, or untraceable. Separate control-read time, actual data exposure, update provenance, and adaptation delay. A worker acknowledgment at step 100 is not a clean learning window when steps 101–110 consume old batches. Late evaluation belongs to its source checkpoint and evaluator, not the arrival step. Unknown provenance limits comparison claims.

### Decide what is worth testing

Assess objective headroom, applicability, distinguishing evidence, implementation and measurement cost, propagation/adaptation delay, remaining samples, confounders, and recovery cost. Use qualitative ranking rather than invented numerical precision. A diagnostic probe can be valuable without immediate task gain. Choose an informative change in the smallest mechanism-level scope; a negligible numerical nudge is not automatically a useful test. Keep restart-only candidates and their implementation requirements visible even when today's action is observation.

## Measurement discipline

- Register local mechanism predictions and final objective predictions separately, with baselines, thresholds, exposure/sample requirements and windows that fit the budget. Account for drift, autocorrelation, staggered uptake and adaptation lag; fixed band lengths or multiples of a difference standard deviation are heuristics, not universal significance tests.
- For every rate, state the denominator and missingness/filtering rules. Track attempted, completed, admitted, invalid, timed-out and excluded samples, with reasons; reconcile these counts rather than treating them all as disjoint if they overlap. Separate task failure, infrastructure failure and exclusion so failed samples do not silently disappear.
- Verify metric precision, dynamic range and detector coverage. Recompute baseline and treatment with the same detector, validated on positive/negative examples. When feasible use controls, matched phases, independent permitted evidence and repetitions; document concurrent changes and unresolved effects. A reversal can help but does not erase learned carryover or uniquely prove causality.
- Under independent identically distributed failures at rate 6% and perfect detection, 30 clean samples occur with probability `0.94**30 ≈ 15.63%`; the chance of detecting at least one is about 84.37%. Correlation, sampling bias and incomplete detector coverage change this calculation.
- Gradient clipping bounds the gradient supplied to the optimizer, not generally the final parameter displacement; momentum, preconditioning, learning rate and optimizer state matter. Check applied/skipped updates separately from gradient norms and entropy.
- A process reward need not have positive mean. Examine relative incentives, bypass/termination paths, effective advantage signal and task performance. Repairing misleading metrics cannot repair a genuinely wrong incentive; preserve useful dense signal on its own merits.
- Rapid entropy growth or a fixed-duration response pattern is a candidate empirical fingerprint. Calibrate by phase and subgroup, require persistence and independent corroboration before diagnosing collapse, and distinguish token entropy from behavioral diversity.

## Recovery and reusable experience

First locate the defect's impact: observation only, future interactions/data/rewards, or applied learning updates. Repair measurement without automatically restarting. For changed data/rewards, choose continuing, restoring or reinitializing using contamination, compatibility, the experimental objective and budget, not a universal fresh-run rule.

Distinguish **revert control**, **restore checkpoint**, and **start fresh**. Restoring a control does not undo gradients, optimizer moments, replay contents, curriculum exposure, or past interactions. A reward change at step 300 reverted at 350 remains in the lineage unless compatible model/optimizer/scheduler/data state is also restored. Record what was restored, what remains inherited, and the resulting lineage. Verify checkpoint completeness for the actual trainer format before claiming recoverability.

Reuse the project's experience, change log, inventory, and next-run plan. Label entries as observation, hypothesis, or supported lesson, with supported/refuted/unresolved verdicts where tested, source/version, applicable conditions, counterexamples, and pending window. Read related entries before a new hypothesis; explain changed conditions before repeating a refuted intervention. Do not promote a single observation to a universal principle or import disallowed evaluation feedback into steering memory. Persist review position, pending predictions, application state and actual next wake-up owner for handoff.
