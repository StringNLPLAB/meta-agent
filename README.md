# Meta-Agent Skill

A general skill for setting up, launching, supervising, and improving long-running model training experiments. It combines the project's training recipe, prior experiment experience, current evidence, and applicable method references to discover mechanism hypotheses, maps them to actual code and runtime controls, then tests worthwhile changes and verifies uptake and outcomes.

The workflow applies across pretraining, self-supervised learning, supervised learning and SFT, reinforcement learning, and agent training. Candidate components depend on the project: objectives and loss functions, data sampling and preprocessing, optimizer schedules, batching, resource use and throughput, and training measurements. Reward, rollout, environment, and user-simulator changes are additional candidates for projects that have those components. Each scheduled review can apply a justified modification, collect evidence, observe a pending window, or plan a restart-required intervention. The experiment contract defines feedback boundaries; EAU reproduction uses training-side feedback only.

Load [.agent/skills/meta-agent/SKILL.md](.agent/skills/meta-agent/SKILL.md) in Codex, or copy `.agent/skills/meta-agent/` into its skill directory. Then ask Codex to run the experiment and provide the training code location. Describe objectives, resources, budgets, review timing, and allowed changes in natural language; Codex translates them into the required configuration and commands.

## Setup

Codex reads the bundled [EAU reference paper](.agent/skills/meta-agent/references/eau_manuscript_ref.pdf) once during setup and saves its version, key mechanisms, applicability conditions, and limitations in the existing experience record or intervention map. This setup is automatic; you do not need to run a separate command.

EAU is a method reference for generating and testing improvement hypotheses, not a required training recipe. Its mechanisms are candidates only where the current project supports them; an experiment does not need RL, rewards, an interactive environment, or a user simulator.

Subsequent experiments and periodic reviews reuse that cached summary rather than rereading the paper. Setup is repeated only when the paper version changes or the saved summary is missing. Each experiment still needs its own component discovery against the current training recipe, code, resources, and evidence.

## Start an experiment

Give Codex a request like this, replacing the two paths:

```text
Read /absolute/path/to/meta-agent/.agent/skills/meta-agent/SKILL.md and use it
to run and improve the experiment in /absolute/path/to/training-checkout.
Handle it end to end, including initial component discovery, training launch,
periodic reviews and modifications, and Harbor trajectory recording. Reuse
the existing recipe and settings where they are available.
```

You can add constraints in the same request, for example:

```text
Use GPUs 0–3 for at most four hours. Review every 20 optimizer steps or ten
minutes, whichever comes first. You may add training-side instrumentation
and use verified runtime controls, but preserve the evaluation protocol and
do not use independent test feedback to steer the experiment.
At each review, apply changes justified by the observed training evidence
within this scope, then verify their uptake and effects.
```

Codex handles the setup and execution:

1. Reuse the setup summary and applicable experience; inspect the training recipe, code, and available resources; resolve the experiment contract, run identity, paths, budgets, and review cadence. Complete setup first if its summary is missing or the paper version changed.
2. Run read-only discovery before training and save the candidate component map.
3. Launch the existing training recipe, then arrange continuing supervision through an actual host-managed process or scheduler with ATIF export enabled.
4. At each configured time or step boundary, review logs and available training evidence, such as sampled examples, predictions, per-batch diagnostics, or interaction trajectories; reassess candidate modifications, register predictions, apply justified changes within the authorized scope through a verified project gate, and verify runtime uptake and subsequent outcomes. Track training progress, the supervisor owner and next wake-up throughout.
5. Save completion or handoff records, remaining uncertainties, and the selected checkpoint under the declared protocol.

Periodic modifications can target training measurements, objectives and losses, data sampling and preprocessing, optimizer schedules, batching, or resource and throughput controls, where the experiment authorizes changes. For RL or interactive agent projects, reward logic, rollout collection, environment behavior, and user simulators may also be relevant. Codex chooses targets from the current intervention map, applicable experience and method summaries, and training evidence. It records the intended mechanism, prediction, observation window, and recovery plan before applying a change. A review can also keep settings unchanged while an observation window is pending or the evidence does not support another modification.

You do not need to write JSON, supply CLI flags, start training manually, or keep a terminal open. Codex reuses suitable project settings and asks only when a necessary objective, budget, permission, data source, or credential is missing. Reading the skill provides instructions; the request to run an experiment authorizes execution within its stated scope.

The project gate is a programmatic validation and execution interface, not a prompt for manual confirmation at every review. Codex derives its allowed operations from your existing authorization and can implement a compatible gate within that scope. It verifies which changes can reach running workers; code changes that require a restart follow the experiment's authorized restart plan.

The host needs Python 3, an authenticated Codex CLI or compatible agent backend, the training environment, and a supported way to own continuing jobs. Codex checks each project's launch interface, progress logs, diagnostic sources, and available runtime controls; the skill does not supply a tested integration for every training framework. It must establish and verify the job owner before claiming unattended supervision. The bundled runner remains a foreground review/gate process; the outer Codex agent launches training and arranges its lifetime. Skill text alone cannot keep a process alive after its host session ends. See the [implementation reference](.agent/skills/meta-agent/references/supervision.md) for the CLI, contract, execution gate, and recovery details used by Codex.

## Export meta-agent trajectories

When trajectory recording is requested, Codex enables `--export-atif`. Each pass produces `requests/<ID>/trajectory.json` and an `export.json` coverage manifest, with a run-level `trajectories.json` index under the training checkout's `reports/supervision/<RUN>/`. These record the **meta-agent's** messages, tool calls, and decisions; training diagnostic samples or interaction trajectories remain separate review inputs. The exporter uses the matching Codex session when available and labels incomplete coverage.

You can also ask: “Export the existing meta-agent requests and prepare a Harbor viewer bundle for this run.” Codex can do this without new agent calls; Harbor is needed only to open its viewer or perform Harbor-specific validation. ATIF generation uses Python's standard library. See [export details and commands](.agent/skills/meta-agent/references/supervision.md#export-meta-agent-trajectories) for coverage limits and Harbor integration.

The [intervention discovery guide](.agent/skills/meta-agent/references/intervention-discovery.md) explains candidate selection. The optional [EAU / verl adapter](.agent/skills/meta-agent/references/eau-adapter.md) documents project-specific helpers and layouts; its watchers check records without invoking the reasoning agent.

## Validation

Run offline helper regression checks with `python3 -m unittest discover -s .agent/skills/meta-agent/scripts/tests -v`. Fixtures use temporary training roots and do not connect to training processes.

Originally developed for **EAU: Environment-Agent-User Co-evolution for Training User-Centric Agents**. This branch contains the skill and supervision helpers; training frameworks, weights, and run outputs remain outside the package.

## Citation

If you use this skill in your research, please cite:

```bibtex
@misc{cai2026eauenvironmentagentusercoevolutiontraining,
  title = {EAU: Environment-Agent-User Co-evolution for Training User-Centric Agents},
  author = {Min Cai and Dan Zhang and Gao Xing and Libin Ma and Zhaohui Xu and Xi Ye},
  year = {2026},
  eprint = {2610.16523},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2610.16523}
}
```

## License

Original project contributions are licensed under the [MIT License](LICENSE.txt). Existing third-party notices and licensing terms are retained in [NOTICE](NOTICE) and [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). The bundled reference paper retains its own copyright and distribution terms.
