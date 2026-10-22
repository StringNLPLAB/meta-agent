# Minimal τ³ text training and supervised runtime

This example connects a real, locally stored Hugging Face causal language model to an external **τ³-bench 1.0.1** checkout (Python package `tau2`). It performs sampled text/tool interaction, native user simulation, native environment grading, and actual REINFORCE gradient updates. It covers **text, half-duplex, official train tasks**. Voice and knowledge domains are outside this example.

The policy being trained, the user simulator, and the meta-agent are three separate models. A local Flint service can power the meta-agent through OpenCode; it does not become the training policy or the simulated user automatically. No model server, weights, credentials, benchmark data, or distributed training stack is bundled or started implicitly.

Tell the outer coding agent where the benchmark checkout and existing local policy weights live, which user/model endpoints to use, your resource budget, and the review cadence in natural language. The agent inspects the project, prepares the runtime launch configuration, and runs the owned experiment. Users do not need to write JSON or keep a manual polling loop alive. The runtime performs discovery before launching training, periodically invokes OpenCode to inspect newly written logs/trajectories, and records a final review after training exits. See [runtime ownership](../../runtime/README.md).

## Real training boundary

Install the example's requirements into the chosen training interpreter, and install the external **1.0.1** benchmark with its `gym` extra according to its own instructions. The upstream package declares Python 3.12 or 3.13; this integration was tested with Python 3.12. Provide an existing local policy directory containing the tokenizer/model; loading uses `local_files_only=True` and disables remote code. The native user simulator requires its own explicitly selected OpenAI-compatible model service and `openai/` model prefix. An optional credential environment variable is read by the trainer; the key is never added to the launch file or manifest. The example requires a python-dotenv version supporting `PYTHON_DOTENV_DISABLED`, which is set before importing the benchmark.

The adapter checks the package version, source hashes and exact official `train` task IDs. It also fingerprints the actual domain implementation and domain data files, including tasks, database and policy, without exporting their hidden contents. Missing/invalid splits fail instead of falling back to all tasks. It preserves native Gym orchestration and the native `UserSimulator`, while selecting **ENV** (default) or **ACTION** as an explicit training reward proxy. These rewards are not a full official benchmark score. There is no NL judge in the selected proxy. Native service/evaluator failures fail training instead of silently becoming zero-reward episodes. Genuine task failure or the native step limit can still produce zero reward. `--seed` controls the PyTorch policy RNG; native user/service behavior and hardware execution are not guaranteed deterministic.

Only visible policy, tool schemas and conversation history reach the policy. Hidden evaluation actions do not construct its candidate actions, and the supervisor contract excludes hidden criteria, private user prompts and held-out task outcomes. These are source/permission boundaries, not an operating-system sandbox: use separate host/container mounts if benchmark task files must be inaccessible to the meta-agent process.

The learning rule is deliberately small: zero-baseline REINFORCE, summing log probabilities of the generated action tokens and averaging episode losses per optimizer step. It has no PPO clipping, reference KL, distributed rollout, automatic mixed precision, evaluation selection, or automatic resume. It does not reproduce InfoPO/InfoGRPO. Model sampling uses the raw categorical distribution at temperature 1.0, with dropout disabled. All-zero task rewards can produce zero gradients; this example does not imply that a tiny model will solve realistic customer-service tasks. Full model training, per-step full-parameter hashing (including GPU-to-CPU copies), and full checkpoints are intended for small demonstration models; select a bounded model/batch that fits your hardware and increase `--save-every` for larger checkpoints.

For implementers, the outer agent can generate a launch file using:

```bash
python examples/tau3_minimal/prepare.py \
  --root /absolute/path/to/training-checkout --run tau_text_demo \
  --output /absolute/path/to/launch.json \
  --trainer-python /absolute/path/to/python3.12 \
  --benchmark-root /absolute/path/to/tau2-bench \
  --policy-model /absolute/path/to/local-policy \
  --user-model openai/user-model --user-base-url http://127.0.0.1:31000/v1 \
  --meta-model flint --meta-base-url http://127.0.0.1:30000/v1 \
  --device cuda --steps 6 --episodes-per-step 2 \
  --every-steps 2 --max-calls 16
```

These are illustrative paths and service IDs, not claims that any server is running. Place this example inside the training checkout so OpenCode can inspect its code. `prepare.py` creates only the launch file. Pass it to `python runtime/run_experiment.py --config /absolute/path/to/launch.json` from this repository. The generated supervision mode is read-only proposal mode; applying changes needs a separately authorized project gate and contract.

## Explicit synthetic smoke

`--smoke` uses a tiny two-action CPU PyTorch policy, a synthetic observation-dependent reward, and real `loss.backward()` / `optimizer.step()`. It needs no benchmark, tokenizer, policy weights, or user service. It tests the training/log/trajectory/checkpoint pipeline and can be supervised by a real OpenCode CLI using a separately provided local inference endpoint. Every manifest, trajectory and log identifies it as `synthetic_optimizer_smoke`; **it is not a τ³ training result**.

```bash
python examples/tau3_minimal/train.py \
  --smoke --run smoke --output-dir /tmp/new-smoke-run --steps 3
```

To prepare the complete runtime smoke, use `prepare.py` with `--smoke`, explicit `--meta-model` / `--meta-base-url`, and optionally `--step-delay-seconds 1` so a fast CPU training loop leaves time for periodic evidence reads. The delay defaults to zero and is not needed for real training. The runtime owns the trainer process and has bounded trainer/final timeouts; supervisor failure uses `stop_owned_trainer`.

## Artifacts and verification

Default runtime artifacts are:

- `reports/runtime/RUN/trainer.stdout`: flushed `step:N - {...}` progress and metrics.
- `reports/tau3_minimal/RUN/manifest.json`: actual optimizer step, kind, objective, benchmark/source identity, parameter hashes and checkpoint lineage.
- `reports/tau3_minimal/RUN/trajectories/step_XXXXXX_episode_XXX.json`: full visible policy context, observations/actions/messages, reward, behavior hash and target optimizer-step identity. The `rollout_before_update` stage means the associated update is still pending; a flushed completed `step:N` log and the manifest establish completion. Synthetic records identify their synthetic environment.
- `reports/tau3_minimal/RUN/checkpoints/step_XXXXXX.pt`: actual model/optimizer state and parent checkpoint. Loading arbitrary checkpoint files is outside the read-only meta-agent workflow.
- Existing supervisor records/ATIF exports: coding-agent requests, decisions and visible tool activity, separate from training-policy trajectories.

Training output must be a new directory. Checkpoints default to every step for this small example; change `--save-every` for larger models. A new run has fresh identity; automatic resume is intentionally absent. Checkpoints contain RNG state and lineage for inspection but do not constitute a restart interface.

Run `python -m unittest discover -s tests -p test_tau3_minimal.py -v` for an actual CPU optimizer smoke plus dependency-free native API boundary fixtures. These checks cover train-split selection, proxy reward dispatch, failure propagation, public-field export and generated runtime contracts. They do **not** establish successful real benchmark/model-server training. A real deployment still requires the benchmark's declared dependencies, existing local model weights and a reachable user service; report that validation separately.

`test_tau3_policy.py` checks the actual HF sampling/log-probability and gradient path using a tiny model and tokenizer created locally with random weights. With the external benchmark's dependencies installed, `TAU3_BENCHMARK_ROOT=/absolute/path/to/tau2-bench python -m unittest discover -s tests -p test_tau3_native.py -v` exercises the real native training-task API, user simulator, orchestration, proxy evaluator, and localhost HTTP user-service fixture. These tests validate interfaces and updates, not learned benchmark performance. The optional `test_runtime_opencode.py` uses an actual OpenCode 1.19.3 CLI with a localhost mock coding-model API to verify periodic reads of training-generated smoke trajectories and final ATIF export.

The integration was newly written against the source files identified in [provenance.json](provenance.json). The benchmark remains an external dependency with its own [MIT license](https://github.com/sierra-research/tau2-bench/blob/main/LICENSE); no benchmark implementation, task data or model weights are copied into this example.
