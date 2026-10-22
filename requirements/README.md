The root package installs the vendored InfoPO training engine as `verl`. Use a
fresh Python 3.12 environment for GPU training; this keeps its older CUDA stack
separate from the minimal example and from the coding agent's environment.

For the complete τ³ recipe, install the external benchmark and bridge in the
same dependency resolution:

```bash
bash requirements/install_training.sh \
  --python /absolute/path/to/venv/bin/python \
  --benchmark-root /absolute/path/to/tau2-bench
```

The benchmark must be the `tau2` 1.0.1 source checkout with its official `data/`.
The script validates that identity and installs its `[gym]` extra together with
the local bridge and trainer. Omitting `--benchmark-root` installs only the
trainer profile. The script never starts a server or downloads model weights.

Run installation commands from the repository root:

```bash
python -m pip install 'setuptools>=77,<81' wheel packaging ninja
python -m pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1
python -m pip install --no-build-isolation -c requirements/constraints-sglang.txt -e '.[sglang]'
```

After the first two preparation commands, `python -m pip install
--no-build-isolation -r requirements/verl-sglang.txt` selects the same profile.
Use the PyTorch wheel index matching the target CUDA installation when needed.
For CPU configuration inspection without a rollout engine, use
`python -m pip install -e .`.

SGLang 0.4.7's official package metadata requires PyTorch 2.7.1, Transformers
4.52.3, SGL kernel 0.1.7 and FlashInfer 0.2.6.post1. The handover's original
requirements also installed vLLM 0.9.1 (PyTorch 2.7.0) and its setup script pinned
PyTorch 2.6.0; these cannot be resolved together. This profile selects SGLang.
The separate `vllm` extra is for an alternative environment, and is not the
τ³ recipe's rollout backend. Installing both extras is unsupported.

The profile pins `compressed-tensors==0.9.2`, matching the API generation used
by SGLang 0.4.7's adapter. Newer releases import Transformers modules absent
from the required Transformers 4.52.3, despite permitting it in their metadata.

The τ³ benchmark's LiteLLM 1.80.15–1.82.6 range requires the OpenAI Python 2.x
client. The profile permits `openai>=2.8,<3`; SGLang's 0.4.7 metadata places no
conflicting upper bound on that client. Do not add an OpenAI 1.x pin when
installing the benchmark into this environment.

The original FSDP engine uses FlashAttention 2, including imports in its actor
and critic implementations. The SGLang profile therefore includes
`flash-attn==2.7.4.post1`; it needs a CUDA toolkit/compiler and the target
PyTorch installed before building. `--no-build-isolation` lets its build find
that PyTorch installation. Liger remains optional through the `gpu` extra.
Model weights and benchmark data are not
downloaded by package installation. See the τ³ training example for its local
model, benchmark and user-simulator requirements.

`constraints-sglang.txt` records direct compatibility pins. The accompanying
`tau3-sglang-py312.lock` records the 209-package joint metadata resolution of
the trainer, SGLang and native τ³ `[gym]` dependencies on Python 3.12/Linux
x86_64; the installer also applies this lock on that platform. The complete
profile requires Python 3.12: the native benchmark imports the `audioop`
standard-library module, which Python 3.13 removed. This verifies dependency resolution;
full GPU training still requires NVIDIA drivers, a CUDA toolkit, compatible
wheels, local weights and a reachable user simulator on the target host.
