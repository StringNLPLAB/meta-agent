#!/usr/bin/env bash
# Install the NVIDIA SGLang/InfoPO stack in the caller's selected environment.
set -euo pipefail

training_python="${PYTHON:-python3}"
training_benchmark_root=""
while (($#)); do
  case "$1" in
    --python)
      if (($# < 2)); then
        echo "--python requires an interpreter path" >&2
        exit 2
      fi
      training_python="$2"
      shift 2
      ;;
    --benchmark-root)
      if (($# < 2)); then
        echo "--benchmark-root requires the external tau2-bench checkout" >&2
        exit 2
      fi
      training_benchmark_root="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: requirements/install_training.sh [--python /path/to/venv/bin/python] [--benchmark-root /path/to/tau2-bench]"
      echo "Requires Python 3.12, an NVIDIA CUDA toolkit and compatible GPU drivers."
      echo "Installs dependencies only; does not download weights or launch training."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

training_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$training_python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 in a dedicated environment for this training profile"'
training_specs=(-e "$training_root[sglang]")
training_constraints=(-c "$training_root/requirements/constraints-sglang.txt")
if "$training_python" -c 'import platform,sys; sys.exit(not (sys.version_info[:2] == (3,12) and sys.platform == "linux" and platform.machine() == "x86_64"))'; then
  training_constraints+=(-c "$training_root/requirements/tau3-sglang-py312.lock")
fi
if [[ -n "$training_benchmark_root" ]]; then
  "$training_python" -c 'import pathlib,sys,tomllib; p=pathlib.Path(sys.argv[1]); d=tomllib.loads((p/"pyproject.toml").read_text()); assert d["project"]["name"] == "tau2" and d["project"]["version"] == "1.0.1", "Expected the tau2 1.0.1 benchmark checkout"; assert (p/"data").is_dir(), "Benchmark data/ is required"' "$training_benchmark_root"
  training_specs+=(-e "$training_benchmark_root[gym]" -e "$training_root/gyms/Tau2Gym")
fi

# FlashAttention's build must see the selected Torch installation.
"$training_python" -m pip install 'setuptools>=77,<81' wheel packaging ninja
"$training_python" -m pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1
"$training_python" -m pip install --no-build-isolation \
  "${training_constraints[@]}" \
  "${training_specs[@]}"

"$training_python" -m pip check
"$training_python" -c 'import torch, sglang, flash_attn; print("torch", torch.__version__, "sglang", sglang.__version__, "flash_attn", flash_attn.__version__); print("CUDA available:", torch.cuda.is_available())'
