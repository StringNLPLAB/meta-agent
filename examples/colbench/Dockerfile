FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git build-essential ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/colbench
ENV PATH=/opt/colbench/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=4
RUN python -m pip install --upgrade pip && \
    python -m pip install 'setuptools>=77,<81' wheel packaging ninja 'numpy<2' && \
    python -m pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1
WORKDIR /code
COPY pyproject.toml NOTICE LICENSE.txt README.md ./
COPY requirements ./requirements
COPY verl ./verl
COPY meta_agent ./meta_agent
COPY gyms/ColBenchGym ./gyms/ColBenchGym
COPY UserRL ./UserRL
COPY examples/__init__.py ./examples/__init__.py
COPY examples/colbench ./examples/colbench
COPY examples/data_preprocess ./examples/data_preprocess
# Shared model identity and fresh-agent utilities contain no held-out tasks.
COPY examples/tau2/__init__.py examples/tau2/models.py examples/tau2/model_identity.json examples/tau2/agent.py ./examples/tau2/
RUN python -m pip install --no-build-isolation -r requirements/verl-sglang.txt && \
    python -m pip install -r requirements/colbench-cpu.txt -e gyms/ColBenchGym
CMD ["python", "examples/colbench/train.py", "--help"]
