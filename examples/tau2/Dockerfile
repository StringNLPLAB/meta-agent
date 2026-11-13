FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git build-essential ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/tau2
ENV PATH=/opt/tau2/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=4
RUN python -m pip install --upgrade pip && \
    python -m pip install 'setuptools>=77,<81' wheel packaging ninja 'numpy<2' && \
    python -m pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1
WORKDIR /code
COPY pyproject.toml NOTICE LICENSE.txt README.md ./
COPY requirements ./requirements
COPY verl ./verl
COPY meta_agent ./meta_agent
COPY gyms ./gyms
COPY UserRL ./UserRL
COPY examples/tau2 ./examples/tau2
COPY eval ./eval
RUN python -m pip install --no-build-isolation -r requirements/tau2.txt
CMD ["python", "examples/tau2/train.py", "--help"]
