"""Serve a local CoSER user or Qwen3 policy with the pinned SGLang profile."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def command(args, template):
    name = "coser-usersim" if args.role == "user" else "tau2-policy"
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", str(Path(args.model).resolve()),
            "--served-model-name", name, "--host", "127.0.0.1", "--port", str(args.port),
            "--tp", str(args.tp), "--dtype", "bfloat16", "--trust-remote-code",
            "--mem-fraction-static", "0.70", "--context-length", "131072" if args.role == "user" else "40960",
            "--max-running-requests", "8", "--tool-call-parser", "llama3" if args.role == "user" else "qwen25",
            "--chat-template", str(template), "--log-level", "warning"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("user", "policy"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--thinking", action="store_true", help="Policy-only control; default is non-thinking")
    parser.add_argument("--receipt", required=True, help="New local service provenance JSON")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path(args.receipt).exists():
        raise ValueError("Use a fresh service receipt path")
    model = Path(args.model).resolve(strict=True)
    if args.role == "user":
        config = json.loads((model / "config.json").read_text())
        if config.get("model_type") != "llama" or config.get("hidden_size") != 4096:
            raise ValueError("Expected the genuine CoSER Llama-3.1-8B checkpoint")
        if hashlib.sha256((model / "config.json").read_bytes()).hexdigest() != "a6f261303b790f7e300534ad56e0b4bc4d8228af052e30288ea09663704fc9c2":
            raise ValueError("CoSER config identity differs")
        template = ROOT / "meta_agent/user_runtime/coser_llama31_tools.jinja"
    else:
        tokenizer = json.loads((model / "tokenizer_config.json").read_text())
        source = tokenizer.get("chat_template")
        if not isinstance(source, str) or "enable_thinking" not in source:
            raise ValueError("Expected a Qwen3 chat template with an explicit thinking control")
        # The server owns this flag, so every request has the declared behavior.
        template = Path(tempfile.mkdtemp(prefix="tau2-policy-template-")) / "policy.jinja"
        template.write_text("{% set enable_thinking = " + ("true" if args.thinking else "false") + " %}\n" + source)
    argv = command(args, template)
    receipt = dict(role=args.role, model=str(model), served_model_name="coser-usersim" if args.role == "user" else "tau2-policy",
                   thinking=args.thinking if args.role == "policy" else None,
                   template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                   config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
                   argv=argv, dry_run=args.dry_run)
    path = Path(args.receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(receipt))
        return
    if importlib.metadata.version("sglang") != "0.4.7":
        raise ValueError("Use SGLang 0.4.7 from the documented training profile")
    env = dict(os.environ)
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(name, None)
    os.execve(sys.executable, argv, env)


if __name__ == "__main__":
    main()
