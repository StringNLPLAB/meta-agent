"""Fresh-process adapter for a natural-language coding-agent CLI.

Example backend argv file: ["pi", "--print", "{prompt}"]. No resume/session
identifier is passed. This launcher does not create an OS filesystem sandbox.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--backend-command", required=True, help="JSON argv with exactly one {prompt} argument")
    parser.add_argument("--pi-models", help="Optional explicit models.json; copy provider configuration only, never conversation/session state")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    context = json.loads(Path(request["context_path"]).read_text())
    roots = request["editable_roots"]
    run = Path(roots[0]).parent
    backend = json.loads(Path(args.backend_command).read_text())
    if not isinstance(backend, list) or not backend or not all(isinstance(x, str) for x in backend):
        raise ValueError("Backend command must be an argv array")
    if backend.count("{prompt}") != 1 or any(x in {"--resume", "--continue", "--session", "-c", "-r"} for x in backend):
        raise ValueError("Use one literal {prompt} argv item and a fresh-session backend command")
    prompt = json.dumps(dict(instructions=context["instructions"], step=request["step"],
                            drafts=roots, training_evidence=context["artifacts"]), ensure_ascii=False)
    with tempfile.TemporaryDirectory(prefix="tau2-agent-session-") as session:
        if args.pi_models:
            source = Path(args.pi_models)
            if not isinstance(json.loads(source.read_text()), dict):
                raise ValueError("Pi model/provider configuration must be a JSON object")
            shutil.copyfile(source, Path(session) / "models.json")
        env = dict(os.environ, PI_CODING_AGENT_DIR=session, OPENCODE_CONFIG_DIR=session,
                   XDG_STATE_HOME=session, XDG_DATA_HOME=session)
        if env.get("META_AGENT_API_KEY"):
            env["OPENAI_API_KEY"] = env["META_AGENT_API_KEY"]
        if env.get("META_AGENT_BASE_URL"):
            env["OPENAI_BASE_URL"] = env["META_AGENT_BASE_URL"]
            env["OPENAI_API_BASE"] = env["META_AGENT_BASE_URL"]
        # API/provider settings should be supplied explicitly through the CLI's
        # supported environment or config arguments, not a prior session folder.
        result = subprocess.run([prompt if x == "{prompt}" else x for x in backend], cwd=run, env=env)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
