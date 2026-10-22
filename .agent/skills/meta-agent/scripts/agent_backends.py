"""Agent invocation adapters. OpenCode runs against an explicitly selected local API.

Each review has a fresh OpenCode session and private configuration. Its tool
permissions permit observation only; the supervisor remains the sole gate owner.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit


OPENCODE_VERSION = "1.19.3"


def command_array(value):
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed or not all(isinstance(x, str) and x for x in parsed):
        raise ValueError("Backend command must be a nonempty JSON array of literal arguments")
    return parsed


@dataclass
class Backend:
    name: str
    settings: dict

    def command(self, directory):
        raise NotImplementedError


class CodexBackend(Backend):
    def __init__(self, model=None):
        super().__init__("codex", {"model": model} if model else {})

    def command(self, directory):
        return ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check",
                "--output-schema", str(directory / "response.schema.json"),
                "--output-last-message", str(directory / "response.json"), "--json",
                *(["--model", self.settings["model"]] if self.settings.get("model") else []), "-"]


class CustomBackend(Backend):
    def __init__(self, command):
        super().__init__("custom", {"command": command})

    def command(self, directory):
        return self.settings["command"]


class OpenCodeBackend(Backend):
    def __init__(self, args):
        endpoint = urlsplit(args.opencode_base_url or "")
        if (endpoint.scheme not in ("http", "https") or not endpoint.hostname or endpoint.username is not None
                or endpoint.password is not None or endpoint.query or endpoint.fragment):
            raise ValueError("OpenCode needs --opencode-base-url as an http(s) API URL without credentials/query/fragment")
        if not args.agent_model or not args.agent_model.strip():
            raise ValueError("OpenCode needs --agent-model with the model ID served by the local endpoint")
        key = args.opencode_api_key_env
        if key and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("--opencode-api-key-env must be an environment variable name, not a key")
        if key and (key.startswith(("XDG_", "META_AGENT_")) or
                    key.startswith("OPENCODE_") and key != "OPENCODE_API_KEY"):
            raise ValueError("Use a credential variable such as FLINT_API_KEY, separate from runtime configuration")
        if args.opencode_output <= 0 or args.opencode_context <= args.opencode_output:
            raise ValueError("OpenCode context must exceed its positive output limit")
        super().__init__("opencode", {
            "command": command_array(args.opencode_command), "model": args.agent_model,
            "base_url": args.opencode_base_url.rstrip("/"), "api_key_env": key,
            "context": args.opencode_context, "output": args.opencode_output,
            "protocol_version": OPENCODE_VERSION,
        })

    def command(self, directory):
        return [sys.executable, str(Path(__file__).resolve()), "--opencode-request", str(directory),
                "--settings", json.dumps(self.settings)]


def backend_from_args(args):
    local_options = (args.opencode_base_url or args.opencode_api_key_env or args.opencode_command != '["opencode"]'
                     or args.opencode_context != 32768 or args.opencode_output != 8192)
    if args.agent_command:
        if args.backend != "codex" or args.agent_model or local_options:
            raise ValueError("--agent-command is a separate custom backend; do not combine it with backend/model options")
        return CustomBackend(command_array(args.agent_command))
    if args.backend == "opencode":
        return OpenCodeBackend(args)
    if local_options:
        raise ValueError("OpenCode options require --backend opencode")
    return CodexBackend(args.agent_model)


def opencode_config(settings, training_root, skill_root, allowed_paths=()):
    model = "meta-agent-local/" + settings["model"]
    read_paths = {"*": "deny"}
    for root in (training_root, skill_root):
        read_paths[str(root)] = "allow"
        read_paths[str(root) + "/*"] = "allow"
    for path in allowed_paths:
        path = Path(path).resolve()
        # OpenCode's external_directory permission matches a file's parent
        # directory, rather than its exact filename. Existing directories use
        # their own boundary; planned files/directories need a readable parent.
        root = path if path.is_dir() else path.parent
        read_paths[str(root)] = "allow"
        read_paths[str(root) + "/*"] = "allow"
    options = {"baseURL": settings["base_url"]}
    if settings.get("api_key_env"):
        options["apiKey"] = "{env:" + settings["api_key_env"] + "}"
    else:
        options["apiKey"] = "unused"
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": model, "small_model": model, "enabled_providers": ["meta-agent-local"],
        "provider": {"meta-agent-local": {"name": "Meta-agent local endpoint",
            "npm": "@ai-sdk/openai-compatible", "options": options,
            "models": {settings["model"]: {"name": settings["model"], "tool_call": True, "temperature": True,
                "reasoning": True, "interleaved": {"field": "reasoning_content"},
                "limit": {"context": settings["context"], "output": settings["output"]}}}}},
        "default_agent": "meta-agent-review", "share": "disabled", "autoupdate": False,
        "snapshot": False, "plugin": [], "mcp": {}, "instructions": [], "lsp": False,
        # read matches paths relative to the detected worktree, while
        # external_directory checks absolute boundaries before opening a file.
        "permission": {"*": "deny", "read": "allow", "glob": "allow", "grep": "allow",
                       "list": "allow", "external_directory": read_paths},
        "agent": {"meta-agent-review": {"description": "Read-only training experiment review",
            "mode": "primary", "model": model, "steps": 20,
            **({"temperature": 1.0, "top_p": 0.95} if settings["model"].split("/")[-1].lower() == "flint" else {}),
            "prompt": "Complete the supplied review using permitted read-only tools. Return only its JSON response."}},
    }


def opencode_environment(directory, config):
    # Prevent inherited project/global integrations from changing this pass. Do
    # not repurpose HOME; OpenCode exposes a separate test-home path override.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("OPENCODE_") and key not in ("META_AGENT_RESPONSE", "META_AGENT_REQUEST")}
    private = directory / "opencode-private"
    for key, relative in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                          ("XDG_CACHE_HOME", "cache"), ("XDG_STATE_HOME", "state"),
                          ("OPENCODE_TEST_HOME", "home")):
        path = private / relative
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    environment.update({key: "1" for key in (
        "OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_CLAUDE_CODE", "OPENCODE_DISABLE_DEFAULT_PLUGINS",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS", "OPENCODE_DISABLE_LSP_DOWNLOAD", "OPENCODE_DISABLE_MODELS_FETCH",
        "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_DISABLE_AUTOCOMPACT", "OPENCODE_PURE")})
    environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    return environment


def parse_opencode_response(stdout):
    """Require a completed, single-session CLI trace and one final JSON object."""
    texts, sessions, finishes = {}, set(), []
    usage = {}
    seen_finishes = set()
    for number, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            raise ValueError(f"OpenCode stdout line {number} is not a JSON event") from None
        if not isinstance(event, dict):
            raise ValueError("OpenCode stdout events must be objects")
        if event.get("type") not in ("step_start", "step_finish", "text", "reasoning", "tool_use", "error"):
            raise ValueError("OpenCode returned an unsupported event; inspect the captured trace before continuing")
        session = event.get("sessionID")
        if not isinstance(session, str) or not session:
            raise ValueError("OpenCode events must identify their session")
        sessions.add(session)
        if event.get("type") == "error":
            raise ValueError("OpenCode reported an error; inspect the captured agent.stdout")
        part = event.get("part")
        if not isinstance(part, dict) or part.get("sessionID", session) != session:
            raise ValueError("OpenCode event part is missing or identifies a different session")
        expected_type = {"step_start": "step-start", "step_finish": "step-finish", "tool_use": "tool",
                         "text": "text", "reasoning": "reasoning"}[event["type"]]
        if part.get("type") != expected_type or not all(isinstance(part.get(key), str) and part[key]
                                                       for key in ("id", "messageID")):
            raise ValueError("OpenCode event part must have its expected type and stable part/message IDs")
        if event.get("type") == "text" and isinstance(part.get("text"), str):
            texts[part["id"]] = (part["messageID"], part["text"])
        if event.get("type") == "step_finish":
            identifier = part.get("id")
            if identifier in seen_finishes:
                continue
            if identifier:
                seen_finishes.add(identifier)
            finishes.append((part.get("reason"), part["messageID"]))
            tokens = part.get("tokens") or {}
            if isinstance(tokens, dict):
                measured = {"input": tokens.get("input"), "output": tokens.get("output"),
                            "reasoning": tokens.get("reasoning"),
                            "cache_read": (tokens.get("cache") or {}).get("read") if isinstance(tokens.get("cache"), dict) else None,
                            "cache_write": (tokens.get("cache") or {}).get("write") if isinstance(tokens.get("cache"), dict) else None}
                for key, value in measured.items():
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        usage[key] = usage.get(key, 0) + value
    if len(sessions) != 1:
        raise ValueError("OpenCode must return exactly one session per review")
    if not finishes or finishes[-1][0] != "stop":
        raise ValueError("OpenCode did not finish normally (missing stop, token limit or unfinished tool turn)")
    if not texts:
        raise ValueError("OpenCode returned no final text response")
    final_message = finishes[-1][1]
    text = "".join(text for message, text in texts.values() if message == final_message).strip()
    if not text:
        raise ValueError("OpenCode's final completed message has no JSON response; earlier proposals are not final")
    # Some models use one conventional fence despite the JSON-only instruction.
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4].strip()
    try:
        response = json.loads(text)
    except ValueError:
        raise ValueError("OpenCode's final text must be one JSON object matching response.schema.json") from None
    if not isinstance(response, dict):
        raise ValueError("OpenCode's final response must be a JSON object")
    return response, next(iter(sessions)), usage


def run_opencode(directory, settings):
    directory = directory.resolve()
    request = json.loads((directory / "request.json").read_text())
    schema = json.loads((directory / "response.schema.json").read_text())
    prompt = sys.stdin.read()
    # The roots are controller-created request metadata, never model output.
    config = opencode_config(settings, Path(request["training_root"]), Path(request["skill_root"]),
                             request.get("allowed_read_paths", ()))
    environment = opencode_environment(directory, config)
    if settings.get("api_key_env") and settings["api_key_env"] not in environment:
        # OPENCODE_API_KEY is a legitimate credential name despite the prefix.
        # Never overwrite one of the adapter's isolation controls.
        name = settings["api_key_env"]
        if name not in environment and name in os.environ:
            environment[name] = os.environ[name]
    if settings.get("api_key_env") and not environment.get(settings["api_key_env"]):
        raise ValueError("The configured local model API-key environment variable is unset or empty")
    (directory / "opencode.config.json").write_text(json.dumps(config, indent=2) + "\n")
    workspace = directory / "opencode-private" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    # OpenCode run prefers PWD to process.cwd() unless --dir is explicit.
    environment["PWD"] = str(workspace)
    metadata = {"backend": "opencode", "model": settings["model"], "protocol_version": OPENCODE_VERSION,
                "request_id": request["id"], "session_policy": "fresh_per_request", "status": "prepared"}
    metadata_path = directory / "agent.backend.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    version = subprocess.run([*settings["command"], "--version"], input="", text=True, capture_output=True,
                             env=environment, cwd=workspace, timeout=20)
    reported = version.stdout.strip()
    if version.returncode or reported != OPENCODE_VERSION:
        raise ValueError(f"OpenCode {OPENCODE_VERSION} is required by this adapter; CLI reported {reported[:80]!r}")
    metadata.update(version=reported, status="running")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    instructions = ("\nReturn exactly one JSON object, without prose or Markdown fences, as the final text. "
                    "The JSON must match this response schema. Tool use remains read-only.\n<response_schema>\n"
                    + json.dumps(schema) + "\n</response_schema>\n")
    command = [*settings["command"], "run", "--format", "json", "--dir", str(workspace), "--model", config["model"],
               "--agent", "meta-agent-review", "--title", request["id"], "--thinking"]
    # The outer supervisor owns this whole process group and its timeout. Pipe
    # through CLI events immediately so interruptions still leave a useful trace.
    with (directory / "opencode.stdout").open("wb") as captured:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   env=environment, cwd=workspace)
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write((prompt + instructions).encode())
        process.stdin.close()
        for chunk in iter(lambda: process.stdout.read1(65536), b""):
            captured.write(chunk)
            captured.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        code = process.wait()
    if code:
        raise ValueError(f"OpenCode exited {code}; raw events and stderr are retained")
    response, session, usage = parse_opencode_response((directory / "opencode.stdout").read_text())
    # Full schema/phase/tag checks run in the supervisor before any gate dispatch.
    metadata.update(status="completed", session_id=session, usage=usage)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (directory / "response.json").write_text(json.dumps(response, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencode-request", required=True, type=Path)
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()
    try:
        run_opencode(args.opencode_request, json.loads(args.settings))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"[OPENCODE] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
