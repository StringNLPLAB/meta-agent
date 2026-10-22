"""Optional real CLI regression: META_AGENT_OPENCODE_BIN=/path/to/opencode-1.19.3.

Only a local HTTP protocol fixture is used; no model inference or GPU is needed.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
REASONING = "Inspect current training evidence before deciding."


@unittest.skipUnless(os.environ.get("META_AGENT_OPENCODE_BIN"),
                     "set META_AGENT_OPENCODE_BIN to an OpenCode 1.19.3 binary")
class OpenCodeCLIIntegrationTests(unittest.TestCase):
    def test_real_read_permissions_cadence_and_native_export(self):
        with tempfile.TemporaryDirectory(prefix="opencode-cli-integration-") as name:
            temporary = Path(name)
            binary = Path(os.environ["META_AGENT_OPENCODE_BIN"]).expanduser().resolve()
            if not binary.is_file() or not os.access(binary, os.X_OK):
                self.skipTest("META_AGENT_OPENCODE_BIN is not an executable file")
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            for key in tuple(environment):
                if key.startswith(("OPENCODE_", "EAU_")):
                    environment.pop(key)
            for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
                environment[key] = str(temporary / key)
            environment["OPENCODE_TEST_HOME"] = str(temporary / "home")
            try:
                version = subprocess.run([str(binary), "--version"], env=environment,
                    cwd=temporary, capture_output=True, text=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired) as error:
                self.skipTest(f"cannot inspect optional OpenCode CLI: {error}")
            if version.returncode or version.stdout.strip() != "1.19.3":
                self.skipTest("this protocol regression requires OpenCode 1.19.3")

            root, data, outside = (temporary / part for part in ("training", "data", "outside"))
            for directory in (root, data / "samples", outside):
                directory.mkdir(parents=True)
            trainer = root / "trainer.py"
            trainer.write_text("learning_rate = 0.001\n")
            secret = outside / "secret.txt"
            secret.write_text("outside-scope-must-not-be-read")
            written, shelled, plugin_ran = (temporary / part for part in ("written", "shelled", "plugin-ran"))
            # PWD, project settings and global integrations must not redirect the child.
            environment.update(PWD=str(SCRIPTS.parents[3]), OPENCODE_PERMISSION='{"*":"allow"}')
            plugin = ("import {writeFileSync} from 'node:fs'; export default async () => {"
                      f"writeFileSync({json.dumps(str(plugin_ran))}, 'unexpected'); return {{}}; }};\n")
            for directory in (root / ".opencode", Path(environment["XDG_CONFIG_HOME"]) / "opencode",
                              Path(environment["OPENCODE_TEST_HOME"]) / ".opencode"):
                (directory / "plugins").mkdir(parents=True)
                path = directory / "plugins" / "unexpected.js"
                path.write_text(plugin)
                (directory / "opencode.json").write_text(json.dumps({"plugin": [str(path)]}))
            environment["OPENCODE_CONFIG"] = str(root / ".opencode" / "opencode.json")
            contract = dict(objective="offline local CLI protocol verification", budget="40 steps, 3 calls",
                guardrails=["read-only observations"], allowed_feedback=["training logs and samples"],
                excluded_feedback=["held-out evaluation"], evaluation_protocol="independent evaluator",
                checkpoint_selection="final checkpoint", allowed_operations=[])
            (root / "contract.json").write_text(json.dumps(contract))
            log, sample = data / "train.log", data / "samples" / "step20.json"
            recorded, server_errors = [], []

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    try:
                        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                        recorded.append((self.path, body))
                        users = "\n".join(m["content"] for m in body["messages"] if m["role"] == "user")
                        context = json.loads(re.search(r"<context>\s*(.*?)\s*</context>", users, re.S).group(1))
                        phase = context["request"]["phase"]
                        first = not any(m["role"] == "tool" for m in body["messages"])
                        calls = []
                        if first:
                            paths = [trainer] if phase == "discovery" else [log, sample]
                            specs = [(f"allowed_{i}", "read", {"filePath": str(path)})
                                     for i, path in enumerate(paths)]
                            if phase == "discovery":
                                specs += [("denied_read", "read", {"filePath": str(secret)}),
                                    ("denied_write", "write", {"filePath": str(written), "content": "bad"}),
                                    ("denied_edit", "edit", {"filePath": str(trainer),
                                        "oldString": "0.001", "newString": "9"}),
                                    ("denied_bash", "bash", {"command": f"touch {shelled}",
                                        "description": "Verify unavailable shell tool"})]
                            calls = [{"index": i, "id": identifier, "type": "function",
                                "function": {"name": tool, "arguments": json.dumps(arguments)}}
                                for i, (identifier, tool, arguments) in enumerate(specs)]
                        response = dict(phase=phase, summary="Read current evidence using actual CLI tools",
                            intervention_map="| Component | Decision |\n| optimizer | retain current learning rate |\n",
                            sources=dict(paper=["bundled versioned mechanism guide; cached setup summary"],
                                experience=["unavailable: offline fixture"], code=[str(trainer)],
                                logs=[str(log)], trajectories=[str(sample)] if phase == "review" else []),
                            review=dict(decision="leave_unchanged", tier="training", knob="learning_rate",
                                change="reason: no demonstrated defect; next: next scheduled review",
                                evidence="trajectory: offline fixture read through tools", prediction="—",
                                verification="no application attempted"),
                            action=dict(operation="", arguments_json="{}"), pending_predictions=[],
                            experience_update="", next_condition="next scheduled review", continue_supervision=True)
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()

                        def emit(delta, finish=None):
                            event = {"id": "chatcmpl-local-mock", "object": "chat.completion.chunk",
                                "created": int(time.time()), "model": "flint", "choices":
                                [{"index": 0, "delta": delta, "finish_reason": finish}]}
                            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())

                        emit({"role": "assistant"})
                        emit({"reasoning_content": REASONING})
                        emit({"tool_calls": calls} if first else {"content": json.dumps(response)})
                        emit({}, "tool_calls" if first else "stop")
                        usage = {"id": "chatcmpl-local-mock", "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": "flint", "choices": [], "usage":
                            {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125,
                             "prompt_tokens_details": {"cached_tokens": 10},
                             "completion_tokens_details": {"reasoning_tokens": 5}}}
                        self.wfile.write(("data: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n").encode())
                        self.wfile.flush()
                    except Exception as error:
                        server_errors.append(repr(error))
                        self.send_error(500)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                command = [sys.executable, str(SCRIPTS / "supervise.py"), "--root", str(root), "--run", "LOCAL",
                    "--log", str(log), "--trajectories", str(sample.parent), "--contract", str(root / "contract.json"),
                    "--every-steps", "20", "--backend", "opencode", "--agent-model", "flint",
                    "--opencode-base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--opencode-command", json.dumps([str(binary)]), "--agent-timeout", "40",
                    "--max-calls", "3", "--once", "--export-atif"]
                for phase in ("discovery", "review"):
                    if phase == "review":
                        log.write_text("step:20 - loss:0.5\n")
                        sample.write_text('{"sample":"clean local mock sample","loss":0.5}')
                    result = subprocess.run(command, env=environment, cwd=SCRIPTS.parents[3],
                        capture_output=True, text=True, timeout=55)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(server_errors, server_errors)
                self.assertEqual(len(recorded), 4, "unexpected auxiliary request or repeated model turn")
                for path, body in recorded:
                    self.assertEqual(path, "/v1/chat/completions")
                    self.assertEqual(body["model"], "flint")
                    self.assertEqual({tool["function"]["name"] for tool in body["tools"]}, {"glob", "grep", "read"})
                    self.assertIn("opencode-private/workspace", body["messages"][0]["content"])
                    for message in body["messages"]:
                        if message["role"] == "assistant":
                            self.assertEqual(message["reasoning_content"], REASONING)

                requests = list((root / "reports/supervision/LOCAL/requests").iterdir())
                self.assertEqual(len(requests), 2)
                sessions, phases = set(), set()
                for directory in requests:
                    request = json.loads((directory / "request.json").read_text())
                    phases.add((request["phase"], request["step"]))
                    events = [json.loads(line) for line in (directory / "opencode.stdout").read_text().splitlines()]
                    tools = {event["part"]["callID"]: event["part"] for event in events if event["type"] == "tool_use"}
                    expected = [trainer] if request["phase"] == "discovery" else [log, sample]
                    for index, path in enumerate(expected):
                        state = tools[f"allowed_{index}"]["state"]
                        self.assertEqual(state["status"], "completed", state)
                        self.assertIn(path.read_text().strip(), state["output"])
                    if request["phase"] == "discovery":
                        for identifier in ("denied_read", "denied_write", "denied_edit", "denied_bash"):
                            self.assertEqual(tools[identifier]["state"]["status"], "error")
                        self.assertNotIn(secret.read_text(), tools["denied_read"]["state"]["error"])
                    metadata = json.loads((directory / "agent.backend.json").read_text())
                    sessions.add(metadata["session_id"])
                    trajectory = json.loads((directory / "trajectory.json").read_text())
                    self.assertEqual(trajectory["extra"]["capture"], "opencode_stdout")
                    self.assertTrue(any(step.get("reasoning_content") for step in trajectory["steps"]))
                    self.assertEqual(trajectory["final_metrics"]["total_prompt_tokens"], 200)
                    self.assertEqual(trajectory["final_metrics"]["total_completion_tokens"], 50)
                    self.assertEqual(trajectory["final_metrics"]["total_cached_tokens"], 20)
                self.assertEqual(phases, {("discovery", 0), ("review", 20)})
                self.assertEqual(len(sessions), 2, "each review must have a fresh native session")
                self.assertEqual(trainer.read_text(), "learning_rate = 0.001\n")
                for sentinel in (written, shelled, plugin_ran):
                    self.assertFalse(sentinel.exists(), str(sentinel))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
