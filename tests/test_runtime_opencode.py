"""Optional owned-training integration with real OpenCode and a mock local API.

The trainer performs CPU parameter updates on an explicitly synthetic task.
The HTTP server supplies protocol decisions, not model inference or scores.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest


REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("META_AGENT_OPENCODE_BIN"),
                     "set META_AGENT_OPENCODE_BIN to an OpenCode 1.19.3 binary")
@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU smoke requires PyTorch")
class OwnedOpenCodeIntegrationTests(unittest.TestCase):
    def test_training_generated_trajectories_are_read_at_periodic_reviews(self):
        binary = Path(os.environ["META_AGENT_OPENCODE_BIN"]).expanduser().resolve()
        self.assertTrue(binary.is_file() and os.access(binary, os.X_OK))
        with tempfile.TemporaryDirectory(prefix="meta-owned-opencode-") as name:
            root = Path(name)
            example = root / "examples/tau3_minimal"
            shutil.copytree(REPO / "examples/tau3_minimal", example,
                            ignore=shutil.ignore_patterns("__pycache__"))
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            for key in tuple(environment):
                if key.startswith(("OPENCODE_", "EAU_")):
                    environment.pop(key)
            recorded, errors = [], []

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    try:
                        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                        users = "\n".join(m["content"] for m in body["messages"] if m["role"] == "user")
                        context = json.loads(re.search(r"<context>\s*(.*?)\s*</context>", users, re.S).group(1))
                        phase = context["request"]["phase"]
                        first = not any(m["role"] == "tool" for m in body["messages"])
                        recorded.append({"body": body, "phase": phase, "context": context})
                        paths = ([example / "train.py"] if phase == "discovery" else
                                 [Path(context["training_log"]),
                                  *map(Path, context["recent_trajectory_files"][:1])])
                        calls = [{"index": i, "id": f"evidence_{i}", "type": "function",
                                  "function": {"name": "read", "arguments": json.dumps({"filePath": str(path)})}}
                                 for i, path in enumerate(paths)] if first else []
                        response = dict(phase=phase, summary="Read owned synthetic training evidence",
                            intervention_map="| Component | Decision |\n| optimizer | observe CPU smoke updates |\n",
                            sources=dict(paper=["cached EAU setup mechanism guide"],
                                experience=["unavailable: isolated synthetic integration"],
                                code=[str(example / "train.py")], logs=[context["training_log"]],
                                trajectories=[str(p) for p in paths[1:]] or ["unavailable: pre-training discovery"]),
                            review=dict(decision="leave_unchanged", tier="training", knob="optimizer",
                                change="reason: lifecycle smoke only; next: next scheduled boundary",
                                evidence="trajectory: actual file read, synthetic training only",
                                prediction="—", verification="no intervention applied"),
                            action=dict(operation="", arguments_json="{}"), pending_predictions=[],
                            experience_update="", next_condition="next boundary or trainer completion",
                            continue_supervision=True)
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()

                        def emit(delta, finish=None):
                            event = {"id": "chatcmpl-owned-smoke", "object": "chat.completion.chunk",
                                "created": int(time.time()), "model": "flint", "choices":
                                [{"index": 0, "delta": delta, "finish_reason": finish}]}
                            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())

                        emit({"role": "assistant"})
                        emit({"reasoning_content": "Read the recorded synthetic optimizer evidence."})
                        emit({"tool_calls": calls} if first else {"content": json.dumps(response)})
                        emit({}, "tool_calls" if first else "stop")
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except Exception as error:
                        errors.append(repr(error))
                        self.send_error(500)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                config_path = root / "launch.json"
                prepared = subprocess.run([sys.executable, str(example / "prepare.py"),
                    "--output", str(config_path), "--root", str(root), "--run", "OWNED_SMOKE",
                    "--backend", "opencode", "--meta-base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--meta-model", "flint", "--smoke", "--steps", "8",
                    "--episodes-per-step", "8", "--step-delay-seconds", "1"],
                    env=environment, capture_output=True, text=True, timeout=15)
                self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
                config = json.loads(config_path.read_text())
                flags = config["supervisor_args"]
                # Step boundaries ensure every periodic call has completed training samples.
                for flag in ("--every-seconds", "--every-steps", "--max-calls", "--agent-timeout"):
                    if flag in flags:
                        offset = flags.index(flag)
                        del flags[offset:offset + 2]
                flags.extend(["--every-steps", "2", "--max-calls", "12", "--agent-timeout", "40",
                              "--opencode-command", json.dumps([str(binary)])])
                config.update(trainer_timeout_seconds=60, final_timeout_seconds=60)
                config_path.write_text(json.dumps(config))
                result = subprocess.run([sys.executable, str(REPO / "runtime/run_experiment.py"),
                    "--config", str(config_path)], env=environment, capture_output=True, text=True, timeout=110)
                runtime = root / "reports/runtime/OWNED_SMOKE"
                diagnostics = "\n".join(p.read_text(errors="replace") for p in runtime.glob("*.stderr"))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr + diagnostics)
                self.assertFalse(errors, errors)
                state = json.loads((runtime / "state.json").read_text())
                self.assertEqual(state["status"], "completed")
                self.assertEqual(state["trainer_returncode"], 0)
                self.assertFalse(state["owner"]["running"])
                supervision = root / "reports/supervision/OWNED_SMOKE"
                requests = [json.loads(p.read_text()) for p in supervision.glob("requests/*/request.json")]
                self.assertEqual(sum(r["phase"] == "discovery" for r in requests), 1)
                self.assertEqual(sum(r["phase"] == "final" for r in requests), 1)
                self.assertTrue(any(r["phase"] == "review" and r["step"] >= 2 for r in requests))
                for request in requests:
                    directory = Path(request["directory"])
                    events = [json.loads(line) for line in (directory / "opencode.stdout").read_text().splitlines()]
                    reads = [e["part"]["state"] for e in events if e["type"] == "tool_use"]
                    self.assertTrue(reads)
                    self.assertTrue(all(read["status"] == "completed" for read in reads), reads)
                    if request["phase"] in ("review", "final"):
                        self.assertGreaterEqual(len(reads), 2)
                        self.assertIn("synthetic", reads[1]["output"])
                    trajectory = json.loads((directory / "trajectory.json").read_text())
                    self.assertEqual(trajectory["extra"]["capture"], "opencode_stdout")
                    self.assertTrue(any(step.get("tool_calls") for step in trajectory["steps"]))
                self.assertTrue((supervision / "trajectories.json").exists())
                self.assertTrue(all(item["body"]["model"] == "flint" for item in recorded))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
