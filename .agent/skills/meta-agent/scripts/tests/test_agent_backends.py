"""Offline backend and controller tests; no real model or OpenCode installation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from agent_backends import backend_from_args, opencode_config, opencode_environment, parse_opencode_response
from supervise import parser, pid_start
import test_supervise as fixtures


def events(response, *, reason="stop", tokens=None):
    common = {"timestamp": 1700000000000, "sessionID": "ses_one"}
    return [
        {**common, "type": "text", "part": {"id": "text_one", "type": "text", "sessionID": "ses_one",
            "messageID": "msg_final", "text": json.dumps(response)}},
        {**common, "type": "step_finish", "part": {"id": "finish_one", "type": "step-finish",
            "messageID": "msg_final", "sessionID": "ses_one",
            "reason": reason, **({"tokens": tokens} if tokens is not None else {})}},
    ]


def encoded(records):
    return "\n".join(json.dumps(event) for event in records)


class BackendContractTests(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(["--root", "/tmp", "--run", "RUN", "--log", "train.log",
            "--trajectories", "samples", "--contract", "contract.json", "--every-steps", "20", "--once", *extra])

    def local(self, *extra):
        return backend_from_args(self.args("--backend", "opencode", "--agent-model", "flint",
            "--opencode-base-url", "http://127.0.0.1:30000/v1", *extra))

    def test_explicit_backend_selection_and_invalid_routes(self):
        default = backend_from_args(self.args())
        self.assertEqual(default.name, "codex")
        self.assertIn("read-only", default.command(Path("/tmp/request")))
        selected = backend_from_args(self.args("--agent-model", "chosen-model"))
        self.assertIn("chosen-model", selected.command(Path("/tmp/request")))
        custom = backend_from_args(self.args("--agent-command", '["custom", "literal;argument"]'))
        self.assertEqual(custom.command(Path("/tmp/request")), ["custom", "literal;argument"])
        for flags in (("--opencode-base-url", "http://localhost/v1"),
                      ("--backend", "opencode"),
                      ("--backend", "opencode", "--agent-command", '["custom"]'),
                      ("--agent-model", "chosen", "--agent-command", '["custom"]')):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                backend_from_args(self.args(*flags))
        for endpoint in ("file:///tmp", "http://user:secret@localhost/v1", "http://localhost/v1?key=secret"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                self.local("--opencode-base-url", endpoint)
        with self.assertRaises(ValueError):
            self.local("--opencode-context", "1024", "--opencode-output", "2048")

    def test_private_configuration_locks_model_and_read_only_tools(self):
        backend = self.local("--opencode-api-key-env", "FLINT_API_KEY")
        config = opencode_config(backend.settings, Path("/training"), Path("/skill"))
        self.assertEqual(config["enabled_providers"], ["meta-agent-local"])
        self.assertEqual(config["model"], config["small_model"])
        self.assertEqual(config["permission"]["*"], "deny")
        self.assertEqual(config["permission"]["read"], "allow")
        self.assertEqual(config["permission"]["external_directory"]["/training/*"], "allow")
        self.assertEqual(config["permission"]["external_directory"]["*"], "deny")
        self.assertEqual(config["agent"]["meta-agent-review"]["model"], config["model"])
        self.assertEqual(config["provider"]["meta-agent-local"]["options"]["apiKey"], "{env:FLINT_API_KEY}")
        with tempfile.TemporaryDirectory() as name, patch.dict(os.environ,
                {"OPENCODE_CONFIG": "/untrusted/global.json", "OPENCODE_PERMISSION": '{"*":"allow"}',
                 "FLINT_API_KEY": "secret-not-an-artifact"}):
            environment = opencode_environment(Path(name), config)
            self.assertEqual(environment["HOME"], os.environ["HOME"])
            self.assertNotIn("OPENCODE_CONFIG", environment)
            self.assertNotIn("OPENCODE_PERMISSION", environment)
            self.assertEqual(environment["OPENCODE_DISABLE_PROJECT_CONFIG"], "1")
            self.assertEqual(environment["OPENCODE_PURE"], "1")
            self.assertNotIn("secret-not-an-artifact", json.dumps(config))
            for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "OPENCODE_TEST_HOME"):
                self.assertTrue(Path(environment[key]).is_relative_to(Path(name)))

    def test_final_json_and_usage_without_inventing_missing_counters(self):
        records = events({"phase": "review"}, tokens={"input": 12, "output": 5, "cache": {"read": 3}})
        response, session, usage = parse_opencode_response(encoded(records + records[-1:]))
        self.assertEqual(response, {"phase": "review"})
        self.assertEqual(session, "ses_one")
        self.assertEqual(usage, {"input": 12, "output": 5, "cache_read": 3})
        self.assertEqual(parse_opencode_response(encoded(events({})))[2], {})
        split = events({})
        split[0]["part"]["text"] = '{"phase":'
        second = json.loads(json.dumps(split[0]))
        second["part"].update(id="text_two", text='"review"}')
        split.insert(1, second)
        self.assertEqual(parse_opencode_response(encoded(split))[0], {"phase": "review"})

    def test_unfinished_errors_mixed_sessions_and_unknown_events_are_rejected(self):
        malformed = [events({}, reason="length"), events({})[:1], [],
            [{"type": "error", "sessionID": "ses_one", "error": {"name": "APIError"}}],
            [{"type": "unexpected", "sessionID": "ses_one"}, *events({})]]
        nested = events({})
        nested[0]["part"]["sessionID"] = "ses_other"
        malformed.append(nested)
        mixed = events({})
        mixed[1]["sessionID"] = "ses_other"
        mixed[1]["part"]["sessionID"] = "ses_other"
        malformed.append(mixed)
        stale = events({"review": {"decision": "apply"}})
        stale[-1]["part"]["messageID"] = "new_empty_final"
        malformed.append(stale)
        wrong_type = events({})
        wrong_type[0]["part"]["type"] = "tool"
        malformed.append(wrong_type)
        for records in malformed:
            with self.subTest(records=records), self.assertRaises(ValueError):
                parse_opencode_response(encoded(records))
        for text in ("not JSON", '[]', '{"type":"text","sessionID":"ses_one"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_opencode_response(text)


FAKE_OPENCODE = r'''
import json, os, pathlib, re, subprocess, sys
root = pathlib.Path(__ROOT__)
settings = json.loads((root/'agent-settings.json').read_text())
if '--version' in sys.argv:
    print(settings.get('opencode_version', '1.19.3'))
    raise SystemExit(0)
config = json.loads(os.environ['OPENCODE_CONFIG_CONTENT'])
assert config['permission']['*'] == 'deny'
assert config['enabled_providers'] == ['meta-agent-local']
assert config['model'] == config['small_model']
assert '--title' in sys.argv
assert os.environ['OPENCODE_DISABLE_PROJECT_CONFIG'] == '1'
prompt = sys.stdin.read()
context = json.loads(re.search(r'<context>\s*(.*?)\s*</context>', prompt, re.S).group(1))
request = context['request']
target = pathlib.Path(os.environ['XDG_STATE_HOME'])/'fixture-response.json'
result = subprocess.run([sys.executable, str(root/'agent.py'), '--output-last-message'], input=prompt,
    text=True, capture_output=True, env=dict(os.environ, META_AGENT_RESPONSE=str(target)))
if result.returncode:
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)
response = json.loads(target.read_text())
session = 'ses_' + request['id']
def emit(kind, part=None, **extra):
    record = dict(type=kind, timestamp=1700000000000, sessionID=session, **extra)
    if part is not None:
        record['part'] = dict(part, sessionID=session, messageID='message_'+request['id'])
    print(json.dumps(record), flush=True)
emit('step_start', dict(id='start', type='step-start'))
log = pathlib.Path(context['training_log'])
emit('tool_use', dict(id='tool', type='tool', callID='read_call', tool='read',
    state=dict(status='completed', input=dict(filePath=str(log)),
               output=log.read_text() if log.exists() else 'unavailable before training')))
if settings.get('opencode_error'):
    emit('error', error=dict(name='APIError'))
elif settings.get('opencode_badfinal'):
    emit('text', dict(id='text', type='text', text='not a JSON object'))
else:
    emit('text', dict(id='text', type='text', text=json.dumps(response)))
emit('step_finish', dict(id='finish', type='step-finish', reason='stop', cost=0,
    tokens=dict(input=120, output=25, reasoning=0, cache=dict(read=0, write=0))))
'''


class OpenCodeSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SupervisorTests('test_discovery_precedes_training_and_reads_skill')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        script = self.root / 'fake-opencode.py'
        script.write_text(FAKE_OPENCODE.replace('__ROOT__', repr(str(self.root))))
        base = list(self.fixture.base)
        index = base.index('--agent-command')
        del base[index:index + 2]
        self.base = base + ['--backend', 'opencode', '--agent-model', 'flint', '--opencode-base-url',
            'http://127.0.0.1:30000/v1', '--opencode-command', json.dumps([sys.executable, str(script)]), '--export-atif']

    def call(self, *extra, expect=0):
        return self.fixture.call(*extra, expect=expect, base=self.base)

    def requests(self):
        return list((self.root / 'reports/supervision/RUN/requests').iterdir())

    def test_local_backend_discovery_review_and_atif_are_bound_to_each_request(self):
        self.call()
        self.fixture.append_step(20)
        (self.fixture.trajectories / 'batch.json').write_text('{"input":"x","target":1,"prediction":0}')
        self.call()
        self.assertEqual([call['phase'] for call in self.fixture.calls()], ['discovery', 'review'])
        sessions = set()
        for directory in self.requests():
            request = json.loads((directory / 'request.json').read_text())
            metadata = json.loads((directory / 'agent.backend.json').read_text())
            self.assertEqual(request['backend'], 'opencode')
            sessions.add(metadata['session_id'])
            self.assertEqual(metadata['model'], 'flint')
            trajectory = json.loads((directory / 'trajectory.json').read_text())
            self.assertTrue(any(step.get('tool_calls') for step in trajectory['steps']))
            self.assertEqual(trajectory['agent']['extra']['backend'], 'opencode')
        self.assertEqual(len(sessions), 2)
        self.assertEqual(self.fixture.state()['next_review_step'], 40)

    def test_apply_still_goes_through_the_existing_project_gate(self):
        self.fixture.settings(apply=True)
        self.call(*self.fixture.apply_args())
        self.assertFalse(self.fixture.calls('executor-calls.jsonl'))
        self.fixture.append_step(20)
        self.call(*self.fixture.apply_args())
        self.assertEqual(len(self.fixture.calls('executor-calls.jsonl')), 1)
        self.assertEqual(json.loads((self.root / 'control.json').read_text())['version'], 2)

    def test_model_failure_or_wrong_response_never_dispatches_the_gate(self):
        self.call(*self.fixture.apply_args())
        self.fixture.append_step(20)
        self.fixture.settings(apply=True, opencode_badfinal=True)
        self.call(*self.fixture.apply_args(), expect=2)
        self.assertFalse(self.fixture.calls('executor-calls.jsonl'))
        self.assertEqual(json.loads((self.root / 'control.json').read_text())['version'], 1)
        self.assertEqual(self.fixture.state()['status'], 'needs_reconciliation')
        manifests = [json.loads((p / 'export.json').read_text()) for p in self.requests()]
        self.assertTrue(any(m['coverage']['partial'] for m in manifests))

    def test_unsupported_cli_fails_before_model_invocation(self):
        self.fixture.settings(opencode_version='2.0.0')
        self.call(expect=2)
        self.assertFalse(self.fixture.calls())
        self.assertEqual(self.fixture.state()['calls'], 1)
        self.assertFalse(any((p / 'response.json').exists() for p in self.requests()))

    def test_timeout_cleans_up_nested_cli_and_model_children(self):
        self.fixture.settings(sleep=10)
        self.call('--agent-timeout', '2', expect=2)
        child = int((self.root / 'agent-pid.txt').read_text())
        self.assertIsNone(pid_start(child))
        self.assertEqual(self.fixture.state()['status'], 'needs_reconciliation')
        self.assertTrue(any((p / 'trajectory.json').exists() for p in self.requests()))


if __name__ == '__main__':
    unittest.main()
