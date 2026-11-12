"""Real-process checks for single-flight audits and late-publication protection."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from meta_agent.nonblocking_audit import NonBlockingAudit, training_metrics


FAKE = r'''
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('phase'); p.add_argument('--request'); p.add_argument('--response'); p.add_argument('--publication')
a=p.parse_args(); req=json.loads(Path(a.request).read_text()); ctx=json.loads(Path(req['context_path']).read_text())
if a.phase == 'review':
    if ctx.get('spawn_child'):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
        Path(ctx['child_pid_path']).write_text(str(child.pid))
    time.sleep(ctx.get('sleep',0))
    if ctx.get('fail'): sys.exit(7)
    out=dict(schema_version=1,request_id=req['id'],run=req['run'],step=req['step'],
             decision=ctx.get('decision','observe'),proposal=None)
    if out['decision']=='apply': out['proposal']={'value':2}
    if ctx.get('wrong_run'): out['run']='another_round'
    Path(a.response).write_text(json.dumps(out))
else:
    time.sleep(ctx.get('publish_sleep',0))
    paths=list(req['state_hashes'])
    hashes=lambda:{v:hashlib.sha256(Path(v).read_bytes()).hexdigest() for v in paths}
    status='rejected'
    if ctx.get('external_before_publish'): Path(paths[0]).write_text('external writer')
    observed=hashes()
    state=json.loads(Path(req['coordinator_state_path']).read_text())
    if state['publication_allowed'] and state['run']==req['run'] and hashes()==req['state_hashes']:
        target=Path(paths[0]); tmp=target.with_suffix('.next'); tmp.write_text('updated'); os.replace(tmp,target)
        status='applied'
    elif observed != req['state_hashes']: status='stale'
    if ctx.get('bad_receipt'): sys.exit(0)
    out=dict(schema_version=1,request_id=req['id'],run=req['run'],status=status,
             before_state_hashes=req['state_hashes'],after_state_hashes=hashes(),
             observed_before_state_hashes=observed)
    Path(a.publication).write_text(json.dumps(out))
'''


class NonBlockingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fake = self.root / "fake.py"
        self.fake.write_text(FAKE)
        self.context = self.root / "context.json"
        self.mutable = self.root / "mutable.json"
        self.mutable.write_text("original")
        self.audits = []
        self.addCleanup(self.cleanup_audits)

    def cleanup_audits(self):
        for audit in self.audits:
            try:
                audit.close(timeout=0)
            except RuntimeError:
                pass

    def make(self, *, context=None, publisher=True, **overrides):
        self.context.write_text(json.dumps(dict(run="r1", **(context or {}))))
        command = [sys.executable, str(self.fake), "review", "--request", "{request}", "--response", "{response}"]
        publish = [sys.executable, str(self.fake), "publish", "--request", "{request}",
                   "--response", "{response}", "--publication", "{publication}"] if publisher else []
        args = dict(run="r1", directory=self.root / "audit", context_path=self.context,
                    mutable_paths=[self.mutable], command=command, publish_command=publish,
                    every=20, stop=100, timeout=5)
        args.update(overrides)
        audit = NonBlockingAudit(**args)
        self.audits.append(audit)
        return audit

    def drive(self, audit, predicate, *, timeout=3):
        end = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= end:
                self.fail("audit did not reach expected state")
            audit.poll()
            time.sleep(0.01)

    def events(self, audit):
        return [json.loads(line) for line in audit.events_path.read_text().splitlines()]

    def test_training_continues_and_busy_requests_coalesce(self):
        audit = self.make(context={"sleep": 0.35})
        started = time.monotonic()
        audit.on_step(20, {"task_reward/success": 0.1})
        audit.on_step(40, {})
        audit.on_step(60, {})
        self.assertLess(time.monotonic() - started, 0.3)
        self.assertEqual(audit.state["submitted"], 1)
        self.assertEqual(audit.state["pending_step"], 60)
        self.assertEqual(audit.state["coalesced"], 1)
        self.drive(audit, lambda: audit.state["completed"] == 2)
        requests = [json.loads(p.read_text()) for p in (audit.directory / "requests").glob("*/request.json")]
        self.assertEqual(sorted(r["step"] for r in requests), [20, 60])

    def test_first_completed_batch_triggers_without_waiting_for_cadence(self):
        audit = self.make(context={"sleep": 0.1})
        audit.on_step(1, {"task_reward/success": 0})
        self.assertEqual(audit.state["submitted"], 1)
        self.assertEqual(audit._active["request"]["step"], 1)
        audit.on_step(2, {})
        self.assertIsNone(audit.state["pending_step"])
        self.assertEqual(audit.state["submitted"], 1)

    def test_stale_base_rejects_publication(self):
        audit = self.make(context={"decision": "apply", "sleep": 0.15})
        audit.on_step(20, {})
        self.mutable.write_text("external update")
        self.drive(audit, lambda: audit.state["stale"] == 1)
        self.assertEqual(self.mutable.read_text(), "external update")
        self.assertFalse(any(p.name == "publication.json" for p in audit.directory.rglob("*")))

    def test_valid_publication_has_request_bound_receipt(self):
        audit = self.make(context={"decision": "apply"})
        audit.on_step(20, {})
        self.drive(audit, lambda: audit.state["completed"] == 1)
        self.assertEqual(self.mutable.read_text(), "updated")
        receipt = json.loads(next(audit.directory.glob("requests/*/publication.json")).read_text())
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["after_state_hashes"][str(self.mutable)],
                         hashlib.sha256(self.mutable.read_bytes()).hexdigest())

    def test_publication_without_receipt_requires_reconciliation(self):
        audit = self.make(context={"decision": "apply", "bad_receipt": True},
                          failure_policy="continue_unchanged")
        audit.on_step(20, {})
        with self.assertRaisesRegex(RuntimeError, "Invalid publisher result"):
            self.drive(audit, lambda: audit.state["status"] == "needs_reconciliation")
        self.assertEqual(audit.state["status"], "needs_reconciliation")
        self.assertFalse(audit.state["publication_allowed"])

    def test_publisher_can_report_a_newer_observed_base_without_writing(self):
        audit = self.make(context={"decision": "apply", "external_before_publish": True})
        audit.on_step(20, {})
        self.drive(audit, lambda: audit.state["completed"] == 1)
        self.assertEqual(audit.state["stale"], 1)
        self.assertEqual(audit.state["failures"], 0)
        self.assertEqual(self.mutable.read_text(), "external writer")

    def test_final_step_never_publishes_and_has_read_only_final_request(self):
        audit = self.make(context={"decision": "apply", "sleep": 0.1})
        audit.on_step(20, {})
        audit.before_step(100, is_final=True)
        self.assertFalse(json.loads(audit.state_path.read_text())["publication_allowed"])
        audit.on_step(100, {}, is_final=True)
        audit.close(timeout=2)
        self.assertEqual(self.mutable.read_text(), "original")
        self.assertEqual(audit.state["final_review_outcome"], "apply_discarded_read_only")
        final = [json.loads(p.read_text()) for p in audit.directory.glob("requests/*/request.json")
                 if json.loads(p.read_text())["is_final"]]
        self.assertEqual(len(final), 1)

    def test_pending_publisher_cancel_does_not_mutate_final_rollout(self):
        audit = self.make(context={"decision": "apply", "publish_sleep": 2})
        audit.on_step(20, {})
        self.drive(audit, lambda: bool(audit._active and audit._active["phase"] == "publish"))
        audit.before_step(100, is_final=True)
        audit.on_step(100, {}, is_final=True)
        audit.close(timeout=2)
        self.assertEqual(self.mutable.read_text(), "original")
        self.assertEqual(audit.state["failures"], 0)

    def test_close_kills_process_group_and_preserves_cancellation(self):
        child_pid = self.root / "child.pid"
        audit = self.make(context={"sleep": 60, "spawn_child": True, "child_pid_path": str(child_pid)})
        audit.on_step(100, {}, is_final=True)
        self.drive(audit, child_pid.exists)
        child = int(child_pid.read_text())
        pid = audit._active["process"].pid
        started = time.monotonic()
        audit.close(timeout=0.02)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(audit.state["cancelled"], 1)
        self.assertEqual(audit.state["final_review_outcome"], "cancelled")
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        # SIGKILL has been sent to the whole group. The kernel may still report
        # an exiting orphan as runnable briefly; wait for exit or zombie state
        # rather than depending on the scheduler at this exact instruction.
        child_stat = Path(f"/proc/{child}/stat")
        deadline = time.monotonic() + 1
        while True:
            try: state = child_stat.read_text().rsplit(') ', 1)[1].split()[0]
            except FileNotFoundError: break
            if state == 'Z': break
            if time.monotonic() >= deadline:
                self.fail('A cancelled reviewer descendant remained runnable')
            time.sleep(.005)

    def test_wrong_run_response_is_a_failure_without_publish(self):
        audit = self.make(context={"decision": "apply", "wrong_run": True},
                          failure_policy="continue_unchanged")
        audit.on_step(20, {})
        self.drive(audit, lambda: audit.state["failures"] == 1)
        self.assertEqual(self.mutable.read_text(), "original")
        self.assertFalse(any(e["event"] == "publication_finished" for e in self.events(audit)))

    def test_context_snapshot_and_training_only_metrics(self):
        audit = self.make(context={"sleep": 0.05})
        audit.on_step(1, {"task_reward/success": 0.2, "val-core/reward": 1,
                          "test/pass": 1, "evaluation/success": 1, "actor/nan": float("nan")})
        audit.on_step(20, {"actor/entropy": 0.01})
        request = json.loads(next(audit.directory.glob("requests/*/request.json")).read_text())
        self.context.write_text(json.dumps({"run": "another_round"}))
        self.assertEqual(json.loads(Path(request["context_path"]).read_text())["run"], "r1")
        rows = [json.loads(x) for x in audit.metrics_path.read_text().splitlines()]
        self.assertEqual(rows[0]["metrics"], {"task_reward/success": 0.2})
        self.assertEqual(rows[1]["step"], 20)

    def test_second_coordinator_cannot_own_same_round(self):
        self.make()
        with self.assertRaisesRegex(RuntimeError, "Another audit coordinator"):
            self.make()

    def test_disabled_hook_leaves_legacy_training_untouched(self):
        path = Path(__file__).parents[1] / "meta_agent/training_audit.py"
        spec = importlib.util.spec_from_file_location("audit_hook_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(os.environ, {"EAU_ASYNC_AUDIT_COMMAND": ""}):
            module.before_training_step(1)
            module.audit_training_step(1, {"reward": 1})
            module.close_training_audit()
        self.assertIsNone(module._coordinator)


if __name__ == "__main__":
    unittest.main()
