"""Real-process direct editing without a JSON completion or proxy publisher."""
import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from meta_agent.direct_file_audit import DirectFileAudit


EDITOR = r'''
import argparse,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');a=p.parse_args()
r=json.loads(Path(a.request).read_text());c=json.loads(Path(r['context_path']).read_text())
time.sleep(c.get('delay',0.05))
target=Path(r['editable_roots'][0])/'control.json'
target.write_text(json.dumps({'version':r['step']}))
print('Changed the declared control after examining the supplied training context.')
if c.get('fail_after_write'):sys.exit(7)
'''


class DirectFileAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.live = self.root / 'live'; self.live.mkdir()
        self.control = self.live / 'control.json'; self.control.write_text('{"version":0}')
        self.editor = self.root / 'editor.py'; self.editor.write_text(EDITOR)
        self.context = self.root / 'context.json'
        self.audits = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for audit in self.audits:
            try: audit.close(timeout=0)
            except RuntimeError: pass

    def make(self, **context):
        self.context.write_text(json.dumps(dict(run='r1', source_scope='training_only', **context)))
        audit = DirectFileAudit(run='r1', directory=self.root / 'audits', context_path=self.context,
            editable_roots=[self.live], command=[sys.executable, str(self.editor), '--request', '{request}'],
            first=1, every=3, stop=10, timeout=5, failure_policy='continue_unchanged')
        self.audits.append(audit)
        return audit

    def drive(self, audit, predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            if time.monotonic() > deadline: self.fail('Direct editor did not reach the expected state')
            audit.poll(); time.sleep(.01)

    def test_natural_language_editor_changes_files_without_json_or_a_publisher(self):
        audit = self.make()
        audit.on_step(1, {'train/loss': .5})
        self.drive(audit, lambda: audit.state['completed'] == 1)
        self.assertEqual(json.loads(self.control.read_text()), {'version': 1})
        request = next((audit.directory / 'requests').iterdir())
        self.assertFalse((request / 'response.json').exists())
        self.assertIn('Changed the declared control', (request / 'review.stdout').read_text())
        receipt = json.loads((request / 'direct_receipt.json').read_text())
        self.assertTrue(receipt['observed_files_changed'])
        self.assertFalse(receipt['file_write_is_worker_uptake'])
        self.assertEqual(receipt['learning_benefit'], 'unverified')
        self.assertTrue((audit.directory / 'direct_edits.jsonl').exists())

    def test_training_triggers_do_not_wait_for_the_editor_and_requests_coalesce(self):
        audit = self.make(delay=.3)
        started = time.monotonic()
        audit.on_step(1, {})
        audit.on_step(3, {})
        audit.on_step(6, {})
        self.assertLess(time.monotonic() - started, .25)
        self.assertEqual(audit.state['submitted'], 1)
        self.assertEqual(audit.state['pending_step'], 6)
        self.assertEqual(audit.state['coalesced'], 1)
        self.drive(audit, lambda: audit.state['completed'] == 2)
        requests = sorted(json.loads(p.read_text())['step'] for p in audit.directory.glob('requests/*/request.json'))
        self.assertEqual(requests, [1, 6])

    def test_final_boundary_reaps_the_editor_before_return_and_never_launches_a_final_writer(self):
        audit = self.make(delay=3)
        audit.on_step(1, {})
        pid = audit.state['active_request']['pid']
        audit.before_step(10, is_final=True)
        with self.assertRaises(ProcessLookupError): os.kill(pid, 0)
        self.assertEqual(json.loads(self.control.read_text()), {'version': 0})
        audit.on_step(10, {}, is_final=True)
        self.assertEqual(audit.state['submitted'], 1)
        self.assertFalse(audit.state['agent_writes_allowed'])
        self.assertEqual(audit.state['final_review_outcome'], 'skipped_direct_editor_closed')

    def test_failed_editor_with_actual_writes_requires_reconciliation(self):
        audit = self.make(fail_after_write=True)
        audit.on_step(1, {})
        with self.assertRaises(RuntimeError): self.drive(audit, lambda: audit.state['failures'] == 1)
        self.assertEqual(audit.state['status'], 'needs_reconciliation')
        self.assertEqual(json.loads(self.control.read_text()), {'version': 1})

    def test_editable_roots_cannot_cover_audit_state_or_overlap(self):
        self.context.write_text('{"run":"r1"}')
        for roots in [[self.root], [self.live, self.live]]:
            with self.assertRaises(ValueError):
                DirectFileAudit(run='r1', directory=self.root / 'audits', context_path=self.context,
                    editable_roots=roots, command=[sys.executable, str(self.editor)])

    def test_environment_hook_selects_direct_editor_with_natural_completion(self):
        path = Path(__file__).parents[1] / 'meta_agent/training_audit.py'
        spec = importlib.util.spec_from_file_location('direct_hook_under_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.context.write_text('{"run":"r1","source_scope":"training_only"}')
        env = dict(EAU_ASYNC_AUDIT_COMMAND=json.dumps([sys.executable, str(self.editor), '--request', '{request}']),
            EAU_ASYNC_AUDIT_DIRECT_ROOTS=json.dumps([str(self.live)]), EAU_ASYNC_AUDIT_RUN='r1',
            EAU_ASYNC_AUDIT_DIR=str(self.root / 'audits'), EAU_AUDIT_CONTEXT_PATH=str(self.context),
            EAU_ASYNC_AUDIT_MUTABLE_PATHS='[]', EAU_ASYNC_AUDIT_PUBLISH_COMMAND='[]',
            EAU_STEP_REVIEW_STATE='', EAU_ASYNC_AUDIT_STOP='10', EAU_ASYNC_AUDIT_DRAIN_TIMEOUT='0')
        with patch.dict(os.environ, env):
            module.before_training_step(1)
            self.audits.append(module._coordinator)
            self.assertIsInstance(module._coordinator, DirectFileAudit)
            module.audit_training_step(1, {'train/loss': .5})
            self.drive(module._coordinator, lambda: module._coordinator.state['completed'] == 1)
            module.close_training_audit()
        self.assertEqual(json.loads(self.control.read_text()), {'version': 1})


if __name__ == '__main__': unittest.main()
