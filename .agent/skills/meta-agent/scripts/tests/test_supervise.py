"""Exercise the scheduler and child protocol offline against real temporary files/processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from supervise import Supervisor, parser, pid_start

AGENT = r'''
import json, os, pathlib, re, sys, time
prompt = sys.stdin.read()
context = json.loads(re.search(r'<context>\s*(.*?)\s*</context>', prompt, re.S).group(1))
root = pathlib.Path(context['training_root'])
settings = json.loads((root/'agent-settings.json').read_text())
request = context['request']
phase = request['phase']
with (root/'agent-calls.jsonl').open('a') as f:
    f.write(json.dumps({'phase': phase, 'step': request['step'], 'loaded_skill': '# Meta-Agent' in prompt,
                        'context': context, 'argv': sys.argv[1:]})+'\n')
if settings.get('sleep'):
    (root/'agent-pid.txt').write_text(str(os.getpid()))
    time.sleep(settings['sleep'])
if settings.get('fail'):
    sys.stderr.write('offline agent failure\n')
    sys.exit(7)
log = pathlib.Path(context['training_log'])
log_evidence = log.read_text()[-300:] if log.exists() else 'unavailable: training has not started'
paths = context['recent_trajectory_files']
trajectory = pathlib.Path(paths[-1]).read_text() if paths else 'unavailable: no training trajectory yet'
review = dict(decision='leave_unchanged', tier='user', knob='disclosure',
              change='reason: collect enough clean exposure; next: next scheduled review',
              evidence='trajectory: '+trajectory.replace(';', ',').replace('|', '/'),
              prediction='—', verification='pending observation, no application')
action = dict(operation='', arguments_json='{}')
if settings.get('apply') and phase == 'review':
    review.update(decision='apply',
        change='target: useful learning; old: version 1; new: version 2; mechanism: count repeated facts; reason: repeated training replies; next: 200 clean episodes; recovery: revert control only',
        prediction='repeat rate below 0.4; window: clean steps 21-60 and 200 admitted episodes',
        verification='application: pending; attempt: not sent, awaiting configured gate')
    action = dict(operation=settings.get('operation', 'set_repeat_budget'), arguments_json='{"expected":1,"value":2}')
if phase == 'reconcile':
    review.update(decision='observe', change='reason: inspect previous request before any new action; window: uptake still pending; next: new reader acknowledgment',
                  evidence='trajectory: '+trajectory.replace(';', ',').replace('|', '/')+'; previous request artifacts inspected')
if settings.get('invalid'):
    review['change'] = 'reason: incomplete result'
if settings.get('false_applied') and phase == 'review':
    review['verification'] = 'application: fully_applied; attempt: assumed from writable file'
if settings.get('tags_in_verification') and phase == 'review':
    review['change'] = review['change'].replace('; reason: repeated training replies; next: 200 clean episodes', '')
    review['verification'] += '; reason: repeated training replies; next: 200 clean episodes'
response = dict(phase=phase, summary='read log and trajectory: '+trajectory[:60],
    intervention_map='| Component | Mechanism | Applicability | Runtime | Evidence | Value | Recovery | Decision |\n| user reply | repeat accounting | inspect current samples | restart-only | local source | test information gain | retain lineage | instrument |\n',
    sources=dict(paper=['EAU Appendix A.2; versioned discovery guide'], experience=['unavailable: new fixture'],
                 code=['fixture/user.py repeat branch'], logs=[str(log)+': '+log_evidence], trajectories=[trajectory]),
    review=review, action=action, pending_predictions=['H1 pending clean exposure through step 60'],
    experience_update='Observation: fixture repeat facts; applicability: synthetic run only.',
    next_condition='next cadence or sufficient clean exposure', continue_supervision=not settings.get('handoff', False))
if settings.get('empty_map'):
    response['intervention_map']=''
if settings.get('missing_trajectories'):
    response['sources']['trajectories']=[]
if settings.get('wrong_phase'):
    response['phase']='final'
if settings.get('advance'):
    with log.open('a') as f:
        f.write(f"step:{request['step']+settings['advance']} - global_seqlen:100\n")
if settings.get('file_response') or '--output-last-message' in sys.argv:
    pathlib.Path(os.environ['META_AGENT_RESPONSE']).write_text(json.dumps(response))
    print('{"type":"turn.completed"}')
else:
    print(json.dumps(response))
'''

EXECUTOR = r'''
import json, os, pathlib, sys, time
payload=json.load(sys.stdin)
root=pathlib.Path.cwd()
settings=json.loads((root/'executor-settings.json').read_text())
assert payload['operation'] == 'set_repeat_budget'
table=(root/'reports/change_log.md').read_text()
assert payload['request_id']+'-proposal' in table and 'repeat rate below 0.4' in table
with (root/'executor-calls.jsonl').open('a') as f:
    f.write(json.dumps(payload)+'\n')
if not settings.get('fail'):
    config=root/'control.json'
    assert json.loads(config.read_text())['version']==payload['arguments']['expected']
    temporary=root/'control.tmp'
    temporary.write_text(json.dumps({'version': payload['arguments']['value']}))
    temporary.replace(config)
if settings.get('sleep_after_write'):
    time.sleep(settings['sleep_after_write'])
receipt=dict(application='failed' if settings.get('fail') else 'fully_applied', attempt='fixture gate request completed',
             boundary='next fixture episode', scope='fixture_reader', confirmed='fixture_reader',
             uptake='fixture reader read back version 2', recovery='restore old control, retain prior learning')
if settings.get('missing_uptake'):
    receipt.pop('uptake')
if settings.get('malformed_fields'):
    receipt.update(boundary=21, scope=['fixture_reader'], confirmed=['fixture_reader'], uptake={'version':2})
if settings.get('incomplete_pending'):
    receipt={key:receipt[key] for key in ('application', 'attempt', 'recovery')}
    receipt['application']='pending'
if settings.get('pending'):
    receipt.update(application='pending', boundary='', confirmed='', uptake='')
print(json.dumps(receipt))
sys.exit(1 if settings.get('fail') or settings.get('contradictory_exit') else 0)
'''


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='supervise-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root/'train.log'
        self.trajectories = self.root/'trajectories'
        self.trajectories.mkdir()
        (self.root/'agent.py').write_text(AGENT)
        (self.root/'executor.py').write_text(EXECUTOR)
        (self.root/'control.json').write_text('{"version":1}')
        self.settings()
        self.executor_settings()
        self.contract = dict(objective='full-task completion', budget='200 steps and 20 agent calls',
                             guardrails=['preserve useful clarification'], allowed_feedback=['training log and trajectories'],
                             excluded_feedback=['independent evaluation scores and trajectories'],
                             evaluation_protocol='fixed held-out evaluator, no steering access',
                             checkpoint_selection='final checkpoint without test selection',
                             allowed_operations=['set_repeat_budget'])
        self.contract_path = self.root/'contract.json'
        self.contract_path.write_text(json.dumps(self.contract))
        self.state_path = self.root/'reports/supervision/RUN/state.json'
        self.table = self.root/'reports/change_log.md'
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        for name in ('EAU_ROOT', 'EAU_GYM_ROOT', 'EAU_FIRST_REVIEW', 'EAU_REVIEW_INTERVAL', 'EAU_MANDATE_INTERVAL'):
            self.env.pop(name, None)
        self.base = ['--root', str(self.root), '--run', 'RUN', '--log', str(self.log),
                     '--trajectories', str(self.trajectories), '--contract', str(self.contract_path),
                     '--every-steps', '20', '--agent-command', json.dumps([sys.executable, str(self.root/'agent.py')]),
                     '--max-calls', '20', '--once']

    def settings(self, **values):
        (self.root/'agent-settings.json').write_text(json.dumps(values))

    def executor_settings(self, **values):
        (self.root/'executor-settings.json').write_text(json.dumps(values))

    def append_step(self, step):
        with self.log.open('a') as f:
            f.write(f'step:{step} - global_seqlen:100\n')

    def call(self, *extra, expect=0, base=None):
        result = subprocess.run([sys.executable, str(SCRIPTS/'supervise.py'), *(base or self.base), *map(str, extra)],
                                env=self.env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, expect, result.stdout+result.stderr)
        return result

    def calls(self, filename='agent-calls.jsonl'):
        path = self.root/filename
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def state(self):
        return json.loads(self.state_path.read_text())

    def apply_args(self):
        return ['--mode', 'apply', '--executor-command', json.dumps([sys.executable, str(self.root/'executor.py')])]

    def test_discovery_precedes_training_and_reads_skill(self):
        self.call()
        self.assertFalse(self.log.exists())
        self.assertEqual([x['phase'] for x in self.calls()], ['discovery'])
        self.assertTrue(self.calls()[0]['loaded_skill'])
        self.assertTrue(self.state()['discovery_complete'])
        self.assertFalse(self.state()['supervisor']['running'])
        self.assertIn('external host/caller', self.state()['supervisor']['next_wake_owner'])
        self.assertEqual(self.calls()[0]['context']['schedule']['every_steps'], 20)
        self.assertTrue((self.root/'reports/intervention_map_RUN.md').exists())
        self.assertFalse(self.calls('executor-calls.jsonl'))
        self.assertIn('versioned mechanism guide', next((self.root/'reports/supervision/RUN/requests').glob('*/prompt.txt')).read_text())

    def test_steps_read_actual_log_trajectory_and_preserve_pending_predictions(self):
        self.call()
        self.append_step(19)
        self.call()
        self.assertEqual(len(self.calls()), 1)
        (self.trajectories/'step20.json').write_text('{"reply":"same fact twice"}')
        self.append_step(20)
        self.call()
        self.assertEqual([x['phase'] for x in self.calls()], ['discovery', 'review'])
        responses=list((self.root/'reports/supervision/RUN/requests').glob('*/response.json'))
        review=next(json.loads(p.read_text()) for p in responses if json.loads(p.read_text())['phase']=='review')
        self.assertIn('same fact twice', review['sources']['trajectories'][0])
        self.assertIn('step:20', review['sources']['logs'][0])
        self.assertEqual(self.state()['pending_predictions'], ['H1 pending clean exposure through step 60'])
        self.call()
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual(self.state()['next_review_step'], 40)

    def test_time_schedule_reviews_even_without_new_step(self):
        args=parser().parse_args([*self.base, '--every-seconds', '60'])
        now=[100.0]
        supervisor=Supervisor(args, clock=lambda: now[0])
        supervisor.load()
        supervisor.tick()
        now[0]=159.0
        supervisor.tick()
        self.assertEqual(len(self.calls()), 1)
        now[0]=160.0
        supervisor.tick()
        self.assertEqual([x['phase'] for x in self.calls()], ['discovery', 'review'])
        self.assertEqual(supervisor.state['next_review_at'], 220)

    def test_time_only_schedule_is_supported(self):
        base=self.base.copy()
        index=base.index('--every-steps')
        del base[index:index+2]
        self.call('--every-seconds', '60', base=base)
        self.assertIsNone(self.state().get('next_review_step'))
        self.assertIsNotNone(self.state()['next_review_at'])

    def test_continuous_mode_dispatches_real_child_processes_serially(self):
        self.settings(advance=20)
        base=[x for x in self.base if x!='--once']
        self.call('--max-calls', '3', '--poll-seconds', '0.01', base=base)
        self.assertEqual([x['phase'] for x in self.calls()], ['discovery','review','review'])
        self.assertEqual(self.state()['status'], 'budget_exhausted')
        self.assertEqual(self.state()['calls'], 3)

    def test_step_jumps_acknowledge_gaps_without_backfilled_reviews(self):
        self.call()
        self.append_step(65)
        self.call()
        text=self.table.read_text()
        self.assertIn('| RUN | 20 | supervision | missed review |', text)
        self.assertIn('| RUN | 40 | supervision | missed review |', text)
        self.assertIn('| RUN | 60 | supervision | missed review |', text)
        self.assertEqual(self.calls()[-1]['step'], 65)
        self.assertEqual(self.state()['next_review_step'], 80)

    def test_resume_anchor_excludes_unexecuted_boundaries(self):
        self.log.write_text('Setting global step to 225\nstep:230 - global_seqlen:100\n')
        self.call()
        self.assertEqual(self.state()['next_review_step'], 240)
        self.assertNotIn('| RUN | 20 |', self.table.read_text())
        self.append_step(240)
        self.call()
        self.assertEqual(self.calls()[-1]['step'], 240)

    def test_resume_anchor_arriving_after_preflight_shifts_next_boundary(self):
        self.call()
        self.log.write_text('Setting global step to 100\n')
        self.call()
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.state()['next_review_step'], 120)
        self.append_step(120)
        self.call()
        self.assertEqual(self.calls()[-1]['step'], 120)
        self.assertNotIn('missed review', self.table.read_text())

    def test_resume_anchor_after_completed_metrics_is_rejected(self):
        self.append_step(20)
        self.call()
        with self.log.open('a') as handle:
            handle.write('Setting global step to 100\n')
        self.call(expect=2)
        self.assertEqual(len(self.calls()), 1)

    def test_partial_step_line_waits_for_completion(self):
        self.call()
        self.log.write_text('step:20 - global_seqlen:')
        self.call()
        self.assertEqual(len(self.calls()), 1)
        with self.log.open('a') as f:
            f.write('100\n')
        self.call()
        self.assertEqual(len(self.calls()), 2)

    def test_replaced_truncated_and_regressed_logs_fail_instead_of_switching_runs(self):
        self.append_step(20)
        self.call()
        before=len(self.calls())
        self.log.replace(self.root/'old.log')
        self.append_step(40)
        self.call(expect=2)
        self.assertEqual(len(self.calls()), before)

    def test_regressed_step_fails(self):
        self.append_step(20)
        self.call()
        self.append_step(19)
        self.call(expect=2)
        self.assertEqual(len(self.calls()), 1)

    def test_log_identity_and_saved_binding_cannot_cross_runs(self):
        self.call()
        other=self.root/'other.log'
        other.write_text('step:100 - global_seqlen:100\n')
        self.call('--log', other, expect=2)
        self.assertEqual(len(self.calls()), 1)
        named=self.root/'train_fixture_20261021_120000.log'
        self.call('--log', named, expect=2)

    def test_explicit_settings_adoption_reassesses_before_any_execution(self):
        self.call()
        self.contract['guardrails'].append('new restriction')
        self.contract_path.write_text(json.dumps(self.contract))
        self.call(expect=2)
        self.call('--adopt-settings')
        self.assertEqual([x['phase'] for x in self.calls()], ['discovery', 'discovery'])
        self.assertEqual(len(self.state()['binding_history']), 1)
        self.assertFalse(self.calls('executor-calls.jsonl'))

    def test_propose_mode_does_not_execute(self):
        self.call()
        self.settings(apply=True)
        self.append_step(20)
        self.call(expect=2)
        self.assertEqual(json.loads((self.root/'control.json').read_text()), {'version': 1})
        self.assertFalse(self.calls('executor-calls.jsonl'))

    def test_apply_requires_explicit_executor_and_operation_scope(self):
        self.call('--mode', 'apply', expect=2)
        self.assertFalse(self.calls())
        self.contract['allowed_operations']=[]
        self.contract_path.write_text(json.dumps(self.contract))
        self.call(*self.apply_args(), expect=2)
        self.assertFalse(self.calls())

    def test_apply_preregisters_then_gate_writes_and_records_receipt(self):
        self.settings(apply=True)
        args=self.apply_args()
        self.call(*args)
        self.assertFalse(self.calls('executor-calls.jsonl'))
        self.append_step(20)
        self.call(*args)
        self.assertEqual(json.loads((self.root/'control.json').read_text()), {'version': 2})
        self.assertEqual(len(self.calls('executor-calls.jsonl')), 1)
        text=self.table.read_text()
        self.assertIn('application: pending', text)
        self.assertIn('application: fully_applied', text)
        self.call(*args)
        self.assertEqual(len(self.calls('executor-calls.jsonl')), 1)
        self.assertTrue(list((self.root/'reports/supervision/RUN/requests').glob('*/receipt.json')))

    def test_receipt_preserves_valid_review_tags_in_verification(self):
        args=self.apply_args()
        self.settings(apply=True, tags_in_verification=True)
        self.call(*args)
        self.append_step(20)
        self.call(*args)
        self.assertEqual(self.state()['status'], 'watching')
        self.assertIn('application: fully_applied', self.table.read_text())
        self.assertIn('planned_recovery: revert control only', self.table.read_text())
        from check_mandate import _review_rows, validate_review
        rows=_review_rows(path=self.table)
        self.assertEqual(len(rows), 3)
        self.assertFalse([error for row in rows for error in validate_review(row)])

    def test_operation_outside_contract_is_not_executed(self):
        args=self.apply_args()
        self.call(*args)
        self.settings(apply=True, operation='unrelated_write')
        self.append_step(20)
        self.call(*args, expect=2)
        self.assertFalse(self.calls('executor-calls.jsonl'))

    def test_gate_receives_current_step_when_analysis_advances_training(self):
        args=self.apply_args()
        self.call(*args)
        self.settings(apply=True, advance=20)
        self.append_step(20)
        self.call(*args)
        payload=self.calls('executor-calls.jsonl')[0]
        self.assertEqual(payload['observation_step'], 20)
        self.assertEqual(payload['step'], 40)
        self.assertIn('decision_step: 40', self.table.read_text())

    def test_no_application_after_end_boundary_arrives_during_analysis(self):
        args=self.apply_args()+['--stop-step','40']
        self.call(*args)
        self.settings(apply=True, advance=20)
        self.append_step(20)
        self.call(*args)
        self.assertFalse(self.calls('executor-calls.jsonl'))
        self.assertEqual(json.loads((self.root/'control.json').read_text()), {'version':1})
        self.assertIn('not sent: configured end boundary', self.table.read_text())
        self.settings()
        self.call(*args)
        self.assertEqual(self.calls()[-1]['phase'], 'final')

    def test_success_receipt_with_failing_exit_does_not_claim_full_application(self):
        args=self.apply_args()
        self.settings(apply=True)
        self.executor_settings(contradictory_exit=True)
        self.call(*args)
        self.append_step(20)
        self.call(*args, expect=2)
        self.assertNotIn('application: fully_applied', self.table.read_text())
        self.assertIsNotNone(self.state()['active_request'])

    def test_missing_uptake_receipt_is_not_published_as_fully_applied(self):
        args=self.apply_args()
        self.settings(apply=True)
        self.executor_settings(missing_uptake=True)
        self.call(*args)
        self.append_step(20)
        self.call(*args, expect=2)
        self.assertIn('application: pending', self.table.read_text())
        self.assertNotIn('application: fully_applied', self.table.read_text())
        self.assertEqual(self.state()['status'], 'needs_reconciliation')

    def test_receipt_does_not_coerce_structured_values_into_uptake_proof(self):
        args=self.apply_args()
        self.settings(apply=True)
        self.executor_settings(malformed_fields=True)
        self.call(*args)
        self.append_step(20)
        self.call(*args, expect=2)
        self.assertNotIn('application: fully_applied', self.table.read_text())
        self.assertEqual(self.state()['status'], 'needs_reconciliation')
        self.assertEqual(len(self.calls('executor-calls.jsonl')), 1)
        self.call(*args)
        self.assertEqual(self.calls()[-1]['phase'], 'reconcile')
        self.assertEqual(len(self.calls('executor-calls.jsonl')), 1)

    def test_pending_receipt_requires_explicit_unknown_fields(self):
        args=self.apply_args()
        self.settings(apply=True)
        self.executor_settings(incomplete_pending=True)
        self.call(*args)
        self.append_step(20)
        self.call(*args, expect=2)
        self.assertEqual(self.state()['status'], 'needs_reconciliation')
        self.assertFalse(list((self.root/'reports/supervision/RUN/requests').glob('*/receipt.json')))

    def test_complete_pending_receipt_keeps_unconfirmed_uptake_unknown(self):
        args=self.apply_args()
        self.settings(apply=True)
        self.executor_settings(pending=True)
        self.call(*args)
        self.append_step(20)
        self.call(*args)
        self.assertEqual(self.state()['status'], 'watching')
        receipts=list((self.root/'reports/supervision/RUN/requests').glob('*/receipt.json'))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(json.loads(receipts[0].read_text())['uptake'], '')
        self.assertNotIn('application: fully_applied', self.table.read_text())

    def test_timeout_after_gate_write_reconciles_without_repeating_operation(self):
        args=self.apply_args()
        self.settings(apply=True)
        self.call(*args, '--executor-timeout', '0.2')
        self.executor_settings(sleep_after_write=5)
        self.append_step(20)
        self.call(*args, '--executor-timeout', '0.2', expect=2)
        self.assertEqual(json.loads((self.root/'control.json').read_text()), {'version': 2})
        self.assertEqual(len(self.calls('executor-calls.jsonl')), 1)
        self.call(*args, '--executor-timeout', '0.2')
        self.assertEqual(self.calls()[-1]['phase'], 'reconcile')
        self.assertIsNotNone(self.calls()[-1]['context']['previous_request'])
        self.assertEqual(len(self.calls('executor-calls.jsonl')), 1)
        self.assertIsNone(self.state()['active_request'])

    def test_failed_application_receipt_is_a_completed_review(self):
        args=self.apply_args()
        self.settings(apply=True)
        self.executor_settings(fail=True)
        self.call(*args)
        self.append_step(20)
        self.call(*args)
        self.assertIn('application: failed', self.table.read_text())
        self.assertEqual(self.state()['status'], 'watching')
        self.assertEqual(json.loads((self.root/'control.json').read_text()), {'version': 1})

    def test_agent_failure_does_not_complete_discovery_or_auto_retry(self):
        self.settings(fail=True)
        self.call(expect=2)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse(self.state()['discovery_complete'])
        self.assertIsNotNone(self.state()['active_request'])
        self.assertFalse(self.table.exists())
        self.settings()
        self.call()
        self.assertEqual(self.calls()[-1]['phase'], 'discovery')
        self.assertIsNotNone(self.calls()[-1]['context']['previous_request'])

    def test_invalid_discovery_never_unlocks_monitoring(self):
        self.settings(empty_map=True)
        self.call(expect=2)
        self.assertFalse(self.state()['discovery_complete'])
        self.assertFalse(self.table.exists())

    def test_review_must_report_trajectory_evidence_or_absence(self):
        self.call()
        self.settings(missing_trajectories=True)
        self.append_step(20)
        self.call(expect=2)
        self.assertIsNotNone(self.state()['active_request'])

    def test_budget_counts_failed_agent_calls_and_never_claims_run_completed(self):
        self.settings(fail=True)
        self.call('--max-calls', '1', expect=2)
        self.settings()
        self.call('--max-calls', '1')
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.state()['status'], 'budget_exhausted')
        self.assertIsNotNone(self.state()['active_request'])

    def test_final_review_is_read_only_and_idempotent(self):
        self.call('--stop-step', '40')
        self.append_step(40)
        self.call('--stop-step', '40')
        self.assertEqual([x['phase'] for x in self.calls()], ['discovery','final'])
        self.assertEqual(self.state()['status'], 'completed')
        self.call('--stop-step', '40')
        self.assertEqual(len(self.calls()), 2)
        self.assertFalse(self.calls('executor-calls.jsonl'))

    def test_completion_marker_and_idle_handoff(self):
        marker=self.root/'done'
        self.call('--completion-file', str(marker))
        marker.touch()
        self.call('--completion-file', str(marker))
        self.assertEqual(self.calls()[-1]['phase'], 'final')

    def test_idle_timeout_hands_off_supervision_only(self):
        args=parser().parse_args([*self.base, '--idle-timeout', '60'])
        now=[100.0]
        supervisor=Supervisor(args, clock=lambda:now[0])
        supervisor.load()
        supervisor.tick()
        now[0]=161.0
        self.assertTrue(supervisor.tick())
        self.assertEqual(supervisor.state['status'], 'idle_handoff')
        self.assertEqual(len(self.calls()), 1)

    def test_frequent_timer_reviews_cannot_preempt_idle_handoff(self):
        args=parser().parse_args([*self.base, '--every-seconds', '5', '--idle-timeout', '10'])
        now=[100.0]
        supervisor=Supervisor(args, clock=lambda:now[0])
        supervisor.load()
        supervisor.tick()
        now[0]=105.0
        supervisor.tick()
        self.assertEqual(len(self.calls()), 2)
        now[0]=110.0
        self.assertTrue(supervisor.tick())
        self.assertEqual(supervisor.state['status'], 'idle_handoff')
        self.assertEqual(len(self.calls()), 2)

    def test_custom_agent_file_protocol_and_default_codex_argv(self):
        self.settings(file_response=True)
        self.call()
        self.assertEqual(len(self.calls()), 1)
        # Reuse the fake agent as a fake Codex binary to verify the real adapter arguments.
        binary=self.root/'bin'
        binary.mkdir()
        codex=binary/'codex'
        codex.write_text('#!'+sys.executable+'\n'+AGENT)
        codex.chmod(0o755)
        self.env['PATH']=str(binary)+os.pathsep+self.env.get('PATH','')
        base=self.base.copy()
        index=base.index('--agent-command')
        del base[index:index+2]
        self.call('--adopt-settings', base=base)
        command=self.calls()[-1]['argv']
        self.assertEqual(command[0], 'exec')
        self.assertEqual(command[command.index('--sandbox')+1], 'read-only')
        self.assertIn('--output-schema', command)
        self.assertIn('--output-last-message', command)
        self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', command)

    def test_record_paths_cannot_escape_reports(self):
        self.contract['records']={'change_log':'../other/change_log.md'}
        self.contract_path.write_text(json.dumps(self.contract))
        self.call(expect=2)
        self.assertFalse(self.calls())

    def test_running_child_and_controller_lock_prevent_overlapping_reviews(self):
        self.settings(sleep=5)
        command=[sys.executable, str(SCRIPTS/'supervise.py'), *self.base, '--agent-timeout','10']
        process=subprocess.Popen(command, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        deadline=time.monotonic()+4
        while not (self.root/'agent-pid.txt').exists() and time.monotonic()<deadline:
            time.sleep(0.02)
        self.assertTrue((self.root/'agent-pid.txt').exists())
        self.call(expect=2)
        # A controller crash must leave its child holding the run lock until that child exits.
        process.kill()
        process.wait(timeout=3)
        self.call(expect=2)
        active=self.state()['active_request']
        os.killpg(active['child_pid'], signal.SIGTERM)
        self.settings()
        deadline=time.monotonic()+3
        while pid_start(active['child_pid']) and time.monotonic()<deadline:
            status=Path(f"/proc/{active['child_pid']}/stat").read_text().rsplit(') ',1)[1].split()[0]
            if status=='Z':
                break
            time.sleep(0.02)
        # A zombie cannot perform more work; the OS releases the inherited lock.
        self.call()
        self.assertIsNotNone(self.calls()[-1]['context']['previous_request'])

    def test_timeout_cleans_only_its_owned_agent_process(self):
        self.settings(sleep=5)
        self.call('--agent-timeout','0.2', expect=2)
        pid=int((self.root/'agent-pid.txt').read_text())
        try:
            status=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[0]
        except FileNotFoundError:
            status='gone'
        self.assertIn(status, ('gone','Z'))
        self.assertFalse(self.state()['discovery_complete'])

    def test_sigterm_stops_only_owned_child_and_records_handoff(self):
        self.settings(sleep=5)
        command=[sys.executable, str(SCRIPTS/'supervise.py'), *self.base, '--agent-timeout','10']
        process=subprocess.Popen(command, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        deadline=time.monotonic()+4
        while not (self.root/'agent-pid.txt').exists() and time.monotonic()<deadline:
            time.sleep(0.02)
        self.assertTrue((self.root/'agent-pid.txt').exists())
        process.terminate()
        self.assertEqual(process.wait(timeout=5), 130)
        self.assertEqual(self.state()['status'], 'interrupted')
        self.assertFalse(self.state()['supervisor']['running'])
        pid=int((self.root/'agent-pid.txt').read_text())
        self.assertIsNone(pid_start(pid))


if __name__=='__main__':
    unittest.main()
