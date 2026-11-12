"""Non-blocking direct-file editor with natural-language completion.

The process/session owner, evidence snapshots and trigger cadence are shared
with the proposal coordinator. A project still owns editor isolation, writable
mounts, schemas, validation, and update-level consumer pinning.
"""
from __future__ import annotations

import json
from pathlib import Path

from meta_agent.audit_files import _directory, live_census
from meta_agent.nonblocking_audit import NonBlockingAudit, _atomic_json


class DirectFileAudit(NonBlockingAudit):
    """Run one direct editor at a time; never interpret prose as a JSON proposal."""

    def __init__(self, *, editable_roots, **kwargs):
        self.editable_roots = [_directory(path) for path in editable_roots]
        if not self.editable_roots or len(set(self.editable_roots)) != len(self.editable_roots):
            raise ValueError('Declare distinct editable directory roots')
        audit_dir = Path(kwargs['directory']).absolute()
        context = Path(kwargs['context_path']).absolute()
        for root in self.editable_roots:
            if audit_dir.is_relative_to(root) or root.is_relative_to(audit_dir) or context.is_relative_to(root):
                raise ValueError('Audit records and the trusted context must be outside editable roots')
            if any(other != root and root.is_relative_to(other) for other in self.editable_roots):
                raise ValueError('Editable roots must not overlap')
        if kwargs.pop('publish_command', None) or kwargs.pop('mutable_paths', None):
            raise ValueError('Direct editing cannot also configure a proposal publisher')
        self._inventory = self._census()
        super().__init__(mutable_paths=[], publish_command=[], **kwargs)
        self.state.update(mode='direct_files', publication_allowed=False, agent_writes_allowed=True,
                          observed_change_sets=0, file_observation_is_worker_uptake=False,
                          learning_benefit='unverified', final_review_outcome=None)
        _atomic_json(self.directory / 'initial_live_files.json', self._inventory)
        self._save()

    def _census(self):
        result = live_census(self.editable_roots, maximum_files=4096,
                             maximum_file_bytes=16777216, maximum_total_bytes=134217728)
        if not result['complete']:
            raise ValueError('Live file observation was incomplete')
        return {name: dict(kind=row['kind'], bytes=row.get('bytes'), sha256=row.get('sha256'))
                for name, row in result['files'].items()}

    def _observe_files(self, reason):
        try:
            after = self._census()
        except (OSError, ValueError) as error:
            self._event('live_census_incomplete', reason=reason, error_type=type(error).__name__)
            self.state['last_census_complete'] = False
            return
        changes = {name: dict(before=self._inventory.get(name), after=after.get(name))
                   for name in sorted(set(self._inventory) | set(after))
                   if self._inventory.get(name) != after.get(name)}
        self.state['last_census_complete'] = True
        if changes:
            self.state['observed_change_sets'] += 1
            row = dict(run=self.run, observed_after_completed_step=self.state['latest_step'],
                       request_id=self._active['request']['id'] if self._active else None,
                       reason=reason, actual_file_changes=changes,
                       file_write_is_worker_uptake=False, learning_benefit='unverified')
            with (self.directory / 'direct_edits.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        self._inventory = after

    def _spawn(self, command, phase):
        active = self._active
        active['before_inventory'] = dict(self._inventory)
        active['request'].update(mode='direct_files', editable_roots=list(map(str, self.editable_roots)),
                                 agent_writes_allowed=True, final_response='natural_language')
        _atomic_json(active['directory'] / 'request.json', active['request'])
        super()._spawn(command, phase)

    def _start_pending(self):
        if self._pending and self._pending['is_final']:
            self.state['final_review_outcome'] = 'skipped_direct_editor_closed'
            self._event('final_editor_not_dispatched', step=self._pending['step'])
            self._pending = None
            self.state['pending_step'] = None
            return
        super()._start_pending()

    def _finish_review(self, active, code):
        self._kill_child()
        self._observe_files('editor_exit')
        changed = self._inventory != active.get('before_inventory', self._inventory)
        cancelled = active.get('cancelled_at_close', False)
        _atomic_json(active['directory'] / 'direct_receipt.json', dict(
            schema_version=1, request_id=active['request']['id'], run=self.run,
            returncode=code, cancelled_at_close=cancelled, observed_files_changed=changed,
            observation_complete=self.state.get('last_census_complete', True),
            file_write_is_worker_uptake=False, learning_benefit='unverified',
            response_semantics='Native stdout/stderr retained; no JSON response is required'))
        self._release_child()
        if cancelled:
            self.state['cancelled'] += 1
            self._event('direct_editor_cancelled_at_close', files_changed=changed)
        elif code:
            self._fail(f'Direct editor exited {code}; inspect its native output',
                       fatal=changed or not self.state.get('last_census_complete', True))
        else:
            self.state['completed'] += 1
            self._event('direct_editor_completed', files_changed=changed,
                        worker_uptake_verified=False, learning_benefit='unverified')

    def _close_publication(self):
        was_closing = self._closing
        super()._close_publication()
        self.state['agent_writes_allowed'] = False
        self._save()
        if not was_closing and self._active is not None:
            active = self._active
            # Direct reviewers are writers. Reap the whole owned editor group
            # before allowing the final rollout/update to begin.
            active['cancelled_at_close'] = True
            self._kill_child()
            code = active['process'].wait(timeout=5)
            self._finish_review(active, code)
            self._save()

    def poll(self, *, start_pending=True):
        self._observe_files('training_boundary')
        super().poll(start_pending=start_pending)
