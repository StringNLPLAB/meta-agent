"""Training input selection, bounded snapshots and document-only experience."""
import json
from pathlib import Path
import tempfile
import unittest

from meta_agent.audit_evidence import build_training_context


class AuditEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_selected_training_inputs_are_bounded_snapshots_with_provenance(self):
        (self.root / 'train.log').write_text('x' * 20000)
        (self.root / 'rollouts.jsonl').write_text('{"action":"ask"}\n')
        (self.root / 'experience.md').write_text('Earlier run: observed a delayed effect; retain the uncertainty.')
        packet = build_training_context(root=self.root, run='r2', logs=['train.log'],
            trajectories=['rollouts.jsonl'], experience=['experience.md'])
        log = packet['artifacts']['logs'][0]
        self.assertEqual(len(log['text']), 16384)
        self.assertEqual(log['source_size_bytes'], 20000)
        self.assertTrue(log['truncated'])
        self.assertEqual(log['byte_start'], 20000 - 16384)
        self.assertEqual(packet['source_scope'], 'training_only')
        original = json.dumps(packet)
        (self.root / 'experience.md').write_text('Changed after request construction.')
        self.assertEqual(json.dumps(packet), original)

    def test_metric_snapshot_filters_evaluation_namespaces_and_nonfinite_scalars(self):
        (self.root / 'metrics.jsonl').write_text(json.dumps(dict(step=1, metrics={
            'train/loss': .3, 'val-core/accuracy': 1, 'test/pass': 1,
            'evaluation/success': 1, 'train/nan': float('nan')})) + '\n')
        packet = build_training_context(root=self.root, run='r1', metrics=['metrics.jsonl'])
        window = packet['artifacts']['metrics'][0]
        self.assertNotIn('text', window)
        self.assertEqual(window['rows'], [dict(step=1, metrics={'train/loss': .3})])

    def test_executable_experience_evaluation_paths_traversal_and_symlinks_are_rejected(self):
        (self.root / 'experience.py').write_text('pass\n')
        (self.root / 'test_results.json').write_text('{}')
        (self.root / 'real.log').write_text('Training evidence')
        (self.root / 'link.log').symlink_to(self.root / 'real.log')
        for kwargs in [dict(experience=['experience.py']), dict(logs=['test_results.json']),
                       dict(logs=['../escape.log']), dict(logs=['link.log']),
                       dict(logs=[str(self.root / 'real.log')])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_training_context(root=self.root, run='r1', **kwargs)

    def test_oversized_selection_fails_instead_of_silently_expanding_context(self):
        (self.root / 'train.log').write_text('x' * 20000)
        with self.assertRaises(ValueError):
            build_training_context(root=self.root, run='r1', logs=['train.log'], max_packet_bytes=1024)
        with self.assertRaises(ValueError):
            build_training_context(root=self.root, run='r1', logs=['train.log'] * 9)


if __name__ == '__main__': unittest.main()
