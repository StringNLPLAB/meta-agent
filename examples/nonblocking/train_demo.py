"""Run a tiny offline trainer/reviewer/publisher demonstration with no model calls."""
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from meta_agent.audit_evidence import build_training_context
from meta_agent.nonblocking_audit import NonBlockingAudit


def main():
    with tempfile.TemporaryDirectory(prefix='meta-agent-audit-demo-') as directory:
        root = Path(directory)
        live = root / 'control.json'
        live.write_text(json.dumps({'learning_rate': 0.001}) + '\n')
        (root / 'train.log').write_text('Offline example: the trainer reloads its scalar at each update.\n')
        (root / 'experience.md').write_text('Completed earlier experiment: assess one learning-rate change at a time.\n')
        context = root / 'context.json'
        context.write_text(json.dumps(build_training_context(root=root, run='demo',
            logs=['train.log'], experience=['experience.md'])))
        examples = ROOT / 'examples/nonblocking'
        audit = NonBlockingAudit(run='demo', directory=root / 'audits', context_path=context,
            mutable_paths=[live], every=2, first=1, stop=12,
            command=[sys.executable, str(examples / 'reviewer.py'), '--request', '{request}', '--response', '{response}'],
            publish_command=[sys.executable, str(examples / 'publisher.py'), '--request', '{request}',
                             '--response', '{response}', '--publication', '{publication}'])
        updates = []
        try:
            for step in range(1, 13):
                audit.before_step(step, is_final=step == 12)
                # A real consumer pins its effective controls for a whole update.
                effective = json.loads(live.read_text())['learning_rate']
                updates.append(dict(step=step, learning_rate=effective))
                time.sleep(0.04)
                audit.on_step(step, {'train/loss': 1 / step}, is_final=step == 12)
        finally:
            audit.close(timeout=1)
        print(json.dumps(dict(offline_fixture=True, optimizer_updates=12, model_calls=0,
            updates=updates, audits_submitted=audit.state['submitted'],
            requests_coalesced=audit.state['coalesced'], final_read_only=True)))


if __name__ == '__main__': main()
