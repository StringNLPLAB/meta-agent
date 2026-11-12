"""Offline protocol fixture: a production reviewer runs the configured agent here."""
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
parser.add_argument('--response', required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
context = json.loads(Path(request['context_path']).read_text())
if context.get('run') != request['run'] or context.get('source_scope') != 'training_only':
    raise ValueError('Reviewer context must bind the current training run')
# Simulate a slower review to demonstrate that training can keep progressing.
# This is a deterministic integration fixture, not an actual model diagnosis.
time.sleep(0.15)
response = dict(schema_version=1, request_id=request['id'], run=request['run'],
                step=request['step'], decision='apply', proposal={'learning_rate': 0.0008})
Path(args.response).write_text(json.dumps(response) + '\n')
