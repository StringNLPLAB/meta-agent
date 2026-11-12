"""Example project gate: validate one scalar and atomically publish its JSON."""
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
parser.add_argument('--response', required=True)
parser.add_argument('--publication', required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
response = json.loads(Path(args.response).read_text())
if (response.get('schema_version') != 1 or response.get('request_id') != request['id']
        or response.get('run') != request['run'] or response.get('step') != request['step']
        or response.get('decision') != 'apply'):
    raise ValueError('Proposal envelope differs from the bound request')
before = request['state_hashes']
if len(before) != 1: raise ValueError('This example gate owns exactly one JSON control')
target = Path(next(iter(before)))
if not target.is_absolute() or target.resolve(strict=True) != target or not target.is_file():
    raise ValueError('The example control must be a canonical regular file')
def hashes():
    return {str(target): hashlib.sha256(target.read_bytes()).hexdigest()}
observed = hashes()
state = json.loads(Path(request['coordinator_state_path']).read_text())
proposal = response.get('proposal', {})
value = proposal.get('learning_rate')
valid = (set(proposal) == {'learning_rate'} and type(value) in (int, float) and 0 < value <= 0.001)
status = 'rejected'
if observed != before:
    status = 'stale'
elif (valid and state.get('run') == request['run'] and state.get('publication_allowed')
      and not request['is_final']):
    temporary = target.with_name(target.name + '.next')
    temporary.write_text(json.dumps(proposal) + '\n')
    # The coordinator already holds commit.lock. Do not acquire it again here.
    # Recheck both the state and live hash immediately before atomic replacement.
    current = json.loads(Path(request['coordinator_state_path']).read_text())
    if current.get('publication_allowed') and hashes() == before:
        os.replace(temporary, target)
        status = 'applied'
    else:
        temporary.unlink(missing_ok=True)
        status = 'stale'
receipt = dict(schema_version=1, request_id=request['id'], run=request['run'], status=status,
               before_state_hashes=before, observed_before_state_hashes=observed, after_state_hashes=hashes())
Path(args.publication).write_text(json.dumps(receipt) + '\n')
