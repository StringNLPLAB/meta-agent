"""Bounded snapshots of explicitly selected training artifacts for a fresh audit.

This is an input contract, not a filesystem sandbox. The caller is responsible
for supplying training-only files and enforcing the reviewer's access boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from meta_agent.nonblocking_audit import training_metrics


_FORBIDDEN = re.compile(r'^(?:eval(?:uation)?|test(?:ing)?|val(?:idation)?|checkpoints?|models?|credentials?|secrets?)(?:[_.-]|$)', re.I)
_LIMITS = dict(logs=16384, trajectories=65536, metrics=32768, experience=16384)


def _selected_path(root: Path, name, kind):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Evidence paths must be explicit relative training paths')
    if any(_FORBIDDEN.match(part) or part in {'.git', '.secrets'} for part in relative.parts):
        raise ValueError('Evidence cannot name evaluation, model or credential artifacts')
    path = root / relative
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError('Evidence requires a regular canonical file without symlinks')
    if kind == 'experience' and path.suffix not in {'.md', '.txt'}:
        raise ValueError('Prior experience must be a selected prose document, never executable code')
    return path


def _window(path, limit, *, tail):
    with path.open('rb') as stream:
        before = path.stat()
        start = max(0, before.st_size - limit) if tail else 0
        stream.seek(start)
        raw = stream.read(limit)
    return dict(source_size_bytes=before.st_size, source_mtime_ns=before.st_mtime_ns,
                byte_start=start, byte_end=start + len(raw),
                truncated=start > 0 or start + len(raw) < before.st_size,
                window_sha256=hashlib.sha256(raw).hexdigest(), full_file_hashed=False,
                text=raw.decode('utf-8', errors='replace'))


def build_training_context(*, root, run, logs=(), trajectories=(), metrics=(), experience=(),
                           max_packet_bytes=262144):
    """Read only the named byte windows; a later invocation gets a new snapshot.

    ``experience`` contains prose selected from completed earlier work. Passing
    a document carries knowledge forward; this function never copies a policy,
    optimizer, live configuration, executable patch or agent session.
    """
    root = Path(root)
    if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError('The training evidence root must be an absolute canonical directory')
    if not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run):
        raise ValueError('An explicit run identity is required')
    if type(max_packet_bytes) is not int or not 1024 <= max_packet_bytes <= 1048576:
        raise ValueError('The evidence packet needs a bounded byte budget')
    packet = dict(schema_version=1, run=run, source_scope='training_only', artifacts={},
                  knowledge_transfer='Selected prose documents only; no automatic executable or session inheritance',
                  access_semantics='Selected training inputs; not an operating-system isolation boundary')
    for kind, names in dict(logs=logs, trajectories=trajectories, metrics=metrics, experience=experience).items():
        names = list(names)
        if len(names) > 8: raise ValueError('Select at most eight artifacts per category')
        records = []
        for name in names:
            path = _selected_path(root, name, kind)
            window = _window(path, _LIMITS[kind], tail=kind != 'experience')
            if kind == 'metrics':
                rows = []
                lines = window['text'].splitlines()
                if window['byte_start'] > 0:
                    lines = lines[1:]
                for line in lines:
                    try: row = json.loads(line)
                    except ValueError: continue
                    if isinstance(row, dict) and isinstance(row.get('metrics'), dict):
                        rows.append(dict(step=row.get('step'), metrics=training_metrics(row['metrics'])))
                window.pop('text')
                window['rows'] = rows
            records.append(dict(path=Path(name).as_posix(), **window))
        packet['artifacts'][kind] = records
    if len(json.dumps(packet, ensure_ascii=False).encode()) > max_packet_bytes:
        raise ValueError('Selected evidence exceeds the total packet budget; select smaller inputs')
    return packet
