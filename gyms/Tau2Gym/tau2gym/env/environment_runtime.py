"""Load one immutable training-environment mechanism for an episode.

Only the live pointer is mutable. Modules, relative imports and configuration
are loaded from verified snapshot bytes; existing episode instances retain
their implementation when another version is published. This is an integrity
boundary for trusted experiment code, not a Python security sandbox.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
from datetime import datetime, timezone

try:
    from . import usersim_runtime as verified
except ImportError:  # Preparation and gate probes import from the env directory.
    import usersim_runtime as verified


class EnvironmentBundleError(ValueError):
    """The environment pointer, snapshot or mechanism interface is invalid."""


_LOCK = threading.RLock()
_FACTORIES = {}
_POINTER_FIELDS = {'schema_version', 'run', 'version', 'bundle_sha256',
                   'snapshot', 'files', 'entrypoint'}


def _file_map(value):
    if not isinstance(value, dict) or not {'environment.py', 'environment_config.json'} <= value.keys():
        raise EnvironmentBundleError('Environment bundle requires environment.py and environment_config.json')
    for name, digest in value.items():
        if not isinstance(name, str):
            raise EnvironmentBundleError('Environment file paths must be strings')
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or relative.as_posix() != name
                or name == 'change.json' or relative.suffix not in verified._ALLOWED_SUFFIXES
                or any(not verified._COMPONENT.fullmatch(part) or part in {'.', '..'}
                       for part in relative.parts)):
            raise EnvironmentBundleError(f'Invalid environment bundle file path: {name!r}')
        if not isinstance(digest, str) or not verified._DIGEST.fullmatch(digest):
            raise EnvironmentBundleError(f'Invalid environment SHA256: {name}')
    return dict(value)


def _factory(digest, snapshot, sources):
    identity = (str(snapshot), digest)
    with _LOCK:
        if identity in _FACTORIES:
            return _FACTORIES[identity]
        location = hashlib.sha256(str(snapshot).encode()).hexdigest()
        package = '_eau_environment_' + digest + '_' + location
        loader = verified._BundleModules(package, snapshot, sources)
        sys.meta_path.insert(0, loader)
        try:
            entry = importlib.import_module(package + '.environment')
            factory = getattr(entry, 'EnvironmentMechanism', None)
            if not isinstance(factory, type) or any(
                    not callable(getattr(factory, name, None))
                    for name in ('on_reset', 'before_action', 'after_action')):
                raise EnvironmentBundleError('EnvironmentMechanism must be a class with all three episode hooks')
        except BaseException:
            sys.meta_path.remove(loader)
            for name in list(sys.modules):
                if name == package or name.startswith(package + '.'):
                    sys.modules.pop(name, None)
            raise
        _FACTORIES[identity] = factory
        return factory


def load_environment_bundle(pointer_path, expected_run=None):
    """Return ``(EnvironmentMechanism, settings, metadata)`` for one reset.

    The pointer must be ``<run>/live/environment_bundle.json`` and its snapshot
    ``<run>/environment_versions/versions/vNNNNNN``. Validation also runs on
    cache hits and failures never fall back to another mechanism.
    """
    value = os.fspath(pointer_path)
    if not os.path.isabs(value):
        value = str(Path.cwd()) + '/' + value
    try:
        pointer = verified._absolute(value, 'Environment bundle pointer')
        if pointer.name != 'environment_bundle.json' or pointer.parent.name != 'live':
            raise EnvironmentBundleError('Environment pointer must be <run>/live/environment_bundle.json')
        manifest = verified._json(verified._read_pointer(pointer), 'environment bundle pointer')
        if (not isinstance(manifest, dict) or set(manifest) != _POINTER_FIELDS
                or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1):
            raise EnvironmentBundleError('Invalid environment bundle pointer schema')
        run, version = manifest.get('run'), manifest.get('version')
        if (not isinstance(run, str) or not verified._COMPONENT.fullmatch(run)
                or run in {'.', '..'} or run != pointer.parent.parent.name
                or expected_run is not None and run != expected_run):
            raise EnvironmentBundleError('Environment bundle run does not match its pointer or expected run')
        if type(version) is not int or not 0 <= version <= 999999:
            raise EnvironmentBundleError('Environment version must be an integer between 0 and 999999')
        if manifest.get('entrypoint') != 'environment.py':
            raise EnvironmentBundleError('Environment entrypoint must be environment.py')
        snapshot = verified._absolute(manifest.get('snapshot'), 'Environment snapshot')
        required = pointer.parent.parent / 'environment_versions/versions' / f'v{version:06d}'
        if snapshot != required:
            raise EnvironmentBundleError('Environment snapshot must be the matching version in this run')
        files = _file_map(manifest['files'])
        digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if manifest.get('bundle_sha256') != digest:
            raise EnvironmentBundleError('Environment bundle manifest SHA256 mismatch')
        sources, change_sha256 = verified._snapshot_bytes(snapshot, files)
        settings = verified._json(sources['environment_config.json'], 'environment_config.json')
        if (not isinstance(settings, dict) or type(settings.get('version')) is not int
                or settings['version'] != version):
            raise EnvironmentBundleError('environment_config.version must equal the bundle version')
        try:
            json.dumps(settings, allow_nan=False)
        except ValueError as error:
            raise EnvironmentBundleError('Environment configuration must contain finite JSON values') from error
        factory = _factory(digest, snapshot, sources)
    except (OSError, verified.UserSimBundleError) as error:
        raise EnvironmentBundleError(f'Cannot load environment bundle: {error}') from error
    except EnvironmentBundleError:
        raise
    except Exception as error:
        raise EnvironmentBundleError(f'Cannot initialize environment mechanism: {error}') from error
    metadata = {'run': run, 'environment_bundle_version': version,
                'environment_bundle_sha256': digest, 'environment_snapshot': str(snapshot),
                'environment_files': files, 'environment_entrypoint': 'environment.py',
                'environment_code_sha256': files['environment.py'],
                'environment_config_sha256': files['environment_config.json']}
    if change_sha256 is not None:
        metadata['environment_change_sha256'] = change_sha256
    return factory, copy.deepcopy(settings), metadata


def record_environment_uptake(env, metadata):
    directory = os.environ.get('EAU_ENVIRONMENT_UPTAKE_DIR')
    if not directory:
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    record = {**metadata, 'time': datetime.now(timezone.utc).isoformat(),
              'episode_id': env.episode_id, 'task_id': str(env.task.id),
              'domain': env.config.domain, 'pid': os.getpid(),
              'process_reward_weights': copy.deepcopy(env._episode_process_weights),
              'max_steps': env._episode_max_steps,
              'rank_environment': {key: os.environ[key] for key in
                  ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'RAY_WORKER_ID', 'RAY_NODE_ID', 'RAY_RANK')
                  if key in os.environ}}
    payload = (json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    descriptor = os.open(Path(directory) / f'{os.getpid()}.jsonl',
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        remaining = memoryview(payload)
        while remaining:
            size = os.write(descriptor, remaining)
            if size <= 0:
                raise OSError('Environment uptake append made no progress')
            remaining = remaining[size:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
