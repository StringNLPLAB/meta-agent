"""Load an immutable, verified user-simulator bundle for one episode.

The live pointer is read once per call. Python modules execute from the bytes
verified in that call, under a package bound to its snapshot and bundle digest.
Existing users therefore keep their original classes when the pointer changes.
This loader verifies artifact integrity; it is not a sandbox for Python code.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import threading
from typing import Any, Callable


class UserSimBundleError(ValueError):
    """The live pointer or its immutable snapshot failed validation."""


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_ALLOWED_SUFFIXES = {".py", ".json", ".txt", ".jinja"}
_LOCK = threading.RLock()
_FACTORIES: dict[tuple[str, str], Callable[..., Any]] = {}


def _json(raw: bytes, label: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise UserSimBundleError(f"Duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UserSimBundleError(f"Invalid JSON in {label}: {exc}") from exc


def _absolute(value: str, label: str) -> Path:
    if (not isinstance(value, str) or not value or "\\" in value
            or "\x00" in value or not value.startswith("/")
            or PurePosixPath(value).as_posix() != value
            or ".." in PurePosixPath(value).parts):
        raise UserSimBundleError(f"{label} must be a canonical absolute path")
    return Path(value)


def _open_directory(path: Path) -> int:
    """Walk using directory descriptors so no path component follows a link."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_at(directory: int, name: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise UserSimBundleError(f"Bundle file must be a regular file: {name}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _read_pointer(path: Path) -> bytes:
    descriptor = _open_directory(path.parent)
    try:
        return _read_at(descriptor, path.name)
    finally:
        os.close(descriptor)


def _file_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise UserSimBundleError("Bundle files must be a nonempty JSON object")
    for name, digest in value.items():
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or relative.as_posix() != name
                or any(not _COMPONENT.fullmatch(part) or part in {".", ".."}
                       for part in relative.parts)
                or relative.suffix not in _ALLOWED_SUFFIXES):
            raise UserSimBundleError(f"Invalid bundle file path: {name!r}")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise UserSimBundleError(f"Invalid SHA256 for bundle file: {name}")
    if not {"flow_user.py", "env_policy.json"}.issubset(value):
        raise UserSimBundleError("Bundle requires flow_user.py and env_policy.json")
    return dict(value)


def _snapshot_bytes(snapshot: Path, expected: dict[str, str]):
    actual: dict[str, bytes] = {}
    change_sha256 = None
    directories = {str(parent) for name in expected
                   for parent in PurePosixPath(name).parents if str(parent) != "."}

    def visit(descriptor: int, prefix: str = "") -> None:
        nonlocal change_sha256
        for name in sorted(os.listdir(descriptor)):
            relative = f"{prefix}{name}"
            mode = os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                if relative not in directories:
                    raise UserSimBundleError(f"Unlisted bundle directory: {relative}")
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=descriptor)
                try:
                    visit(child, relative + "/")
                finally:
                    os.close(child)
            elif not stat.S_ISREG(mode):
                raise UserSimBundleError(f"Symlink or nonregular bundle entry: {relative}")
            elif relative == "change.json" and relative not in expected:
                # The gate owns this root transaction receipt. It is outside
                # the executable bundle digest and is never imported as code.
                change_sha256 = hashlib.sha256(_read_at(descriptor, name)).hexdigest()
            elif relative not in expected:
                raise UserSimBundleError(f"Unlisted bundle file: {relative}")
            else:
                raw = _read_at(descriptor, name)
                if hashlib.sha256(raw).hexdigest() != expected[relative]:
                    raise UserSimBundleError(f"Bundle file SHA256 mismatch: {relative}")
                actual[relative] = raw

    descriptor = _open_directory(snapshot)
    try:
        visit(descriptor)
    finally:
        os.close(descriptor)
    if actual.keys() != expected.keys():
        raise UserSimBundleError(f"Missing bundle files: {sorted(expected.keys() - actual.keys())}")
    return actual, change_sha256


class _BundleModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve only verified Python bytes, including imports made during reply()."""

    def __init__(self, package: str, snapshot: Path, sources: dict[str, bytes]):
        self.package = package
        self.snapshot = snapshot
        self.modules: dict[str, tuple[str | None, bool]] = {package: (None, True)}
        self.sources = sources
        for path in sources:
            relative = PurePosixPath(path)
            if relative.suffix != ".py":
                continue
            parts = list(relative.with_suffix("").parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            if any(not part.isidentifier() for part in parts):
                raise UserSimBundleError(f"Python helper is not importable: {path}")
            name = ".".join([package, *parts])
            if name in self.modules and self.modules[name][0] is not None:
                raise UserSimBundleError(f"Conflicting Python modules: {path}")
            self.modules[name] = (path, is_package)
            for length in range(1, len(parts)):
                self.modules.setdefault(".".join([package, *parts[:length]]), (None, True))
        # A module and a directory cannot both provide the same import name.
        for name, (_path, is_package) in self.modules.items():
            if not is_package and any(other.startswith(name + ".") for other in self.modules):
                raise UserSimBundleError(f"Conflicting Python package: {name}")

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.package and not fullname.startswith(self.package + "."):
            return None
        if fullname not in self.modules:
            raise ModuleNotFoundError(f"No verified bundle module named {fullname!r}", name=fullname)
        source, is_package = self.modules[fullname]
        origin = str(self.snapshot / source) if source else str(self.snapshot)
        spec = importlib.util.spec_from_loader(fullname, self, origin=origin,
                                              is_package=is_package)
        spec.has_location = True
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source, is_package = self.modules[module.__name__]
        if is_package:
            # Imports are resolved by this finder, never by a mutable sys.path.
            module.__path__ = []
        if source is not None:
            exec(compile(self.sources[source], str(self.snapshot / source), "exec", dont_inherit=True),
                 module.__dict__)


def _factory(digest: str, snapshot: Path, sources: dict[str, bytes]):
    # Identical bytes in separate runs must still have separate module paths:
    # __file__ and package-relative template reads belong to this snapshot.
    identity = (str(snapshot), digest)
    with _LOCK:
        if identity in _FACTORIES:
            return _FACTORIES[identity]
        location = hashlib.sha256(str(snapshot).encode("utf-8")).hexdigest()
        package = "_eau_usersim_" + digest + "_" + location
        loader = _BundleModules(package, snapshot, sources)
        sys.meta_path.insert(0, loader)
        try:
            entry = importlib.import_module(package + ".flow_user")
            factory = getattr(entry, "Tau2FlowUser", None)
            if not callable(factory):
                raise UserSimBundleError("flow_user.py must export callable Tau2FlowUser")
        except BaseException:
            sys.meta_path.remove(loader)
            for name in list(sys.modules):
                if name == package or name.startswith(package + "."):
                    sys.modules.pop(name, None)
            raise
        _FACTORIES[identity] = factory
        return factory


def load_usersim_bundle(pointer_path: str | os.PathLike[str], expected_run: str | None = None):
    """Return ``(Tau2FlowUser, policy, metadata)`` from one verified pointer.

    A snapshot must be ``<run>/versions/versions/vNNNNNN`` beside the run's
    ``live/usersim_bundle.json`` pointer. Validation runs even on cache hits.
    No errors fall back to an older simulator or an unrelated policy file.
    """
    pointer_string = os.fspath(pointer_path)
    if not os.path.isabs(pointer_string):
        pointer_string = str(Path.cwd()) + "/" + pointer_string
    pointer = _absolute(pointer_string, "Bundle pointer")
    if pointer.name != "usersim_bundle.json" or pointer.parent.name != "live":
        raise UserSimBundleError("Bundle pointer must be <run>/live/usersim_bundle.json")
    try:
        manifest = _json(_read_pointer(pointer), "bundle pointer")
        if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise UserSimBundleError("Unsupported user-simulator bundle schema")
        run = manifest.get("run")
        if (not isinstance(run, str) or not _COMPONENT.fullmatch(run)
                or run in {".", ".."} or pointer.parent.parent.name != run
                or (expected_run is not None and run != expected_run)):
            raise UserSimBundleError("Bundle run does not match the live pointer or expected run")
        version = manifest.get("version")
        if type(version) is not int or not 0 <= version <= 999999:
            raise UserSimBundleError("Bundle version must be an integer between 0 and 999999")
        snapshot = _absolute(manifest.get("snapshot"), "Bundle snapshot")
        required = pointer.parent.parent / "versions" / "versions" / f"v{version:06d}"
        if snapshot != required:
            raise UserSimBundleError("Bundle snapshot must be the matching version in the same run")
        if manifest.get("entrypoint") != "flow_user.py":
            raise UserSimBundleError("Bundle entrypoint must be flow_user.py")
        files = _file_map(manifest.get("files"))
        digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if manifest.get("bundle_sha256") != digest:
            raise UserSimBundleError("Bundle manifest SHA256 mismatch")
        sources, change_sha256 = _snapshot_bytes(snapshot, files)
        policy = _json(sources["env_policy.json"], "env_policy.json")
        if not isinstance(policy, dict) or not isinstance(policy.get("tau2_flow", {}), dict):
            raise UserSimBundleError("Bundle policy and tau2_flow must be JSON objects")
        try:
            factory = _factory(digest, snapshot, sources)
        except UserSimBundleError:
            raise
        except Exception as exc:
            raise UserSimBundleError(f"Cannot initialize user-simulator bundle: {exc}") from exc
    except OSError as exc:
        raise UserSimBundleError(f"Cannot read user-simulator bundle safely: {exc}") from exc
    policy_encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    metadata = {
        "run": run,
        "usersim_bundle_version": version,
        "usersim_bundle_sha256": digest,
        "usersim_snapshot": str(snapshot),
        "usersim_files": files,
        "usersim_entrypoint": "flow_user.py",
        "usersim_code_sha256": files["flow_user.py"],
        "usersim_config_sha256": files["env_policy.json"],
        "code_sha256": files["flow_user.py"],
        "config_version": policy.get("version"),
        "config_sha256": files["env_policy.json"],
        "policy_sha256": files["env_policy.json"],
        "policy_snapshot_sha256": hashlib.sha256(policy_encoded).hexdigest(),
        "policy_sha256_encoding": "raw config bytes",
        "config_path": str(snapshot / "env_policy.json"),
    }
    if change_sha256 is not None:
        metadata["usersim_change_sha256"] = change_sha256
    return factory, copy.deepcopy(policy), metadata
