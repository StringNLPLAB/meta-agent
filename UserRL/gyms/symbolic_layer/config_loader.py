"""
Hot-reloadable JSON config loader for the symbolic layer.

The config file is stat'ed on every access; when its (mtime, size) stamp
changes the file is re-read and parsed, and only after a full successful
parse is the config atomically swapped in. If re-reading or parsing fails,
the last successfully loaded config is kept and a warning is printed.

Thread-safe: all state transitions happen under a single lock, and the
config object is replaced (never mutated in place), so readers always see a
consistent snapshot.
"""

import json
import os
import threading
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# USERRL_META_ARM -- which experimental arm this run belongs to.
#
# Reserved for the planned three-arm comparison (see
# InfoPO/reports/TODO_arms.md). The switch lives HERE, at the loader, rather
# than in the meta-agent's discipline, because "I will remember not to write
# the hot layer during this run" is not a control -- it is an intention, and
# this session already recorded what happens to controls that depend on
# somebody remembering.
#
#   "online" (default)  hot layers are re-read on every access. Current design:
#                       the meta-agent can change the environment MID-run.
#   "frozen"            hot layers are read ONCE, at first access, and never
#                       re-stat'ed. The arm may still configure everything
#                       BEFORE the run; mid-run writes are silently inert.
#   "off"               hot layers are never read at all; built-in defaults
#                       only. The closest thing to plain PPO on this stack.
#
# Read once at import: an arm that could change mid-run would defeat its own
# purpose.
_META_ARM = os.environ.get("USERRL_META_ARM", "online").strip().lower()


class ConfigLoader:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._config: Optional[Dict[str, Any]] = None
        self._stamp: Optional[Tuple[int, int]] = None
        # off 臂**完全不读 JSON**，只用内置默认值（= 空字典，各处回落到 gym 原值）。
        # 之前这里无条件加载、get() 再 `return dict(self._config or {})`，
        # 于是 off 臂返回的是**文件内容**而不是默认值——一个名实不符的对照臂
        # 比没有对照臂更糟，所以在构造处就切断。
        if _META_ARM == "off":
            self._config = {}
            return
        # fail fast at startup if the config cannot be loaded at all
        self._reload_locked(force=True)

    def _file_stamp(self) -> Tuple[int, int]:
        stat = os.stat(self._path)
        return (stat.st_mtime_ns, stat.st_size)

    def _reload_locked(self, force: bool = False) -> None:
        """Reload if the file changed. Caller must hold the lock."""
        try:
            stamp = self._file_stamp()
        except OSError as e:
            if self._config is None:
                raise
            print(f"[symbolic_layer] cannot stat {self._path} ({e}); keeping last good config")
            return
        if not force and stamp == self._stamp:
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                new_config = json.load(f)
            if not isinstance(new_config, dict):
                raise ValueError("config root must be a JSON object")
        except Exception as e:
            if self._config is None:
                raise
            # remember the stamp so we only retry when the file changes again
            self._stamp = stamp
            print(f"[symbolic_layer] failed to reload {self._path} ({e}); keeping last good config")
            return
        # parse succeeded: atomically swap
        self._config = new_config
        self._stamp = stamp

    def get(self) -> Dict[str, Any]:
        if _META_ARM == "off":
            return {}
        if _META_ARM == "frozen" and self._config is not None:
            return self._config
        """Return the current config snapshot, hot-reloading if changed."""
        with self._lock:
            self._reload_locked()
            return self._config

    def reload(self) -> None:
        """Explicit reload (normally unnecessary — get() auto-detects)."""
        with self._lock:
            self._reload_locked(force=True)


_DEFAULT_LOADER: Optional[ConfigLoader] = None
_DEFAULT_LOADER_LOCK = threading.Lock()


def get_loader() -> ConfigLoader:
    """Return the process-wide loader for the bundled config.json."""
    global _DEFAULT_LOADER
    with _DEFAULT_LOADER_LOCK:
        if _DEFAULT_LOADER is None:
            _DEFAULT_LOADER = ConfigLoader(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            )
        return _DEFAULT_LOADER


def get_config() -> Dict[str, Any]:
    """Return the current symbolic-layer config (hot-reloaded on change)."""
    return get_loader().get()
