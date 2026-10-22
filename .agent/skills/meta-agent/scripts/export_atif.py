#!/usr/bin/env python3
"""Conservatively export one supervisor request to ATIF, without Harbor dependencies.

Only a Codex session whose session_meta ID matches this invocation's thread.started
ID may be collected. OpenCode is converted solely from this request's stdout.
This is an interchange export, not a Harbor trial/reward runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

VERSION = "1"
ARTIFACTS = ("request.json", "prompt.txt", "agent.stdout", "agent.stderr", "agent.backend.json", "response.json",
             "operation.json", "receipt.json", "executor.stdout", "executor.stderr")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe(root, relative):
    """Refuse artifact symlinks, including directory components, even for reads."""
    path = root / relative
    if path != root and root not in path.parents:
        raise ValueError("artifact path must stay inside request directory")
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError(f"symlink artifact is not allowed: {candidate}")
    return path


def _read(root, name):
    path = _safe(root, name)
    return path.read_bytes() if path.is_file() else None


def _write(root, name, data):
    path = _safe(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".atif-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _json_object(data, name, warnings):
    if data is None:
        return None
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError):
        warnings.append(f"{name}: invalid/incomplete JSON retained as raw artifact")
        return None
    if not isinstance(value, dict):
        warnings.append(f"{name}: expected an object; retained as raw artifact")
        return None
    return value


def _events(data, name, warnings):
    result = []
    if data is None:
        return result
    for number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                result.append(event)
            else:
                warnings.append(f"{name}:{number}: non-object event retained as raw")
        except (ValueError, UnicodeError):
            warnings.append(f"{name}:{number}: incomplete/non-JSON line retained as raw")
    return result


def _strip_encrypted(value):
    if isinstance(value, dict):
        return {k: _strip_encrypted(v) for k, v in value.items() if k != "encrypted_content"}
    if isinstance(value, list):
        return [_strip_encrypted(v) for v in value]
    return value


def _session_id(events):
    identities = {e.get("payload", {}).get("id") for e in events
                  if e.get("type") == "session_meta" and isinstance(e.get("payload"), dict)}
    identities.discard(None)
    return next(iter(identities)) if len(identities) == 1 else None


def _native(root, thread_id, sessions_dir, warnings):
    """Prefer a validated private archive; never select shared sessions by mtime."""
    if not thread_id:
        return None, None
    # A digest filename is safe even when an untrusted backend supplies the ID.
    relative = "native/sessions/" + _sha(thread_id.encode()) + ".jsonl"
    archive = _read(root, relative)
    if archive is not None:
        events = _events(archive, relative, warnings)
        if _session_id(events) == thread_id:
            metadata = _json_object(_read(root, "native/capture.json"), "native/capture.json", warnings)
            if metadata and metadata.get("path") == relative and metadata.get("archived_sha256") == _sha(archive):
                warnings.extend(metadata.get("warnings", []))
                return events, metadata
            return events, {"path": relative, "native_session_id": thread_id,
                            "archived_sha256": _sha(archive), "capture": "filtered_native_session_snapshot",
                            "basis": "existing private archive; original source checksum unavailable"}
        warnings.append("archived native session identity mismatch; archive was not overwritten")
        return None, None
    if sessions_dir is None:
        warnings.append("native session collection not configured; using stdout fallback")
        return None, None
    source_root = Path(sessions_dir)
    if source_root.is_symlink() or not source_root.is_dir():
        warnings.append("native sessions directory unavailable or symlink; using stdout fallback")
        return None, None
    matches = []
    for candidate in sorted(source_root.rglob("*.jsonl")):
        if not candidate.is_file() or candidate.is_symlink() or any(p.is_symlink() for p in candidate.parents if p != source_root.parent):
            continue
        try:
            # Native Codex JSONL begins with session_meta. Inspect its bounded
            # header before reading a complete unrelated shared conversation.
            with candidate.open("rb") as handle:
                header = handle.readline(1024 * 1024)
                metadata = json.loads(header)
                if not isinstance(metadata, dict) or metadata.get("type") != "session_meta" or not isinstance(metadata.get("payload"), dict) or metadata["payload"].get("id") != thread_id:
                    continue
                matches.append(candidate)
        except (OSError, ValueError, UnicodeError):
            continue
    if len(matches) != 1:
        warnings.append(f"expected one native session for exact thread ID; found {len(matches)}; using stdout fallback")
        return None, None
    candidate = matches[0]
    try:
        original = candidate.read_bytes()
    except OSError:
        warnings.append("matched native source became unavailable; using stdout fallback")
        return None, None
    events = _events(original, "native source", [])
    if _session_id(events) != thread_id:
        warnings.append("matched native source has inconsistent session IDs; using stdout fallback")
        return None, None
    # Keep unsupported event types for later conversion, but never export opaque reasoning blobs.
    cleaned = [_strip_encrypted(event) for event in events]
    archived = b"".join((json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n").encode() for e in cleaned)
    capture_warnings = []
    if any("encrypted_content" in json.dumps(e) for e in events):
        capture_warnings.append("native encrypted_content omitted; only visible reasoning summaries can be converted")
    # Warn on partial JSONL input as well; the original bytes remain at the caller's source.
    _events(original, "native source", capture_warnings)
    capture_warnings = [w.replace("retained as raw", "omitted from filtered snapshot; original remains at source")
                        for w in capture_warnings]
    warnings.extend(capture_warnings)
    _write(root, relative, archived)
    source = {"path": relative, "native_session_id": thread_id, "archived_sha256": _sha(archived),
              "source_sha256": _sha(original), "source_path": str(candidate.absolute()),
              "capture": "filtered_native_session_snapshot", "encrypted_content_omitted": True,
              "basis": "exact session_meta.payload.id; parsed JSONL snapshot; opaque encrypted_content removed",
              "warnings": capture_warnings}
    _write(root, "native/capture.json", _json_bytes(source))
    return cleaned, source


def _timestamp(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            return None
    if not isinstance(value, str):
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value
    except ValueError:
        return None


def _step(steps, source, message="", *, timestamp=None, extra=None):
    step = {"step_id": len(steps) + 1, "source": source, "message": message}
    normalized = _timestamp(timestamp)
    if normalized:
        step["timestamp"] = normalized
    if extra:
        step["extra"] = extra
    steps.append(step)
    return step


def _text(content, warnings, location):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("input_text", "output_text", "text", "summary_text") and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                warnings.append(f"{location}: unsupported/multimodal content retained only in raw artifact")
        return "\n".join(parts)
    warnings.append(f"{location}: unsupported content retained only in raw artifact")
    return ""


def _arguments(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
        return {"input": value}
    return {}


def _call(steps, calls, identifier, name, arguments, timestamp=None, extra=None):
    if not isinstance(identifier, str) or not identifier:
        return None
    if identifier in calls:
        step = calls[identifier]
        if arguments:
            step["tool_calls"][0]["arguments"] = _arguments(arguments)
        if name:
            step["tool_calls"][0]["function_name"] = name
        return step
    step = _step(steps, "agent", timestamp=timestamp, extra=extra)
    step["tool_calls"] = [{"tool_call_id": identifier, "function_name": name,
                           "arguments": _arguments(arguments)}]
    calls[identifier] = step
    return step


def _output(step, identifier, content, extra=None):
    if step is None:
        return
    result = {"source_call_id": identifier, "content": content}
    if extra:
        result["extra"] = extra
    step["observation"] = {"results": [result]}


def _convert_native(events, steps, warnings):
    calls = {}
    model = None
    version = "unknown"
    for event in events:
        kind = event.get("type")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if kind == "session_meta":
            version = str(payload.get("cli_version") or "unknown")
        elif kind == "turn_context" and isinstance(payload.get("model"), str):
            model = payload["model"]
        elif kind == "response_item":
            item_type = payload.get("type")
            timestamp = event.get("timestamp")
            if item_type == "message":
                role = {"assistant": "agent", "user": "user", "developer": "system", "system": "system"}.get(payload.get("role"))
                if role:
                    message = _text(payload.get("content"), warnings, "native message")
                    if message:
                        if role == "user" and any(s["source"] == "user" and s["message"] == message
                                                  and s.get("extra", {}).get("artifact") == "prompt.txt" for s in steps):
                            continue
                        _step(steps, role, message, timestamp=timestamp,
                              extra={"native_type": "message", "channel": payload.get("channel")})
                else:
                    warnings.append("native message role unsupported; retained in raw artifact")
            elif item_type in ("function_call", "custom_tool_call"):
                identifier = payload.get("call_id")
                if not _call(steps, calls, identifier, str(payload.get("name") or item_type),
                             payload.get("arguments") if item_type == "function_call" else payload.get("input"),
                             timestamp, {"native_type": item_type}):
                    warnings.append("native tool call lacks call_id; retained in raw artifact")
            elif item_type in ("function_call_output", "custom_tool_call_output"):
                identifier = payload.get("call_id")
                if identifier not in calls:
                    warnings.append("native tool output has no matching recorded call; retained in raw artifact")
                else:
                    _output(calls[identifier], identifier, _text(payload.get("output"), warnings, "native tool output"))
            elif item_type == "reasoning":
                summary = _text(payload.get("summary"), warnings, "native reasoning summary")
                if summary:
                    step = _step(steps, "agent", timestamp=timestamp,
                                 extra={"native_type": "reasoning", "visibility": "summary"})
                    step["reasoning_content"] = summary
            else:
                warnings.append(f"native response_item type {item_type!r} unsupported; retained in raw artifact")
    return model, version


def _convert_stdout(events, steps, warnings):
    calls = {}
    items = {}
    usage = None
    for event in events:
        kind = event.get("type")
        item = event.get("item")
        if kind == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
        if kind not in ("item.started", "item.updated", "item.completed") or not isinstance(item, dict):
            if kind in ("error", "turn.failed"):
                warnings.append(f"stdout reports {kind}; trajectory may be partial")
            continue
        identifier = item.get("id")
        item_type = item.get("type")
        if item_type in ("agent_message", "reasoning"):
            if not isinstance(identifier, str):
                identifier = f"anonymous-message-{len(items)}"
            text = item.get("text")
            if not isinstance(text, str):
                warnings.append(f"stdout {item_type}: unsupported text retained as raw")
                continue
            if identifier not in items:
                items[identifier] = _step(steps, "agent", extra={"stdout_type": item_type, "item_id": identifier})
            if item_type == "reasoning":
                items[identifier]["reasoning_content"] = text
                items[identifier]["extra"]["visibility"] = "summary"
            else:
                items[identifier]["message"] = text
        elif item_type in ("command_execution", "mcp_tool_call"):
            if item_type == "command_execution":
                name, arguments = "command_execution", {"command": item["command"]} if "command" in item else {}
            else:
                name = str(item.get("server", "mcp")) + "." + str(item.get("tool", "unknown"))
                if identifier in calls and ("server" not in item or "tool" not in item):
                    name = calls[identifier]["tool_calls"][0]["function_name"]
                arguments = item.get("arguments", {})
            step = _call(steps, calls, identifier, name, arguments,
                         extra={"stdout_type": item_type, "item_id": identifier})
            if step is None:
                warnings.append("stdout tool call lacks item id; retained as raw")
                continue
            step["extra"]["status"] = item.get("status", kind.removeprefix("item."))
            if kind == "item.completed":
                if item_type == "command_execution":
                    if "aggregated_output" in item:
                        _output(step, identifier, _text(item["aggregated_output"], warnings, "stdout command output"),
                                {"exit_code": item.get("exit_code")})
                elif "result" in item or "error" in item:
                    result = item.get("result")
                    if isinstance(result, dict) and "content" in result:
                        text = _text(result["content"], warnings, "stdout MCP output")
                    elif isinstance(result, str):
                        text = result
                    else:
                        text = json.dumps(_strip_encrypted(result if result is not None else item.get("error")), ensure_ascii=False)
                    _output(step, identifier, text)
        elif item_type not in ("plan",):
            warnings.append(f"stdout item type {item_type!r} unsupported; retained in raw artifact")
    return usage



def _opencode_timestamp(value):
    # OpenCode run --format json emits Date.now(), in milliseconds. Keep the
    # existing seconds-based supervisor/Codex timestamp handling unchanged.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _timestamp(value / 1000)
    return None


def _opencode_usage(part, warnings):
    """Keep reported native usage without inventing unknown token semantics/cost."""
    usage = {}
    tokens = part.get("tokens")
    if isinstance(tokens, dict):
        for key in ("input", "output", "reasoning", "total"):
            value = tokens.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[key] = value
            elif key in tokens:
                warnings.append(f"OpenCode tokens.{key}: unsupported count retained as raw")
        cache = tokens.get("cache")
        if isinstance(cache, dict):
            for key in ("read", "write"):
                value = cache.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage["cache_" + key] = value
                elif key in cache:
                    warnings.append(f"OpenCode tokens.cache.{key}: unsupported count retained as raw")
    elif tokens is not None:
        warnings.append("OpenCode tokens: unsupported usage retained as raw")
    cost = part.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
        # OpenCode computes this from configured model prices; it is not proof
        # of actual billing or the compute cost of a locally deployed model.
        usage["reported_cost_usd"] = cost
    elif cost is not None:
        warnings.append("OpenCode cost: unsupported value retained as raw")
    return usage



def _opencode_token_metrics(usage, version):
    # Official v1.19.3 Session.getUsage removes cache tokens from input and
    # reasoning tokens from output. ATIF includes both in its standard totals.
    # Retain native categories only when version/required counters are unknown.
    if version != "1.19.3":
        return {}
    metrics = {}
    if all(key in usage for key in ("input", "cache_read", "cache_write")):
        metrics["prompt_tokens"] = usage["input"] + usage["cache_read"] + usage["cache_write"]
    if all(key in usage for key in ("output", "reasoning")):
        metrics["completion_tokens"] = usage["output"] + usage["reasoning"]
    if "cache_read" in usage:
        metrics["cached_tokens"] = usage["cache_read"]
    return metrics

def _convert_opencode(events, steps, warnings, metadata):
    """Convert v1 run NDJSON, never searching OpenCode's global session store.

    The official CLI emits complete text/tool parts, not token deltas. Part IDs
    therefore identify snapshots; repeated step_finish parts must not double
    count usage. CLI stdout omits subagent internals and optional reasoning.
    """
    metadata = metadata or {}
    version = metadata.get("version")
    version = version if isinstance(version, str) and version else "unknown"
    model = metadata.get("model")
    model = model if isinstance(model, str) and model else None
    identities, envelope_ids = set(), set()
    for event in events:
        identity = event.get("sessionID")
        if isinstance(identity, str) and identity:
            identities.add(identity)
            envelope_ids.add(identity)
        part = event.get("part")
        if isinstance(part, dict) and isinstance(part.get("sessionID"), str) and part["sessionID"]:
            identities.add(part["sessionID"])
    claimed = metadata.get("session_id")
    if isinstance(claimed, str) and claimed:
        identities.add(claimed)
    if len(identities) != 1 or len(envelope_ids) != 1:
        warnings.append("OpenCode mixed/missing session IDs; stdout retained as raw without merging sessions")
        return model, version, None, None
    session_id = next(iter(identities))
    calls, parts, usage_parts = {}, {}, {}
    last_finish = None
    known = {"text": "text", "reasoning": "reasoning", "tool_use": "tool",
             "step_start": "step-start", "step_finish": "step-finish"}
    for event in events:
        kind = event.get("type")
        if event.get("sessionID") != session_id:
            warnings.append("OpenCode event missing matching sessionID; retained as raw")
            continue
        timestamp = _opencode_timestamp(event.get("timestamp"))
        if timestamp is None:
            warnings.append("OpenCode event timestamp unavailable; retained without invented timestamp")
        if kind == "error":
            warnings.append("OpenCode stdout reports error; trajectory may be partial")
            _step(steps, "system", json.dumps(_strip_encrypted(event.get("error")), ensure_ascii=False),
                  timestamp=timestamp, extra={"stdout_type": "error", "backend_error": True})
            continue
        if not isinstance(kind, str) or kind not in known:
            warnings.append(f"OpenCode event type {kind!r} unsupported; retained in raw artifact")
            continue
        part = event.get("part")
        if not isinstance(part, dict) or part.get("type") != known[kind]:
            warnings.append(f"OpenCode {kind}: missing/unsupported part retained as raw")
            continue
        identifier = part.get("id")
        if not isinstance(identifier, str) or not identifier:
            warnings.append(f"OpenCode {kind}: missing part id retained as raw")
            continue
        if identifier in parts and parts[identifier]["extra"]["stdout_type"] != kind:
            warnings.append("OpenCode conflicting part types unsupported; retained as raw")
            continue
        extra = {"stdout_type": kind, "part_id": identifier, "message_id": part.get("messageID")}
        if kind in ("text", "reasoning"):
            text = part.get("text")
            if not isinstance(text, str):
                warnings.append(f"OpenCode {kind}: unsupported text retained as raw")
                continue
            step = parts.get(identifier)
            if step is None:
                step = parts[identifier] = _step(steps, "agent", timestamp=timestamp, extra=extra)
            if kind == "reasoning":
                step["reasoning_content"] = text
                step["extra"]["visibility"] = "visible_backend_text"
            else:
                step["message"] = text
        elif kind == "tool_use":
            state = part.get("state")
            name, call_id = part.get("tool"), part.get("callID")
            if not isinstance(state, dict) or not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
                warnings.append("OpenCode tool_use: missing/unsupported tool state or callID retained as raw")
                continue
            if call_id in calls and calls[call_id]["extra"]["part_id"] != identifier:
                warnings.append("OpenCode conflicting callID across parts unsupported; retained as raw")
                continue
            step = _call(steps, calls, call_id, name, state.get("input"), timestamp, extra)
            parts[identifier] = step
            step["extra"]["status"] = state.get("status")
            if not isinstance(state.get("input"), dict):
                warnings.append("OpenCode tool input unavailable/unsupported; captured fields retained as raw")
            if state.get("status") == "completed" and isinstance(state.get("output"), str):
                _output(step, call_id, state["output"], {"status": "completed"})
            elif state.get("status") == "error" and isinstance(state.get("error"), str):
                _output(step, call_id, state["error"], {"status": "error"})
                warnings.append("OpenCode tool reports error; trajectory may be partial")
            else:
                warnings.append("OpenCode tool result unavailable/unsupported; retained as raw")
            if state.get("attachments"):
                warnings.append("OpenCode tool attachments unsupported; retained in raw artifact")
        else:
            step = parts.get(identifier)
            if step is None:
                step = parts[identifier] = _step(steps, "agent", timestamp=timestamp, extra=extra)
            if kind == "step_finish":
                last_finish = part.get("reason")
                step["extra"]["finish_reason"] = last_finish
                usage = _opencode_usage(part, warnings)
                usage_parts[identifier] = usage
                if usage:
                    step["metrics"] = _opencode_token_metrics(usage, version)
                    step["metrics"]["extra"] = {"opencode_usage": usage,
                                                "basis": "reported step_finish; cost uses configured prices"}
    if last_finish != "stop":
        warnings.append("OpenCode terminal step_finish reason=stop unavailable; stdout may be truncated")
    warnings.append("OpenCode stdout covers emitted terminal tool snapshots; in-progress tools, un-emitted reasoning and subagent internals unavailable")
    # Native counts/cost remain separately attributable. Publish standard token
    # totals only when every observed finish has all required measured counters.
    totals = {"extra": {"opencode_usage": {
        key: sum(item[key] for item in usage_parts.values() if key in item)
        for key in sorted({key for item in usage_parts.values() for key in item})},
        "observed_usage_steps": len(usage_parts),
        "basis": "observed unique step_finish parts; reported tokens, configured-price cost"}}
    converted = [_opencode_token_metrics(item, version) for item in usage_parts.values()]
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        if converted and all(key in item for item in converted):
            totals["total_" + key] = sum(item[key] for item in converted)
    return model, version, totals if any(usage_parts.values()) else None, session_id


def _controller(root, steps, warnings, request):
    request_id = request["id"]
    operation = _json_object(_read(root, "operation.json"), "operation.json", warnings)
    receipt = _json_object(_read(root, "receipt.json"), "receipt.json", warnings)
    receipt_validated = receipt is not None
    if receipt is None:
        receipt = _json_object(_read(root, "executor.stdout"), "executor.stdout", warnings)
        if receipt is not None:
            warnings.append("controller receipt recovered from executor.stdout; supervisor validation not confirmed")
    if operation is None:
        if receipt is not None:
            warnings.append("controller receipt has no operation artifact; retained without invented tool call")
            _step(steps, "system", json.dumps(_strip_encrypted(receipt), ensure_ascii=False, sort_keys=True),
                  extra={"actor": "controller", "request_id": request_id,
                         "artifact": "receipt.json" if receipt_validated else "executor.stdout",
                         "supervisor_validated": receipt_validated, "dispatch_status": "no_operation_recorded"})
        return
    identifier = "controller-gate:" + request_id
    executor_process = request.get("processes", {}).get("executor") if isinstance(request.get("processes"), dict) else None
    step = _step(steps, "agent", "Controller gate operation recorded; dispatch and uptake follow captured evidence.",
                 extra={"actor": "controller", "request_id": request_id,
                        "dispatch_status": "receipt_recorded" if receipt is not None else "attempt_recorded" if executor_process else "prepared_dispatch_unconfirmed"})
    step["llm_call_count"] = 0
    step["tool_calls"] = [{"tool_call_id": identifier, "function_name": "controller.gate",
                           "arguments": _strip_encrypted(operation)}]
    if receipt is not None:
        _output(step, identifier, json.dumps(_strip_encrypted(receipt), ensure_ascii=False, sort_keys=True),
                {"receipt_is_runtime_claim": True, "supervisor_validated": receipt_validated,
                 "runtime_verified": False})
    else:
        warnings.append("controller operation has no receipt; dispatch outcome remains unknown")


def export_request(request_dir, *, run, sessions_dir=None, status=None):
    """Write trajectory.json/export.json for one request and return its manifest.

    Re-export is deterministic for unchanged artifacts. A verified private native
    archive wins over subsequent shared-session changes. No global Codex state is
    copied, and unavailable internals remain explicitly unavailable.
    """
    supplied = Path(request_dir).absolute()
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValueError("request directory must be an existing directory, not a symlink")
    root = supplied.resolve()
    warnings = []
    request = _json_object(_read(root, "request.json"), "request.json", warnings)
    if request is None or not isinstance(request.get("id"), str) or not request["id"]:
        raise ValueError("request.json must contain a nonempty string id")
    request_id = request["id"]
    if not isinstance(run, str) or not run:
        raise ValueError("run must be a nonempty string")
    if request.get("run") not in (None, run):
        raise ValueError("request run identity does not match requested run")
    stdout = _events(_read(root, "agent.stdout"), "agent.stdout", warnings)
    thread_ids = {e.get("thread_id") for e in stdout if e.get("type") == "thread.started"
                  and isinstance(e.get("thread_id"), str) and e["thread_id"]}
    thread_id = next(iter(thread_ids)) if len(thread_ids) == 1 else None
    if len(thread_ids) > 1:
        warnings.append("multiple stdout thread IDs; no shared native session was collected")
    opencode = request.get("backend") == "opencode"
    native, source = (None, None) if opencode else _native(root, thread_id, sessions_dir, warnings)
    steps = []
    prompt = _read(root, "prompt.txt")
    if prompt is not None:
        _step(steps, "user", prompt.decode("utf-8", errors="replace"),
              timestamp=request.get("started_at"), extra={"actor": "supervisor", "artifact": "prompt.txt"})
    else:
        warnings.append("prompt.txt missing; request context unavailable")
    model, version, usage = None, "unknown", None
    if opencode:
        thread_id = None
        metadata = _json_object(_read(root, "agent.backend.json"), "agent.backend.json", warnings)
        model, version, usage, session_id = _convert_opencode(stdout, steps, warnings, metadata)
        capture = "opencode_stdout"
        if session_id:
            thread_id = session_id
        source = {"path": "agent.stdout", "native_session_id": session_id,
                  "basis": "only this invocation's NDJSON; no global OpenCode storage collection"}
    elif native is not None:
        model, version = _convert_native(native, steps, warnings)
        capture = "native"
        # stdout is still authoritative for usage emitted by this exact invocation.
        usage = next((e["usage"] for e in reversed(stdout) if e.get("type") == "turn.completed" and isinstance(e.get("usage"), dict)), None)
    else:
        usage = _convert_stdout(stdout, steps, warnings)
        capture = "stdout" if any(e.get("type", "").startswith("item.") for e in stdout) else "artifacts_only"
        warnings.append("native session unavailable; internal tool coverage is limited to observed stdout events")
    response = _json_object(_read(root, "response.json"), "response.json", warnings)
    if response is not None:
        text = json.dumps(_strip_encrypted(response), ensure_ascii=False, sort_keys=True)
        already_recorded = False
        for step in steps:
            if step["source"] == "agent":
                try:
                    already_recorded = json.loads(step["message"]) == _strip_encrypted(response)
                except (ValueError, TypeError):
                    pass
                if already_recorded:
                    break
        if not already_recorded:
            _step(steps, "agent", text, extra={"artifact": "response.json", "structured_response": True})
    elif capture == "artifacts_only":
        warnings.append("no structured response/internal events; export contains only available request artifacts")
    _controller(root, steps, warnings, request)
    if not steps:
        _step(steps, "system", "No interaction content captured.", extra={"capture_gap": True})
    outcome = status or request.get("outcome") or request.get("stage") or "unknown"
    for step in steps:
        if step.get("tool_calls") and not step.get("observation"):
            warnings.append("recorded tool call has no captured output; result is unavailable")
    complete = outcome in ("completed", "success", "succeeded", "complete")
    conversion_gap = any(any(term in warning for term in ("unsupported", "unavailable", "incomplete", "no matching", "missing"))
                         for warning in warnings)
    coverage = {"basis": "visible recorded events only; no inferred private reasoning or LLM call counts",
                "internal_tools": "native_response_items" if native is not None else "opencode_stdout_events" if opencode and thread_id else "stdout_events_only" if capture == "stdout" else "unavailable",
                "partial": not complete or native is None or conversion_gap,
                "lifecycle_complete": complete,
                "multimodal": "unsupported content retained in raw artifacts, not represented as text"}
    warnings = list(dict.fromkeys(warnings))
    trajectory = {"schema_version": "ATIF-v1.7", "session_id": run, "trajectory_id": request_id,
                  "agent": {"name": "meta-agent", "version": version,
                            "extra": {"exporter_version": VERSION, "backend": request.get("backend", "unknown")}},
                  "steps": steps, "extra": {"request_id": request_id, "run": run,
                  "phase": request.get("phase"), "step": request.get("step"), "status": outcome,
                  "capture": capture, "coverage": coverage, "warnings": warnings}}
    trajectory["extra"]["request_lifecycle"] = {key: _strip_encrypted(request[key])
                                              for key in ("started_at", "finished_at", "outcome", "error", "processes", "skill_sha")
                                              if key in request}
    if model:
        trajectory["agent"]["model_name"] = model
    if thread_id:
        trajectory["extra"]["native_session_id"] = thread_id
    previous = request.get("previous_request")
    if isinstance(previous, dict) and isinstance(previous.get("id"), str):
        trajectory["extra"]["previous_request_id"] = previous["id"]
    if opencode and usage:
        trajectory["final_metrics"] = usage
    elif usage:
        mapping = {"input_tokens": "total_prompt_tokens", "output_tokens": "total_completion_tokens",
                   "cached_input_tokens": "total_cached_tokens"}
        metrics = {target: usage[key] for key, target in mapping.items()
                   if isinstance(usage.get(key), int) and not isinstance(usage[key], bool) and usage[key] >= 0}
        if metrics:
            trajectory["final_metrics"] = metrics
    _write(root, "trajectory.json", _json_bytes(trajectory))
    hashes = {name: _sha(data) for name in (*ARTIFACTS, "trajectory.json")
              if (data := _read(root, name)) is not None}
    if source and source.get("archived_sha256"):
        hashes[source["path"]] = source["archived_sha256"]
        metadata = _read(root, "native/capture.json")
        if metadata is not None:
            hashes["native/capture.json"] = _sha(metadata)
    manifest = {"exporter_version": VERSION, "schema_version": "ATIF-v1.7", "request_id": request_id,
                "run": run, "status": outcome, "capture": capture, "source": source or {"path": "agent.stdout"},
                "coverage": coverage, "warnings": warnings, "artifact_sha256": hashes,
                "trajectory": "trajectory.json"}
    _write(root, "export.json", _json_bytes(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-dir", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--codex-sessions", type=Path)
    parser.add_argument("--status")
    args = parser.parse_args(argv)
    print(json.dumps(export_request(args.request_dir, run=args.run,
                                   sessions_dir=args.codex_sessions, status=args.status), ensure_ascii=False))


if __name__ == "__main__":
    main()
