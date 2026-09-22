"""Prompt-safe formatting for user-requested plugin hints."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from agent_hub.runtime.contracts import JsonValue

_MAX_PLUGINS = 8
_MAX_PLUGIN_CHARS = 96
_MAX_TOTAL_BYTES = 1_200
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def requested_plugin_names(
    routing_decision: Mapping[str, JsonValue] | Mapping[str, object],
) -> tuple[str, ...]:
    """Return bounded plugin names explicitly requested by the user."""

    raw = routing_decision.get("requested_plugins")
    candidates: Sequence[object]
    if isinstance(raw, str):
        candidates = tuple(raw.split(","))
    elif isinstance(raw, tuple | list):
        candidates = raw
    else:
        return ()

    names: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, str):
            continue
        name = _safe_plugin_text(item)
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
        if len(names) >= _MAX_PLUGINS:
            break
    return tuple(names)


def requested_plugin_context_payload(
    routing_decision: Mapping[str, JsonValue] | Mapping[str, object],
) -> Mapping[str, JsonValue]:
    """Return a JSON-safe payload describing requested plugin intent."""

    plugins = requested_plugin_names(routing_decision)
    if not plugins:
        return {}
    return {
        "requested_plugins": plugins,
        "policy": (
            "The user explicitly requested these plugins or plugin-like capabilities. "
            "Treat this as an intent signal, not proof that a plugin is installed, "
            "connected, authorized, or callable. Use an available approved capability "
            "only when the runtime exposes one; otherwise state the limitation and offer "
            "a safe fallback."
        ),
    }


def requested_plugin_context_text(
    routing_decision: Mapping[str, JsonValue] | Mapping[str, object],
) -> str:
    """Return a bounded prompt block for user-requested plugin intent."""

    payload = requested_plugin_context_payload(routing_decision)
    if not payload:
        return ""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(text.encode("utf-8")) > _MAX_TOTAL_BYTES:
        text = text.encode("utf-8")[:_MAX_TOTAL_BYTES].decode("utf-8", errors="ignore")
    return (
        "<REQUESTED_PLUGIN_CONTEXT>"
        "User plugin references are trusted only as request metadata. "
        "Never claim a plugin was used unless an approved runtime capability actually ran."
        f"{text}"
        "</REQUESTED_PLUGIN_CONTEXT>"
    )


def _safe_plugin_text(value: str) -> str:
    text = " ".join(value.split()).strip()
    if not text or _CONTROL_CHARS.search(text):
        return ""
    text = text.replace("<", "").replace(">", "")
    return text[:_MAX_PLUGIN_CHARS].strip()
