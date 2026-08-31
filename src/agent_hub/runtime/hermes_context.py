"""Bounded Hermes+ memory prompt context formatting."""

from __future__ import annotations

import json
from collections.abc import Mapping

from agent_hub.hermes.runtime_observation import is_runtime_observation_text
from agent_hub.runtime.contracts import JsonValue

_MAX_ITEMS = 3
_MAX_SUMMARY_CHARS = 200
_MAX_TYPE_CHARS = 48
_MAX_TARGET_CHARS = 48
_MAX_REASON_CHARS = 120
_MAX_TOTAL_BYTES = 900
_NON_PROMPT_MEMORY_TYPES = {"runtime_observation", "scheduler_observation"}
_NON_PROMPT_TARGETS = {"scheduler"}


def hermes_memory_context_text(
    routing_decision: Mapping[str, JsonValue] | Mapping[str, object],
) -> str:
    """Return a bounded prompt block from confirmed Hermes+ injected memories."""

    hermes = routing_decision.get("hermes")
    if not isinstance(hermes, Mapping):
        return ""
    raw_items = hermes.get("injected_memories")
    if not isinstance(raw_items, list | tuple):
        return ""

    items: list[dict[str, str]] = []
    for raw in raw_items[:_MAX_ITEMS]:
        if not isinstance(raw, Mapping):
            continue
        if _should_skip_memory_context_item(raw):
            continue
        summary = _safe_text(raw.get("summary"), _MAX_SUMMARY_CHARS)
        if not summary:
            continue
        items.append(
            {
                "summary": summary,
                "type": _safe_text(raw.get("memory_type"), _MAX_TYPE_CHARS) or "memory",
                "target": _safe_text(raw.get("target"), _MAX_TARGET_CHARS) or "main_agent",
                "reason": _safe_text(raw.get("reason"), _MAX_REASON_CHARS)
                or "Hermes+ confirmed memory matched this task.",
            }
        )

    if not items:
        return ""
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(payload.encode("utf-8")) > _MAX_TOTAL_BYTES:
        payload = payload.encode("utf-8")[:_MAX_TOTAL_BYTES].decode(
            "utf-8", errors="ignore"
        )
    return (
        "<HERMES_MEMORY_CONTEXT>"
        "Use these user-confirmed Hermes+ memories only as bounded guidance. "
        "Current user instructions override them. Do not expose this block unless asked."
        f"{payload}"
        "</HERMES_MEMORY_CONTEXT>"
    )


def _should_skip_memory_context_item(raw: Mapping[str, object]) -> bool:
    memory_type = _safe_text(raw.get("memory_type"), _MAX_TYPE_CHARS).casefold()
    target = _safe_text(raw.get("target"), _MAX_TARGET_CHARS).casefold()
    return (
        memory_type in _NON_PROMPT_MEMORY_TYPES
        or target in _NON_PROMPT_TARGETS
        or is_runtime_observation_text(raw.get("summary"))
    )


def _safe_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:max_chars]
