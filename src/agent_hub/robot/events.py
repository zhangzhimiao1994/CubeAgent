from __future__ import annotations

from collections.abc import Mapping, Sequence


def extract_assistant_text(events: Sequence[Mapping[str, object]]) -> str:
    """Return the latest main-agent answer text from public run events."""

    chosen = ""
    for event in events:
        if _event_kind(event) != "artifact.created":
            continue
        if _producer(event) == "conversation_history":
            continue
        text = _artifact_text(event)
        if text:
            chosen = text
    return chosen


def text_delta(previous: str, current: str) -> str:
    if current.startswith(previous):
        return current[len(previous) :]
    return ""


def event_is_terminal(event: Mapping[str, object]) -> bool:
    return _event_kind(event) in {
        "runtime.completed",
        "runtime.failed",
        "runtime.cancelled",
    }


def event_failed(event: Mapping[str, object]) -> bool:
    return _event_kind(event) == "runtime.failed"


def _event_kind(event: Mapping[str, object]) -> str:
    raw = event.get("kind")
    if isinstance(raw, str) and raw:
        return raw
    raw = event.get("type")
    return raw if isinstance(raw, str) else ""


def _producer(event: Mapping[str, object]) -> str:
    artifact = event.get("artifact")
    if isinstance(artifact, Mapping):
        producer = artifact.get("producer")
        if isinstance(producer, str):
            return producer
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        producer = payload.get("producer")
        if isinstance(producer, str):
            return producer
    return ""


def _artifact_text(event: Mapping[str, object]) -> str:
    artifact = event.get("artifact")
    if isinstance(artifact, Mapping):
        text = _mapping_text(artifact)
        if text:
            return text
        content = artifact.get("content")
        if isinstance(content, Mapping):
            text = _mapping_text(content)
            if text:
                return text
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        text = _mapping_text(payload)
        if text:
            return text
    return _mapping_text(event)


def _mapping_text(payload: Mapping[str, object]) -> str:
    for key in ("output", "result", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    content = payload.get("content")
    if isinstance(content, Mapping):
        value = content.get("text")
        if isinstance(value, str) and value.strip():
            return value
    return ""


__all__ = [
    "event_failed",
    "event_is_terminal",
    "extract_assistant_text",
    "text_delta",
]
