from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Envelope(BaseModel):
    type: str
    turn_id: str | None = None
    reply_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


DeviceMessageType = Literal[
    "hello",
    "utterance.start",
    "utterance.audio",
    "utterance.end",
    "audio_chunk",
    "audio.end",
    "final_transcript",
    "barge_in",
    "state.patch",
    "ping",
]

CloudMessageType = Literal[
    "hello.ok",
    "state",
    "text_delta",
    "final",
    "cancelled",
    "audio_delta",
    "audio.final",
    "transcript",
    "assistant.audio",
    "assistant.text",
    "assistant.end",
    "state.sync",
    "memory.hint",
    "proactive.say",
    "cancel",
    "error",
    "pong",
]
