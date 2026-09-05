from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class AudioFormat(StrEnum):
    PCM16 = "pcm16"
    WAV = "wav"
    WEBM = "webm"
    MP3 = "mp3"


@dataclass(frozen=True, slots=True)
class SpeechAudio:
    data: bytes
    format: AudioFormat
    sample_rate_hz: int = 16000
    channels: int = 1


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    provider: str = ""
    model: str = ""


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    data: bytes
    format: AudioFormat
    mime_type: str
    final: bool = False


class SpeechToText(Protocol):
    async def transcribe(self, audio: SpeechAudio) -> Transcript: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[SpeechChunk]: ...


__all__ = [
    "AudioFormat",
    "SpeechAudio",
    "SpeechChunk",
    "SpeechToText",
    "TextToSpeech",
    "Transcript",
]
