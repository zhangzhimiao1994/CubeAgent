from __future__ import annotations

from collections.abc import AsyncIterator

from agent_hub.robot.voice.types import AudioFormat, SpeechAudio, SpeechChunk, Transcript


class FakeSpeechToText:
    def __init__(self, transcript: str = "hello") -> None:
        self.transcript = transcript
        self.calls = 0
        self.received: list[SpeechAudio] = []

    async def transcribe(self, audio: SpeechAudio) -> Transcript:
        self.calls += 1
        self.received.append(audio)
        return Transcript(text=self.transcript, provider="fake", model="fake")


class FakeTextToSpeech:
    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self.chunks = list(chunks or [b"fake-audio"])
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[SpeechChunk]:
        self.texts.append(text)
        last_index = len(self.chunks) - 1
        for index, data in enumerate(self.chunks):
            yield SpeechChunk(
                data=data,
                format=AudioFormat.MP3,
                mime_type="audio/mpeg",
                final=index == last_index,
            )


__all__ = ["FakeSpeechToText", "FakeTextToSpeech"]
