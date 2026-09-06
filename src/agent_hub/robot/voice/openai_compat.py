from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from agent_hub.robot.voice.audio import pcm16_to_wav
from agent_hub.robot.voice.errors import VoiceProviderError
from agent_hub.robot.voice.types import AudioFormat, SpeechAudio, SpeechChunk, Transcript

_MIME_BY_FORMAT = {
    AudioFormat.PCM16: ("speech.wav", "audio/wav"),
    AudioFormat.WAV: ("speech.wav", "audio/wav"),
    AudioFormat.WEBM: ("speech.webm", "audio/webm"),
    AudioFormat.MP3: ("speech.mp3", "audio/mpeg"),
}


class OpenAICompatibleSpeechToText:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "whisper-1",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._transport = transport
        self._timeout_seconds = timeout_seconds

    async def transcribe(self, audio: SpeechAudio) -> Transcript:
        filename, content_type = _MIME_BY_FORMAT.get(
            audio.format, ("speech.wav", "audio/wav")
        )
        payload = audio.data
        if audio.format is AudioFormat.PCM16:
            payload = pcm16_to_wav(
                audio.data,
                sample_rate_hz=audio.sample_rate_hz,
                channels=audio.channels,
            )
            filename, content_type = "speech.wav", "audio/wav"
        async with httpx.AsyncClient(
            timeout=self._timeout_seconds,
            transport=self._transport,
            follow_redirects=True,
        ) as client:
            response = await client.post(
                openai_audio_url(self._base_url, "audio/transcriptions"),
                headers={"authorization": f"Bearer {self._api_key}"},
                files={"file": (filename, payload, content_type)},
                data={"model": self._model},
            )
        if response.status_code >= 400:
            raise VoiceProviderError("speech-to-text provider request failed")
        try:
            body = response.json()
        except ValueError as error:
            raise VoiceProviderError("speech-to-text provider returned invalid JSON") from error
        text = ""
        if isinstance(body, dict):
            raw = body.get("text")
            if isinstance(raw, str):
                text = raw.strip()
        return Transcript(text=text, provider="openai_compatible", model=self._model)


class OpenAICompatibleTextToSpeech:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._voice = voice
        self._response_format = response_format
        self._transport = transport
        self._timeout_seconds = timeout_seconds

    async def synthesize(self, text: str) -> AsyncIterator[SpeechChunk]:
        async with httpx.AsyncClient(
            timeout=self._timeout_seconds,
            transport=self._transport,
            follow_redirects=True,
        ) as client:
            response = await client.post(
                openai_audio_url(self._base_url, "audio/speech"),
                headers={"authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "voice": self._voice,
                    "input": text,
                    "response_format": self._response_format,
                },
            )
        if response.status_code >= 400:
            raise VoiceProviderError("text-to-speech provider request failed")
        audio_format = parse_tts_format(self._response_format)
        yield SpeechChunk(
            data=response.content,
            format=audio_format,
            mime_type=_tts_mime(audio_format),
            final=True,
        )


def openai_audio_url(base_url: str, resource: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/{resource}"
    return f"{root}/v1/{resource}"


def parse_tts_format(value: str) -> AudioFormat:
    raw = value.strip().lower()
    if raw in {"wav", "wave"}:
        return AudioFormat.WAV
    if raw in {"pcm", "pcm16"}:
        return AudioFormat.PCM16
    return AudioFormat.MP3


def _tts_mime(audio_format: AudioFormat) -> str:
    if audio_format is AudioFormat.WAV:
        return "audio/wav"
    if audio_format is AudioFormat.PCM16:
        return "audio/pcm"
    return "audio/mpeg"


__all__ = [
    "OpenAICompatibleSpeechToText",
    "OpenAICompatibleTextToSpeech",
    "openai_audio_url",
]
