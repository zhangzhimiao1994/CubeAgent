from __future__ import annotations

from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx

from agent_hub.robot.voice.errors import VoiceProviderError
from agent_hub.robot.voice.openai_compat import parse_tts_format
from agent_hub.robot.voice.types import AudioFormat, SpeechChunk


class DashScopeTextToSpeech:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "cosyvoice-v2",
        voice: str = "longxiaochun",
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
        audio_format = parse_tts_format(self._response_format)
        async with httpx.AsyncClient(
            timeout=self._timeout_seconds,
            transport=self._transport,
            follow_redirects=True,
        ) as client:
            response = await client.post(
                f"{dashscope_origin(self._base_url)}/api/v1/services/audio/tts/SpeechSynthesizer",
                headers={"authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "input": {"text": text},
                    "parameters": {
                        "voice": self._voice,
                        "format": _dashscope_format(audio_format),
                    },
                },
            )
        if response.status_code >= 400:
            raise VoiceProviderError("text-to-speech provider request failed")
        yield SpeechChunk(
            data=response.content,
            format=audio_format,
            mime_type=_dashscope_mime(audio_format),
            final=True,
        )


def dashscope_origin(base_url: str) -> str:
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        raise VoiceProviderError("invalid DashScope base URL")
    return f"{parts.scheme}://{parts.netloc}"


def _dashscope_format(audio_format: AudioFormat) -> str:
    if audio_format is AudioFormat.WAV:
        return "wav"
    if audio_format is AudioFormat.PCM16:
        return "pcm"
    return "mp3"


def _dashscope_mime(audio_format: AudioFormat) -> str:
    if audio_format is AudioFormat.WAV:
        return "audio/wav"
    if audio_format is AudioFormat.PCM16:
        return "audio/pcm"
    return "audio/mpeg"


__all__ = ["DashScopeTextToSpeech", "dashscope_origin"]
