from __future__ import annotations

import json

import httpx
import pytest

from agent_hub.robot.voice.audio import pcm16_to_wav
from agent_hub.robot.voice.openai_compat import (
    OpenAICompatibleSpeechToText,
    OpenAICompatibleTextToSpeech,
)
from agent_hub.robot.voice.types import AudioFormat, SpeechAudio


@pytest.mark.asyncio
async def test_openai_compatible_stt_posts_multipart_and_reads_text() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, json={"text": "晚上好"})

    client = OpenAICompatibleSpeechToText(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="whisper-1",
        transport=httpx.MockTransport(handler),
    )
    pcm = b"\x00\x01" * 16
    result = await client.transcribe(
        SpeechAudio(data=pcm, format=AudioFormat.PCM16, sample_rate_hz=16000)
    )

    assert result.text == "晚上好"
    assert result.provider == "openai_compatible"
    assert result.model == "whisper-1"
    assert requests[0].headers["content-type"].startswith("multipart/form-data")
    wav = pcm16_to_wav(pcm, sample_rate_hz=16000)
    assert wav[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_openai_compatible_tts_posts_json_and_yields_audio() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/audio/speech"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "tts-1"
        assert payload["voice"] == "alloy"
        assert payload["input"] == "你好。"
        return httpx.Response(200, content=b"ID3fake-mp3", headers={"content-type": "audio/mpeg"})

    client = OpenAICompatibleTextToSpeech(
        api_key="sk-test",
        base_url="https://api.openai.com",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        transport=httpx.MockTransport(handler),
    )
    chunks = [chunk async for chunk in client.synthesize("你好。")]

    assert len(chunks) == 1
    assert chunks[0].data == b"ID3fake-mp3"
    assert chunks[0].format is AudioFormat.MP3
    assert chunks[0].mime_type == "audio/mpeg"
    assert chunks[0].final is True
