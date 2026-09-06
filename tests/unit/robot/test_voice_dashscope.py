from __future__ import annotations

import json

import httpx
import pytest

from agent_hub.robot.voice.dashscope import DashScopeTextToSpeech
from agent_hub.robot.voice.types import AudioFormat


@pytest.mark.asyncio
async def test_dashscope_tts_posts_native_speech_synthesizer() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/api/v1/services/audio/tts/SpeechSynthesizer"
        assert request.headers["authorization"] == "Bearer sk-cn"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "cosyvoice-v2"
        assert payload["input"]["text"] == "今天天气不错。"
        return httpx.Response(200, content=b"cosy-audio", headers={"content-type": "audio/mpeg"})

    client = DashScopeTextToSpeech(
        api_key="sk-cn",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="cosyvoice-v2",
        voice="longxiaochun",
        transport=httpx.MockTransport(handler),
    )
    chunks = [chunk async for chunk in client.synthesize("今天天气不错。")]

    assert chunks[0].data == b"cosy-audio"
    assert chunks[0].format is AudioFormat.MP3
