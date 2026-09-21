from __future__ import annotations

import json

import httpx
import pytest

from agent_hub.multimodal.audio_providers import (
    MiniMaxAudioGenerationClient,
    MiniMaxAudioGenerationError,
)


@pytest.mark.asyncio
async def test_minimax_audio_client_submits_and_stores_hex_audio(tmp_path) -> None:  # type: ignore[no-untyped-def]
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/t2a_v2"
        assert request.headers["Authorization"] == "Bearer sk-live"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload == {
            "model": "speech-2.8-turbo",
            "text": "这是一段科普旁白。",
            "stream": False,
            "voice_setting": {
                "voice_id": "male-qn-qingse",
                "speed": 1.0,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
        }
        return httpx.Response(
            200,
            json={
                "data": {
                    "audio": "49443303000000000000",
                    "status": 2,
                },
                "trace_id": "trace-1",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        )

    client = MiniMaxAudioGenerationClient(transport=httpx.MockTransport(handler))

    artifact = await client.generate_text_to_audio(
        api_key="sk-live",
        api_base="https://api.minimax.chat/v1",
        model="speech-2.8-turbo",
        prompt="这是一段科普旁白。",
        output_dir=tmp_path,
    )

    assert artifact.kind == "audio"
    assert artifact.provider == "minimax"
    assert artifact.model == "speech-2.8-turbo"
    assert artifact.task_id == "trace-1"
    assert artifact.mime_type == "audio/mpeg"
    assert artifact.path.suffix == ".mp3"
    assert artifact.path.read_bytes() == bytes.fromhex("49443303000000000000")
    assert [request.url.path for request in requests] == ["/v1/t2a_v2"]


@pytest.mark.asyncio
async def test_minimax_audio_client_reports_missing_audio(tmp_path) -> None:  # type: ignore[no-untyped-def]
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {"status": 2},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        )

    client = MiniMaxAudioGenerationClient(transport=httpx.MockTransport(handler))

    with pytest.raises(MiniMaxAudioGenerationError, match="missing audio"):
        await client.generate_text_to_audio(
            api_key="sk-live",
            api_base="https://api.minimax.chat/v1",
            model="speech-2.8-turbo",
            prompt="这是一段科普旁白。",
            output_dir=tmp_path,
        )
