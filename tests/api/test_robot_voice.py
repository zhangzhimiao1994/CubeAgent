from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_hub.app import create_app
from agent_hub.robot.voice.fake import FakeSpeechToText, FakeTextToSpeech
from agent_hub.robot.voice.gateway import RobotVoiceGateway
from agent_hub.robot.voice.types import AudioFormat, SpeechChunk
from agent_hub.settings import Settings
from tests.api.test_robot_channel import RobotRunService, _register, _until


class HangingTextToSpeech:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[SpeechChunk]:
        self.texts.append(text)
        self.started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if False:
            yield SpeechChunk(data=b"", format=AudioFormat.MP3, mime_type="audio/mpeg", final=True)


def _voice_app(
    run_service: RobotRunService,
    voice: RobotVoiceGateway,
) -> FastAPI:
    return create_app(
        settings=Settings.model_construct(environment="test"),
        auth_service=object(),
        rate_limiter=object(),
        config_service=object(),
        admin_resource_service=object(),
        user_admin_service=object(),
        run_service=run_service,
        database_probe=_ok_probe,
        redis_probe=_ok_probe,
        robot_voice=voice,
    )


async def _ok_probe() -> None:
    return None


def test_audio_end_transcribes_and_submits_direct_run() -> None:
    stt = FakeSpeechToText(transcript="晚上好")
    run_service = RobotRunService()
    with TestClient(_voice_app(run_service, RobotVoiceGateway(stt=stt, tts=None))) as client:
        token = _register(client, device_id="pi-stt")
        with client.websocket_connect(
            "/api/robot/v1/ws",
            headers={"X-Device-Token": token},
        ) as websocket:
            assert websocket.receive_json()["type"] == "hello.ok"
            websocket.send_json(
                {
                    "type": "audio_chunk",
                    "turn_id": "voice-1",
                    "audio": base64.b64encode(b"pcm-frame-1").decode("ascii"),
                    "format": "pcm16",
                    "sample_rate_hz": 16000,
                }
            )
            websocket.send_json(
                {
                    "type": "audio.end",
                    "turn_id": "voice-1",
                    "audio": base64.b64encode(b"pcm-frame-2").decode("ascii"),
                    "format": "pcm16",
                }
            )
            transcript = _until(websocket, "transcript", limit=12)
            assert transcript["text"] == "晚上好"
            final = _until(websocket, "final", limit=12)
            assert final["text"]

    assert stt.calls == 1
    assert stt.received and stt.received[0].data == b"pcm-frame-1pcm-frame-2"
    assert len(run_service.submit_calls) == 1
    assert run_service.submit_calls[0]["message"] == "晚上好"


def test_audio_end_without_stt_uses_debug_text() -> None:
    run_service = RobotRunService()
    with TestClient(_voice_app(run_service, RobotVoiceGateway(stt=None, tts=None))) as client:
        token = _register(client, device_id="pi-text-fallback")
        with client.websocket_connect(
            "/api/robot/v1/ws",
            headers={"X-Device-Token": token},
        ) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "audio.end",
                    "turn_id": "voice-2",
                    "text": "调试文本",
                }
            )
            _until(websocket, "final", limit=12)

    assert run_service.submit_calls[0]["message"] == "调试文本"


def test_audio_end_without_stt_or_text_errors() -> None:
    run_service = RobotRunService()
    with TestClient(_voice_app(run_service, RobotVoiceGateway(stt=None, tts=None))) as client:
        token = _register(client, device_id="pi-no-stt")
        with client.websocket_connect(
            "/api/robot/v1/ws",
            headers={"X-Device-Token": token},
        ) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "audio.end", "turn_id": "voice-3"})
            error = _until(websocket, "error", limit=8)
            assert "stt unavailable" in str(error.get("message"))

    assert run_service.submit_calls == []


def test_reply_tts_emits_audio_delta_and_final() -> None:
    tts = FakeTextToSpeech(chunks=[b"aa", b"bb"])
    run_service = RobotRunService()
    with TestClient(_voice_app(run_service, RobotVoiceGateway(stt=None, tts=tts))) as client:
        token = _register(client, device_id="pi-tts")
        with client.websocket_connect(
            "/api/robot/v1/ws",
            headers={"X-Device-Token": token},
        ) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {"type": "final_transcript", "text": "你好", "turn_id": "tts-1"}
            )
            messages: list[dict[str, Any]] = []
            for _ in range(20):
                messages.append(websocket.receive_json())
                if any(item.get("type") == "audio.final" for item in messages):
                    break
            deltas = [item for item in messages if item.get("type") == "audio_delta"]
            assert [base64.b64decode(item["audio"]) for item in deltas] == [b"aa", b"bb"]
            assert any(item.get("type") == "audio.final" for item in messages)
            assert any(item.get("type") == "final" for item in messages)

    assert tts.texts
    assert any("你好" in text or "我在" in text for text in tts.texts)


def test_barge_in_cancels_hanging_tts() -> None:
    hanging = HangingTextToSpeech()
    run_service = RobotRunService()
    with TestClient(_voice_app(run_service, RobotVoiceGateway(stt=None, tts=hanging))) as client:
        token = _register(client, device_id="pi-barge-tts")
        with client.websocket_connect(
            "/api/robot/v1/ws",
            headers={"X-Device-Token": token},
        ) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {"type": "final_transcript", "text": "讲个故事", "turn_id": "barge-tts"}
            )
            _until(websocket, "text_delta", limit=12)
            assert hanging.started.wait(timeout=2)
            websocket.send_json({"type": "barge_in", "turn_id": "barge-tts"})
            cancelled = _until(websocket, "cancelled", limit=12)
            assert cancelled["type"] == "cancelled"

    assert hanging.cancelled.wait(timeout=2)
    assert run_service.cancel_calls
