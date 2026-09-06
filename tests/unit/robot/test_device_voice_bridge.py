from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

_DEVICE_SRC = Path(__file__).resolve().parents[3] / "device" / "robot_runtime" / "src"
if str(_DEVICE_SRC) not in sys.path:
    sys.path.insert(0, str(_DEVICE_SRC))

from robot_runtime.audio.playback import NullAudioPlayback  # type: ignore[import-not-found]
from robot_runtime.bridge.client import (  # type: ignore[import-not-found]
    LoggingBridgeClient,
    envelope_from_cloud,
)
from robot_runtime.bridge.protocol import Envelope  # type: ignore[import-not-found]
from robot_runtime.runtime import RobotRuntime  # type: ignore[import-not-found]
from robot_runtime.vad.barge_in import SimpleBargeIn  # type: ignore[import-not-found]
from robot_runtime.vad.turn_taking import EnergyTurnTaking  # type: ignore[import-not-found]


@pytest.mark.asyncio
async def test_capture_sends_audio_chunk_and_audio_end() -> None:
    runtime = RobotRuntime()
    assert isinstance(runtime.bridge, LoggingBridgeClient)
    runtime.turn_taking = EnergyTurnTaking(silence_frames_to_end=1, energy_threshold=1.0)
    speech = b"\x00\x10" * 80
    await runtime.on_frame(speech, energy=10.0)
    await runtime.on_frame(b"\x00\x00" * 80, energy=0.0)

    types = [item.type for item in runtime.bridge.sent]
    assert "audio_chunk" in types
    assert "audio.end" in types
    chunk = next(item for item in runtime.bridge.sent if item.type == "audio_chunk")
    audio = base64.b64decode(str(chunk.payload["audio"]))
    assert audio == speech
    assert chunk.payload["format"] == "pcm16"


@pytest.mark.asyncio
async def test_audio_delta_enqueues_playback_and_barge_in_clears() -> None:
    playback = NullAudioPlayback()
    runtime = RobotRuntime(playback=playback)
    runtime.turn_taking = EnergyTurnTaking(silence_frames_to_end=99, energy_threshold=1.0)
    runtime.barge_in = SimpleBargeIn(min_speech_frames=1)
    await playback.start()
    await runtime._on_cloud(
        envelope_from_cloud(
            {
                "type": "audio_delta",
                "audio": base64.b64encode(b"mp3-bytes").decode("ascii"),
                "format": "mp3",
                "turn_id": "t1",
            }
        )
    )
    assert playback.is_playing
    await runtime.on_frame(b"\x00\x10" * 80, energy=10.0)
    assert not playback.is_playing
    assert any(item.type == "barge_in" for item in runtime.bridge.sent)


def test_envelope_from_cloud_merges_audio_fields() -> None:
    envelope = envelope_from_cloud(
        {
            "type": "audio_delta",
            "audio": "YWE=",
            "format": "mp3",
            "mime_type": "audio/mpeg",
            "turn_id": "t9",
        }
    )
    assert isinstance(envelope, Envelope)
    assert envelope.payload["audio"] == "YWE="
    assert envelope.payload["format"] == "mp3"
    assert envelope.payload["mime_type"] == "audio/mpeg"
