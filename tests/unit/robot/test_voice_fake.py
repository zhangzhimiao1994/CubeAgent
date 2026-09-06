from __future__ import annotations

import pytest

from agent_hub.robot.voice.fake import FakeSpeechToText, FakeTextToSpeech
from agent_hub.robot.voice.types import AudioFormat, SpeechAudio


@pytest.mark.asyncio
async def test_fake_stt_returns_configured_transcript() -> None:
    stt = FakeSpeechToText(transcript="你好")
    result = await stt.transcribe(SpeechAudio(data=b"pcm", format=AudioFormat.PCM16))
    assert result.text == "你好"
    assert stt.calls == 1


@pytest.mark.asyncio
async def test_fake_tts_yields_configured_chunks() -> None:
    tts = FakeTextToSpeech(chunks=[b"aa", b"bb"])
    out = [chunk async for chunk in tts.synthesize("hello")]
    assert [item.data for item in out] == [b"aa", b"bb"]
    assert tts.texts == ["hello"]
