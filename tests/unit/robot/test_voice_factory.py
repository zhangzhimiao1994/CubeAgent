from __future__ import annotations

from pydantic import SecretStr

from agent_hub.robot.voice.factory import build_robot_voice
from agent_hub.robot.voice.fake import FakeSpeechToText, FakeTextToSpeech
from agent_hub.settings import Settings


def test_build_robot_voice_none_when_providers_disabled() -> None:
    settings = Settings.model_construct(environment="test")
    assert build_robot_voice(settings) is None


def test_build_robot_voice_fake_adapters() -> None:
    settings = Settings.model_construct(
        environment="test",
        robot_stt_provider="fake",
        robot_tts_provider="fake",
    )
    gateway = build_robot_voice(settings)
    assert gateway is not None
    assert isinstance(gateway.stt, FakeSpeechToText)
    assert isinstance(gateway.tts, FakeTextToSpeech)


def test_build_robot_voice_skips_openai_without_key() -> None:
    settings = Settings.model_construct(
        environment="test",
        robot_stt_provider="openai_compatible",
        robot_stt_api_key=None,
        robot_tts_provider="openai_compatible",
        robot_tts_api_key=SecretStr(""),
    )
    assert build_robot_voice(settings) is None
