from __future__ import annotations

import logging

from pydantic import SecretStr

from agent_hub.robot.voice.dashscope import DashScopeTextToSpeech
from agent_hub.robot.voice.fake import FakeSpeechToText, FakeTextToSpeech
from agent_hub.robot.voice.gateway import RobotVoiceGateway
from agent_hub.robot.voice.openai_compat import (
    OpenAICompatibleSpeechToText,
    OpenAICompatibleTextToSpeech,
)
from agent_hub.robot.voice.types import SpeechToText, TextToSpeech
from agent_hub.settings import Settings

_LOGGER = logging.getLogger(__name__)


def build_robot_voice(settings: Settings) -> RobotVoiceGateway | None:
    stt = _build_stt(settings)
    tts = _build_tts(settings)
    if stt is None and tts is None:
        return None
    return RobotVoiceGateway(stt=stt, tts=tts)


def _build_stt(settings: Settings) -> SpeechToText | None:
    provider = _provider_name(getattr(settings, "robot_stt_provider", "none"))
    if provider in {"", "none"}:
        return None
    if provider == "fake":
        return FakeSpeechToText()
    if provider == "openai_compatible":
        api_key = _secret(getattr(settings, "robot_stt_api_key", None))
        if not api_key:
            _LOGGER.warning("robot STT provider %s skipped: missing API key", provider)
            return None
        return OpenAICompatibleSpeechToText(
            api_key=api_key,
            base_url=str(getattr(settings, "robot_stt_base_url", "") or "https://api.openai.com/v1"),
            model=str(getattr(settings, "robot_stt_model", "") or "whisper-1"),
        )
    _LOGGER.warning("unknown robot STT provider %s", provider)
    return None


def _build_tts(settings: Settings) -> TextToSpeech | None:
    provider = _provider_name(getattr(settings, "robot_tts_provider", "none"))
    if provider in {"", "none"}:
        return None
    if provider == "fake":
        return FakeTextToSpeech()
    api_key = _secret(getattr(settings, "robot_tts_api_key", None))
    if provider in {"openai_compatible", "dashscope"} and not api_key:
        _LOGGER.warning("robot TTS provider %s skipped: missing API key", provider)
        return None
    base_url = str(getattr(settings, "robot_tts_base_url", "") or "https://api.openai.com/v1")
    model = str(getattr(settings, "robot_tts_model", "") or "tts-1")
    voice = str(getattr(settings, "robot_tts_voice", "") or "alloy")
    response_format = str(getattr(settings, "robot_tts_format", "") or "mp3")
    if provider == "openai_compatible":
        return OpenAICompatibleTextToSpeech(
            api_key=api_key,
            base_url=base_url,
            model=model,
            voice=voice,
            response_format=response_format,
        )
    if provider == "dashscope":
        if not base_url or "openai.com" in base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return DashScopeTextToSpeech(
            api_key=api_key,
            base_url=base_url,
            model=model if model != "tts-1" else "cosyvoice-v2",
            voice=voice if voice != "alloy" else "longxiaochun",
            response_format=response_format,
        )
    _LOGGER.warning("unknown robot TTS provider %s", provider)
    return None


def _provider_name(value: object) -> str:
    return str(value or "none").strip().lower()


def _secret(value: object) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value().strip()
    if isinstance(value, str):
        return value.strip()
    return ""


__all__ = ["build_robot_voice"]
