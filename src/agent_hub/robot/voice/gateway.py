from __future__ import annotations

from dataclasses import dataclass

from agent_hub.robot.voice.types import SpeechToText, TextToSpeech


@dataclass(slots=True)
class RobotVoiceGateway:
    stt: SpeechToText | None = None
    tts: TextToSpeech | None = None

    @property
    def can_transcribe(self) -> bool:
        return self.stt is not None

    @property
    def can_speak(self) -> bool:
        return self.tts is not None


__all__ = ["RobotVoiceGateway"]
