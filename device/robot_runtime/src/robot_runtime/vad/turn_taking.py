from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class TurnDecision:
    speech_active: bool
    turn_ended: bool


class TurnTaking(Protocol):
    """Decide whether the user finished speaking."""

    def reset(self) -> None: ...

    def observe(self, pcm_frame: bytes, *, energy: float | None = None) -> TurnDecision: ...


class EnergyTurnTaking:
    """Simple energy VAD stub for bring-up on Pi."""

    def __init__(self, *, silence_frames_to_end: int = 12, energy_threshold: float = 500.0) -> None:
        self._silence_frames_to_end = silence_frames_to_end
        self._energy_threshold = energy_threshold
        self._seen_speech = False
        self._silence = 0

    def reset(self) -> None:
        self._seen_speech = False
        self._silence = 0

    def observe(self, pcm_frame: bytes, *, energy: float | None = None) -> TurnDecision:
        del pcm_frame
        level = 0.0 if energy is None else energy
        if level >= self._energy_threshold:
            self._seen_speech = True
            self._silence = 0
            return TurnDecision(speech_active=True, turn_ended=False)
        if not self._seen_speech:
            return TurnDecision(speech_active=False, turn_ended=False)
        self._silence += 1
        ended = self._silence >= self._silence_frames_to_end
        if ended:
            self.reset()
        return TurnDecision(speech_active=False, turn_ended=ended)
