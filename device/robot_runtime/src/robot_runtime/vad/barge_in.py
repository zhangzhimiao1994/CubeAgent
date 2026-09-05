from __future__ import annotations

from typing import Protocol


class BargeInDetector(Protocol):
    """Detect user speech while assistant is speaking."""

    def reset(self) -> None: ...

    def should_interrupt(self, *, speech_active: bool, assistant_playing: bool) -> bool: ...


class SimpleBargeIn:
    def __init__(self, *, min_speech_frames: int = 3) -> None:
        self._min_speech_frames = min_speech_frames
        self._speech_frames = 0

    def reset(self) -> None:
        self._speech_frames = 0

    def should_interrupt(self, *, speech_active: bool, assistant_playing: bool) -> bool:
        if not assistant_playing:
            self.reset()
            return False
        if speech_active:
            self._speech_frames += 1
        else:
            self._speech_frames = 0
        return self._speech_frames >= self._min_speech_frames
