from __future__ import annotations

from enum import StrEnum


class LoopState(StrEnum):
    IDLE = "idle"
    HEARING = "hearing"
    CAPTURING = "capturing"
    THINKING = "thinking"
    SPEAKING = "speaking"


class InteractionLoop:
    """Listen → turn-take → think → speak, with barge-in."""

    def __init__(self) -> None:
        self.state = LoopState.IDLE

    def on_speech_start(self) -> None:
        if self.state in {LoopState.IDLE, LoopState.HEARING, LoopState.SPEAKING}:
            self.state = LoopState.CAPTURING

    def on_turn_end(self) -> None:
        if self.state is LoopState.CAPTURING:
            self.state = LoopState.THINKING

    def on_assistant_start(self) -> None:
        if self.state is LoopState.THINKING:
            self.state = LoopState.SPEAKING

    def on_assistant_end(self) -> None:
        if self.state is LoopState.SPEAKING:
            self.state = LoopState.IDLE

    def on_barge_in(self) -> None:
        if self.state is LoopState.SPEAKING:
            self.state = LoopState.CAPTURING
