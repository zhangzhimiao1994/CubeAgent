from __future__ import annotations

from typing import Protocol


class EchoCanceller(Protocol):
    def process(self, mic_frame: bytes, reference_frame: bytes | None = None) -> bytes: ...


class PassthroughEchoCanceller:
    def process(self, mic_frame: bytes, reference_frame: bytes | None = None) -> bytes:
        del reference_frame
        return mic_frame
