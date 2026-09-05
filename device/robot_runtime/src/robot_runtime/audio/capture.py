from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class AudioCapture(Protocol):
    """Microphone capture for Raspberry Pi."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def frames(self) -> AsyncIterator[bytes]:
        """Yield PCM frames (mono s16le by default)."""
        ...


class NullAudioCapture:
    """Placeholder until ALSA/PortAudio is wired."""

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def frames(self) -> AsyncIterator[bytes]:
        if False:
            yield b""
