from __future__ import annotations

from typing import Protocol


class AudioPlayback(Protocol):
    """Speaker playback queue."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def enqueue(self, pcm: bytes) -> None: ...

    async def clear(self) -> None:
        """Drop queued audio; used by barge-in."""
        ...

    @property
    def is_playing(self) -> bool: ...


class NullAudioPlayback:
    def __init__(self) -> None:
        self._playing = False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self._playing = False

    async def enqueue(self, pcm: bytes) -> None:
        del pcm
        self._playing = True

    async def clear(self) -> None:
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing
