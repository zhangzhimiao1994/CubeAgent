from __future__ import annotations


class VoiceProviderError(Exception):
    """Safe provider failure; never include secrets or raw HTTP bodies."""


__all__ = ["VoiceProviderError"]
