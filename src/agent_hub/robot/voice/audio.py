from __future__ import annotations

import base64
import binascii
import re
import struct

from agent_hub.robot.voice.types import AudioFormat

_B64_STD = re.compile(r"^[A-Za-z0-9+/]+=*$")
_B64_URL = re.compile(r"^[A-Za-z0-9_-]+=*$")


def pcm16_to_wav(pcm: bytes, *, sample_rate_hz: int = 16000, channels: int = 1) -> bytes:
    byte_rate = sample_rate_hz * channels * 2
    block_align = channels * 2
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate_hz,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return header + pcm


def wav_payload_size(wav: bytes) -> int:
    if len(wav) < 44:
        return 0
    return int.from_bytes(wav[40:44], "little")


def decode_audio_b64(value: str) -> bytes | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    padded = cleaned + "=" * ((4 - len(cleaned) % 4) % 4)
    if _B64_STD.fullmatch(padded):
        try:
            return base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            return None
    if _B64_URL.fullmatch(padded):
        try:
            return base64.urlsafe_b64decode(padded)
        except (binascii.Error, ValueError):
            return None
    return None


def parse_audio_format(value: object) -> AudioFormat:
    raw = str(value or AudioFormat.PCM16).strip().lower()
    try:
        return AudioFormat(raw)
    except ValueError:
        return AudioFormat.PCM16


__all__ = [
    "decode_audio_b64",
    "parse_audio_format",
    "pcm16_to_wav",
    "wav_payload_size",
]
