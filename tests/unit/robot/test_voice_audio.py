from __future__ import annotations

from agent_hub.robot.voice.audio import decode_audio_b64, pcm16_to_wav, wav_payload_size


def test_pcm16_to_wav_writes_riff_header_and_payload() -> None:
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    wav = pcm16_to_wav(pcm, sample_rate_hz=16000, channels=1)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert wav[36:40] == b"data"
    assert wav_payload_size(wav) == len(pcm)
    assert wav[44:] == pcm


def test_decode_audio_b64_accepts_standard_and_urlsafe() -> None:
    raw = b"hello-audio"
    assert decode_audio_b64("aGVsbG8tYXVkaW8=") == raw
    assert decode_audio_b64("aGVsbG8tYXVkaW8") == raw
    assert decode_audio_b64("not-base64!!!") is None
    assert decode_audio_b64("") is None
