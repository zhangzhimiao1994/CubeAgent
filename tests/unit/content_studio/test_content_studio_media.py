from __future__ import annotations

import pytest

pytest.skip(
    "superseded by test_content_studio_media_v2.py after preview/final split",
    allow_module_level=True,
)

import json
import math
import shutil
import wave
from pathlib import Path

from PIL import Image

from agent_hub.content_studio.media import (
    AudioClip,
    ClaimReference,
    CommandResult,
    ContentStudioMediaAdapter,
    ContentStudioMediaError,
    MediaTimeline,
    RenderRequest,
    SubtitleCue,
    VisualClip,
)


class FakeMediaRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: tuple[str, ...], *, timeout_seconds: int) -> CommandResult:
        del timeout_seconds
        self.commands.append(command)
        executable = Path(command[0]).name.casefold()
        if executable == "ffprobe":
            return CommandResult(stdout=json.dumps(_ffprobe_payload()), stderr="")
        if executable == "ffmpeg":
            if command[-1] == "-":
                return CommandResult(stdout="", stderr="")
            output = Path(command[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"rendered:{output.name}".encode())
            return CommandResult(stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")


def test_render_returns_playable_local_preview_and_final_metadata(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    runner = FakeMediaRunner()
    adapter = ContentStudioMediaAdapter(runner=runner, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")

    result = adapter.render(_render_request(image=image, audio=audio), tmp_path / "out")

    assert result.preview.path.is_file()
    assert result.final.path.is_file()
    assert result.preview.sha256 == _sha256(result.preview.path)
    assert result.final.sha256 == _sha256(result.final.path)
    assert result.final.mime_type == "video/mp4"
    assert result.final.technical_params["width"] == 1080
    assert result.final.technical_params["height"] == 1920
    assert result.final.technical_params["video_codec"] == "h264"
    assert result.final.technical_params["audio_codec"] == "aac"
    assert result.final.ffprobe["format"]["duration"] == "2.000000"
    ffmpeg_commands = [command for command in runner.commands if Path(command[0]).name == "ffmpeg"]
    assert len(ffmpeg_commands) >= 4
    joined = "\n".join(" ".join(command) for command in ffmpeg_commands)
    assert "scale=1080:1920:force_original_aspect_ratio=decrease" in joined
    assert "subtitles=" in joined
    assert "-c:v libx264" in joined
    assert "-c:a aac" in joined
    assert result.qc.passed is True
    assert result.qc.check("subtitle_safe_area").status == "passed"
    assert result.qc.check("claim_coverage").status == "passed"
    assert result.qc.check("black_frame").status == "passed"
    assert result.qc.check("silence").status == "passed"
    assert result.qc.check("face_identity").status == "not_detected"
    assert result.qc.check("factual_truth").status == "not_detected"


def test_subtitle_safe_area_uses_real_coordinates(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    request = _render_request(
        image=image,
        audio=audio,
        subtitle=SubtitleCue(
            cue_id="SUB001",
            start_ms=0,
            duration_ms=2000,
            text="字幕跑出了安全区",
            x=80,
            y=1840,
            width=920,
            height=120,
            claim_ids=("CLAIM001",),
        ),
    )

    result = ContentStudioMediaAdapter(
        runner=FakeMediaRunner(), ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe"
    ).render(request, tmp_path / "out")

    assert result.qc.passed is False
    safe_area = result.qc.check("subtitle_safe_area")
    assert safe_area.status == "failed"
    assert "SUB001" in safe_area.details


def test_voiceover_audio_must_be_provided_not_synthetic_tts(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    synthetic_voice = tmp_path / "voice.wav"
    synthetic_voice.write_bytes(b"wav")
    request = _render_request(
        image=image,
        audio=synthetic_voice,
        audio_clip=AudioClip(
            clip_id="VOICE001",
            path=synthetic_voice,
            mime_type="audio/wav",
            start_ms=0,
            duration_ms=2000,
            role="voiceover",
            synthetic=True,
            demo_signal=False,
        ),
    )

    with pytest.raises(ContentStudioMediaError, match="voiceover audio must be caller-provided"):
        ContentStudioMediaAdapter(
            runner=FakeMediaRunner(), ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe"
        ).render(request, tmp_path / "out")


def test_claim_coverage_does_not_claim_truth_detection(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    request = _render_request(
        image=image,
        audio=audio,
        claims=(ClaimReference(claim_id="CLAIM002", text="没有字幕覆盖的事实点"),),
    )

    result = ContentStudioMediaAdapter(
        runner=FakeMediaRunner(), ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe"
    ).render(request, tmp_path / "out")

    assert result.qc.passed is False
    assert result.qc.check("claim_coverage").status == "failed"
    assert "CLAIM002" in result.qc.check("claim_coverage").details
    assert result.qc.check("factual_truth").status == "not_detected"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are not installed locally",
)
def test_real_ffmpeg_demo_signal_renders_playable_mp4(tmp_path: Path) -> None:
    image = tmp_path / "demo.png"
    Image.new("RGB", (216, 384), color=(24, 48, 96)).save(image)
    audio = tmp_path / "demo-signal.wav"
    _write_demo_signal_wav(audio, seconds=1)
    request = _render_request(
        image=image,
        audio=audio,
        duration_ms=1000,
        audio_clip=AudioClip(
            clip_id="DEMO_SIGNAL",
            path=audio,
            mime_type="audio/wav",
            start_ms=0,
            duration_ms=1000,
            role="demo_signal",
            synthetic=False,
            demo_signal=True,
        ),
        subtitle=SubtitleCue(
            cue_id="SUB001",
            start_ms=0,
            duration_ms=1000,
            text="Demo signal only",
            x=80,
            y=1500,
            width=920,
            height=180,
            claim_ids=("CLAIM001",),
        ),
    )

    result = ContentStudioMediaAdapter().render(request, tmp_path / "rendered")

    assert result.final.path.is_file()
    assert result.final.size_bytes > 0
    assert result.final.technical_params["width"] == 1080
    assert result.final.technical_params["height"] == 1920
    assert result.final.technical_params["video_codec"] == "h264"
    assert result.final.technical_params["audio_codec"] == "aac"
    assert result.qc.check("demo_signal").status == "warning"
    assert result.qc.check("face_identity").status == "not_detected"
    assert result.qc.check("factual_truth").status == "not_detected"


def _render_request(
    *,
    image: Path,
    audio: Path,
    duration_ms: int = 2000,
    audio_clip: AudioClip | None = None,
    subtitle: SubtitleCue | None = None,
    claims: tuple[ClaimReference, ...] = (
        ClaimReference(claim_id="CLAIM001", text="被字幕覆盖的事实点"),
    ),
) -> RenderRequest:
    return RenderRequest(
        title="Content Studio Demo",
        output_basename="content-studio-demo",
        timeline=MediaTimeline(
            width=1080,
            height=1920,
            duration_ms=duration_ms,
            visuals=(
                VisualClip(
                    clip_id="VIS001",
                    path=image,
                    mime_type="image/png",
                    start_ms=0,
                    duration_ms=duration_ms,
                ),
            ),
            audio=(
                audio_clip
                or AudioClip(
                    clip_id="VOICE001",
                    path=audio,
                    mime_type="audio/wav",
                    start_ms=0,
                    duration_ms=duration_ms,
                    role="voiceover",
                    synthetic=False,
                    demo_signal=False,
                ),
            ),
            subtitles=(
                subtitle
                or SubtitleCue(
                    cue_id="SUB001",
                    start_ms=0,
                    duration_ms=duration_ms,
                    text="这是一条安全区内的字幕",
                    x=80,
                    y=1500,
                    width=920,
                    height=220,
                    claim_ids=("CLAIM001",),
                ),
            ),
            claims=claims,
        ),
    )


def _ffprobe_payload() -> dict[str, object]:
    return {
        "streams": (
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1080,
                "height": 1920,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "30/1",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
            },
        ),
        "format": {"duration": "2.000000", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_demo_signal_wav(path: Path, *, seconds: int) -> None:
    sample_rate = 48_000
    amplitude = 10_000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_rate * seconds):
            sample = int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        output.writeframes(bytes(frames))
