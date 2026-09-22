from __future__ import annotations

import json
import math
import shutil
import wave
from pathlib import Path

import pytest
from PIL import Image

from agent_hub.content_studio.media import (
    AudioClip,
    ClaimReference,
    CommandResult,
    ContentStudioMediaAdapter,
    ContentStudioMediaError,
    FinalRenderApproval,
    MediaTimeline,
    RenderRequest,
    SubtitleCue,
    VisualClip,
)


class FakeMediaRunner:
    def __init__(
        self,
        *,
        width: int = 1080,
        height: int = 1920,
        duration: str = "2.000000",
        silence_stderr: str = "",
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.width = width
        self.height = height
        self.duration = duration
        self.silence_stderr = silence_stderr

    def run(self, command: tuple[str, ...], *, timeout_seconds: int) -> CommandResult:
        del timeout_seconds
        self.commands.append(command)
        executable = Path(command[0]).name.casefold()
        if executable == "ffprobe":
            return CommandResult(stdout=json.dumps(_ffprobe_payload(self.width, self.height, self.duration)), stderr="")
        if executable == "ffmpeg":
            if "-af" in command and any("silencedetect" in item for item in command):
                return CommandResult(stdout="", stderr=self.silence_stderr)
            if command[-1] == "-":
                return CommandResult(stdout="", stderr="")
            output = Path(command[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            if "%03d" in output.name:
                for index in range(1, 4):
                    output.with_name(output.name.replace("%03d", f"{index:03d}")).write_bytes(
                        f"frame:{index}".encode()
                    )
                return CommandResult(stdout="", stderr="")
            output.write_bytes(f"rendered:{output.name}".encode())
            return CommandResult(stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")


def test_preview_render_returns_local_mp4_metadata_without_final(tmp_path: Path) -> None:
    image, audio = _media_files(tmp_path)
    runner = FakeMediaRunner()
    adapter = ContentStudioMediaAdapter(runner=runner, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")

    preview = adapter.render_preview(_request(image=image, audio=audio), tmp_path / "out")

    assert preview.preview.path.is_file()
    assert preview.preview.path.name.endswith("-preview.mp4")
    assert preview.preview.sha256 == _sha256(preview.preview.path)
    assert preview.preview.technical_params["width"] == 1080
    assert preview.preview.technical_params["height"] == 1920
    assert preview.preview.technical_params["video_codec"] == "h264"
    assert preview.preview.technical_params["audio_codec"] == "aac"
    assert preview.qc.technical_passed is True
    assert preview.qc.needs_review is True
    assert preview.qc.check("face_identity").status == "not_detected"
    assert preview.qc.check("factual_truth").status == "not_detected"
    assert not list((tmp_path / "out").glob("*-final.mp4"))
    joined = "\n".join(" ".join(command) for command in runner.commands)
    assert "scale=1080:1920:force_original_aspect_ratio=decrease" in joined
    assert "subtitles=" in joined
    assert "-c:v libx264" in joined
    assert "-c:a aac" in joined


def test_still_image_visuals_use_motion_filter_instead_of_static_hold(tmp_path: Path) -> None:
    image, audio = _media_files(tmp_path)
    runner = FakeMediaRunner()
    adapter = ContentStudioMediaAdapter(runner=runner, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")

    adapter.render_preview(_request(image=image, audio=audio, duration_ms=6000), tmp_path / "out")

    visual_commands = [
        command
        for command in runner.commands
        if Path(command[-1]).name.startswith("visual-0")
    ]
    assert visual_commands
    assert all(command.index("-t") > command.index("-vf") for command in visual_commands)
    filters = " ".join(
        command[command.index("-vf") + 1]
        for command in visual_commands
        if "-vf" in command
    )
    assert "zoompan=" in filters
    assert "fps=30" in filters


def test_qc_fails_when_still_visual_cadence_is_too_slow(tmp_path: Path) -> None:
    image, audio = _media_files(tmp_path)
    adapter = ContentStudioMediaAdapter(
        runner=FakeMediaRunner(duration="6.000000"),
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
    )

    preview = adapter.render_preview(_request(image=image, audio=audio, duration_ms=6000), tmp_path / "out")

    cadence = preview.qc.check("visual_change_cadence")
    assert cadence.status == "failed"
    assert "VIS001" in cadence.details
    assert preview.qc.technical_passed is False


def test_final_render_requires_matching_approval(tmp_path: Path) -> None:
    image, audio = _media_files(tmp_path)
    runner = FakeMediaRunner()
    adapter = ContentStudioMediaAdapter(runner=runner, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")
    request = _request(image=image, audio=audio)
    preview = adapter.render_preview(request, tmp_path / "out")

    with pytest.raises(ContentStudioMediaError, match="approved preview"):
        adapter.render_final(
            request,
            tmp_path / "out",
            approval=FinalRenderApproval(
                approved=True,
                approved_by="reviewer",
                preview_sha256="bad",
                technical_passed=True,
            ),
        )

    final = adapter.render_final(
        request,
        tmp_path / "out",
        approval=FinalRenderApproval(
            approved=True,
            approved_by="reviewer",
            preview_sha256=preview.preview.sha256,
            technical_passed=preview.qc.technical_passed,
        ),
    )

    assert final.final.path.is_file()
    assert final.final.path.name.endswith("-final.mp4")
    assert final.qc.technical_passed is True


def test_pack_dimensions_come_from_timeline_not_global_constant(tmp_path: Path) -> None:
    image, audio = _media_files(tmp_path)
    runner = FakeMediaRunner(width=1920, height=1080)
    request = _request(image=image, audio=audio, width=1920, height=1080)

    preview = ContentStudioMediaAdapter(
        runner=runner, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe"
    ).render_preview(request, tmp_path / "out")

    assert preview.preview.technical_params["width"] == 1920
    assert preview.preview.technical_params["height"] == 1080
    joined = "\n".join(" ".join(command) for command in runner.commands)
    assert "scale=1920:1080:force_original_aspect_ratio=decrease" in joined


def test_timeline_gaps_overlaps_and_extra_audio_are_explicit_errors(tmp_path: Path) -> None:
    image, audio = _media_files(tmp_path)
    second_audio = tmp_path / "voice-2.wav"
    second_audio.write_bytes(b"wav")

    with pytest.raises(ContentStudioMediaError, match="visual timeline must fully cover"):
        ContentStudioMediaAdapter(runner=FakeMediaRunner(), ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe").render_preview(
            _request(
                image=image,
                audio=audio,
                visuals=(
                    VisualClip("VIS001", image, "image/png", 0, 800),
                    VisualClip("VIS002", image, "image/png", 1000, 1000),
                ),
            ),
            tmp_path / "out-gap",
        )

    with pytest.raises(ContentStudioMediaError, match="visual timeline must fully cover"):
        ContentStudioMediaAdapter(runner=FakeMediaRunner(), ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe").render_preview(
            _request(
                image=image,
                audio=audio,
                visuals=(
                    VisualClip("VIS001", image, "image/png", 0, 1200),
                    VisualClip("VIS002", image, "image/png", 1000, 1000),
                ),
            ),
            tmp_path / "out-overlap",
        )

    with pytest.raises(ContentStudioMediaError, match="exactly one audio track"):
        ContentStudioMediaAdapter(runner=FakeMediaRunner(), ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe").render_preview(
            _request(
                image=image,
                audio=audio,
                audio_tracks=(
                    AudioClip("VOICE001", audio, "audio/wav", 0, 2000, source="caller_tts"),
                    AudioClip("MUSIC001", second_audio, "audio/wav", 0, 2000, source="caller_audio"),
                ),
            ),
            tmp_path / "out-audio",
        )


def test_parallel_previews_use_isolated_work_dirs(tmp_path: Path) -> None:
    image, audio = _media_files(tmp_path)
    runner = FakeMediaRunner()
    adapter = ContentStudioMediaAdapter(runner=runner, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")

    first = adapter.render_preview(_request(image=image, audio=audio), tmp_path / "out")
    second = adapter.render_preview(_request(image=image, audio=audio), tmp_path / "out")

    assert first.preview.path != second.preview.path
    work_dirs = {
        Path(command[-1]).parent
        for command in runner.commands
        if Path(command[0]).name == "ffmpeg" and command[-1] != "-"
    }
    assert len({path for path in work_dirs if "content-studio-work-" in path.name}) >= 2


def test_caller_tts_is_allowed_but_demo_signal_is_marked_review(tmp_path: Path) -> None:
    image, audio = _media_files(tmp_path)
    adapter = ContentStudioMediaAdapter(runner=FakeMediaRunner(), ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")

    caller_tts = adapter.render_preview(
        _request(
            image=image,
            audio=audio,
            audio_tracks=(AudioClip("VOICE001", audio, "audio/wav", 0, 2000, source="caller_tts"),),
        ),
        tmp_path / "tts",
    )
    assert caller_tts.qc.check("demo_signal").status == "passed"

    demo = adapter.render_preview(
        _request(
            image=image,
            audio=audio,
            audio_tracks=(AudioClip("DEMO001", audio, "audio/wav", 0, 2000, source="demo_signal"),),
        ),
        tmp_path / "demo",
    )
    assert demo.qc.check("demo_signal").status == "warning"
    assert demo.qc.needs_review is True


def test_silence_qc_ignores_short_narration_pauses(tmp_path: Path) -> None:
    runner = FakeMediaRunner(silence_stderr="[silencedetect] silence_duration: 0.62")
    adapter = ContentStudioMediaAdapter(runner=runner, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")

    check = adapter._silence_check(tmp_path / "preview.mp4", 30)

    assert check.status == "passed"
    silence_commands = [command for command in runner.commands if "-af" in command]
    assert len(silence_commands) == 1
    assert "silencedetect=n=-45dB:d=3.0" in silence_commands[0]


def test_silence_qc_fails_long_silence(tmp_path: Path) -> None:
    runner = FakeMediaRunner(silence_stderr="[silencedetect] silence_start: 12.0")
    adapter = ContentStudioMediaAdapter(runner=runner, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")

    check = adapter._silence_check(tmp_path / "preview.mp4", 30)

    assert check.status == "failed"


def test_video_reviewer_extracts_frames_and_checks_subtitle_text(tmp_path: Path) -> None:
    image, audio = _media_files(tmp_path)
    runner = FakeMediaRunner()
    adapter = ContentStudioMediaAdapter(runner=runner, ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")

    preview = adapter.render_preview(
        _request(
            image=image,
            audio=audio,
            subtitles=(
                SubtitleCue(
                    "SUB001",
                    0,
                    2000,
                    "这是一条带有口口占位符的字幕",
                    80,
                    1500,
                    920,
                    220,
                    ("CLAIM001",),
                ),
            ),
        ),
        tmp_path / "reviewed",
    )

    frame_check = preview.qc.check("video_reviewer_frame_sampling")
    text_check = preview.qc.check("subtitle_text_review")
    assert frame_check.status == "passed"
    assert "3 frames" in frame_check.details
    assert len(preview.qc.review_frames) == 3
    assert all(frame.path.name.startswith("review-frame-") for frame in preview.qc.review_frames)
    assert text_check.status == "failed"
    assert "SUB001" in text_check.details
    assert any(
        "fps=1/" in " ".join(command)
        for command in runner.commands
        if Path(command[0]).name == "ffmpeg"
    )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are not installed locally",
)
def test_real_ffmpeg_demo_signal_preview_is_playable(tmp_path: Path) -> None:
    image = tmp_path / "demo.png"
    Image.new("RGB", (216, 384), color=(24, 48, 96)).save(image)
    audio = tmp_path / "demo-signal.wav"
    _write_demo_signal_wav(audio, seconds=1)
    request = _request(
        image=image,
        audio=audio,
        duration_ms=1000,
        audio_tracks=(AudioClip("DEMO001", audio, "audio/wav", 0, 1000, source="demo_signal"),),
        subtitles=(
            SubtitleCue("SUB001", 0, 1000, "Demo signal only", 80, 1500, 920, 180, ("CLAIM001",)),
        ),
    )

    preview = ContentStudioMediaAdapter().render_preview(request, tmp_path / "rendered")

    assert preview.preview.path.is_file()
    assert preview.preview.size_bytes > 0
    assert preview.preview.technical_params["width"] == 1080
    assert preview.preview.technical_params["height"] == 1920
    assert preview.preview.technical_params["video_codec"] == "h264"
    assert preview.preview.technical_params["audio_codec"] == "aac"
    assert preview.qc.check("demo_signal").status == "warning"
    assert preview.qc.check("face_identity").status == "not_detected"


def _media_files(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    return image, audio


def _request(
    *,
    image: Path,
    audio: Path,
    width: int = 1080,
    height: int = 1920,
    duration_ms: int = 2000,
    visuals: tuple[VisualClip, ...] | None = None,
    audio_tracks: tuple[AudioClip, ...] | None = None,
    subtitles: tuple[SubtitleCue, ...] | None = None,
) -> RenderRequest:
    return RenderRequest(
        title="Content Studio Demo",
        output_basename="content-studio-demo",
        timeline=MediaTimeline(
            width=width,
            height=height,
            duration_ms=duration_ms,
            visuals=visuals
            or (VisualClip("VIS001", image, "image/png", 0, duration_ms),),
            audio=audio_tracks
            or (AudioClip("VOICE001", audio, "audio/wav", 0, duration_ms, source="caller_tts"),),
            subtitles=subtitles
            or (
                SubtitleCue(
                    "SUB001",
                    0,
                    duration_ms,
                    "这是一条安全区内的字幕",
                    80,
                    min(height - 420, 1500),
                    width - 160,
                    220,
                    ("CLAIM001",),
                ),
            ),
            claims=(ClaimReference("CLAIM001", "被字幕覆盖的事实点"),),
        ),
    )


def _ffprobe_payload(width: int, height: int, duration: str) -> dict[str, object]:
    return {
        "streams": (
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": width,
                "height": height,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
            },
        ),
        "format": {"duration": duration, "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
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
