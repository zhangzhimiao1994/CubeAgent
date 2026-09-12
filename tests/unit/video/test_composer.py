from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_hub.video.composer import (
    VideoClipInput,
    VideoComposer,
    VideoComposeRequest,
    VideoCompositionError,
)


def test_composer_requires_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mp4")
    monkeypatch.setattr("agent_hub.video.composer.shutil.which", lambda name: None)

    request = VideoComposeRequest(
        title="Trial",
        clips=(VideoClipInput(storage_key="source/run/clip", path=clip, mime_type="video/mp4"),),
        output_filename="trial.mp4",
    )

    with pytest.raises(VideoCompositionError, match="ffmpeg is not installed"):
        VideoComposer().compose(request, tmp_path)


def test_composer_rejects_unsupported_mime_type(tmp_path: Path) -> None:
    clip = tmp_path / "clip.gif"
    clip.write_bytes(b"gif")
    request = VideoComposeRequest(
        title="Trial",
        clips=(VideoClipInput(storage_key="source/run/clip", path=clip, mime_type="image/gif"),),
        output_filename="trial.mp4",
    )

    with pytest.raises(VideoCompositionError, match="unsupported clip MIME type"):
        VideoComposer(ffmpeg_binary="ffmpeg").compose(request, tmp_path)


def test_composer_normalizes_segments_and_concatenates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"mp4")
    calls: list[tuple[str, ...]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, timeout
        calls.append(tuple(command))
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"composed")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("agent_hub.video.composer.shutil.which", lambda name: "ffmpeg")
    monkeypatch.setattr("agent_hub.video.composer.subprocess.run", fake_run)

    request = VideoComposeRequest(
        title="Trial",
        clips=(
            VideoClipInput(
                storage_key="tenant/run/image",
                path=image,
                mime_type="image/png",
                duration_seconds=2,
            ),
            VideoClipInput(storage_key="tenant/run/video", path=video, mime_type="video/mp4"),
        ),
        output_filename="trial.mp4",
        aspect_ratio="9:16",
        image_duration_seconds=3,
    )

    output = VideoComposer().compose(request, tmp_path / "out")

    assert output.read_bytes() == b"composed"
    assert output.name == "trial.mp4"
    assert len(calls) == 3
    assert calls[0][:2] == ("ffmpeg", "-y")
    assert "-loop" in calls[0]
    assert "-t" in calls[0]
    assert "2" in calls[0]
    assert "-c:v" in calls[0]
    assert "libx264" in calls[0]
    assert "-movflags" in calls[0]
    assert "+faststart" in calls[0]
    assert "-c:v" in calls[1]
    assert "libx264" in calls[1]
    assert "-an" in calls[1]
    assert calls[-1][:2] == ("ffmpeg", "-y")
    assert "-f" in calls[-1]
    assert "concat" in calls[-1]


def test_composer_original_aspect_ratio_still_uses_common_canvas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.mp4"
    first.write_bytes(b"first")
    second = tmp_path / "second.mp4"
    second.write_bytes(b"second")
    calls: list[tuple[str, ...]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, timeout
        calls.append(tuple(command))
        Path(command[-1]).write_bytes(b"segment")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("agent_hub.video.composer.shutil.which", lambda name: "ffmpeg")
    monkeypatch.setattr("agent_hub.video.composer.subprocess.run", fake_run)

    request = VideoComposeRequest(
        title="Trial",
        clips=(
            VideoClipInput(storage_key="tenant/run/first", path=first, mime_type="video/mp4"),
            VideoClipInput(storage_key="tenant/run/second", path=second, mime_type="video/mp4"),
        ),
        output_filename="trial.mp4",
    )

    VideoComposer().compose(request, tmp_path / "out")

    assert "-vf" in calls[0]
    assert any("scale=1280:720:force_original_aspect_ratio=decrease" in arg for arg in calls[0])
    assert "-vf" in calls[1]
    assert any("scale=1280:720:force_original_aspect_ratio=decrease" in arg for arg in calls[1])
