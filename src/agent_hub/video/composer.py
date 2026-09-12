from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agent_hub.files.generated import safe_generated_filename

VIDEO_MP4_MIME_TYPE = "video/mp4"
SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
SUPPORTED_CLIP_MIME_TYPES = SUPPORTED_IMAGE_MIME_TYPES | frozenset({VIDEO_MP4_MIME_TYPE})
SUPPORTED_ASPECT_RATIOS = frozenset({"original", "16:9", "9:16"})
MAX_CLIPS = 32
MIN_IMAGE_DURATION_SECONDS = 1
MAX_IMAGE_DURATION_SECONDS = 10
DEFAULT_IMAGE_DURATION_SECONDS = 3
DEFAULT_TIMEOUT_SECONDS = 300


class VideoCompositionError(RuntimeError):
    """Raised when video composition cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class VideoClipInput:
    storage_key: str
    path: Path
    mime_type: str
    duration_seconds: int | None = None
    filename: str | None = None

    def __post_init__(self) -> None:
        if type(self.storage_key) is not str or not self.storage_key.strip():
            raise VideoCompositionError("clip storage_key must be nonblank")
        if not isinstance(self.path, Path):
            raise VideoCompositionError("clip path must be a Path")
        if type(self.mime_type) is not str or not self.mime_type.strip():
            raise VideoCompositionError("clip mime_type must be nonblank")
        if self.duration_seconds is not None:
            _validate_image_duration(self.duration_seconds)
        if self.filename is not None:
            safe_generated_filename(self.filename)


@dataclass(frozen=True, slots=True)
class VideoComposeRequest:
    title: str
    clips: tuple[VideoClipInput, ...]
    output_filename: str
    aspect_ratio: str = "original"
    image_duration_seconds: int = DEFAULT_IMAGE_DURATION_SECONDS
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if type(self.title) is not str or not self.title.strip():
            raise VideoCompositionError("title must be nonblank")
        if not self.clips or len(self.clips) > MAX_CLIPS:
            raise VideoCompositionError("clips must contain 1 to 32 entries")
        if not all(isinstance(clip, VideoClipInput) for clip in self.clips):
            raise VideoCompositionError("clips must contain only video clip inputs")
        safe_generated_filename(self.output_filename)
        if not self.output_filename.casefold().endswith(".mp4"):
            raise VideoCompositionError("output filename must end in .mp4")
        if self.aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
            raise VideoCompositionError("aspect_ratio must be original, 16:9, or 9:16")
        _validate_image_duration(self.image_duration_seconds)
        if (
            type(self.timeout_seconds) is not int
            or self.timeout_seconds < 1
            or self.timeout_seconds > 1800
        ):
            raise VideoCompositionError("timeout_seconds must be between 1 and 1800")


class VideoComposer:
    def __init__(self, *, ffmpeg_binary: str | None = None) -> None:
        self._ffmpeg_binary = ffmpeg_binary

    def compose(self, request: VideoComposeRequest, output_dir: Path) -> Path:
        if not isinstance(output_dir, Path):
            raise VideoCompositionError("output_dir must be a Path")
        ffmpeg = self._resolve_ffmpeg()
        _validate_clip_files(request.clips)
        output_dir.mkdir(parents=True, exist_ok=True)
        segment_dir = output_dir / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)

        segment_paths = [
            self._normalize_segment(ffmpeg, request, clip, segment_dir, index)
            for index, clip in enumerate(request.clips)
        ]
        concat_file = output_dir / "concat.txt"
        concat_file.write_text(
            "".join(f"file {shlex.quote(path.as_posix())}\n" for path in segment_paths),
            encoding="utf-8",
        )
        output = output_dir / request.output_filename
        self._run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output),
            ],
            timeout=request.timeout_seconds,
        )
        if not output.is_file():
            raise VideoCompositionError("ffmpeg did not produce an output video")
        return output

    def _resolve_ffmpeg(self) -> str:
        if self._ffmpeg_binary is not None:
            return self._ffmpeg_binary
        found = shutil.which("ffmpeg")
        if found is None:
            raise VideoCompositionError("ffmpeg is not installed")
        return found

    def _normalize_segment(
        self,
        ffmpeg: str,
        request: VideoComposeRequest,
        clip: VideoClipInput,
        segment_dir: Path,
        index: int,
    ) -> Path:
        _validate_clip_mime_type(clip.mime_type)
        output = segment_dir / f"segment-{index:03d}.mp4"
        if clip.mime_type in SUPPORTED_IMAGE_MIME_TYPES:
            duration = clip.duration_seconds or request.image_duration_seconds
            command = [
                ffmpeg,
                "-y",
                "-loop",
                "1",
                "-t",
                str(duration),
                "-i",
                str(clip.path),
                *self._video_filter_args(request.aspect_ratio),
                "-r",
                "30",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ]
        else:
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(clip.path),
                *self._video_filter_args(request.aspect_ratio),
                "-r",
                "30",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(output),
            ]
        self._run(command, timeout=request.timeout_seconds)
        return output

    def _video_filter_args(self, aspect_ratio: str) -> list[str]:
        if aspect_ratio == "9:16":
            return [
                "-vf",
                (
                    "scale=1080:1920:force_original_aspect_ratio=decrease,"
                    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1"
                ),
            ]
        if aspect_ratio == "16:9":
            return [
                "-vf",
                (
                    "scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1"
                ),
            ]
        return [
            "-vf",
            (
                "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1"
            ),
        ]

    def _run(self, command: list[str], *, timeout: int) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise VideoCompositionError("ffmpeg composition timed out") from error
        except subprocess.CalledProcessError as error:
            stderr = _clean_stderr(error.stderr)
            message = "ffmpeg composition failed"
            if stderr:
                message = f"{message}: {stderr}"
            raise VideoCompositionError(message) from error


def _validate_image_duration(value: int) -> None:
    if (
        type(value) is not int
        or value < MIN_IMAGE_DURATION_SECONDS
        or value > MAX_IMAGE_DURATION_SECONDS
    ):
        raise VideoCompositionError("image duration must be between 1 and 10 seconds")


def _validate_clip_files(clips: tuple[VideoClipInput, ...]) -> None:
    for clip in clips:
        if not clip.path.is_file():
            raise VideoCompositionError("clip source file does not exist")


def _validate_clip_mime_type(mime_type: str) -> None:
    if mime_type not in SUPPORTED_CLIP_MIME_TYPES:
        raise VideoCompositionError("unsupported clip MIME type")


def _clean_stderr(value: bytes | str | None) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    return " ".join(text.strip().split())[:500]


__all__ = [
    "DEFAULT_IMAGE_DURATION_SECONDS",
    "MAX_CLIPS",
    "SUPPORTED_CLIP_MIME_TYPES",
    "SUPPORTED_IMAGE_MIME_TYPES",
    "VIDEO_MP4_MIME_TYPE",
    "VideoClipInput",
    "VideoComposeRequest",
    "VideoComposer",
    "VideoCompositionError",
]
