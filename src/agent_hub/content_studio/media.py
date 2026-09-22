"""Local media adapter for Content Studio preview/final MP4 rendering.

This module is intentionally standalone: it does not register routes, change
runtime defaults, call paid providers, or synthesize voiceover audio. Callers
provide timeline media files; this adapter renders local MP4 artifacts and
returns technical metadata plus bounded QC results.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4 as _uuid4

from agent_hub.files.generated import safe_generated_filename
from agent_hub.video.composer import (
    VideoClipInput as _VideoClipInput,
)
from agent_hub.video.composer import (
    VideoComposer as _VideoComposer,
)
from agent_hub.video.composer import (
    VideoComposeRequest as _VideoComposeRequest,
)

MP4_MIME_TYPE = "video/mp4"
SUPPORTED_VISUAL_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "video/mp4"})
SUPPORTED_AUDIO_MIME_TYPES = frozenset({"audio/wav", "audio/mpeg", "audio/aac", "audio/mp4"})
PACK_WIDTH = 1080
PACK_HEIGHT = 1920
FPS = 30
DEFAULT_TIMEOUT_SECONDS = 300
SILENCE_QC_MIN_SECONDS = 3.0
_SAFE_AREA_MARGIN_X = 54
_SAFE_AREA_MARGIN_TOP = 96
_SAFE_AREA_MARGIN_BOTTOM = 160


class ContentStudioMediaError(RuntimeError):
    """Raised when a local Content Studio media render cannot be completed."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    stdout: str
    stderr: str


class MediaCommandRunner(Protocol):
    def run(self, command: tuple[str, ...], *, timeout_seconds: int) -> CommandResult: ...


class SubprocessMediaRunner:
    def run(self, command: tuple[str, ...], *, timeout_seconds: int) -> CommandResult:
        try:
            completed = subprocess.run(
                list(command),
                check=True,
                capture_output=True,
                timeout=timeout_seconds,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as error:
            raise ContentStudioMediaError("media command timed out") from error
        except subprocess.CalledProcessError as error:
            stderr = _clean_text(error.stderr)
            message = "media command failed"
            if stderr:
                message = f"{message}: {stderr}"
            raise ContentStudioMediaError(message) from error
        return CommandResult(stdout=completed.stdout, stderr=completed.stderr)


@dataclass(frozen=True, slots=True)
class _LegacyVisualClip:
    clip_id: str
    path: Path
    mime_type: str
    start_ms: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class _LegacyAudioClip:
    clip_id: str
    path: Path
    mime_type: str
    start_ms: int
    duration_ms: int
    role: str
    synthetic: bool = False
    demo_signal: bool = False


@dataclass(frozen=True, slots=True)
class _LegacySubtitleCue:
    cue_id: str
    start_ms: int
    duration_ms: int
    text: str
    x: int
    y: int
    width: int
    height: int
    claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _LegacyClaimReference:
    claim_id: str
    text: str


@dataclass(frozen=True, slots=True)
class _LegacyMediaTimeline:
    width: int
    height: int
    duration_ms: int
    visuals: tuple[_LegacyVisualClip, ...]
    audio: tuple[_LegacyAudioClip, ...]
    subtitles: tuple[_LegacySubtitleCue, ...]
    claims: tuple[_LegacyClaimReference, ...] = ()


@dataclass(frozen=True, slots=True)
class _LegacyRenderRequest:
    title: str
    output_basename: str
    timeline: _LegacyMediaTimeline
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class _LegacyMediaArtifact:
    path: Path
    mime_type: str
    sha256: str
    size_bytes: int
    ffprobe: Mapping[str, object]
    technical_params: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _LegacyMediaQCCheck:
    name: str
    status: str
    details: str


@dataclass(frozen=True, slots=True)
class _LegacyMediaQCResult:
    checks: tuple[MediaQCCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.status in {"passed", "warning", "not_detected"} for check in self.checks)

    def check(self, name: str) -> MediaQCCheck:
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class RenderResult:
    preview: MediaArtifact
    final: MediaArtifact
    qc: MediaQCResult


class _LegacyOneShotContentStudioMediaAdapter:
    """Render Content Studio timelines to local preview and final MP4 files."""

    def __init__(
        self,
        *,
        runner: MediaCommandRunner | None = None,
        ffmpeg_binary: str | None = None,
        ffprobe_binary: str | None = None,
    ) -> None:
        self._runner = runner or SubprocessMediaRunner()
        self._ffmpeg_binary = ffmpeg_binary
        self._ffprobe_binary = ffprobe_binary

    def _legacy_one_shot_render_removed(
        self, request: RenderRequest, output_dir: Path
    ) -> RenderResult:
        _validate_request(request)
        ffmpeg = self._resolve_binary(self._ffmpeg_binary, "ffmpeg")
        ffprobe = self._resolve_binary(self._ffprobe_binary, "ffprobe")
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = output_dir / "content-studio-work"
        work_dir.mkdir(parents=True, exist_ok=True)
        subtitle_path = _write_ass_subtitles(request.timeline.subtitles, work_dir / "subtitles.ass")
        visual_path = self._render_visual_track(ffmpeg, request, work_dir)
        basename = safe_generated_filename(request.output_basename)
        preview_path = output_dir / f"{basename}-preview.mp4"
        final_path = output_dir / f"{basename}-final.mp4"
        self._mux_with_audio_and_subtitles(
            ffmpeg,
            request,
            visual_path,
            subtitle_path,
            preview_path,
            crf="28",
        )
        self._mux_with_audio_and_subtitles(
            ffmpeg,
            request,
            visual_path,
            subtitle_path,
            final_path,
            crf="20",
        )
        preview = self._artifact_for(ffprobe, preview_path, request.timeout_seconds)
        final = self._artifact_for(ffprobe, final_path, request.timeout_seconds)
        qc = self._qc(ffmpeg, request, final)
        return RenderResult(preview=preview, final=final, qc=qc)

    def _render_visual_track(self, ffmpeg: str, request: RenderRequest, work_dir: Path) -> Path:
        segment_paths: list[Path] = []
        for index, visual in enumerate(request.timeline.visuals):
            segment = work_dir / f"visual-{index:03d}.mp4"
            duration = f"{visual.duration_ms / 1000:.3f}"
            filter_expr = _canvas_filter()
            if visual.mime_type == "video/mp4":
                command: tuple[str, ...] = (
                    ffmpeg,
                    "-y",
                    "-i",
                    str(visual.path),
                    "-t",
                    duration,
                    "-vf",
                    filter_expr,
                    "-r",
                    str(FPS),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(segment),
                )
            else:
                command = (
                    ffmpeg,
                    "-y",
                    "-loop",
                    "1",
                    "-t",
                    duration,
                    "-i",
                    str(visual.path),
                    "-vf",
                    filter_expr,
                    "-r",
                    str(FPS),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(segment),
                )
            self._runner.run(command, timeout_seconds=request.timeout_seconds)
            segment_paths.append(segment)
        concat_file = work_dir / "visual-concat.txt"
        concat_file.write_text(
            "".join(f"file '{_ffconcat_path(path)}'\n" for path in segment_paths),
            encoding="utf-8",
        )
        visual_track = work_dir / "visual-track.mp4"
        self._runner.run(
            (
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
                str(visual_track),
            ),
            timeout_seconds=request.timeout_seconds,
        )
        return visual_track

    def _mux_with_audio_and_subtitles(
        self,
        ffmpeg: str,
        request: RenderRequest,
        visual_path: Path,
        subtitle_path: Path,
        output_path: Path,
        *,
        crf: str,
    ) -> None:
        audio = request.timeline.audio[0]
        command = (
            ffmpeg,
            "-y",
            "-i",
            str(visual_path),
            "-i",
            str(audio.path),
            "-vf",
            f"{_canvas_filter()},subtitles={_ffmpeg_filter_path(subtitle_path)}",
            "-t",
            f"{request.timeline.duration_ms / 1000:.3f}",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        )
        self._runner.run(command, timeout_seconds=request.timeout_seconds)

    def _artifact_for(self, ffprobe: str, path: Path, timeout_seconds: int) -> MediaArtifact:
        if not path.is_file():
            raise ContentStudioMediaError("renderer did not produce a local playable file")
        ffprobe_payload = self._ffprobe(ffprobe, path, timeout_seconds)
        return MediaArtifact(
            path=path,
            mime_type=MP4_MIME_TYPE,
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
            ffprobe=ffprobe_payload,
            technical_params=_technical_params(ffprobe_payload),
        )

    def _ffprobe(self, ffprobe: str, path: Path, timeout_seconds: int) -> Mapping[str, object]:
        result = self._runner.run(
            (
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ),
            timeout_seconds=timeout_seconds,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ContentStudioMediaError("ffprobe did not return JSON") from error
        if not isinstance(payload, Mapping):
            raise ContentStudioMediaError("ffprobe result is invalid")
        return payload

    def _qc(self, ffmpeg: str, request: RenderRequest, artifact: MediaArtifact) -> MediaQCResult:
        checks = [
            _expected_encoding_check(artifact),
            _subtitle_safe_area_check(request.timeline.subtitles, request.timeline.width, request.timeline.height),
            _claim_coverage_check(request.timeline.claims, request.timeline.subtitles),
            self._black_frame_check(ffmpeg, artifact.path, request.timeout_seconds),
            self._silence_check(ffmpeg, artifact.path, request.timeout_seconds),
            _demo_signal_check(request.timeline.audio),
            MediaQCCheck(
                name="face_identity",
                status="not_detected",
                details="人脸/人物身份一致性未由本地媒体适配器检测。",
            ),
            MediaQCCheck(
                name="factual_truth",
                status="not_detected",
                details="事实真实性未由本地媒体适配器检测；仅检查字幕是否覆盖声明 ID。",
            ),
        ]
        return MediaQCResult(checks=tuple(checks))

    def _black_frame_check(self, ffmpeg: str, path: Path, timeout_seconds: int) -> MediaQCCheck:
        result = self._runner.run(
            (
                ffmpeg,
                "-v",
                "info",
                "-i",
                str(path),
                "-vf",
                "blackdetect=d=0.25:pix_th=0.10",
                "-an",
                "-f",
                "null",
                "-",
            ),
            timeout_seconds=timeout_seconds,
        )
        text = f"{result.stdout}\n{result.stderr}"
        if "black_start:" in text:
            return MediaQCCheck("black_frame", "failed", _clean_text(text))
        return MediaQCCheck("black_frame", "passed", "未检测到超过阈值的黑帧区间。")

    def _silence_check(self, ffmpeg: str, path: Path, timeout_seconds: int) -> MediaQCCheck:
        result = self._runner.run(
            (
                ffmpeg,
                "-v",
                "info",
                "-i",
                str(path),
                "-af",
                f"silencedetect=n=-45dB:d={SILENCE_QC_MIN_SECONDS}",
                "-f",
                "null",
                "-",
            ),
            timeout_seconds=timeout_seconds,
        )
        text = f"{result.stdout}\n{result.stderr}"
        if "silence_start:" in text:
            return MediaQCCheck("silence", "failed", _clean_text(text))
        return MediaQCCheck("silence", "passed", "未检测到超过阈值的静音区间。")

    def _resolve_binary(self, configured: str | None, name: str) -> str:
        if configured is not None:
            return configured
        found = shutil.which(name)
        if found is None:
            raise ContentStudioMediaError(f"{name} is not installed")
        return found


def _validate_request(request: RenderRequest) -> None:
    if not isinstance(request, RenderRequest):
        raise ContentStudioMediaError("request must be RenderRequest")
    if not request.title.strip():
        raise ContentStudioMediaError("title must be nonblank")
    safe_generated_filename(request.output_basename)
    timeline = request.timeline
    if timeline.width != PACK_WIDTH or timeline.height != PACK_HEIGHT:
        raise ContentStudioMediaError("timeline must match the locked 1080x1920 Pack canvas")
    if type(timeline.duration_ms) is not int or timeline.duration_ms <= 0:
        raise ContentStudioMediaError("timeline duration must be positive")
    if not timeline.visuals:
        raise ContentStudioMediaError("timeline requires at least one visual clip")
    if not timeline.audio:
        raise ContentStudioMediaError("timeline requires caller-provided audio")
    if not timeline.subtitles:
        raise ContentStudioMediaError("timeline requires burned subtitle cues")
    for visual in timeline.visuals:
        _validate_clip_time(visual.start_ms, visual.duration_ms)
        _validate_existing_file(visual.path, "visual")
        if visual.mime_type not in SUPPORTED_VISUAL_MIME_TYPES:
            raise ContentStudioMediaError("unsupported visual clip MIME type")
    for audio in timeline.audio:
        _validate_clip_time(audio.start_ms, audio.duration_ms)
        _validate_existing_file(audio.path, "audio")
        if audio.mime_type not in SUPPORTED_AUDIO_MIME_TYPES:
            raise ContentStudioMediaError("unsupported audio clip MIME type")
        if audio.source == "demo_signal":
            raise ContentStudioMediaError("voiceover audio must be caller-provided")
    for cue in timeline.subtitles:
        _validate_clip_time(cue.start_ms, cue.duration_ms)
        if not cue.text.strip():
            raise ContentStudioMediaError("subtitle text must be nonblank")
    if (
        type(request.timeout_seconds) is not int
        or request.timeout_seconds < 1
        or request.timeout_seconds > 1800
    ):
        raise ContentStudioMediaError("timeout_seconds must be between 1 and 1800")


def _validate_clip_time(start_ms: int, duration_ms: int) -> None:
    if type(start_ms) is not int or start_ms < 0:
        raise ContentStudioMediaError("clip start_ms must be a non-negative integer")
    if type(duration_ms) is not int or duration_ms <= 0:
        raise ContentStudioMediaError("clip duration_ms must be a positive integer")


def _validate_existing_file(path: Path, label: str) -> None:
    if not isinstance(path, Path) or not path.is_file():
        raise ContentStudioMediaError(f"{label} file is unavailable")


def _canvas_filter() -> str:
    return (
        f"scale={PACK_WIDTH}:{PACK_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={PACK_WIDTH}:{PACK_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )


def _write_ass_subtitles(cues: Sequence[SubtitleCue], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {PACK_WIDTH}",
        f"PlayResY: {PACK_HEIGHT}",
        "",
        "[V4+ Styles]",
        (
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding"
        ),
        (
            "Style: Default,Arial,56,&H00FFFFFF,&H00FFFFFF,&H00111111,&H88000000,"
            "0,0,0,0,100,100,0,0,1,3,1,2,80,80,160,1"
        ),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for cue in cues:
        box_path = f"m 0 0 l {cue.width} 0 l {cue.width} {cue.height} l 0 {cue.height}"
        style = "{\\an7" f"\\pos({cue.x},{cue.y})" "\\p1}" f"{box_path}" "{\\p0}"
        text = _escape_ass_text(cue.text)
        lines.append(
            "Dialogue: 0,"
            f"{_ass_time(cue.start_ms)},{_ass_time(cue.start_ms + cue.duration_ms)},"
            f"Default,{cue.cue_id},0,0,0,,{style}\\N{{\\an7\\pos({cue.x + 24},{cue.y + 24})}}{text}"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _ass_time(ms: int) -> str:
    total_cs = ms // 10
    cs = total_cs % 100
    total_seconds = total_cs // 100
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def _escape_ass_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _ffconcat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def _ffmpeg_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    value = value.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return f"'{value}'"


def _technical_params(ffprobe: Mapping[str, object]) -> Mapping[str, object]:
    video = _stream(ffprobe, "video")
    audio = _stream(ffprobe, "audio")
    return {
        "width": _int_from_mapping(video, "width"),
        "height": _int_from_mapping(video, "height"),
        "video_codec": str(video.get("codec_name", "")),
        "audio_codec": str(audio.get("codec_name", "")),
        "pix_fmt": str(video.get("pix_fmt", "")),
        "sample_rate": str(audio.get("sample_rate", "")),
        "channels": _int_from_mapping(audio, "channels"),
    }


def _stream(ffprobe: Mapping[str, object], kind: str) -> Mapping[str, object]:
    streams = ffprobe.get("streams")
    if not isinstance(streams, Sequence):
        return {}
    for stream in streams:
        if isinstance(stream, Mapping) and stream.get("codec_type") == kind:
            return stream
    return {}


def _int_from_mapping(value: Mapping[str, object], key: str) -> int | None:
    raw = value.get(key)
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


def _expected_encoding_check(artifact: MediaArtifact) -> MediaQCCheck:
    params = artifact.technical_params
    failures: list[str] = []
    if params.get("width") != PACK_WIDTH:
        failures.append("width")
    if params.get("height") != PACK_HEIGHT:
        failures.append("height")
    if params.get("video_codec") != "h264":
        failures.append("video_codec")
    if params.get("audio_codec") != "aac":
        failures.append("audio_codec")
    if failures:
        return MediaQCCheck("encoding", "failed", "编码参数不符合 Pack 要求: " + ",".join(failures))
    return MediaQCCheck("encoding", "passed", "1080x1920 H.264/AAC 编码参数符合要求。")


def _subtitle_safe_area_check(
    subtitles: Sequence[SubtitleCue],
    width: int,
    height: int,
) -> MediaQCCheck:
    violations: list[str] = []
    min_x = _SAFE_AREA_MARGIN_X
    max_x = width - _SAFE_AREA_MARGIN_X
    min_y = _SAFE_AREA_MARGIN_TOP
    max_y = height - _SAFE_AREA_MARGIN_BOTTOM
    for cue in subtitles:
        if (
            cue.x < min_x
            or cue.y < min_y
            or cue.x + cue.width > max_x
            or cue.y + cue.height > max_y
        ):
            violations.append(cue.cue_id)
    if violations:
        return MediaQCCheck(
            "subtitle_safe_area",
            "failed",
            "字幕越过安全区: " + ", ".join(violations),
        )
    return MediaQCCheck("subtitle_safe_area", "passed", "所有字幕坐标均位于安全区内。")


def _claim_coverage_check(
    claims: Sequence[ClaimReference],
    subtitles: Sequence[SubtitleCue],
) -> MediaQCCheck:
    covered = {claim_id for cue in subtitles for claim_id in cue.claim_ids}
    missing = [claim.claim_id for claim in claims if claim.claim_id not in covered]
    if missing:
        return MediaQCCheck("claim_coverage", "failed", "字幕未覆盖声明 ID: " + ", ".join(missing))
    return MediaQCCheck("claim_coverage", "passed", "所有声明 ID 都被字幕时间轴覆盖。")


def _demo_signal_check(audio: Sequence[AudioClip]) -> MediaQCCheck:
    if any(item.source == "demo_signal" for item in audio):
        return MediaQCCheck(
            "demo_signal",
            "warning",
            "音频为测试用确定性 demo signal，不应作为真实人声或 TTS。",
        )
    return MediaQCCheck("demo_signal", "passed", "音频由调用方提供，未标记为 demo signal。")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    return " ".join(text.strip().split())[:800]


_LEGACY_ALL = [
    "AudioClip",
    "ClaimReference",
    "CommandResult",
    "ContentStudioMediaAdapter",
    "ContentStudioMediaError",
    "MediaArtifact",
    "MediaCommandRunner",
    "MediaQCCheck",
    "MediaQCResult",
    "MediaTimeline",
    "RenderRequest",
    "RenderResult",
    "SubtitleCue",
    "SubprocessMediaRunner",
    "VisualClip",
]


# V2 two-phase Content Studio media adapter.
#
# The first implementation above was intentionally small but mixed preview and
# final output.  The names below replace the module exports with the stricter
# two-stage contract: preview render first, final render only after trusted
# approval of the preview technical QC.

@dataclass(frozen=True, slots=True)
class VisualClip:
    clip_id: str
    path: Path
    mime_type: str
    start_ms: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class AudioClip:
    clip_id: str
    path: Path
    mime_type: str
    start_ms: int
    duration_ms: int
    source: str


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    cue_id: str
    start_ms: int
    duration_ms: int
    text: str
    x: int
    y: int
    width: int
    height: int
    claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaimReference:
    claim_id: str
    text: str


@dataclass(frozen=True, slots=True)
class MediaTimeline:
    width: int
    height: int
    duration_ms: int
    visuals: tuple[VisualClip, ...]
    audio: tuple[AudioClip, ...]
    subtitles: tuple[SubtitleCue, ...]
    claims: tuple[ClaimReference, ...] = ()


@dataclass(frozen=True, slots=True)
class RenderRequest:
    title: str
    output_basename: str
    timeline: MediaTimeline
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class FinalRenderApproval:
    approved: bool
    approved_by: str
    preview_sha256: str
    technical_passed: bool


@dataclass(frozen=True, slots=True)
class MediaArtifact:
    path: Path
    mime_type: str
    sha256: str
    size_bytes: int
    ffprobe: Mapping[str, object]
    technical_params: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MediaQCCheck:
    name: str
    status: str
    details: str


@dataclass(frozen=True, slots=True)
class VideoReviewFrame:
    frame_id: str
    path: Path
    timestamp_ms: int
    sha256: str


@dataclass(frozen=True, slots=True)
class MediaQCResult:
    checks: tuple[MediaQCCheck, ...]
    review_frames: tuple[VideoReviewFrame, ...] = ()

    @property
    def technical_passed(self) -> bool:
        return not any(check.status == "failed" for check in self.checks)

    @property
    def needs_review(self) -> bool:
        return any(check.status in {"warning", "not_detected"} for check in self.checks)

    def check(self, name: str) -> MediaQCCheck:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class PreviewRenderResult:
    preview: MediaArtifact
    qc: MediaQCResult


@dataclass(frozen=True, slots=True)
class FinalRenderResult:
    final: MediaArtifact
    qc: MediaQCResult


class ContentStudioMediaAdapter:
    def __init__(
        self,
        *,
        runner: MediaCommandRunner | None = None,
        ffmpeg_binary: str | None = None,
        ffprobe_binary: str | None = None,
        video_composer: _VideoComposer | None = None,
    ) -> None:
        self._runner = runner or SubprocessMediaRunner()
        self._uses_external_runner = runner is not None
        self._ffmpeg_binary = ffmpeg_binary
        self._ffprobe_binary = ffprobe_binary
        self._video_composer = video_composer
        self._preview_sha256s: set[str] = set()

    def render_preview(self, request: RenderRequest, output_dir: Path) -> PreviewRenderResult:
        artifact = self._render_stage(request, output_dir, stage="preview", crf="28")
        qc = self._qc(request, artifact)
        self._preview_sha256s.add(artifact.sha256)
        return PreviewRenderResult(preview=artifact, qc=qc)

    def render_final(
        self,
        request: RenderRequest,
        output_dir: Path,
        *,
        approval: FinalRenderApproval,
    ) -> FinalRenderResult:
        if (
            not approval.approved
            or not approval.technical_passed
            or not approval.approved_by.strip()
            or approval.preview_sha256 not in self._preview_sha256s
        ):
            raise ContentStudioMediaError("final render requires an approved preview")
        artifact = self._render_stage(request, output_dir, stage="final", crf="20")
        return FinalRenderResult(final=artifact, qc=self._qc(request, artifact))

    def _render_stage(
        self,
        request: RenderRequest,
        output_dir: Path,
        *,
        stage: str,
        crf: str,
    ) -> MediaArtifact:
        _v2_validate_request(request)
        ffmpeg = self._resolve_binary(self._ffmpeg_binary, "ffmpeg")
        ffprobe = self._resolve_binary(self._ffprobe_binary, "ffprobe")
        output_dir.mkdir(parents=True, exist_ok=True)
        render_id = _uuid4().hex[:12]
        work_dir = output_dir / f"content-studio-work-{render_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        subtitles = _v2_write_ass_subtitles(request.timeline, work_dir / "subtitles.ass")
        visual = self._render_visual_track(ffmpeg, request, work_dir)
        output = output_dir / f"{safe_generated_filename(request.output_basename)}-{render_id}-{stage}.mp4"
        self._mux(ffmpeg, request, visual, subtitles, output, crf=crf)
        return self._artifact_for(ffprobe, output, request.timeout_seconds)

    def _render_visual_track(self, ffmpeg: str, request: RenderRequest, work_dir: Path) -> Path:
        if not self._uses_external_runner:
            composer = self._video_composer or _VideoComposer(ffmpeg_binary=ffmpeg)
            compose_request = _VideoComposeRequest(
                title=request.title,
                clips=tuple(
                    _VideoClipInput(
                        storage_key=clip.clip_id,
                        path=clip.path,
                        mime_type=clip.mime_type,
                        duration_seconds=max(1, round(clip.duration_ms / 1000)),
                    )
                    for clip in request.timeline.visuals
                ),
                output_filename="visual-track.mp4",
                aspect_ratio="9:16" if request.timeline.height >= request.timeline.width else "16:9",
                image_duration_seconds=max(1, round(request.timeline.visuals[0].duration_ms / 1000)),
                timeout_seconds=request.timeout_seconds,
            )
            return composer.compose(compose_request, work_dir)
        segments: list[Path] = []
        for index, clip in enumerate(request.timeline.visuals):
            segment = work_dir / f"visual-{index:03d}.mp4"
            command = _v2_visual_command(ffmpeg, request, clip, segment)
            self._runner.run(command, timeout_seconds=request.timeout_seconds)
            segments.append(segment)
        concat = work_dir / "visual-concat.txt"
        concat.write_text(
            "".join(f"file '{path.resolve().as_posix()}'\n" for path in segments),
            encoding="utf-8",
        )
        visual_track = work_dir / "visual-track.mp4"
        self._runner.run(
            (ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(visual_track)),
            timeout_seconds=request.timeout_seconds,
        )
        return visual_track

    def _mux(
        self,
        ffmpeg: str,
        request: RenderRequest,
        visual: Path,
        subtitles: Path,
        output: Path,
        *,
        crf: str,
    ) -> None:
        audio = request.timeline.audio[0]
        self._runner.run(
            (
                ffmpeg,
                "-y",
                "-i",
                str(visual),
                "-i",
                str(audio.path),
                "-vf",
                f"{_v2_canvas_filter(request.timeline)},subtitles={_ffmpeg_filter_path(subtitles)}",
                "-t",
                f"{request.timeline.duration_ms / 1000:.3f}",
                "-r",
                str(FPS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                crf,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(output),
            ),
            timeout_seconds=request.timeout_seconds,
        )

    def _artifact_for(self, ffprobe: str, path: Path, timeout_seconds: int) -> MediaArtifact:
        if not path.is_file():
            raise ContentStudioMediaError("renderer did not produce a local playable file")
        payload = self._ffprobe(ffprobe, path, timeout_seconds)
        return MediaArtifact(
            path=path,
            mime_type=MP4_MIME_TYPE,
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
            ffprobe=payload,
            technical_params=_v2_technical_params(payload),
        )

    def _ffprobe(self, ffprobe: str, path: Path, timeout_seconds: int) -> Mapping[str, object]:
        result = self._runner.run(
            (
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ),
            timeout_seconds=timeout_seconds,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ContentStudioMediaError("ffprobe did not return JSON") from error
        if not isinstance(payload, Mapping):
            raise ContentStudioMediaError("ffprobe result is invalid")
        return payload

    def _qc(self, request: RenderRequest, artifact: MediaArtifact) -> MediaQCResult:
        frame_check, review_frames = self._video_reviewer_frame_sampling(
            artifact.path,
            request.timeline.duration_ms,
            request.timeout_seconds,
        )
        checks = [
            _v2_encoding_check(request, artifact),
            _v2_duration_fps_check(request, artifact),
            _v2_subtitle_safe_area_check(request.timeline),
            _v2_subtitle_layout_check(request.timeline),
            _subtitle_text_review_check(request.timeline.subtitles),
            _claim_coverage_check(request.timeline.claims, request.timeline.subtitles),
            _visual_change_cadence_check(request.timeline),
            frame_check,
            self._black_frame_check(artifact.path, request.timeout_seconds),
            self._silence_check(artifact.path, request.timeout_seconds),
            _v2_demo_signal_check(request.timeline.audio),
            MediaQCCheck("subtitle_visual_contrast", "not_detected", "未抽帧做字幕像素级对比度/OCR 检测。"),
            MediaQCCheck("face_identity", "not_detected", "人脸/人物身份一致性未由本地媒体适配器检测。"),
            MediaQCCheck("factual_truth", "not_detected", "事实真实性未由本地媒体适配器检测。"),
        ]
        return MediaQCResult(checks=tuple(checks), review_frames=review_frames)

    def _video_reviewer_frame_sampling(
        self,
        path: Path,
        duration_ms: int,
        timeout_seconds: int,
    ) -> tuple[MediaQCCheck, tuple[VideoReviewFrame, ...]]:
        ffmpeg = self._resolve_binary(self._ffmpeg_binary, "ffmpeg")
        review_dir = path.parent / f"{path.stem}-review-frames"
        review_dir.mkdir(parents=True, exist_ok=True)
        pattern = review_dir / "review-frame-%03d.jpg"
        duration_seconds = max(1.0, duration_ms / 1000)
        target_count = 3 if duration_seconds < 45 else min(8, max(3, round(duration_seconds / 12)))
        interval_seconds = max(1, round(duration_seconds / target_count))
        self._runner.run(
            (
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-vf",
                f"fps=1/{interval_seconds},scale=360:-1",
                "-frames:v",
                str(target_count),
                str(pattern),
            ),
            timeout_seconds=timeout_seconds,
        )
        frames = tuple(sorted(review_dir.glob("review-frame-*.jpg")))
        if not frames:
            return (
                MediaQCCheck("video_reviewer_frame_sampling", "failed", "视频审核员未能抽取任何审核帧。"),
                (),
            )
        review_frames = tuple(
            VideoReviewFrame(
                frame_id=f"FRAME{index:03d}",
                path=frame,
                timestamp_ms=round((index - 1) * duration_ms / max(1, len(frames))),
                sha256=_sha256(frame),
            )
            for index, frame in enumerate(frames, start=1)
        )
        status = "passed" if len(review_frames) >= min(3, target_count) else "warning"
        return (
            MediaQCCheck(
                "video_reviewer_frame_sampling",
                status,
                f"视频审核员已抽取 {len(review_frames)} frames 用于画面、字幕、错字和异常内容复核。",
            ),
            review_frames,
        )

    def _black_frame_check(self, path: Path, timeout_seconds: int) -> MediaQCCheck:
        ffmpeg = self._resolve_binary(self._ffmpeg_binary, "ffmpeg")
        result = self._runner.run(
            (ffmpeg, "-v", "info", "-i", str(path), "-vf", "blackdetect=d=0.25:pix_th=0.10", "-an", "-f", "null", "-"),
            timeout_seconds=timeout_seconds,
        )
        text = f"{result.stdout}\n{result.stderr}"
        if "black_start:" in text:
            return MediaQCCheck("black_frame", "failed", _clean_text(text))
        return MediaQCCheck("black_frame", "passed", "未检测到超过阈值的黑帧区间。")

    def _silence_check(self, path: Path, timeout_seconds: int) -> MediaQCCheck:
        ffmpeg = self._resolve_binary(self._ffmpeg_binary, "ffmpeg")
        result = self._runner.run(
            (
                ffmpeg,
                "-v",
                "info",
                "-i",
                str(path),
                "-af",
                f"silencedetect=n=-45dB:d={SILENCE_QC_MIN_SECONDS}",
                "-f",
                "null",
                "-",
            ),
            timeout_seconds=timeout_seconds,
        )
        text = f"{result.stdout}\n{result.stderr}"
        if "silence_start:" in text:
            return MediaQCCheck("silence", "failed", _clean_text(text))
        return MediaQCCheck("silence", "passed", "未检测到超过阈值的静音区间。")

    def _resolve_binary(self, configured: str | None, name: str) -> str:
        if configured is not None:
            return configured
        found = shutil.which(name)
        if found is None:
            raise ContentStudioMediaError(f"{name} is not installed")
        return found


def _v2_validate_request(request: RenderRequest) -> None:
    if not request.title.strip():
        raise ContentStudioMediaError("title must be nonblank")
    safe_generated_filename(request.output_basename)
    timeline = request.timeline
    if timeline.width < 320 or timeline.height < 320:
        raise ContentStudioMediaError("timeline Pack dimensions are invalid")
    if timeline.duration_ms <= 0:
        raise ContentStudioMediaError("timeline duration must be positive")
    if len(timeline.audio) != 1:
        raise ContentStudioMediaError("timeline currently supports exactly one audio track")
    if not timeline.visuals:
        raise ContentStudioMediaError("timeline requires at least one visual clip")
    if not timeline.subtitles:
        raise ContentStudioMediaError("timeline requires burned subtitle cues")
    _v2_validate_visual_coverage(timeline)
    audio = timeline.audio[0]
    _v2_validate_file(audio.path, "audio")
    if audio.mime_type not in SUPPORTED_AUDIO_MIME_TYPES:
        raise ContentStudioMediaError("unsupported audio clip MIME type")
    if audio.start_ms != 0 or audio.duration_ms < timeline.duration_ms:
        raise ContentStudioMediaError("audio track must start at 0 and cover the timeline duration")
    if audio.source not in {"caller_voice", "caller_tts", "caller_audio", "demo_signal"}:
        raise ContentStudioMediaError("audio source must identify caller-provided audio or demo_signal")
    for visual in timeline.visuals:
        _v2_validate_file(visual.path, "visual")
        if visual.mime_type not in SUPPORTED_VISUAL_MIME_TYPES:
            raise ContentStudioMediaError("unsupported visual clip MIME type")
    for subtitle in timeline.subtitles:
        if not subtitle.text.strip():
            raise ContentStudioMediaError("subtitle text must be nonblank")
    if request.timeout_seconds < 1 or request.timeout_seconds > 1800:
        raise ContentStudioMediaError("timeout_seconds must be between 1 and 1800")


def _v2_validate_visual_coverage(timeline: MediaTimeline) -> None:
    expected = 0
    for clip in sorted(timeline.visuals, key=lambda item: item.start_ms):
        if clip.start_ms != expected or clip.duration_ms <= 0:
            raise ContentStudioMediaError("visual timeline must fully cover duration without gaps or overlaps")
        expected = clip.start_ms + clip.duration_ms
    if expected != timeline.duration_ms:
        raise ContentStudioMediaError("visual timeline must fully cover duration without gaps or overlaps")


def _v2_validate_file(path: Path, label: str) -> None:
    if not isinstance(path, Path) or not path.is_file():
        raise ContentStudioMediaError(f"{label} file is unavailable")


def _v2_visual_command(ffmpeg: str, request: RenderRequest, clip: VisualClip, output: Path) -> tuple[str, ...]:
    duration = f"{clip.duration_ms / 1000:.3f}"
    prefix = (ffmpeg, "-y")
    if clip.mime_type == "video/mp4":
        input_args: tuple[str, ...] = ("-i", str(clip.path))
        visual_filter = _v2_canvas_filter(request.timeline)
    else:
        input_args = ("-loop", "1", "-i", str(clip.path))
        visual_filter = _v2_image_motion_filter(request.timeline, clip)
    return (
        *prefix,
        *input_args,
        "-vf",
        visual_filter,
        "-t",
        duration,
        "-r",
        str(FPS),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    )


def _v2_canvas_filter(timeline: MediaTimeline) -> str:
    return (
        f"scale={timeline.width}:{timeline.height}:force_original_aspect_ratio=decrease,"
        f"pad={timeline.width}:{timeline.height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )


def _v2_image_motion_filter(timeline: MediaTimeline, clip: VisualClip) -> str:
    frame_count = max(1, math.ceil(clip.duration_ms * FPS / 1000))
    return (
        f"scale={timeline.width}:{timeline.height}:force_original_aspect_ratio=increase,"
        f"crop={timeline.width}:{timeline.height},"
        "zoompan="
        "z='min(zoom+0.0008,1.08)':"
        "x='iw/2-(iw/zoom/2)':"
        "y='ih/2-(ih/zoom/2)':"
        f"d={frame_count}:s={timeline.width}x{timeline.height}:fps={FPS},"
        "setsar=1"
    )


def _v2_write_ass_subtitles(timeline: MediaTimeline, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {timeline.width}",
        f"PlayResY: {timeline.height}",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: Default,Arial,56,&H00FFFFFF,&H00FFFFFF,&H00111111,&HAA000000,0,0,0,0,100,100,0,0,1,3,1,2,80,80,160,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for cue in timeline.subtitles:
        text = _v2_wrap_subtitle(cue.text, cue.width)
        lines.append(
            "Dialogue: 0,"
            f"{_ass_time(cue.start_ms)},{_ass_time(cue.start_ms + cue.duration_ms)},"
            f"Default,{cue.cue_id},0,0,0,,{{\\an7\\pos({cue.x},{cue.y})}}{_escape_ass_text(text)}"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _v2_wrap_subtitle(text: str, width: int) -> str:
    max_chars = max(4, width // 36)
    words = list(text)
    rows = ["".join(words[index : index + max_chars]) for index in range(0, len(words), max_chars)]
    return "\\N".join(rows)


def _v2_technical_params(ffprobe: Mapping[str, object]) -> Mapping[str, object]:
    video = _stream(ffprobe, "video")
    audio = _stream(ffprobe, "audio")
    return {
        "width": _int_from_mapping(video, "width"),
        "height": _int_from_mapping(video, "height"),
        "video_codec": str(video.get("codec_name", "")),
        "audio_codec": str(audio.get("codec_name", "")),
        "pix_fmt": str(video.get("pix_fmt", "")),
        "sample_rate": str(audio.get("sample_rate", "")),
        "channels": _int_from_mapping(audio, "channels"),
        "fps": _v2_parse_rate(str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "")),
        "duration_seconds": _v2_duration_seconds(ffprobe),
    }


def _v2_parse_rate(value: str) -> float | None:
    if "/" in value:
        left, right = value.split("/", 1)
        try:
            denominator = float(right)
            return None if denominator == 0 else float(left) / denominator
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _v2_duration_seconds(ffprobe: Mapping[str, object]) -> float | None:
    raw_format = ffprobe.get("format")
    if not isinstance(raw_format, Mapping):
        return None
    raw = raw_format.get("duration")
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return None


def _v2_encoding_check(request: RenderRequest, artifact: MediaArtifact) -> MediaQCCheck:
    params = artifact.technical_params
    failures: list[str] = []
    if params.get("width") != request.timeline.width:
        failures.append("width")
    if params.get("height") != request.timeline.height:
        failures.append("height")
    if params.get("video_codec") != "h264":
        failures.append("video_codec")
    if params.get("audio_codec") != "aac":
        failures.append("audio_codec")
    if failures:
        return MediaQCCheck("encoding", "failed", "编码参数不符合 Pack 要求: " + ",".join(failures))
    return MediaQCCheck("encoding", "passed", "视频编码参数符合锁定 Pack。")


def _v2_duration_fps_check(request: RenderRequest, artifact: MediaArtifact) -> MediaQCCheck:
    duration = artifact.technical_params.get("duration_seconds")
    fps = artifact.technical_params.get("fps")
    expected = request.timeline.duration_ms / 1000
    if not isinstance(duration, int | float) or abs(float(duration) - expected) > 0.25:
        return MediaQCCheck("duration", "failed", "输出时长与 timeline 不一致。")
    if not isinstance(fps, int | float) or abs(float(fps) - FPS) > 0.1:
        return MediaQCCheck("duration", "failed", "输出 FPS 与 timeline 不一致。")
    return MediaQCCheck("duration", "passed", "输出时长和 FPS 符合 timeline。")


def _v2_subtitle_safe_area_check(timeline: MediaTimeline) -> MediaQCCheck:
    min_x = max(24, round(timeline.width * 0.05))
    max_x = timeline.width - min_x
    min_y = max(48, round(timeline.height * 0.05))
    max_y = timeline.height - max(80, round(timeline.height * 0.08))
    bad = [
        cue.cue_id
        for cue in timeline.subtitles
        if cue.x < min_x or cue.y < min_y or cue.x + cue.width > max_x or cue.y + cue.height > max_y
    ]
    if bad:
        return MediaQCCheck("subtitle_safe_area", "failed", "字幕越过安全区: " + ", ".join(bad))
    return MediaQCCheck("subtitle_safe_area", "passed", "字幕框坐标位于安全区内。")


def _v2_subtitle_layout_check(timeline: MediaTimeline) -> MediaQCCheck:
    bad: list[str] = []
    for cue in timeline.subtitles:
        max_chars = max(4, cue.width // 36)
        line_count = max(1, (len(cue.text) + max_chars - 1) // max_chars)
        if line_count * 68 > cue.height:
            bad.append(cue.cue_id)
    if bad:
        return MediaQCCheck("subtitle_layout", "failed", "字幕估算布局可能溢出: " + ", ".join(bad))
    return MediaQCCheck("subtitle_layout", "passed", "字幕文本按框宽换行后未超过声明高度。")


def _subtitle_text_review_check(subtitles: Sequence[SubtitleCue]) -> MediaQCCheck:
    suspicious: list[str] = []
    for cue in subtitles:
        text = cue.text.strip()
        if any(marker in text for marker in ("口口", "□□", "??", "？？", "�")):
            suspicious.append(cue.cue_id)
            continue
        if re.search(r"([，。！？,.!?])\1{1,}", text):
            suspicious.append(cue.cue_id)
            continue
        if len(text) > 34 and not any(separator in text for separator in ("，", "。", "！", "？", ",", ".", "!", "?")):
            suspicious.append(cue.cue_id)
    if suspicious:
        return MediaQCCheck(
            "subtitle_text_review",
            "failed",
            "视频审核员发现疑似错字、乱码、占位符或异常长句字幕: " + ", ".join(suspicious),
        )
    return MediaQCCheck("subtitle_text_review", "passed", "字幕文本未发现常见错字/乱码/占位符风险。")


def _v2_demo_signal_check(audio: Sequence[AudioClip]) -> MediaQCCheck:
    if any(item.source == "demo_signal" for item in audio):
        return MediaQCCheck("demo_signal", "warning", "音频为测试 demo signal，不代表真实人声/TTS。")
    return MediaQCCheck("demo_signal", "passed", "音频由调用方提供。")


def _visual_change_cadence_check(timeline: MediaTimeline) -> MediaQCCheck:
    slow_stills = [
        clip.clip_id
        for clip in timeline.visuals
        if clip.mime_type.startswith("image/") and clip.duration_ms > 5_000
    ]
    if slow_stills:
        return MediaQCCheck(
            "visual_change_cadence",
            "failed",
            "静态图片视觉节奏超过 5 秒，会呈现图片拉长/PPT 感: " + ", ".join(slow_stills),
        )
    return MediaQCCheck("visual_change_cadence", "passed", "静态图片视觉节奏符合 3-5 秒变化要求。")


__all__ = [
    "AudioClip",
    "ClaimReference",
    "CommandResult",
    "ContentStudioMediaAdapter",
    "ContentStudioMediaError",
    "FinalRenderApproval",
    "FinalRenderResult",
    "MediaArtifact",
    "MediaCommandRunner",
    "MediaQCCheck",
    "MediaQCResult",
    "MediaTimeline",
    "PreviewRenderResult",
    "RenderRequest",
    "SubprocessMediaRunner",
    "SubtitleCue",
    "VideoReviewFrame",
    "VisualClip",
]
