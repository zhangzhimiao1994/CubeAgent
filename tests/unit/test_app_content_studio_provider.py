# mypy: disable-error-code="arg-type, return-value, no-untyped-call, no-untyped-def"

from __future__ import annotations

import wave
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from agent_hub.app import (
    _ConfigBackedContentStudioProductionProvider,
    _content_studio_research_bundle,
    _qc_report_from_media,
    _render_request_from_project,
    _write_demo_signal_wav,
)
from agent_hub.content_studio import (
    AssetManifest,
    AssetRecord,
    ContentStudioService,
    Evidence,
    InMemoryContentProjectStore,
    PackRegistry,
    ProjectStatus,
    ResearchQuestion,
    ScriptDraft,
    ScriptSegment,
    Timeline,
    VoiceTrack,
)
from agent_hub.content_studio.media import MediaQCCheck, MediaQCResult
from agent_hub.multimodal.generation import MultimediaGenerationKind, MultimediaGenerationResult


def test_demo_signal_wav_contains_audible_samples(tmp_path: Path) -> None:
    audio = tmp_path / "demo-signal.wav"

    _write_demo_signal_wav(audio, seconds=1)

    with wave.open(str(audio), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        samples = [
            int.from_bytes(frames[index : index + 2], byteorder="little", signed=True)
            for index in range(0, len(frames), 2)
        ]

    assert any(sample != 0 for sample in samples)


@pytest.mark.asyncio
async def test_content_studio_voice_uses_file_backed_audio_generation_when_available(tmp_path: Path) -> None:
    audio = tmp_path / "voice.wav"
    _write_demo_signal_wav(audio, seconds=1)
    provider = _ConfigBackedContentStudioProductionProvider(
        list_models=lambda: (),
        secret_service=object(),
        tenant_id=uuid4(),
        redis_client=object(),
        multimedia_generation_executor=FakeAudioMultimedia(audio),
        output_dir=tmp_path,
    )
    project = _voice_ready_project()

    voiced = await provider._ensure_voice(project)

    assert voiced.status is ProjectStatus.VOICE_READY
    assert voiced.voice_track is not None
    assert voiced.voice_track.audio_artifact_id == str(audio)
    assert voiced.voice_track.mime_type == "audio/wav"
    assert voiced.voice_track.source == "caller_tts"
    assert not any("BLOCKER" in item for item in voiced.voice_track.pronunciation_report)


@pytest.mark.asyncio
async def test_content_studio_tts_input_is_only_approved_narration(tmp_path: Path) -> None:
    audio = tmp_path / "voice.wav"
    _write_demo_signal_wav(audio, seconds=1)
    multimedia = FakeAudioMultimedia(audio)
    provider = _ConfigBackedContentStudioProductionProvider(
        list_models=lambda: (),
        secret_service=object(),
        tenant_id=uuid4(),
        redis_client=object(),
        multimedia_generation_executor=multimedia,
        output_dir=tmp_path,
    )
    project = _voice_ready_project(
        topic="做一条约60秒的AIGC抖音科普视频，主题是2026年前后AI Agent、多模态生成和企业落地的主要变化；要求研究充分、表达通俗、事实有来源。",
        narration=(
            "2026年前后，AI Agent 的变化不是更会聊天，而是开始接进真实工作流。",
            "多模态生成也从单张图，变成图片、语音和视频一起配合生产。",
        ),
    )

    voiced = await provider._ensure_voice(project)

    assert voiced.status is ProjectStatus.VOICE_READY
    assert multimedia.prompts == [
        (
            "2026年前后，AI Agent 的变化不是更会聊天，而是开始接进真实工作流。\n"
            "多模态生成也从单张图，变成图片、语音和视频一起配合生产。"
        )
    ]
    assert "做一条约60秒" not in multimedia.prompts[0]
    assert "主题：" not in multimedia.prompts[0]
    assert "请为" not in multimedia.prompts[0]


@pytest.mark.asyncio
async def test_content_studio_voice_keeps_demo_blocker_without_audio_generation(tmp_path: Path) -> None:
    provider = _ConfigBackedContentStudioProductionProvider(
        list_models=lambda: (),
        secret_service=object(),
        tenant_id=uuid4(),
        redis_client=object(),
        multimedia_generation_executor=FakeNoAudioMultimedia(),
        output_dir=tmp_path,
    )
    project = _voice_ready_project()

    voiced = await provider._ensure_voice(project)

    assert voiced.voice_track is not None
    assert voiced.voice_track.source == "demo_signal"
    assert any("BLOCKER" in item for item in voiced.voice_track.pronunciation_report)


def test_production_research_bundle_keeps_source_coverage_for_workbench() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="AIGC research metadata",
        topic="AIGC research metadata",
        source_urls=("https://openai.com/release-notes/example",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="production",
    )
    questions = (ResearchQuestion("RQ001", "What changed?"),)
    evidence = (
        Evidence(
            evidence_id="EV001",
            source_url="https://openai.com/release-notes/example",
            source_type="release_notes",
            publisher="OpenAI",
            published_at=None,
            retrieved_at="2026-09-21T10:00:00Z",
            content_hash="hash1",
            locator="body",
            excerpt="Release note evidence.",
            license="source_terms",
        ),
        Evidence(
            evidence_id="EV002",
            source_url="https://github.com/openai/example/releases/tag/v1",
            source_type="github_release",
            publisher="GitHub",
            published_at=None,
            retrieved_at="2026-09-21T10:01:00Z",
            content_hash="hash2",
            locator="body",
            excerpt="GitHub release evidence.",
            license="source_terms",
        ),
    )

    bundle = _content_studio_research_bundle(project, questions, evidence)

    assert bundle.source_priority[:3] == ("official_docs", "official_blog", "release_notes")
    coverage = {item.source_type: item.status for item in bundle.source_coverage}
    assert coverage["release_notes"] == "covered"
    assert coverage["github_release"] == "covered"
    assert coverage["official_docs"] == "missing"
    assert {item.source_url for item in bundle.source_candidates} >= {
        "https://openai.com/release-notes/example",
        "https://github.com/openai/example/releases/tag/v1",
    }


def test_video_reviewer_failures_are_visible_and_block_final_approval() -> None:
    project = _voice_ready_project()
    media_qc = MediaQCResult(
        checks=(
            MediaQCCheck("encoding", "passed", "ok"),
            MediaQCCheck("video_reviewer_frame_sampling", "passed", "视频审核员已抽取 3 frames。"),
            MediaQCCheck("subtitle_text_review", "failed", "SUB001 疑似错字或乱码。"),
        )
    )

    report = _qc_report_from_media(project, media_qc)

    assert "video reviewer: video_reviewer_frame_sampling passed - 视频审核员已抽取 3 frames。" in report.checked_items
    assert "video reviewer: subtitle_text_review failed - SUB001 疑似错字或乱码。" in report.checked_items
    assert "video reviewer QC failed: subtitle_text_review" in report.blockers


def test_render_request_splits_still_assets_into_short_visual_beats(tmp_path: Path) -> None:
    project = _voice_ready_project()
    assets = tuple(
        replace(
            project.asset_manifest.assets[0],
            asset_id=f"ASSET{index:03d}",
            request_id=f"ASREQ{index:03d}",
            technical_params={
                "file_path": str(tmp_path / f"asset-{index}.png"),
                "mime": "image/png",
            },
        )
        for index in (1, 2)
    )
    project = replace(
        project,
        asset_manifest=AssetManifest(assets=assets),
        voice_track=VoiceTrack(
            audio_artifact_id=str(tmp_path / "voice.wav"),
            timestamp_level="sentence",
            pronunciation_report=(),
        ),
        timeline=Timeline(
            width=1080,
            height=1920,
            duration_ms=12_000,
            tracks={
                "narration": ("voice",),
                "primary_visual": tuple(asset.asset_id for asset in assets),
                "subtitle": project.script.subtitle_lines if project.script else (),
            },
        ),
    )

    request = _render_request_from_project(project)

    assert len(request.timeline.visuals) > len(assets)
    assert max(clip.duration_ms for clip in request.timeline.visuals) <= 5_000
    assert request.timeline.visuals[0].start_ms == 0
    assert (
        request.timeline.visuals[-1].start_ms + request.timeline.visuals[-1].duration_ms
        == project.timeline.duration_ms
    )


def _voice_ready_project(
    *,
    topic: str = "AIGC voice smoke",
    narration: tuple[str, ...] = ("AIGC 正在从一次性问答，变成能接进工作流的工具。",),
):
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="AIGC voice",
        topic=topic,
        source_urls=(),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="production",
    )
    asset = AssetRecord(
        asset_id="ASSET001",
        request_id="ASREQ001",
        source="fixture",
        acquisition_method="generated_image",
        url_or_provider_task_id="fixture",
        content_hash="hash",
        technical_params={"file_path": "asset.png", "mime": "image/png"},
        rights_status="approved",
        generation_params={},
    )
    return replace(
        project,
        script_approved=True,
        rights_approved=True,
        asset_manifest=AssetManifest(assets=(asset,)),
        script=ScriptDraft(
            hooks=("AIGC 变了。", "一分钟看懂 AIGC。", "别只看热闹。"),
            segments=tuple(
                ScriptSegment(f"SEG{index:03d}", text, True, ("CL001",))
                for index, text in enumerate(narration, start=1)
            ),
            subtitle_lines=narration,
        ),
    )


class FakeAudioMultimedia:
    def __init__(self, audio: Path) -> None:
        self.audio = audio
        self.prompts: list[str] = []

    async def default_logical_model(self, kind: str) -> str | None:
        assert kind == "audio"
        return "tts-primary"

    async def generate(
        self,
        *,
        kind: MultimediaGenerationKind,
        logical_model: str,
        prompt: str,
    ) -> MultimediaGenerationResult:
        assert kind is MultimediaGenerationKind.AUDIO
        assert logical_model == "tts-primary"
        assert prompt.strip()
        self.prompts.append(prompt)
        return MultimediaGenerationResult(
            kind=kind,
            logical_model=logical_model,
            deployment_id="tts-deployment",
            text=str(self.audio),
            file_path=self.audio,
            filename=self.audio.name,
            mime_type="audio/wav",
        )


class FakeNoAudioMultimedia:
    async def default_logical_model(self, kind: str) -> str | None:
        assert kind == "audio"
        return None
