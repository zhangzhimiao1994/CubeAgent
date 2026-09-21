from __future__ import annotations

import wave
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from agent_hub.app import _ConfigBackedContentStudioProductionProvider, _write_demo_signal_wav
from agent_hub.content_studio import (
    AssetManifest,
    AssetRecord,
    ContentStudioService,
    InMemoryContentProjectStore,
    PackRegistry,
    ProjectStatus,
)
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


def _voice_ready_project():
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="AIGC voice",
        topic="AIGC voice smoke",
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
    )


class FakeAudioMultimedia:
    def __init__(self, audio: Path) -> None:
        self.audio = audio

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
