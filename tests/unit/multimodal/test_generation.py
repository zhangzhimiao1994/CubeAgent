from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from agent_hub.models.gateway import GatewayCompletion
from agent_hub.models.types import ModelCapability, ModelRequest, ModelResponse
from agent_hub.multimodal.generation import (
    InMemoryMultimediaGenerationJobStore,
    MultimediaArtifact,
    MultimediaDailyLimitExceeded,
    MultimediaGenerationExecutor,
    MultimediaGenerationJobStatus,
    MultimediaGenerationKind,
)


class GatewayStub:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        self.requests.append(request)
        return GatewayCompletion(
            response=ModelResponse(text="artifact://generated-video"),
            deployment_id="video-primary-1",
            logical_model=request.logical_model,
            provider_id="minimax",
            provider_model="minimax/MiniMax-Hailuo-02",
            cost_usd=Decimal("0.010000"),
        )


async def test_video_generation_request_requires_video_capability() -> None:
    gateway = GatewayStub()
    executor = MultimediaGenerationExecutor(gateway)

    result = await executor.generate(
        kind=MultimediaGenerationKind.VIDEO,
        logical_model="video-primary",
        prompt="generate a short product video",
    )

    assert result.text == "artifact://generated-video"
    assert gateway.requests[0].logical_model == "video-primary"
    assert gateway.requests[0].required_capabilities == frozenset({
        ModelCapability.VIDEO_GENERATION
    })


async def test_audio_generation_request_requires_audio_generation_capability() -> None:
    gateway = GatewayStub()
    executor = MultimediaGenerationExecutor(gateway)

    result = await executor.generate(
        kind=MultimediaGenerationKind.AUDIO,
        logical_model="audio-primary",
        prompt="generate a short intro sound",
    )

    assert result.text == "artifact://generated-video"
    assert gateway.requests[0].logical_model == "audio-primary"
    assert gateway.requests[0].required_capabilities == frozenset({
        ModelCapability.AUDIO_GENERATION
    })


async def test_generation_job_store_receives_executor_artifacts_for_main_agent() -> None:
    gateway = GatewayStub()
    store = InMemoryMultimediaGenerationJobStore()
    executor = MultimediaGenerationExecutor(gateway, job_store=store)

    queued = executor.submit(
        kind=MultimediaGenerationKind.VIDEO,
        logical_model="video-primary",
        prompt="generate a short product video",
    )
    assert queued.status is MultimediaGenerationJobStatus.QUEUED

    completed = await executor.run_job(queued.id, executor_id="media-agent-1")

    assert completed.status is MultimediaGenerationJobStatus.SUCCEEDED
    assert completed.executor_id == "media-agent-1"
    assert completed.artifacts == (
        MultimediaArtifact(
            kind=MultimediaGenerationKind.VIDEO,
            uri="artifact://generated-video",
            text="artifact://generated-video",
            logical_model="video-primary",
            deployment_id="video-primary-1",
        ),
    )
    assert store.get(queued.id) == completed


async def test_generation_job_cannot_be_run_twice_by_competing_executor_agents() -> None:
    gateway = GatewayStub()
    executor = MultimediaGenerationExecutor(gateway)
    queued = executor.submit(
        kind=MultimediaGenerationKind.VIDEO,
        logical_model="video-primary",
        prompt="generate a short product video",
    )

    await executor.run_job(queued.id, executor_id="media-agent-1")

    with pytest.raises(RuntimeError, match="not queued"):
        await executor.run_job(queued.id, executor_id="media-agent-2")

    assert len(gateway.requests) == 1


def test_generation_job_store_expires_after_24_hours_and_removes_files(tmp_path: Path) -> None:
    current = datetime(2026, 9, 4, 8, 0, tzinfo=UTC)

    def now() -> datetime:
        return current

    store = InMemoryMultimediaGenerationJobStore(now=now)
    queued = store.create(
        kind=MultimediaGenerationKind.IMAGE,
        logical_model="image-primary",
        prompt="generate a poster",
    )
    media_path = tmp_path / "poster.png"
    media_path.write_bytes(b"image")
    completed = store.succeed(
        queued.id,
        artifacts=(
            MultimediaArtifact(
                kind=MultimediaGenerationKind.IMAGE,
                uri=media_path.as_uri(),
                file_path=media_path,
                filename=media_path.name,
                mime_type="image/png",
            ),
        ),
    )

    assert completed.expires_at == current + timedelta(hours=24)
    assert store.get(queued.id).status is MultimediaGenerationJobStatus.SUCCEEDED

    current = current + timedelta(hours=24, seconds=1)

    with pytest.raises(KeyError, match="unknown multimedia generation job"):
        store.get(queued.id)
    assert not media_path.exists()


async def test_generation_prompt_is_required() -> None:
    executor = MultimediaGenerationExecutor(GatewayStub())

    with pytest.raises(ValueError, match="prompt"):
        await executor.generate(
            kind=MultimediaGenerationKind.IMAGE,
            logical_model="image-primary",
            prompt=" ",
        )


async def test_generation_daily_limit_blocks_the_fourth_video_request() -> None:
    gateway = GatewayStub()
    executor = MultimediaGenerationExecutor(gateway, daily_limit=3)

    for index in range(3):
        await executor.generate(
            kind=MultimediaGenerationKind.VIDEO,
            logical_model="video-primary",
            prompt=f"generate video {index}",
        )

    with pytest.raises(MultimediaDailyLimitExceeded, match="daily multimedia generation limit"):
        await executor.generate(
            kind=MultimediaGenerationKind.VIDEO,
            logical_model="video-primary",
            prompt="generate video 4",
        )

    assert len(gateway.requests) == 3
