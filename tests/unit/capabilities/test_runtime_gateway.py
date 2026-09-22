# mypy: disable-error-code="index, call-overload, operator, dict-item"

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID
from zipfile import ZipFile

import pytest

import agent_hub.capabilities.runtime as runtime_module
from agent_hub.capabilities.runtime import (
    RuntimeAssetVisualReview,
    RuntimeCapabilityError,
    RuntimeCapabilityGateway,
)
from agent_hub.files.generated import (
    DOCX_MIME_TYPE,
    MP4_MIME_TYPE,
    PNG_MIME_TYPE,
    PPTX_MIME_TYPE,
    ZIP_MIME_TYPE,
    GeneratedFileStore,
)
from agent_hub.multimodal.generation import (
    MultimediaArtifact,
    MultimediaGenerationJob,
    MultimediaGenerationJobStatus,
    MultimediaGenerationKind,
)
from agent_hub.runtime.contracts import JsonValue
from agent_hub.skills.sandbox.base import SkillInvocation, SkillResult
from agent_hub.video.composer import VideoComposeRequest
from tests.unit.skills.test_package import skill_zip

TENANT_ID = UUID("66666666-6666-4666-8666-666666666666")
RUN_ID = UUID("77777777-7777-4777-8777-777777777777")


class FakeSandbox:
    def __init__(self, *, stdout: str = '{"ok":true}') -> None:
        self.invocations: list[SkillInvocation] = []
        self.stdout = stdout

    async def run(self, invocation: SkillInvocation) -> SkillResult:
        self.invocations.append(invocation)
        return SkillResult(
            exit_code=0,
            stdout=self.stdout,
            stderr="",
            timed_out=False,
        )

    async def terminate(self, execution_id: str) -> None:
        del execution_id


class FakeMultimediaExecutor:
    def __init__(self, media_path: Path) -> None:
        self.media_path = media_path
        self.created_at = datetime(2026, 9, 4, 8, 0, tzinfo=UTC)
        self.expires_at = self.created_at + timedelta(hours=24)
        self.submitted: list[tuple[MultimediaGenerationKind, str, str]] = []
        self.run_requests: list[tuple[str, str]] = []
        self._jobs: dict[str, MultimediaGenerationJob] = {}

    async def default_logical_model_for_multimedia(
        self,
        *,
        kind: MultimediaGenerationKind,
    ) -> str:
        return f"{kind.value}_primary"

    def submit(
        self,
        *,
        kind: MultimediaGenerationKind,
        logical_model: str,
        prompt: str,
    ) -> MultimediaGenerationJob:
        self.submitted.append((kind, logical_model, prompt))
        index = len(self.submitted)
        job_id = "media_test" if index == 1 else f"media_test_{index}"
        job = MultimediaGenerationJob(
            id=job_id,
            kind=kind,
            logical_model=logical_model,
            prompt=prompt,
            status=MultimediaGenerationJobStatus.QUEUED,
            created_at=self.created_at,
            expires_at=self.expires_at,
        )
        self._jobs[job_id] = job
        return job

    async def run_job(
        self,
        job_id: str,
        *,
        executor_id: str,
    ) -> MultimediaGenerationJob:
        self.run_requests.append((job_id, executor_id))
        job = self._jobs[job_id]
        return MultimediaGenerationJob(
            id=job_id,
            kind=job.kind,
            logical_model=job.logical_model,
            prompt=job.prompt,
            status=MultimediaGenerationJobStatus.SUCCEEDED,
            executor_id=executor_id,
            created_at=self.created_at,
            expires_at=self.expires_at,
            artifacts=(
                MultimediaArtifact(
                    kind=job.kind,
                    uri=(
                        "artifact://generated-video"
                        if job.kind is MultimediaGenerationKind.VIDEO
                        else f"artifact://{job_id}"
                    ),
                    text=(
                        "artifact://generated-video"
                        if job.kind is MultimediaGenerationKind.VIDEO
                        else f"artifact://{job_id}"
                    ),
                    logical_model=job.logical_model,
                    deployment_id=f"{job.logical_model}_1",
                    file_path=self.media_path,
                    filename=self.media_path.name,
                    mime_type=(
                        "video/mp4"
                        if job.kind is MultimediaGenerationKind.VIDEO
                        else "image/png"
                    ),
                ),
            ),
        )


class HangingMultimediaExecutor(FakeMultimediaExecutor):
    async def run_job(
        self,
        job_id: str,
        *,
        executor_id: str,
    ) -> MultimediaGenerationJob:
        self.run_requests.append((job_id, executor_id))
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class SlowCancellationMultimediaExecutor(FakeMultimediaExecutor):
    def __init__(self, media_path: Path) -> None:
        super().__init__(media_path)
        self.cancelled = asyncio.Event()
        self.cancelled_count = 0
        self.active_jobs = 0
        self.max_active_jobs = 0

    async def run_job(
        self,
        job_id: str,
        *,
        executor_id: str,
    ) -> MultimediaGenerationJob:
        self.run_requests.append((job_id, executor_id))
        self.active_jobs += 1
        self.max_active_jobs = max(self.max_active_jobs, self.active_jobs)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            self.cancelled_count += 1
            await asyncio.sleep(0.5)
            raise
        finally:
            self.active_jobs -= 1
        raise AssertionError("unreachable")


class ParallelTrackingMultimediaExecutor(FakeMultimediaExecutor):
    def __init__(self, media_path: Path) -> None:
        super().__init__(media_path)
        self.active_jobs = 0
        self.max_active_jobs = 0

    async def run_job(
        self,
        job_id: str,
        *,
        executor_id: str,
    ) -> MultimediaGenerationJob:
        self.active_jobs += 1
        self.max_active_jobs = max(self.max_active_jobs, self.active_jobs)
        try:
            await asyncio.sleep(0.05)
            return await super().run_job(job_id, executor_id=executor_id)
        finally:
            self.active_jobs -= 1


class DelayedMultimediaExecutor(FakeMultimediaExecutor):
    def __init__(self, media_path: Path, *, delay_seconds: float) -> None:
        super().__init__(media_path)
        self.delay_seconds = delay_seconds
        self.active_jobs = 0
        self.max_active_jobs = 0

    async def run_job(
        self,
        job_id: str,
        *,
        executor_id: str,
    ) -> MultimediaGenerationJob:
        self.active_jobs += 1
        self.max_active_jobs = max(self.max_active_jobs, self.active_jobs)
        try:
            await asyncio.sleep(self.delay_seconds)
            return await super().run_job(job_id, executor_id=executor_id)
        finally:
            self.active_jobs -= 1


class RateLimitedOnceMultimediaExecutor(FakeMultimediaExecutor):
    def __init__(
        self,
        media_path: Path,
        *,
        failing_prompt: str,
        failure_message: str = "DashScope image submit failed: Requests rate limit exceeded",
    ) -> None:
        super().__init__(media_path)
        self.failing_prompt = failing_prompt
        self.failure_message = failure_message
        self.failures_by_prompt: dict[str, int] = {}

    async def run_job(
        self,
        job_id: str,
        *,
        executor_id: str,
    ) -> MultimediaGenerationJob:
        self.run_requests.append((job_id, executor_id))
        job = self._jobs[job_id]
        if job.prompt == self.failing_prompt and self.failures_by_prompt.get(job.prompt, 0) == 0:
            self.failures_by_prompt[job.prompt] = 1
            raise RuntimeError(self.failure_message)
        return MultimediaGenerationJob(
            id=job_id,
            kind=job.kind,
            logical_model=job.logical_model,
            prompt=job.prompt,
            status=MultimediaGenerationJobStatus.SUCCEEDED,
            executor_id=executor_id,
            created_at=self.created_at,
            expires_at=self.expires_at,
            artifacts=(
                MultimediaArtifact(
                    kind=job.kind,
                    uri=f"artifact://{job_id}",
                    text=f"artifact://{job_id}",
                    logical_model=job.logical_model,
                    deployment_id=f"{job.logical_model}_1",
                    file_path=self.media_path,
                    filename=self.media_path.name,
                    mime_type="image/png",
                ),
            ),
        )


class FakeAssetVisualReviewer:
    def __init__(self, review: RuntimeAssetVisualReview | Exception) -> None:
        self.review = review
        self.requests: list[dict[str, object]] = []

    async def review_image_asset(
        self,
        *,
        tenant_id: UUID,
        label: str,
        prompt: str,
        filename: str,
        mime_type: str,
        data: bytes,
        image_url: str | None = None,
    ) -> RuntimeAssetVisualReview:
        self.requests.append(
            {
                "tenant_id": tenant_id,
                "label": label,
                "prompt": prompt,
                "filename": filename,
                "mime_type": mime_type,
                "data": data,
                "image_url": image_url,
            }
        )
        if isinstance(self.review, Exception):
            raise self.review
        return self.review


class SequencedAssetVisualReviewer:
    def __init__(self, reviews: tuple[RuntimeAssetVisualReview, ...]) -> None:
        self.reviews = list(reviews)
        self.requests: list[dict[str, object]] = []

    async def review_image_asset(
        self,
        *,
        tenant_id: UUID,
        label: str,
        prompt: str,
        filename: str,
        mime_type: str,
        data: bytes,
        image_url: str | None = None,
    ) -> RuntimeAssetVisualReview:
        self.requests.append(
            {
                "tenant_id": tenant_id,
                "label": label,
                "prompt": prompt,
                "filename": filename,
                "mime_type": mime_type,
                "data": data,
                "image_url": image_url,
            }
        )
        return self.reviews.pop(0)


class FakeVideoComposer:
    def __init__(self) -> None:
        self.requests: list[tuple[VideoComposeRequest, Path]] = []

    def compose(self, request: VideoComposeRequest, output_dir: Path) -> Path:
        self.requests.append((request, output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / request.output_filename
        output.write_bytes(b"composed video")
        return output


class ThreadCheckingVideoComposer:
    def __init__(self, main_thread_id: int) -> None:
        self.main_thread_id = main_thread_id
        self.thread_ids: list[int] = []

    def compose(self, request: VideoComposeRequest, output_dir: Path) -> Path:
        self.thread_ids.append(threading.get_ident())
        assert threading.get_ident() != self.main_thread_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / request.output_filename
        output.write_bytes(b"threaded composed video")
        return output


async def test_runtime_gateway_executes_calculator_without_external_side_effects(tmp_path: Path) -> None:
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path)

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="planner",
        name="calculator",
        arguments={"expression": "2 + 3 * 4"},
        idempotency_key="calc_1",
    )

    assert result == {"value": "14"}
    assert gateway.is_replay_safe("calculator") is True


async def test_runtime_gateway_executes_multimedia_generation_tool(tmp_path: Path) -> None:
    media_path = tmp_path / "generated-video.mp4"
    media_path.write_bytes(b"video")
    media_executor = FakeMultimediaExecutor(media_path)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
    )

    assert gateway.is_available(TENANT_ID, "generate_multimedia") is True
    assert gateway.is_replay_safe("generate_multimedia") is True

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="multimedia_generator",
        name="generate_multimedia",
        arguments={
            "kind": "video",
            "logical_model": "video_primary",
            "generation_prompt": "生成 5 秒产品视频",
        },
        idempotency_key="media_1",
    )

    assert media_executor.submitted == [
        (MultimediaGenerationKind.VIDEO, "video_primary", "生成 5 秒产品视频")
    ]
    assert media_executor.run_requests == [("media_test", "multimedia_generator")]
    assert result["job_id"] == "media_test"
    assert result["status"] == "succeeded"
    assert result["summary"] == "Generated video artifact with video_primary."
    file_metadata = result["file"]
    assert isinstance(file_metadata, dict)
    assert file_metadata["filename"] == "generated-video.mp4"
    assert file_metadata["mime_type"] == "video/mp4"
    assert file_metadata["size_bytes"] == len(b"video")
    assert file_metadata["sha256"] == (
        "0cab1c9617404faf2b24e221e189ca5945813e14d3f766345b09ca13bbe28ffc"
    )
    assert isinstance(file_metadata["artifact_id"], str)
    assert file_metadata["download_url"] == (
        f"/api/v1/admin/runs/{RUN_ID}/artifacts/{file_metadata['artifact_id']}/download"
    )
    assert isinstance(file_metadata["expires_at"], str)
    assert datetime.fromisoformat(file_metadata["expires_at"]) == media_executor.expires_at
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert isinstance(artifact, dict)
    assert artifact["kind"] == "video"
    assert artifact["uri"] == "artifact://generated-video"
    assert artifact["text"] == "artifact://generated-video"
    assert artifact["logical_model"] == "video_primary"
    assert artifact["deployment_id"] == "video_primary_1"
    assert artifact["filename"] == "generated-video.mp4"
    assert artifact["mime_type"] == "video/mp4"
    assert artifact["size_bytes"] == len(b"video")
    assert artifact["sha256"] == "0cab1c9617404faf2b24e221e189ca5945813e14d3f766345b09ca13bbe28ffc"
    assert artifact["artifact_id"] == file_metadata["artifact_id"]
    assert artifact["storage_key"] == file_metadata["storage_key"]
    assert artifact["download_url"] == file_metadata["download_url"]
    assert artifact["expires_at"] == file_metadata["expires_at"]
    assert artifact["file"] == file_metadata
    assert result["metadata"] == file_metadata


async def test_runtime_gateway_multimedia_generation_times_out_hung_image_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "hung.png"
    media_path.write_bytes(b"image")
    media_executor = HangingMultimediaExecutor(media_path)
    monkeypatch.setattr(runtime_module, "_MULTIMEDIA_IMAGE_JOB_TIMEOUT_SECONDS", 0.01)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成 1 张产品图片",
        },
        idempotency_key="media_hung_image",
    )

    assert media_executor.submitted == [
        (MultimediaGenerationKind.IMAGE, "image_primary", "生成 1 张产品图片")
    ]
    assert media_executor.run_requests == [("media_test", "asset_generator")]
    assert result["status"] == "failed"
    assert result["review_status"] == "needs_user_revision"
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 1
    assert artifacts[0]["status"] == "failed"
    assert "image generation timed out" in artifacts[0]["generation_error"]


async def test_runtime_gateway_multimedia_timeout_does_not_wait_for_slow_provider_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "slow-cancel.png"
    media_path.write_bytes(b"image")
    media_executor = SlowCancellationMultimediaExecutor(media_path)
    monkeypatch.setattr(runtime_module, "_MULTIMEDIA_IMAGE_JOB_TIMEOUT_SECONDS", 0.01)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
    )

    result = await asyncio.wait_for(
        gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="asset_generator",
            name="generate_multimedia",
            arguments={
                "kind": "image",
                "logical_model": "image_primary",
                "generation_prompt": "生成 1 张普通图片",
            },
            idempotency_key="media_slow_cancel_image",
        ),
        timeout=0.2,
    )

    assert result["review_status"] == "needs_user_revision"
    await asyncio.wait_for(media_executor.cancelled.wait(), timeout=0.05)
    await asyncio.sleep(0.6)


async def test_runtime_gateway_multimedia_batch_timeout_returns_failed_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "slow-batch.png"
    media_path.write_bytes(b"image")
    media_executor = SlowCancellationMultimediaExecutor(media_path)
    monkeypatch.setattr(runtime_module, "_MULTIMEDIA_IMAGE_JOB_TIMEOUT_SECONDS", 0.01)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
    )

    result = await asyncio.wait_for(
        gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="asset_generator",
            name="generate_multimedia",
            arguments={
                "kind": "image",
                "logical_model": "image_primary",
                "generation_prompt": "生成全量资产图",
                "artifact_count": 5,
                "artifact_prompts": (
                    "生成图片 1",
                    "生成图片 2",
                    "生成图片 3",
                    "生成图片 4",
                    "生成图片 5",
                ),
            },
            idempotency_key="media_slow_batch_image",
        ),
        timeout=0.2,
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 5
    assert result["review_status"] == "needs_user_revision"
    assert all(item["status"] == "failed" for item in artifacts)
    assert all("generation_error" in item for item in artifacts)
    assert media_executor.max_active_jobs == 5
    await asyncio.sleep(0.05)
    assert media_executor.cancelled_count == 5
    await asyncio.sleep(0.6)


async def test_runtime_gateway_multimedia_batch_timeout_allows_all_image_waves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "asset-wave.png"
    media_path.write_bytes(b"image")
    media_executor = DelayedMultimediaExecutor(media_path, delay_seconds=0.02)
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="符合资产图要求",
            issues=(),
            confidence=0.95,
        )
    )
    monkeypatch.setattr(runtime_module, "_MULTIMEDIA_IMAGE_JOB_TIMEOUT_SECONDS", 0.05)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await asyncio.wait_for(
        gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="asset_generator",
            name="generate_multimedia",
            arguments={
                "kind": "image",
                "logical_model": "image_primary",
                "generation_prompt": "生成全量资产图",
                "artifact_count": 8,
                "artifact_prompts": (
                    "生成男主角色锁定资产图",
                    "生成女主角色锁定资产图",
                    "生成反派角色锁定资产图",
                    "生成服装妆造资产图",
                    "生成场景资产图",
                    "生成道具资产图",
                    "生成动作资产图",
                    "生成特效资产图",
                ),
            },
            idempotency_key="media_asset_waves",
        ),
        timeout=0.5,
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 8
    assert media_executor.max_active_jobs == 8
    assert len(media_executor.run_requests) == 8


async def test_runtime_gateway_multimedia_retries_image_provider_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "rate-limited.png"
    media_path.write_bytes(b"image")
    failing_prompt = "生成男主角色锁定资产图"
    media_executor = RateLimitedOnceMultimediaExecutor(media_path, failing_prompt=failing_prompt)
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="符合资产图要求",
            issues=(),
            confidence=0.93,
        )
    )
    monkeypatch.setattr(runtime_module, "_MULTIMEDIA_IMAGE_PROVIDER_RETRY_BACKOFF_SECONDS", 0)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成角色资产",
            "artifact_count": 2,
            "artifact_prompts": (
                failing_prompt,
                "生成女主角色锁定资产图",
            ),
        },
        idempotency_key="media_rate_limit_retry",
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 2
    assert media_executor.failures_by_prompt == {failing_prompt: 1}
    assert [prompt for _kind, _model, prompt in media_executor.submitted].count(failing_prompt) == 2


async def test_runtime_gateway_multimedia_retries_dashscope_image_query_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "query-retried.png"
    media_path.write_bytes(b"image")
    failing_prompt = "生成女主角色锁定资产图"
    media_executor = RateLimitedOnceMultimediaExecutor(
        media_path,
        failing_prompt=failing_prompt,
        failure_message="DashScope task query failed",
    )
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="符合资产图要求",
            issues=(),
            confidence=0.93,
        )
    )
    monkeypatch.setattr(runtime_module, "_MULTIMEDIA_IMAGE_PROVIDER_RETRY_BACKOFF_SECONDS", 0)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成角色资产",
            "artifact_count": 1,
            "artifact_prompts": (failing_prompt,),
        },
        idempotency_key="media_dashscope_query_retry",
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 1
    assert media_executor.failures_by_prompt == {failing_prompt: 1}
    assert [prompt for _kind, _model, prompt in media_executor.submitted].count(failing_prompt) == 2


async def test_runtime_gateway_multimedia_retries_image_provider_read_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "read-timeout-retried.png"
    media_path.write_bytes(b"image")
    failing_prompt = "生成角色锁定资产图"
    media_executor = RateLimitedOnceMultimediaExecutor(
        media_path,
        failing_prompt=failing_prompt,
        failure_message="capability execution failed (ReadTimeout)",
    )
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="符合资产图要求",
            issues=(),
            confidence=0.93,
        )
    )
    monkeypatch.setattr(runtime_module, "_MULTIMEDIA_IMAGE_PROVIDER_RETRY_BACKOFF_SECONDS", 0)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成角色资产",
            "artifact_count": 1,
            "artifact_prompts": (failing_prompt,),
        },
        idempotency_key="media_read_timeout_retry",
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 1
    assert media_executor.failures_by_prompt == {failing_prompt: 1}
    assert [prompt for _kind, _model, prompt in media_executor.submitted].count(failing_prompt) == 2


async def test_runtime_gateway_multimedia_preserves_batch_when_one_image_prompt_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "partial-success.png"
    media_path.write_bytes(b"image")
    failing_prompt = "生成特效资产图"
    media_executor = RateLimitedOnceMultimediaExecutor(
        media_path,
        failing_prompt=failing_prompt,
        failure_message="permanent provider failure",
    )
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="符合资产图要求",
            issues=(),
            confidence=0.93,
        )
    )
    monkeypatch.setattr(runtime_module, "_MULTIMEDIA_IMAGE_PROVIDER_RETRY_BACKOFF_SECONDS", 0)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成全量资产",
            "artifact_count": 2,
            "artifact_prompts": (
                failing_prompt,
                "生成镜头资产图",
            ),
            "artifact_labels": (
                "特效资产",
                "镜头资产",
            ),
        },
        idempotency_key="media_partial_failure",
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 2
    assert result["review_status"] == "needs_user_revision"
    assert artifacts[0]["label"] == "特效资产"
    assert artifacts[0]["visual_review"]["passed"] is False
    assert artifacts[0]["generation_error"] == "permanent provider failure"
    assert artifacts[1]["label"] == "镜头资产"
    assert artifacts[1]["visual_review"]["passed"] is True


async def test_runtime_gateway_composes_video_from_generated_artifacts(tmp_path: Path) -> None:
    generated_artifact_dir = tmp_path / "generated"
    store = GeneratedFileStore(generated_artifact_dir)
    video_source = store.store_bytes(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        artifact_id=UUID("77777777-7777-4777-8777-000000000001"),
        filename="clip.mp4",
        mime_type=MP4_MIME_TYPE,
        data=b"video",
    )
    image_source = store.store_bytes(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        artifact_id=UUID("77777777-7777-4777-8777-000000000002"),
        filename="image.png",
        mime_type=PNG_MIME_TYPE,
        data=b"image",
    )
    composer = FakeVideoComposer()
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=generated_artifact_dir,
        video_composer=composer,
    )

    assert gateway.is_available(TENANT_ID, "compose_video") is True
    assert gateway.is_replay_safe("compose_video") is True

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="video_compositor",
        name="compose_video",
        arguments={
            "title": "Launch Reel",
            "filename": "launch-reel.mp4",
            "aspect_ratio": "9:16",
            "image_duration_seconds": 4,
            "clips": (
                {
                    "storage_key": video_source.storage_key,
                    "filename": video_source.filename,
                    "mime_type": video_source.mime_type,
                },
                {
                    "storage_key": image_source.storage_key,
                    "filename": image_source.filename,
                    "mime_type": image_source.mime_type,
                    "duration_seconds": 2,
                },
            ),
        },
        idempotency_key="compose_1",
    )

    assert len(composer.requests) == 1
    request, output_dir = composer.requests[0]
    assert output_dir.name.startswith("agent-hub-video-")
    assert request.title == "Launch Reel"
    assert request.output_filename == "launch-reel.mp4"
    assert request.aspect_ratio == "9:16"
    assert request.image_duration_seconds == 4
    assert [clip.mime_type for clip in request.clips] == [MP4_MIME_TYPE, PNG_MIME_TYPE]
    assert request.clips[0].path == generated_artifact_dir / video_source.storage_key
    assert request.clips[1].path == generated_artifact_dir / image_source.storage_key

    file_metadata = _assert_file_result(result, expected_mime_type=MP4_MIME_TYPE)
    assert file_metadata["filename"] == "launch-reel.mp4"
    assert file_metadata["size_bytes"] == len(b"composed video")
    assert result["summary"] == "Composed video artifact launch-reel.mp4."
    assert result["presentation"] == "final_attachment"
    assert (generated_artifact_dir / str(file_metadata["storage_key"])).read_bytes() == b"composed video"


async def test_runtime_gateway_runs_video_composition_off_event_loop(tmp_path: Path) -> None:
    generated_artifact_dir = tmp_path / "generated"
    store = GeneratedFileStore(generated_artifact_dir)
    video_source = store.store_bytes(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        artifact_id=UUID("77777777-7777-4777-8777-000000000003"),
        filename="clip.mp4",
        mime_type=MP4_MIME_TYPE,
        data=b"video",
    )
    composer = ThreadCheckingVideoComposer(threading.get_ident())
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=generated_artifact_dir,
        video_composer=composer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="video_compositor",
        name="compose_video",
        arguments={
            "title": "Launch Reel",
            "clips": (
                {
                    "storage_key": video_source.storage_key,
                    "mime_type": video_source.mime_type,
                },
            ),
        },
        idempotency_key="compose_threaded",
    )

    assert composer.thread_ids
    file_metadata = _assert_file_result(result, expected_mime_type=MP4_MIME_TYPE)
    assert file_metadata["size_bytes"] == len(b"threaded composed video")


async def test_runtime_gateway_rejects_clip_mime_type_that_conflicts_with_storage_filename(
    tmp_path: Path,
) -> None:
    generated_artifact_dir = tmp_path / "generated"
    store = GeneratedFileStore(generated_artifact_dir)
    zip_source = store.store_bytes(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        artifact_id=UUID("77777777-7777-4777-8777-000000000004"),
        filename="archive.zip",
        mime_type=ZIP_MIME_TYPE,
        data=b"zip",
    )
    composer = FakeVideoComposer()
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=generated_artifact_dir,
        video_composer=composer,
    )

    with pytest.raises(RuntimeCapabilityError, match="mime_type does not match clip filename"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="video_compositor",
            name="compose_video",
            arguments={
                "title": "Launch Reel",
                "clips": (
                    {
                        "storage_key": zip_source.storage_key,
                        "mime_type": MP4_MIME_TYPE,
                    },
                ),
            },
            idempotency_key="compose_bad_mime",
        )

    assert composer.requests == []


async def test_runtime_gateway_compose_video_requires_generated_artifact_store(
    tmp_path: Path,
) -> None:
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills")

    with pytest.raises(RuntimeCapabilityError, match="generated artifact store is not configured"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="video_compositor",
            name="compose_video",
            arguments={
                "title": "Launch Reel",
                "clips": (
                    {
                        "storage_key": "tenant/run/artifact/clip.mp4",
                        "mime_type": MP4_MIME_TYPE,
                    },
                ),
            },
            idempotency_key="compose_missing_store",
        )


async def test_runtime_gateway_multimedia_generation_tool_keeps_legacy_prompt_compatible(tmp_path: Path) -> None:
    media_path = tmp_path / "generated-video.mp4"
    media_path.write_bytes(b"video")
    media_executor = FakeMultimediaExecutor(media_path)
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        multimedia_generation_executor=media_executor,
    )

    await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="multimedia_generator",
        name="generate_multimedia",
        arguments={
            "kind": "video",
            "logical_model": "video_primary",
            "prompt": "生成 5 秒产品视频",
        },
        idempotency_key="media_legacy",
    )

    assert media_executor.submitted == [
        (MultimediaGenerationKind.VIDEO, "video_primary", "生成 5 秒产品视频")
    ]


async def test_runtime_gateway_multimedia_generation_tool_runs_each_artifact_prompt(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "character-sheet.png"
    media_path.write_bytes(b"image")
    media_executor = FakeMultimediaExecutor(media_path)
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="符合角色锁定资产要求",
            issues=(),
            confidence=0.91,
            logical_model="vision_primary",
            deployment_id="vision_primary_1",
        )
    )
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="multimedia_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "为男女主生成角色参考设定表",
            "artifact_count": 2,
            "artifact_prompts": (
                "为男主单独生成一张角色参考设定表",
                "为女主单独生成一张角色参考设定表",
            ),
            "artifact_labels": (
                "角色锁定资产：男主",
                "角色锁定资产：女主",
            ),
        },
        idempotency_key="media_multi_character_sheet",
    )

    assert media_executor.submitted == [
        (MultimediaGenerationKind.IMAGE, "image_primary", "为男主单独生成一张角色参考设定表"),
        (MultimediaGenerationKind.IMAGE, "image_primary", "为女主单独生成一张角色参考设定表"),
    ]
    assert media_executor.run_requests == [
        ("media_test", "multimedia_generator"),
        ("media_test_2", "multimedia_generator"),
    ]
    assert result["job_id"] == "media_test"
    assert result["job_ids"] == ("media_test", "media_test_2")
    assert result["summary"] == "Generated 2 image artifacts with image_primary."
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 2
    assert [artifact["title"] for artifact in artifacts] == [
        "角色锁定资产：男主",
        "角色锁定资产：女主",
    ]
    assert [artifact["label"] for artifact in artifacts] == [
        "角色锁定资产：男主",
        "角色锁定资产：女主",
    ]
    assert [artifact["generation_prompt"] for artifact in artifacts] == [
        "为男主单独生成一张角色参考设定表",
        "为女主单独生成一张角色参考设定表",
    ]
    assert [artifact["visual_review"]["passed"] for artifact in artifacts] == [True, True]
    assert [artifact["visual_review"]["summary"] for artifact in artifacts] == [
        "符合角色锁定资产要求",
        "符合角色锁定资产要求",
    ]
    assert [request["label"] for request in visual_reviewer.requests] == [
        "角色锁定资产：男主",
        "角色锁定资产：女主",
    ]
    assert [request["data"] for request in visual_reviewer.requests] == [b"image", b"image"]


async def test_runtime_gateway_multimedia_generation_runs_image_assets_in_parallel(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "character-sheet.png"
    media_path.write_bytes(b"image")
    media_executor = ParallelTrackingMultimediaExecutor(media_path)
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="符合资产图要求",
            issues=(),
            confidence=0.9,
        )
    )
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成全量资产图",
            "artifact_count": 3,
            "artifact_prompts": (
                "生成男主角色锁定资产图",
                "生成女主角色锁定资产图",
                "生成场景资产图",
            ),
            "artifact_labels": (
                "角色锁定资产：男主",
                "角色锁定资产：女主",
                "场景资产",
            ),
        },
        idempotency_key="media_parallel_assets",
    )

    assert media_executor.max_active_jobs >= 2
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert [artifact["title"] for artifact in artifacts] == [
        "角色锁定资产：男主",
        "角色锁定资产：女主",
        "场景资产",
    ]
    assert [artifact["visual_review"]["summary"] for artifact in artifacts] == [
        "符合资产图要求",
        "符合资产图要求",
        "符合资产图要求",
    ]
    assert [request["label"] for request in visual_reviewer.requests] == [
        "角色锁定资产：男主",
        "角色锁定资产：女主",
        "场景资产",
    ]


def test_runtime_gateway_multimedia_image_asset_parallelism_scales_for_large_packs() -> None:
    assert runtime_module._multimedia_parallelism(MultimediaGenerationKind.IMAGE, 1) == 1
    assert runtime_module._multimedia_parallelism(MultimediaGenerationKind.IMAGE, 3) == 3
    assert runtime_module._multimedia_parallelism(MultimediaGenerationKind.IMAGE, 10) == 9


async def test_runtime_gateway_multimedia_reviews_non_character_production_assets(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "prop-sheet.png"
    media_path.write_bytes(b"image")
    media_executor = FakeMultimediaExecutor(media_path)
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="道具和特效资产符合要求",
            issues=(),
            confidence=0.92,
        )
    )
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成短剧专业资产图",
            "artifact_count": 2,
            "artifact_prompts": (
                "生成道具资产设定板，覆盖剧情关键物、随身物和法器细节，不要画成角色动作剧照",
                "生成特效资产设定板，覆盖能量形态、颜色层级和触发动作，不要画成战斗海报",
            ),
            "artifact_labels": ("道具资产", "特效资产"),
        },
        idempotency_key="media_non_character_assets_reviewed",
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert [artifact["title"] for artifact in artifacts] == ["道具资产", "特效资产"]
    assert [artifact["visual_review"]["passed"] for artifact in artifacts] == [True, True]
    assert [request["label"] for request in visual_reviewer.requests] == [
        "道具资产",
        "特效资产",
    ]


async def test_runtime_gateway_multimedia_retries_asset_image_when_visual_review_rejects(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "asset.png"
    media_path.write_bytes(b"image")
    media_executor = FakeMultimediaExecutor(media_path)
    visual_reviewer = SequencedAssetVisualReviewer(
        (
            RuntimeAssetVisualReview(
                passed=False,
                summary="不是资产图",
                issues=("像单张剧照", "缺少角色三视图和锁定信息"),
                confidence=0.88,
                logical_model="vision_primary",
                deployment_id="vision_primary_1",
            ),
            RuntimeAssetVisualReview(
                passed=True,
                summary="修正后符合角色锁定资产要求",
                issues=(),
                confidence=0.93,
                logical_model="vision_primary",
                deployment_id="vision_primary_1",
            ),
        )
    )
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成角色锁定资产图",
            "artifact_count": 1,
            "artifact_prompts": ("生成角色锁定资产图，不要电影剧照",),
            "artifact_labels": ("角色锁定资产",),
        },
        idempotency_key="media_visual_review_retry",
    )

    assert [job_id for job_id, _actor in media_executor.run_requests] == [
        "media_test",
        "media_test_2",
    ]
    assert "视觉审核未通过" in media_executor.submitted[1][2]
    assert "不是资产图" in media_executor.submitted[1][2]
    assert "像单张剧照" in media_executor.submitted[1][2]
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 1
    assert artifacts[0]["uri"] == "artifact://media_test_2"
    assert artifacts[0]["visual_review"]["summary"] == "修正后符合角色锁定资产要求"
    assert [request["label"] for request in visual_reviewer.requests] == [
        "角色锁定资产",
        "角色锁定资产",
    ]


async def test_runtime_gateway_multimedia_retries_only_rejected_asset_prompt(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "asset.png"
    media_path.write_bytes(b"image")
    media_executor = FakeMultimediaExecutor(media_path)
    visual_reviewer = SequencedAssetVisualReviewer(
        (
            RuntimeAssetVisualReview(
                passed=True,
                summary="男主资产合格",
                issues=(),
                confidence=0.91,
            ),
            RuntimeAssetVisualReview(
                passed=False,
                summary="女主资产不合格",
                issues=("像单张写真",),
                confidence=0.86,
            ),
            RuntimeAssetVisualReview(
                passed=True,
                summary="女主资产修正合格",
                issues=(),
                confidence=0.94,
            ),
        )
    )
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成男女主角色锁定资产图",
            "artifact_count": 2,
            "artifact_prompts": (
                "生成男主角色锁定资产图",
                "生成女主角色锁定资产图",
            ),
            "artifact_labels": (
                "角色锁定资产：男主",
                "角色锁定资产：女主",
            ),
        },
        idempotency_key="media_retry_only_rejected_asset",
    )

    assert media_executor.submitted == [
        (MultimediaGenerationKind.IMAGE, "image_primary", "生成男主角色锁定资产图"),
        (MultimediaGenerationKind.IMAGE, "image_primary", "生成女主角色锁定资产图"),
        (
            MultimediaGenerationKind.IMAGE,
            "image_primary",
            (
                "生成女主角色锁定资产图\n\n"
                "视觉审核未通过，正在第 2 次重新生成同一项资产：角色锁定资产：女主。\n"
                "上一版问题：女主资产不合格：像单张写真\n"
                "请修正上述问题后重新生成合格资产图；不要输出电影剧照、宣传海报、随机写真、"
                "混合角色图片或与该资产类别无关的画面。"
            ),
        ),
    ]
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 2
    assert [artifact["uri"] for artifact in artifacts] == [
        "artifact://media_test",
        "artifact://media_test_3",
    ]
    assert [artifact["visual_review"]["summary"] for artifact in artifacts] == [
        "男主资产合格",
        "女主资产修正合格",
    ]


async def test_runtime_gateway_multimedia_merges_preserved_assets_without_regenerating(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "fixed-asset.png"
    media_path.write_bytes(b"fixed image")
    media_executor = FakeMultimediaExecutor(media_path)
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="动作资产修正合格",
            issues=(),
            confidence=0.93,
        )
    )
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
        asset_visual_reviewer=visual_reviewer,
    )
    preserved_asset = {
        "kind": "image",
        "uri": "artifact://previous-character",
        "filename": "character-sheet.png",
        "mime_type": "image/png",
        "sha256": "a" * 64,
        "label": "角色锁定资产：林渊",
        "title": "角色锁定资产：林渊",
        "visual_review": {
            "passed": True,
            "summary": "角色资产合格",
            "issues": (),
            "confidence": 0.92,
        },
    }

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "只修正失败资产",
            "artifact_count": 1,
            "artifact_prompts": ("重新生成动作资产，不要电影剧照",),
            "artifact_labels": ("动作资产",),
            "preserved_artifacts": (preserved_asset,),
        },
        idempotency_key="media_preserve_passed_assets",
    )

    assert media_executor.submitted == [
        (MultimediaGenerationKind.IMAGE, "image_primary", "重新生成动作资产，不要电影剧照")
    ]
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert [artifact["label"] for artifact in artifacts] == ["角色锁定资产：林渊", "动作资产"]
    assert artifacts[0]["uri"] == "artifact://previous-character"
    assert artifacts[1]["visual_review"]["summary"] == "动作资产修正合格"
    assert result["preserved_artifact_count"] == 1
    assert result["generated_artifact_count"] == 1


async def test_runtime_gateway_multimedia_visual_review_preserves_rejected_asset_image(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "bad-asset.png"
    media_path.write_bytes(b"not a character sheet")
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=False,
            summary="不是资产图",
            issues=("像单张剧照", "缺少角色三视图和锁定信息"),
            confidence=0.88,
            logical_model="vision_primary",
            deployment_id="vision_primary_1",
        )
    )
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=FakeMultimediaExecutor(media_path),
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成全量资产图",
            "artifact_count": 1,
            "artifact_prompts": ("生成角色锁定资产图，不要电影剧照",),
            "artifact_labels": ("角色锁定资产",),
        },
        idempotency_key="media_bad_visual_review",
    )

    assert [request["label"] for request in visual_reviewer.requests] == [
        "角色锁定资产",
        "角色锁定资产",
    ]
    assert result["review_status"] == "needs_user_revision"
    assert result["review_failed_artifact_count"] == 1
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 1
    assert artifacts[0]["title"] == "角色锁定资产"
    assert artifacts[0]["visual_review"]["passed"] is False
    assert artifacts[0]["visual_review"]["summary"] == "不是资产图"


async def test_runtime_gateway_multimedia_infers_asset_review_when_labels_are_missing(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "asset.png"
    media_path.write_bytes(b"image")
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="符合资产图要求",
            issues=(),
            confidence=0.9,
        )
    )
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=FakeMultimediaExecutor(media_path),
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成全量资产图",
            "artifact_count": 1,
            "artifact_prompts": ("生成角色锁定资产图，不要电影剧照",),
        },
        idempotency_key="media_inferred_asset_review",
    )

    assert [request["label"] for request in visual_reviewer.requests] == ["角色锁定资产"]
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert artifacts[0]["title"] == "角色锁定资产"
    assert artifacts[0]["visual_review"]["summary"] == "符合资产图要求"


async def test_runtime_gateway_multimedia_infers_non_character_asset_labels_from_prompt_variants(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "asset.png"
    media_path.write_bytes(b"image")
    visual_reviewer = FakeAssetVisualReviewer(
        RuntimeAssetVisualReview(
            passed=True,
            summary="符合资产图要求",
            issues=(),
            confidence=0.9,
        )
    )
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=FakeMultimediaExecutor(media_path),
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成短剧专业资产图",
            "artifact_count": 5,
            "artifact_prompts": (
                "生成道具设定板，覆盖剧情关键物、随身物和法器细节",
                "生成动作姿态参考板，覆盖奔跑、转身、递物和施法动作分解",
                "生成特效设定板，覆盖能量形态、光效层级和转场特效",
                "生成场景设定板，覆盖主要地点、关键空间和背景元素",
                "生成镜头语言设定板，覆盖景别、机位、镜头运动和构图参考",
            ),
        },
        idempotency_key="media_inferred_non_character_assets",
    )

    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert [artifact["title"] for artifact in artifacts] == [
        "道具资产",
        "动作资产",
        "特效资产",
        "场景资产",
        "镜头资产",
    ]
    assert [request["label"] for request in visual_reviewer.requests] == [
        "道具资产",
        "动作资产",
        "特效资产",
        "场景资产",
        "镜头资产",
    ]
    assert all(artifact["visual_review"]["passed"] is True for artifact in artifacts)


async def test_runtime_gateway_multimedia_visual_review_failure_preserves_asset_image(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "asset.png"
    media_path.write_bytes(b"image")
    visual_reviewer = FakeAssetVisualReviewer(ValueError("bad review json"))
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=FakeMultimediaExecutor(media_path),
        asset_visual_reviewer=visual_reviewer,
    )

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="asset_generator",
        name="generate_multimedia",
        arguments={
            "kind": "image",
            "logical_model": "image_primary",
            "generation_prompt": "生成角色锁定资产图",
            "artifact_count": 1,
            "artifact_prompts": ("生成角色锁定资产图，不要电影剧照",),
            "artifact_labels": ("角色锁定资产",),
        },
        idempotency_key="media_visual_review_failure",
    )

    assert result["review_status"] == "needs_user_revision"
    assert result["review_failed_artifact_count"] == 1
    artifacts = result["artifacts"]
    assert isinstance(artifacts, tuple)
    assert len(artifacts) == 1
    assert artifacts[0]["title"] == "角色锁定资产"
    assert artifacts[0]["visual_review"]["passed"] is False
    assert artifacts[0]["visual_review"]["summary"] == "角色锁定资产 视觉审核执行失败"
    assert artifacts[0]["visual_review"]["issues"] == (
        "visual asset review failed for 角色锁定资产: bad review json",
    )
    assert [request["label"] for request in visual_reviewer.requests] == ["角色锁定资产"]


async def test_runtime_gateway_multimedia_labels_must_match_artifact_prompts(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "character-sheet.png"
    media_path.write_bytes(b"image")
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=FakeMultimediaExecutor(media_path),
    )

    with pytest.raises(RuntimeCapabilityError, match="artifact_labels must match"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="multimedia_generator",
            name="generate_multimedia",
            arguments={
                "kind": "image",
                "logical_model": "image_primary",
                "generation_prompt": "生成完整资产图",
                "artifact_count": 2,
                "artifact_prompts": ("角色锁定资产", "场景资产"),
                "artifact_labels": ("角色锁定资产",),
            },
            idempotency_key="media_label_mismatch",
        )


async def test_runtime_gateway_multimedia_tool_requires_executor(tmp_path: Path) -> None:
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills")

    assert gateway.is_available(TENANT_ID, "generate_multimedia") is False

    with pytest.raises(RuntimeCapabilityError, match="multimedia generation executor is not configured"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="multimedia_generator",
            name="generate_multimedia",
            arguments={
                "kind": "image",
                "logical_model": "image_primary",
                "prompt": "生成产品图",
            },
            idempotency_key="media_missing_executor",
        )


async def test_runtime_gateway_reads_only_configured_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("safe", encoding="utf-8")
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills", workspace_root=workspace)

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="researcher",
        name="read_context",
        arguments={"path": "note.txt"},
        idempotency_key="read_1",
    )

    assert result == {"path": "note.txt", "text": "safe", "truncated": False}


async def test_runtime_gateway_read_context_accepts_query_without_workspace(tmp_path: Path) -> None:
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills")

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="planner",
        name="read_context",
        arguments={"query": "activity plan constraints"},
        idempotency_key="context_1",
    )

    assert result["query"] == "activity plan constraints"
    assert result["matches"] == ()
    assert result["truncated"] is False


async def test_runtime_gateway_invokes_installed_skill_through_sandbox(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / str(TENANT_ID)
    skill_dir.mkdir(parents=True)
    (skill_dir / "docx.zip").write_bytes(skill_zip())
    sandbox = FakeSandbox()
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills", skill_sandbox=sandbox)

    assert gateway.is_available(TENANT_ID, "read_context") is True
    assert gateway.is_available(TENANT_ID, "docx") is True
    assert gateway.is_available(TENANT_ID, "pdf") is False
    assert gateway.is_available(TENANT_ID, "web.search") is False

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="writer",
        name="docx",
        arguments={"task": "draft"},
        idempotency_key="skill_1",
    )

    assert result["result"] == {"ok": True}
    assert len(sandbox.invocations) == 1
    assert sandbox.invocations[0].package_path == skill_dir / "docx.zip"
    assert sandbox.invocations[0].input["arguments"] == {"task": "draft"}


async def test_runtime_gateway_normalizes_skill_json_arrays_to_contract(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / str(TENANT_ID)
    skill_dir.mkdir(parents=True)
    (skill_dir / "docx.zip").write_bytes(skill_zip())
    sandbox = FakeSandbox(stdout='{"items":[1,{"nested":["a","b"]}]}')
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills", skill_sandbox=sandbox)

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="writer",
        name="docx",
        arguments={"task": "draft"},
        idempotency_key="skill_json_arrays",
    )

    assert result["result"] == {"items": (1, {"nested": ("a", "b")})}


async def test_runtime_gateway_rejects_non_finite_skill_stdout_numbers(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / str(TENANT_ID)
    skill_dir.mkdir(parents=True)
    (skill_dir / "docx.zip").write_bytes(skill_zip())
    sandbox = FakeSandbox(stdout='{"value":NaN}')
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills", skill_sandbox=sandbox)

    with pytest.raises(RuntimeCapabilityError, match="skill stdout is not JSON serializable"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="writer",
            name="docx",
            arguments={"task": "draft"},
            idempotency_key="skill_nan_stdout",
        )


async def test_runtime_gateway_generates_docx_artifact(tmp_path: Path) -> None:
    generated_artifact_dir = tmp_path / "generated"
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=generated_artifact_dir,
    )

    assert gateway.is_available(TENANT_ID, "document.generate_docx") is True
    assert gateway.is_replay_safe("document.generate_docx") is True

    arguments: Mapping[str, JsonValue] = {
        "title": "Launch Memo",
        "subtitle": "Runtime generated",
        "filename": "launch-memo.docx",
        "sections": (
            {
                "heading": "Summary",
                "paragraphs": ("The gateway generated this document.",),
                "bullets": ("Stored as an artifact",),
            },
        ),
    }

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="writer",
        name="document.generate_docx",
        arguments=arguments,
        idempotency_key="docx_1",
    )

    file_metadata = _assert_file_result(result, expected_mime_type=DOCX_MIME_TYPE)
    assert file_metadata["filename"] == "launch-memo.docx"
    assert result["presentation"] == "final_attachment"
    storage_key = file_metadata["storage_key"]
    assert isinstance(storage_key, str)
    output = generated_artifact_dir / storage_key
    assert output.is_file()
    with ZipFile(output) as package:
        assert "[Content_Types].xml" in package.namelist()
        assert "word/document.xml" in package.namelist()


async def test_runtime_gateway_generates_pptx_artifact(tmp_path: Path) -> None:
    generated_artifact_dir = tmp_path / "generated"
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=generated_artifact_dir,
    )

    assert gateway.is_available(TENANT_ID, "presentation.generate_pptx") is True
    assert gateway.is_replay_safe("presentation.generate_pptx") is True

    arguments: Mapping[str, JsonValue] = {
        "title": "Technical Blueprint",
        "template_id": "technical-blueprint",
        "slides": (
            {
                "title": "Gateway",
                "bullets": ("DOCX and PPTX tools", "Generated artifact storage"),
            },
        ),
    }

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="designer",
        name="presentation.generate_pptx",
        arguments=arguments,
        idempotency_key="pptx_1",
    )

    file_metadata = _assert_file_result(result, expected_mime_type=PPTX_MIME_TYPE)
    assert file_metadata["filename"] == "technical-blueprint.pptx"
    assert result["presentation"] == "final_attachment"
    storage_key = file_metadata["storage_key"]
    assert isinstance(storage_key, str)
    output = generated_artifact_dir / storage_key
    assert output.is_file()
    with ZipFile(output) as package:
        assert "[Content_Types].xml" in package.namelist()
        assert "ppt/presentation.xml" in package.namelist()
        assert "ppt/slides/slide1.xml" in package.namelist()


async def test_runtime_gateway_generates_project_zip_final_artifact(tmp_path: Path) -> None:
    generated_artifact_dir = tmp_path / "generated"
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=generated_artifact_dir,
    )

    assert gateway.is_available(TENANT_ID, "project.generate_zip") is True
    assert gateway.is_replay_safe("project.generate_zip") is True

    result = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="engineer",
        name="project.generate_zip",
        arguments={
            "title": "Hello World",
            "files": {
                "main.py": "print('hello world')\n",
                "README.md": "# Hello World\n\nRun `python main.py`.\n",
            },
        },
        idempotency_key="project_zip_1",
    )

    file_metadata = _assert_file_result(result, expected_mime_type=ZIP_MIME_TYPE)
    assert file_metadata["filename"] == "hello-world.zip"
    assert result["presentation"] == "final_attachment"
    storage_key = file_metadata["storage_key"]
    assert isinstance(storage_key, str)
    output = generated_artifact_dir / storage_key
    assert output.is_file()
    with ZipFile(output) as archive:
        assert archive.namelist() == ["README.md", "main.py"]
        assert archive.read("main.py") == b"print('hello world')\n"


async def test_runtime_gateway_rejects_unsafe_project_zip_paths(tmp_path: Path) -> None:
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
    )

    for index, path in enumerate(("../main.py", "/tmp/main.py", "src/../../main.py", "NUL.txt")):
        with pytest.raises(RuntimeCapabilityError, match="file path|reserved"):
            await gateway.execute(
                tenant_id=TENANT_ID,
                run_id=RUN_ID,
                actor="engineer",
                name="project.generate_zip",
                arguments={"title": "Unsafe", "files": {path: "content"}},
                idempotency_key=f"unsafe_{index}",
            )


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({}, "files must contain 1 to 64 entries"),
        ({f"file-{index}.txt": "x" for index in range(65)}, "files must contain 1 to 64 entries"),
        ({"large.txt": "x" * 256_001}, "file content is too large"),
        (
            {f"chunk-{index}.txt": "x" * 250_000 for index in range(9)},
            "project content is too large",
        ),
    ],
)
async def test_runtime_gateway_rejects_oversized_project_zip_payloads(
    tmp_path: Path, files: dict[str, str], message: str
) -> None:
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
    )

    with pytest.raises(RuntimeCapabilityError, match=message):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="engineer",
            name="project.generate_zip",
            arguments={"title": "Oversized", "files": files},
            idempotency_key="project_zip_limits",
        )


async def test_runtime_gateway_rejects_invalid_project_zip_presentation(tmp_path: Path) -> None:
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
    )

    with pytest.raises(
        RuntimeCapabilityError,
        match="presentation must be step_detail or final_attachment",
    ):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="engineer",
            name="project.generate_zip",
            arguments={
                "title": "Invalid Presentation",
                "files": {"main.py": "print('hello')\n"},
                "presentation": "chat_inline",
            },
            idempotency_key="project_zip_bad_presentation",
        )


async def test_runtime_gateway_office_tools_require_configured_artifact_store(
    tmp_path: Path,
) -> None:
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills")

    assert gateway.is_available(TENANT_ID, "document.generate_docx") is True

    with pytest.raises(RuntimeCapabilityError, match="generated artifact store is not configured"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="writer",
            name="document.generate_docx",
            arguments={"title": "Launch Memo"},
            idempotency_key="docx_unconfigured",
        )


async def test_runtime_gateway_rejects_unsafe_docx_filename_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
    )
    build_calls: list[Path] = []

    def fake_build_docx(_blueprint: object, output: Path) -> None:
        build_calls.append(output)

    monkeypatch.setattr(runtime_module, "build_docx", fake_build_docx)

    with pytest.raises(RuntimeCapabilityError, match="filename must not contain path segments"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="writer",
            name="document.generate_docx",
            arguments={"title": "Launch Memo", "filename": "../escape.docx"},
            idempotency_key="docx_unsafe_filename",
        )

    assert build_calls == []
    assert not (tmp_path / "generated").exists()


async def test_runtime_gateway_rejects_unsafe_pptx_filename_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
    )
    build_calls: list[Path] = []

    def fake_build_pptx(_blueprint: object, output: Path) -> None:
        build_calls.append(output)

    monkeypatch.setattr(runtime_module, "build_pptx", fake_build_pptx)

    with pytest.raises(RuntimeCapabilityError, match="filename must not contain path segments"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="designer",
            name="presentation.generate_pptx",
            arguments={"title": "Launch Deck", "filename": "..\\escape.pptx"},
            idempotency_key="pptx_unsafe_filename",
        )

    assert build_calls == []
    assert not (tmp_path / "generated").exists()


async def test_runtime_gateway_pptx_template_error_does_not_echo_input(tmp_path: Path) -> None:
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
    )

    with pytest.raises(RuntimeCapabilityError) as error:
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="designer",
            name="presentation.generate_pptx",
            arguments={"title": "Launch Deck", "template_id": "secret-template-token"},
            idempotency_key="pptx_invalid_template",
        )

    assert str(error.value) == "template_id is invalid"
    assert "secret-template-token" not in str(error.value)


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("document.generate_docx", {"title": ""}, "title must not be empty"),
        ("presentation.generate_pptx", {"template_id": "dark-launch"}, "title must be a string"),
    ],
)
async def test_runtime_gateway_office_tools_raise_stable_errors_for_invalid_payloads(
    tmp_path: Path,
    name: str,
    arguments: Mapping[str, JsonValue],
    message: str,
) -> None:
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
    )

    with pytest.raises(RuntimeCapabilityError, match=message):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="writer",
            name=name,
            arguments=arguments,
            idempotency_key="invalid_payload",
        )


async def test_runtime_gateway_rejects_unknown_dotted_skill_ids(tmp_path: Path) -> None:
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills")

    assert gateway.is_available(TENANT_ID, "web.search") is False
    with pytest.raises(RuntimeCapabilityError, match="capability name is invalid"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="planner",
            name="web.search",
            arguments={"query": "blocked"},
            idempotency_key="blocked_dotted_name",
        )


def _assert_file_result(
    result: object,
    *,
    expected_mime_type: str,
) -> dict[str, str | int]:
    assert isinstance(result, dict)
    assert isinstance(result["artifact_id"], str)
    assert result["summary"]
    assert result["file"] == result["metadata"]
    file_metadata = result["file"]
    assert isinstance(file_metadata, dict)
    for key in (
        "filename",
        "mime_type",
        "size_bytes",
        "sha256",
        "storage_key",
        "download_url",
    ):
        assert key in file_metadata
    assert file_metadata["artifact_id"] == result["artifact_id"]
    assert file_metadata["mime_type"] == expected_mime_type
    assert isinstance(file_metadata["size_bytes"], int)
    assert file_metadata["size_bytes"] > 0
    assert isinstance(file_metadata["sha256"], str)
    assert len(file_metadata["sha256"]) == 64
    assert isinstance(file_metadata["storage_key"], str)
    assert str(TENANT_ID) in file_metadata["storage_key"]
    assert str(RUN_ID) in file_metadata["storage_key"]
    assert isinstance(file_metadata["download_url"], str)
    assert result["artifact_id"] in file_metadata["download_url"]
    return file_metadata
