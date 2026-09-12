from __future__ import annotations

import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID
from zipfile import ZipFile

import pytest

import agent_hub.capabilities.runtime as runtime_module
from agent_hub.capabilities.runtime import RuntimeCapabilityError, RuntimeCapabilityGateway
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
    assert result["artifacts"] == (
        {
            "kind": "video",
            "uri": "artifact://generated-video",
            "text": "artifact://generated-video",
            "logical_model": "video_primary",
            "deployment_id": "video_primary_1",
            "filename": "generated-video.mp4",
            "mime_type": "video/mp4",
            "size_bytes": len(b"video"),
            "sha256": "0cab1c9617404faf2b24e221e189ca5945813e14d3f766345b09ca13bbe28ffc",
            "download_url": file_metadata["download_url"],
            "expires_at": file_metadata["expires_at"],
            "file": file_metadata,
        },
    )
    assert result["metadata"] == file_metadata


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
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        generated_artifact_dir=tmp_path / "generated",
        multimedia_generation_executor=media_executor,
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
