from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol, cast
from uuid import UUID, uuid4

from agent_hub.capabilities.tools.calculator import Calculator
from agent_hub.capabilities.tools.workspace_read import WorkspaceReader
from agent_hub.documents.docx import DocxBlueprint, build_docx
from agent_hub.documents.pptx import PptxBlueprint, build_pptx
from agent_hub.files.generated import (
    ALLOWED_GENERATED_FILE_MIME_TYPES,
    DOCX_MIME_TYPE,
    JPEG_MIME_TYPE,
    MP4_MIME_TYPE,
    PNG_MIME_TYPE,
    PPTX_MIME_TYPE,
    WEBP_MIME_TYPE,
    ZIP_MIME_TYPE,
    GeneratedFileStore,
    safe_generated_filename,
)
from agent_hub.multimodal.generation import (
    MultimediaArtifact,
    MultimediaGenerationJob,
    MultimediaGenerationKind,
)
from agent_hub.runtime.contracts import JsonValue
from agent_hub.skills.sandbox.base import SkillInvocation, SkillSandbox
from agent_hub.skills.sandbox.systemd import SystemdSkillSandbox
from agent_hub.video.composer import (
    VideoClipInput,
    VideoComposer,
    VideoComposeRequest,
    VideoCompositionError,
)

_SAFE_CAPABILITY_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_DOCX_TOOL = "document.generate_docx"
_PPTX_TOOL = "presentation.generate_pptx"
_PROJECT_ZIP_TOOL = "project.generate_zip"
_MULTIMEDIA_TOOL = "generate_multimedia"
_COMPOSE_VIDEO_TOOL = "compose_video"
_MULTIMEDIA_ARTIFACT_TTL = timedelta(hours=24)
_MAX_PROJECT_FILES = 64
_MAX_PROJECT_FILE_BYTES = 256_000
_MAX_PROJECT_ZIP_SOURCE_BYTES = 2_000_000
_MAX_VIDEO_CLIPS = 32
_MAX_MULTIMEDIA_ARTIFACT_COUNT = 8
_VIDEO_CLIP_EXTENSIONS = {
    MP4_MIME_TYPE: (".mp4",),
    PNG_MIME_TYPE: (".png",),
    JPEG_MIME_TYPE: (".jpg", ".jpeg"),
    WEBP_MIME_TYPE: (".webp",),
}
_DOTTED_BUILT_INS = frozenset({_DOCX_TOOL, _PPTX_TOOL, _PROJECT_ZIP_TOOL})
_REPLAY_SAFE = frozenset({
    "calculator",
    "calculator_evaluate",
    "read_context",
    "workspace_read",
    _DOCX_TOOL,
    _PPTX_TOOL,
    _PROJECT_ZIP_TOOL,
    _MULTIMEDIA_TOOL,
    _COMPOSE_VIDEO_TOOL,
})
_VIDEO_CLIP_MIME_TYPES = frozenset({MP4_MIME_TYPE, PNG_MIME_TYPE, JPEG_MIME_TYPE, WEBP_MIME_TYPE})


class RuntimeCapabilityError(RuntimeError):
    """Stable runtime capability failure."""


class RuntimeMultimediaGenerationExecutor(Protocol):
    async def default_logical_model_for_multimedia(
        self,
        *,
        kind: MultimediaGenerationKind,
    ) -> str: ...

    def submit(
        self,
        *,
        kind: MultimediaGenerationKind,
        logical_model: str,
        prompt: str,
    ) -> MultimediaGenerationJob: ...

    async def run_job(
        self,
        job_id: str,
        *,
        executor_id: str,
    ) -> MultimediaGenerationJob: ...


class RuntimeVideoComposer(Protocol):
    def compose(self, request: VideoComposeRequest, output_dir: Path) -> Path: ...


class RuntimeCapabilityGateway:
    """Production capability executor for non-dangerous built-ins and approved skills."""

    def __init__(
        self,
        *,
        skill_store_dir: Path,
        workspace_root: Path | None = None,
        generated_artifact_dir: Path | None = None,
        skill_sandbox: SkillSandbox | None = None,
        calculator: Calculator | None = None,
        multimedia_generation_executor: RuntimeMultimediaGenerationExecutor | None = None,
        video_composer: RuntimeVideoComposer | None = None,
    ) -> None:
        self._skill_store_dir = skill_store_dir
        self._workspace_root = workspace_root
        self._generated_file_store = (
            GeneratedFileStore(generated_artifact_dir) if generated_artifact_dir is not None else None
        )
        self._skill_sandbox = skill_sandbox or SystemdSkillSandbox()
        self._calculator = calculator or Calculator()
        self._multimedia_generation_executor = multimedia_generation_executor
        self._video_composer = video_composer or VideoComposer()

    def is_replay_safe(self, name: str) -> bool:
        return name in _REPLAY_SAFE

    def is_available(self, tenant_id: UUID, name: str) -> bool:
        if name == _MULTIMEDIA_TOOL:
            return self._multimedia_generation_executor is not None
        if name == _COMPOSE_VIDEO_TOOL:
            return True
        if name in _REPLAY_SAFE:
            return True
        if _SAFE_CAPABILITY_NAME.fullmatch(name) is None:
            return False
        return self._skill_package_path(tenant_id, name).is_file()

    async def execute(
        self,
        *,
        tenant_id: UUID,
        run_id: UUID,
        actor: str,
        name: str,
        arguments: Mapping[str, JsonValue],
        idempotency_key: str,
    ) -> Mapping[str, JsonValue]:
        _require_safe("actor", actor)
        _require_safe("capability name", name)
        _require_safe("idempotency key", idempotency_key, max_length=160)
        if name in {"calculator", "calculator_evaluate"}:
            return self._execute_calculator(arguments)
        if name == "read_context":
            return self._execute_read_context(arguments)
        if name == "workspace_read":
            return self._execute_workspace_read(arguments)
        if name == _DOCX_TOOL:
            return self._execute_generate_docx(tenant_id, run_id, arguments)
        if name == _PPTX_TOOL:
            return self._execute_generate_pptx(tenant_id, run_id, arguments)
        if name == _PROJECT_ZIP_TOOL:
            return self._execute_generate_project_zip(tenant_id, run_id, arguments)
        if name == _MULTIMEDIA_TOOL:
            return await self._execute_generate_multimedia(tenant_id, run_id, actor, arguments)
        if name == _COMPOSE_VIDEO_TOOL:
            return await self._execute_compose_video(tenant_id, run_id, arguments)
        return await self._execute_skill(
            tenant_id=tenant_id,
            run_id=run_id,
            actor=actor,
            skill_id=name,
            arguments=arguments,
            idempotency_key=idempotency_key,
        )

    async def default_multimedia_logical_model(self, kind: str) -> str | None:
        executor = self._require_multimedia_generation_executor()
        selector = getattr(executor, "default_logical_model", None)
        if not callable(selector):
            return None
        result = selector(kind)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, str) and result.strip() else None

    def _execute_calculator(self, arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        expression = arguments.get("expression")
        if not isinstance(expression, str):
            raise RuntimeCapabilityError("calculator requires expression")
        result = self._calculator.evaluate(expression)
        return {"value": str(result.value)}

    def _execute_read_context(self, arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        path = arguments.get("path")
        if isinstance(path, str):
            return self._execute_workspace_read(arguments)
        query = arguments.get("query")
        if query is None:
            query = arguments.get("text")
        if query is not None and (not isinstance(query, str) or not query.strip()):
            raise RuntimeCapabilityError("read_context query must be a nonblank string")
        return {
            "query": query.strip() if isinstance(query, str) else None,
            "matches": (),
            "summary": "No additional runtime context is available for this query.",
            "truncated": False,
        }

    def _execute_workspace_read(self, arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if self._workspace_root is None:
            raise RuntimeCapabilityError("workspace reader is not configured")
        path = arguments.get("path")
        if not isinstance(path, str):
            raise RuntimeCapabilityError("workspace reader requires path")
        result = WorkspaceReader(self._workspace_root).read(path)
        return {
            "path": result.relative_path,
            "text": result.text,
            "truncated": result.truncated,
        }

    def _execute_generate_docx(
        self,
        tenant_id: UUID,
        run_id: UUID,
        arguments: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        store = self._require_generated_file_store()
        title = _required_string(arguments, "title")
        sections = _optional_mapping_list(arguments, "sections")
        blueprint = DocxBlueprint(
            title=title,
            subtitle=_optional_string(arguments, "subtitle"),
            sections=sections,
        )
        filename = _filename(arguments, title=title, extension=".docx")
        artifact_id = uuid4()
        with tempfile.TemporaryDirectory(prefix="agent-hub-docx-") as temporary_dir:
            output = Path(temporary_dir) / filename
            try:
                build_docx(blueprint, output)
            except ValueError as error:
                raise RuntimeCapabilityError(str(error)) from None
            metadata = store.store_bytes(
                tenant_id=tenant_id,
                run_id=run_id,
                artifact_id=artifact_id,
                filename=filename,
                mime_type=DOCX_MIME_TYPE,
                data=output.read_bytes(),
            )
        result = dict(
            _file_result(
                artifact_id=artifact_id,
                metadata=metadata.to_public_dict(),
                summary=f"Generated DOCX artifact {metadata.filename}.",
            )
        )
        result["presentation"] = _generated_file_presentation(
            arguments,
            default="final_attachment",
        )
        return result

    def _execute_generate_pptx(
        self,
        tenant_id: UUID,
        run_id: UUID,
        arguments: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        store = self._require_generated_file_store()
        title = _required_string(arguments, "title")
        slides = _optional_mapping_list(arguments, "slides")
        blueprint = PptxBlueprint(
            title=title,
            subtitle=_optional_string(arguments, "subtitle"),
            template_id=_optional_string(arguments, "template_id") or "consulting-clean",
            slides=slides,
        )
        filename = _filename(arguments, title=title, extension=".pptx")
        artifact_id = uuid4()
        with tempfile.TemporaryDirectory(prefix="agent-hub-pptx-") as temporary_dir:
            output = Path(temporary_dir) / filename
            try:
                build_pptx(blueprint, output)
            except ValueError as error:
                if str(error).startswith("unknown PPTX template:"):
                    raise RuntimeCapabilityError("template_id is invalid") from None
                raise RuntimeCapabilityError(str(error)) from None
            metadata = store.store_bytes(
                tenant_id=tenant_id,
                run_id=run_id,
                artifact_id=artifact_id,
                filename=filename,
                mime_type=PPTX_MIME_TYPE,
                data=output.read_bytes(),
            )
        result = dict(
            _file_result(
                artifact_id=artifact_id,
                metadata=metadata.to_public_dict(),
                summary=f"Generated PPTX artifact {metadata.filename}.",
            )
        )
        result["presentation"] = _generated_file_presentation(
            arguments,
            default="final_attachment",
        )
        return result

    def _execute_generate_project_zip(
        self,
        tenant_id: UUID,
        run_id: UUID,
        arguments: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        store = self._require_generated_file_store()
        title = _required_string(arguments, "title")
        files = _project_files(arguments)
        filename = _filename(arguments, title=title, extension=".zip")
        artifact_id = uuid4()
        with tempfile.TemporaryDirectory(prefix="agent-hub-project-") as temporary_dir:
            output = Path(temporary_dir) / filename
            with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path, data in sorted(files.items()):
                    archive.writestr(path, data)
            metadata = store.store_bytes(
                tenant_id=tenant_id,
                run_id=run_id,
                artifact_id=artifact_id,
                filename=filename,
                mime_type=ZIP_MIME_TYPE,
                data=output.read_bytes(),
            )
        result = dict(
            _file_result(
                artifact_id=artifact_id,
                metadata=metadata.to_public_dict(),
                summary=f"Generated project ZIP artifact {metadata.filename}.",
            )
        )
        result["presentation"] = _generated_file_presentation(
            arguments,
            default="final_attachment",
        )
        return result

    async def _execute_generate_multimedia(
        self,
        tenant_id: UUID,
        run_id: UUID,
        actor: str,
        arguments: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        executor = self._require_multimedia_generation_executor()
        kind = _multimedia_kind(arguments)
        logical_model = _required_string(arguments, "logical_model").strip()
        prompt_field = "generation_prompt" if "generation_prompt" in arguments else "prompt"
        prompt = _required_string(arguments, prompt_field).strip()
        prompts = _multimedia_generation_prompts(arguments, fallback_prompt=prompt)
        media_results: list[Mapping[str, JsonValue]] = []
        first_file_metadata: dict[str, JsonValue] | None = None
        completed_jobs: list[MultimediaGenerationJob] = []
        for item_prompt in prompts:
            job = executor.submit(kind=kind, logical_model=logical_model, prompt=item_prompt)
            completed = await executor.run_job(job.id, executor_id=actor)
            completed_jobs.append(completed)
            expires_at = (
                completed.expires_at.isoformat() if completed.expires_at is not None else None
            )
            for index, artifact in enumerate(completed.artifacts):
                file_metadata = _stored_multimedia_file_metadata(
                    self._generated_file_store,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    artifact=artifact,
                    expires_at=expires_at,
                )
                if file_metadata is not None and first_file_metadata is None:
                    first_file_metadata = file_metadata
                media_results.append(
                    _multimedia_artifact_result(
                        artifact,
                        job_id=completed.id,
                        artifact_index=index,
                        expires_at=expires_at,
                        file_metadata=file_metadata,
                    )
                )
        first_completed = completed_jobs[0]
        artifact_total = max(1, len(media_results))
        summary = (
            f"Generated {kind.value} artifact with {logical_model}."
            if artifact_total == 1
            else f"Generated {artifact_total} {kind.value} artifacts with {logical_model}."
        )
        result: dict[str, JsonValue] = {
            "job_id": first_completed.id,
            "kind": first_completed.kind.value,
            "logical_model": first_completed.logical_model,
            "status": first_completed.status.value,
            "executor_id": first_completed.executor_id,
            "summary": summary,
            "artifacts": tuple(media_results),
            "presentation": "final_attachment",
        }
        if len(completed_jobs) > 1:
            result["job_ids"] = tuple(job.id for job in completed_jobs)
        if first_file_metadata is not None:
            result["artifact_id"] = first_file_metadata["artifact_id"]
            result["file"] = first_file_metadata
            result["metadata"] = first_file_metadata
        return result

    async def _execute_compose_video(
        self,
        tenant_id: UUID,
        run_id: UUID,
        arguments: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        store = self._require_generated_file_store()
        title = _required_string(arguments, "title")
        filename = _filename(arguments, title=title, extension=".mp4")
        clips = _video_clip_inputs(arguments, store=store, tenant_id=tenant_id, run_id=run_id)
        request = VideoComposeRequest(
            title=title,
            clips=clips,
            output_filename=filename,
            aspect_ratio=_optional_string(arguments, "aspect_ratio") or "original",
            image_duration_seconds=_optional_int(
                arguments,
                "image_duration_seconds",
                default=3,
            ),
        )
        artifact_id = uuid4()
        with tempfile.TemporaryDirectory(prefix="agent-hub-video-") as temporary_dir:
            output_dir = Path(temporary_dir)
            try:
                output = await asyncio.to_thread(self._video_composer.compose, request, output_dir)
            except VideoCompositionError as error:
                raise RuntimeCapabilityError(str(error)) from None
            try:
                data = output.read_bytes()
            except OSError:
                raise RuntimeCapabilityError("composed video output is unavailable") from None
            metadata = store.store_bytes(
                tenant_id=tenant_id,
                run_id=run_id,
                artifact_id=artifact_id,
                filename=filename,
                mime_type=MP4_MIME_TYPE,
                data=data,
            )
        result = dict(
            _file_result(
                artifact_id=artifact_id,
                metadata=metadata.to_public_dict(),
                summary=f"Composed video artifact {metadata.filename}.",
            )
        )
        result["presentation"] = _generated_file_presentation(arguments, default="final_attachment")
        return result

    def _require_generated_file_store(self) -> GeneratedFileStore:
        if self._generated_file_store is None:
            raise RuntimeCapabilityError("generated artifact store is not configured")
        return self._generated_file_store

    def _require_multimedia_generation_executor(self) -> RuntimeMultimediaGenerationExecutor:
        if self._multimedia_generation_executor is None:
            raise RuntimeCapabilityError("multimedia generation executor is not configured")
        return self._multimedia_generation_executor

    async def default_logical_model_for_multimedia(
        self,
        *,
        tenant_id: UUID,
        kind: str,
    ) -> str:
        del tenant_id
        executor = self._require_multimedia_generation_executor()
        try:
            generation_kind = MultimediaGenerationKind(kind)
        except ValueError:
            raise RuntimeCapabilityError("kind must be image, video, or audio") from None
        return await executor.default_logical_model_for_multimedia(kind=generation_kind)

    async def _execute_skill(
        self,
        *,
        tenant_id: UUID,
        run_id: UUID,
        actor: str,
        skill_id: str,
        arguments: Mapping[str, JsonValue],
        idempotency_key: str,
    ) -> Mapping[str, JsonValue]:
        package_path = self._skill_package_path(tenant_id, skill_id)
        if not package_path.is_file():
            raise RuntimeCapabilityError("skill is not installed or approved")
        archive_bytes = package_path.read_bytes()
        package_sha256 = hashlib.sha256(archive_bytes).hexdigest()
        execution_id = _execution_id(actor, skill_id, idempotency_key)
        writable_tmp_path = self._skill_store_dir / str(tenant_id) / "tmp" / execution_id
        writable_tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            result = await self._skill_sandbox.run(
                SkillInvocation(
                    execution_id=execution_id,
                    package_path=package_path,
                    package_sha256=package_sha256,
                    input={
                        "run_id": str(run_id),
                        "actor": actor,
                        "skill": skill_id,
                        "arguments": _json_dict(arguments),
                    },
                    timeout_seconds=300,
                    output_limit_bytes=1_000_000,
                    memory_limit_bytes=512 * 1024 * 1024,
                    cpu_quota_percent=100,
                    writable_tmp_path=writable_tmp_path,
                )
            )
        finally:
            shutil.rmtree(writable_tmp_path, ignore_errors=True)
        if result.timed_out:
            raise RuntimeCapabilityError("skill execution timed out")
        if result.exit_code != 0:
            raise RuntimeCapabilityError("skill execution failed")
        parsed = _parse_stdout(result.stdout)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result": parsed,
        }

    def _skill_package_path(self, tenant_id: UUID, skill_id: str) -> Path:
        root = (self._skill_store_dir / str(tenant_id)).resolve()
        target = (root / f"{skill_id}.zip").resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise RuntimeCapabilityError("skill path is invalid") from None
        return target


def _require_safe(name: str, value: str, *, max_length: int = 128) -> None:
    if name == "capability name" and value in _DOTTED_BUILT_INS:
        return
    if (
        not isinstance(value, str)
        or len(value) > max_length
        or _SAFE_CAPABILITY_NAME.fullmatch(value) is None
    ):
        raise RuntimeCapabilityError(f"{name} is invalid")


def _required_string(arguments: Mapping[str, JsonValue], field_name: str) -> str:
    value = arguments.get(field_name)
    if not isinstance(value, str):
        raise RuntimeCapabilityError(f"{field_name} must be a string")
    if not value.strip():
        raise RuntimeCapabilityError(f"{field_name} must not be empty")
    return value


def _optional_string(arguments: Mapping[str, JsonValue], field_name: str) -> str | None:
    value = arguments.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeCapabilityError(f"{field_name} must be a string")
    return value


def _optional_int(
    arguments: Mapping[str, JsonValue],
    field_name: str,
    *,
    default: int,
) -> int:
    value = arguments.get(field_name)
    if value is None:
        return default
    if type(value) is not int:
        raise RuntimeCapabilityError(f"{field_name} must be an integer")
    return value


def _optional_mapping_list(
    arguments: Mapping[str, JsonValue],
    field_name: str,
) -> list[dict[str, object]]:
    value = arguments.get(field_name)
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise RuntimeCapabilityError(f"{field_name} must be a list")
    items: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RuntimeCapabilityError(f"{field_name} items must be objects")
        items.append(dict(item))
    return items


def _video_clip_inputs(
    arguments: Mapping[str, JsonValue],
    *,
    store: GeneratedFileStore,
    tenant_id: UUID,
    run_id: UUID,
) -> tuple[VideoClipInput, ...]:
    value = arguments.get("clips")
    if not isinstance(value, list | tuple) or not value or len(value) > _MAX_VIDEO_CLIPS:
        raise RuntimeCapabilityError("clips must contain 1 to 32 entries")
    clips: list[VideoClipInput] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RuntimeCapabilityError("clips items must be objects")
        storage_key = _clip_required_string(item, "storage_key")
        mime_type = _clip_required_string(item, "mime_type")
        if mime_type not in _VIDEO_CLIP_MIME_TYPES:
            raise RuntimeCapabilityError("unsupported clip MIME type")
        _validate_clip_storage_filename(storage_key, mime_type)
        path = _resolve_generated_clip_path(
            store,
            tenant_id=tenant_id,
            run_id=run_id,
            storage_key=storage_key,
        )
        clips.append(
            VideoClipInput(
                storage_key=storage_key,
                path=path,
                mime_type=mime_type,
                duration_seconds=_clip_optional_int(item, "duration_seconds"),
                filename=_clip_optional_string(item, "filename"),
            )
        )
    return tuple(clips)


def _validate_clip_storage_filename(storage_key: str, mime_type: str) -> None:
    filename = PurePosixPath(storage_key).name.casefold()
    expected_extensions = _VIDEO_CLIP_EXTENSIONS.get(mime_type, ())
    if not expected_extensions or not filename.endswith(expected_extensions):
        raise RuntimeCapabilityError("mime_type does not match clip filename")


def _resolve_generated_clip_path(
    store: GeneratedFileStore,
    *,
    tenant_id: UUID,
    run_id: UUID,
    storage_key: str,
) -> Path:
    try:
        parts = PurePosixPath(storage_key).parts
        if len(parts) != 4:
            raise ValueError
        stored_tenant_id = UUID(parts[0])
        stored_run_id = UUID(parts[1])
        artifact_id = UUID(parts[2])
        if stored_tenant_id != tenant_id or stored_run_id != run_id:
            raise ValueError
        return store.resolve_for(tenant_id, run_id, artifact_id, storage_key)
    except (FileNotFoundError, ValueError):
        raise RuntimeCapabilityError("clip storage_key is invalid or unavailable") from None


def _clip_required_string(arguments: Mapping[str, JsonValue], field_name: str) -> str:
    value = arguments.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeCapabilityError(f"{field_name} must be a nonblank string")
    return value.strip()


def _clip_optional_string(arguments: Mapping[str, JsonValue], field_name: str) -> str | None:
    value = arguments.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeCapabilityError(f"{field_name} must be a nonblank string")
    return value.strip()


def _clip_optional_int(arguments: Mapping[str, JsonValue], field_name: str) -> int | None:
    value = arguments.get(field_name)
    if value is None:
        return None
    if type(value) is not int:
        raise RuntimeCapabilityError(f"{field_name} must be an integer")
    return value


def _multimedia_generation_prompts(
    arguments: Mapping[str, JsonValue],
    *,
    fallback_prompt: str,
) -> tuple[str, ...]:
    raw_prompts = arguments.get("artifact_prompts")
    raw_count = arguments.get("artifact_count")
    if raw_prompts is not None:
        if not isinstance(raw_prompts, list | tuple):
            raise RuntimeCapabilityError("artifact_prompts must be a list")
        prompts = tuple(_nonblank_prompt(item, "artifact_prompts item") for item in raw_prompts)
        if not 1 <= len(prompts) <= _MAX_MULTIMEDIA_ARTIFACT_COUNT:
            raise RuntimeCapabilityError("artifact_prompts must contain 1 to 8 entries")
        if raw_count is not None and raw_count != len(prompts):
            raise RuntimeCapabilityError("artifact_count must match artifact_prompts length")
        return prompts
    count = _optional_int(arguments, "artifact_count", default=1)
    if not 1 <= count <= _MAX_MULTIMEDIA_ARTIFACT_COUNT:
        raise RuntimeCapabilityError("artifact_count must be between 1 and 8")
    if count == 1:
        return (fallback_prompt,)
    return tuple(
        f"{fallback_prompt}\n\n输出第 {index}/{count} 个独立产物。"
        for index in range(1, count + 1)
    )


def _nonblank_prompt(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise RuntimeCapabilityError(f"{field_name} must be a nonblank string")
    return value.strip()


def _multimedia_kind(arguments: Mapping[str, JsonValue]) -> MultimediaGenerationKind:
    value = _required_string(arguments, "kind").strip()
    try:
        return MultimediaGenerationKind(value)
    except ValueError:
        raise RuntimeCapabilityError("kind must be image, video, or audio") from None


def _multimedia_artifact_result(
    artifact: MultimediaArtifact,
    *,
    job_id: str,
    artifact_index: int,
    expires_at: str | None = None,
    file_metadata: Mapping[str, JsonValue] | None = None,
) -> Mapping[str, JsonValue]:
    if file_metadata is not None:
        return {
            "kind": artifact.kind.value,
            "uri": artifact.uri,
            "text": artifact.text,
            "logical_model": artifact.logical_model,
            "deployment_id": artifact.deployment_id,
            "filename": file_metadata["filename"],
            "mime_type": file_metadata["mime_type"],
            "size_bytes": file_metadata["size_bytes"],
            "sha256": file_metadata["sha256"],
            "download_url": file_metadata["download_url"],
            "expires_at": expires_at,
            "file": dict(file_metadata),
        }
    download_url: str | None = None
    size_bytes: int | None = None
    digest: str | None = None
    filename = artifact.filename
    mime_type = artifact.mime_type
    if (
        artifact.file_path is not None
        and filename is not None
        and mime_type is not None
        and artifact.file_path.is_file()
    ):
        try:
            filename = safe_generated_filename(filename)
            if mime_type in ALLOWED_GENERATED_FILE_MIME_TYPES:
                data = artifact.file_path.read_bytes()
                size_bytes = len(data)
                digest = hashlib.sha256(data).hexdigest()
                download_url = (
                    f"/api/v1/admin/multimedia/jobs/{job_id}/artifacts/{artifact_index}/download"
                )
        except (OSError, ValueError):
            filename = None
            mime_type = None
    return {
        "kind": artifact.kind.value,
        "uri": artifact.uri,
        "text": artifact.text,
        "logical_model": artifact.logical_model,
        "deployment_id": artifact.deployment_id,
        "filename": filename,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "sha256": digest,
        "download_url": download_url,
        "expires_at": expires_at,
    }


def _stored_multimedia_file_metadata(
    store: GeneratedFileStore | None,
    *,
    tenant_id: UUID,
    run_id: UUID,
    artifact: MultimediaArtifact,
    expires_at: str | None,
) -> dict[str, JsonValue] | None:
    if (
        store is None
        or artifact.file_path is None
        or artifact.filename is None
        or artifact.mime_type is None
        or not artifact.file_path.is_file()
    ):
        return None
    try:
        artifact_id = uuid4()
        metadata = store.store_bytes(
            tenant_id=tenant_id,
            run_id=run_id,
            artifact_id=artifact_id,
            filename=artifact.filename,
            mime_type=artifact.mime_type,
            data=artifact.file_path.read_bytes(),
        )
    except (OSError, ValueError):
        return None
    return {
        "artifact_id": str(artifact_id),
        **metadata.to_public_dict(),
        "expires_at": expires_at
        if expires_at is not None
        else (datetime.now(UTC) + _MULTIMEDIA_ARTIFACT_TTL).isoformat(),
    }


def _filename(arguments: Mapping[str, JsonValue], *, title: str, extension: str) -> str:
    value = arguments.get("filename")
    if value is not None:
        if not isinstance(value, str):
            raise RuntimeCapabilityError("filename must be a string")
        try:
            return safe_generated_filename(value)
        except ValueError as error:
            raise RuntimeCapabilityError(str(error)) from None
    basename = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
    if not basename:
        basename = "artifact"
    try:
        return safe_generated_filename(f"{basename[:80]}{extension}")
    except ValueError as error:
        raise RuntimeCapabilityError(str(error)) from None


def _generated_file_presentation(
    arguments: Mapping[str, JsonValue], *, default: str = "step_detail"
) -> str:
    value = arguments.get("presentation")
    if value is None:
        return default
    if value in {"step_detail", "final_attachment"}:
        return str(value)
    raise RuntimeCapabilityError("presentation must be step_detail or final_attachment")


def _project_files(arguments: Mapping[str, JsonValue]) -> dict[str, bytes]:
    raw_files = arguments.get("files")
    if not isinstance(raw_files, Mapping):
        raise RuntimeCapabilityError("files must be an object")
    if not raw_files or len(raw_files) > _MAX_PROJECT_FILES:
        raise RuntimeCapabilityError("files must contain 1 to 64 entries")
    files: dict[str, bytes] = {}
    total_bytes = 0
    for raw_path, raw_content in raw_files.items():
        if not isinstance(raw_path, str):
            raise RuntimeCapabilityError("file paths must be strings")
        path = _project_archive_path(raw_path)
        if isinstance(raw_content, str):
            data = raw_content.encode("utf-8")
        else:
            raise RuntimeCapabilityError("file contents must be strings")
        if len(data) > _MAX_PROJECT_FILE_BYTES:
            raise RuntimeCapabilityError("file content is too large")
        total_bytes += len(data)
        if total_bytes > _MAX_PROJECT_ZIP_SOURCE_BYTES:
            raise RuntimeCapabilityError("project content is too large")
        files[path] = data
    return files


def _project_archive_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if not normalized or normalized != path.strip():
        raise RuntimeCapabilityError("file path is invalid")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(path)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or any(part in {"", ".", ".."} for part in posix.parts)
        or len(posix.parts) > 8
    ):
        raise RuntimeCapabilityError("file path is invalid")
    for part in posix.parts:
        try:
            safe_generated_filename(part)
        except ValueError as error:
            raise RuntimeCapabilityError(str(error)) from None
    return posix.as_posix()


def _file_result(
    *,
    artifact_id: UUID,
    metadata: dict[str, str | int],
    summary: str,
) -> Mapping[str, JsonValue]:
    public_metadata: dict[str, JsonValue] = {
        "artifact_id": str(artifact_id),
        **metadata,
    }
    return {
        "artifact_id": str(artifact_id),
        "file": public_metadata,
        "metadata": public_metadata,
        "summary": summary,
    }


def _execution_id(actor: str, skill_id: str, idempotency_key: str) -> str:
    digest = hashlib.sha256(f"{actor}:{skill_id}:{idempotency_key}".encode()).hexdigest()[:24]
    return f"skill_{digest}"


def _json_dict(value: Mapping[str, JsonValue]) -> dict[str, object]:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    parsed = json.loads(encoded)
    if not isinstance(parsed, dict):
        raise RuntimeCapabilityError("capability arguments are invalid")
    return parsed


def _parse_stdout(value: str) -> JsonValue:
    if not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return _normalize_json_value(parsed)


def _normalize_json_value(value: object) -> JsonValue:
    if value is None or type(value) in {bool, int, str}:
        return cast(JsonValue, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeCapabilityError("skill stdout is not JSON serializable")
        return value
    if isinstance(value, list):
        return tuple(_normalize_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    raise RuntimeCapabilityError("skill stdout is not JSON serializable")
