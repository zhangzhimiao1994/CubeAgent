from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol, cast
from uuid import UUID, uuid4

from agent_hub.capabilities.tools.calculator import Calculator
from agent_hub.capabilities.tools.workspace_read import WorkspaceReader
from agent_hub.content_studio import (
    AsyncContentStudioService,
    AsyncInMemoryContentProjectStore,
    ClaimStatus,
    ProjectStatus,
)
from agent_hub.content_studio.packs import load_pack_registry
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
from agent_hub.runtime.production import production_metadata_for_label
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
_CONTENT_STUDIO_TOOL = "content_studio"
_MULTIMEDIA_ARTIFACT_TTL = timedelta(hours=24)
_MAX_PROJECT_FILES = 64
_MAX_PROJECT_FILE_BYTES = 256_000
_MAX_PROJECT_ZIP_SOURCE_BYTES = 2_000_000
_MAX_VIDEO_CLIPS = 32
_MAX_MULTIMEDIA_ARTIFACT_COUNT = 24
_MAX_VISUAL_ASSET_GENERATION_ATTEMPTS = 2
_MULTIMEDIA_IMAGE_JOB_TIMEOUT_SECONDS = 600
_MULTIMEDIA_VIDEO_JOB_TIMEOUT_SECONDS = 1_200
_MULTIMEDIA_AUDIO_JOB_TIMEOUT_SECONDS = 420
_MULTIMEDIA_IMAGE_PROVIDER_RETRY_ATTEMPTS = 2
_MULTIMEDIA_IMAGE_PROVIDER_RETRY_BACKOFF_SECONDS = 15
_MULTIMEDIA_IMAGE_PROVIDER_PROMPT_BYTES = 2_400
_MULTIMEDIA_VIDEO_PROVIDER_PROMPT_BYTES = 6_000
_MULTIMEDIA_AUDIO_PROVIDER_PROMPT_BYTES = 3_000
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


@dataclass(frozen=True, slots=True)
class RuntimeAssetVisualReview:
    passed: bool
    summary: str
    issues: tuple[str, ...] = ()
    confidence: float | None = None
    logical_model: str | None = None
    deployment_id: str | None = None


class RuntimeAssetVisualReviewer(Protocol):
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
    ) -> RuntimeAssetVisualReview: ...


@dataclass(frozen=True, slots=True)
class _MultimediaPromptExecution:
    prompt_index: int
    media_results: tuple[Mapping[str, JsonValue], ...]
    accepted_jobs: tuple[MultimediaGenerationJob, ...]
    attempted_jobs: tuple[MultimediaGenerationJob, ...]
    review_failed_results: tuple[Mapping[str, JsonValue], ...]
    first_file_metadata: dict[str, JsonValue] | None = None


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
        asset_visual_reviewer: RuntimeAssetVisualReviewer | None = None,
        video_composer: RuntimeVideoComposer | None = None,
        content_studio_service: AsyncContentStudioService | None = None,
        content_studio_execution_mode: str = "production",
        content_studio_owner_user_id: UUID | None = None,
    ) -> None:
        self._skill_store_dir = skill_store_dir
        self._workspace_root = workspace_root
        self._generated_file_store = (
            GeneratedFileStore(generated_artifact_dir) if generated_artifact_dir is not None else None
        )
        self._skill_sandbox = skill_sandbox or SystemdSkillSandbox()
        self._calculator = calculator or Calculator()
        self._multimedia_generation_executor = multimedia_generation_executor
        self._asset_visual_reviewer = asset_visual_reviewer
        self._video_composer = video_composer or VideoComposer()
        if content_studio_execution_mode not in {"demo", "production"}:
            raise ValueError("content_studio_execution_mode must be demo or production")
        self._content_studio = content_studio_service or AsyncContentStudioService(
            registry=load_pack_registry(),
            store=AsyncInMemoryContentProjectStore(),
            execution_mode=content_studio_execution_mode,
        )
        self._content_studio_execution_mode = content_studio_execution_mode
        self._content_studio_owner_user_id = content_studio_owner_user_id

    def is_replay_safe(self, name: str) -> bool:
        return name in _REPLAY_SAFE

    def is_available(self, tenant_id: UUID, name: str) -> bool:
        if name == _MULTIMEDIA_TOOL:
            return self._multimedia_generation_executor is not None
        if name == _COMPOSE_VIDEO_TOOL:
            return True
        if name == _CONTENT_STUDIO_TOOL:
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
        if name == _CONTENT_STUDIO_TOOL:
            return await self._execute_content_studio(tenant_id, arguments)
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

    async def _execute_content_studio(
        self,
        tenant_id: UUID,
        arguments: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        operation = _required_string(arguments, "operation").strip()
        owner_user_id = self._content_studio_owner_user_id
        if owner_user_id is None:
            raise RuntimeCapabilityError(
                "content_studio runtime owner principal is not configured; use the Content Studio workspace for project operations"
            )
        with _content_studio_service_scope(
            self._content_studio,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        ):
            return await self._execute_scoped_content_studio(
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                operation=operation,
                arguments=arguments,
            )

    async def _execute_scoped_content_studio(
        self,
        *,
        tenant_id: UUID,
        owner_user_id: UUID,
        operation: str,
        arguments: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        if operation == "create_content_project":
            raw_sources = arguments.get("source_urls", ())
            if raw_sources is None:
                raw_sources = ()
            if not isinstance(raw_sources, (list, tuple)) or any(
                type(item) is not str or not item.strip() for item in raw_sources
            ):
                raise RuntimeCapabilityError("source_urls must be a list of strings")
            project = await _call_content_studio_service(
                self._content_studio.create_content_project,
                title=_required_string(arguments, "title"),
                topic=_required_string(arguments, "topic"),
                source_urls=tuple(str(item).strip() for item in raw_sources),
                domain=_optional_string(arguments, "domain") or "aigc",
                format=_optional_string(arguments, "format") or "explainer",
                platform=_optional_string(arguments, "platform") or "douyin",
                channel=_optional_string(arguments, "channel") or "ai_frontier",
                style=_optional_string(arguments, "style") or "fast_minimal",
                tenant_id=str(tenant_id),
                owner_user_id=str(owner_user_id),
                execution_mode=self._content_studio_execution_mode,
            )
            return _content_project_payload(project)
        project_id = _required_string(arguments, "project_id")
        project = await self._content_studio.get_content_project(project_id)
        _validate_content_project_ownership(
            project,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        if operation == "get_content_project":
            return _content_project_payload(project)
        if operation == "run_content_project":
            until = ProjectStatus(_optional_string(arguments, "until") or ProjectStatus.QC_REVIEW.value)
            return _content_project_payload(
                await _call_content_studio_service(
                    self._content_studio.run_content_project,
                    project_id=project_id,
                    until=until,
                )
            )
        if operation == "revise_script":
            return _content_project_payload(
                await _call_content_studio_service(
                    self._content_studio.revise_script,
                    project_id=project_id,
                    instruction=_required_string(arguments, "instruction"),
                )
            )
        if operation == "revise_storyboard":
            return _content_project_payload(
                await _call_content_studio_service(
                    self._content_studio.revise_storyboard,
                    project_id=project_id,
                    instruction=_required_string(arguments, "instruction"),
                )
            )
        if operation == "regenerate_asset":
            return _content_project_payload(
                await _call_content_studio_service(
                    self._content_studio.regenerate_asset,
                    project_id=project_id,
                    asset_id=_required_string(arguments, "asset_id"),
                    instruction=_required_string(arguments, "instruction"),
                )
            )
        if operation == "regenerate_voice":
            regenerate_voice = getattr(self._content_studio, "regenerate_voice", None)
            if not callable(regenerate_voice):
                raise RuntimeCapabilityError("content_studio regenerate_voice is unavailable")
            return _content_project_payload(
                await _call_content_studio_service(
                    regenerate_voice,
                    project_id=project_id,
                    instruction=_required_string(arguments, "instruction"),
                )
            )
        if operation == "render_preview":
            return _content_project_payload(
                await _call_content_studio_service(
                    self._content_studio.render_preview,
                    project_id=project_id,
                )
            )
        if operation == "approve_script":
            self._require_content_studio_trusted_human_approval()
            return _content_project_payload(
                await _call_content_studio_service(
                    self._content_studio.approve_script,
                    project_id=project_id,
                )
            )
        if operation == "approve_rights":
            self._require_content_studio_trusted_human_approval()
            approve_rights = getattr(self._content_studio, "approve_rights", None)
            if not callable(approve_rights):
                raise RuntimeCapabilityError("content_studio approve_rights is unavailable")
            raw_asset_ids = arguments.get("asset_ids")
            if not isinstance(raw_asset_ids, (list, tuple)) or any(
                type(item) is not str or not item.strip() for item in raw_asset_ids
            ):
                raise RuntimeCapabilityError("asset_ids must be a list of strings")
            return _content_project_payload(
                await _call_content_studio_service(
                    approve_rights,
                    project_id=project_id,
                    asset_ids=tuple(str(item).strip() for item in raw_asset_ids),
                    note=_required_string(arguments, "note"),
                )
            )
        if operation == "approve_final":
            self._require_content_studio_trusted_human_approval()
            return _content_project_payload(
                await _call_content_studio_service(
                    self._content_studio.approve_final,
                    project_id=project_id,
                )
            )
        if operation == "retry_stage":
            stage = ProjectStatus(_required_string(arguments, "stage"))
            return _content_project_payload(
                await _call_content_studio_service(
                    self._content_studio.retry_stage,
                    project_id=project_id,
                    stage=stage,
                )
            )
        if operation == "replace_claim_status":
            raw_evidence_ids = arguments.get("evidence_ids", ())
            if raw_evidence_ids is None:
                raw_evidence_ids = ()
            if not isinstance(raw_evidence_ids, (list, tuple)) or any(
                type(item) is not str or not item.strip() for item in raw_evidence_ids
            ):
                raise RuntimeCapabilityError("evidence_ids must be a list of strings")
            return _content_project_payload(
                await _call_content_studio_service(
                    self._content_studio.replace_claim_status,
                    project_id=project_id,
                    claim_id=_required_string(arguments, "claim_id"),
                    status=ClaimStatus(_required_string(arguments, "status")),
                    note=_required_string(arguments, "note"),
                    evidence_ids=tuple(str(item).strip() for item in raw_evidence_ids),
                )
            )
        raise RuntimeCapabilityError("content_studio operation is invalid")

    def _require_content_studio_trusted_human_approval(self) -> None:
        raise RuntimeCapabilityError(
            "content_studio approvals require trusted workspace permission context"
        )

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
        labels = _multimedia_artifact_labels(
            arguments,
            expected_count=len(prompts),
            prompts=prompts,
        )
        preserved_results = _preserved_multimedia_artifacts(
            arguments,
            kind=kind,
            generated_count=len(prompts),
        )
        prompts = tuple(
            _bounded_multimedia_provider_prompt(kind, item_prompt)
            for item_prompt in prompts
        )
        semaphore = asyncio.Semaphore(_multimedia_parallelism(kind, len(prompts)))
        execution_tasks = tuple(
            asyncio.create_task(
                self._execute_multimedia_prompt(
                    executor=executor,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    actor=actor,
                    kind=kind,
                    logical_model=logical_model,
                    prompt_index=prompt_index,
                    item_prompt=item_prompt,
                    prompt_label=labels[prompt_index] if labels is not None else None,
                    semaphore=semaphore,
                )
            )
            for prompt_index, item_prompt in enumerate(prompts)
        )
        _done, pending = await asyncio.wait(
            execution_tasks,
            timeout=_multimedia_batch_timeout_seconds(kind, len(prompts), labels=labels),
            return_when=asyncio.ALL_COMPLETED,
        )
        timed_out_tasks = set(pending)
        if pending:
            for task in pending:
                task.cancel()
                task.add_done_callback(self._consume_background_multimedia_execution_result)
        executions: list[_MultimediaPromptExecution] = []
        for prompt_index, task in enumerate(execution_tasks):
            if task in timed_out_tasks or task.cancelled():
                executions.append(
                    _failed_multimedia_prompt_execution(
                        kind=kind,
                        prompt_index=prompt_index,
                        prompt_label=labels[prompt_index] if labels is not None else None,
                        item_prompt=prompts[prompt_index],
                        error=RuntimeCapabilityError(
                            f"{kind.value} generation timed out for "
                            f"{labels[prompt_index] if labels is not None else '媒体资产'}"
                        ),
                    )
                )
                continue
            exception = task.exception()
            if exception is not None:
                executions.append(
                    _failed_multimedia_prompt_execution(
                        kind=kind,
                        prompt_index=prompt_index,
                        prompt_label=labels[prompt_index] if labels is not None else None,
                        item_prompt=prompts[prompt_index],
                        error=exception,
                    )
                )
                continue
            executions.append(task.result())
        media_results: list[Mapping[str, JsonValue]] = list(preserved_results)
        first_file_metadata: dict[str, JsonValue] | None = None
        accepted_jobs: list[MultimediaGenerationJob] = []
        attempted_jobs: list[MultimediaGenerationJob] = []
        review_failed_results: list[Mapping[str, JsonValue]] = []
        for execution in sorted(executions, key=lambda item: item.prompt_index):
            media_results.extend(execution.media_results)
            accepted_jobs.extend(execution.accepted_jobs)
            attempted_jobs.extend(execution.attempted_jobs)
            review_failed_results.extend(execution.review_failed_results)
            if first_file_metadata is None and execution.first_file_metadata is not None:
                first_file_metadata = execution.first_file_metadata
        first_completed = (
            accepted_jobs[0]
            if accepted_jobs
            else attempted_jobs[0]
            if attempted_jobs
            else None
        )
        artifact_total = max(1, len(media_results))
        preserved_count = len(preserved_results)
        generated_count = max(0, artifact_total - preserved_count)
        summary = (
            f"Generated {kind.value} artifact with {logical_model}."
            if artifact_total == 1
            else f"Generated {artifact_total} {kind.value} artifacts with {logical_model}."
        )
        if preserved_count:
            summary = (
                f"Generated {generated_count} new {kind.value} artifacts and reused "
                f"{preserved_count} approved {kind.value} artifacts with {logical_model}."
            )
        result: dict[str, JsonValue] = {
            "job_id": first_completed.id if first_completed is not None else "unavailable",
            "kind": first_completed.kind.value if first_completed is not None else kind.value,
            "logical_model": (
                first_completed.logical_model if first_completed is not None else logical_model
            ),
            "status": (
                first_completed.status.value if first_completed is not None else "failed"
            ),
            "executor_id": first_completed.executor_id if first_completed is not None else actor,
            "summary": summary,
            "artifacts": tuple(media_results),
            "presentation": "final_attachment",
        }
        if preserved_count:
            result["preserved_artifact_count"] = preserved_count
            result["generated_artifact_count"] = generated_count
        if review_failed_results:
            result["review_status"] = "needs_user_revision"
            result["review_failed_artifact_count"] = len(review_failed_results)
            if preserved_count:
                result["summary"] = (
                    f"Generated {generated_count} new {kind.value} artifacts and reused "
                    f"{preserved_count} approved {kind.value} artifacts with {logical_model}; "
                    f"{len(review_failed_results)} require user review/regeneration."
                )
            else:
                result["summary"] = (
                    f"Generated {artifact_total} {kind.value} artifacts with {logical_model}; "
                    f"{len(review_failed_results)} require user review/regeneration."
                )
        if len(accepted_jobs) > 1:
            result["job_ids"] = tuple(job.id for job in accepted_jobs)
        if len(attempted_jobs) > len(accepted_jobs):
            result["attempted_job_ids"] = tuple(job.id for job in attempted_jobs)
        if first_file_metadata is not None:
            result["artifact_id"] = first_file_metadata["artifact_id"]
            result["file"] = first_file_metadata
            result["metadata"] = first_file_metadata
        return result

    async def _execute_multimedia_prompt(
        self,
        *,
        executor: RuntimeMultimediaGenerationExecutor,
        tenant_id: UUID,
        run_id: UUID,
        actor: str,
        kind: MultimediaGenerationKind,
        logical_model: str,
        prompt_index: int,
        item_prompt: str,
        prompt_label: str | None,
        semaphore: asyncio.Semaphore,
    ) -> _MultimediaPromptExecution:
        media_results: list[Mapping[str, JsonValue]] = []
        first_file_metadata: dict[str, JsonValue] | None = None
        accepted_jobs: list[MultimediaGenerationJob] = []
        attempted_jobs: list[MultimediaGenerationJob] = []
        review_failed_results: list[Mapping[str, JsonValue]] = []
        async with semaphore:
            attempt_prompt = item_prompt
            for attempt_index in range(_MAX_VISUAL_ASSET_GENERATION_ATTEMPTS):
                provider_attempt = 1
                while True:
                    provider_prompt = _bounded_multimedia_provider_prompt(kind, attempt_prompt)
                    job = executor.submit(
                        kind=kind,
                        logical_model=logical_model,
                        prompt=provider_prompt,
                    )
                    try:
                        completed = await self._run_multimedia_job_with_hard_timeout(
                            executor,
                            job_id=job.id,
                            executor_id=actor,
                            timeout_seconds=_multimedia_prompt_timeout_seconds(
                                kind,
                                prompt_label,
                            ),
                        )
                        break
                    except TimeoutError:
                        raise RuntimeCapabilityError(
                            f"{kind.value} generation timed out for {prompt_label or '媒体资产'}"
                        ) from None
                    except Exception as exc:
                        if (
                            provider_attempt >= _MULTIMEDIA_IMAGE_PROVIDER_RETRY_ATTEMPTS
                            or not _is_retryable_multimedia_provider_error(kind, exc)
                        ):
                            raise
                        await asyncio.sleep(
                            _multimedia_provider_retry_backoff_seconds(provider_attempt)
                        )
                        provider_attempt += 1
                attempted_jobs.append(completed)
                expires_at = (
                    completed.expires_at.isoformat() if completed.expires_at is not None else None
                )
                attempt_results: list[Mapping[str, JsonValue]] = []
                attempt_first_file_metadata: dict[str, JsonValue] | None = None
                failed_review: RuntimeAssetVisualReview | None = None
                failed_result: Mapping[str, JsonValue] | None = None
                for index, artifact in enumerate(completed.artifacts):
                    file_metadata = _stored_multimedia_file_metadata(
                        self._generated_file_store,
                        tenant_id=tenant_id,
                        run_id=run_id,
                        artifact=artifact,
                        expires_at=expires_at,
                    )
                    if file_metadata is not None and attempt_first_file_metadata is None:
                        attempt_first_file_metadata = file_metadata
                    try:
                        visual_review = await self._review_multimedia_image_asset(
                            tenant_id=tenant_id,
                            run_id=run_id,
                            kind=kind,
                            label=prompt_label,
                            prompt=provider_prompt,
                            artifact=artifact,
                            file_metadata=file_metadata,
                        )
                    except RuntimeCapabilityError as exc:
                        visual_review = _visual_review_unavailable(
                            label=prompt_label,
                            error=exc,
                        )
                    result_item = _multimedia_artifact_result(
                        artifact,
                        job_id=completed.id,
                        artifact_index=index,
                        expires_at=expires_at,
                        file_metadata=file_metadata,
                        label=_multimedia_result_label(
                            prompt_label,
                            artifact_index=index,
                            artifact_count=len(completed.artifacts),
                        ),
                        generation_prompt=provider_prompt,
                        visual_review=visual_review,
                    )
                    if visual_review is not None and not visual_review.passed:
                        failed_review = visual_review
                        failed_result = result_item
                        break
                    attempt_results.append(result_item)
                if failed_review is None:
                    media_results.extend(attempt_results)
                    accepted_jobs.append(completed)
                    if first_file_metadata is None and attempt_first_file_metadata is not None:
                        first_file_metadata = attempt_first_file_metadata
                    break
                if _visual_review_is_unavailable(failed_review):
                    if failed_result is not None:
                        review_failed_results.append(failed_result)
                        media_results.append(failed_result)
                        if first_file_metadata is None and attempt_first_file_metadata is not None:
                            first_file_metadata = attempt_first_file_metadata
                        break
                    raise RuntimeCapabilityError(
                        "visual asset review unavailable for "
                        f"{prompt_label}: {_visual_review_failure_reason(failed_review)}"
                    )
                if attempt_index + 1 >= _MAX_VISUAL_ASSET_GENERATION_ATTEMPTS:
                    if failed_result is not None:
                        review_failed_results.append(failed_result)
                        media_results.append(failed_result)
                        if first_file_metadata is None and attempt_first_file_metadata is not None:
                            first_file_metadata = attempt_first_file_metadata
                        break
                    raise RuntimeCapabilityError(
                        "visual asset review failed after "
                        f"{_MAX_VISUAL_ASSET_GENERATION_ATTEMPTS} attempts for "
                        f"{prompt_label}: {_visual_review_failure_reason(failed_review)}"
                    )
                attempt_prompt = _visual_asset_retry_prompt(
                    item_prompt,
                    label=prompt_label,
                    review=failed_review,
                    next_attempt=attempt_index + 2,
                )
        return _MultimediaPromptExecution(
            prompt_index=prompt_index,
            media_results=tuple(media_results),
            accepted_jobs=tuple(accepted_jobs),
            attempted_jobs=tuple(attempted_jobs),
            review_failed_results=tuple(review_failed_results),
            first_file_metadata=first_file_metadata,
        )

    async def _run_multimedia_job_with_hard_timeout(
        self,
        executor: RuntimeMultimediaGenerationExecutor,
        *,
        job_id: str,
        executor_id: str,
        timeout_seconds: int,
    ) -> MultimediaGenerationJob:
        task: asyncio.Task[MultimediaGenerationJob] = asyncio.create_task(
            executor.run_job(job_id, executor_id=executor_id)
        )
        try:
            done, _pending = await asyncio.wait({task}, timeout=timeout_seconds)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(self._consume_background_multimedia_job_result)
            raise
        if task in done:
            return task.result()
        task.cancel()
        task.add_done_callback(self._consume_background_multimedia_job_result)
        raise TimeoutError

    def _consume_background_multimedia_job_result(
        self,
        task: asyncio.Task[MultimediaGenerationJob],
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 - background callback must consume provider failures.
            return

    def _consume_background_multimedia_execution_result(
        self,
        future: asyncio.Future[_MultimediaPromptExecution],
    ) -> None:
        try:
            future.result()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 - background callback must consume provider failures.
            return

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

    async def _review_multimedia_image_asset(
        self,
        *,
        tenant_id: UUID,
        run_id: UUID,
        kind: MultimediaGenerationKind,
        label: str | None,
        prompt: str,
        artifact: MultimediaArtifact,
        file_metadata: Mapping[str, JsonValue] | None,
    ) -> RuntimeAssetVisualReview | None:
        if self._asset_visual_reviewer is None:
            return None
        if kind is not MultimediaGenerationKind.IMAGE or not _requires_visual_asset_review(label):
            return None
        if (
            file_metadata is None
            or self._generated_file_store is None
            or artifact.filename is None
            or artifact.mime_type not in {PNG_MIME_TYPE, JPEG_MIME_TYPE, WEBP_MIME_TYPE}
        ):
            raise RuntimeCapabilityError("visual asset review requires a stored image file")
        storage_key = file_metadata.get("storage_key")
        artifact_id = file_metadata.get("artifact_id")
        if type(storage_key) is not str or type(artifact_id) is not str:
            raise RuntimeCapabilityError("visual asset review requires generated file metadata")
        try:
            path = self._generated_file_store.resolve_for(
                tenant_id,
                run_id,
                UUID(artifact_id),
                storage_key,
            )
        except (ValueError, FileNotFoundError):
            raise RuntimeCapabilityError("visual asset review image file is unavailable") from None
        try:
            data = path.read_bytes()
        except OSError:
            raise RuntimeCapabilityError("visual asset review image file is unavailable") from None
        try:
            review = await self._asset_visual_reviewer.review_image_asset(
                tenant_id=tenant_id,
                label=label.strip() if label else "图片资产",
                prompt=prompt,
                filename=artifact.filename,
                mime_type=artifact.mime_type,
                data=data,
                image_url=_public_multimedia_artifact_url(artifact.uri),
            )
        except RuntimeCapabilityError:
            raise
        except Exception as exc:
            raise RuntimeCapabilityError(
                f"visual asset review failed for {label}: {exc}"
            ) from exc
        return review

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
            raise RuntimeCapabilityError(
                f"artifact_prompts must contain 1 to {_MAX_MULTIMEDIA_ARTIFACT_COUNT} entries"
            )
        if raw_count is not None and raw_count != len(prompts):
            raise RuntimeCapabilityError("artifact_count must match artifact_prompts length")
        return prompts
    count = _optional_int(arguments, "artifact_count", default=1)
    if not 1 <= count <= _MAX_MULTIMEDIA_ARTIFACT_COUNT:
        raise RuntimeCapabilityError(
            f"artifact_count must be between 1 and {_MAX_MULTIMEDIA_ARTIFACT_COUNT}"
        )
    if count == 1:
        return (fallback_prompt,)
    return tuple(
        f"{fallback_prompt}\n\n输出第 {index}/{count} 个独立产物。"
        for index in range(1, count + 1)
    )


def _bounded_multimedia_provider_prompt(
    kind: MultimediaGenerationKind,
    prompt: str,
) -> str:
    max_bytes = {
        MultimediaGenerationKind.IMAGE: _MULTIMEDIA_IMAGE_PROVIDER_PROMPT_BYTES,
        MultimediaGenerationKind.VIDEO: _MULTIMEDIA_VIDEO_PROVIDER_PROMPT_BYTES,
        MultimediaGenerationKind.AUDIO: _MULTIMEDIA_AUDIO_PROVIDER_PROMPT_BYTES,
    }[kind]
    prompt = prompt.strip()
    if len(prompt.encode("utf-8")) <= max_bytes:
        return prompt
    marker = "\n\n[过长上下文已压缩，保留核心生成要求和关键约束]\n\n"
    marker_bytes = len(marker.encode("utf-8"))
    tail_budget = max(240, (max_bytes - marker_bytes) // 3)
    head_budget = max(240, max_bytes - marker_bytes - tail_budget)
    return (
        _truncate_utf8_head(prompt, max_bytes=head_budget)
        + marker
        + _truncate_utf8_tail(prompt, max_bytes=tail_budget)
    ).strip()


def _truncate_utf8_head(value: str, *, max_bytes: int) -> str:
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    total = 0
    chars: list[str] = []
    for character in value:
        size = len(character.encode("utf-8"))
        if total + size > max_bytes:
            break
        chars.append(character)
        total += size
    return "".join(chars).rstrip()


def _truncate_utf8_tail(value: str, *, max_bytes: int) -> str:
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    total = 0
    chars: list[str] = []
    for character in reversed(value):
        size = len(character.encode("utf-8"))
        if total + size > max_bytes:
            break
        chars.append(character)
        total += size
    return "".join(reversed(chars)).lstrip()


def _multimedia_artifact_labels(
    arguments: Mapping[str, JsonValue],
    *,
    expected_count: int,
    prompts: tuple[str, ...],
) -> tuple[str, ...] | None:
    raw_labels = arguments.get("artifact_labels")
    if raw_labels is None:
        return _inferred_multimedia_artifact_labels(prompts)
    if not isinstance(raw_labels, list | tuple):
        raise RuntimeCapabilityError("artifact_labels must be a list")
    labels = tuple(_nonblank_prompt(item, "artifact_labels item") for item in raw_labels)
    if len(labels) != expected_count:
        raise RuntimeCapabilityError("artifact_labels must match artifact_prompts length")
    return labels


def _inferred_multimedia_artifact_labels(prompts: tuple[str, ...]) -> tuple[str, ...] | None:
    labels: list[str | None] = []
    for prompt in prompts:
        labels.append(_inferred_multimedia_artifact_label(prompt))
    if not any(label is not None for label in labels):
        return None
    return tuple(label if label is not None else "图片资产" for label in labels)


def _preserved_multimedia_artifacts(
    arguments: Mapping[str, JsonValue],
    *,
    kind: MultimediaGenerationKind,
    generated_count: int,
) -> tuple[Mapping[str, JsonValue], ...]:
    raw_items = arguments.get("preserved_artifacts")
    if raw_items is None:
        return ()
    if not isinstance(raw_items, list | tuple):
        raise RuntimeCapabilityError("preserved_artifacts must be a list")
    if len(raw_items) + generated_count > _MAX_MULTIMEDIA_ARTIFACT_COUNT:
        raise RuntimeCapabilityError(
            f"preserved_artifacts plus artifact_prompts must contain at most "
            f"{_MAX_MULTIMEDIA_ARTIFACT_COUNT} entries"
        )
    preserved: list[Mapping[str, JsonValue]] = []
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, Mapping):
            raise RuntimeCapabilityError("preserved_artifacts item must be an object")
        item_kind = raw_item.get("kind")
        if isinstance(item_kind, str) and item_kind.strip() and item_kind != kind.value:
            raise RuntimeCapabilityError("preserved_artifacts item kind must match kind")
        label = raw_item.get("label") or raw_item.get("title") or raw_item.get("filename")
        if not isinstance(label, str) or not label.strip():
            raise RuntimeCapabilityError("preserved_artifacts item must include a label")
        cleaned: dict[str, JsonValue] = {}
        for key, value in raw_item.items():
            if isinstance(key, str) and _is_json_value(value):
                cleaned[key] = value
        cleaned["kind"] = kind.value
        cleaned.setdefault("label", label.strip())
        cleaned.setdefault("title", label.strip())
        cleaned.setdefault("preserved_from_previous_attempt", True)
        cleaned.setdefault("preserved_artifact_index", index)
        preserved.append(cleaned)
    return tuple(preserved)


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, tuple | list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _inferred_multimedia_artifact_label(prompt: str) -> str | None:
    normalized = prompt.casefold()
    semantic_candidates: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "角色锁定资产",
            (
                "角色参考设定",
                "角色参考图",
                "角色设定板",
                "定妆",
                "三视",
                "character model sheet",
                "character sheet",
            ),
        ),
        (
            "服装妆造资产",
            (
                "服装妆造",
                "服装设定板",
                "妆造设定板",
                "配饰",
                "妆发",
                "costume sheet",
                "wardrobe sheet",
            ),
        ),
        (
            "场景资产",
            (
                "场景设定板",
                "主要地点",
                "关键空间",
                "空间层次",
                "背景元素",
                "environment sheet",
                "scene sheet",
            ),
        ),
        (
            "道具资产",
            (
                "道具设定板",
                "道具图",
                "剧情关键物",
                "随身物",
                "法器细节",
                "物件细节",
                "prop sheet",
                "props sheet",
                "key props",
            ),
        ),
        (
            "动作资产",
            (
                "动作姿态参考板",
                "动作参考板",
                "姿态参考板",
                "动作分解",
                "姿态线",
                "pose sheet",
                "action sheet",
            ),
        ),
        (
            "特效资产",
            (
                "特效设定板",
                "法术设定",
                "能量形态",
                "光效",
                "转场特效",
                "vfx sheet",
                "effect sheet",
            ),
        ),
        (
            "镜头资产",
            (
                "镜头语言设定板",
                "景别",
                "机位",
                "镜头运动",
                "构图参考",
                "camera sheet",
                "shot language",
            ),
        ),
        (
            "表演节奏与风格锁定资产",
            (
                "表演节奏",
                "风格锁定",
                "关键表情",
                "眼神",
                "肢体状态",
                "声音节奏",
                "performance sheet",
                "style lock",
            ),
        ),
    )
    for label, terms in semantic_candidates:
        if any(term.casefold() in normalized for term in terms):
            return label
    candidates = (
        "角色锁定资产",
        "角色资产",
        "服装妆造资产",
        "场景资产",
        "道具资产",
        "动作资产",
        "特效资产",
        "镜头资产",
        "表演节奏与风格锁定资产",
        "分镜图",
        "character model sheet",
        "asset sheet",
        "asset pack",
        "storyboard",
    )
    for candidate in candidates:
        if candidate.casefold() in normalized:
            return candidate
    if any(term in normalized for term in ("资产", "锁定", "设定表", "设定板", "参考板", "asset")):
        return "图片资产"
    return None


def _multimedia_job_timeout_seconds(kind: MultimediaGenerationKind) -> int:
    if kind is MultimediaGenerationKind.IMAGE:
        return _MULTIMEDIA_IMAGE_JOB_TIMEOUT_SECONDS
    if kind is MultimediaGenerationKind.VIDEO:
        return _MULTIMEDIA_VIDEO_JOB_TIMEOUT_SECONDS
    if kind is MultimediaGenerationKind.AUDIO:
        return _MULTIMEDIA_AUDIO_JOB_TIMEOUT_SECONDS
    raise RuntimeCapabilityError("kind must be image, video, or audio")


def _multimedia_prompt_timeout_seconds(
    kind: MultimediaGenerationKind,
    prompt_label: str | None,
) -> int:
    base_timeout = _multimedia_job_timeout_seconds(kind)
    if kind is MultimediaGenerationKind.IMAGE and prompt_label is not None:
        normalized = prompt_label.casefold()
        if "角色锁定资产" in normalized or "character model sheet" in normalized:
            return max(base_timeout, 1_200)
        if "表演节奏" in normalized or "风格锁定" in normalized:
            return max(base_timeout, 1_200)
    return base_timeout


def _multimedia_batch_timeout_seconds(
    kind: MultimediaGenerationKind,
    prompt_count: int,
    *,
    labels: tuple[str, ...] | None = None,
) -> int:
    per_job_timeout = _multimedia_job_timeout_seconds(kind)
    if kind is MultimediaGenerationKind.IMAGE:
        if labels:
            per_job_timeout = max(
                _multimedia_prompt_timeout_seconds(kind, label)
                for label in labels
            )
        wave_count = math.ceil(max(1, prompt_count) / _multimedia_parallelism(kind, prompt_count))
        return per_job_timeout * wave_count
    return per_job_timeout * max(1, prompt_count)


def _is_retryable_multimedia_provider_error(
    kind: MultimediaGenerationKind,
    error: Exception,
) -> bool:
    if kind is not MultimediaGenerationKind.IMAGE:
        return False
    lowered = f"{type(error).__module__}.{type(error).__name__}: {error}".casefold()
    return any(
        marker in lowered
        for marker in (
            "rate limit",
            "requests rate limit exceeded",
            "too many requests",
            "throttl",
            "resource_exhausted",
            "temporarily unavailable",
            "service unavailable",
            "provider overloaded",
            "upstream overloaded",
            "task query failed",
            "readtimeout",
            "read timeout",
            "timed out",
            "timeout",
            "remoteprotocolerror",
            "server disconnected",
            "connection reset",
            "connection aborted",
            "transport failed",
        )
    )


def _multimedia_provider_retry_backoff_seconds(attempt_index: int) -> float:
    return _MULTIMEDIA_IMAGE_PROVIDER_RETRY_BACKOFF_SECONDS * max(1, attempt_index)


def _first_failed_task(
    tasks: set[asyncio.Task[_MultimediaPromptExecution]],
) -> asyncio.Task[_MultimediaPromptExecution] | None:
    for task in tasks:
        if task.cancelled():
            continue
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            continue
        if exception is not None:
            return task
    return None


def _failed_multimedia_prompt_execution(
    *,
    kind: MultimediaGenerationKind,
    prompt_index: int,
    prompt_label: str | None,
    item_prompt: str,
    error: BaseException,
) -> _MultimediaPromptExecution:
    label = prompt_label or f"{kind.value}资产 {prompt_index + 1}"
    error_text = " ".join(str(error).split())[:500] or type(error).__name__
    review = RuntimeAssetVisualReview(
        passed=False,
        summary=f"{label} 生成失败，需单项重试。",
        issues=(error_text,),
        confidence=0.0,
    )
    item: dict[str, JsonValue] = {
        "kind": kind.value,
        "label": label,
        "title": label,
        "status": "failed",
        "generation_prompt": item_prompt,
        "visual_review": _visual_review_payload(review),
        "generation_error": error_text,
    }
    production_metadata = production_metadata_for_label(label, item_prompt)
    if production_metadata:
        item["production_metadata"] = production_metadata
    return _MultimediaPromptExecution(
        prompt_index=prompt_index,
        media_results=(item,),
        accepted_jobs=(),
        attempted_jobs=(),
        review_failed_results=(item,),
        first_file_metadata=None,
    )


def _multimedia_parallelism(kind: MultimediaGenerationKind, prompt_count: int) -> int:
    if prompt_count <= 1:
        return 1
    if kind is MultimediaGenerationKind.IMAGE:
        return min(9, prompt_count)
    if kind in {MultimediaGenerationKind.VIDEO, MultimediaGenerationKind.AUDIO}:
        return 1
    raise RuntimeCapabilityError("kind must be image, video, or audio")


def _multimedia_result_label(
    label: str | None,
    *,
    artifact_index: int,
    artifact_count: int,
) -> str | None:
    if label is None:
        return None
    if artifact_count <= 1:
        return label
    return f"{label} {artifact_index + 1}"


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
    label: str | None = None,
    generation_prompt: str | None = None,
    visual_review: RuntimeAssetVisualReview | None = None,
) -> Mapping[str, JsonValue]:
    production_metadata = (
        production_metadata_for_label(label, generation_prompt)
        if label is not None and generation_prompt is not None
        else {}
    )
    if file_metadata is not None:
        result: dict[str, JsonValue] = {
            "kind": artifact.kind.value,
            "uri": artifact.uri,
            "text": artifact.text,
            "logical_model": artifact.logical_model,
            "deployment_id": artifact.deployment_id,
            "artifact_id": file_metadata["artifact_id"],
            "storage_key": file_metadata["storage_key"],
            "filename": file_metadata["filename"],
            "mime_type": file_metadata["mime_type"],
            "size_bytes": file_metadata["size_bytes"],
            "sha256": file_metadata["sha256"],
            "download_url": file_metadata["download_url"],
            "expires_at": expires_at,
            "file": dict(file_metadata),
        }
        if label is not None:
            result["label"] = label
            result["title"] = label
        if generation_prompt is not None:
            result["generation_prompt"] = generation_prompt
        if production_metadata:
            result["production_metadata"] = production_metadata
        if visual_review is not None:
            result["visual_review"] = _visual_review_payload(visual_review)
        return result
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
    result = {
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
    if label is not None:
        result["label"] = label
        result["title"] = label
    if generation_prompt is not None:
        result["generation_prompt"] = generation_prompt
    if production_metadata:
        result["production_metadata"] = production_metadata
    if visual_review is not None:
        result["visual_review"] = _visual_review_payload(visual_review)
    return result


def _requires_visual_asset_review(label: str | None) -> bool:
    if label is None:
        return False
    normalized = label.casefold()
    return any(
        term in normalized
        for term in (
            "资产",
            "锁定",
            "设定",
            "角色",
            "场景",
            "道具",
            "动作",
            "特效",
            "分镜",
            "asset",
            "storyboard",
            "character",
            "scene",
            "prop",
            "effect",
        )
    )


def _visual_review_failure_reason(review: RuntimeAssetVisualReview) -> str:
    issues = "；".join(review.issues)
    return review.summary if not issues else f"{review.summary}：{issues}"


def _visual_review_unavailable(
    *,
    label: str | None,
    error: Exception,
) -> RuntimeAssetVisualReview:
    label_text = label.strip() if label else "图片资产"
    reason = str(error).strip() or type(error).__name__
    return RuntimeAssetVisualReview(
        passed=False,
        summary=f"{label_text} 视觉审核执行失败",
        issues=(reason[:1000],),
        confidence=0.0,
    )


def _visual_review_is_unavailable(review: RuntimeAssetVisualReview) -> bool:
    return review.confidence == 0.0 and "视觉审核执行失败" in review.summary


def _public_multimedia_artifact_url(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if re.match(r"^https?://", candidate, flags=re.IGNORECASE) is None:
        return None
    return candidate


def _visual_asset_retry_prompt(
    original_prompt: str,
    *,
    label: str | None,
    review: RuntimeAssetVisualReview,
    next_attempt: int,
) -> str:
    label_text = label.strip() if label else "图片资产"
    reason = _visual_review_failure_reason(review)
    guardrails = _visual_asset_retry_guardrails(label_text)
    return (
        f"{original_prompt}\n\n"
        f"视觉审核未通过，正在第 {next_attempt} 次重新生成同一项资产：{label_text}。\n"
        f"上一版问题：{reason}\n"
        f"{guardrails}\n"
        "请修正上述问题后重新生成合格资产图；不要输出电影剧照、宣传海报、随机写真、"
        "混合角色图片或与该资产类别无关的画面。图内只允许少量大号中文短标签；"
        "如果无法稳定生成可读文字，优先使用无文字图标、色块、箭头和结构化留白。"
    )


def _visual_asset_retry_guardrails(label: str) -> str:
    normalized = label.casefold()
    common = (
        "返修硬约束：纯白/浅灰/透明感纯色背景；只表达当前资产类别；"
        "禁止场景背景、海报构图、电影剧照、随机写真、测试水印、英文错字、伪字和密集小字。"
    )
    if "角色" in normalized or "character" in normalized or "定妆" in normalized:
        return (
            f"{common} 角色身份资产必须是单角色 Character Identity 参考表："
            "正脸、左右45度、侧脸、半身、全身、少量表情必须是同一张脸；"
            "服装/道具只能辅助身份，不能出现街景、室内、雨景、战斗场面或其他角色。"
        )
    if "服装" in normalized or "妆造" in normalized or "costume" in normalized:
        return (
            f"{common} 服装妆造只展示 Look/Costume：服装正反面、配饰、鞋履、材质、色卡；"
            "优先无头模特、衣架、平铺或局部特写，不能重新设计角色脸。"
        )
    if "道具" in normalized or "prop" in normalized:
        return (
            f"{common} 道具资产只展示独立物件、材质特写、比例尺和状态变化；"
            "证件不得使用随机真人头像，文字不可读时改用空白占位和清晰大标签。"
        )
    if "动作" in normalized or "pose" in normalized or "action" in normalized:
        return (
            f"{common} 动作资产用剪影/线稿/动作人偶表达关键帧、重心、方向箭头；"
            "不要可辨识陌生人脸，不要把动作板画成战斗海报。"
        )
    if "特效" in normalized or "effect" in normalized or "vfx" in normalized:
        return (
            f"{common} 特效资产只展示可复用光效层、粒子方向、强弱等级、透明叠加和触发点；"
            "不要人物剧照、街景或单张战斗画面。"
        )
    if "音频" in normalized or "字幕" in normalized or "subtitle" in normalized:
        return (
            f"{common} 音频字幕资产必须是 9:16 安全区、字幕断句、时间码、对白/旁白/BGM/SFX 轨道板；"
            "不得混入服装 Look 术语、播放器皮肤或角色写真。"
        )
    if "连续性" in normalized or "质检" in normalized or "qc" in normalized:
        return (
            f"{common} 连续性与质检资产必须是检查清单/流程图：Identity、Look、道具、场景、镜头、字幕、特效继承关系；"
            "不要用失败案例图片或复杂剧情剧照。"
        )
    if "表演" in normalized or "节奏" in normalized or "style" in normalized:
        return (
            f"{common} 表演节奏资产必须是导演节奏板：情绪曲线、节拍、停顿、表演强度、镜头节奏；"
            "不要普通人像写真或海报。"
        )
    return common


def _visual_review_payload(review: RuntimeAssetVisualReview) -> Mapping[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "passed": review.passed,
        "summary": review.summary,
        "issues": tuple(review.issues),
    }
    if review.confidence is not None:
        payload["confidence"] = review.confidence
    if review.logical_model is not None:
        payload["logical_model"] = review.logical_model
    if review.deployment_id is not None:
        payload["deployment_id"] = review.deployment_id
    return payload


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


def _content_project_payload(project: object) -> Mapping[str, JsonValue]:
    value = _jsonify_content_value(project)
    if not isinstance(value, Mapping):
        raise RuntimeCapabilityError("content_studio result is invalid")
    return value


async def _call_content_studio_service(method: object, **kwargs: object) -> object:
    if not callable(method):
        raise RuntimeCapabilityError("content_studio operation is unavailable")
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        filtered_kwargs = kwargs
    else:
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        filtered_kwargs = kwargs if accepts_var_kwargs else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
    result = method(**filtered_kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _content_studio_service_scope(
    service: object,
    *,
    tenant_id: UUID,
    owner_user_id: UUID,
) -> contextlib.AbstractContextManager[object]:
    store = getattr(service, "_store", None)
    scoped_to = getattr(store, "scoped_to", None)
    if callable(scoped_to):
        return cast(
            contextlib.AbstractContextManager[object],
            scoped_to(tenant_id=tenant_id, owner_user_id=owner_user_id),
        )
    return contextlib.nullcontext()


def _validate_content_project_ownership(
    project: object,
    *,
    tenant_id: UUID,
    owner_user_id: UUID,
) -> None:
    project_tenant_id = _content_project_identity_field(project, "tenant_id")
    project_owner_user_id = _content_project_identity_field(project, "owner_user_id")
    if project_tenant_id and project_tenant_id != str(tenant_id):
        raise RuntimeCapabilityError("content_studio project ownership mismatch")
    if project_owner_user_id and project_owner_user_id != str(owner_user_id):
        raise RuntimeCapabilityError("content_studio project ownership mismatch")


def _content_project_identity_field(project: object, field: str) -> str:
    if isinstance(project, Mapping):
        value = project.get(field)
    else:
        value = getattr(project, field, None)
    if value is None or value == "":
        return ""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        try:
            return str(UUID(value))
        except ValueError:
            return value
    raise RuntimeCapabilityError("content_studio project ownership mismatch")


def _jsonify_content_value(value: object) -> JsonValue:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, UUID):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            str(key): _jsonify_content_value(item)
            for key, item in asdict(value).items()
        }
    if value is None or type(value) in {bool, int, str}:
        return cast(JsonValue, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeCapabilityError("content_studio result is invalid")
        return value
    if isinstance(value, tuple | list | frozenset | set):
        return tuple(_jsonify_content_value(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _jsonify_content_value(item) for key, item in value.items()}
    raise RuntimeCapabilityError("content_studio result is invalid")


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
