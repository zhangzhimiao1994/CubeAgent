from __future__ import annotations

import inspect
from collections.abc import Awaitable
from contextlib import nullcontext
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field, field_validator

from agent_hub.api.dependencies import require_permission
from agent_hub.api.errors import PublicAPIError, error_responses
from agent_hub.auth.models import AuthenticatedPrincipal
from agent_hub.content_studio import (
    AsyncContentStudioService,
    ClaimStatus,
    ContentProject,
    ProjectStatus,
    content_project_to_payload,
)
from agent_hub.content_studio.repository import ContentProjectConflict

router = APIRouter(
    prefix="/api/v1/content-studio",
    tags=["content-studio"],
    responses=error_responses(401, 405, 500, 503),
)


class ContentStudioServiceProtocol(Protocol):
    async def create_content_project(
        self,
        *,
        title: str,
        topic: str,
        source_urls: tuple[str, ...],
        domain: str,
        format: str,
        platform: str,
        channel: str,
        style: str,
        tenant_id: str = "",
        owner_user_id: str = "",
    ) -> ContentProject: ...

    async def get_content_project(self, project_id: str) -> ContentProject: ...

    async def run_content_project(
        self,
        project_id: str,
        *,
        until: ProjectStatus = ProjectStatus.QC_REVIEW,
    ) -> ContentProject: ...

    async def revise_script(self, project_id: str, *, instruction: str) -> ContentProject: ...

    async def approve_script(self, project_id: str) -> ContentProject: ...

    async def revise_storyboard(self, project_id: str, *, instruction: str) -> ContentProject: ...

    async def regenerate_asset(
        self,
        project_id: str,
        *,
        asset_id: str,
        instruction: str,
    ) -> ContentProject: ...

    async def render_preview(self, project_id: str) -> ContentProject: ...

    async def approve_rights(
        self,
        project_id: str,
        *,
        asset_ids: tuple[str, ...],
        note: str,
    ) -> ContentProject: ...

    async def regenerate_voice(
        self, project_id: str, *, instruction: str = ""
    ) -> ContentProject: ...

    async def approve_final(self, project_id: str) -> ContentProject: ...

    async def retry_stage(self, project_id: str, stage: ProjectStatus) -> ContentProject: ...

    async def replace_claim_status(
        self,
        project_id: str,
        *,
        claim_id: str,
        status: ClaimStatus,
        note: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> ContentProject: ...


class CreateContentProjectRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    topic: str = Field(min_length=1, max_length=2000)
    source_urls: tuple[str, ...] = ()
    domain: str = "aigc"
    format: str = "explainer"
    platform: str = "douyin"
    channel: str = "ai_frontier"
    style: str = "fast_minimal"

    @field_validator("title", "topic", "domain", "format", "platform", "channel", "style")
    @classmethod
    def _trim_nonblank(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed

    @field_validator("source_urls")
    @classmethod
    def _trim_source_urls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.strip() for item in value if item.strip())


class RunContentProjectRequest(BaseModel):
    until: ProjectStatus = ProjectStatus.QC_REVIEW


class InstructionRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=4000)

    @field_validator("instruction")
    @classmethod
    def _trim_instruction(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("instruction must not be blank")
        return trimmed


class RegenerateAssetRequest(InstructionRequest):
    asset_id: str = Field(min_length=1, max_length=128)


class RetryStageRequest(BaseModel):
    stage: ProjectStatus


class RevisionRequest(BaseModel):
    revision: int = Field(ge=1)


class ApproveRightsRequest(RevisionRequest):
    asset_ids: tuple[str, ...] = Field(min_length=1)
    note: str = Field(min_length=1, max_length=4000)

    @field_validator("asset_ids")
    @classmethod
    def _trim_asset_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        trimmed = tuple(item.strip() for item in value if item.strip())
        if not trimmed:
            raise ValueError("asset_ids must not be empty")
        return trimmed

    @field_validator("note")
    @classmethod
    def _trim_note(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("note must not be blank")
        return trimmed


class RegenerateVoiceRequest(InstructionRequest):
    revision: int = Field(ge=1)


class ReplaceClaimStatusRequest(BaseModel):
    status: ClaimStatus
    note: str = Field(min_length=1, max_length=4000)
    evidence_ids: tuple[str, ...] = ()

    @field_validator("note")
    @classmethod
    def _trim_note(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("note must not be blank")
        return trimmed

    @field_validator("evidence_ids")
    @classmethod
    def _trim_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.strip() for item in value if item.strip())


def _content_studio_service(request: Request) -> ContentStudioServiceProtocol:
    service = getattr(request.app.state, "content_studio_service", None)
    if service is None:
        raise PublicAPIError(503, "service_unavailable", "service unavailable")
    return cast(AsyncContentStudioService, service)


@router.post(
    "/projects",
    status_code=status.HTTP_201_CREATED,
    response_model=None,
    responses=error_responses(403, 413, 422),
)
async def create_content_project(
    body: CreateContentProjectRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    try:
        with _service_scope(service, principal):
            project = await _call_service(
                service.create_content_project,
                title=body.title,
                topic=body.topic,
                source_urls=body.source_urls,
                domain=body.domain,
                format=body.format,
                platform=body.platform,
                channel=body.channel,
                style=body.style,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
    except KeyError as error:
        raise PublicAPIError(
            422,
            "content_studio_pack_not_found",
            f"content studio pack was not found: {error}",
        ) from None
    except RuntimeError as error:
        raise PublicAPIError(
            409,
            "content_studio_provider_unavailable",
            str(error) or "content studio provider is not configured",
        ) from None
    return _project_payload(project)


@router.get(
    "/projects/{project_id}",
    response_model=None,
    responses=error_responses(403, 404),
)
async def get_content_project(
    project_id: str,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:read"))],
) -> dict[str, object]:
    project = await _scoped_project(service, principal, project_id)
    return _project_payload(project)


@router.post(
    "/projects/{project_id}/run",
    response_model=None,
    responses=error_responses(403, 404, 409, 413, 422),
)
async def run_content_project(
    project_id: str,
    body: RunContentProjectRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    if _requires_script_approval(body.until):
        project = await _scoped_project(service, principal, project_id)
        if _stage_before(project.status, ProjectStatus.SCRIPT_APPROVED):
            raise PublicAPIError(
                409,
                "content_project_conflict",
                "script must be approved before asset stages",
            )
    with _service_scope(service, principal):
        project = await _project_op(
            _call_service(
                service.run_content_project,
                project_id,
                until=body.until,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(project)


@router.post(
    "/projects/{project_id}/revise-script",
    response_model=None,
    responses=error_responses(403, 404, 409, 413, 422),
)
async def revise_script(
    project_id: str,
    body: InstructionRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    with _service_scope(service, principal):
        project = await _project_op(
            _call_service(
                service.revise_script,
                project_id,
                instruction=body.instruction,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(project)


@router.post("/projects/{project_id}/approve-script", response_model=None)
async def approve_script(
    project_id: str,
    body: RevisionRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    project = await _scoped_project(service, principal, project_id)
    if project.revision != body.revision:
        raise PublicAPIError(409, "content_project_conflict", "content project revision changed")
    with _service_scope(service, principal):
        result = await _project_op(
            _call_service(
                service.approve_script,
                project_id,
                revision=body.revision,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(result)


@router.post("/projects/{project_id}/revise-storyboard", response_model=None)
async def revise_storyboard(
    project_id: str,
    body: InstructionRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    with _service_scope(service, principal):
        project = await _project_op(
            _call_service(
                service.revise_storyboard,
                project_id,
                instruction=body.instruction,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(project)


@router.post("/projects/{project_id}/regenerate-asset", response_model=None)
async def regenerate_asset(
    project_id: str,
    body: RegenerateAssetRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    with _service_scope(service, principal):
        project = await _project_op(
            _call_service(
                service.regenerate_asset,
                project_id,
                asset_id=body.asset_id,
                instruction=body.instruction,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(project)


@router.post("/projects/{project_id}/render-preview", response_model=None)
async def render_preview(
    project_id: str,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    project = await _scoped_project(service, principal, project_id)
    if project.asset_manifest is not None and any(
        asset.rights_status not in {"approved", "cleared", "owned", "licensed"}
        for asset in project.asset_manifest.assets
    ):
        raise PublicAPIError(
            409,
            "content_project_conflict",
            "asset rights must be approved before preview",
        )
    with _service_scope(service, principal):
        project = await _project_op(
            _call_service(
                service.render_preview,
                project_id,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(project)


@router.post("/projects/{project_id}/approve-rights", response_model=None)
async def approve_rights(
    project_id: str,
    body: ApproveRightsRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    project = await _scoped_project(service, principal, project_id)
    if project.revision != body.revision:
        raise PublicAPIError(409, "content_project_conflict", "content project revision changed")
    with _service_scope(service, principal):
        result = await _project_op(
            _call_service(
                service.approve_rights,
                project_id,
                asset_ids=body.asset_ids,
                note=body.note,
                revision=body.revision,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(result)


@router.post("/projects/{project_id}/regenerate-voice", response_model=None)
async def regenerate_voice(
    project_id: str,
    body: RegenerateVoiceRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    project = await _scoped_project(service, principal, project_id)
    if project.revision != body.revision:
        raise PublicAPIError(409, "content_project_conflict", "content project revision changed")
    with _service_scope(service, principal):
        result = await _project_op(
            _call_service(
                service.regenerate_voice,
                project_id,
                instruction=body.instruction,
                revision=body.revision,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(result)


@router.post("/projects/{project_id}/approve-final", response_model=None)
async def approve_final(
    project_id: str,
    body: RevisionRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    project = await _scoped_project(service, principal, project_id)
    if project.revision != body.revision:
        raise PublicAPIError(409, "content_project_conflict", "content project revision changed")
    if (
        project.status is not ProjectStatus.QC_REVIEW
        or project.qc_report is None
        or project.qc_report.blockers
    ):
        raise PublicAPIError(
            409,
            "content_project_conflict",
            "clean QC review is required before final approval",
        )
    with _service_scope(service, principal):
        result = await _project_op(
            _call_service(
                service.approve_final,
                project_id,
                revision=body.revision,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(result)


@router.post("/projects/{project_id}/retry-stage", response_model=None)
async def retry_stage(
    project_id: str,
    body: RetryStageRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    with _service_scope(service, principal):
        project = await _project_op(
            _call_service(
                service.retry_stage,
                project_id,
                body.stage,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(project)


@router.post("/projects/{project_id}/claims/{claim_id}", response_model=None)
async def replace_claim_status(
    project_id: str,
    claim_id: str,
    body: ReplaceClaimStatusRequest,
    service: Annotated[ContentStudioServiceProtocol, Depends(_content_studio_service)],
    principal: Annotated[AuthenticatedPrincipal, Depends(require_permission("run:create"))],
) -> dict[str, object]:
    if body.status is ClaimStatus.SUPPORTED and not body.evidence_ids:
        raise PublicAPIError(
            409,
            "content_project_conflict",
            "supported claims require evidence",
        )
    project = await _scoped_project(service, principal, project_id)
    if body.evidence_ids:
        evidence_ids = _project_evidence_ids(project)
        missing = set(body.evidence_ids) - evidence_ids
        if missing:
            raise PublicAPIError(
                409,
                "content_project_conflict",
                f"evidence was not found: {min(missing)}",
            )
    with _service_scope(service, principal):
        result = await _project_op(
            _call_service(
                service.replace_claim_status,
                project_id,
                claim_id=claim_id,
                status=body.status,
                note=body.note,
                evidence_ids=body.evidence_ids,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )
    return _project_payload(result)


async def _project_op(awaitable: Awaitable[ContentProject]) -> ContentProject:
    try:
        return await awaitable
    except KeyError:
        raise PublicAPIError(
            404, "content_project_not_found", "content project was not found"
        ) from None
    except ContentProjectConflict as error:
        raise PublicAPIError(409, "content_project_conflict", str(error)) from None
    except ValueError as error:
        raise PublicAPIError(409, "content_project_conflict", str(error)) from None
    except RuntimeError as error:
        raise PublicAPIError(
            409,
            "content_studio_provider_unavailable",
            str(error) or "content studio provider is not configured",
        ) from None


def _project_payload(project: ContentProject) -> dict[str, object]:
    raw = content_project_to_payload(project)["project"]
    assert isinstance(raw, dict)
    return raw


async def _call_service(method: object, *args: object, **kwargs: object) -> Any:
    return await method(*args, **_supported_kwargs(method, kwargs))  # type: ignore[misc]


def _supported_kwargs(method: object, kwargs: dict[str, object]) -> dict[str, object]:
    signature = inspect.signature(method)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _service_scope(
    service: ContentStudioServiceProtocol,
    principal: AuthenticatedPrincipal,
) -> object:
    scoped_to = getattr(getattr(service, "_store", None), "scoped_to", None)
    if scoped_to is None:
        return nullcontext()
    return scoped_to(
        tenant_id=principal.tenant_id,
        owner_user_id=principal.user_id,
    )


async def _scoped_project(
    service: ContentStudioServiceProtocol,
    principal: AuthenticatedPrincipal,
    project_id: str,
) -> ContentProject:
    with _service_scope(service, principal):
        return await _project_op(
            _call_service(
                service.get_content_project,
                project_id,
                tenant_id=str(principal.tenant_id),
                owner_user_id=str(principal.user_id),
            )
        )


def _requires_script_approval(status: ProjectStatus) -> bool:
    return status in {
        ProjectStatus.STORYBOARD_READY,
        ProjectStatus.ASSETS_READY,
        ProjectStatus.VOICE_READY,
        ProjectStatus.TIMELINE_READY,
        ProjectStatus.PREVIEW_RENDERED,
        ProjectStatus.QC_REVIEW,
    }


def _stage_before(current: ProjectStatus, target: ProjectStatus) -> bool:
    order = (
        ProjectStatus.DRAFT,
        ProjectStatus.RESEARCHING,
        ProjectStatus.RESEARCH_READY,
        ProjectStatus.FACT_CHECKED,
        ProjectStatus.PLAN_READY,
        ProjectStatus.SCRIPT_READY,
        ProjectStatus.SCRIPT_APPROVED,
        ProjectStatus.STORYBOARD_READY,
        ProjectStatus.ASSETS_READY,
        ProjectStatus.VOICE_READY,
        ProjectStatus.TIMELINE_READY,
        ProjectStatus.PREVIEW_RENDERED,
        ProjectStatus.QC_REVIEW,
        ProjectStatus.FINAL_APPROVED,
        ProjectStatus.FINAL_RENDERED,
    )
    if current not in order or target not in order:
        return False
    return order.index(current) < order.index(target)


def _project_evidence_ids(project: ContentProject) -> set[str]:
    if project.evidence_graph is None:
        return set()
    return {evidence.evidence_id for evidence in project.evidence_graph.evidence}


__all__ = ["router"]
