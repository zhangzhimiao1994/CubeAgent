from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent_hub.content_studio import (
    ContentProject,
    ContentProjectConflict,
    ContentProjectSummary,
    content_project_from_payload,
    content_project_summary_from_payload,
    content_project_to_payload,
)
from agent_hub.db.models import AdminResourceRow

_CONTENT_STUDIO_KIND = "workflow"
_CONTENT_STUDIO_PROJECT_PREFIX = "content_studio_project:"
_CONTENT_STUDIO_SCHEMA = "content_studio.project.v1"
_scope_tenant_id: ContextVar[UUID | None] = ContextVar("content_studio_tenant_id", default=None)
_scope_owner_user_id: ContextVar[UUID | None] = ContextVar(
    "content_studio_owner_user_id", default=None
)


class PersistentContentProjectStore:
    """Persist Content Studio project snapshots in the existing resource table."""

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], *, tenant_id: UUID
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id

    @contextmanager
    def scoped_to(
        self,
        *,
        tenant_id: UUID,
        owner_user_id: UUID,
    ) -> Iterator[None]:
        tenant_token = _scope_tenant_id.set(tenant_id)
        owner_token = _scope_owner_user_id.set(owner_user_id)
        try:
            yield
        finally:
            _scope_owner_user_id.reset(owner_token)
            _scope_tenant_id.reset(tenant_token)

    async def save(self, project: ContentProject) -> ContentProject:
        tenant_id, owner_user_id = self._scope()
        _assert_project_scope(project, tenant_id=tenant_id, owner_user_id=owner_user_id)
        resource_id = content_project_resource_id(project.project_id)
        async with self._session_factory() as session, session.begin():
            existing = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == tenant_id)
                    .where(AdminResourceRow.kind == _CONTENT_STUDIO_KIND)
                    .where(AdminResourceRow.resource_id == resource_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            current_payload = None if existing is None else dict(existing.payload)
            if current_payload is not None and not project_belongs_to_scope(
                current_payload,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
            ):
                raise KeyError(project.project_id)
            payload = scoped_content_project_payload(
                project,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                current_payload=current_payload,
            )
            saved_project = replace(project, revision=content_project_revision(payload))
            payload = {
                **payload,
                "resource_id": resource_id,
                "storage_kind": _CONTENT_STUDIO_KIND,
            }
            if existing is None:
                session.add(
                    AdminResourceRow(
                        id=uuid4(),
                        tenant_id=tenant_id,
                        kind=_CONTENT_STUDIO_KIND,
                        resource_id=resource_id,
                        payload=payload,
                    )
                )
            else:
                existing.payload = payload
            await session.flush()
        return saved_project

    async def get(self, project_id: str) -> ContentProject:
        tenant_id, owner_user_id = self._scope()
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == tenant_id)
                    .where(AdminResourceRow.kind == _CONTENT_STUDIO_KIND)
                    .where(AdminResourceRow.resource_id == content_project_resource_id(project_id))
                )
            ).scalar_one_or_none()
        if row is None:
            raise KeyError(project_id)
        payload = dict(row.payload)
        if not project_belongs_to_scope(
            payload,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        ):
            raise KeyError(project_id)
        return content_project_from_payload(payload)

    async def metadata_for(self, project_id: str) -> dict[str, object]:
        tenant_id, owner_user_id = self._scope()
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == tenant_id)
                    .where(AdminResourceRow.kind == _CONTENT_STUDIO_KIND)
                    .where(AdminResourceRow.resource_id == content_project_resource_id(project_id))
                )
            ).scalar_one_or_none()
        if row is None:
            raise KeyError(project_id)
        payload = dict(row.payload)
        if not project_belongs_to_scope(
            payload,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        ):
            raise KeyError(project_id)
        return {
            "tenant_id": payload["tenant_id"],
            "owner_user_id": payload["owner_user_id"],
            "revision": content_project_revision(payload),
        }

    async def list_recent(
        self,
        *,
        tenant_id: str = "",
        owner_user_id: str = "",
        limit: int = 50,
    ) -> tuple[ContentProjectSummary, ...]:
        scoped_tenant_id, scoped_owner_user_id = self._scope()
        limit = max(1, min(limit, 100))
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == scoped_tenant_id)
                    .where(AdminResourceRow.kind == _CONTENT_STUDIO_KIND)
                    .where(AdminResourceRow.resource_id.like(f"{_CONTENT_STUDIO_PROJECT_PREFIX}%"))
                    .order_by(AdminResourceRow.updated_at.desc(), AdminResourceRow.created_at.desc())
                )
            ).scalars().all()
        summaries: list[ContentProjectSummary] = []
        for row in rows:
            payload = dict(row.payload)
            if not project_belongs_to_scope(
                payload,
                tenant_id=scoped_tenant_id,
                owner_user_id=scoped_owner_user_id,
            ):
                continue
            updated_at = row.updated_at.isoformat() if row.updated_at else None
            summaries.append(content_project_summary_from_payload(payload, updated_at=updated_at))
            if len(summaries) >= limit:
                break
        return tuple(summaries)

    def _scope(self) -> tuple[UUID, UUID]:
        tenant_id = _scope_tenant_id.get() or self._tenant_id
        owner_user_id = _scope_owner_user_id.get()
        if owner_user_id is None:
            raise KeyError("content studio owner scope is required")
        return tenant_id, owner_user_id


def content_project_resource_id(project_id: str) -> str:
    return f"{_CONTENT_STUDIO_PROJECT_PREFIX}{project_id}"


def scoped_content_project_payload(
    project: ContentProject,
    *,
    tenant_id: UUID,
    owner_user_id: UUID,
    current_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _assert_project_scope(project, tenant_id=tenant_id, owner_user_id=owner_user_id)
    if current_payload is not None and not project_belongs_to_scope(
        current_payload,
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
    ):
        raise KeyError(project.project_id)
    current_revision = 0 if current_payload is None else content_project_revision(current_payload)
    if project.revision != current_revision:
        raise ContentProjectConflict("content project revision changed")
    revision = current_revision + 1
    saved_project = replace(
        project,
        tenant_id=str(tenant_id),
        owner_user_id=str(owner_user_id),
        revision=revision,
    )
    payload = content_project_to_payload(saved_project)
    payload["schema_version"] = _CONTENT_STUDIO_SCHEMA
    payload["tenant_id"] = str(tenant_id)
    payload["owner_user_id"] = str(owner_user_id)
    payload["revision"] = revision
    return payload


def content_project_scope(payload: Mapping[str, object]) -> tuple[str | None, str | None]:
    tenant_id = payload.get("tenant_id")
    owner_user_id = payload.get("owner_user_id")
    return (
        tenant_id if isinstance(tenant_id, str) and tenant_id else None,
        owner_user_id if isinstance(owner_user_id, str) and owner_user_id else None,
    )


def content_project_revision(payload: Mapping[str, object]) -> int:
    revision = payload.get("revision")
    if not isinstance(revision, int):
        raw_project = payload.get("project")
        if isinstance(raw_project, Mapping):
            revision = raw_project.get("revision")
    if not isinstance(revision, int) or revision < 1:
        return 0
    return revision


def project_belongs_to_scope(
    payload: Mapping[str, object],
    *,
    tenant_id: UUID,
    owner_user_id: UUID,
) -> bool:
    payload_tenant_id, payload_owner_user_id = content_project_scope(payload)
    if payload_tenant_id != str(tenant_id) or payload_owner_user_id != str(owner_user_id):
        return False
    raw_project = payload.get("project")
    if not isinstance(raw_project, Mapping):
        return False
    return raw_project.get("tenant_id") == str(tenant_id) and raw_project.get(
        "owner_user_id"
    ) == str(owner_user_id)


def _assert_project_scope(
    project: ContentProject,
    *,
    tenant_id: UUID,
    owner_user_id: UUID,
) -> None:
    if project.tenant_id != str(tenant_id):
        raise KeyError(project.project_id)
    if project.owner_user_id != str(owner_user_id):
        raise KeyError(project.project_id)


__all__ = [
    "ContentProjectConflict",
    "PersistentContentProjectStore",
    "content_project_resource_id",
    "content_project_revision",
    "content_project_scope",
    "project_belongs_to_scope",
    "scoped_content_project_payload",
]
