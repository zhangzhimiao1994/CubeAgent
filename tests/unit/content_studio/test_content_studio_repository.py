# mypy: disable-error-code="index"

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from agent_hub.content_studio import (
    ContentProject,
    ContentStudioService,
    InMemoryContentProjectStore,
    PackRegistry,
    content_project_from_payload,
)
from agent_hub.content_studio.repository import (
    ContentProjectConflict,
    content_project_resource_id,
    content_project_revision,
    content_project_scope,
    project_belongs_to_scope,
    scoped_content_project_payload,
)


def test_content_project_resource_id_is_stable_and_namespaced() -> None:
    assert content_project_resource_id("project-123") == "content_studio_project:project-123"


def test_scoped_payload_preserves_owner_and_initial_revision_from_project_fields() -> None:
    tenant_id = uuid4()
    user_id = uuid4()
    project = _project(tenant_id=tenant_id, owner_user_id=user_id)

    payload = scoped_content_project_payload(project, tenant_id=tenant_id, owner_user_id=user_id)

    assert payload["tenant_id"] == str(tenant_id)
    assert payload["owner_user_id"] == str(user_id)
    assert payload["revision"] == 1
    assert content_project_scope(payload) == (str(tenant_id), str(user_id))
    assert content_project_revision(payload) == 1
    assert payload["project"]["tenant_id"] == str(tenant_id)
    assert payload["project"]["owner_user_id"] == str(user_id)
    assert payload["project"]["revision"] == 1


def test_scoped_payload_increments_revision_from_current_project_revision() -> None:
    tenant_id = uuid4()
    user_id = uuid4()
    project = _project(tenant_id=tenant_id, owner_user_id=user_id)
    current = scoped_content_project_payload(project, tenant_id=tenant_id, owner_user_id=user_id)
    stored = content_project_from_payload(current)

    updated = scoped_content_project_payload(
        replace(stored, title="Updated title"),
        tenant_id=tenant_id,
        owner_user_id=user_id,
        current_payload=current,
    )

    assert updated["revision"] == 2
    assert updated["project"]["title"] == "Updated title"


def test_scoped_payload_rejects_stale_revision() -> None:
    tenant_id = uuid4()
    user_id = uuid4()
    current = scoped_content_project_payload(
        _project(tenant_id=tenant_id, owner_user_id=user_id),
        tenant_id=tenant_id,
        owner_user_id=user_id,
    )

    with pytest.raises(ContentProjectConflict, match="content project revision changed"):
        scoped_content_project_payload(
            _project(tenant_id=tenant_id, owner_user_id=user_id),
            tenant_id=tenant_id,
            owner_user_id=user_id,
            current_payload=current,
        )


def test_project_scope_fails_closed_when_owner_is_missing() -> None:
    tenant_id = uuid4()
    user_id = uuid4()
    payload = scoped_content_project_payload(
        _project(tenant_id=tenant_id, owner_user_id=user_id),
        tenant_id=tenant_id,
        owner_user_id=user_id,
    )
    payload.pop("owner_user_id")

    assert project_belongs_to_scope(payload, tenant_id=tenant_id, owner_user_id=user_id) is False


def test_project_scope_rejects_cross_user_same_tenant_and_cross_tenant() -> None:
    tenant_id = uuid4()
    user_id = uuid4()
    payload = scoped_content_project_payload(
        _project(tenant_id=tenant_id, owner_user_id=user_id),
        tenant_id=tenant_id,
        owner_user_id=user_id,
    )

    assert project_belongs_to_scope(payload, tenant_id=tenant_id, owner_user_id=uuid4()) is False
    assert project_belongs_to_scope(payload, tenant_id=uuid4(), owner_user_id=user_id) is False


def test_scoped_payload_rejects_project_owner_relabeling() -> None:
    tenant_id = uuid4()
    user_id = uuid4()
    project = _project(tenant_id=tenant_id, owner_user_id=uuid4())

    with pytest.raises(KeyError):
        scoped_content_project_payload(project, tenant_id=tenant_id, owner_user_id=user_id)


def _project(*, tenant_id: object, owner_user_id: object) -> ContentProject:
    service = ContentStudioService(
        registry=PackRegistry.mvp(),
        store=InMemoryContentProjectStore(),
        execution_mode="demo",
    )
    return service.create_content_project(
        title="AIGC explainer",
        topic="Explain official AIGC launch notes",
        source_urls=("https://example.com/official-launch",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        tenant_id=str(tenant_id),
        owner_user_id=str(owner_user_id),
    )
