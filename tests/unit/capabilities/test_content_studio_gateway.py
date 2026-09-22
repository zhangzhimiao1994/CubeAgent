# mypy: disable-error-code="index, call-overload, union-attr"

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from agent_hub.capabilities.runtime import RuntimeCapabilityGateway
from agent_hub.content_studio import (
    AsyncContentStudioService,
    AsyncInMemoryContentProjectStore,
    PackRegistry,
)

TENANT_ID = UUID("66666666-6666-4666-8666-666666666666")
OTHER_TENANT_ID = UUID("66666666-6666-4666-8666-999999999999")
OWNER_USER_ID = UUID("88888888-8888-4888-8888-888888888888")
OTHER_OWNER_USER_ID = UUID("99999999-9999-4999-8999-999999999999")
RUN_ID = UUID("77777777-7777-4777-8777-777777777777")


class RecordingContentStudioService:
    def __init__(self) -> None:
        self.projects: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def create_content_project(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("create_content_project", dict(kwargs)))
        project = {
            "project_id": "project-owned",
            "status": "DRAFT",
            "tenant_id": kwargs["tenant_id"],
            "owner_user_id": kwargs["owner_user_id"],
            "execution_mode": kwargs["execution_mode"],
            "packs": {"domain": {"version": "1.0.0"}},
        }
        self.projects["project-owned"] = project
        return project

    async def get_content_project(self, project_id: str) -> dict[str, object]:
        self.calls.append(("get_content_project", {"project_id": project_id}))
        return self.projects[project_id]

    async def run_content_project(self, project_id: str, *, until: object) -> dict[str, object]:
        self.calls.append(("run_content_project", {"project_id": project_id, "until": until}))
        project = dict(self.projects[project_id])
        project["status"] = str(getattr(until, "value", until))
        self.projects[project_id] = project
        return project

    async def approve_rights(
        self,
        project_id: str,
        *,
        asset_ids: tuple[str, ...],
        note: str,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "approve_rights",
                {"project_id": project_id, "asset_ids": asset_ids, "note": note},
            )
        )
        project = dict(self.projects[project_id])
        project["rights_approved"] = True
        self.projects[project_id] = project
        return project

    async def regenerate_voice(self, project_id: str, *, instruction: str) -> dict[str, object]:
        self.calls.append(
            ("regenerate_voice", {"project_id": project_id, "instruction": instruction})
        )
        project = dict(self.projects[project_id])
        project["voice_track"] = {"audio_artifact_id": "voice-regenerated"}
        self.projects[project_id] = project
        return project


async def test_content_studio_gateway_rejects_without_runtime_owner_principal(
    tmp_path: Path,
) -> None:
    gateway = RuntimeCapabilityGateway(skill_store_dir=tmp_path / "skills")

    assert gateway.is_available(TENANT_ID, "content_studio") is True
    assert gateway.is_replay_safe("content_studio") is False
    with pytest.raises(RuntimeError, match="runtime owner principal is not configured"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="main_agent",
            name="content_studio",
            arguments={
                "operation": "create_content_project",
                "title": "AIGC explainer",
                "topic": "Explain the official AIGC launch",
            },
            idempotency_key="content_create_without_owner",
        )


async def test_content_studio_capability_creates_runs_and_revises_project(tmp_path: Path) -> None:
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        content_studio_execution_mode="demo",
        content_studio_owner_user_id=OWNER_USER_ID,
    )

    assert gateway.is_available(TENANT_ID, "content_studio") is True
    assert gateway.is_replay_safe("content_studio") is False

    created = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="main_agent",
        name="content_studio",
        arguments={
            "operation": "create_content_project",
            "title": "AIGC explainer",
            "topic": "Explain the official AIGC launch",
            "source_urls": ("https://example.com/official-launch",),
            "domain": "aigc",
            "format": "explainer",
            "platform": "douyin",
            "channel": "ai_frontier",
            "style": "fast_minimal",
        },
        idempotency_key="content_create",
    )

    assert created["status"] == "DRAFT"
    assert created["packs"]["domain"]["version"] == "1.0.0"
    project_id = created["project_id"]
    assert isinstance(project_id, str)

    blocked_at_script = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="main_agent",
        name="content_studio",
        arguments={
            "operation": "run_content_project",
            "project_id": project_id,
            "until": "QC_REVIEW",
        },
        idempotency_key="content_run",
    )

    assert blocked_at_script["status"] == "SCRIPT_READY"
    assert blocked_at_script["script"]["hooks"]
    assert blocked_at_script["storyboard"] is None

    with pytest.raises(RuntimeError, match="trusted workspace permission context"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="main_agent",
            name="content_studio",
            arguments={
                "operation": "approve_script",
                "project_id": project_id,
            },
            idempotency_key="content_script_approval",
        )

    revised = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="main_agent",
        name="content_studio",
        arguments={
            "operation": "revise_script",
            "project_id": project_id,
            "instruction": "换一个更强但不夸张的开头",
        },
        idempotency_key="content_revise",
    )

    assert revised["status"] == "SCRIPT_READY"
    assert revised["script"]["hooks"][0].startswith("更强但不夸张")
    assert revised["storyboard"] is None


async def test_content_studio_capability_resumes_project_from_shared_store(tmp_path: Path) -> None:
    store = AsyncInMemoryContentProjectStore()
    first_gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills-1",
        content_studio_service=AsyncContentStudioService(
            registry=PackRegistry.mvp(),
            store=store,
            execution_mode="demo",
        ),
        content_studio_execution_mode="demo",
        content_studio_owner_user_id=OWNER_USER_ID,
    )
    created = await first_gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="main_agent",
        name="content_studio",
        arguments={
            "operation": "create_content_project",
            "title": "AIGC explainer",
            "topic": "Explain the official AIGC launch",
            "source_urls": ("https://example.com/official-launch",),
        },
        idempotency_key="content_create",
    )
    await first_gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="main_agent",
        name="content_studio",
        arguments={
            "operation": "run_content_project",
            "project_id": created["project_id"],
            "until": "SCRIPT_READY",
        },
        idempotency_key="content_script",
    )
    second_gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills-2",
        content_studio_service=AsyncContentStudioService(
            registry=PackRegistry.mvp(),
            store=store,
            execution_mode="demo",
        ),
        content_studio_execution_mode="demo",
        content_studio_owner_user_id=OWNER_USER_ID,
    )
    resumed = await second_gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="main_agent",
        name="content_studio",
        arguments={
            "operation": "get_content_project",
            "project_id": created["project_id"],
        },
        idempotency_key="content_qc",
    )

    assert resumed["status"] == "SCRIPT_READY"
    assert resumed["script"]["hooks"]


async def test_content_studio_gateway_uses_runtime_identity_not_tool_arguments(
    tmp_path: Path,
) -> None:
    service = RecordingContentStudioService()
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        content_studio_service=service,  # type: ignore[arg-type]
        content_studio_owner_user_id=OWNER_USER_ID,
        content_studio_execution_mode="demo",
    )

    created = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="content_producer",
        name="content_studio",
        arguments={
            "operation": "create_content_project",
            "title": "AIGC explainer",
            "topic": "Explain the official AIGC launch",
            "tenant_id": str(OTHER_TENANT_ID),
            "owner_user_id": str(OTHER_OWNER_USER_ID),
            "execution_mode": "production",
        },
        idempotency_key="content_create_identity",
    )

    assert created["tenant_id"] == str(TENANT_ID)
    assert created["owner_user_id"] == str(OWNER_USER_ID)
    assert created["execution_mode"] == "demo"
    create_call = service.calls[0][1]
    assert create_call["tenant_id"] == str(TENANT_ID)
    assert create_call["owner_user_id"] == str(OWNER_USER_ID)
    assert create_call["execution_mode"] == "demo"


async def test_content_studio_gateway_rejects_cross_owner_existing_project(
    tmp_path: Path,
) -> None:
    service = RecordingContentStudioService()
    service.projects["project-other-owner"] = {
        "project_id": "project-other-owner",
        "status": "DRAFT",
        "tenant_id": TENANT_ID,
        "owner_user_id": OTHER_OWNER_USER_ID,
    }
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        content_studio_service=service,  # type: ignore[arg-type]
        content_studio_owner_user_id=OWNER_USER_ID,
    )

    with pytest.raises(RuntimeError, match="content_studio project ownership mismatch"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="content_producer",
            name="content_studio",
            arguments={
                "operation": "run_content_project",
                "project_id": "project-other-owner",
                "until": "QC_REVIEW",
            },
            idempotency_key="content_cross_owner",
        )


async def test_content_studio_gateway_creates_production_project_and_blocks_at_run_without_provider(
    tmp_path: Path,
) -> None:
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        content_studio_execution_mode="production",
        content_studio_owner_user_id=OWNER_USER_ID,
    )

    created = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="content_producer",
        name="content_studio",
        arguments={
            "operation": "create_content_project",
            "title": "Production video",
            "topic": "Explain official AIGC release notes",
        },
        idempotency_key="content_prod_create",
    )
    assert created["execution_mode"] == "production"
    assert created["tenant_id"] == str(TENANT_ID)
    assert created["owner_user_id"] == str(OWNER_USER_ID)

    blocked = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="content_producer",
        name="content_studio",
        arguments={
            "operation": "run_content_project",
            "project_id": created["project_id"],
            "until": "SCRIPT_READY",
        },
        idempotency_key="content_prod_run",
    )

    assert blocked["status"] == "FAILED_BLOCKED"
    assert blocked["error_code"] == "provider_not_configured"
    assert blocked["research_bundle"] is None
    assert blocked["script"] is None


async def test_content_studio_gateway_routes_rights_and_voice_operations(
    tmp_path: Path,
) -> None:
    service = RecordingContentStudioService()
    service.projects["project-owned"] = {
        "project_id": "project-owned",
        "status": "ASSETS_READY",
        "tenant_id": TENANT_ID,
        "owner_user_id": OWNER_USER_ID,
    }
    gateway = RuntimeCapabilityGateway(
        skill_store_dir=tmp_path / "skills",
        content_studio_service=service,  # type: ignore[arg-type]
        content_studio_owner_user_id=OWNER_USER_ID,
    )

    with pytest.raises(RuntimeError, match="trusted workspace permission context"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="content_producer",
            name="content_studio",
            arguments={
                "operation": "approve_rights",
                "project_id": "project-owned",
                "asset_ids": ("ASSET001",),
                "note": "looks fine",
            },
            idempotency_key="content_rights_no_approval",
        )

    with pytest.raises(RuntimeError, match="trusted workspace permission context"):
        await gateway.execute(
            tenant_id=TENANT_ID,
            run_id=RUN_ID,
            actor="content_producer",
            name="content_studio",
            arguments={
                "operation": "approve_rights",
                "project_id": "project-owned",
                "asset_ids": ("ASSET001",),
                "note": "human reviewed copyright",
                "user_approved": True,
            },
            idempotency_key="content_rights_approved",
        )
    voiced = await gateway.execute(
        tenant_id=TENANT_ID,
        run_id=RUN_ID,
        actor="content_producer",
        name="content_studio",
        arguments={
            "operation": "regenerate_voice",
            "project_id": "project-owned",
            "instruction": "只重新生成配音，不要改画面",
        },
        idempotency_key="content_voice",
    )

    assert voiced["voice_track"]["audio_artifact_id"] == "voice-regenerated"
    approve_calls = [call for call in service.calls if call[0] == "approve_rights"]
    voice_calls = [call for call in service.calls if call[0] == "regenerate_voice"]
    assert approve_calls == []
    assert voice_calls[-1] == (
        "regenerate_voice",
        {"project_id": "project-owned", "instruction": "只重新生成配音，不要改画面"},
    )
