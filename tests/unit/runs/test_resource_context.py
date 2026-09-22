from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

from agent_hub.runs.resource_context import ResourceContextArtifactLoader

TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")


@dataclass(frozen=True, slots=True)
class CatalogSkill:
    id: str
    name: str
    status: str
    current_version_id: str | None = None


def skill_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "cross-system-hub/SKILL.md",
            "---\nname: cross-system-hub\ndescription: 跨系统协作\n---\n\n"
            "# Cross System Hub\n\n读取跨系统上下文并给出执行策略。\n",
        )
        archive.writestr(
            "cross-system-hub/references/rules.md",
            "内部规则：先确认权限，再读取被明确引用的资源。",
        )
        archive.writestr("cross-system-hub/assets/avatar.png", b"\x89PNG\r\n")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_resource_context_loads_requested_skill_instruction_and_internal_text(
    tmp_path: Path,
) -> None:
    store = tmp_path / "skills"
    tenant_dir = store / str(TENANT_ID)
    tenant_dir.mkdir(parents=True)
    (tenant_dir / "skill_cross_system_hub.zip").write_bytes(skill_archive())
    loader = ResourceContextArtifactLoader(skill_store_dir=store)

    artifacts = await loader(
        tenant_id=TENANT_ID,
        routing_decision={"requested_skills": "cross-system-hub"},
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.producer == "requested_resource_context"
    text = str(artifact.content["text"])
    assert "REQUESTED_RESOURCE_CONTEXT" in text
    assert "cross-system-hub/SKILL.md" in text
    assert "读取跨系统上下文并给出执行策略" in text
    assert "cross-system-hub/references/rules.md" in text
    assert "先确认权限" in text
    assert "assets/avatar.png" in text


@pytest.mark.asyncio
async def test_resource_context_reads_explicit_local_files_only_from_allowed_roots(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "handoff.md").write_text("当前计划：继续增强交互。", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    loader = ResourceContextArtifactLoader(
        skill_store_dir=tmp_path / "skills",
        workspace_roots=(workspace,),
    )

    artifacts = await loader(
        tenant_id=TENANT_ID,
        routing_decision={
            "requested_files": ["handoff.md", "../outside.md", str(outside)],
        },
    )

    text = "\n".join(str(artifact.content["text"]) for artifact in artifacts)
    assert "当前计划：继续增强交互" in text
    assert "requested local file was not readable" in text
    assert "secret" not in text


@pytest.mark.asyncio
async def test_resource_context_does_not_load_unapproved_skill_when_catalog_is_configured(
    tmp_path: Path,
) -> None:
    store = tmp_path / "skills"
    tenant_dir = store / str(TENANT_ID)
    tenant_dir.mkdir(parents=True)
    (tenant_dir / "skill_cross_system_hub.zip").write_bytes(skill_archive())

    async def list_skills() -> tuple[CatalogSkill, ...]:
        return (CatalogSkill(id="skill_cross_system_hub", name="cross-system-hub", status="scanned"),)

    loader = ResourceContextArtifactLoader(skill_store_dir=store, list_skills=list_skills)

    artifacts = await loader(
        tenant_id=TENANT_ID,
        routing_decision={"requested_skills": "cross-system-hub"},
    )

    text = str(artifacts[0].content["text"])
    assert "not enabled or readable" in text
    assert "读取跨系统上下文并给出执行策略" not in text
