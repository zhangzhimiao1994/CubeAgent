from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_hub.cognitive.repository import InMemoryExperienceRepository
from agent_hub.cognitive.service import ExperienceService
from agent_hub.cognitive.types import CognitiveEvidence, ExperienceKind, ExperienceStatus


@pytest.mark.asyncio
async def test_create_candidate_experience_is_not_runtime_active() -> None:
    service = ExperienceService(InMemoryExperienceRepository(), now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()

    record = await service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.ERROR_HANDLING,
        summary="reviewer 超时时先压缩上下文再分块审查。",
        lesson="大输入会让 reviewer 步骤超时。",
        strategy="先压缩输入，再拆分审查。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="timeout"),),
        tags=("reviewer", "timeout"),
        applies_to_modes=("dispatch", "hybrid"),
        applies_to_agents=("quality_reviewer",),
    )

    assert record.status is ExperienceStatus.CANDIDATE
    assert record.active_for_runtime is False


@pytest.mark.asyncio
async def test_confirm_experience_makes_it_runtime_active() -> None:
    repository = InMemoryExperienceRepository()
    service = ExperienceService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()
    record = await service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.USER_PREFERENCE,
        summary="用户偏好先给结论。",
        lesson="用户多次要求先给结论。",
        strategy="回答先给结论，再给关键证据。",
        evidence=(CognitiveEvidence(source_type="feedback", source_id="fb-1", note="explicit confirmation"),),
    )

    confirmed = await service.confirm(record.id, tenant_id=tenant_id, user_id=user_id)

    assert confirmed.status is ExperienceStatus.CONFIRMED
    assert confirmed.active_for_runtime is True


@pytest.mark.asyncio
async def test_record_use_outcome_updates_counts_and_confidence() -> None:
    repository = InMemoryExperienceRepository()
    service = ExperienceService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()
    record = await service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.WORKFLOW_STRATEGY,
        summary="大任务先拆小块。",
        lesson="大输入容易超时。",
        strategy="先拆分，再分配给子 agent。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="large task"),),
    )
    confirmed = await service.confirm(record.id, tenant_id=tenant_id, user_id=user_id)

    updated = await service.record_use_outcome(
        confirmed.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=True,
        evidence=CognitiveEvidence(source_type="run", source_id="run-2", note="worked again"),
    )

    assert updated.use_count == 1
    assert updated.success_count == 1
    assert updated.failure_count == 0
    assert updated.confidence > confirmed.confidence
    assert updated.last_used_at is not None
    assert updated.last_verified_at is not None


@pytest.mark.asyncio
async def test_tenant_user_isolation_blocks_foreign_confirm() -> None:
    repository = InMemoryExperienceRepository()
    service = ExperienceService(repository, now=lambda: datetime.now(UTC))
    record = await service.create_candidate(
        tenant_id=uuid4(),
        user_id=uuid4(),
        kind=ExperienceKind.ERROR_HANDLING,
        summary="错误要先修复。",
        lesson="审查发现问题不能甩给用户。",
        strategy="先诊断和修复，再报告。",
        evidence=(CognitiveEvidence(source_type="feedback", source_id="msg-1", note="explicit correction"),),
    )

    with pytest.raises(PermissionError):
        await service.confirm(record.id, tenant_id=uuid4(), user_id=uuid4())
