from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_hub.cognitive.repository import InMemoryExperienceRepository
from agent_hub.cognitive.service import CognitiveStateService, ExperienceService
from agent_hub.cognitive.types import (
    CognitiveEvidence,
    CognitiveMemoryScope,
    ExperienceKind,
    ExperienceStatus,
    SkillCandidateRecord,
)


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


@pytest.mark.asyncio
async def test_user_scoped_experience_is_only_listed_for_owner() -> None:
    repository = InMemoryExperienceRepository()
    service = ExperienceService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    owner_user_id = uuid4()
    other_user_id = uuid4()
    record = await service.create_candidate(
        tenant_id=tenant_id,
        user_id=owner_user_id,
        kind=ExperienceKind.USER_PREFERENCE,
        summary="用户偏好状态报告先给结论。",
        lesson="该用户明确纠正过冗长过程输出。",
        strategy="对该用户先输出结论、证据和下一步。",
        evidence=(CognitiveEvidence(source_type="feedback", source_id="msg-1", note="explicit correction"),),
        memory_scope=CognitiveMemoryScope.USER,
    )
    await service.confirm(record.id, tenant_id=tenant_id, user_id=owner_user_id)

    owner_records = await service.list_records(tenant_id=tenant_id, user_id=owner_user_id)
    other_records = await service.list_records(tenant_id=tenant_id, user_id=other_user_id)

    assert [item.id for item in owner_records] == [record.id]
    assert other_records == ()


@pytest.mark.asyncio
async def test_root_scoped_experience_is_listed_for_other_users_in_same_tenant() -> None:
    repository = InMemoryExperienceRepository()
    service = ExperienceService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    owner_user_id = uuid4()
    other_user_id = uuid4()
    record = await service.create_candidate(
        tenant_id=tenant_id,
        user_id=owner_user_id,
        kind=ExperienceKind.ERROR_HANDLING,
        summary="模型空输出时切换备用路径。",
        lesson="同一租户内的模型空输出是通用运行经验。",
        strategy="先重试同模型；仍为空时切换同能力备用模型。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="empty response"),),
        memory_scope=CognitiveMemoryScope.ROOT,
    )
    await service.confirm(record.id, tenant_id=tenant_id, user_id=owner_user_id)

    other_records = await service.list_records(tenant_id=tenant_id, user_id=other_user_id)

    assert [item.id for item in other_records] == [record.id]


@pytest.mark.asyncio
async def test_belief_observations_accumulate_evidence_and_contradictions() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    repository = InMemoryCognitiveRecordRepository()
    service = CognitiveStateService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()

    first = await service.record_belief_observation(
        tenant_id=tenant_id,
        user_id=user_id,
        subject="user.workflow_preference",
        claim="用户偏好默认使用子 agent。",
        evidence=CognitiveEvidence(source_type="feedback", source_id="msg-1", note="explicit preference"),
        supported=True,
    )
    reinforced = await service.record_belief_observation(
        tenant_id=tenant_id,
        user_id=user_id,
        subject="user.workflow_preference",
        claim="用户偏好默认使用子 agent。",
        evidence=CognitiveEvidence(source_type="feedback", source_id="msg-2", note="same preference repeated"),
        supported=True,
    )
    contradicted = await service.record_belief_observation(
        tenant_id=tenant_id,
        user_id=user_id,
        subject="user.workflow_preference",
        claim="用户偏好默认使用子 agent。",
        evidence=CognitiveEvidence(source_type="feedback", source_id="msg-3", note="requested inline work"),
        supported=False,
    )

    assert reinforced.id == first.id
    assert reinforced.version == first.version + 1
    assert reinforced.confidence > first.confidence
    assert contradicted.id == first.id
    assert len(contradicted.evidence) == 2
    assert len(contradicted.contradictions) == 1
    assert contradicted.confidence < reinforced.confidence
    assert contradicted.last_verified_at is not None


@pytest.mark.asyncio
async def test_skill_candidate_use_outcomes_update_counts_and_confidence() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    now = datetime.now(UTC)
    repository = InMemoryCognitiveRecordRepository()
    service = CognitiveStateService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()
    skill = SkillCandidateRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=user_id,
        name="reviewer-timeout-recovery",
        purpose="处理 reviewer 超时。",
        steps=("压缩输入", "拆分审查", "重试"),
        output_contract="输出修复结果。",
        confidence=0.62,
        evidence=(CognitiveEvidence(source_type="experience", source_id="exp-1", note="candidate created"),),
        status="candidate",
        created_at=now,
        updated_at=now,
    )
    await repository.upsert(skill)

    succeeded = await service.record_skill_use_outcome(
        skill.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=True,
        evidence=CognitiveEvidence(source_type="run", source_id="run-1", note="worked"),
    )
    failed = await service.record_skill_use_outcome(
        skill.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=False,
        evidence=CognitiveEvidence(source_type="run", source_id="run-2", note="did not work"),
    )

    assert succeeded.use_count == 1
    assert succeeded.success_count == 1
    assert succeeded.confidence > skill.confidence
    assert failed.use_count == 2
    assert failed.success_count == 1
    assert failed.failure_count == 1
    assert len(failed.contradictions) == 1
    assert failed.confidence < succeeded.confidence
