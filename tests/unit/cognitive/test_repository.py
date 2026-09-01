from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_hub.cognitive import InMemoryCognitiveRecordRepository
from agent_hub.cognitive.types import (
    BeliefRecord,
    CognitiveEvidence,
    CognitiveMemoryScope,
    RelationshipStateRecord,
    SkillCandidateRecord,
    StrategyRecord,
    StrategyStatus,
)


@pytest.mark.asyncio
async def test_cognitive_record_repository_scopes_user_and_root_beliefs() -> None:
    now = datetime.now(UTC)
    tenant_id = uuid4()
    owner_user_id = uuid4()
    other_user_id = uuid4()
    repository = InMemoryCognitiveRecordRepository()
    owner_belief = BeliefRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=owner_user_id,
        memory_scope=CognitiveMemoryScope.USER,
        subject="user.confirmation_style",
        claim="用户偏好不要反复确认。",
        confidence=0.76,
        evidence=(CognitiveEvidence(source_type="feedback", source_id="msg-1", note="explicit preference"),),
        contradictions=(),
        status="active",
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )
    root_belief = BeliefRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=owner_user_id,
        memory_scope=CognitiveMemoryScope.ROOT,
        subject="project.boundary",
        claim="CubeAgent 仓库只做纯对话 Agent。",
        confidence=0.9,
        evidence=(CognitiveEvidence(source_type="handoff", source_id="HANDOFF", note="project boundary"),),
        contradictions=(),
        status="active",
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )

    await repository.upsert(owner_belief)
    await repository.upsert(root_belief)

    owner_records = await repository.list_for_user(BeliefRecord, tenant_id=tenant_id, user_id=owner_user_id)
    other_records = await repository.list_for_user(BeliefRecord, tenant_id=tenant_id, user_id=other_user_id)

    assert [item.id for item in owner_records] == [owner_belief.id, root_belief.id]
    assert [item.id for item in other_records] == [root_belief.id]


@pytest.mark.asyncio
async def test_cognitive_record_repository_keeps_record_types_separate() -> None:
    now = datetime.now(UTC)
    tenant_id = uuid4()
    user_id = uuid4()
    repository = InMemoryCognitiveRecordRepository()
    relationship = RelationshipStateRecord(
        id=f"relationship:{user_id}",
        tenant_id=tenant_id,
        user_id=user_id,
        memory_scope=CognitiveMemoryScope.USER,
        familiarity=0.4,
        preferred_language="zh-CN",
        preferred_confirmation_style="minimal",
        shared_milestones=("完成 Hermes+ 记忆注入",),
        recent_friction_points=(),
        last_interaction_at=now,
        created_at=now,
        updated_at=now,
    )
    skill = SkillCandidateRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=user_id,
        memory_scope=CognitiveMemoryScope.USER,
        name="memory-injection-smoke",
        purpose="验证经验注入是否影响未来运行。",
        steps=("创建经验", "确认经验", "触发相似任务", "检查 prompt 注入"),
        required_inputs=("tenant_id", "user_id"),
        output_contract="返回注入证据和结果。",
        confidence=0.7,
        evidence=(CognitiveEvidence(source_type="experience", source_id="exp-1", note="manual smoke"),),
        contradictions=(),
        status="candidate",
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )

    await repository.upsert(relationship)
    await repository.upsert(skill)

    relationships = await repository.list_for_user(
        RelationshipStateRecord, tenant_id=tenant_id, user_id=user_id
    )
    skills = await repository.list_for_user(SkillCandidateRecord, tenant_id=tenant_id, user_id=user_id)

    assert relationships == (relationship,)
    assert skills == (skill,)


@pytest.mark.asyncio
async def test_cognitive_record_repository_scopes_user_and_root_strategies() -> None:
    now = datetime.now(UTC)
    tenant_id = uuid4()
    owner_user_id = uuid4()
    other_user_id = uuid4()
    other_tenant_id = uuid4()
    repository = InMemoryCognitiveRecordRepository()
    owner_strategy = StrategyRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=owner_user_id,
        memory_scope=CognitiveMemoryScope.USER,
        name="private-status-summary",
        context="该用户询问项目状态。",
        strategy="先给结论，再列验证证据。",
        rationale="该用户纠正过冗长过程输出。",
        status=StrategyStatus.ACTIVE,
        confidence=0.82,
        evidence=(CognitiveEvidence(source_type="feedback", source_id="msg-1", note="explicit preference"),),
        contradictions=(),
        tags=("status", "summary"),
        applies_to_modes=("direct",),
        created_at=now,
        updated_at=now,
    )
    root_strategy = StrategyRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=owner_user_id,
        memory_scope=CognitiveMemoryScope.ROOT,
        name="repository-boundary",
        context="任务涉及 CubeAgent 仓库范围。",
        strategy="只做纯对话 Agent 改动，不做 harness 或部署。",
        rationale="仓库边界是 root 级约束。",
        status=StrategyStatus.ACTIVE,
        confidence=0.9,
        evidence=(CognitiveEvidence(source_type="handoff", source_id="handoff-1", note="project boundary"),),
        contradictions=(),
        tags=("cubeagent", "boundary"),
        applies_to_modes=("hybrid", "dispatch"),
        created_at=now,
        updated_at=now,
    )

    await repository.upsert(owner_strategy)
    await repository.upsert(root_strategy)

    owner_records = await repository.list_for_user(
        StrategyRecord, tenant_id=tenant_id, user_id=owner_user_id
    )
    other_user_records = await repository.list_for_user(
        StrategyRecord, tenant_id=tenant_id, user_id=other_user_id
    )
    other_tenant_records = await repository.list_for_user(
        StrategyRecord, tenant_id=other_tenant_id, user_id=other_user_id
    )

    assert [item.id for item in owner_records] == [owner_strategy.id, root_strategy.id]
    assert [item.id for item in other_user_records] == [root_strategy.id]
    assert other_tenant_records == ()
