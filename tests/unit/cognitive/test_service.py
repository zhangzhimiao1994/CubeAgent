from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_hub.cognitive.repository import InMemoryExperienceRepository
from agent_hub.cognitive.service import (
    CognitiveStateService,
    ExperienceService,
    SkillPromotionNotReady,
    SkillPromotionService,
)
from agent_hub.cognitive.types import (
    CognitiveEvidence,
    CognitiveMemoryScope,
    ExperienceKind,
    ExperienceStatus,
    OutcomeAssessmentRecord,
    OutcomeVerdict,
    SkillCandidateRecord,
    StrategyStatus,
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
async def test_repeated_failed_experience_use_deprecates_runtime_learning() -> None:
    repository = InMemoryExperienceRepository()
    service = ExperienceService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()
    record = await service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.ERROR_HANDLING,
        summary="弱证据策略不应长期污染运行时。",
        lesson="连续失败说明经验不可靠。",
        strategy="失败后继续使用同一策略。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="candidate"),),
        confidence=0.42,
    )
    confirmed = await service.confirm(record.id, tenant_id=tenant_id, user_id=user_id)

    first = await service.record_use_outcome(
        confirmed.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=False,
        evidence=CognitiveEvidence(source_type="run", source_id="run-2", note="failed"),
    )
    second = await service.record_use_outcome(
        first.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=False,
        evidence=CognitiveEvidence(source_type="run", source_id="run-3", note="failed again"),
    )
    third = await service.record_use_outcome(
        second.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=False,
        evidence=CognitiveEvidence(source_type="run", source_id="run-4", note="failed repeatedly"),
    )

    assert third.status is ExperienceStatus.DEPRECATED
    assert third.active_for_runtime is False


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


@pytest.mark.asyncio
async def test_relationship_state_merges_user_habits_and_shared_context() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    repository = InMemoryCognitiveRecordRepository()
    service = CognitiveStateService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()

    first = await service.update_relationship_state(
        tenant_id=tenant_id,
        user_id=user_id,
        preferred_language="zh-CN",
        preferred_confirmation_style="minimal",
        shared_milestones=("完成 Hermes+ 记忆隔离。",),
        recent_friction_points=("调度经验不应写入普通对话记忆。",),
        familiarity_delta=0.08,
    )
    second = await service.update_relationship_state(
        tenant_id=tenant_id,
        user_id=user_id,
        preferred_confirmation_style="no-confirmation-for-approved-scope",
        shared_milestones=("完成 Hermes+ 记忆隔离。", "完成认知状态服务。"),
        recent_friction_points=("调度经验不应写入普通对话记忆。",),
        familiarity_delta=0.08,
    )

    assert second.id == first.id
    assert second.familiarity > first.familiarity
    assert second.preferred_language == "zh-CN"
    assert second.preferred_confirmation_style == "no-confirmation-for-approved-scope"
    assert second.shared_milestones == ("完成 Hermes+ 记忆隔离。", "完成认知状态服务。")
    assert second.recent_friction_points == ("调度经验不应写入普通对话记忆。",)


@pytest.mark.asyncio
async def test_world_state_merges_facts_and_closes_open_items() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    repository = InMemoryCognitiveRecordRepository()
    service = CognitiveStateService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()

    first = await service.update_world_state(
        tenant_id=tenant_id,
        user_id=user_id,
        scope="cubeagent.project",
        facts=("CubeAgent 是纯对话 Agent 仓库。",),
        open_items=("实现统一 Memory/Experience Router。", "后续合并 Memory/Hermes。"),
        future_events=("harness 改造走独立项目。",),
        evidence=CognitiveEvidence(source_type="handoff", source_id="HANDOFF", note="project boundary"),
    )
    second = await service.update_world_state(
        tenant_id=tenant_id,
        user_id=user_id,
        scope="cubeagent.project",
        facts=("CubeAgent 是纯对话 Agent 仓库。", "生产只保留当前 release。"),
        open_items=("实现 Skill Candidate 晋升。",),
        completed_items=("实现统一 Memory/Experience Router。",),
        future_events=("harness 改造走独立项目。",),
        evidence=CognitiveEvidence(source_type="handoff", source_id="HANDOFF", note="state update"),
    )

    assert second.id == first.id
    assert second.facts == ("CubeAgent 是纯对话 Agent 仓库。", "生产只保留当前 release。")
    assert second.open_items == ("后续合并 Memory/Hermes。", "实现 Skill Candidate 晋升。")
    assert second.future_events == ("harness 改造走独立项目。",)
    assert len(second.evidence) == 2
    assert second.last_verified_at is not None


@pytest.mark.asyncio
async def test_promote_successful_experiences_to_skill_candidate() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    experience_repository = InMemoryExperienceRepository()
    cognitive_repository = InMemoryCognitiveRecordRepository()
    experience_service = ExperienceService(experience_repository, now=lambda: datetime.now(UTC))
    promotion_service = SkillPromotionService(
        experience_repository,
        cognitive_repository,
        now=lambda: datetime.now(UTC),
    )
    tenant_id = uuid4()
    user_id = uuid4()
    first = await experience_service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.WORKFLOW_STRATEGY,
        summary="大输入先压缩再拆分。",
        lesson="大输入直接审查容易超时。",
        strategy="先压缩输入，再拆成较小审查块。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="created"),),
        tags=("review", "timeout"),
        applies_to_modes=("hybrid",),
        applies_to_agents=("quality_reviewer",),
    )
    first = await experience_service.confirm(first.id, tenant_id=tenant_id, user_id=user_id)
    await experience_service.record_use_outcome(
        first.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=True,
        evidence=CognitiveEvidence(source_type="run", source_id="run-2", note="worked"),
    )
    second = await experience_service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.ERROR_HANDLING,
        summary="reviewer 超时后重试较小任务。",
        lesson="分块后 reviewer 更稳定。",
        strategy="如果 reviewer 超时，降低输入规模后重试。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-3", note="created"),),
        tags=("review", "timeout"),
        applies_to_modes=("hybrid",),
        applies_to_agents=("quality_reviewer",),
    )
    second = await experience_service.confirm(second.id, tenant_id=tenant_id, user_id=user_id)
    await experience_service.record_use_outcome(
        second.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=True,
        evidence=CognitiveEvidence(source_type="run", source_id="run-4", note="worked"),
    )

    skill = await promotion_service.promote_from_experiences(
        tenant_id=tenant_id,
        user_id=user_id,
        experience_ids=(first.id, second.id),
        name="reviewer-timeout-recovery",
        purpose="让 reviewer 超时后优先压缩、拆分、重试。",
        output_contract="输出审查结论、修复建议和无法审查的残余风险。",
        required_inputs=("审查对象", "失败原因"),
    )

    assert skill.name == "reviewer-timeout-recovery"
    assert skill.status == "candidate"
    assert skill.confidence >= 0.65
    assert skill.steps == ("先压缩输入，再拆成较小审查块。", "如果 reviewer 超时，降低输入规模后重试。")
    assert skill.required_inputs == ("审查对象", "失败原因")
    assert {item.source_id for item in skill.evidence} == {str(first.id), str(second.id)}


@pytest.mark.asyncio
async def test_skill_promotion_requires_enough_successful_experience() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    experience_repository = InMemoryExperienceRepository()
    cognitive_repository = InMemoryCognitiveRecordRepository()
    experience_service = ExperienceService(experience_repository, now=lambda: datetime.now(UTC))
    promotion_service = SkillPromotionService(
        experience_repository,
        cognitive_repository,
        now=lambda: datetime.now(UTC),
    )
    tenant_id = uuid4()
    user_id = uuid4()
    record = await experience_service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.WORKFLOW_STRATEGY,
        summary="单次经验不足以晋升技能。",
        lesson="只有一次成功，证据不足。",
        strategy="继续收集使用结果。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="created"),),
    )
    record = await experience_service.confirm(record.id, tenant_id=tenant_id, user_id=user_id)

    with pytest.raises(SkillPromotionNotReady):
        await promotion_service.promote_from_experiences(
            tenant_id=tenant_id,
            user_id=user_id,
            experience_ids=(record.id,),
            name="not-ready",
            purpose="不应晋升。",
            output_contract="不应创建 skill。",
        )


@pytest.mark.asyncio
async def test_skill_promotion_updates_existing_candidate_without_duplicate() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    experience_repository = InMemoryExperienceRepository()
    cognitive_repository = InMemoryCognitiveRecordRepository()
    experience_service = ExperienceService(experience_repository, now=lambda: datetime.now(UTC))
    promotion_service = SkillPromotionService(
        experience_repository,
        cognitive_repository,
        now=lambda: datetime.now(UTC),
    )
    tenant_id = uuid4()
    user_id = uuid4()
    first = await experience_service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.WORKFLOW_STRATEGY,
        summary="重复晋升应更新同名 skill。",
        lesson="同名 skill 不应重复。",
        strategy="先压缩输入。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="created"),),
        confidence=0.8,
    )
    first = await experience_service.confirm(first.id, tenant_id=tenant_id, user_id=user_id)
    await experience_service.record_use_outcome(
        first.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=True,
        evidence=CognitiveEvidence(source_type="run", source_id="run-2", note="worked"),
    )
    second = await experience_service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.WORKFLOW_STRATEGY,
        summary="重复晋升补充步骤。",
        lesson="新经验可补充旧 skill。",
        strategy="再拆分审查块。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-3", note="created"),),
        confidence=0.8,
    )
    second = await experience_service.confirm(second.id, tenant_id=tenant_id, user_id=user_id)
    await experience_service.record_use_outcome(
        second.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=True,
        evidence=CognitiveEvidence(source_type="run", source_id="run-4", note="worked"),
    )

    original = await promotion_service.promote_from_experiences(
        tenant_id=tenant_id,
        user_id=user_id,
        experience_ids=(first.id,),
        name="reviewer-recovery",
        purpose="审查恢复。",
        output_contract="输出审查结果。",
        minimum_successes=1,
    )
    updated = await promotion_service.promote_from_experiences(
        tenant_id=tenant_id,
        user_id=user_id,
        experience_ids=(first.id, second.id),
        name="reviewer-recovery",
        purpose="审查恢复。",
        output_contract="输出审查结果。",
        minimum_successes=1,
    )

    skills = await cognitive_repository.list_for_user(
        SkillCandidateRecord,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    assert len(skills) == 1
    assert updated.id == original.id
    assert updated.version == original.version + 1
    assert updated.steps == ("先压缩输入。", "再拆分审查块。")


@pytest.mark.asyncio
async def test_strategy_library_candidate_is_not_selected_until_confirmed() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    repository = InMemoryCognitiveRecordRepository()
    service = CognitiveStateService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()

    candidate = await service.create_strategy_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        name="reviewer-timeout-recovery",
        context="reviewer 审查大输入时超时。",
        strategy="先压缩输入，再拆分审查块。",
        rationale="大输入直接审查容易超时。",
        evidence=(CognitiveEvidence(source_type="reflection", source_id="ref-1", note="timeout analysis"),),
        tags=("reviewer", "timeout"),
        applies_to_modes=("hybrid", "dispatch"),
        confidence=0.78,
    )

    before_confirm = await service.select_strategies_for_task(
        tenant_id=tenant_id,
        user_id=user_id,
        request="reviewer 审查时又 timeout 了",
        mode="hybrid",
    )
    confirmed = await service.confirm_strategy(candidate.id, tenant_id=tenant_id, user_id=user_id)
    after_confirm = await service.select_strategies_for_task(
        tenant_id=tenant_id,
        user_id=user_id,
        request="reviewer 审查时又 timeout 了",
        mode="hybrid",
    )

    assert candidate.status is StrategyStatus.CANDIDATE
    assert candidate.active_for_runtime is False
    assert before_confirm == ()
    assert confirmed.status is StrategyStatus.ACTIVE
    assert confirmed.active_for_runtime is True
    assert [item.id for item in after_confirm] == [candidate.id]


@pytest.mark.asyncio
async def test_strategy_library_selects_active_relevant_visible_high_confidence_strategies() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    repository = InMemoryCognitiveRecordRepository()
    service = CognitiveStateService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    owner_user_id = uuid4()
    other_user_id = uuid4()
    visible = await service.create_strategy_candidate(
        tenant_id=tenant_id,
        user_id=owner_user_id,
        memory_scope=CognitiveMemoryScope.ROOT,
        name="cubeagent-boundary",
        context="CubeAgent 仓库任务。",
        strategy="只做纯对话 Agent，不做 harness，不部署。",
        rationale="当前仓库边界要求。",
        evidence=(CognitiveEvidence(source_type="handoff", source_id="handoff-1", note="project boundary"),),
        tags=("cubeagent", "harness", "deploy"),
        applies_to_modes=("hybrid",),
        confidence=0.9,
    )
    visible = await service.confirm_strategy(visible.id, tenant_id=tenant_id, user_id=owner_user_id)
    low_confidence = await service.create_strategy_candidate(
        tenant_id=tenant_id,
        user_id=owner_user_id,
        name="weak-boundary",
        context="CubeAgent 仓库任务。",
        strategy="弱信号不应进入选择。",
        rationale="证据不足。",
        evidence=(CognitiveEvidence(source_type="note", source_id="n-1", note="weak"),),
        tags=("cubeagent",),
        applies_to_modes=("hybrid",),
        confidence=0.52,
    )
    await service.confirm_strategy(low_confidence.id, tenant_id=tenant_id, user_id=owner_user_id)
    wrong_mode = await service.create_strategy_candidate(
        tenant_id=tenant_id,
        user_id=owner_user_id,
        name="direct-only",
        context="direct 模式任务。",
        strategy="direct 专用策略不应进入 hybrid。",
        rationale="模式不匹配。",
        evidence=(CognitiveEvidence(source_type="note", source_id="n-2", note="mode"),),
        tags=("cubeagent",),
        applies_to_modes=("direct",),
        confidence=0.9,
    )
    await service.confirm_strategy(wrong_mode.id, tenant_id=tenant_id, user_id=owner_user_id)
    irrelevant = await service.create_strategy_candidate(
        tenant_id=tenant_id,
        user_id=owner_user_id,
        name="billing-response",
        context="用户询问账单。",
        strategy="先解释账单口径。",
        rationale="账单类问题需要口径。",
        evidence=(CognitiveEvidence(source_type="note", source_id="n-3", note="billing"),),
        tags=("billing",),
        applies_to_modes=("hybrid",),
        confidence=0.9,
    )
    await service.confirm_strategy(irrelevant.id, tenant_id=tenant_id, user_id=owner_user_id)

    selected = await service.select_strategies_for_task(
        tenant_id=tenant_id,
        user_id=other_user_id,
        request="继续 CubeAgent 认知层，仓库只做纯对话 Agent，不要做 harness 或部署",
        mode="hybrid",
    )

    assert [item.id for item in selected] == [visible.id]


@pytest.mark.asyncio
async def test_strategy_library_reject_and_use_outcome_update_learning_state() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    timestamps = iter(
        [
            datetime(2026, 9, 2, 10, 0, tzinfo=UTC),
            datetime(2026, 9, 2, 10, 1, tzinfo=UTC),
            datetime(2026, 9, 2, 10, 2, tzinfo=UTC),
            datetime(2026, 9, 2, 10, 3, tzinfo=UTC),
            datetime(2026, 9, 2, 10, 4, tzinfo=UTC),
        ]
    )
    repository = InMemoryCognitiveRecordRepository()
    service = CognitiveStateService(repository, now=lambda: next(timestamps))
    tenant_id = uuid4()
    user_id = uuid4()
    rejected_candidate = await service.create_strategy_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        name="obsolete",
        context="旧任务。",
        strategy="旧策略。",
        rationale="不再适用。",
        evidence=(CognitiveEvidence(source_type="note", source_id="n-1", note="obsolete"),),
        tags=("obsolete",),
        confidence=0.7,
    )
    rejected = await service.reject_strategy(
        rejected_candidate.id,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    active = await service.create_strategy_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        name="reviewer-timeout-recovery",
        context="reviewer 审查大输入时超时。",
        strategy="先压缩输入，再拆分审查块。",
        rationale="大输入直接审查容易超时。",
        evidence=(CognitiveEvidence(source_type="reflection", source_id="ref-1", note="timeout analysis"),),
        tags=("reviewer", "timeout"),
        applies_to_modes=("hybrid",),
        confidence=0.78,
    )
    active = await service.confirm_strategy(active.id, tenant_id=tenant_id, user_id=user_id)

    failed = await service.record_strategy_use_outcome(
        active.id,
        tenant_id=tenant_id,
        user_id=user_id,
        succeeded=False,
        evidence=CognitiveEvidence(source_type="run", source_id="run-1", note="strategy failed"),
    )

    assert rejected.status is StrategyStatus.REJECTED
    assert rejected.active_for_runtime is False
    assert failed.use_count == 1
    assert failed.success_count == 0
    assert failed.failure_count == 1
    assert failed.confidence < active.confidence
    assert failed.status is StrategyStatus.ACTIVE
    assert failed.version == active.version + 1
    assert len(failed.evidence) == 1
    assert len(failed.contradictions) == 1
    assert failed.last_used_at == datetime(2026, 9, 2, 10, 4, tzinfo=UTC)
    assert failed.last_verified_at == datetime(2026, 9, 2, 10, 4, tzinfo=UTC)


@pytest.mark.asyncio
async def test_repeated_failed_strategy_use_marks_strategy_contested() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    repository = InMemoryCognitiveRecordRepository()
    service = CognitiveStateService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()
    strategy = await service.create_strategy_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        name="weak-reviewer-recovery",
        context="reviewer 审查大输入时超时。",
        strategy="继续完整审查。",
        rationale="旧策略证据不足。",
        evidence=(CognitiveEvidence(source_type="reflection", source_id="ref-1", note="weak"),),
        tags=("reviewer", "timeout"),
        applies_to_modes=("hybrid",),
        confidence=0.52,
    )
    strategy = await service.confirm_strategy(strategy.id, tenant_id=tenant_id, user_id=user_id)

    for index in range(3):
        strategy = await service.record_strategy_use_outcome(
            strategy.id,
            tenant_id=tenant_id,
            user_id=user_id,
            succeeded=False,
            evidence=CognitiveEvidence(source_type="run", source_id=f"run-{index}", note="failed"),
        )

    selected = await service.select_strategies_for_task(
        tenant_id=tenant_id,
        user_id=user_id,
        request="reviewer timeout",
        mode="hybrid",
    )

    assert strategy.status is StrategyStatus.CONTESTED
    assert strategy.active_for_runtime is False
    assert selected == ()


@pytest.mark.asyncio
async def test_outcome_assessment_is_persisted_and_scoped() -> None:
    from agent_hub.cognitive import InMemoryCognitiveRecordRepository

    repository = InMemoryCognitiveRecordRepository()
    service = CognitiveStateService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    owner_user_id = uuid4()
    other_user_id = uuid4()

    assessment = await service.record_outcome_assessment(
        tenant_id=tenant_id,
        user_id=owner_user_id,
        source_run_id="run-1",
        target_type="strategy",
        target_id="strategy-1",
        verdict=OutcomeVerdict.PARTIAL,
        note="completed with reviewer fallback",
        evidence=(CognitiveEvidence(source_type="run_event", source_id="event-1", note="step.failed recovered"),),
        confidence_delta=-0.04,
    )

    owner_records = await repository.list_for_user(
        OutcomeAssessmentRecord,
        tenant_id=tenant_id,
        user_id=owner_user_id,
    )
    other_records = await repository.list_for_user(
        OutcomeAssessmentRecord,
        tenant_id=tenant_id,
        user_id=other_user_id,
    )

    assert assessment.verdict is OutcomeVerdict.PARTIAL
    assert [item.id for item in owner_records] == [assessment.id]
    assert other_records == ()
