from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_hub.cognitive.types import (
    BeliefRecord,
    CognitiveEvidence,
    ExperienceKind,
    ExperienceRecord,
    ExperienceStatus,
    RelationshipStateRecord,
    SkillCandidateRecord,
    StrategyRecord,
    StrategyStatus,
    WorldStateRecord,
)


def test_experience_requires_bounded_confidence_and_evidence() -> None:
    now = datetime.now(UTC)
    record = ExperienceRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        kind=ExperienceKind.ERROR_HANDLING,
        status=ExperienceStatus.CANDIDATE,
        summary="reviewer 超时时先压缩上下文再分块审查。",
        lesson="大输入导致 reviewer 超时。",
        strategy="先压缩输入，再拆分审查任务。",
        confidence=0.72,
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="reviewer timeout"),),
        contradictions=(),
        source_run_ids=("run-1",),
        source_memory_ids=(),
        tags=("reviewer", "timeout"),
        applies_to_modes=("hybrid", "dispatch"),
        applies_to_agents=("quality_reviewer",),
        use_count=0,
        success_count=0,
        failure_count=0,
        last_used_at=None,
        last_verified_at=now,
        version=1,
        created_at=now,
        updated_at=now,
    )

    assert record.summary == "reviewer 超时时先压缩上下文再分块审查。"
    assert record.active_for_runtime is False


def test_experience_rejects_unbounded_confidence() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError):
        ExperienceRecord(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            kind=ExperienceKind.USER_PREFERENCE,
            status=ExperienceStatus.ACTIVE,
            summary="用户偏好简洁回答。",
            lesson="用户多次要求精简。",
            strategy="先给结论，再给必要证据。",
            confidence=1.4,
            evidence=(),
            contradictions=(),
            source_run_ids=(),
            source_memory_ids=(),
            tags=("communication",),
            applies_to_modes=(),
            applies_to_agents=(),
            use_count=0,
            success_count=0,
            failure_count=0,
            last_used_at=None,
            last_verified_at=now,
            version=1,
            created_at=now,
            updated_at=now,
        )


def test_belief_tracks_evidence_contradictions_and_verification() -> None:
    now = datetime.now(UTC)
    belief = BeliefRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        subject="user.workflow_preference",
        claim="用户偏好默认使用子 agent 分工。",
        confidence=0.81,
        evidence=(CognitiveEvidence(source_type="feedback", source_id="msg-1", note="explicit preference"),),
        contradictions=(),
        status="active",
        use_count=2,
        success_count=1,
        failure_count=1,
        last_used_at=now,
        version=3,
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )

    assert belief.confidence == 0.81
    assert belief.contradictions == ()
    assert belief.use_count == 2
    assert belief.version == 3


def test_relationship_world_and_skill_records_are_separate_from_runtime_permissions() -> None:
    now = datetime.now(UTC)
    tenant_id = uuid4()
    user_id = uuid4()
    relationship = RelationshipStateRecord(
        id=f"relationship:{user_id}",
        tenant_id=tenant_id,
        user_id=user_id,
        familiarity=0.5,
        preferred_language="zh-CN",
        preferred_confirmation_style="minimal",
        shared_milestones=("完成 Hermes+ 记忆注入",),
        recent_friction_points=("不要把问题甩给用户",),
        last_interaction_at=now,
        created_at=now,
        updated_at=now,
    )
    world = WorldStateRecord(
        id="world:project:cubeagent",
        tenant_id=tenant_id,
        user_id=user_id,
        scope="project:cubeagent",
        facts=("CubeAgent 仓库只做纯对话 Agent。",),
        open_items=("持续学习系统待实现",),
        future_events=(),
        last_verified_at=now,
        evidence=(CognitiveEvidence(source_type="handoff", source_id="HANDOFF", note="project boundary"),),
        created_at=now,
        updated_at=now,
    )
    skill = SkillCandidateRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=user_id,
        name="reviewer-timeout-recovery",
        purpose="处理 reviewer 超时",
        steps=("压缩输入", "拆分审查", "重试或降级"),
        required_inputs=("失败步骤", "输入大小", "模型"),
        output_contract="给出可执行修复或明确跳过原因",
        confidence=0.73,
        evidence=(CognitiveEvidence(source_type="experience", source_id="exp-1", note="repeated success"),),
        contradictions=(),
        use_count=0,
        success_count=0,
        failure_count=0,
        last_used_at=None,
        last_verified_at=now,
        version=1,
        status="candidate",
        created_at=now,
        updated_at=now,
    )

    assert relationship.preferred_language == "zh-CN"
    assert world.scope == "project:cubeagent"
    assert skill.status == "candidate"


def test_strategy_record_starts_as_candidate_and_tracks_outcomes() -> None:
    now = datetime.now(UTC)
    record = StrategyRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        name="large-task-split-first",
        context="用户要求完成大范围系统测试或多模块改造。",
        strategy="先拆分任务，分别验证，再汇总结论。",
        rationale="大任务直接执行容易超时或遗漏。",
        status=StrategyStatus.CANDIDATE,
        confidence=0.66,
        evidence=(CognitiveEvidence(source_type="reflection", source_id="ref-1", note="failure analysis"),),
        contradictions=(),
        tags=("large-task", "split", "verify"),
        applies_to_modes=("hybrid", "dispatch"),
        use_count=3,
        success_count=2,
        failure_count=1,
        last_used_at=now,
        last_verified_at=now,
        version=2,
        created_at=now,
        updated_at=now,
    )

    assert record.active_for_runtime is False
    assert record.success_count + record.failure_count <= record.use_count
    assert record.status is StrategyStatus.CANDIDATE
