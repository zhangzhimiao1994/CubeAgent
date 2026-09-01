from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from agent_hub.cognitive.context_router import route_cognitive_context
from agent_hub.cognitive.types import (
    BeliefRecord,
    CognitiveEvidence,
    ExperienceKind,
    ExperienceRecord,
    ExperienceStatus,
    RelationshipStateRecord,
    SkillCandidateRecord,
    WorldStateRecord,
)
from agent_hub.memory.types import MemoryCategory, MemoryLayer, MemoryRecord


def test_context_router_selects_bounded_relevant_context_across_sources() -> None:
    now = datetime.now(UTC)
    result = route_cognitive_context(
        request="reviewer 审查又超时了，按项目边界处理，不要改 harness",
        mode="hybrid",
        agent_ids=("quality_reviewer",),
        memories=(
            _memory("用户要求默认使用子 agent，但本仓库不能直接改代码 harness。", now=now),
            _memory("无关的旅游计划。", now=now),
        ),
        experiences=(
            _experience("reviewer 超时时先压缩上下文再分块审查。", now=now),
        ),
        beliefs=(
            _belief("project.boundary", "CubeAgent 是纯对话 Agent，不实现 harness 代码执行。", now=now),
        ),
        relationship_states=(
            _relationship(("完成 Hermes+ 记忆隔离。",), ("用户不接受把可修复问题甩回去。",), now=now),
        ),
        world_states=(
            _world(
                "cubeagent.project",
                facts=("CubeAgent 当前生产只保留当前 release。",),
                open_items=("后续合并 Memory/Hermes。",),
                future_events=("harness 改造走独立项目。",),
                now=now,
            ),
        ),
        skill_candidates=(
            _skill("reviewer-timeout-recovery", ("先压缩输入。", "再拆分审查块。"), now=now),
        ),
        limit=5,
    )

    assert len(result.selected) == 5
    assert {item.source for item in result.selected} >= {"memory", "experience", "belief", "skill"}
    assert result.selected[0].score >= result.selected[-1].score
    assert any(item.source == "world_state" and "harness" in item.summary for item in result.selected)
    assert any(item.source == "memory" and "旅游" in item.summary for item in result.skipped)


def test_context_router_skips_unsafe_or_low_quality_candidates() -> None:
    now = datetime.now(UTC)
    deleted_memory = _memory("reviewer 超时处理规则。", now=now).model_copy(
        update={"deleted_at": now}
    )
    result = route_cognitive_context(
        request="reviewer 超时",
        mode="hybrid",
        agent_ids=("quality_reviewer",),
        memories=(deleted_memory,),
        experiences=(
            _experience(
                "候选经验不能注入。",
                status=ExperienceStatus.CANDIDATE,
                now=now,
            ),
        ),
        beliefs=(
            _belief("project.boundary", "低置信信念不能注入。", confidence=0.3, now=now),
        ),
        skill_candidates=(
            _skill("low-confidence-skill", ("重试。",), confidence=0.3, now=now),
        ),
        limit=5,
    )

    assert result.selected == ()
    assert {item.reason for item in result.skipped} >= {
        "记忆已删除或过期",
        "经验尚未确认",
        "信念置信度不足",
        "技能候选置信度不足",
    }


def _memory(text: str, *, now: datetime) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        layer=MemoryLayer.CORE,
        category=MemoryCategory.PREFERENCE,
        text=text,
        confidence=0.85,
        created_at=now,
        updated_at=now,
        heat=0.8,
    )


def _experience(
    summary: str,
    *,
    status: ExperienceStatus = ExperienceStatus.CONFIRMED,
    now: datetime,
) -> ExperienceRecord:
    return ExperienceRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        kind=ExperienceKind.ERROR_HANDLING,
        status=status,
        summary=summary,
        lesson="reviewer timeout",
        strategy="compress then split",
        confidence=0.86,
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="timeout"),),
        tags=("reviewer", "timeout", "审查"),
        applies_to_modes=("dispatch", "hybrid"),
        applies_to_agents=("quality_reviewer",),
        use_count=2,
        success_count=2,
        failure_count=0,
        last_used_at=None,
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )


def _belief(
    subject: str,
    claim: str,
    *,
    confidence: float = 0.82,
    now: datetime,
) -> BeliefRecord:
    return BeliefRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        subject=subject,
        claim=claim,
        confidence=confidence,
        evidence=(CognitiveEvidence(source_type="feedback", source_id="msg-1", note="explicit"),),
        status="active",
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )


def _relationship(
    shared_milestones: tuple[str, ...],
    recent_friction_points: tuple[str, ...],
    *,
    now: datetime,
) -> RelationshipStateRecord:
    return RelationshipStateRecord(
        id="relationship:unit",
        tenant_id=uuid4(),
        user_id=uuid4(),
        familiarity=0.6,
        preferred_language="zh-CN",
        preferred_confirmation_style="minimal",
        shared_milestones=shared_milestones,
        recent_friction_points=recent_friction_points,
        last_interaction_at=now,
        created_at=now,
        updated_at=now,
    )


def _world(
    scope: str,
    *,
    facts: tuple[str, ...],
    open_items: tuple[str, ...],
    future_events: tuple[str, ...],
    now: datetime,
) -> WorldStateRecord:
    return WorldStateRecord(
        id="world:unit",
        tenant_id=uuid4(),
        user_id=uuid4(),
        scope=scope,
        facts=facts,
        open_items=open_items,
        future_events=future_events,
        evidence=(CognitiveEvidence(source_type="handoff", source_id="HANDOFF", note="project state"),),
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )


def _skill(
    name: str,
    steps: tuple[str, ...],
    *,
    confidence: float = 0.78,
    now: datetime,
) -> SkillCandidateRecord:
    return SkillCandidateRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        name=name,
        purpose="处理 reviewer 超时。",
        steps=steps,
        output_contract="输出审查结论。",
        confidence=confidence,
        evidence=(CognitiveEvidence(source_type="experience", source_id="exp-1", note="worked"),),
        success_count=2,
        failure_count=0,
        use_count=2,
        status="candidate",
        created_at=now,
        updated_at=now,
    )
