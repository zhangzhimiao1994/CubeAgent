from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from agent_hub.cognitive.router import route_experiences
from agent_hub.cognitive.types import (
    CognitiveEvidence,
    ExperienceKind,
    ExperienceRecord,
    ExperienceStatus,
)


def _experience(
    summary: str,
    *,
    status: ExperienceStatus = ExperienceStatus.CONFIRMED,
    confidence: float = 0.86,
    failures: int = 0,
) -> ExperienceRecord:
    now = datetime.now(UTC)
    return ExperienceRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        kind=ExperienceKind.ERROR_HANDLING,
        status=status,
        summary=summary,
        lesson="reviewer timeout",
        strategy="compress then split",
        confidence=confidence,
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="timeout"),),
        contradictions=(),
        source_run_ids=("run-1",),
        source_memory_ids=(),
        tags=("reviewer", "timeout", "审查"),
        applies_to_modes=("dispatch", "hybrid"),
        applies_to_agents=("quality_reviewer",),
        use_count=2,
        success_count=2 - failures,
        failure_count=failures,
        last_used_at=None,
        last_verified_at=now,
        version=1,
        created_at=now,
        updated_at=now,
    )


def test_router_selects_relevant_confirmed_experience() -> None:
    result = route_experiences(
        request="审查输出时 reviewer 又超时了",
        mode="dispatch",
        agent_ids=("quality_reviewer",),
        experiences=(_experience("reviewer 超时时先压缩上下文再分块审查。"),),
    )

    assert result.selected[0].summary == "reviewer 超时时先压缩上下文再分块审查。"
    assert result.selected[0].score >= 0.7
    assert result.skipped == ()


def test_router_skips_unconfirmed_experience() -> None:
    result = route_experiences(
        request="审查输出时 reviewer 又超时了",
        mode="dispatch",
        agent_ids=("quality_reviewer",),
        experiences=(_experience("候选经验", status=ExperienceStatus.CANDIDATE),),
    )

    assert result.selected == ()
    assert result.skipped[0].reason == "经验尚未确认"


def test_router_skips_conflicting_experience() -> None:
    result = route_experiences(
        request="这次必须直连，不要 hybrid 混合模式",
        mode="direct",
        agent_ids=(),
        experiences=(_experience("大任务优先使用 hybrid 混合模式。"),),
    )

    assert result.selected == ()
    assert result.skipped[0].reason == "当前用户指令覆盖这条经验"
