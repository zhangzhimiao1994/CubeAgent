from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from agent_hub.cognitive.governance import (
    AntiLearningService,
    ConfidenceCalibrationService,
    ConflictResolutionEngine,
    ConflictResolutionStatus,
)
from agent_hub.cognitive.types import (
    BeliefRecord,
    CognitiveEvidence,
    CognitiveMemoryScope,
    StrategyStatus,
)


def test_conflict_resolution_marks_close_competing_beliefs_unresolved() -> None:
    now = datetime.now(UTC)
    first = _belief("user.prefers.detail", "用户偏好详细解释。", confidence=0.72, now=now)
    second = _belief("user.prefers.detail", "用户偏好极简结论。", confidence=0.69, now=now)

    decision = ConflictResolutionEngine().resolve_beliefs((first, second))

    assert decision.status is ConflictResolutionStatus.UNRESOLVED
    assert decision.winning_record_id is None
    assert {str(first.id), str(second.id)} <= set(decision.conflicting_record_ids)


def test_conflict_resolution_selects_stronger_verified_belief() -> None:
    now = datetime.now(UTC)
    winner = _belief("project.boundary", "CubeAgent 只做纯对话 Agent。", confidence=0.91, now=now)
    weaker = _belief("project.boundary", "CubeAgent 同时做 harness。", confidence=0.48, now=now)

    decision = ConflictResolutionEngine().resolve_beliefs((weaker, winner))

    assert decision.status is ConflictResolutionStatus.RESOLVED
    assert decision.winning_record_id == str(winner.id)


def test_confidence_calibration_rewards_success_and_penalizes_old_contradicted_records() -> None:
    now = datetime.now(UTC)
    service = ConfidenceCalibrationService()

    reinforced = service.calibrate(
        confidence=0.62,
        success_count=4,
        failure_count=0,
        contradiction_count=0,
        last_verified_at=now,
        now=now,
    )
    degraded = service.calibrate(
        confidence=0.62,
        success_count=0,
        failure_count=3,
        contradiction_count=2,
        last_verified_at=now - timedelta(days=120),
        now=now,
    )

    assert reinforced > 0.62
    assert degraded < 0.62


def test_anti_learning_degrades_repeatedly_failing_active_strategy() -> None:
    status = AntiLearningService().strategy_status(
        current_status=StrategyStatus.ACTIVE,
        confidence=0.37,
        success_count=1,
        failure_count=4,
    )

    assert status is StrategyStatus.CONTESTED


def _belief(subject: str, claim: str, *, confidence: float, now: datetime) -> BeliefRecord:
    return BeliefRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        memory_scope=CognitiveMemoryScope.USER,
        subject=subject,
        claim=claim,
        confidence=confidence,
        evidence=(CognitiveEvidence(source_type="feedback", source_id="msg-1", note="explicit"),),
        contradictions=(),
        status="active",
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )
