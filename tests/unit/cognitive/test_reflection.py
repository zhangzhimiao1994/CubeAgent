from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from agent_hub.cognitive.reflection import ReflectionEngine, reflect_from_feedback
from agent_hub.cognitive.types import (
    CognitiveEvidence,
    OutcomeAssessmentRecord,
    OutcomeVerdict,
)


def test_user_correction_creates_counterfactual_reflection() -> None:
    tenant_id = uuid4()
    user_id = uuid4()

    reflection = reflect_from_feedback(
        tenant_id=tenant_id,
        user_id=user_id,
        source_run_id="run-1",
        user_feedback="不是让你甩锅给我，发现问题应该先解决。",
        outcome="negative",
        now=lambda: datetime.now(UTC),
    )

    assert reflection.tenant_id == tenant_id
    assert reflection.user_id == user_id
    assert reflection.trigger == "user_correction"
    assert "先解决" in reflection.counterfactual
    assert reflection.confidence >= 0.6


def test_user_satisfaction_creates_positive_pattern() -> None:
    reflection = reflect_from_feedback(
        tenant_id=uuid4(),
        user_id=uuid4(),
        source_run_id="run-2",
        user_feedback="这样可以，继续按这个方式处理。",
        outcome="positive",
        now=lambda: datetime.now(UTC),
    )

    assert reflection.trigger == "user_satisfaction"
    assert reflection.positive_patterns


def test_success_outcome_creates_positive_reflection() -> None:
    tenant_id = uuid4()
    user_id = uuid4()
    assessment = OutcomeAssessmentRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=user_id,
        source_run_id="run-success",
        target_type="run",
        target_id="run-success",
        verdict=OutcomeVerdict.SUCCESS,
        note="completed with visible output",
        evidence=(CognitiveEvidence(source_type="run_event", source_id="1", note="runtime.completed"),),
        confidence_delta=0.05,
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
    )

    reflection = ReflectionEngine(now=lambda: datetime(2026, 9, 2, tzinfo=UTC)).reflect_from_outcome(
        assessment
    )

    assert reflection.trigger == "outcome_success"
    assert reflection.outcome == OutcomeVerdict.SUCCESS.value
    assert reflection.positive_patterns
    assert not reflection.negative_patterns


def test_failure_outcome_creates_counterfactual_reflection() -> None:
    assessment = OutcomeAssessmentRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        source_run_id="run-failed",
        target_type="run",
        target_id="run-failed",
        verdict=OutcomeVerdict.FAILURE,
        note="terminal runtime failure",
        evidence=(CognitiveEvidence(source_type="run_event", source_id="1", note="runtime.failed"),),
        confidence_delta=-0.1,
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
    )

    reflection = ReflectionEngine(now=lambda: datetime(2026, 9, 2, tzinfo=UTC)).reflect_from_outcome(
        assessment
    )

    assert reflection.trigger == "outcome_failure"
    assert "重新处理" in reflection.counterfactual
    assert reflection.negative_patterns
