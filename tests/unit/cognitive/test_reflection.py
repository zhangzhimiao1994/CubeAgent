from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from agent_hub.cognitive.reflection import reflect_from_feedback


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
