from __future__ import annotations

from datetime import UTC, datetime

from agent_hub.cognitive.metacognition import CognitiveGateLevel, MetacognitionService


def test_metacognition_keeps_simple_tasks_lightweight() -> None:
    decision = MetacognitionService(now=lambda: datetime(2026, 9, 2, tzinfo=UTC)).assess(
        request="解释一下这个概念",
        mode="direct",
        uncertainty_signals=(),
        conflict_count=0,
        last_verified_at=datetime(2026, 9, 2, tzinfo=UTC),
    )

    assert decision.level is CognitiveGateLevel.LIGHT
    assert decision.actions == ("answer_directly",)
    assert decision.context_budget <= 3


def test_metacognition_escalates_complex_uncertain_or_stale_tasks() -> None:
    decision = MetacognitionService(now=lambda: datetime(2026, 9, 2, tzinfo=UTC)).assess(
        request="继续完成整个系统三阶段改造，测试、审查、提交和 CI 都要处理",
        mode="hybrid",
        uncertainty_signals=("missing_current_state", "model_error_seen"),
        conflict_count=2,
        last_verified_at=datetime(2026, 5, 1, tzinfo=UTC),
    )

    assert decision.level is CognitiveGateLevel.ADVANCED
    assert "retrieve_memory" in decision.actions
    assert "verify_state" in decision.actions
    assert "delegate_agent" in decision.actions
    assert decision.context_budget > 3
    assert decision.reasons
