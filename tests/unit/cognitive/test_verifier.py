from __future__ import annotations

from agent_hub.cognitive.types import OutcomeVerdict
from agent_hub.cognitive.verifier import OutcomeVerifier


def test_outcome_verifier_marks_completed_run_with_output_as_success() -> None:
    assessment = OutcomeVerifier().assess(
        terminal_status="completed",
        events=({"kind": "runtime.completed", "message": "run completed"},),
        artifacts=({"kind": "text", "title": "final answer", "text": "交付结果"},),
    )

    assert assessment.verdict is OutcomeVerdict.SUCCESS
    assert assessment.learnable is True
    assert assessment.confidence >= 0.75


def test_outcome_verifier_accepts_nested_text_artifact_content() -> None:
    assessment = OutcomeVerifier().assess(
        terminal_status="completed",
        events=({"kind": "runtime.completed", "message": "run completed"},),
        artifacts=(
            {
                "kind": "text",
                "title": "final answer",
                "content": {"text": "真实 artifact payload 输出"},
            },
        ),
    )

    assert assessment.verdict is OutcomeVerdict.SUCCESS


def test_outcome_verifier_marks_completed_run_with_recovered_failures_as_partial() -> None:
    assessment = OutcomeVerifier().assess(
        terminal_status="completed",
        events=(
            {"kind": "step.failed", "message": "reviewer timeout"},
            {"kind": "runtime.completed", "message": "completed with fallback"},
        ),
        artifacts=({"kind": "text", "title": "fallback answer", "text": "部分完成"},),
    )

    assert assessment.verdict is OutcomeVerdict.PARTIAL
    assert assessment.learnable is True
    assert "failure" in assessment.failure_or_gap_reason


def test_outcome_verifier_marks_failed_terminal_state_as_failure() -> None:
    assessment = OutcomeVerifier().assess(
        terminal_status="failed",
        events=({"kind": "runtime.failed", "message": "model response text is empty"},),
        artifacts=(),
    )

    assert assessment.verdict is OutcomeVerdict.FAILURE
    assert assessment.learnable is True
    assert assessment.confidence >= 0.8


def test_outcome_verifier_requires_evidence_for_success() -> None:
    assessment = OutcomeVerifier().assess(
        terminal_status="completed",
        events=({"kind": "runtime.completed", "message": "done"},),
        artifacts=(),
    )

    assert assessment.verdict is OutcomeVerdict.INSUFFICIENT_EVIDENCE
    assert assessment.learnable is False
