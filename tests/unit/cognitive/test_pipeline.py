from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from agent_hub.cognitive.pipeline import CognitiveLearningPipeline, OutcomeCritic
from agent_hub.cognitive.repository import (
    InMemoryCognitiveRecordRepository,
    InMemoryExperienceRepository,
)
from agent_hub.cognitive.service import CognitiveStateService, ExperienceService
from agent_hub.cognitive.types import OutcomeAssessmentRecord, OutcomeVerdict, ReflectionRecord
from agent_hub.domain.runs import RunStatus, TaskMode


class FakeRunEvidenceRepository:
    def __init__(
        self,
        *,
        events: tuple[dict[str, object], ...],
        artifacts: tuple[dict[str, object], ...] = (),
    ) -> None:
        self._events = events
        self._artifacts = artifacts

    async def events(self, tenant_id: UUID, run_id: UUID) -> tuple[dict[str, object], ...]:
        del tenant_id, run_id
        return self._events

    async def artifacts(self, tenant_id: UUID, run_id: UUID) -> tuple[dict[str, object], ...]:
        del tenant_id, run_id
        return self._artifacts


def test_outcome_critic_assesses_failure_with_evidence() -> None:
    verdict, confidence, note, evidence_notes, learnable = OutcomeCritic().assess(
        status=RunStatus.FAILED,
        events=({"kind": "runtime.failed", "message": "model response text is empty"},),
        artifacts=(),
    )

    assert verdict is OutcomeVerdict.FAILURE
    assert confidence >= 0.8
    assert note
    assert "runtime.failed" in evidence_notes
    assert learnable is True


@pytest.mark.asyncio
async def test_learning_pipeline_reflects_failure_and_creates_candidate_experience() -> None:
    cognitive_repository = InMemoryCognitiveRecordRepository()
    experience_repository = InMemoryExperienceRepository()
    cognitive_service = CognitiveStateService(
        cognitive_repository,
        now=lambda: datetime(2026, 9, 2, tzinfo=UTC),
    )
    experience_service = ExperienceService(
        experience_repository,
        now=lambda: datetime(2026, 9, 2, tzinfo=UTC),
    )
    tenant_id = uuid4()
    user_id = uuid4()
    run_id = uuid4()

    result = await CognitiveLearningPipeline(
        cognitive_service=cognitive_service,
        experience_service=experience_service,
        run_repository=FakeRunEvidenceRepository(
            events=(
                {
                    "kind": "runtime.failed",
                    "message": "hybrid direct failed: model response budget is unverifiable",
                },
            ),
        ),
    ).process_terminal_run(
        tenant_id=tenant_id,
        actor_id=user_id,
        run_id=run_id,
        status=RunStatus.FAILED,
        mode=TaskMode.HYBRID,
        routing_decision={"workflow_id": "no-workflow", "selected_agent_ids": ["reviewer"]},
    )

    reflections = await cognitive_repository.list_for_user(
        ReflectionRecord,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    outcomes = await cognitive_repository.list_for_user(
        OutcomeAssessmentRecord,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    experiences = await experience_repository.list_for_user(tenant_id, user_id)

    assert result is not None
    assert result.outcome.record.verdict is OutcomeVerdict.FAILURE
    assert reflections == (result.reflection,)
    assert outcomes == (result.outcome.record,)
    assert len(experiences) == 1
    assert experiences[0].status.value == "candidate"
    assert experiences[0].source_run_ids == (str(run_id),)
    assert "budget" in experiences[0].lesson


@pytest.mark.asyncio
async def test_learning_pipeline_skips_candidate_learning_without_visible_evidence() -> None:
    cognitive_repository = InMemoryCognitiveRecordRepository()
    experience_repository = InMemoryExperienceRepository()
    cognitive_service = CognitiveStateService(
        cognitive_repository,
        now=lambda: datetime(2026, 9, 2, tzinfo=UTC),
    )
    experience_service = ExperienceService(
        experience_repository,
        now=lambda: datetime(2026, 9, 2, tzinfo=UTC),
    )
    tenant_id = uuid4()
    user_id = uuid4()

    result = await CognitiveLearningPipeline(
        cognitive_service=cognitive_service,
        experience_service=experience_service,
        run_repository=FakeRunEvidenceRepository(
            events=({"kind": "runtime.completed", "message": "done"},),
        ),
    ).process_terminal_run(
        tenant_id=tenant_id,
        actor_id=user_id,
        run_id=uuid4(),
        status=RunStatus.COMPLETED,
        mode=TaskMode.DIRECT,
        routing_decision={"workflow_id": "no-workflow"},
    )

    reflections = await cognitive_repository.list_for_user(
        ReflectionRecord,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    experiences = await experience_repository.list_for_user(tenant_id, user_id)

    assert result is not None
    assert result.outcome.record.verdict is OutcomeVerdict.INSUFFICIENT_EVIDENCE
    assert result.reflection is None
    assert result.experience is None
    assert reflections == ()
    assert experiences == ()
