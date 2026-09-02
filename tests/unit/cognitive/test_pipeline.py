from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from agent_hub.cognitive.context_router import route_cognitive_context
from agent_hub.cognitive.pipeline import CognitiveLearningPipeline, OutcomeCritic
from agent_hub.cognitive.repository import (
    InMemoryCognitiveRecordRepository,
    InMemoryExperienceRepository,
)
from agent_hub.cognitive.service import CognitiveStateService, ExperienceService
from agent_hub.cognitive.types import OutcomeAssessmentRecord, OutcomeVerdict, ReflectionRecord
from agent_hub.domain.runs import RunStatus, TaskMode
from agent_hub.memory.maintenance import MemoryMaintenanceService
from agent_hub.memory.repository import InMemoryMemoryRepository
from agent_hub.memory.service import MemoryService
from agent_hub.memory.types import MemoryCategory, MemoryLayer, MemoryRecord


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


@pytest.mark.asyncio
async def test_learning_injection_failure_feedback_and_memory_maintenance_closed_loop() -> None:
    now = [datetime(2026, 9, 2, tzinfo=UTC)]
    cognitive_repository = InMemoryCognitiveRecordRepository()
    experience_repository = InMemoryExperienceRepository()
    memory_repository = InMemoryMemoryRepository()
    cognitive_service = CognitiveStateService(cognitive_repository, now=lambda: now[0])
    experience_service = ExperienceService(experience_repository, now=lambda: now[0])
    memory_service = MemoryService(memory_repository, now=lambda: now[0])
    pipeline = CognitiveLearningPipeline(
        cognitive_service=cognitive_service,
        experience_service=experience_service,
    )
    tenant_id = uuid4()
    user_id = uuid4()

    learned = await pipeline.process_terminal_run(
        tenant_id=tenant_id,
        actor_id=user_id,
        run_id=uuid4(),
        status=RunStatus.COMPLETED,
        mode=TaskMode.HYBRID,
        routing_decision={"used_experience_ids": []},
        events=(
            {
                "kind": "runtime.completed",
                "message": "hybrid reviewer timeout recovery completed with visible output",
            },
        ),
        artifacts=({"kind": "text", "title": "answer", "text": "压缩上下文后完成审查"},),
    )
    assert learned is not None
    assert learned.experience is not None

    confirmed = await experience_service.confirm(learned.experience.id, tenant_id=tenant_id, user_id=user_id)
    routed = route_cognitive_context(
        request="runtime_outcome success hybrid",
        mode=TaskMode.HYBRID.value,
        agent_ids=("quality_reviewer",),
        experiences=(confirmed,),
        limit=3,
    )
    assert routed.selected
    assert routed.selected[0].source == "experience"

    now[0] += timedelta(minutes=5)
    failed = await pipeline.process_terminal_run(
        tenant_id=tenant_id,
        actor_id=user_id,
        run_id=uuid4(),
        status=RunStatus.FAILED,
        mode=TaskMode.HYBRID,
        routing_decision={"injected_experience_ids": [str(confirmed.id)]},
        events=(
            {
                "kind": "runtime.failed",
                "message": "hybrid dispatch failed: reviewer step timed out",
            },
        ),
        artifacts=(),
    )
    assert failed is not None
    updated_experience = (await experience_repository.list_for_user(tenant_id, user_id))[0]
    assert updated_experience.failure_count == 1
    assert updated_experience.confidence < confirmed.confidence
    assert updated_experience.contradictions

    for index in range(3):
        added = await memory_service.add_candidate(
            tenant_id=tenant_id,
            user_id=user_id,
            text=f"reviewer 超时经验原始片段 {index}：先压缩上下文。",
            category=MemoryCategory.LESSON,
            project_id="cubeagent",
            conversation_id="conv-review",
            confidence=0.72,
        )
        assert added.record is not None
    maintained = await MemoryMaintenanceService(memory_repository, now=lambda: now[0]).maintain(apply=True)
    assert maintained.compressed == 1
    all_memories = await memory_repository.list_all()
    summaries = [record for record in all_memories if record.category is MemoryCategory.SUMMARY]
    assert len(summaries) == 1
    assert len(summaries[0].source_memory_ids) == 3
    for source_id in summaries[0].source_memory_ids:
        source = await memory_repository.get(source_id)
        assert source is not None
        assert source.archived_at is not None


@pytest.mark.asyncio
async def test_learning_injection_outcome_and_memory_maintenance_loop() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    cognitive_repository = InMemoryCognitiveRecordRepository()
    experience_repository = InMemoryExperienceRepository()
    cognitive_service = CognitiveStateService(cognitive_repository, now=lambda: now)
    experience_service = ExperienceService(experience_repository, now=lambda: now)
    tenant_id = uuid4()
    user_id = uuid4()
    run_id = uuid4()

    learned = await CognitiveLearningPipeline(
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
        routing_decision={"workflow_id": "no-workflow"},
    )
    assert learned is not None
    assert learned.experience is not None
    confirmed = await experience_service.confirm(learned.experience.id, tenant_id=tenant_id, user_id=user_id)

    routed = route_cognitive_context(
        request="runtime_outcome failure hybrid 怎么处理",
        mode="hybrid",
        agent_ids=("main_agent",),
        experiences=(confirmed,),
        limit=3,
        total_context_budget=2,
    )
    unrelated = route_cognitive_context(
        request="帮我写一首生日诗",
        mode="direct",
        agent_ids=("main_agent",),
        experiences=(confirmed,),
        limit=3,
    )
    assert [item.id for item in routed.selected] == [f"cognitive_experience:{confirmed.id}"]
    assert unrelated.selected == ()

    second_run_id = uuid4()
    await CognitiveLearningPipeline(
        cognitive_service=cognitive_service,
        experience_service=experience_service,
        run_repository=FakeRunEvidenceRepository(
            events=(
                {
                    "kind": "runtime.failed",
                    "message": "same strategy failed again after injection",
                },
            ),
        ),
    ).process_terminal_run(
        tenant_id=tenant_id,
        actor_id=user_id,
        run_id=second_run_id,
        status=RunStatus.FAILED,
        mode=TaskMode.HYBRID,
        routing_decision={"injected_experience_ids": [str(confirmed.id)]},
    )
    updated = await experience_repository.get(confirmed.id)
    assert updated is not None
    assert updated.failure_count == 1
    assert updated.confidence < confirmed.confidence

    memory_repository = InMemoryMemoryRepository()
    for index in range(3):
        await memory_repository.upsert(
            MemoryRecord(
                id=uuid4(),
                tenant_id=tenant_id,
                user_id=user_id,
                layer=MemoryLayer.EPISODIC,
                category=MemoryCategory.LESSON,
                text=f"重复运行记忆 {index}：runtime failure 需要压缩。",
                confidence=0.72,
                created_at=now,
                updated_at=now,
                heat=0.62,
                project_id="cubeagent",
                conversation_id="conv-loop",
            )
        )
    maintained = await MemoryMaintenanceService(memory_repository, now=lambda: now).maintain(apply=True)
    memory_records = await memory_repository.list_for_user(tenant_id, user_id)

    assert maintained.compressed == 1
    assert sum(1 for record in memory_records if record.category is MemoryCategory.SUMMARY) == 1
    assert sum(1 for record in memory_records if record.archived_at is not None) == 3
