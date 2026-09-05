from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from agent_hub.domain.runs import RunStatus, TaskMode
from agent_hub.runs.repository import RunRecord
from agent_hub.runs.service import HermesMemoryInjection, HermesRunAdvice, RunService
from agent_hub.runtime.defaults import UnavailableRuntime
from agent_hub.runtime.registry import RuntimeRegistry


class RecordingQueue:
    def __init__(self) -> None:
        self.enqueued: list[UUID] = []

    async def enqueue_run(self, run_id: UUID, *, idempotency_key: str) -> None:
        del idempotency_key
        self.enqueued.append(run_id)


class RecordingRepository:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []

    async def create_run(
        self,
        *,
        tenant_id: UUID,
        actor_id: UUID,
        request: str,
        mode: TaskMode | None,
        status: RunStatus,
        idempotency_key: str | None,
        routing_decision: dict[str, object] | None = None,
        enqueue: bool,
    ) -> RunRecord:
        del idempotency_key, enqueue
        self.created.append(
            {
                "request": request,
                "mode": mode,
                "status": status,
                "routing_decision": routing_decision,
            }
        )
        return RunRecord(
            id=uuid4(),
            tenant_id=tenant_id,
            actor_id=actor_id,
            request=request,
            mode=mode,
            status=status,
            version=1,
            created_at=datetime.now(UTC),
            routing_decision=routing_decision,
        )


class RecordingHermesAdvisor:
    def __init__(self, advice: HermesRunAdvice) -> None:
        self.advice = advice
        self.calls: list[dict[str, object]] = []

    async def advise(
        self,
        *,
        tenant_id: UUID,
        actor_id: UUID,
        message: str,
        mode: TaskMode,
        agent_ids: tuple[str, ...],
        workflow_id: str | None,
    ) -> HermesRunAdvice:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "actor_id": actor_id,
                "message": message,
                "mode": mode,
                "agent_ids": agent_ids,
                "workflow_id": workflow_id,
            }
        )
        return self.advice

    async def record_outcome(self, outcome: object) -> None:
        del outcome


async def test_direct_robot_submit_keeps_direct_mode_and_injects_hermes_memories() -> None:
    repository = RecordingRepository()
    advisor = RecordingHermesAdvisor(
        HermesRunAdvice(
            recommended_mode=TaskMode.DISPATCH,
            confidence=0.9,
            reasons=("companion recall",),
            requires_approval=False,
            injected_memories=(
                HermesMemoryInjection(
                    id="user_likes_tea",
                    summary="用户喜欢喝热茶，晚上少喝咖啡。",
                    memory_type="preference",
                    target="main_agent",
                    score=0.88,
                    reason="陪伴偏好",
                ),
            ),
        )
    )
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=None,
        task_queue=RecordingQueue(),
        hermes_advisor=advisor,
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="晚上好，给我倒杯什么好？",
        mode=TaskMode.DIRECT,
        conversation_id="ch-robot-abc",
        channel_context={"source_channel": "robot"},
        skip_evolution_proposal=True,
        idempotency_key="robot-hermes-1",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DIRECT
    assert len(advisor.calls) == 1
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["source_channel"] == "robot"
    hermes = routing["hermes"]
    assert isinstance(hermes, dict)
    memories = hermes["injected_memories"]
    assert isinstance(memories, list)
    assert memories[0]["id"] == "user_likes_tea"
    assert memories[0]["summary"] == "用户喜欢喝热茶，晚上少喝咖啡。"


async def test_direct_web_submit_does_not_call_hermes() -> None:
    repository = RecordingRepository()
    advisor = RecordingHermesAdvisor(
        HermesRunAdvice(
            recommended_mode=TaskMode.DIRECT,
            confidence=0.9,
            reasons=("unused",),
            requires_approval=False,
        )
    )
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=None,
        task_queue=RecordingQueue(),
        hermes_advisor=advisor,
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="plain web chat",
        mode=TaskMode.DIRECT,
        conversation_id="conv-web",
        skip_evolution_proposal=True,
        idempotency_key="web-direct-1",
    )

    assert submitted.mode is TaskMode.DIRECT
    assert advisor.calls == []
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert "hermes" not in routing
