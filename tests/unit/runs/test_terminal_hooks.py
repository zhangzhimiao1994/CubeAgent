from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self
from uuid import UUID, uuid4

import pytest

from agent_hub.cognitive.pipeline import CognitiveLearningTerminalHook
from agent_hub.domain.runs import RunStatus, TaskMode
from agent_hub.runs.repository import ConversationContextItem, RunRecord
from agent_hub.runs.service import (
    HermesRunOutcome,
    RunService,
    _conversation_history_artifact,
)
from agent_hub.runtime.contracts import (
    Artifact,
    EventKind,
    RunEvent,
    RuntimeCheckpoint,
    TaskContext,
)
from agent_hub.runtime.registry import RuntimeRegistry

TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")
ACTOR_ID = UUID("22222222-2222-4222-8222-222222222222")


@dataclass(slots=True)
class FakeRunRow:
    id: UUID
    tenant_id: UUID
    actor_id: UUID | None
    request: str
    mode: str | None
    status: str
    version: int
    created_at: datetime
    routing_decision: dict[str, object] | None


class FakeTransaction:
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


    def begin(self) -> FakeTransaction:
        return self


class ExecutableFakeRepository:
    def __init__(
        self,
        *,
        routing_decision: dict[str, object],
        checkpoint: RuntimeCheckpoint | None = None,
        raw_artifacts: tuple[dict[str, object], ...] = (),
        conversation_context: tuple[ConversationContextItem, ...] = (),
    ) -> None:
        self.run_id = uuid4()
        self.row = FakeRunRow(
            id=self.run_id,
            tenant_id=TENANT_ID,
            actor_id=ACTOR_ID,
            request="run evolution round",
            mode=TaskMode.DISPATCH.value,
            status=RunStatus.QUEUED.value,
            version=1,
            created_at=datetime.now(UTC),
            routing_decision=routing_decision,
        )
        self.events: list[RunEvent] = []
        self.checkpoint = checkpoint
        self.raw_artifact_payloads = raw_artifacts
        self.conversation_context_items = conversation_context

    async def run_transaction(self) -> FakeTransaction:
        return FakeTransaction()

    async def claim_for_execution(
        self,
        session: FakeTransaction,
        run_id: UUID,
        *,
        allow_running_recovery: bool,
    ) -> tuple[FakeRunRow, RuntimeCheckpoint | None] | RunRecord:
        del session, allow_running_recovery
        assert run_id == self.run_id
        self.row.status = RunStatus.RUNNING.value
        self.row.version += 1
        return self.row, self.checkpoint

    async def get_for_update(self, session: FakeTransaction, run_id: UUID) -> FakeRunRow:
        del session
        assert run_id == self.run_id
        return self.row

    async def persist_event(
        self,
        session: FakeTransaction,
        *,
        tenant_id: UUID,
        run_id: UUID,
        event: RunEvent,
    ) -> None:
        del session
        assert tenant_id == TENANT_ID
        assert run_id == self.run_id
        self.events.append(event)
        if event.checkpoint is not None:
            self.checkpoint = event.checkpoint

    async def raw_artifacts(
        self, tenant_id: UUID, run_id: UUID
    ) -> tuple[dict[str, object], ...]:
        assert tenant_id == TENANT_ID
        assert run_id == self.run_id
        return self.raw_artifact_payloads

    async def conversation_context(
        self,
        tenant_id: UUID,
        conversation_id: str,
        *,
        before_run_id: UUID,
        limit: int = 6,
    ) -> tuple[ConversationContextItem, ...]:
        del limit
        assert tenant_id == TENANT_ID
        assert before_run_id == self.run_id
        assert conversation_id
        return self.conversation_context_items

    async def next_event_sequence(self, session: FakeTransaction, run_id: UUID) -> int:
        del session
        assert run_id == self.run_id
        return max((event.sequence for event in self.events), default=0) + 1

    async def update_status(
        self,
        tenant_id: UUID,
        run_id: UUID,
        status: RunStatus,
        *,
        mode: TaskMode | None = None,
    ) -> RunRecord:
        del mode
        assert tenant_id == TENANT_ID
        assert run_id == self.run_id
        self.row.status = status.value
        self.row.version += 1
        return self._record()

    async def fail_run(
        self,
        run_id: UUID,
        *,
        reason: str,
        diagnostics: object | None = None,
    ) -> RunRecord:
        del reason, diagnostics
        assert run_id == self.run_id
        self.row.status = RunStatus.FAILED.value
        self.row.version += 1
        return self._record()

    async def get(self, tenant_id: UUID, run_id: UUID) -> RunRecord:
        assert tenant_id == TENANT_ID
        assert run_id == self.run_id
        return self._record()

    def _record(self) -> RunRecord:
        return RunRecord(
            id=self.row.id,
            tenant_id=self.row.tenant_id,
            actor_id=self.row.actor_id,
            request=self.row.request,
            mode=None if self.row.mode is None else TaskMode(self.row.mode),
            status=RunStatus(self.row.status),
            version=self.row.version,
            created_at=self.row.created_at,
            routing_decision=None
            if self.row.routing_decision is None
            else dict(self.row.routing_decision),
        )


class RuntimeCompletes:
    mode = TaskMode.DISPATCH

    async def run(self, context: TaskContext) -> AsyncIterator[RunEvent]:
        yield RunEvent(kind=EventKind.RUNTIME_COMPLETED, sequence=1, run_id=context.run_id)

    async def save_checkpoint(self) -> RuntimeCheckpoint:
        raise AssertionError("not used")

    async def restore_checkpoint(self, checkpoint: RuntimeCheckpoint) -> None:
        del checkpoint

    async def cancel(self) -> None:
        raise AssertionError("not used")


class RuntimeCapturesArtifacts(RuntimeCompletes):
    def __init__(self) -> None:
        self.artifacts: tuple[Artifact, ...] = ()

    async def run(self, context: TaskContext) -> AsyncIterator[RunEvent]:
        self.artifacts = context.artifacts
        yield RunEvent(kind=EventKind.RUNTIME_COMPLETED, sequence=1, run_id=context.run_id)


class RuntimeReportsCapacityPressure:
    mode = TaskMode.DISPATCH

    async def run(self, context: TaskContext) -> AsyncIterator[RunEvent]:
        yield RunEvent(
            kind=EventKind.STEP_FAILED,
            sequence=1,
            run_id=context.run_id,
            actor="planner",
            step_id="planner_step",
            reason="model gateway failed: model capacity unavailable",
        )
        yield RunEvent(
            kind=EventKind.RUNTIME_FAILED,
            sequence=2,
            run_id=context.run_id,
            reason="model gateway failed: model capacity unavailable",
        )

    async def save_checkpoint(self) -> RuntimeCheckpoint:
        raise AssertionError("not used")

    async def restore_checkpoint(self, checkpoint: RuntimeCheckpoint) -> None:
        del checkpoint

    async def cancel(self) -> None:
        raise AssertionError("not used")


class RuntimeRequiresHydratedArtifactOnRestore(RuntimeCompletes):
    def __init__(self, *required_artifact_ids: UUID) -> None:
        self.required_artifact_ids = tuple(str(item) for item in required_artifact_ids)
        self.context_artifact_ids: tuple[str, ...] = ()
        self.restored: RuntimeCheckpoint | None = None

    async def run(self, context: TaskContext) -> AsyncIterator[RunEvent]:
        self.context_artifact_ids = tuple(str(artifact.id) for artifact in context.artifacts)
        if any(item not in self.context_artifact_ids for item in self.required_artifact_ids):
            raise RuntimeError("checkpoint artifacts were not hydrated")
        yield RunEvent(
            kind=EventKind.STEP_RETRYING,
            sequence=1,
            run_id=context.run_id,
            actor="multimedia_generator",
            step_id="multimedia_generator_step",
            reason="user rejected artifact review; regenerating stage",
        )
        yield RunEvent(kind=EventKind.RUNTIME_COMPLETED, sequence=2, run_id=context.run_id)

    async def restore_checkpoint(self, checkpoint: RuntimeCheckpoint) -> None:
        self.restored = checkpoint


class RuntimeRequestsArtifactReview(RuntimeCompletes):
    async def run(self, context: TaskContext) -> AsyncIterator[RunEvent]:
        yield RunEvent(
            kind=EventKind.APPROVAL_REQUESTED,
            sequence=1,
            run_id=context.run_id,
            actor="character_designer",
            approval_id="artifact-review-test",
            action="artifact_review",
            reason="user review required for intermediate artifact",
            payload={
                "approval_kind": "runtime_artifact_review",
                "stage_id": "character_model_sheet",
                "artifact_id": str(uuid4()),
                "next_action": "approve_or_revise_artifact",
            },
        )


class RuntimeCompletesAfterCapabilityApproval(RuntimeCompletes):
    async def run(self, context: TaskContext) -> AsyncIterator[RunEvent]:
        yield RunEvent(
            kind=EventKind.APPROVAL_REQUESTED,
            sequence=1,
            run_id=context.run_id,
            actor="operator",
            approval_id="approval_1",
            action="tool_execute",
            reason="requires_user_approval",
        )
        yield RunEvent(
            kind=EventKind.APPROVAL_RESOLVED,
            sequence=2,
            run_id=context.run_id,
            actor="operator",
            approval_id="approval_1",
            decision="approved",
        )
        yield RunEvent(
            kind=EventKind.RUNTIME_COMPLETED,
            sequence=3,
            run_id=context.run_id,
        )


class RecordingHermesAdvisor:
    def __init__(self) -> None:
        self.outcomes: list[HermesRunOutcome] = []

    async def advise(self, **kwargs: object) -> None:
        del kwargs

    async def record_outcome(self, outcome: HermesRunOutcome) -> None:
        self.outcomes.append(outcome)

class RecordingHook:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        *,
        tenant_id: UUID,
        actor_id: UUID | None,
        run_id: UUID,
        status: RunStatus,
        mode: TaskMode | None,
        routing_decision: dict[str, object],
    ) -> None:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "actor_id": actor_id,
                "run_id": run_id,
                "status": status,
                "mode": mode,
                "routing_decision": routing_decision,
            }
        )
        if self.fail:
            raise RuntimeError("hook failed")


class SlowCognitivePipeline:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def process_terminal_run(
        self,
        *,
        tenant_id: UUID,
        run_id: UUID,
        status: RunStatus,
        mode: TaskMode | None,
        routing_decision: dict[str, object],
        actor_id: UUID | None = None,
    ) -> object:
        del tenant_id, actor_id, run_id, status, mode, routing_decision
        self.started.set()
        await self.release.wait()
        return None


@pytest.mark.asyncio
async def test_execute_notifies_terminal_hooks_after_completed_run() -> None:
    repository = ExecutableFakeRepository(
        routing_decision={"source": "evolution", "evolution_run_id": "evolution_1"}
    )
    hook = RecordingHook()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((RuntimeCompletes(),)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
        terminal_run_hooks=(hook,),
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.COMPLETED
    assert hook.calls == [
        {
            "tenant_id": TENANT_ID,
            "actor_id": ACTOR_ID,
            "run_id": repository.run_id,
            "status": RunStatus.COMPLETED,
            "mode": TaskMode.DISPATCH,
            "routing_decision": {"source": "evolution", "evolution_run_id": "evolution_1"},
        }
    ]


@pytest.mark.asyncio
async def test_execute_waits_for_user_approval_after_runtime_artifact_review_request() -> None:
    repository = ExecutableFakeRepository(routing_decision={"source": "video_pipeline"})
    hook = RecordingHook()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((RuntimeRequestsArtifactReview(),)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
        terminal_run_hooks=(hook,),
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.WAITING_APPROVAL
    assert repository.row.status == RunStatus.WAITING_APPROVAL.value
    assert hook.calls == []


@pytest.mark.asyncio
async def test_execute_continues_after_non_artifact_approval_events() -> None:
    repository = ExecutableFakeRepository(routing_decision={"source": "manual"})
    hook = RecordingHook()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((RuntimeCompletesAfterCapabilityApproval(),)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
        terminal_run_hooks=(hook,),
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.COMPLETED
    assert repository.row.status == RunStatus.COMPLETED.value
    assert [event.kind for event in repository.events] == [
        EventKind.APPROVAL_REQUESTED,
        EventKind.APPROVAL_RESOLVED,
        EventKind.RUNTIME_COMPLETED,
    ]
    assert hook.calls == [
        {
            "tenant_id": TENANT_ID,
            "actor_id": ACTOR_ID,
            "run_id": repository.run_id,
            "status": RunStatus.COMPLETED,
            "mode": TaskMode.DISPATCH,
            "routing_decision": {"source": "manual"},
        }
    ]


@pytest.mark.asyncio
async def test_execute_hydrates_persisted_run_artifacts_when_restoring_checkpoint() -> None:
    artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="multimedia_generator",
        content={"text": "Generated image artifact with kilin-ima."},
    )
    repository = ExecutableFakeRepository(
        routing_decision={
            "source": "video_pipeline",
            "artifact_review_feedback": {
                "stage_id": "multimedia_generator_step",
                "artifact_id": str(artifact.id),
                "feedback": "角色脸型和服装不一致，退回重新生成角色参考设定表。",
            },
        },
        raw_artifacts=(artifact.to_payload(),),
    )
    checkpoint = RuntimeCheckpoint(
        id=uuid4(),
        runtime_type="crew",
        runtime_version="7",
        run_id=repository.run_id,
        tenant_id=TENANT_ID,
        mode=TaskMode.DISPATCH,
        state={
            "phase": "completed",
            "artifact_registry": {str(artifact.id): artifact.content_sha256},
        },
    )
    repository.checkpoint = checkpoint
    runtime = RuntimeRequiresHydratedArtifactOnRestore(artifact.id)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((runtime,)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.COMPLETED
    assert runtime.restored == checkpoint
    assert str(artifact.id) in runtime.context_artifact_ids
    event_kinds = [event.kind for event in repository.events]
    assert event_kinds[:2] == [
        EventKind.STEP_RETRYING,
        EventKind.RUNTIME_COMPLETED,
    ]
    assert EventKind.RUNTIME_FAILED not in event_kinds


@pytest.mark.asyncio
async def test_execute_recovers_missing_external_lineage_source_on_checkpoint_restore() -> (
    None
):
    conversation_id = "conv-review-retry"
    prior_run_id = uuid4()
    context_items = (
        ConversationContextItem(
            run_id=prior_run_id,
            request="先生成一个短剧剧本。",
            artifacts=(
                {
                    "producer": "main_agent",
                    "content": {"text": "已完成剧本：角色是年轻医生。"},
                },
            ),
        ),
    )
    repository = ExecutableFakeRepository(
        routing_decision={
            "source": "video_pipeline",
            "conversation_id": conversation_id,
        },
        conversation_context=context_items,
    )
    source_artifact = _conversation_history_artifact(
        run_id=repository.run_id,
        conversation_id=conversation_id,
        current_request=repository.row.request,
        context_items=context_items,
        history_token_budget=4096,
    )
    assert source_artifact is not None
    stale_source_id = uuid4()
    assert stale_source_id != source_artifact.id
    generated = Artifact(
        id=uuid4(),
        type="text",
        producer="multimedia_generator",
        content={"text": "Generated character sheet artifact with kilin-ima."},
        source_ids=(str(stale_source_id),),
    )
    repository.raw_artifact_payloads = (generated.to_payload(),)
    repository.row.routing_decision = {
        "source": "video_pipeline",
        "conversation_id": conversation_id,
        "artifact_review_feedback": {
            "stage_id": "multimedia_generator_step",
            "artifact_id": str(generated.id),
            "feedback": "角色脸型和服装不一致，退回重新生成角色参考设定表。",
        },
    }
    checkpoint = RuntimeCheckpoint(
        id=uuid4(),
        runtime_type="crew",
        runtime_version="7",
        run_id=repository.run_id,
        tenant_id=TENANT_ID,
        mode=TaskMode.DISPATCH,
        state={
            "phase": "completed",
            "artifact_registry": {str(generated.id): generated.content_sha256},
        },
    )
    repository.checkpoint = checkpoint
    runtime = RuntimeRequiresHydratedArtifactOnRestore(generated.id, stale_source_id)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((runtime,)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.COMPLETED
    assert runtime.restored == checkpoint
    assert str(generated.id) in runtime.context_artifact_ids
    assert str(stale_source_id) in runtime.context_artifact_ids


@pytest.mark.asyncio
async def test_execute_loads_current_attachment_artifacts_into_runtime_context() -> None:
    repository = ExecutableFakeRepository(
        routing_decision={
            "source": "manual",
            "conversation_id": "conv-with-file",
            "attachment_ids": ["att_11111111111111111111111111111111"],
        }
    )
    runtime = RuntimeCapturesArtifacts()
    attachment_artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="uploaded_attachment",
        content={
            "text": "附件：screen.png\n类型：image\n说明：用户本轮上传的文件。",
            "attachment_id": "att_11111111111111111111111111111111",
        },
    )

    async def load_attachments(
        *,
        tenant_id: UUID,
        attachment_ids: tuple[str, ...],
    ) -> tuple[Artifact, ...]:
        assert tenant_id == TENANT_ID
        assert attachment_ids == ("att_11111111111111111111111111111111",)
        return (attachment_artifact,)

    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((runtime,)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
        attachment_artifact_loader=load_attachments,
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.COMPLETED
    assert attachment_artifact in runtime.artifacts


@pytest.mark.asyncio
async def test_execute_loads_previous_conversation_attachment_artifacts_into_runtime_context() -> None:
    previous_attachment_id = "att_22222222222222222222222222222222"
    repository = ExecutableFakeRepository(
        routing_decision={
            "source": "manual",
            "conversation_id": "conv-with-file",
            "attachment_ids": [],
        },
        conversation_context=(
            ConversationContextItem(
                run_id=uuid4(),
                request="安装这个 skill 压缩包",
                artifacts=(),
                routing_decision={
                    "conversation_id": "conv-with-file",
                    "attachment_ids": [previous_attachment_id],
                },
            ),
        ),
    )
    runtime = RuntimeCapturesArtifacts()
    previous_attachment_artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="uploaded_attachment",
        content={
            "text": "用户本轮上传了附件，以下内容与当前对话消息直接关联。\n文件名：safe-skill.zip",
            "attachment_id": previous_attachment_id,
        },
    )

    async def load_attachments(
        *,
        tenant_id: UUID,
        attachment_ids: tuple[str, ...],
    ) -> tuple[Artifact, ...]:
        assert tenant_id == TENANT_ID
        assert attachment_ids == (previous_attachment_id,)
        return (previous_attachment_artifact,)

    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((runtime,)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
        attachment_artifact_loader=load_attachments,
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.COMPLETED
    assert len(runtime.artifacts) >= 1
    restored_attachment = runtime.artifacts[0]
    assert restored_attachment.producer == "conversation_uploaded_attachment"
    assert restored_attachment.content["context_scope"] == "previous_conversation_attachment"
    assert "前序交互上传过附件" in str(restored_attachment.content["text"])


@pytest.mark.asyncio
async def test_execute_loads_requested_resource_artifacts_into_runtime_context() -> None:
    repository = ExecutableFakeRepository(
        routing_decision={
            "source": "manual",
            "conversation_id": "conv-with-resource",
            "requested_skills": "cross-system-hub",
            "requested_files": "handoff.md",
        }
    )
    runtime = RuntimeCapturesArtifacts()
    resource_artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="requested_resource_context",
        content={
            "text": "<REQUESTED_RESOURCE_CONTEXT>cross-system-hub handoff.md</REQUESTED_RESOURCE_CONTEXT>",
        },
    )

    async def load_resources(
        *,
        tenant_id: UUID,
        routing_decision: Mapping[str, object],
    ) -> tuple[Artifact, ...]:
        assert tenant_id == TENANT_ID
        assert routing_decision["requested_skills"] == "cross-system-hub"
        assert routing_decision["requested_files"] == "handoff.md"
        return (resource_artifact,)

    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((runtime,)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
        resource_context_loader=load_resources,
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.COMPLETED
    assert resource_artifact in runtime.artifacts


@pytest.mark.asyncio
async def test_terminal_hook_failure_does_not_fail_completed_run() -> None:
    repository = ExecutableFakeRepository(
        routing_decision={"source": "evolution", "evolution_run_id": "evolution_1"}
    )
    hook = RecordingHook(fail=True)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((RuntimeCompletes(),)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
        terminal_run_hooks=(hook,),
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.COMPLETED
    assert hook.calls


@pytest.mark.asyncio
async def test_cognitive_terminal_hook_does_not_block_completed_run() -> None:
    repository = ExecutableFakeRepository(routing_decision={"source": "manual"})
    pipeline = SlowCognitivePipeline()
    hook = CognitiveLearningTerminalHook(pipeline)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((RuntimeCompletes(),)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
        terminal_run_hooks=(hook,),
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.COMPLETED
    await asyncio.wait_for(pipeline.started.wait(), timeout=1)
    assert hook.pending_count == 1
    pipeline.release.set()
    await hook.drain()
    assert hook.pending_count == 0


@pytest.mark.asyncio
async def test_execute_persists_observer_notice_for_capacity_pressure() -> None:
    repository = ExecutableFakeRepository(routing_decision={"source": "manual"})
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((RuntimeReportsCapacityPressure(),)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.FAILED
    notices = [event for event in repository.events if event.kind == "observer.notice"]
    assert len(notices) == 1
    notice = notices[0]
    assert [event.kind for event in repository.events] == [
        EventKind.STEP_FAILED,
        EventKind.RUNTIME_FAILED,
        "observer.notice",
    ]
    assert notice.sequence == 3
    assert notice.payload["trigger"] == "model_capacity_pressure"
    assert notice.payload["action"] == "reschedule_or_reassign_model"
    assert notice.payload["source_sequence"] == 1
    assert "message" not in notice.payload
    assert "prompt" not in notice.payload

@pytest.mark.asyncio
async def test_execute_records_observer_notices_in_hermes_scheduler_outcome() -> None:
    repository = ExecutableFakeRepository(routing_decision={"source": "manual"})
    hermes = RecordingHermesAdvisor()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((RuntimeReportsCapacityPressure(),)),
        router=None,
        task_queue=object(),  # type: ignore[arg-type]
        hermes_advisor=hermes
    )

    submitted = await service.execute(repository.run_id)

    assert submitted.status is RunStatus.FAILED
    assert len(hermes.outcomes) == 1
    outcome = hermes.outcomes[0]
    assert outcome.status is RunStatus.FAILED
    assert outcome.scheduler_notices == (
        {
            "schema_version": 1,
            "trigger": "model_capacity_pressure",
            "action": "reschedule_or_reassign_model",
            "severity": "warning",
            "source_kind": "step.failed",
            "source_sequence": 1,
            "event_count": 1,
            "failure_events": 1,
            "retry_events": 0,
            "message_events": 0,
            "artifact_events": 0,
            "actor": "planner",
        },
    )
    assert "正文" not in repr(outcome.scheduler_notices)
    assert "prompt" not in repr(outcome.scheduler_notices)
