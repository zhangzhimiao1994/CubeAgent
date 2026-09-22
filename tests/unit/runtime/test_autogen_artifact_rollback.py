from __future__ import annotations

import asyncio
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from agent_hub.domain.runs import TaskMode
from agent_hub.models.gateway import GatewayCompletion
from agent_hub.models.litellm_client import ModelTransportError
from agent_hub.models.types import ModelRequest, ModelResponse, TokenUsage
from agent_hub.runtime.artifacts import (
    ArtifactReference,
    ArtifactRepositoryError,
    InMemoryArtifactRepository,
)
from agent_hub.runtime.autogen.adapter import (
    AutoGenDiscussionRuntime,
    DiscussionParticipant,
    DiscussionPlan,
    RuntimeExecutionError,
    _can_complete_with_partial_discussion,
    _discussion_has_enough_distinct_outputs,
    _should_fail_on_autogen_cleanup,
)
from agent_hub.runtime.contracts import Artifact, EventKind, RunEvent, TaskContext

TENANT_ID = UUID("00000000-0000-4000-8000-000000000041")
RUN_ID = UUID("00000000-0000-4000-8000-000000000042")


class UnusedGateway:
    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        return GatewayCompletion(
            response=ModelResponse(text="unused", usage=TokenUsage(1, 1, 2)),
            deployment_id="primary",
            logical_model=request.logical_model,
            provider_id="deepseek",
            provider_model="deepseek-v4-flash",
            cost_usd=Decimal(0),
        )


class ScriptedGateway:
    def __init__(self, replies: list[tuple[str, int, Decimal | None]]) -> None:
        self.replies = list(replies)
        self.requests: list[ModelRequest] = []

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        self.requests.append(request)
        text, tokens, cost = self.replies.pop(0)
        return GatewayCompletion(
            response=ModelResponse(
                text=text,
                usage=TokenUsage(
                    prompt_tokens=max(tokens - 1, 0),
                    completion_tokens=min(tokens, 1),
                    total_tokens=tokens,
                ),
            ),
            deployment_id="shared",
            logical_model=request.logical_model,
            provider_id="openai",
            provider_model="openai/test",
            cost_usd=cost,
        )


class FailingGateway:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        self.requests.append(request)
        raise ModelTransportError("Authorization: Bearer sk-secret", status_code=401)


def _runtime(repository: InMemoryArtifactRepository) -> AutoGenDiscussionRuntime:
    return AutoGenDiscussionRuntime(
        UnusedGateway(),
        DiscussionPlan(
            participants=(
                DiscussionParticipant(
                    id="analyst", role="Analyst", goal="Analyze", logical_model="general"
                ),
                DiscussionParticipant(
                    id="critic", role="Critic", goal="Critique", logical_model="general"
                ),
            ),
            selector_model="general",
        ),
        artifact_repository=repository,
    )


def _scripted_runtime(gateway: object) -> AutoGenDiscussionRuntime:
    return AutoGenDiscussionRuntime(
        gateway,  # type: ignore[arg-type]
        DiscussionPlan(
            participants=(
                DiscussionParticipant(
                    id="analyst", role="Analyst", goal="Analyze", logical_model="general"
                ),
                DiscussionParticipant(
                    id="critic", role="Critic", goal="Critique", logical_model="general"
                ),
            ),
            selector_model="general",
            max_turns=4,
            wall_time_seconds=5.0,
            token_budget=100,
            cost_budget_usd=Decimal(1),
            consensus_votes=2,
        ),
    )


def _context() -> TaskContext:
    return TaskContext(
        run_id=RUN_ID,
        tenant_id=TENANT_ID,
        mode=TaskMode.DISCUSS,
        request="Compare options.",
    )


def _artifact() -> Artifact:
    return Artifact(id=uuid4(), type="text", producer="analyst", content={"text": "safe"})


async def _collect(runtime: AutoGenDiscussionRuntime, ctx: TaskContext) -> list[RunEvent]:
    return [event async for event in runtime.run(ctx)]


def test_partial_discussion_can_complete_after_late_model_gateway_failure() -> None:
    artifact = _artifact()

    assert _can_complete_with_partial_discussion(
        (artifact,), "model gateway failed: model transport failed"
    )
    assert not _can_complete_with_partial_discussion(
        (), "model gateway failed: model transport failed"
    )
    assert not _can_complete_with_partial_discussion((artifact,), "discussion_failed")


def test_discussion_soft_completion_requires_distinct_participants() -> None:
    assert not _discussion_has_enough_distinct_outputs(
        (
            Artifact(id=uuid4(), type="text", producer="analyst", content={"text": "A"}),
            Artifact(id=uuid4(), type="text", producer="critic", content={"text": "B"}),
        ),
        participant_count=2,
        consensus_votes=2,
    )
    assert _discussion_has_enough_distinct_outputs(
        (
            Artifact(id=uuid4(), type="text", producer="analyst", content={"text": "A"}),
            Artifact(id=uuid4(), type="text", producer="critic", content={"text": "B"}),
            Artifact(id=uuid4(), type="text", producer="moderator", content={"text": "C"}),
        ),
        participant_count=6,
        consensus_votes=2,
    )
    assert not _discussion_has_enough_distinct_outputs(
        (
            Artifact(id=uuid4(), type="text", producer="analyst", content={"text": "A"}),
            Artifact(id=uuid4(), type="text", producer="analyst", content={"text": "B"}),
        ),
        participant_count=2,
        consensus_votes=2,
    )


def test_autogen_cleanup_failure_does_not_override_usable_discussion_output() -> None:
    assert not _should_fail_on_autogen_cleanup(
        cleanup_failed=True,
        framework_failed=False,
        framework_timed_out=False,
        message_artifacts=(_artifact(),),
    )
    assert _should_fail_on_autogen_cleanup(
        cleanup_failed=True,
        framework_failed=False,
        framework_timed_out=False,
        message_artifacts=(),
    )
    assert not _should_fail_on_autogen_cleanup(
        cleanup_failed=True,
        framework_failed=True,
        framework_timed_out=False,
        message_artifacts=(),
    )


async def test_autogen_abort_artifact_write_returns_repository_result() -> None:
    repository = InMemoryArtifactRepository()
    runtime = _runtime(repository)
    artifact = _artifact()
    reference = ArtifactReference(id=artifact.id, sha256=artifact.content_sha256)
    first_owner = uuid4()
    second_owner = uuid4()

    await repository.put(TENANT_ID, RUN_ID, artifact, write_id=first_owner)
    await repository.put(TENANT_ID, RUN_ID, artifact, write_id=second_owner)
    runtime._pending_artifact_writes[first_owner] = reference

    assert await runtime._abort_artifact_write(_context(), first_owner) is False


async def test_autogen_store_artifact_preserves_original_error_when_rollback_fails() -> None:
    class FailingRollbackRepository(InMemoryArtifactRepository):
        async def put(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            raise ArtifactRepositoryError("artifact repository capacity exceeded")

        async def abort_write(self, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
            del args, kwargs
            return False

    runtime = _runtime(FailingRollbackRepository())
    runtime._cancel_event = asyncio.Event()

    with pytest.raises(RuntimeExecutionError) as caught:
        await runtime._store_artifact(_context(), _artifact())

    assert (
        str(caught.value)
        == "artifact rollback failed after artifact repository capacity exceeded"
    )


async def test_discussion_gateway_failure_emits_readable_summary_before_failure() -> None:
    events = await _collect(_scripted_runtime(FailingGateway()), _context())

    completed = next(event for event in events if event.kind == "discussion.completed")
    summary = completed.payload["summary"]
    assert isinstance(summary, str)
    assert "讨论阶段未能完成" in summary
    assert completed.payload["reason"] == "model gateway failed: model transport failed (status=401)"
    assert events[-1].kind is EventKind.RUNTIME_FAILED
    assert events[-1].reason == "model gateway failed: model transport failed (status=401)"


async def test_discussion_retries_empty_model_response_before_continuing() -> None:
    gateway = ScriptedGateway(
        [
            ("", 1, Decimal("0.01")),
            ("analyst", 1, Decimal("0.01")),
            ("Facts are A.", 2, Decimal("0.01")),
            ("critic", 1, Decimal("0.01")),
            ("[COMPLETE] Facts are verified.", 2, Decimal("0.01")),
        ]
    )

    events = await _collect(_scripted_runtime(gateway), _context())

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    assert events[-1].reason == "explicit_completion"
    assert "previous model response was empty" in str(gateway.requests[1].messages[-1].content).casefold()
    assert [event.actor for event in events if event.kind is EventKind.MESSAGE_CREATED] == [
        "analyst",
        "critic",
    ]
