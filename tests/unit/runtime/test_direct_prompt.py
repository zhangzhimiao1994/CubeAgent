from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest

from agent_hub.domain.runs import TaskMode
from agent_hub.models.gateway import GatewayCompletion
from agent_hub.models.types import ModelRequest, ModelResponse, TokenUsage
from agent_hub.runtime.contracts import Artifact, EventKind, JsonValue, TaskContext
from agent_hub.runtime.direct import DirectRuntime


class UnusedGateway:
    pass


class UsageLessTextGateway:
    def __init__(self, text: str = "这是一个有效的直连回答。") -> None:
        self.text = text
        self.requests: list[ModelRequest] = []

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        self.requests.append(request)
        return GatewayCompletion(
            response=ModelResponse(text=self.text, usage=None),
            deployment_id="main_deployment",
            logical_model=request.logical_model,
            provider_id="test",
            provider_model="test/model",
        )


class InvalidUsageGateway:
    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        return GatewayCompletion(
            response=ModelResponse(
                text="这是一个有效文本，但 usage 不可信。",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=11),
            ),
            deployment_id="main_deployment",
            logical_model=request.logical_model,
            provider_id="test",
            provider_model="test/model",
        )


def test_direct_prompt_truncates_large_artifact_text_for_capacity_estimation() -> None:
    original_text = "长文本" * 1_000
    artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="planner",
        content={"text": original_text},
    )
    context = TaskContext(
        run_id=uuid4(),
        tenant_id=uuid4(),
        mode=TaskMode.DIRECT,
        request="Synthesize the artifacts.",
        artifacts=(artifact,),
        token_budget=1_000_000,
    )
    runtime = DirectRuntime(UnusedGateway(), logical_model="main")  # type: ignore[arg-type]

    request = runtime._build_request(context).request

    assert request is not None
    user_content = request.messages[-1].content
    assert isinstance(user_content, str)
    assert "[truncated:" in user_content
    assert request.max_output_tokens <= 8192
    assert len(user_content.encode("utf-8")) < len(original_text.encode("utf-8"))
    assert artifact.content["text"] == original_text


def test_direct_prompt_includes_bounded_hermes_memory_context() -> None:
    routing_decision: dict[str, JsonValue] = {
        "hermes": {
            "injected_memories": (
                {
                    "summary": "reviewer 超时时先压缩上下文再分块审查。",
                    "memory_type": "error_handling",
                    "target": "reviewer",
                    "reason": "命中 reviewer 超时经验",
                },
            )
        }
    }
    context = TaskContext(
        run_id=uuid4(),
        tenant_id=uuid4(),
        mode=TaskMode.DIRECT,
        request="审查脚本",
        artifacts=(),
        timeout_seconds=60,
        token_budget=10_000,
        routing_decision=routing_decision,
    )
    runtime = DirectRuntime(UnusedGateway(), logical_model="main")  # type: ignore[arg-type]

    prompt = runtime._build_prompt(context)

    assert prompt.messages is not None
    serialized = "\n".join(cast(str, message.content) for message in prompt.messages)
    assert "HERMES_MEMORY_CONTEXT" in serialized
    assert "reviewer 超时时先压缩上下文再分块审查" in serialized


@pytest.mark.asyncio
async def test_direct_runtime_completes_when_gateway_omits_usage_for_bounded_text() -> None:
    gateway = UsageLessTextGateway()
    runtime = DirectRuntime(gateway, logical_model="main")

    events = [
        event
        async for event in runtime.run(
            TaskContext(
                run_id=uuid4(),
                tenant_id=uuid4(),
                mode=TaskMode.DIRECT,
                request="给出一个简短回答",
                token_budget=20_000,
            )
        )
    ]

    assert [request.logical_model for request in gateway.requests] == ["main"]
    assert any(event.kind is EventKind.ARTIFACT_CREATED for event in events)
    assert events[-1].kind is EventKind.RUNTIME_COMPLETED


@pytest.mark.asyncio
async def test_direct_runtime_uses_request_budget_when_usage_less_chinese_text_is_long() -> None:
    gateway = UsageLessTextGateway("中文长回答" * 1_000)
    runtime = DirectRuntime(gateway, logical_model="main")

    events = [
        event
        async for event in runtime.run(
            TaskContext(
                run_id=uuid4(),
                tenant_id=uuid4(),
                mode=TaskMode.DIRECT,
                request="给出一个中文长回答",
                token_budget=20_000,
            )
        )
    ]

    artifact_event = next(event for event in events if event.kind is EventKind.ARTIFACT_CREATED)
    assert artifact_event.artifact is not None
    assert artifact_event.artifact.content["text"] == "中文长回答" * 1_000
    assert events[-1].kind is EventKind.RUNTIME_COMPLETED


@pytest.mark.asyncio
async def test_direct_runtime_completes_with_conservative_budget_when_provider_usage_is_inconsistent() -> None:
    runtime = DirectRuntime(InvalidUsageGateway(), logical_model="main")

    events = [
        event
        async for event in runtime.run(
            TaskContext(
                run_id=uuid4(),
                tenant_id=uuid4(),
                mode=TaskMode.DIRECT,
                request="给出一个简短回答",
                token_budget=20_000,
            )
        )
    ]

    artifact_event = next(event for event in events if event.kind is EventKind.ARTIFACT_CREATED)
    assert artifact_event.artifact is not None
    assert artifact_event.artifact.content["text"] == "这是一个有效文本，但 usage 不可信。"
    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
