from __future__ import annotations

import asyncio
from collections.abc import Mapping
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

import pytest

from agent_hub.domain.runs import TaskMode
from agent_hub.models.gateway import GatewayCompletion
from agent_hub.models.types import ModelRequest, ModelResponse, TokenUsage, ToolCall
from agent_hub.runtime.contracts import Artifact, EventKind, JsonValue, RunEvent, TaskContext
from agent_hub.runtime.crew.adapter import (
    CrewAgentDefinition,
    CrewDispatchRuntime,
    CrewLLMBridge,
    CrewObjectFactory,
    CrewTaskDefinition,
    RuntimeExecutionError,
    _artifact_final_synthesis_payload,
    _artifact_prompt_payload,
    _artifact_review_packet_payload,
    _normalize_tool_call_arguments,
)
from agent_hub.runtime.crew.plan import AgentSpec, DispatchPlan, DispatchStep

TENANT_ID = UUID("00000000-0000-4000-8000-000000000001")
RUN_ID = UUID("00000000-0000-4000-8000-000000000002")


class UnusedGateway:
    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        return GatewayCompletion(
            response=ModelResponse(text="unused", usage=TokenUsage(1, 1, 2)),
            deployment_id="primary",
            logical_model=request.logical_model,
            provider_id="deepseek",
            provider_model="deepseek/deepseek-v4-flash",
            cost_usd=Decimal(0),
        )


class LargeCandidateGateway:
    def __init__(self) -> None:
        self.large_text = "review candidate body " * 2_000

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        prompt = " ".join(cast(str, message.content) for message in request.messages)
        text = '{"verdict":"approve"}' if "REVIEWER" in prompt else self.large_text
        return GatewayCompletion(
            response=ModelResponse(text=text, usage=TokenUsage(1, 1, 2)),
            deployment_id="primary",
            logical_model=request.logical_model,
            provider_id="deepseek",
            provider_model="deepseek/deepseek-v4-flash",
            cost_usd=Decimal(0),
        )


class ReviewAwareGateway:
    def __init__(self, reviewer_responses: tuple[str, ...] = ('{"verdict":"approve"}',)) -> None:
        self._reviewer_responses = list(reviewer_responses)

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        prompt = " ".join(cast(str, message.content) for message in request.messages)
        if "REVIEWER" in prompt:
            text = self._reviewer_responses.pop(0)
        else:
            text = "draft output " * 200
        return GatewayCompletion(
            response=ModelResponse(text=text, usage=TokenUsage(1, 1, 2)),
            deployment_id="primary",
            logical_model=request.logical_model,
            provider_id="deepseek",
            provider_model="deepseek/deepseek-v4-flash",
            cost_usd=Decimal(0),
        )


class EmptyThenSuccessGateway:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        self.calls += 1
        text = "" if self.calls == 1 else "recovered answer"
        return GatewayCompletion(
            response=ModelResponse(text=text, usage=TokenUsage(1, 1, 2)),
            deployment_id="primary",
            logical_model=request.logical_model,
            provider_id="deepseek",
            provider_model="deepseek/deepseek-v4-flash",
            cost_usd=Decimal(0),
        )


class FailingReviewerStepGateway:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        self.calls.append(request.logical_model)
        if request.logical_model == "deepseek-mutil":
            raise RuntimeExecutionError(
                "model gateway failed: model transport failed "
                "(logical_models=deepseek-mutil; deployments=deepseek-mutil_1)"
            )
        return GatewayCompletion(
            response=ModelResponse(
                text=f"{request.logical_model} usable output",
                usage=TokenUsage(1, 1, 2),
            ),
            deployment_id="primary",
            logical_model=request.logical_model,
            provider_id="deepseek",
            provider_model="deepseek/deepseek-v4-flash",
            cost_usd=Decimal(0),
        )


class FailingGeneration:
    async def execute(
        self,
        step_id: str,
        prompt: str,
        bridge: CrewLLMBridge,
        *,
        agent_id: str | None = None,
        storage_scope: tuple[UUID, UUID],
    ) -> str:
        del step_id, prompt, bridge, agent_id, storage_scope
        raise ValueError("agent identifier must be a safe identifier")


class FailingFactory(CrewObjectFactory):
    def build(
        self,
        agents: tuple[CrewAgentDefinition, ...],
        tasks: tuple[CrewTaskDefinition, ...],
        *,
        share_crew: bool,
        telemetry_disabled: bool,
    ) -> FailingGeneration:
        del agents, tasks, share_crew, telemetry_disabled
        return FailingGeneration()


class TimeoutGeneration:
    async def execute(
        self,
        step_id: str,
        prompt: str,
        bridge: CrewLLMBridge,
        *,
        agent_id: str | None = None,
        storage_scope: tuple[UUID, UUID],
    ) -> str:
        del step_id, prompt, bridge, agent_id, storage_scope
        raise TimeoutError


class TimeoutFactory(CrewObjectFactory):
    def build(
        self,
        agents: tuple[CrewAgentDefinition, ...],
        tasks: tuple[CrewTaskDefinition, ...],
        *,
        share_crew: bool,
        telemetry_disabled: bool,
    ) -> TimeoutGeneration:
        del agents, tasks, share_crew, telemetry_disabled
        return TimeoutGeneration()


class StepTimeoutOnceGeneration:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    async def execute(
        self,
        step_id: str,
        prompt: str,
        bridge: CrewLLMBridge,
        *,
        agent_id: str | None = None,
        storage_scope: tuple[UUID, UUID],
    ) -> str:
        del step_id, agent_id, storage_scope
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            raise TimeoutError
        return await bridge.complete([{"role": "user", "content": prompt}])


class StepTimeoutOnceFactory(CrewObjectFactory):
    def __init__(self) -> None:
        self.generation = StepTimeoutOnceGeneration()

    def build(
        self,
        agents: tuple[CrewAgentDefinition, ...],
        tasks: tuple[CrewTaskDefinition, ...],
        *,
        share_crew: bool,
        telemetry_disabled: bool,
    ) -> StepTimeoutOnceGeneration:
        del agents, tasks, share_crew, telemetry_disabled
        return self.generation


class SlowStepGeneration:
    async def execute(
        self,
        step_id: str,
        prompt: str,
        bridge: CrewLLMBridge,
        *,
        agent_id: str | None = None,
        storage_scope: tuple[UUID, UUID],
    ) -> str:
        del step_id, prompt, bridge, agent_id, storage_scope
        await asyncio.sleep(1)
        return "late"


class SlowStepFactory(CrewObjectFactory):
    def build(
        self,
        agents: tuple[CrewAgentDefinition, ...],
        tasks: tuple[CrewTaskDefinition, ...],
        *,
        share_crew: bool,
        telemetry_disabled: bool,
    ) -> SlowStepGeneration:
        del agents, tasks, share_crew, telemetry_disabled
        return SlowStepGeneration()


class SlowThenFastStepGeneration:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self,
        step_id: str,
        prompt: str,
        bridge: CrewLLMBridge,
        *,
        agent_id: str | None = None,
        storage_scope: tuple[UUID, UUID],
    ) -> str:
        del step_id, prompt, agent_id, storage_scope
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(1)
        return await bridge.complete([{"role": "user", "content": "recover"}])


class SlowThenFastStepFactory(CrewObjectFactory):
    def __init__(self) -> None:
        self.generation = SlowThenFastStepGeneration()

    def build(
        self,
        agents: tuple[CrewAgentDefinition, ...],
        tasks: tuple[CrewTaskDefinition, ...],
        *,
        share_crew: bool,
        telemetry_disabled: bool,
    ) -> SlowThenFastStepGeneration:
        del agents, tasks, share_crew, telemetry_disabled
        return self.generation


class ReviewerTimeoutGeneration:
    async def execute(
        self,
        step_id: str,
        prompt: str,
        bridge: CrewLLMBridge,
        *,
        agent_id: str | None = None,
        storage_scope: tuple[UUID, UUID],
    ) -> str:
        del step_id, storage_scope
        if agent_id == "critic":
            raise TimeoutError
        return await bridge.complete([{"role": "user", "content": prompt}])


class ReviewerTimeoutOnceGeneration:
    def __init__(self) -> None:
        self.review_calls = 0

    async def execute(
        self,
        step_id: str,
        prompt: str,
        bridge: CrewLLMBridge,
        *,
        agent_id: str | None = None,
        storage_scope: tuple[UUID, UUID],
    ) -> str:
        del step_id, storage_scope
        if agent_id == "critic":
            self.review_calls += 1
            if self.review_calls == 1:
                raise TimeoutError
        return await bridge.complete([{"role": "user", "content": prompt}])


class ReviewerTimeoutFactory(CrewObjectFactory):
    def build(
        self,
        agents: tuple[CrewAgentDefinition, ...],
        tasks: tuple[CrewTaskDefinition, ...],
        *,
        share_crew: bool,
        telemetry_disabled: bool,
    ) -> ReviewerTimeoutGeneration:
        del agents, tasks, share_crew, telemetry_disabled
        return ReviewerTimeoutGeneration()


class ReviewerTimeoutOnceFactory(CrewObjectFactory):
    def __init__(self) -> None:
        self.generation = ReviewerTimeoutOnceGeneration()

    def build(
        self,
        agents: tuple[CrewAgentDefinition, ...],
        tasks: tuple[CrewTaskDefinition, ...],
        *,
        share_crew: bool,
        telemetry_disabled: bool,
    ) -> ReviewerTimeoutOnceGeneration:
        del agents, tasks, share_crew, telemetry_disabled
        return self.generation


class CapturingGeneration:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def execute(
        self,
        step_id: str,
        prompt: str,
        bridge: CrewLLMBridge,
        *,
        agent_id: str | None = None,
        storage_scope: tuple[UUID, UUID],
    ) -> str:
        del step_id, agent_id, storage_scope
        self.prompts.append(prompt)
        return await bridge.complete([{"role": "user", "content": prompt}])


class CapturingFactory(CrewObjectFactory):
    def __init__(self) -> None:
        self.generation = CapturingGeneration()

    def build(
        self,
        agents: tuple[CrewAgentDefinition, ...],
        tasks: tuple[CrewTaskDefinition, ...],
        *,
        share_crew: bool,
        telemetry_disabled: bool,
    ) -> CapturingGeneration:
        del agents, tasks, share_crew, telemetry_disabled
        return self.generation


class MultimediaToolGateway:
    def __init__(self, *, legacy_prompt: bool = False, include_legacy_prompt: bool = False) -> None:
        self.legacy_prompt = legacy_prompt
        self.include_legacy_prompt = include_legacy_prompt
        self.requests: list[ModelRequest] = []

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        self.requests.append(request)
        tool_name = request.tools[0].name if request.tools else "generate_multimedia"
        arguments: Mapping[str, JsonValue]
        if self.legacy_prompt:
            arguments = {
                "kind": "image",
                "logical_model": "media_primary",
                "prompt": "生成一张赛博朋克风格海报",
            }
        else:
            arguments = {
                "kind": "image",
                "logical_model": "media_primary",
                "generation_prompt": "生成一张赛博朋克风格海报",
            }
            if self.include_legacy_prompt:
                arguments = dict(arguments)
                arguments["prompt"] = "不应进入运行轨迹的旧字段"
        return GatewayCompletion(
            response=ModelResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        id="call-media",
                        name=tool_name,
                        arguments=arguments,
                    ),
                ),
                usage=TokenUsage(10, 1, 11),
            ),
            deployment_id="primary",
            logical_model=request.logical_model,
            provider_id="deepseek",
            provider_model="deepseek/deepseek-v4-flash",
            cost_usd=Decimal(0),
        )


class MultimediaCapabilities:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Mapping[str, JsonValue]]] = []

    async def execute(  # type: ignore[no-untyped-def]
        self, *, tenant_id, run_id, actor, name, arguments, idempotency_key
    ) -> Mapping[str, JsonValue]:
        del tenant_id, run_id, idempotency_key
        self.calls.append((actor, name, arguments))
        return {
            "job_id": "media-test",
            "kind": arguments["kind"],
            "logical_model": arguments["logical_model"],
            "status": "completed",
            "executor_id": actor,
            "summary": "Generated image artifact with media_primary.",
            "artifacts": (
                {
                    "filename": "poster.png",
                    "mime_type": "image/png",
                    "download_url": "/api/v1/admin/multimedia/jobs/media-test/artifacts/0/download",
                },
            ),
            "presentation": "final_attachment",
        }

    def is_replay_safe(self, name: str) -> bool:
        return name == "generate_multimedia"


class DirectMultimediaCapabilities(MultimediaCapabilities):
    async def default_logical_model_for_multimedia(self, *, tenant_id: UUID, kind: str) -> str:
        assert tenant_id == TENANT_ID
        assert kind in {"image", "video", "audio"}
        return "media_primary"


def _one_step_plan(*, timeout_seconds: float = 60.0) -> DispatchPlan:
    return DispatchPlan(
        agents=(AgentSpec(id="writer", role="writer", goal="Write", logical_model="general"),),
        steps=(
            DispatchStep(
                id="final",
                agent="writer",
                task="Answer",
                final_synthesizer=True,
                token_budget=100,
                timeout_seconds=timeout_seconds,
            ),
        ),
        total_token_budget=100,
        total_timeout_seconds=max(60.0, timeout_seconds * 4),
    )


def _one_step_tool_plan(*, tools: tuple[str, ...], multimedia: bool = False) -> DispatchPlan:
    agent_id = "multimedia_generator" if multimedia else "writer"
    role = "Multimedia Generator" if multimedia else "writer"
    goal = "生成图片和视频产物" if multimedia else "Write"
    task = "生成一张赛博朋克风格海报" if multimedia else "Answer"
    return DispatchPlan(
        agents=(
            AgentSpec(
                id=agent_id,
                role=role,
                goal=goal,
                logical_model="general",
                allowed_tools=tools,
            ),
        ),
        steps=(
            DispatchStep(
                id="final",
                agent=agent_id,
                task=task,
                tools=tools,
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        allowed_tools=tools,
        total_token_budget=100,
    )


def _reviewed_plan() -> DispatchPlan:
    return DispatchPlan(
        agents=(
            AgentSpec(id="writer", role="writer", goal="Write", logical_model="general"),
            AgentSpec(id="critic", role="critic", goal="Review", logical_model="general"),
        ),
        steps=(
            DispatchStep(
                id="draft",
                agent="writer",
                task="Draft",
                reviewer="critic",
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        total_token_budget=200,
    )


def _reviewed_plan_with_retry_budget(reviewer_retries: int = 1) -> DispatchPlan:
    return DispatchPlan(
        agents=(
            AgentSpec(id="writer", role="writer", goal="Write", logical_model="general"),
            AgentSpec(id="critic", role="critic", goal="Review", logical_model="general"),
        ),
        steps=(
            DispatchStep(
                id="draft",
                agent="writer",
                task="Draft",
                reviewer="critic",
                reviewer_retries=reviewer_retries,
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        total_token_budget=200,
    )


def _optional_reviewer_step_plan() -> DispatchPlan:
    return DispatchPlan(
        agents=(
            AgentSpec(id="writer", role="writer", goal="Write", logical_model="general"),
            AgentSpec(
                id="quality_reviewer",
                role="质量审查员",
                goal="Review upstream answer quality",
                logical_model="deepseek-mutil",
            ),
            AgentSpec(
                id="final_writer",
                role="final writer",
                goal="Return final answer",
                logical_model="general",
            ),
        ),
        steps=(
            DispatchStep(
                id="draft",
                agent="writer",
                task="Draft an answer",
                token_budget=100,
            ),
            DispatchStep(
                id="quality_reviewer_step",
                agent="quality_reviewer",
                task="Review the draft and provide quality notes",
                depends_on=("draft",),
                token_budget=100,
            ),
            DispatchStep(
                id="final",
                agent="final_writer",
                task="Return final answer using available upstream artifacts",
                depends_on=("quality_reviewer_step",),
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        total_token_budget=300,
        total_timeout_seconds=300,
    )


def _context(
    *,
    artifacts: tuple[Artifact, ...] = (),
    timeout_seconds: float = 60.0,
    request: str = "Write a short answer",
) -> TaskContext:
    return TaskContext(
        run_id=RUN_ID,
        tenant_id=TENANT_ID,
        mode=TaskMode.DISPATCH,
        request=request,
        artifacts=artifacts,
        timeout_seconds=timeout_seconds,
        token_budget=1000,
    )


def test_openai_tool_call_response_can_have_empty_text() -> None:
    completion = GatewayCompletion(
        response=ModelResponse(
            text="",
            tool_calls=(ToolCall(id="call_1", name="read_context", arguments={"query": "x"}),),
            usage=TokenUsage(10, 1, 11),
        ),
        deployment_id="primary",
        logical_model="general",
        provider_id="deepseek",
        provider_model="deepseek/deepseek-v4-flash",
        cost_usd=Decimal(0),
    )

    response = CrewDispatchRuntime._valid_response(completion)

    assert response.tool_calls[0].name == "read_context"


def test_text_only_empty_model_response_still_fails() -> None:
    completion = GatewayCompletion(
        response=ModelResponse(text="", usage=TokenUsage(10, 0, 10)),
        deployment_id="primary",
        logical_model="general",
        provider_id="deepseek",
        provider_model="deepseek/deepseek-v4-flash",
        cost_usd=Decimal(0),
    )

    with pytest.raises(RuntimeExecutionError, match="model response text is empty"):
        CrewDispatchRuntime._valid_response(completion)


async def _collect(runtime: CrewDispatchRuntime) -> list[RunEvent]:
    return [event async for event in runtime.run(_context())]


async def test_multimedia_final_attachment_tool_uses_safe_arguments_and_finishes_without_text_fallback() -> None:
    capabilities = MultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        MultimediaToolGateway(),
        _one_step_tool_plan(tools=("generate_multimedia",)),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = await _collect(runtime)
    artifacts = tuple(event.artifact for event in events if event.artifact is not None)
    final = next(artifact for artifact in artifacts if artifact.type == "text")

    assert capabilities.calls == [
        (
            "writer",
            "generate_multimedia",
            {
                "kind": "image",
                "logical_model": "media_primary",
                "generation_prompt": "生成一张赛博朋克风格海报",
            },
        )
    ]
    assert "poster.png" in cast(str, final.content["text"])
    assert len([artifact for artifact in artifacts if artifact.type == "model_response"]) == 1


async def test_multimedia_generator_directly_executes_media_tool_without_text_model() -> None:
    class FailingTextGateway:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            self.calls += 1
            raise AssertionError("text gateway must not be called for direct media generation")

    gateway = FailingTextGateway()
    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        gateway,
        _one_step_tool_plan(tools=("generate_multimedia",), multimedia=True),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(request="请生成一张修仙世界女主角照片，清冷仙子气质")
        )
    ]
    artifacts = tuple(event.artifact for event in events if event.artifact is not None)
    final = next(artifact for artifact in artifacts if artifact.type == "text")

    assert gateway.calls == 0
    assert capabilities.calls
    actor, name, arguments = capabilities.calls[0]
    assert actor == "multimedia_generator"
    assert name == "generate_multimedia"
    assert arguments["kind"] == "image"
    assert arguments["logical_model"] == "media_primary"
    assert "修仙世界女主角" in cast(str, arguments["generation_prompt"])
    assert "poster.png" in cast(str, final.content["text"])
    assert (
        "[下载图片：poster.png](/api/v1/admin/multimedia/jobs/media-test/artifacts/0/download)"
        in cast(str, final.content["text"])
    )
    assert any(
        event.kind is EventKind.TOOL_STARTED
        and event.payload.get("direct_dispatch") is True
        for event in events
    )
    created = next(
        event
        for event in events
        if event.kind is EventKind.ARTIFACT_CREATED and event.actor == "multimedia_generator"
    )
    completed = next(
        event
        for event in events
        if event.kind is EventKind.STEP_COMPLETED and event.actor == "multimedia_generator"
    )
    assert created.payload["logical_model"] == "media_primary"
    assert completed.payload["logical_model"] == "media_primary"


@pytest.mark.parametrize(
    ("task_text", "expected_kind"),
    [
        ("给我做一张图片版设定板。", "image"),
        ("出一张赛博朋克产品概念图。", "image"),
        ("生成三张可下载表情包贴纸。", "image"),
        ("做一张商品 3D 渲染图。", "image"),
        ("把这个故事做成 8 秒动画短片成片。", "video"),
        ("为这段开场白合成一段旁白配音。", "audio"),
        ("给品牌发布会做一段 BGM 背景音乐。", "audio"),
    ],
)
async def test_multimedia_generator_direct_kind_inference_covers_business_media_terms(
    task_text: str,
    expected_kind: str,
) -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct media generation")

    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        _one_step_tool_plan(tools=("generate_multimedia",), multimedia=True),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [event async for event in runtime.run(_context(request=task_text))]

    assert events
    assert capabilities.calls[0][2]["kind"] == expected_kind


async def test_multimedia_legacy_prompt_tool_argument_is_normalized_before_evidence() -> None:
    capabilities = MultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        MultimediaToolGateway(legacy_prompt=True),
        _one_step_tool_plan(tools=("generate_multimedia",)),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = await _collect(runtime)
    artifacts = tuple(event.artifact for event in events if event.artifact is not None)
    model_artifact = next(artifact for artifact in artifacts if artifact.type == "model_response")
    tool_call = cast(tuple[Mapping[str, JsonValue], ...], model_artifact.content["tool_calls"])[0]
    arguments = cast(Mapping[str, JsonValue], tool_call["arguments"])

    assert "prompt" not in arguments
    assert arguments["generation_prompt"] == "生成一张赛博朋克风格海报"
    assert capabilities.calls == [
        (
            "writer",
            "generate_multimedia",
            {
                "kind": "image",
                "logical_model": "media_primary",
                "generation_prompt": "生成一张赛博朋克风格海报",
            },
        )
    ]


async def test_multimedia_mixed_prompt_fields_drop_legacy_prompt_before_evidence() -> None:
    capabilities = MultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        MultimediaToolGateway(include_legacy_prompt=True),
        _one_step_tool_plan(tools=("generate_multimedia",)),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = await _collect(runtime)
    artifacts = tuple(event.artifact for event in events if event.artifact is not None)
    model_artifact = next(artifact for artifact in artifacts if artifact.type == "model_response")
    tool_call = cast(tuple[Mapping[str, JsonValue], ...], model_artifact.content["tool_calls"])[0]
    arguments = cast(Mapping[str, JsonValue], tool_call["arguments"])

    assert "prompt" not in arguments
    assert arguments["generation_prompt"] == "生成一张赛博朋克风格海报"
    assert capabilities.calls == [
        (
            "writer",
            "generate_multimedia",
            {
                "kind": "image",
                "logical_model": "media_primary",
                "generation_prompt": "生成一张赛博朋克风格海报",
            },
        )
    ]


def test_multimedia_empty_generation_prompt_falls_back_to_legacy_prompt_without_sensitive_key() -> None:
    arguments = _normalize_tool_call_arguments(
        "generate_multimedia",
        {
            "kind": "image",
            "logical_model": "media_primary",
            "generation_prompt": " ",
            "prompt": "生成一张赛博朋克风格海报",
        },
    )

    assert "prompt" not in arguments
    assert arguments["generation_prompt"] == "生成一张赛博朋克风格海报"


async def test_dispatch_framework_failure_records_safe_root_cause() -> None:
    runtime = CrewDispatchRuntime(
        UnusedGateway(),
        _one_step_plan(),
        crew_factory=FailingFactory(),
    )
    events: list[RunEvent] = []

    with pytest.raises(RuntimeExecutionError) as caught:
        async for event in runtime.run(_context()):
            events.append(event)

    expected = "CrewAI step execution failed: agent identifier must be a safe identifier"
    assert str(caught.value) == expected
    assert events[-1].kind is EventKind.RUNTIME_FAILED
    assert events[-1].reason == expected
    assert any(event.kind is EventKind.STEP_FAILED and event.reason == expected for event in events)


async def test_dispatch_framework_timeout_names_the_step_and_actor() -> None:
    runtime = CrewDispatchRuntime(
        UnusedGateway(),
        _one_step_plan(),
        crew_factory=TimeoutFactory(),
    )
    events: list[RunEvent] = []

    with pytest.raises(RuntimeExecutionError) as caught:
        async for event in runtime.run(_context()):
            events.append(event)

    expected = "CrewAI step timed out: step=final actor=writer"
    assert str(caught.value) == expected
    assert events[-1].kind is EventKind.RUNTIME_FAILED
    assert events[-1].reason == expected
    assert events[-1].payload["error_code"] == "crew.step_timeout"
    assert events[-1].payload["step_id"] == "final"
    assert events[-1].payload["actor"] == "writer"
    assert any(
        event.kind is EventKind.STEP_FAILED
        and event.reason == expected
        and event.payload["error_code"] == "crew.step_timeout"
        and event.payload["step_id"] == "final"
        and event.payload["actor"] == "writer"
        for event in events
    )


async def test_dispatch_step_timeout_retries_with_compact_recovery_prompt() -> None:
    artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="researcher",
        content={"text": "large source context " * 500},
    )
    factory = StepTimeoutOnceFactory()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        _one_step_plan(),
        crew_factory=factory,
    )

    events = [event async for event in runtime.run(_context(artifacts=(artifact,)))]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    assert factory.generation.calls == 2
    retry = next(event for event in events if event.kind is EventKind.STEP_RETRYING)
    assert retry.actor == "writer"
    assert retry.reason == "step execution timed out; retrying with compact recovery"
    assert retry.payload["attempt"] == 2
    assert retry.payload["strategy"] == "compact_retry"
    assert retry.payload["fallback_policy"] == "fail_if_retry_exhausted"
    assert retry.payload["error_code"] == "crew.step_timeout"
    assert retry.payload["step_id"] == "final"
    assert retry.payload["actor"] == "writer"
    assert "compact_retry" in factory.generation.prompts[1]
    assert len(factory.generation.prompts[1].encode("utf-8")) < len(
        factory.generation.prompts[0].encode("utf-8")
    )


async def test_dispatch_step_timeout_recovery_keeps_each_attempt_on_step_deadline() -> None:
    factory = SlowThenFastStepFactory()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        _one_step_plan(timeout_seconds=0.05),
        crew_factory=factory,
    )

    events = [event async for event in runtime.run(_context(timeout_seconds=1.0))]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    assert factory.generation.calls == 2
    retry = next(event for event in events if event.kind is EventKind.STEP_RETRYING)
    assert retry.payload["timeout_policy"] == "use_remaining_step_budget"


async def test_dispatch_step_empty_model_response_retries_before_failing() -> None:
    gateway = EmptyThenSuccessGateway()
    runtime = CrewDispatchRuntime(
        gateway,
        _one_step_plan(),
        crew_factory=CapturingFactory(),
    )

    events = [event async for event in runtime.run(_context())]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    assert gateway.calls == 2
    retry = next(event for event in events if event.kind is EventKind.STEP_RETRYING)
    assert retry.actor == "writer"
    assert retry.reason == "model returned empty response; retrying with explicit output request"
    assert retry.payload["strategy"] == "empty_response_retry"
    assert retry.payload["error_code"] == "model.empty_response"


async def test_optional_reviewer_agent_step_model_failure_is_skipped_with_model_context() -> None:
    gateway = FailingReviewerStepGateway()
    runtime = CrewDispatchRuntime(
        gateway,
        _optional_reviewer_step_plan(),
        crew_factory=CapturingFactory(),
    )

    events = [event async for event in runtime.run(_context())]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    assert "deepseek-mutil" in gateway.calls
    skipped = next(
        event
        for event in events
        if event.kind is EventKind.STEP_COMPLETED and event.actor == "quality_reviewer"
    )
    assert skipped.payload["review_status"] == "skipped"
    assert skipped.payload["fallback_policy"] == "skip_optional_review_step"
    assert skipped.payload["error_code"] == "model.provider_transport_failed"
    assert skipped.payload["logical_models"] == "deepseek-mutil"
    assert skipped.payload["deployments"] == "deepseek-mutil_1"
    created = next(
        event
        for event in events
        if event.kind is EventKind.ARTIFACT_CREATED
        and event.actor == "quality_reviewer"
        and event.artifact is not None
        and event.artifact.type == "text"
    )
    artifact = created.artifact
    assert artifact is not None
    assert artifact.provenance is not None
    assert artifact.provenance.deployment_id == "skipped-optional-review"
    assert "general usable output" in cast(str, artifact.content["text"])


async def test_dispatch_step_retry_is_suppressed_when_runtime_budget_is_exhausted() -> None:
    runtime = CrewDispatchRuntime(
        UnusedGateway(),
        _one_step_plan(),
        crew_factory=SlowStepFactory(),
    )
    context = _context(timeout_seconds=0.05)
    events: list[RunEvent] = []

    with pytest.raises(RuntimeExecutionError) as caught:
        async for event in runtime.run(context):
            events.append(event)

    assert str(caught.value) in {
        "dispatch deadline exhausted",
        "CrewAI step timed out: step=final actor=writer",
    }
    assert not any(event.kind is EventKind.STEP_RETRYING for event in events)
    assert events[-1].kind is EventKind.RUNTIME_FAILED


async def test_final_step_prompt_uses_review_packet_for_source_artifacts() -> None:
    original_text = "large upstream source " * 2_000
    artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="writer",
        content={"text": original_text},
    )
    factory = CapturingFactory()
    runtime = CrewDispatchRuntime(
        UnusedGateway(),
        _one_step_plan(),
        crew_factory=factory,
    )

    events = [
        event
        async for event in runtime.run(_context(artifacts=(artifact,)))
        if event.kind is EventKind.STEP_COMPLETED
    ]

    assert events
    prompt = factory.generation.prompts[0]
    assert "artifact_review_packet" in prompt
    assert "large upstream source large upstream source" in prompt
    assert original_text not in prompt
    assert '"content"' not in prompt


async def test_reviewer_prompt_uses_review_packet_for_candidate_artifact() -> None:
    gateway = LargeCandidateGateway()
    factory = CapturingFactory()
    runtime = CrewDispatchRuntime(
        gateway,
        _reviewed_plan(),
        crew_factory=factory,
    )

    events = [event async for event in runtime.run(_context())]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    reviewer_prompt = next(prompt for prompt in factory.generation.prompts if "REVIEWER" in prompt)
    assert "artifact_review_packet" in reviewer_prompt
    assert "review candidate body review candidate body" in reviewer_prompt
    assert gateway.large_text not in reviewer_prompt
    assert '"content"' not in reviewer_prompt


async def test_reviewer_timeout_is_recorded_and_dispatch_continues() -> None:
    runtime = CrewDispatchRuntime(
        UnusedGateway(),
        _reviewed_plan(),
        crew_factory=ReviewerTimeoutFactory(),
    )

    events = [event async for event in runtime.run(_context())]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    review = next(event for event in events if event.kind is EventKind.REVIEW_COMPLETED)
    assert review.payload["verdict"] == "approve"
    assert review.payload["review_status"] == "timeout_skipped"
    assert review.payload["error_code"] == "crew.step_timeout"
    assert review.payload["step_id"] == "draft.review"
    assert review.payload["actor"] == "critic"


async def test_reviewer_timeout_retries_before_skip() -> None:
    factory = ReviewerTimeoutOnceFactory()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        _reviewed_plan_with_retry_budget(),
        crew_factory=factory,
    )

    events = [event async for event in runtime.run(_context())]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    assert factory.generation.review_calls == 2
    retry = next(event for event in events if event.kind is EventKind.STEP_RETRYING)
    assert retry.actor == "critic"
    assert retry.reason == "reviewer execution failed; retrying review"
    assert retry.payload["review_attempt"] == 2
    review = next(event for event in events if event.kind is EventKind.REVIEW_COMPLETED)
    assert review.payload["verdict"] == "approve"
    assert "review_status" not in review.payload


async def test_reviewer_invalid_json_retries_with_optimized_prompt_before_skip() -> None:
    factory = CapturingFactory()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(("not json", '{"verdict":"approve"}')),
        _reviewed_plan_with_retry_budget(),
        crew_factory=factory,
    )

    events = [event async for event in runtime.run(_context())]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    reviewer_prompts = [prompt for prompt in factory.generation.prompts if "REVIEWER" in prompt]
    assert len(reviewer_prompts) == 2
    assert "Previous reviewer failure" in reviewer_prompts[1]
    assert "Return strict JSON only" in reviewer_prompts[1]
    assert len(reviewer_prompts[1].encode("utf-8")) < len(reviewer_prompts[0].encode("utf-8"))
    retry = next(event for event in events if event.kind is EventKind.STEP_RETRYING)
    assert retry.actor == "critic"
    assert retry.payload["strategy"] == "optimized_retry"
    review = next(event for event in events if event.kind is EventKind.REVIEW_COMPLETED)
    assert review.payload["verdict"] == "approve"


def test_artifact_prompt_payload_truncates_large_text_without_mutating_artifact() -> None:
    original_text = "长文本" * 1_000
    artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="writer",
        content={"text": original_text},
    )

    payload = _artifact_prompt_payload(artifact, max_text_bytes=256)

    content = payload["content"]
    assert isinstance(content, dict)
    text = content["text"]
    assert isinstance(text, str)
    assert len(text.encode("utf-8")) <= 256
    assert "[truncated:" in text
    assert artifact.content["text"] == original_text


def test_final_synthesis_payload_uses_smaller_summary_without_mutating_artifact() -> None:
    original_text = "final synthesis source " * 2_000
    artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="planner",
        content={"text": original_text},
    )

    payload = _artifact_final_synthesis_payload(artifact)

    content = payload["content"]
    assert isinstance(content, dict)
    text = content["text"]
    assert isinstance(text, str)
    assert len(text.encode("utf-8")) <= 2_048
    assert "[truncated:" in text
    assert payload["synthesis_input"] == {
        "mode": "summary",
        "note": "Full artifact is stored separately; this final synthesis input is bounded to keep production model calls reliable.",
    }
    assert artifact.content["text"] == original_text


def test_artifact_review_packet_payload_uses_bounded_preview_without_full_text() -> None:
    original_text = "review source " * 2_000
    source_id = str(uuid4())
    artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="writer",
        content={"text": original_text, "risk": "low"},
        source_ids=(source_id,),
    )

    payload = _artifact_review_packet_payload(artifact)

    assert "content" not in payload
    packet = payload["artifact_review_packet"]
    assert isinstance(packet, dict)
    assert packet["producer"] == "writer"
    assert packet["type"] == "text"
    assert packet["source_ids"] == [source_id]
    assert packet["content_keys"] == ["risk", "text"]
    assert isinstance(packet["preview"], str)
    assert len(packet["preview"].encode("utf-8")) <= 1_200
    assert packet["preview"] != original_text
    assert "[truncated:" in packet["preview"]
    assert artifact.content["text"] == original_text
