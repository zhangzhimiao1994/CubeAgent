# mypy: disable-error-code="index, operator, union-attr, dict-item, call-overload, arg-type"

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

import pytest

from agent_hub.domain.runs import TaskMode
from agent_hub.models.gateway import GatewayCompletion, ModelGatewayError
from agent_hub.models.types import ModelRequest, ModelResponse, TokenUsage, ToolCall
from agent_hub.runtime.artifacts import InMemoryArtifactRepository
from agent_hub.runtime.contracts import (
    Artifact,
    EventKind,
    JsonValue,
    RunEvent,
    RuntimeCheckpoint,
    TaskContext,
)
from agent_hub.runtime.crew import adapter as adapter_module
from agent_hub.runtime.crew.adapter import (
    CapabilityOutcomeUncertain,
    CrewAgentDefinition,
    CrewDispatchRuntime,
    CrewLLMBridge,
    CrewObjectFactory,
    CrewTaskDefinition,
    RuntimeExecutionError,
    _artifact_final_synthesis_payload,
    _artifact_prompt_payload,
    _artifact_review_feedback_from_routing,
    _artifact_review_feedback_text,
    _artifact_review_items_payload,
    _artifact_review_packet_payload,
    _direct_compose_video_arguments,
    _direct_full_production_asset_labels,
    _direct_full_production_asset_prompt_specs,
    _direct_full_production_asset_prompts,
    _direct_multimedia_generation_prompt,
    _direct_multimedia_retry_selection,
    _fallback_review_response_from_text,
    _final_attachment_summary,
    _full_production_asset_character_targets,
    _normalize_compose_video_arguments_with_sources,
    _normalize_tool_call_arguments,
    _prune_invalidated_artifact_lineage,
    _usable_file_artifacts_payload,
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


class DocumentToolGateway:
    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        return GatewayCompletion(
            response=ModelResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="call-docx",
                        name="document.generate_docx",
                        arguments={
                            "title": "Long Report",
                            "filename": "long-report.docx",
                            "sections": (
                                {
                                    "heading": "Summary",
                                    "paragraphs": ("Recovered compact document output.",),
                                },
                            ),
                            "presentation": "final_attachment",
                        },
                    ),
                ),
                usage=TokenUsage(1, 1, 2),
            ),
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


class ControlCharsThenSuccessGateway:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        self.calls += 1
        text = "\x00\u200b\r" if self.calls == 1 else "recovered answer"
        return GatewayCompletion(
            response=ModelResponse(text=text, usage=TokenUsage(1, 1, 2)),
            deployment_id="primary",
            logical_model=request.logical_model,
            provider_id="deepseek",
            provider_model="deepseek/deepseek-v4-flash",
            cost_usd=Decimal(0),
        )


class EmptyErrorThenSuccessGateway:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
        self.calls += 1
        if self.calls == 1:
            raise ModelGatewayError(
                "model response text is empty",
                logical_models=("qwen",),
                deployments=("qwen_1",),
            )
        return GatewayCompletion(
            response=ModelResponse(text="recovered answer", usage=TokenUsage(1, 1, 2)),
            deployment_id="qwen_1",
            logical_model=request.logical_model,
            provider_id="qwen",
            provider_model="qwen/qwen-max",
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


class StepTimeoutAfterModelCallGeneration:
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
            await bridge.complete([{"role": "user", "content": prompt}])
            raise TimeoutError
        return await bridge.complete([{"role": "user", "content": prompt}])


class StepTimeoutAfterModelCallFactory(CrewObjectFactory):
    def __init__(self) -> None:
        self.generation = StepTimeoutAfterModelCallGeneration()

    def build(
        self,
        agents: tuple[CrewAgentDefinition, ...],
        tasks: tuple[CrewTaskDefinition, ...],
        *,
        share_crew: bool,
        telemetry_disabled: bool,
    ) -> StepTimeoutAfterModelCallGeneration:
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
        raw_count = arguments.get("artifact_count")
        artifact_count = raw_count if type(raw_count) is int and raw_count > 0 else 1
        return {
            "job_id": "media-test",
            "kind": arguments["kind"],
            "logical_model": arguments["logical_model"],
            "status": "completed",
            "executor_id": actor,
            "summary": (
                "Generated image artifact with media_primary."
                if artifact_count == 1
                else f"Generated {artifact_count} image artifacts with media_primary."
            ),
            "artifacts": tuple(
                {
                    "filename": f"poster-{index}.png" if artifact_count > 1 else "poster.png",
                    "mime_type": "image/png",
                    "download_url": (
                        f"/api/v1/admin/multimedia/jobs/media-test/artifacts/{index}/download"
                    ),
                }
                for index in range(artifact_count)
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


class SlowDirectMultimediaCapabilities(DirectMultimediaCapabilities):
    async def execute(  # type: ignore[no-untyped-def]
        self, *, tenant_id, run_id, actor, name, arguments, idempotency_key
    ) -> Mapping[str, JsonValue]:
        await asyncio.sleep(0.04)
        return await super().execute(
            tenant_id=tenant_id,
            run_id=run_id,
            actor=actor,
            name=name,
            arguments=arguments,
            idempotency_key=idempotency_key,
        )


class ReviewedAssetPackCapabilities(DirectMultimediaCapabilities):
    def __init__(
        self,
        *,
        failed_labels: tuple[str, ...] = ("动作资产",),
        job_id: str = "media-test",
    ) -> None:
        super().__init__()
        self.failed_labels = set(failed_labels)
        self.job_id = job_id

    async def execute(  # type: ignore[no-untyped-def]
        self, *, tenant_id, run_id, actor, name, arguments, idempotency_key
    ) -> Mapping[str, JsonValue]:
        self.calls.append((actor, name, arguments))
        labels = tuple(
            str(label)
            for label in arguments.get("artifact_labels", ())
            if isinstance(label, str)
        )
        artifacts: list[Mapping[str, JsonValue]] = []
        for index, label in enumerate(labels):
            failed = label in self.failed_labels
            artifacts.append(
                {
                    "filename": f"asset-{index}.png",
                    "mime_type": "image/png",
                    "download_url": (
                        f"/api/v1/admin/multimedia/jobs/{self.job_id}/artifacts/{index}/download"
                    ),
                    "label": label,
                    "visual_review": {
                        "passed": not failed,
                        "summary": f"{label}不符合资产图要求" if failed else "资产合格",
                        "issues": ("不符合资产图要求",) if failed else (),
                        "confidence": 0.9,
                    },
                }
            )
        return {
            "job_id": self.job_id,
            "kind": arguments["kind"],
            "logical_model": arguments["logical_model"],
            "status": "completed",
            "executor_id": actor,
            "summary": f"Generated {len(artifacts)} image artifacts with media_primary.",
            "artifacts": tuple(artifacts),
            "presentation": "final_attachment",
        }


class StoredAssetPackCapabilities(DirectMultimediaCapabilities):
    async def execute(  # type: ignore[no-untyped-def]
        self, *, tenant_id, run_id, actor, name, arguments, idempotency_key
    ) -> Mapping[str, JsonValue]:
        del tenant_id, run_id, idempotency_key
        self.calls.append((actor, name, arguments))
        labels = tuple(
            str(label)
            for label in arguments.get("artifact_labels", ())
            if isinstance(label, str)
        ) or ("角色锁定资产：林渊", "特效资产")
        artifacts: list[Mapping[str, JsonValue]] = []
        for index, label in enumerate(labels):
            digest = f"{index + 1:064x}"[-64:]
            artifact_id = str(uuid4())
            storage_key = f"{TENANT_ID}/{RUN_ID}/{artifact_id}/asset-{index}.png"
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "storage_key": storage_key,
                    "filename": f"asset-{index}.png",
                    "mime_type": "image/png",
                    "size_bytes": 123 + index,
                    "sha256": digest,
                    "download_url": f"/api/v1/admin/runs/{RUN_ID}/artifacts/{artifact_id}/download",
                    "label": label,
                    "title": label,
                    "visual_review": {
                        "passed": True,
                        "summary": "资产合格",
                        "issues": (),
                        "confidence": 0.9,
                    },
                }
            )
        return {
            "job_id": "media-stored",
            "kind": arguments["kind"],
            "logical_model": arguments["logical_model"],
            "status": "completed",
            "executor_id": actor,
            "summary": f"Generated {len(artifacts)} image artifacts with media_primary.",
            "artifacts": tuple(artifacts),
            "presentation": "final_attachment",
        }


def test_final_attachment_summary_lists_all_multimedia_files() -> None:
    summary = _final_attachment_summary(
        [
            {
                "name": "generate_multimedia",
                "result": {
                    "presentation": "final_attachment",
                    "summary": "Generated 2 image artifacts with media_primary.",
                    "artifacts": (
                        {
                            "filename": "male-lead-model-sheet.png",
                            "mime_type": "image/png",
                            "download_url": (
                                "/api/v1/admin/multimedia/jobs/media-test/artifacts/0/download"
                            ),
                            "expires_at": "2026-09-15T00:00:00+00:00",
                        },
                        {
                            "filename": "female-lead-model-sheet.png",
                            "mime_type": "image/png",
                            "download_url": (
                                "/api/v1/admin/multimedia/jobs/media-test-2/artifacts/0/download"
                            ),
                            "expires_at": "2026-09-15T00:00:00+00:00",
                        },
                    ),
                },
            }
        ]
    )

    assert summary is not None
    assert "Generated 2 image artifacts" in summary
    assert "male-lead-model-sheet.png" in summary
    assert "female-lead-model-sheet.png" in summary
    assert summary.count("下载图片") == 2


class ReplaySafeDocumentCapabilities:
    async def execute(
        self,
        *,
        tenant_id: UUID,
        run_id: UUID,
        actor: str,
        name: str,
        arguments: Mapping[str, JsonValue],
        idempotency_key: str,
    ) -> Mapping[str, JsonValue]:
        del tenant_id, run_id, actor, idempotency_key
        assert name == "document.generate_docx"
        assert arguments["presentation"] == "final_attachment"
        return {
            "presentation": "final_attachment",
            "summary": "已生成恢复后的 DOCX 文档。",
            "file": {
                "filename": "long-report.docx",
                "mime_type": (
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ),
            },
        }

    def is_replay_safe(self, name: str) -> bool:
        return name == "document.generate_docx"


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
    checkpoint: RuntimeCheckpoint | None = None,
    routing_decision: Mapping[str, JsonValue] | None = None,
    timeout_seconds: float = 60.0,
    request: str = "Write a short answer",
) -> TaskContext:
    return TaskContext(
        run_id=RUN_ID,
        tenant_id=TENANT_ID,
        mode=TaskMode.DISPATCH,
        request=request,
        artifacts=artifacts,
        checkpoint=checkpoint,
        routing_decision={} if routing_decision is None else routing_decision,
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


def test_text_artifact_sanitizes_unsafe_control_characters() -> None:
    step = DispatchStep(id="director_step", agent="director", task="compose final video plan")
    completion = GatewayCompletion(
        response=ModelResponse(
            text="first\x00line\n\tzero\u200bwidth",
            usage=TokenUsage(10, 1, 11),
        ),
        deployment_id="primary",
        logical_model="general",
        provider_id="deepseek",
        provider_model="deepseek/deepseek-v4-flash",
        cost_usd=Decimal(0),
    )

    artifact = CrewDispatchRuntime._artifact(step, completion, (), version=1)

    assert dict(artifact.content) == {"text": "firstline\n\tzerowidth"}


def test_model_response_artifact_sanitizes_text_before_evidence_storage() -> None:
    completion = GatewayCompletion(
        response=ModelResponse(
            text="first\x00line\r\n\tzero\u200bwidth",
            usage=TokenUsage(10, 1, 11),
        ),
        deployment_id="primary",
        logical_model="general",
        provider_id="deepseek",
        provider_model="deepseek/deepseek-v4-flash",
        cost_usd=Decimal(0),
    )

    artifact = CrewDispatchRuntime._model_artifact(completion, "director", ())

    assert dict(artifact.content)["text"] == "firstline\n\tzerowidth"


def test_model_response_artifact_strips_hidden_think_block_before_storage() -> None:
    completion = GatewayCompletion(
        response=ModelResponse(
            text="<think>private chain of thought</think>\n\n# 可展示剧本\n正文",
            usage=TokenUsage(10, 1, 11),
        ),
        deployment_id="primary",
        logical_model="general",
        provider_id="deepseek",
        provider_model="deepseek/deepseek-v4-flash",
        cost_usd=Decimal(0),
    )

    artifact = CrewDispatchRuntime._model_artifact(completion, "director", ())

    assert dict(artifact.content)["text"] == "# 可展示剧本\n正文"


def test_model_response_rejects_unclosed_hidden_think_block() -> None:
    completion = GatewayCompletion(
        response=ModelResponse(
            text="<think>private chain of thought\n\n# 半截剧本",
            usage=TokenUsage(10, 1, 11),
        ),
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


async def test_multimedia_generator_direct_character_sheet_splits_gender_lead_prompts() -> None:
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

    events = [
        event
        async for event in runtime.run(
            _context(request="为男女主生成角色参考设定表，风格全是二次元，不要太细节也不要太简化")
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    actor, name, arguments = capabilities.calls[0]
    assert actor == "multimedia_generator"
    assert name == "generate_multimedia"
    assert arguments["kind"] == "image"
    assert arguments["artifact_count"] == 2
    artifact_prompts = arguments["artifact_prompts"]
    assert isinstance(artifact_prompts, tuple)
    assert len(artifact_prompts) == 2
    assert "男主" in cast(str, artifact_prompts[0])
    assert "女主" in cast(str, artifact_prompts[1])
    for prompt in artifact_prompts:
        prompt_text = cast(str, prompt)
        assert "一张图只包含一个角色" in prompt_text
        assert "同一画风" in prompt_text
        assert "全二次元" in prompt_text
        assert "中等复杂度" in prompt_text
        assert "重复近景头像" in prompt_text
        assert "与角色设定无关的食物" in prompt_text
        assert "禁止写实主图+二次元表情+线稿三视图" in prompt_text
        assert "Character Identity + Look / Costume + Pose + Scene + Shot Prompt" in prompt_text
        assert "图内尽量不要写文字" in prompt_text
        assert "文字说明放在结构化产物元数据里" in prompt_text


async def test_multimedia_generator_direct_character_design_splits_gender_lead_prompts() -> None:
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

    events = [
        event
        async for event in runtime.run(
            _context(request="根据这个剧本，生成男女主角的角色设定图，风格全是写实")
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    actor, name, arguments = capabilities.calls[0]
    assert actor == "multimedia_generator"
    assert name == "generate_multimedia"
    assert arguments["kind"] == "image"
    assert arguments["artifact_count"] == 2
    artifact_prompts = arguments["artifact_prompts"]
    assert isinstance(artifact_prompts, tuple)
    assert len(artifact_prompts) == 2
    assert "男主" in cast(str, artifact_prompts[0])
    assert "女主" in cast(str, artifact_prompts[1])
    for prompt in artifact_prompts:
        prompt_text = cast(str, prompt)
        assert "本张角色参考设定表/角色设定图的唯一目标角色" in prompt_text
        assert "一张图只包含一个角色" in prompt_text
        assert "全写实" in prompt_text


async def test_multimedia_generator_rejects_incomplete_character_sheet_count() -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct media generation")

    class IncompleteMultimediaCapabilities(DirectMultimediaCapabilities):
        async def execute(  # type: ignore[no-untyped-def]
            self, *, tenant_id, run_id, actor, name, arguments, idempotency_key
        ) -> Mapping[str, JsonValue]:
            del tenant_id, run_id, actor, name, arguments, idempotency_key
            return {
                "job_id": "media-test",
                "kind": "image",
                "logical_model": "media_primary",
                "status": "completed",
                "executor_id": "multimedia_generator",
                "summary": "Generated image artifact with media_primary.",
                "artifacts": (
                    {
                        "filename": "only-one-model-sheet.png",
                        "mime_type": "image/png",
                        "download_url": (
                            "/api/v1/admin/multimedia/jobs/media-test/artifacts/0/download"
                        ),
                    },
                ),
                "presentation": "final_attachment",
            }

    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        _one_step_tool_plan(tools=("generate_multimedia",), multimedia=True),
        capability_gateway=IncompleteMultimediaCapabilities(),
        crew_factory=CapturingFactory(),
    )

    with pytest.raises(CapabilityOutcomeUncertain, match="artifact count is incomplete"):
        async for _event in runtime.run(
            _context(request="为男女主生成角色参考设定表，风格全是写实")
        ):
            pass


def test_character_sheet_prompt_separates_identity_and_look_without_in_image_text() -> None:
    prompt = _direct_multimedia_generation_prompt(
        _context(request="为女主生成角色参考设定表，风格全是二次元，不要太细节也不要太简化"),
        DispatchStep(
            id="multimedia_generator_step",
            agent="multimedia_generator",
            task="User task: 生成女主角色定妆资产图",
            tools=("generate_multimedia",),
            final_synthesizer=True,
        ),
        (),
    )

    assert "Character Identity + Look / Costume + Pose + Scene + Shot Prompt" in prompt
    assert "Character Identity 负责这个人是谁" in prompt
    assert "Look / Costume 只负责当前穿什么" in prompt
    assert "图内尽量不要写文字" in prompt
    assert "文字说明放在结构化产物元数据里" in prompt


async def test_multimedia_generator_direct_person_reference_splits_each_script_role() -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct media generation")

    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "### 主角人设构建\n\n"
                "## 女主：苏念（26岁）\n"
                "- 职业：广告公司资深文案\n"
                "- 外貌：黑长直，浅粉针织衫，温柔但有边界感。\n\n"
                "## 男主：陆沉（29岁）\n"
                "- 职业：品牌公司创始人\n"
                "- 外貌：短黑发，灰色西装，冷静克制。\n\n"
                "## 闺蜜：林小鹿（25岁）\n"
                "- 职业：咖啡店主理人\n"
                "- 外貌：短发，牛仔外套，活泼机灵。"
            )
        },
    )
    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        _one_step_tool_plan(tools=("generate_multimedia",), multimedia=True),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                request="根据剧本生成各个角色的人物参考图和分镜图",
                artifacts=(script,),
            )
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    _actor, _name, arguments = capabilities.calls[0]
    assert arguments["kind"] == "image"
    assert arguments["artifact_count"] == 4
    artifact_prompts = arguments["artifact_prompts"]
    assert isinstance(artifact_prompts, tuple)
    assert len(artifact_prompts) == 4
    assert "唯一目标角色：女主" in cast(str, artifact_prompts[0])
    assert "苏念" in cast(str, artifact_prompts[0])
    assert "浅粉针织衫" in cast(str, artifact_prompts[0])
    assert "唯一目标角色：男主" in cast(str, artifact_prompts[1])
    assert "陆沉" in cast(str, artifact_prompts[1])
    assert "灰色西装" in cast(str, artifact_prompts[1])
    assert "唯一目标角色：闺蜜" in cast(str, artifact_prompts[2])
    assert "林小鹿" in cast(str, artifact_prompts[2])
    assert "牛仔外套" in cast(str, artifact_prompts[2])
    assert "分镜图" in cast(str, artifact_prompts[3])
    assert "不要生成角色定妆照" in cast(str, artifact_prompts[3])
    for prompt in artifact_prompts[:3]:
        prompt_text = cast(str, prompt)
        assert "一张图只包含一个角色" in prompt_text
        assert "不要混入其他角色设定" in prompt_text
        assert "重复近景头像" in prompt_text
        assert "与角色设定无关的食物" in prompt_text


def test_full_asset_character_targets_extract_names_before_descriptive_text() -> None:
    context = _context(
        request=(
            "女主苏念是灵能调查员，穿灰白风衣、银色耳坠。"
            "男主林烬黑色冲锋衣、黄色外卖箱、左手铜钱法器。"
            "每个主要角色必须单独一张，不允许男女主混在同一张图。"
            "请根据这个剧本直接生成全量专业资产图。"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, ())

    assert targets[:2] == ("女主苏念", "男主林烬")
    assert "女主" not in targets
    assert "男主" not in targets
    assert "女主角色" not in targets
    assert "女主混" not in targets
    assert "男主混" not in targets


def test_full_asset_character_targets_prefer_script_role_table_names() -> None:
    context = _context(
        request=(
            "请根据已批准剧本生成全量专业资产图。每个主要角色必须单独一张，"
            "不允许男女主混在同一张图，关键反派或配角设定板也要有。\n\n"
            "| 角色 | 演员表意 | 一句话定位 |\n"
            "|---|---|---|\n"
            "| **陆渊**(男) | 22 岁,夜班保安 | 主线觉醒者 |\n"
            "| **沈清漪**(女) | 24 岁,龙卫特勤队长 | 护道组织骨干 |\n"
            "| **赵乾**(反派) | 26 岁,金融新贵 | 邪修少主 |\n"
            "| 口罩男(反派手下) | 30 岁 | 暗桩 |\n"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, ())

    assert targets == ("陆渊", "沈清漪", "赵乾", "口罩男")
    assert "反派或配" not in targets


def test_full_asset_character_targets_extract_numbered_script_cast_names() -> None:
    script = (
        "## 二、主要角色（供 Step 2 资产拆解使用；本步不出图）\n\n"
        "1. **林越**，25 岁，男。天穹大厦夜班保安。真实身份：九百年前渡劫失败的修仙者。\n"
        "2. **苏晚晴**，24 岁，女。天穹集团执行董事。母亲死后持半枚玉牌。\n"
        "3. **雷豹**，32 岁，男。雷家家主次子，白手套、掌心缠绷带。\n"
        "4. **雷家武者甲**，30 岁上下，壮硕，拳重而不稳。\n"
        "5. **队长**，中年男声，仅对讲机出现，不出镜。\n\n"
        "## 三、场景清单\n"
        "- **S1** 天穹大厦·地下三层封印室\n"
    )
    context = _context(
        request=(
            "请基于已批准剧本拆解并生成全量资产图。"
            "主要角色必须每人单独一张 Character Model Sheet，不能混图。\n\n"
            f"{script}"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, ())

    assert targets == ("林越", "苏晚晴", "雷豹", "雷家武者甲")
    assert "队长" not in targets


def test_full_asset_character_targets_extract_bold_script_cast_names_before_fallback() -> None:
    script = (
        "## 二、角色表（本集出场，供 Step 2 拆解角色锁定用）\n\n"
        "**叶九霄**｜男主，28 岁，九转不灭体觉醒者 / 昆仑墟归来者\n"
        "- 外形：破旧深灰风衣、内搭黑T、指节有旧伤。\n\n"
        "**苏清月**｜女主，26 岁，青云集团总裁，高武四品\n"
        "- 外形：剪裁利落的墨色西装长裙、低马尾。\n\n"
        "**叶天龙**｜反派一号，52 岁，叶家家主\n"
        "- 伪善外壳 + 宗门式威压。\n\n"
        "**陈伯**｜伏笔角色，70 岁，仅画外音，不出镜。\n\n"
        "## 三、完整剧本\n"
        "女主介入冲突，阶段规则要求先资产后视频。\n"
    )
    context = _context(
        request=(
            "实机 smoke：阶段规则是先生成剧本，再生成资产图。"
            "主要角色必须每人单独一张 Character Model Sheet。\n\n"
            f"{script}"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, ())

    assert targets == ("叶九霄", "苏清月", "叶天龙")
    assert "阶段规则" not in targets
    assert "女主介入" not in targets
    assert "陈伯" not in targets


def test_full_asset_prompts_keep_identity_and_director_rules_when_pack_is_large() -> None:
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="screenwriter",
        content={
            "text": (
                "## 二、角色表\n\n"
                "**苏清月**｜女主，26岁，急诊医生，白大褂、低马尾、银针。\n"
                "**林渊**｜男主，22岁，外卖员，藏蓝冲锋衣、黄色外卖箱、青玉断佩。\n"
                "**雷豹**｜反派，32岁，高武打手，掌心缠绷带。\n\n"
                "## 三、场景\n"
                "EP01_SC01：雨夜，林渊背黄色外卖箱冲进巷口，青玉断佩冒出蓝色电弧。\n"
                "EP01_SC02：同一夜，苏清月用银针牵出真气纹。\n"
                "EP01_SC03：第二天回家，苏清月换成浅灰居家服。"
            )
        },
    )
    context = _context(
        request="根据已批准剧本生成全量专业资产图，先把资产生成并验证好。",
        artifacts=(script,),
    )
    step = DispatchStep(
        id="asset_generator_step",
        agent="asset_generator",
        task="根据剧本拆解角色、服装、场景、道具、动作、特效、镜头和表演节奏资产。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    prompts = _direct_full_production_asset_prompts(context, step, (script,), None)

    assert len(prompts) >= 10
    assert all(len(prompt.encode("utf-8")) <= 2_700 for prompt in prompts)
    female_prompt = next(prompt for prompt in prompts if "角色锁定资产：苏清月" in prompt)
    assert "Available Looks / 多造型管理" in female_prompt
    assert "Character ID 只负责脸型" in female_prompt
    assert "Look ID 只负责服装" in female_prompt
    assert "主定妆正脸半身大图" in female_prompt
    assert "医疗办公室、医院走廊" in female_prompt
    assert "图内文字尽量不用英文" in female_prompt
    assert "角色锁定资产只管理人物身份和明确服装 Look" in female_prompt
    assert "不得出现雨伞、雨景、街景、护甲、战术服" in female_prompt
    action_prompt = next(prompt for prompt in prompts if "本张图片资产类别：动作资产" in prompt)
    assert "角色外观必须沿用角色锁定资产" in action_prompt
    assert "黄色外卖箱" in action_prompt
    assert "优先使用无脸灰色剪影/线稿动作人偶" in action_prompt
    assert "只用黄色外卖箱、青玉断佩、银针" in action_prompt
    assert "不得出现雨伞" in action_prompt
    assert "用雨线和湿地面表达雨" in action_prompt
    assert "不得出现古风发冠" in action_prompt
    costume_prompt = next(prompt for prompt in prompts if "本张图片资产类别：服装妆造资产" in prompt)
    assert "必须按角色分区展示" in costume_prompt
    assert "禁止项不得画进画面当反例" in costume_prompt
    assert "即使旁边写“禁止使用”也不合格" in costume_prompt
    assert "不得生成 Character ID 001/A01/B03" in costume_prompt
    assert "不得生成金发西装男" in costume_prompt
    assert "优先使用无头服装平铺" in costume_prompt
    assert "不要使用真人模特照片" in costume_prompt
    scene_prompt = next(prompt for prompt in prompts if "本张图片资产类别：场景资产" in prompt)
    assert "只覆盖剧本出现的地点" in scene_prompt
    assert "不得替换成写字楼大厅、会展广场" in scene_prompt
    assert "不得出现 smoke、v30、test" in scene_prompt
    prop_prompt = next(prompt for prompt in prompts if "本张图片资产类别：道具资产" in prompt)
    assert "只生成剧本明确要求的道具" in prop_prompt
    assert "不要补充能量核心、机械装置" in prop_prompt
    assert "证件照片只能使用空白头像占位" in prop_prompt
    assert "标签不得错位" in prop_prompt
    effects_prompt = next(prompt for prompt in prompts if "本张图片资产类别：特效资产" in prompt)
    assert "不要把雨水剑气画成实体长剑" in effects_prompt
    camera_prompt = next(prompt for prompt in prompts if "本张图片资产类别：镜头资产" in prompt)
    assert "不得出现真人眼睛、真实脸部特写" in camera_prompt
    rhythm_prompt = next(prompt for prompt in prompts if "本张图片资产类别：表演节奏" in prompt)
    assert "不要换成黑西装男性" in rhythm_prompt
    assert "0-3秒Hook" in rhythm_prompt
    assert "35-52秒反转兑现" in rhythm_prompt
    assert "不得写成 35-522" in rhythm_prompt
    assert "不得省略“秒”字" in rhythm_prompt
    assert "禁止英文错字" in rhythm_prompt


def test_prune_invalidated_artifact_lineage_removes_derived_media() -> None:
    model = Artifact(
        id=uuid4(),
        type="model_response",
        producer="asset_generator",
        content={"response": {"text": "direct capability"}},
    )
    tool = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="asset_generator",
        content={"result": {"artifacts": []}},
        source_ids=(str(model.id),),
    )
    image = Artifact(
        id=uuid4(),
        type="image",
        producer="asset_generator",
        content={"uri": "file:///tmp/old.png"},
        source_ids=(str(tool.id),),
    )
    unrelated = Artifact(
        id=uuid4(),
        type="image",
        producer="asset_generator",
        content={"uri": "file:///tmp/keep.png"},
    )
    registry = {str(item.id): item for item in (model, tool, image, unrelated)}
    invalidated = {str(tool.id)}

    _prune_invalidated_artifact_lineage(registry, invalidated)

    assert str(tool.id) not in registry
    assert str(image.id) not in registry
    assert str(image.id) in invalidated
    assert str(model.id) in registry
    assert str(unrelated.id) in registry


def test_full_asset_character_targets_use_full_source_text_for_character_biographies() -> None:
    long_gate_prefix = (
        "# 交付物：短剧《高武：我以修仙证道》第一集\n\n"
        "**执行状态：STEP 1 COMPLETE / GATE: AWAITING REVIEW**\n"
        "阶段规则：先生成完整剧本；该剧本是后续资产、分镜、AI 视频和剪辑的唯一文本基准。\n"
        "关键阻塞：此处是模型自述，不能作为角色名来源。\n"
        + "资产拆解预告、审批门禁、流程说明、视觉审核规则。" * 120
    )
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                long_gate_prefix
                + "\n\n### 1.2 人物小传（供角色锁定表使用）\n\n"
                "- **陈砚**，22，男。江城大学武道系大三，评级 F 级无脉者。\n"
                "- **苏晚晴**，21，女。江南苏家次女，陈砚的挂名未婚妻。\n"
                "- **赵鲲鹏**，23，男。武道系首席，赵家嫡子，B+ 级。\n"
                "- **剑冢器灵（声音）**，苍老、沙哑、仅画外音。\n"
            )
        },
    )
    context = _context(
        request=(
            "必须按顺序执行：1 先生成完整剧本并等待审核；2 剧本通过后从剧本拆解全量制作资产并生成资产图，"
            "主要角色必须每人单独一张角色锁定 Character Model Sheet。"
        ),
        artifacts=(script,),
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="阶段规则：先生成资产图。女主介入只是剧情说明，不是角色名。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, (script,))

    assert targets == ("陈砚", "苏晚晴", "赵鲲鹏")
    assert "阶段规则" not in targets
    assert "女主介入" not in targets
    assert "剑冢器灵" not in targets


def test_full_asset_character_targets_extract_role_column_from_script_table() -> None:
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "## 一、DELIVERABLE：完整剧本\n\n"
                "### 角色表（本集出场）\n"
                "| 代号 | 角色 | 年龄 | 身份 | 本集功能 |\n"
                "|---|---|---|---|---|\n"
                "| C1 | 林渊 | 25 | 前江城第一天才，现外卖骑手，丹田被废 | 主角，觉醒起点 |\n"
                "| C2 | 苏清鸢 | 24 | 苏氏医药集团总裁，医修世家传人 | 女主，唯一识货的人 |\n"
                "| C3 | 赵天霸 | 26 | 赵家嫡孙，罡劲武者 | 反派，当年废林渊之人 |\n"
                "| C4 | 玄尘子 | 声 | 太古剑尊残魂，寄于青铜剑穗 | 仅画外音+虚影 |\n"
            )
        },
    )
    context = _context(
        request="生成剧本后继续生成全量资产图，主要角色每人单独一张 Character Model Sheet。",
        artifacts=(script,),
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, (script,))

    assert targets == ("林渊", "苏清鸢", "赵天霸")
    assert "C1" not in targets
    assert "玄尘子" not in targets


def test_full_asset_character_targets_extract_character_anchor_heading() -> None:
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "### 【人物设定锚点】（下游角色锁定基准）\n"
                "- **陆昭**｜22 男｜灵潮外卖骑手，无品→觉醒｜黑短发、眼尾有旧疤｜眼底藏剑意\n"
                "- **苏清鸢**｜22 女｜天枢武院首席兼秩序司特级巡察使｜黑长直高马尾｜冷、极少失态\n"
                "- **林震岳**｜25 男｜天枢武院少院主，凝气巅峰｜白衬衫黑西裤｜傲慢\n"
                "- **苏清月**｜22 女｜女主 / 退婚线｜本集仅被提及，不出画｜伏笔\n\n"
                "## Risks\n"
                "- **视觉审核缺位风险**：如果没有视觉模型审核，不能继续。\n"
            )
        },
    )
    context = _context(
        request="生成剧本后继续生成全量资产图，主要角色每人单独一张 Character Model Sheet。",
        artifacts=(script,),
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, (script,))

    assert targets == ("陆昭", "苏清鸢", "林震岳")
    assert "视觉审核缺位风险" not in targets
    assert "苏清月" not in targets


def test_full_asset_character_targets_extract_markdown_headings_under_people_table() -> None:
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "## 2. 人物表（含外形锁定要点，供阶段 2 拆解 Character Model Sheet）\n\n"
                "### 林砚（男主，26）\n"
                "- 身份：寰宇大厦 B2 夜班监控员。\n"
                "- 外形锁定：黑发略长、深灰工装外套、青玉断佩。\n\n"
                "### 苏晚（女主，24）\n"
                "- 身份：灵能管控局特勤。\n"
                "- 外形锁定：黑色修身风衣、银色检测仪。\n\n"
                "### 赵鲲（反派，28）\n"
                "- 身份：鼎盛集团少主，A 级武修。\n\n"
                "## 3. 风险与成本\n"
                "### 特效成本风险\n"
                "### 设定密度风险\n"
            )
        },
    )
    context = _context(
        request="生成剧本后继续生成全量资产图，主要角色每人单独一张 Character Model Sheet。",
        artifacts=(script,),
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, (script,))

    assert targets == ("林砚", "苏晚", "赵鲲")
    assert "特效成本风险" not in targets
    assert "设定密度风险" not in targets


def test_full_asset_character_targets_extract_bold_numbered_people_bios() -> None:
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "## 三、人物小传与外观锚点（供 Step 2 角色锁定使用，本步骤不生成图）\n\n"
                "**1. 林渊｜男｜22｜外卖员（真身：九霄仙庭万古仙尊，未觉醒）**\n"
                "- 性格：能忍，忍到极限才动手。\n"
                "- 外观锚点：短碎发、眉骨旧疤、藏蓝外卖骑手冲锋衣。\n"
                "- 表演基调：前段肩线下垂。\n\n"
                "**2. 赵天霸｜男｜24｜龙门武馆少主｜武师初期｜反派**\n"
                "- 性格：纨绔、暴戾、迷信家世。\n"
                "- 外观锚点：大背头、金色武馆纹章黑夹克。\n\n"
                "**3. 苏清月｜女｜21｜江城武道协会见习医师**\n"
                "- 性格：冷静、数据控、职业敏感。\n"
                "- 外观锚点：低马尾、白大褂、银色胸针。\n\n"
                "**4. 林小雨｜女｜12｜林渊之妹｜仅出现在电话与照片（不出镜）**\n"
            )
        },
    )
    context = _context(
        request="生成剧本后继续生成全量资产图，主要角色每人单独一张 Character Model Sheet。",
        artifacts=(script,),
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, (script,))

    assert targets == ("林渊", "赵天霸", "苏清月")
    assert "性格" not in targets
    assert "外观锚点" not in targets
    assert "表演基调" not in targets
    assert "林小雨" not in targets


def test_full_asset_character_targets_extract_inline_age_gender_descriptions_from_request() -> None:
    context = _context(
        request=(
            "根据已确认剧本生成全量专业资产图。剧本：林渊，22岁男，雨夜外卖骑手，"
            "黑短发，眉骨旧疤。苏清月，21岁女，江城武道协会见习医师，低马尾。"
            "赵天霸，24岁男，龙门武馆少主，大背头。关键资产：青玉断佩、蓝色电弧。"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    targets = _full_production_asset_character_targets(context, step, ())

    assert targets == ("林渊", "苏清月", "赵天霸")
    assert "关键资产" not in targets


def test_full_asset_character_targets_backfill_request_roles_when_source_is_partial() -> None:
    context = _context(
        request=(
            "根据已确认剧本生成全量专业资产图。剧本：林渊，22岁男，雨夜外卖骑手。"
            "苏清月，21岁女，江城武道协会见习医师。"
            "赵天霸，24岁男，龙门武馆少主。"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )
    partial_source = Artifact(
        id=uuid4(),
        type="text",
        producer="copywriter",
        content={
            "text": (
                "## 人物设定\n"
                "### 赵天霸\n"
                "龙门武馆少主。\n"
                "### 林渊\n"
                "雨夜外卖骑手。\n"
            )
        },
    )

    targets = _full_production_asset_character_targets(context, step, (partial_source,))

    assert targets == ("赵天霸", "林渊", "苏清月")


def test_full_asset_prompt_specs_include_more_than_three_roles_and_support_assets() -> None:
    context = _context(
        request=(
            "根据已确认剧本生成全量专业资产图。剧本：林渊，22岁男，雨夜外卖骑手。"
            "苏清月，21岁女，武道协会见习医师。赵天霸，24岁男，龙门武馆少主。"
            "沈墨，29岁男，地下拍卖场主持人。秦岚，27岁女，灵纹鉴定师。"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    specs = _direct_full_production_asset_prompt_specs(context, step, ())
    labels = tuple(title for title, _requirement in specs)

    assert labels[:5] == (
        "角色锁定资产：林渊",
        "角色锁定资产：苏清月",
        "角色锁定资产：赵天霸",
        "角色锁定资产：沈墨",
        "角色锁定资产：秦岚",
    )
    assert "服装妆造资产" in labels
    assert "场景资产" in labels
    assert "道具资产" in labels
    assert "动作资产" in labels
    assert "特效资产" in labels
    assert "镜头资产" in labels
    assert "表演节奏与风格锁定资产" in labels
    assert "角色硬锚点（最高优先级，所有模块都必须对应）：林渊,22岁男,雨夜外卖骑手" in specs[0][1]
    assert "角色硬锚点（最高优先级，所有模块都必须对应）：苏清月,21岁女,武道协会见习医师" in specs[1][1]
    assert "1-3 套剧情服装/状态变体" in specs[0][1]
    assert "不能换衣服后换成另一个人" in "\n".join(requirement for _title, requirement in specs)
    assert "不能把全剧都固定成一套衣服" in "\n".join(
        requirement for _title, requirement in specs
    )


def test_full_asset_prompts_include_identity_look_and_production_direction() -> None:
    context = _context(
        request=(
            "根据已确认剧本生成全量专业资产图。剧本：女主苏清月，医生，白大褂值夜班。"
            "男主林渊，外卖员，雨夜背黄色外卖箱。EP01_SC02：同一夜继续追查。"
            "EP01_SC03：第二天苏清月回家换成居家服。"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    prompts = _direct_full_production_asset_prompts(context, step, (), None)
    joined = "\n".join(prompts)

    assert "CHARACTER_ID: CHAR_SQY_001" in joined
    assert "LOOK_ID: LOOK_001" in joined
    assert "IDENTITY LOCK" in joined
    assert "导演/制片" in joined
    assert "Scene Character State" in joined
    assert "不要继承服装参考图中的脸" in joined
    assert "只允许修改服装" in joined


def test_full_asset_labels_expand_role_name_sentences_without_colons() -> None:
    context = _context(
        request=(
            "根据已确认剧本生成全量专业资产图。剧本角色：女主苏清月，21岁，"
            "江城武道协会见习医师，低马尾，白大褂；男主林渊，22岁，"
            "雨夜外卖骑手，黑短发，右手常提黄色外卖箱。"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    labels = _direct_full_production_asset_labels(context, step, ())

    assert labels[:2] == ("角色锁定资产：女主苏清月", "角色锁定资产：男主林渊")


def test_full_asset_labels_strip_instruction_words_from_role_targets() -> None:
    context = _context(
        request=(
            "根据已确认剧本继续生成资产。注意：角色参考设定表必须拆成"
            "女主苏清月和男主林渊两张独立 Character Model Sheet，"
            "不要混成一张图。"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    labels = _direct_full_production_asset_labels(context, step, ())

    assert labels[:2] == ("角色锁定资产：女主苏清月", "角色锁定资产：男主林渊")
    assert "角色锁定资产：女主苏清月和" not in labels
    assert "角色锁定资产：男主林渊两张" not in labels


def test_full_asset_prompt_specs_support_larger_short_drama_casts() -> None:
    context = _context(
        request=(
            "根据已确认剧本生成全量专业资产图。剧本：林渊，22岁男。苏清月，21岁女。"
            "赵天霸，24岁男。沈墨，29岁男。秦岚，27岁女。林母，48岁女。"
            "苏父，52岁男。韩七，31岁男。阿洛，19岁女。"
        )
    )
    step = DispatchStep(
        id="assets",
        agent="asset_generator",
        task="根据剧本生成全量专业资产图。",
        tools=("generate_multimedia",),
        final_synthesizer=True,
    )

    specs = _direct_full_production_asset_prompt_specs(context, step, ())
    labels = tuple(title for title, _requirement in specs)

    assert labels[:9] == (
        "角色锁定资产：林渊",
        "角色锁定资产：苏清月",
        "角色锁定资产：赵天霸",
        "角色锁定资产：沈墨",
        "角色锁定资产：秦岚",
        "角色锁定资产：林母",
        "角色锁定资产：苏父",
        "角色锁定资产：韩七",
        "角色锁定资产：阿洛",
    )
    assert "服装妆造资产" in labels
    assert "表演节奏与风格锁定资产" in labels


def test_direct_multimedia_retry_selection_preserves_passed_assets_and_retries_failed_items() -> None:
    previous = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="asset_generator",
        content={
            "result": {
                "artifacts": (
                    {
                        "kind": "image",
                        "uri": "data:image/png;base64," + ("a" * 50_000),
                        "text": "oversized inline description" * 2000,
                        "download_url": "/api/v1/admin/runs/run/artifacts/artifact/download",
                        "storage_key": "tenant/run/character.png",
                        "filename": "character.png",
                        "mime_type": "image/png",
                        "label": "角色锁定资产：林渊",
                        "generation_prompt": "角色生成提示词" * 2000,
                        "visual_review": {
                            "passed": True,
                            "summary": "角色资产合格",
                            "issues": ("轻微问题" * 200,),
                            "confidence": 0.91,
                        },
                    },
                    {
                        "kind": "image",
                        "uri": "artifact://action",
                        "filename": "action.png",
                        "mime_type": "image/png",
                        "label": "动作资产",
                        "visual_review": {
                            "passed": False,
                            "summary": "动作资产不像分解板",
                            "issues": ("只有剧照",),
                            "confidence": 0.88,
                        },
                    },
                    {
                        "kind": "image",
                        "uri": "artifact://effect",
                        "filename": "effect.png",
                        "mime_type": "image/png",
                        "label": "特效资产",
                        "visual_review": {
                            "passed": False,
                            "summary": "特效资产混入人物写真",
                            "issues": ("缺少可复用特效元素",),
                            "confidence": 0.87,
                        },
                    },
                ),
            },
        },
    )

    selection = adapter_module._direct_multimedia_retry_selection(
        previous_artifacts=(previous,),
        expected_labels=("角色锁定资产：林渊", "动作资产", "特效资产"),
        feedback_text="只重做视觉审核不合格的资产。",
    )

    assert selection is not None
    assert selection.retry_labels == ("动作资产", "特效资产")
    assert [item["label"] for item in selection.preserved_artifacts] == ["角色锁定资产：林渊"]
    preserved = selection.preserved_artifacts[0]
    assert "generation_prompt" not in preserved
    assert "uri" not in preserved
    assert "text" not in preserved
    assert "download_url" not in preserved
    assert preserved["storage_key"] == "tenant/run/character.png"
    review = cast(Mapping[str, JsonValue], preserved["visual_review"])
    assert review["passed"] is True
    assert len(cast(tuple[str, ...], review["issues"])[0]) <= 160


async def test_multimedia_generator_direct_script_image_assets_generate_full_asset_pack() -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct asset generation")

    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "## 剧本\n"
                "女主苏念在咖啡店发现会发光的旧钥匙，男主陆沉追来。"
                "窗外暴雨，钥匙引发蓝色电弧特效，两人奔跑穿过街巷。"
                + (
                    "都市高武修仙补充：霓虹雨夜、龙脉裂缝、外卖员林烬、"
                    "玄烛宗追兵、盛穹集团天台阵法、妹妹病房、铜钱法器、"
                    "蓝色电弧、金色符文、巷战、车流闪避、楼顶坠落救援。"
                )
                * 40
                + "最终尾部资产：祖传玉佩、终局天桥、紫色雷暴结界。"
            )
        },
    )
    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        DispatchPlan(
            agents=(
                AgentSpec(
                    id="asset_generator",
                    role="Asset Generator",
                    goal="Generate the full locked production asset image pack.",
                    logical_model="general",
                    allowed_tools=("generate_multimedia",),
                ),
            ),
            steps=(
                DispatchStep(
                    id="assets",
                    agent="asset_generator",
                    task="根据剧本生成全量专业资产图。",
                    tools=("generate_multimedia",),
                    final_synthesizer=True,
                    token_budget=100,
                ),
            ),
            allowed_tools=("generate_multimedia",),
            total_token_budget=100,
        ),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(request="先生成剧本，然后根据剧本生成图片资产，不要生成视频。", artifacts=(script,))
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    progress = next(event for event in events if event.kind == "custom.progress")
    assert progress.payload["phase"] == "multimedia_generation"
    assert progress.payload["artifact_count"] == 11
    assert progress.payload["parallelism"] == 9
    assert progress.payload["wave_count"] == 2
    assert progress.payload["completed_count"] == 0
    labels = progress.payload["artifact_labels"]
    assert isinstance(labels, tuple)
    assert len(labels) == 11
    assert cast(str, labels[0]).startswith("角色锁定资产：女主")
    assert "服装妆造资产" in labels
    assert "场景资产" in labels
    assert "道具资产" in labels
    assert "动作资产" in labels
    assert "特效资产" in labels
    assert "镜头资产" in labels
    assert "表演节奏与风格锁定资产" in labels
    _actor, name, arguments = capabilities.calls[0]
    assert name == "generate_multimedia"
    assert arguments["kind"] == "image"
    assert arguments["artifact_count"] == 11
    artifact_prompts = arguments["artifact_prompts"]
    assert isinstance(artifact_prompts, tuple)
    assert len(artifact_prompts) == 11
    canonical_arguments = json.dumps(
        arguments,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert len(canonical_arguments.encode("utf-8")) <= 32_768
    joined = "\n".join(cast(str, prompt) for prompt in artifact_prompts)
    for required in (
        "角色锁定资产：女主苏念",
        "角色锁定资产：男主陆沉",
        "服装妆造资产",
        "场景资产",
        "道具资产",
        "动作资产",
        "特效资产",
        "镜头资产",
        "表演节奏与风格锁定资产",
        "情绪变化",
        "声音节奏",
        "BGM 氛围",
    ):
        assert required in joined
    assert "不要只生成角色图" in joined
    assert "不是电影剧照" in joined
    assert "Character Model Sheet" in joined
    assert "干净、低噪声" in joined
    assert "不要把剧本里所有角色、地点、道具、动作、特效和背景都当作细节堆进同一张图" in joined
    assert "旧钥匙" in joined
    assert "蓝色电弧特效" in joined
    assert "终局天桥" in joined
    assert "不要擅自改成黑西装、战术服、奇幻铠甲或无关职业制服" in joined
    assert "不得擅自换成战术服、黑西装、陌生发型或无关人物" in joined
    assert "不要生成通用魔法爆炸集合" in joined
    assert "不要换成黑西装男性、陌生动漫角色或通用情绪模板" in joined
    assert "逐项列出" not in joined
    argument_labels = arguments["artifact_labels"]
    assert isinstance(argument_labels, tuple)
    assert len(argument_labels) == 11
    assert cast(str, argument_labels[0]).startswith("角色锁定资产：女主")
    assert "服装妆造资产" in argument_labels
    assert "场景资产" in argument_labels
    assert "道具资产" in argument_labels
    assert "动作资产" in argument_labels
    assert "特效资产" in argument_labels
    assert "镜头资产" in argument_labels
    assert "表演节奏与风格锁定资产" in argument_labels
    assert "唯一目标角色：女主苏念" in cast(str, artifact_prompts[0])
    assert "一张图只包含这个角色" in cast(str, artifact_prompts[0])
    assert "纯白/浅灰/透明感纯色背景" in cast(str, artifact_prompts[0])
    assert "医疗办公室、医院走廊" in cast(str, artifact_prompts[0])
    assert "职业场所背景" in cast(str, artifact_prompts[0])
    assert "主定妆大图" in cast(str, artifact_prompts[0])
    assert "正/侧/背全身三视图" in cast(str, artifact_prompts[0])
    assert "表情头部变化" in cast(str, artifact_prompts[0])
    assert "服装拆解" in cast(str, artifact_prompts[0])
    assert "随身物/职业道具" in cast(str, artifact_prompts[0])
    assert "材质色卡" in cast(str, artifact_prompts[0])
    assert "不得加入剧本或角色设定之外的随机道具" in cast(str, artifact_prompts[0])
    assert "少量清晰中文标签" in cast(str, artifact_prompts[0])
    assert "唯一目标角色：男主陆沉" in cast(str, artifact_prompts[1])
    assert "不要混入其他角色" in cast(str, artifact_prompts[1])
    assert "纯白/浅灰/透明感纯色背景" in cast(str, artifact_prompts[1])
    assert "空间视角" in cast(str, artifact_prompts[3])
    assert "独立物件 lineup" in cast(str, artifact_prompts[4])
    assert "姿态序列" in cast(str, artifact_prompts[5])
    assert "形态分层" in cast(str, artifact_prompts[6])
    assert "景别机位构图卡" in cast(str, artifact_prompts[7])
    assert "情绪节奏点" in cast(str, artifact_prompts[8])


async def test_multimedia_generator_direct_asset_pack_emits_polling_heartbeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("direct multimedia step should not call text model")

    monkeypatch.setattr(
        adapter_module,
        "_direct_capability_progress_heartbeat_seconds",
        lambda capability_name, base_payload: 0.01,
    )
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "## 人物小传\n"
                "- **林玄**｜男主，外卖员，黑发，灰色冲锋衣。\n"
                "- **苏清月**｜女主，医生，白大褂，冷静。\n"
                "## 剧本\n"
                "林玄在雨夜街巷释放蓝色电弧，苏清月在医院走廊发现符文。"
            )
        },
    )
    capabilities = SlowDirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        DispatchPlan(
            agents=(
                AgentSpec(
                    id="asset_generator",
                    role="Asset Generator",
                    goal="Generate locked image assets.",
                    logical_model="general",
                    allowed_tools=("generate_multimedia",),
                ),
            ),
            steps=(
                DispatchStep(
                    id="assets",
                    agent="asset_generator",
                    task="根据剧本生成全量专业资产图。",
                    tools=("generate_multimedia",),
                    final_synthesizer=True,
                    token_budget=100,
                ),
            ),
            allowed_tools=("generate_multimedia",),
            total_token_budget=100,
        ),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(request="先生成剧本，确认后生成资产图。", artifacts=(script,))
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    progress_events = [event for event in events if event.kind == "custom.progress"]
    assert progress_events[0].payload["status"] == "running"
    polling = [event for event in progress_events if event.payload["status"] == "polling"]
    assert polling
    assert polling[-1].payload["elapsed_seconds"] >= 0
    assert polling[-1].payload["timeout_seconds"] > 0
    assert "仍在轮询" in cast(str, polling[-1].payload["message"])
    assert polling[-1].payload["artifact_labels"] == progress_events[0].payload[
        "artifact_labels"
    ]


async def test_storyboard_artist_does_not_reuse_full_asset_pack_prompts_from_media_pipeline_request() -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct storyboard generation")

    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "## 剧本\n"
                "男主林烬在雨夜城市觉醒古武灵根，蓝色电弧照亮巷口，"
                "玄烛宗追兵逼近。"
            )
        },
    )
    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        DispatchPlan(
            agents=(
                AgentSpec(
                    id="storyboard_artist",
                    role="Storyboard Artist",
                    goal="Generate storyboard frames after locked assets are reviewed.",
                    logical_model="general",
                    allowed_tools=("generate_multimedia",),
                ),
            ),
            steps=(
                DispatchStep(
                    id="storyboard",
                    agent="storyboard_artist",
                    task=(
                        "用户要求先生成剧本，再生成全量资产图，资产确认后生成分镜图，"
                        "根据分镜和资产参考生成 AI 视频片段，最后剪辑成片。"
                    ),
                    tools=("generate_multimedia",),
                    final_synthesizer=True,
                    token_budget=100,
                ),
            ),
            allowed_tools=("generate_multimedia",),
            total_token_budget=100,
        ),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                request=(
                    "先生成剧本，确认之后生成资产，资产确认后生成分镜并制作视频，"
                    "最后剪辑视频。资产图要全。"
                ),
                artifacts=(script,),
            )
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    _actor, name, arguments = capabilities.calls[0]
    assert name == "generate_multimedia"
    assert arguments["kind"] == "image"
    artifact_prompts = cast(tuple[str, ...], arguments["artifact_prompts"])
    assert len(artifact_prompts) == 1
    assert "分镜图产物约束" in artifact_prompts[0]
    assert "分镜画面必须干净" in artifact_prompts[0]
    assert "不要把资产包里的角色设定、道具特写、特效设定、服装板、场景细节全部塞进同一格" in artifact_prompts[0]
    assert "本张图片资产类别：角色锁定资产" not in artifact_prompts[0]
    assert arguments["artifact_labels"] == ("分镜图 1",)


def test_storyboard_prompt_includes_continuity_and_frame_qc_hooks() -> None:
    context = _context(request="根据剧本和资产图生成分镜图。")
    step = DispatchStep(
        id="storyboard",
        agent="storyboard_artist",
        task="根据已审核资产生成分镜图。",
        tools=("generate_multimedia",),
    )

    prompt = adapter_module._direct_storyboard_generation_prompt(context, step, (), None)

    assert "Scene Character State" in prompt
    assert "继承上一场造型" in prompt
    assert "只重试失败镜头" in prompt
    assert "抽帧检测身份/服装/黑帧/静音/字幕" in prompt


async def test_multimedia_generator_direct_person_reference_keeps_split_when_group_is_negated() -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct media generation")

    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "## 女主：苏念（26岁）\n"
                "- 外貌：黑长直，浅粉针织衫。\n\n"
                "## 男主：陆沉（29岁）\n"
                "- 外貌：短黑发，灰色西装。"
            )
        },
    )
    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        _one_step_tool_plan(tools=("generate_multimedia",), multimedia=True),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                request="为每个角色单独生成角色设定图，不要同框合照",
                artifacts=(script,),
            )
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    _actor, _name, arguments = capabilities.calls[0]
    artifact_prompts = cast(tuple[str, ...], arguments["artifact_prompts"])
    assert len(artifact_prompts) == 2
    assert "唯一目标角色：女主" in artifact_prompts[0]
    assert "唯一目标角色：男主" in artifact_prompts[1]


async def test_multimedia_generator_direct_person_reference_reads_structured_script_roles() -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct media generation")

    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "result": {
                "script": {
                    "characters": (
                        {
                            "role": "女主",
                            "name": "苏念",
                            "age": "26岁",
                            "occupation": "广告公司资深文案",
                            "appearance": "黑长直，浅粉针织衫",
                        },
                        {
                            "role": "男主",
                            "name": "陆沉",
                            "age": "29岁",
                            "occupation": "品牌公司创始人",
                            "appearance": "短黑发，灰色西装",
                        },
                    )
                }
            }
        },
    )
    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        _one_step_tool_plan(tools=("generate_multimedia",), multimedia=True),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                request="给所有人物做人设图",
                artifacts=(script,),
            )
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    _actor, _name, arguments = capabilities.calls[0]
    artifact_prompts = cast(tuple[str, ...], arguments["artifact_prompts"])
    assert len(artifact_prompts) == 2
    assert "苏念" in artifact_prompts[0]
    assert "陆沉" in artifact_prompts[1]


async def test_multimedia_generator_direct_gender_lead_group_photo_keeps_single_artifact() -> None:
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

    events = [
        event
        async for event in runtime.run(
            _context(request="根据这个剧本，生成男女主角同框合照，风格全是写实")
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    _actor, _name, arguments = capabilities.calls[0]
    assert "artifact_count" not in arguments
    assert "artifact_prompts" not in arguments


async def test_multimedia_generator_direct_video_comparison_creates_reference_and_no_reference_prompts() -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct media generation")

    source = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="multimedia_generator",
        content={
            "result": {
                "artifacts": (
                    {
                        "filename": "male-lead-sheet.png",
                        "mime_type": "image/png",
                        "storage_key": "tenant/run/artifact/male-lead-sheet.png",
                    },
                ),
            }
        },
    )
    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        _one_step_tool_plan(tools=("generate_multimedia",), multimedia=True),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                request="生成两版 5 秒视频对比：一版带参考图锁定人物，一版不带参考图。",
                artifacts=(source,),
            )
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    _actor, _name, arguments = capabilities.calls[0]
    assert arguments["kind"] == "video"
    assert arguments["artifact_count"] == 2
    artifact_prompts = arguments["artifact_prompts"]
    assert isinstance(artifact_prompts, tuple)
    assert "带参考图" in cast(str, artifact_prompts[0])
    assert "锁定人物" in cast(str, artifact_prompts[0])
    assert "male-lead-sheet.png" in cast(str, artifact_prompts[0])
    assert "不带参考图" in cast(str, artifact_prompts[1])


async def test_shot_video_generator_direct_step_prefers_video_over_asset_and_storyboard_terms() -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct shot video generation")

    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        DispatchPlan(
            agents=(
                AgentSpec(
                    id="shot_video_generator",
                    role="Shot Video Generator",
                    goal="Generate AI video shots using locked assets and storyboard references.",
                    logical_model="general",
                    allowed_tools=("generate_multimedia",),
                ),
            ),
            steps=(
                DispatchStep(
                    id="shot_video_generator_step",
                    agent="shot_video_generator",
                    task="根据剧本、全量锁定资产图和分镜图生成 AI 视频片段。",
                    tools=("generate_multimedia",),
                    final_synthesizer=True,
                    token_budget=100,
                ),
            ),
            allowed_tools=("generate_multimedia",),
            total_token_budget=100,
        ),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(request="根据分镜和资产参考生成 AI 视频片段。")
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    _actor, _name, arguments = capabilities.calls[0]
    assert arguments["kind"] == "video"


async def test_asset_generator_direct_step_prefers_image_even_when_full_request_mentions_video() -> None:
    class FailingTextGateway:
        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            raise AssertionError("text gateway must not be called for direct asset generation")

    capabilities = DirectMultimediaCapabilities()
    runtime = CrewDispatchRuntime(
        FailingTextGateway(),
        DispatchPlan(
            agents=(
                AgentSpec(
                    id="asset_generator",
                    role="Asset Generator",
                    goal="Generate the full locked production asset image pack.",
                    logical_model="general",
                    allowed_tools=("generate_multimedia",),
                ),
            ),
            steps=(
                DispatchStep(
                    id="asset_generator_step",
                    agent="asset_generator",
                    task=(
                        "用户要求先生成剧本，再生成全量资产图，资产确认后生成分镜图，"
                        "根据分镜和资产参考生成 AI 视频片段，最后剪辑成片。"
                    ),
                    tools=("generate_multimedia",),
                    final_synthesizer=True,
                    token_budget=100,
                ),
            ),
            allowed_tools=("generate_multimedia",),
            total_token_budget=100,
        ),
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(request="生成高武都市修仙短剧，最终要 AI 视频片段和成片。")
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    _actor, _name, arguments = capabilities.calls[0]
    assert arguments["kind"] == "image"


async def test_video_compositor_directly_composes_upstream_file_handles_without_text_model() -> None:
    class FailingTextGateway:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            self.calls += 1
            raise AssertionError("text gateway must not be called for direct video composition")

    class DirectComposeCapabilities(DirectMultimediaCapabilities):
        async def execute(  # type: ignore[no-untyped-def]
            self, *, tenant_id, run_id, actor, name, arguments, idempotency_key
        ) -> Mapping[str, JsonValue]:
            del tenant_id, run_id, idempotency_key
            self.calls.append((actor, name, arguments))
            assert name == "compose_video"
            return {
                "job_id": "compose-test",
                "status": "completed",
                "executor_id": actor,
                "summary": "Composed final video.",
                "artifacts": (
                    {
                        "filename": "final.mp4",
                        "mime_type": "video/mp4",
                        "download_url": "/api/v1/admin/compositions/compose-test/artifacts/0/download",
                    },
                ),
                "presentation": "final_attachment",
            }

        def is_replay_safe(self, name: str) -> bool:
            return name == "generate_multimedia"

    source = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="multimedia_generator",
        content={
            "result": {
                "artifacts": (
                    {
                        "artifact_id": "source-video-artifact",
                        "download_url": "/api/v1/admin/runs/run/artifacts/source-video-artifact/download",
                        "filename": "shot-001.mp4",
                        "mime_type": "video/mp4",
                        "storage_key": "tenant/run/source/shot-001.mp4",
                    },
                ),
            }
        },
        source_ids=(),
    )
    gateway = FailingTextGateway()
    capabilities = DirectComposeCapabilities()
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="video_compositor",
                role="Video Compositor",
                goal="合并剪辑上游镜头并输出最终成片",
                logical_model="general",
                allowed_tools=("compose_video",),
            ),
        ),
        steps=(
            DispatchStep(
                id="compose",
                agent="video_compositor",
                task="将上游镜头剪辑成最终 5 秒 MP4",
                tools=("compose_video",),
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        allowed_tools=("compose_video",),
        total_token_budget=100,
    )
    runtime = CrewDispatchRuntime(
        gateway,
        plan,
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                artifacts=(source,),
                request="请把已经生成的上游视频剪辑成片，输出 5 秒 MP4",
            )
        )
    ]
    artifacts = tuple(event.artifact for event in events if event.artifact is not None)
    final = next(artifact for artifact in artifacts if artifact.type == "text")

    assert gateway.calls == 0
    assert len(capabilities.calls) == 1
    actor, name, arguments = capabilities.calls[0]
    assert actor == "video_compositor"
    assert name == "compose_video"
    clips = arguments["clips"]
    assert isinstance(clips, tuple)
    first_clip = cast(Mapping[str, JsonValue], clips[0])
    assert first_clip["storage_key"] == "tenant/run/source/shot-001.mp4"
    assert first_clip["mime_type"] == "video/mp4"
    assert first_clip["filename"] == "shot-001.mp4"
    assert "final.mp4" in cast(str, final.content["text"])
    assert any(
        event.kind is EventKind.TOOL_STARTED
        and event.tool_name == "compose_video"
        and event.payload.get("direct_dispatch") is True
        for event in events
    )


async def test_video_compositor_uses_file_handles_from_same_run_generation_lineage() -> None:
    class FailingTextGateway:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_with_context(self, request: ModelRequest) -> GatewayCompletion:
            del request
            self.calls += 1
            raise AssertionError("text gateway must not be called for direct media chain")

    class DirectMediaChainCapabilities(DirectMultimediaCapabilities):
        async def execute(  # type: ignore[no-untyped-def]
            self, *, tenant_id, run_id, actor, name, arguments, idempotency_key
        ) -> Mapping[str, JsonValue]:
            del tenant_id, run_id, idempotency_key
            self.calls.append((actor, name, arguments))
            if name == "generate_multimedia":
                return {
                    "job_id": "media-test",
                    "kind": arguments["kind"],
                    "logical_model": arguments["logical_model"],
                    "status": "completed",
                    "executor_id": actor,
                    "summary": "Generated video artifact with media_primary.",
                    "artifacts": (
                        {
                            "artifact_id": "source-video-artifact",
                            "download_url": (
                                "/api/v1/admin/runs/run/artifacts/"
                                "source-video-artifact/download"
                            ),
                            "filename": "shot-001.mp4",
                            "mime_type": "video/mp4",
                            "storage_key": "tenant/run/source/shot-001.mp4",
                        },
                    ),
                    "presentation": "final_attachment",
                }
            assert name == "compose_video"
            return {
                "job_id": "compose-test",
                "status": "completed",
                "executor_id": actor,
                "summary": "Composed final video.",
                "artifacts": (
                    {
                        "filename": "final.mp4",
                        "mime_type": "video/mp4",
                        "download_url": "/api/v1/admin/compositions/compose-test/artifacts/0/download",
                    },
                ),
                "presentation": "final_attachment",
            }

        def is_replay_safe(self, name: str) -> bool:
            return name == "generate_multimedia"

    gateway = FailingTextGateway()
    capabilities = DirectMediaChainCapabilities()
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="multimedia_generator",
                role="Multimedia Generator",
                goal="生成视频镜头",
                logical_model="general",
                allowed_tools=("generate_multimedia",),
            ),
            AgentSpec(
                id="video_compositor",
                role="Video Compositor",
                goal="合并剪辑上游镜头并输出最终成片",
                logical_model="general",
                allowed_tools=("compose_video",),
            ),
        ),
        steps=(
            DispatchStep(
                id="generate",
                agent="multimedia_generator",
                task="生成 5 秒视频镜头",
                tools=("generate_multimedia",),
                token_budget=100,
            ),
            DispatchStep(
                id="compose",
                agent="video_compositor",
                task="将上游镜头剪辑成最终 5 秒 MP4",
                depends_on=("generate",),
                tools=("compose_video",),
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        allowed_tools=("generate_multimedia", "compose_video"),
        total_token_budget=200,
    )
    runtime = CrewDispatchRuntime(
        gateway,
        plan,
        capability_gateway=capabilities,
        crew_factory=CapturingFactory(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(request="请生成一个 5 秒视频镜头，然后剪辑成最终 MP4")
        )
    ]
    artifacts = tuple(event.artifact for event in events if event.artifact is not None)
    final = next(
        artifact
        for artifact in artifacts
        if artifact.type == "text" and artifact.producer == "video_compositor"
    )

    assert gateway.calls == 0
    assert [call[1] for call in capabilities.calls] == [
        "generate_multimedia",
        "compose_video",
    ]
    compose_arguments = capabilities.calls[-1][2]
    clips = compose_arguments["clips"]
    assert isinstance(clips, tuple)
    first_clip = cast(Mapping[str, JsonValue], clips[0])
    assert first_clip["storage_key"] == "tenant/run/source/shot-001.mp4"
    assert "final.mp4" in cast(str, final.content["text"])


async def test_user_review_gate_requests_approval_before_downstream_step() -> None:
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="character_designer",
                role="Character Designer",
                goal="Generate character references",
                logical_model="general",
            ),
            AgentSpec(
                id="final_synthesizer",
                role="Final Synthesizer",
                goal="Finish after approval",
                logical_model="general",
            ),
        ),
        steps=(
            DispatchStep(
                id="character_model_sheet",
                agent="character_designer",
                task="Generate Character Model Sheet.",
                requires_user_review=True,
                token_budget=100,
            ),
            DispatchStep(
                id="final_response",
                agent="final_synthesizer",
                task="Continue only after the model sheet is approved.",
                depends_on=("character_model_sheet",),
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        total_token_budget=200,
    )
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        crew_factory=CapturingFactory(),
    )

    events = [event async for event in runtime.run(_context(request="生成角色设定后再剪辑成片"))]

    approval = next(event for event in events if event.kind is EventKind.APPROVAL_REQUESTED)
    assert approval.action == "artifact_review"
    assert approval.actor == "character_designer"
    assert approval.payload["stage_id"] == "character_model_sheet"
    assert approval.payload["artifact_id"]
    assert not any(
        event.kind is EventKind.STEP_STARTED and event.step_id == "final_response"
        for event in events
    )
    assert not any(event.kind is EventKind.RUNTIME_COMPLETED for event in events)


async def test_user_review_gate_requests_approval_for_single_media_delivery() -> None:
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="multimedia_generator",
                role="Multimedia Generator",
                goal="Generate reviewed media",
                logical_model="general",
                allowed_tools=("generate_multimedia",),
            ),
        ),
        steps=(
            DispatchStep(
                id="multimedia_generator_step",
                agent="multimedia_generator",
                task="Generate Character Model Sheet.",
                tools=("generate_multimedia",),
                requires_user_review=True,
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        allowed_tools=("generate_multimedia",),
        total_token_budget=100,
    )
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=DirectMultimediaCapabilities(),
        crew_factory=CapturingFactory(),
    )

    events = [event async for event in runtime.run(_context(request="生成角色参考设定表图片"))]

    approval = next(event for event in events if event.kind is EventKind.APPROVAL_REQUESTED)
    checkpoint = next(
        event.checkpoint
        for event in reversed(events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert approval.action == "artifact_review"
    assert approval.actor == "multimedia_generator"
    assert approval.payload["stage_id"] == "multimedia_generator_step"
    assert approval.payload["artifact_id"]
    assert checkpoint.state["phase"] == "waiting_approval"
    assert checkpoint.state["terminal"] is False
    assert not any(event.kind is EventKind.RUNTIME_COMPLETED for event in events)


async def test_rejected_single_media_delivery_reruns_stage_before_completion() -> None:
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="multimedia_generator",
                role="Multimedia Generator",
                goal="Generate reviewed media",
                logical_model="general",
                allowed_tools=("generate_multimedia",),
            ),
        ),
        steps=(
            DispatchStep(
                id="multimedia_generator_step",
                agent="multimedia_generator",
                task="Generate Character Model Sheet.",
                tools=("generate_multimedia",),
                requires_user_review=True,
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        allowed_tools=("generate_multimedia",),
        total_token_budget=100,
    )
    repository = InMemoryArtifactRepository()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=DirectMultimediaCapabilities(),
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )

    events = [event async for event in runtime.run(_context(request="生成角色参考设定表图片"))]
    approval = next(event for event in events if event.kind is EventKind.APPROVAL_REQUESTED)
    checkpoint = next(
        event.checkpoint
        for event in reversed(events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert checkpoint is not None
    rejected_artifact_id = cast(str, approval.payload["artifact_id"])
    stored_artifacts = tuple(
        event.artifact
        for event in events
        if event.kind is EventKind.ARTIFACT_CREATED and event.artifact is not None
    )
    feedback = "角色脸型和服装不一致，退回重新生成角色参考设定表。"
    restored = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=DirectMultimediaCapabilities(),
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )
    await restored.restore_checkpoint(checkpoint)

    restored_events = [
        event
        async for event in restored.run(
            _context(
                checkpoint=checkpoint,
                artifacts=stored_artifacts,
                request="生成角色参考设定表图片",
                routing_decision={
                    "artifact_review_feedback": {
                        "stage_id": "multimedia_generator_step",
                        "artifact_id": rejected_artifact_id,
                        "feedback": feedback,
                    }
                },
            )
        )
    ]

    retry = next(event for event in restored_events if event.kind is EventKind.STEP_RETRYING)
    assert retry.step_id == "multimedia_generator_step"
    assert retry.reason == "user rejected artifact review; regenerating stage"
    assert retry.sequence > approval.sequence
    refreshed_approval = next(
        event for event in restored_events if event.kind is EventKind.APPROVAL_REQUESTED
    )
    assert refreshed_approval.payload["stage_id"] == "multimedia_generator_step"
    assert refreshed_approval.payload["artifact_id"] != rejected_artifact_id
    assert not any(event.kind is EventKind.RUNTIME_COMPLETED for event in restored_events)


async def test_artifact_review_request_includes_file_items_from_upstream_tool_result() -> None:
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="asset_generator",
                role="Asset Generator",
                goal="Generate reviewed asset images.",
                logical_model="general",
                allowed_tools=("generate_multimedia",),
            ),
        ),
        steps=(
            DispatchStep(
                id="asset_generator_step",
                agent="asset_generator",
                task="根据剧本生成全量专业资产图。",
                tools=("generate_multimedia",),
                requires_user_review=True,
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        allowed_tools=("generate_multimedia",),
        total_token_budget=100,
    )
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="script_writer",
        content={
            "text": "林渊释放蓝色电弧，苏清月用银针救人。"
        },
    )
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=StoredAssetPackCapabilities(),
        crew_factory=CapturingFactory(),
        artifact_repository=InMemoryArtifactRepository(),
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                request="先生成剧本，确认后生成全量资产图。",
                artifacts=(script,),
            )
        )
    ]

    approval = next(event for event in events if event.kind is EventKind.APPROVAL_REQUESTED)
    review_items = approval.payload["review_items"]
    assert isinstance(review_items, list | tuple)
    assert len(review_items) >= 8
    titles = {item["title"] for item in review_items}
    assert any(str(title).startswith("角色锁定资产") for title in titles)
    assert "特效资产" in titles
    assert all(item["mime_type"] == "image/png" for item in review_items)
    assert all(item["filename"].endswith(".png") for item in review_items)


async def test_rejected_asset_pack_retries_only_failed_items_from_checkpoint_tool_result() -> None:
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="asset_generator",
                role="Asset Generator",
                goal="Generate full reviewed production asset image pack.",
                logical_model="general",
                allowed_tools=("generate_multimedia",),
            ),
        ),
        steps=(
            DispatchStep(
                id="asset_generator_step",
                agent="asset_generator",
                task="根据剧本生成全量专业资产图。",
                tools=("generate_multimedia",),
                requires_user_review=True,
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        allowed_tools=("generate_multimedia",),
        total_token_budget=100,
    )
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="script_writer",
        content={
            "text": (
                "## 人物小传\n"
                "- **林渊**｜男主，外卖骑手，黑发，灰色冲锋衣。\n"
                "- **苏清月**｜女主，医修，白大褂，低马尾。\n"
                "## 剧本\n"
                "林渊在雨夜街巷释放蓝色电弧，苏清月用银针救人，"
                "赵天霸带武馆弟子追击。"
            )
        },
    )
    repository = InMemoryArtifactRepository()
    first_capabilities = ReviewedAssetPackCapabilities()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=first_capabilities,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                request="先生成剧本，确认后生成全量资产图。",
                artifacts=(script,),
            )
        )
    ]
    approval = next(event for event in events if event.kind is EventKind.APPROVAL_REQUESTED)
    checkpoint = next(
        event.checkpoint
        for event in reversed(events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert checkpoint is not None
    rejected_artifact_id = cast(str, approval.payload["artifact_id"])
    stored_artifacts = tuple(
        event.artifact
        for event in events
        if event.kind is EventKind.ARTIFACT_CREATED and event.artifact is not None
    )

    retry_capabilities = ReviewedAssetPackCapabilities()
    restored = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=retry_capabilities,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )
    await restored.restore_checkpoint(checkpoint)

    restored_events = [
        event
        async for event in restored.run(
            _context(
                checkpoint=checkpoint,
                artifacts=(script, *stored_artifacts),
                request="先生成剧本，确认后生成全量资产图。",
                routing_decision={
                    "artifact_review_feedback": {
                        "stage_id": "asset_generator_step",
                        "artifact_id": rejected_artifact_id,
                        "feedback": "只重试视觉审核失败的动作资产，其他通过资产不要重做。",
                    }
                },
            )
        )
    ]

    retry = next(event for event in restored_events if event.kind is EventKind.STEP_RETRYING)
    assert retry.step_id == "asset_generator_step"
    _actor, name, arguments = retry_capabilities.calls[0]
    assert name == "generate_multimedia"
    assert arguments["artifact_labels"] == ("动作资产",)
    preserved = cast(tuple[Mapping[str, JsonValue], ...], arguments["preserved_artifacts"])
    assert len(preserved) >= 7
    assert "动作资产" not in {item["label"] for item in preserved}
    assert "角色锁定资产：林渊" in {item["label"] for item in preserved}


async def test_rejected_asset_pack_can_be_rejected_twice_without_losing_partial_retry_state() -> None:
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="asset_generator",
                role="Asset Generator",
                goal="Generate full reviewed production asset image pack.",
                logical_model="general",
                allowed_tools=("generate_multimedia",),
            ),
        ),
        steps=(
            DispatchStep(
                id="asset_generator_step",
                agent="asset_generator",
                task="根据剧本生成全量专业资产图。",
                tools=("generate_multimedia",),
                requires_user_review=True,
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        allowed_tools=("generate_multimedia",),
        total_token_budget=100,
    )
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="script_writer",
        content={
            "text": (
                "## 人物小传\n"
                "- **林渊**｜男主，外卖骑手，黑发，灰色冲锋衣。\n"
                "- **苏清月**｜女主，医修，白大褂，低马尾。\n"
                "## 剧本\n"
                "林渊在雨夜街巷释放蓝色电弧，苏清月用银针救人，"
                "赵天霸带武馆弟子追击。"
            )
        },
    )
    repository = InMemoryArtifactRepository()
    first_capabilities = ReviewedAssetPackCapabilities(
        failed_labels=("动作资产", "特效资产"),
        job_id="media-initial",
    )
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=first_capabilities,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                request="先生成剧本，确认后生成全量资产图。",
                artifacts=(script,),
            )
        )
    ]
    first_approval = next(
        event for event in events if event.kind is EventKind.APPROVAL_REQUESTED
    )
    first_checkpoint = next(
        event.checkpoint
        for event in reversed(events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert first_checkpoint is not None
    first_rejected_artifact_id = cast(str, first_approval.payload["artifact_id"])
    first_stored_artifacts = tuple(
        event.artifact
        for event in events
        if event.kind is EventKind.ARTIFACT_CREATED and event.artifact is not None
    )

    first_retry_capabilities = ReviewedAssetPackCapabilities(
        failed_labels=("特效资产",),
        job_id="media-retry-1",
    )
    first_retry_runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=first_retry_capabilities,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )
    await first_retry_runtime.restore_checkpoint(first_checkpoint)
    first_retry_events = [
        event
        async for event in first_retry_runtime.run(
            _context(
                checkpoint=first_checkpoint,
                artifacts=(script, *first_stored_artifacts),
                request="先生成剧本，确认后生成全量资产图。",
                routing_decision={
                    "artifact_review_feedback": {
                        "stage_id": "asset_generator_step",
                        "artifact_id": first_rejected_artifact_id,
                        "feedback": "只重试视觉审核失败的动作资产和特效资产。",
                    }
                },
            )
        )
    ]

    _actor, name, first_retry_arguments = first_retry_capabilities.calls[0]
    assert name == "generate_multimedia"
    assert first_retry_arguments["artifact_labels"] == ("动作资产", "特效资产")
    second_approval = next(
        event for event in first_retry_events if event.kind is EventKind.APPROVAL_REQUESTED
    )
    second_checkpoint = next(
        event.checkpoint
        for event in reversed(first_retry_events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert second_checkpoint is not None
    second_rejected_artifact_id = cast(str, second_approval.payload["artifact_id"])
    second_stored_artifacts = (
        *first_stored_artifacts,
        *tuple(
            event.artifact
            for event in first_retry_events
            if event.kind is EventKind.ARTIFACT_CREATED and event.artifact is not None
        ),
    )

    second_retry_capabilities = ReviewedAssetPackCapabilities(
        failed_labels=(),
        job_id="media-retry-2",
    )
    second_retry_runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=second_retry_capabilities,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )
    await second_retry_runtime.restore_checkpoint(second_checkpoint)

    second_retry_events = [
        event
        async for event in second_retry_runtime.run(
            _context(
                checkpoint=second_checkpoint,
                artifacts=(script, *second_stored_artifacts),
                request="先生成剧本，确认后生成全量资产图。",
                routing_decision={
                    "artifact_review_feedback": {
                        "stage_id": "asset_generator_step",
                        "artifact_id": second_rejected_artifact_id,
                        "feedback": "特效资产仍不符合要求，只重试特效资产。",
                    }
                },
            )
        )
    ]

    retry = next(event for event in second_retry_events if event.kind is EventKind.STEP_RETRYING)
    assert retry.step_id == "asset_generator_step"
    _actor, name, second_retry_arguments = second_retry_capabilities.calls[0]
    assert name == "generate_multimedia"
    assert second_retry_arguments["artifact_labels"] == ("特效资产",)
    preserved = cast(
        tuple[Mapping[str, JsonValue], ...],
        second_retry_arguments["preserved_artifacts"],
    )
    preserved_labels = {item["label"] for item in preserved}
    assert "动作资产" in preserved_labels
    assert "角色锁定资产：林渊" in preserved_labels
    assert "特效资产" not in preserved_labels
    tool_result = next(
        event.artifact
        for event in second_retry_events
        if event.kind is EventKind.TOOL_COMPLETED and event.artifact is not None
    )
    assert tool_result is not None
    merged_items = cast(
        tuple[Mapping[str, JsonValue], ...],
        tool_result.content["result"]["artifacts"],
    )
    merged_labels = {item["label"] for item in merged_items}
    assert "动作资产" in merged_labels
    assert "特效资产" in merged_labels
    assert "角色锁定资产：林渊" in merged_labels
    assert len(merged_labels) >= 9
    assert not any(event.kind is EventKind.RUNTIME_FAILED for event in second_retry_events)

    third_approval = next(
        event for event in second_retry_events if event.kind is EventKind.APPROVAL_REQUESTED
    )
    third_checkpoint = next(
        event.checkpoint
        for event in reversed(second_retry_events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert third_checkpoint is not None
    third_rejected_artifact_id = cast(str, third_approval.payload["artifact_id"])
    third_stored_artifacts = (
        *second_stored_artifacts,
        *tuple(
            event.artifact
            for event in second_retry_events
            if event.kind is EventKind.ARTIFACT_CREATED and event.artifact is not None
        ),
    )

    third_retry_capabilities = ReviewedAssetPackCapabilities(
        failed_labels=(),
        job_id="media-retry-3",
    )
    third_retry_runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        capability_gateway=third_retry_capabilities,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )
    await third_retry_runtime.restore_checkpoint(third_checkpoint)

    third_retry_events = [
        event
        async for event in third_retry_runtime.run(
            _context(
                checkpoint=third_checkpoint,
                artifacts=(script, *third_stored_artifacts),
                request="先生成剧本，确认后生成全量资产图。",
                routing_decision={
                    "artifact_review_feedback": {
                        "stage_id": "asset_generator_step",
                        "artifact_id": third_rejected_artifact_id,
                        "feedback": "动作资产仍混入错误角色，只重试动作资产。",
                    }
                },
            )
        )
    ]

    _actor, name, third_retry_arguments = third_retry_capabilities.calls[0]
    assert name == "generate_multimedia"
    assert third_retry_arguments["artifact_labels"] == ("动作资产",)
    assert not any(event.kind is EventKind.RUNTIME_FAILED for event in third_retry_events)


def test_preserved_multimedia_merge_deduplicates_capability_preserved_results() -> None:
    preserved = (
        {
            "label": "角色锁定资产：林渊",
            "title": "角色锁定资产：林渊",
            "kind": "image",
            "artifact_id": "role-linyuan",
            "storage_key": "tenant/run/role-linyuan.png",
            "filename": "role-linyuan.png",
        },
        {
            "label": "场景资产",
            "title": "场景资产",
            "kind": "image",
            "artifact_id": "scene",
            "storage_key": "tenant/run/scene.png",
            "filename": "scene.png",
        },
    )
    result = {
        "summary": "Generated 1 new image artifacts and reused 2 approved image artifacts.",
        "artifacts": (
            {
                **preserved[0],
                "preserved_from_previous_attempt": True,
            },
            {
                "label": "动作资产",
                "title": "动作资产",
                "kind": "image",
                "artifact_id": "action-new",
                "storage_key": "tenant/run/action-new.png",
                "filename": "action-new.png",
            },
            {
                **preserved[1],
                "preserved_from_previous_attempt": True,
            },
        ),
    }

    merged = adapter_module._merge_preserved_multimedia_result_artifacts(
        result,
        preserved,
        complete_labels=("角色锁定资产：林渊", "动作资产", "场景资产"),
    )

    merged_items = cast(tuple[Mapping[str, JsonValue], ...], merged["artifacts"])
    assert [item["label"] for item in merged_items] == [
        "角色锁定资产：林渊",
        "动作资产",
        "场景资产",
    ]
    assert merged["summary"] == result["summary"]


def test_artifact_review_items_from_lineage_deduplicates_same_file() -> None:
    shared_file = {
        "storage_key": "tenant/run/shared.png",
        "mime_type": "image/png",
        "filename": "shared.png",
        "sha256": "a" * 64,
        "title": "动作资产",
    }
    source = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="asset_generator",
        content={"result": {"artifacts": (shared_file,)}},
    )
    final = Artifact(
        id=uuid4(),
        type="asset_pack",
        producer="asset_generator",
        content={"file": shared_file},
        source_ids=(str(source.id),),
    )

    review_items = adapter_module._artifact_review_items_payload_from_lineage(
        final,
        (source, final),
    )

    assert len(review_items) == 1
    assert review_items[0]["storage_key"] == "tenant/run/shared.png"


def test_direct_multimedia_retry_selection_retries_items_without_visual_review() -> None:
    previous = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="asset_generator",
        content={
            "result": {
                "artifacts": [
                    {
                        "label": "角色锁定资产：林渊",
                        "kind": "image",
                        "filename": "role-linyuan.png",
                    },
                    {
                        "label": "场景资产",
                        "kind": "image",
                        "filename": "scene.png",
                        "visual_review": {"passed": True, "summary": "资产合格"},
                    },
                ]
            }
        },
    )

    selection = _direct_multimedia_retry_selection(
        previous_artifacts=(previous,),
        expected_labels=("角色锁定资产：林渊", "场景资产"),
        feedback_text="资产需要重新确认，缺少审核结论的不应直接保留。",
    )

    assert selection is not None
    assert selection.retry_labels == ("角色锁定资产：林渊",)
    assert [item["label"] for item in selection.preserved_artifacts] == ["场景资产"]


def test_direct_multimedia_retry_selection_prefers_explicit_review_items() -> None:
    previous_id = uuid4()
    previous = Artifact(
        id=previous_id,
        type="tool_result",
        producer="asset_generator",
        content={
            "result": {
                "artifacts": [
                    {
                        "label": "角色锁定资产：秦岚",
                        "kind": "image",
                        "filename": "qinlan.png",
                        "visual_review": {
                            "passed": False,
                            "summary": "纯白背景，无室内、街景、桌面或环境光影等污染背景。",
                        },
                    },
                    {
                        "label": "道具资产",
                        "kind": "image",
                        "filename": "props.png",
                        "visual_review": {
                            "passed": False,
                            "summary": "道具背景复杂且伪字严重。",
                        },
                    },
                    {
                        "label": "镜头资产",
                        "kind": "image",
                        "filename": "shots.png",
                        "visual_review": {"passed": True, "summary": "资产合格"},
                    },
                ]
            }
        },
    )

    selection = _direct_multimedia_retry_selection(
        previous_artifacts=(previous,),
        expected_labels=("角色锁定资产：秦岚", "道具资产", "镜头资产"),
        feedback_text="只重试道具资产。",
        explicit_retry_labels=("道具资产",),
    )

    assert selection is not None
    assert selection.retry_labels == ("道具资产",)
    assert [item["label"] for item in selection.preserved_artifacts] == [
        "角色锁定资产：秦岚",
        "镜头资产",
    ]

    selection_by_review_item_id = _direct_multimedia_retry_selection(
        previous_artifacts=(previous,),
        expected_labels=("角色锁定资产：秦岚", "道具资产", "镜头资产"),
        feedback_text="只重试被退回的文件 id。",
        explicit_retry_labels=(f"{previous_id}:2",),
    )

    assert selection_by_review_item_id is not None
    assert selection_by_review_item_id.retry_labels == ("道具资产",)
    assert [item["label"] for item in selection_by_review_item_id.preserved_artifacts] == [
        "角色锁定资产：秦岚",
        "镜头资产",
    ]


def test_direct_multimedia_retry_selection_ignores_preserved_labels_in_feedback() -> None:
    labels = (
        "角色锁定资产：女主苏清月",
        "角色锁定资产：男主林渊",
        "服装妆造资产",
        "场景资产",
        "道具资产",
    )
    previous = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="asset_generator",
        content={
            "result": {
                "artifacts": [
                    {
                        "label": label,
                        "kind": "image",
                        "filename": f"{index}.png",
                        "visual_review": {"passed": True, "summary": "资产合格"},
                    }
                    for index, label in enumerate(labels, start=1)
                ]
            }
        },
    )

    selection = _direct_multimedia_retry_selection(
        previous_artifacts=(previous,),
        expected_labels=labels,
        feedback_text=(
            "只重试以下失败资产，保留已通过的场景资产、道具资产："
            "角色锁定资产：女主苏清月；角色锁定资产：男主林渊；服装妆造资产。"
        ),
    )

    assert selection is not None
    assert selection.retry_labels == (
        "角色锁定资产：女主苏清月",
        "角色锁定资产：男主林渊",
        "服装妆造资产",
    )
    assert [item["label"] for item in selection.preserved_artifacts] == [
        "场景资产",
        "道具资产",
    ]


def test_direct_multimedia_retry_selection_includes_user_mentioned_passed_assets() -> None:
    previous = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="asset_generator",
        content={
            "result": {
                "artifacts": [
                    {
                        "label": "服装妆造资产",
                        "title": "服装妆造资产",
                        "kind": "image",
                        "visual_review": {
                            "passed": True,
                            "summary": "旧审核放行但背景有污染",
                            "issues": [],
                            "confidence": 0.72,
                        },
                    },
                    {
                        "label": "动作资产",
                        "title": "动作资产",
                        "kind": "image",
                        "visual_review": {
                            "passed": False,
                            "summary": "动作不像分解板",
                            "issues": ["缺少动作分解"],
                            "confidence": 0.46,
                        },
                    },
                    {
                        "label": "场景资产",
                        "title": "场景资产",
                        "kind": "image",
                        "visual_review": {
                            "passed": True,
                            "summary": "场景资产合格",
                            "issues": [],
                            "confidence": 0.9,
                        },
                    },
                ]
            }
        },
    )

    selection = _direct_multimedia_retry_selection(
        previous_artifacts=(previous,),
        expected_labels=("服装妆造资产", "动作资产", "场景资产"),
        feedback_text="服装妆造资产有办公室和桌面背景污染；动作资产也不合格。只重试这两项。",
    )

    assert selection is not None
    assert selection.retry_labels == ("服装妆造资产", "动作资产")
    assert [item["label"] for item in selection.preserved_artifacts] == ["场景资产"]


async def test_rejected_user_review_checkpoint_reruns_stage_before_downstream_step() -> None:
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="character_designer",
                role="Character Designer",
                goal="Generate character references",
                logical_model="general",
            ),
            AgentSpec(
                id="final_synthesizer",
                role="Final Synthesizer",
                goal="Finish after approval",
                logical_model="general",
            ),
        ),
        steps=(
            DispatchStep(
                id="character_model_sheet",
                agent="character_designer",
                task="Generate Character Model Sheet.",
                requires_user_review=True,
                token_budget=100,
            ),
            DispatchStep(
                id="final_response",
                agent="final_synthesizer",
                task="Continue only after the model sheet is approved.",
                depends_on=("character_model_sheet",),
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        total_token_budget=200,
    )
    repository = InMemoryArtifactRepository()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )

    events = [event async for event in runtime.run(_context(request="生成角色设定后再剪辑成片"))]
    approval = next(event for event in events if event.kind is EventKind.APPROVAL_REQUESTED)
    checkpoint = next(
        event.checkpoint
        for event in reversed(events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert checkpoint is not None
    rejected_artifact_id = cast(str, approval.payload["artifact_id"])
    stored_artifacts = tuple(
        event.artifact
        for event in events
        if event.kind is EventKind.ARTIFACT_CREATED and event.artifact is not None
    )
    feedback = "角色脸部一致性不足，重新生成完整 Character Model Sheet。"
    restored_factory = CapturingFactory()
    restored = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        crew_factory=restored_factory,
        artifact_repository=repository,
    )
    await restored.restore_checkpoint(checkpoint)

    restored_events = [
        event
        async for event in restored.run(
            _context(
                checkpoint=checkpoint,
                artifacts=stored_artifacts,
                request="生成角色设定后再剪辑成片",
                routing_decision={
                    "artifact_review_feedback": {
                        "stage_id": "character_model_sheet",
                        "artifact_id": rejected_artifact_id,
                        "feedback": feedback,
                    }
                },
            )
        )
    ]

    started_steps = [
        event.step_id for event in restored_events if event.kind is EventKind.STEP_STARTED
    ]
    assert started_steps == ["character_model_sheet"]
    retry = next(event for event in restored_events if event.kind is EventKind.STEP_RETRYING)
    assert retry.step_id == "character_model_sheet"
    assert retry.reason == "user rejected artifact review; regenerating stage"
    assert retry.payload["attempt"] == 1
    assert retry.payload["artifact_id"] == rejected_artifact_id
    assert retry.payload["feedback"] == feedback
    regenerated = next(
        event.artifact
        for event in restored_events
        if event.kind is EventKind.ARTIFACT_CREATED
        and event.actor == "character_designer"
        and event.artifact is not None
    )
    assert str(regenerated.id) != rejected_artifact_id
    assert feedback in restored_factory.generation.prompts[0]
    refreshed_approval = next(
        event for event in restored_events if event.kind is EventKind.APPROVAL_REQUESTED
    )
    assert refreshed_approval.payload["artifact_id"] == str(regenerated.id)
    assert refreshed_approval.payload["stage_id"] == "character_model_sheet"
    assert not any(
        event.kind is EventKind.STEP_STARTED and event.step_id == "final_response"
        for event in restored_events
    )
    assert not any(event.kind is EventKind.RUNTIME_COMPLETED for event in restored_events)

    regenerated_checkpoint = next(
        event.checkpoint
        for event in reversed(restored_events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert regenerated_checkpoint is not None
    all_artifacts = (
        *stored_artifacts,
        *(
            event.artifact
            for event in restored_events
            if event.kind is EventKind.ARTIFACT_CREATED and event.artifact is not None
        ),
    )
    approved = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )
    await approved.restore_checkpoint(regenerated_checkpoint)

    approved_events = [
        event
        async for event in approved.run(
            _context(
                checkpoint=regenerated_checkpoint,
                artifacts=all_artifacts,
                request="生成角色设定后再剪辑成片",
                routing_decision={
                    "media_pipeline_plan": {
                        "approved_artifacts": (
                            {
                                "stage_id": "character_model_sheet",
                                "artifact_id": str(regenerated.id),
                            }
                        ),
                    }
                },
            )
        )
    ]

    assert any(
        event.kind is EventKind.STEP_STARTED and event.step_id == "final_response"
        for event in approved_events
    )
    assert approved_events[-1].kind is EventKind.RUNTIME_COMPLETED


async def test_user_rejection_after_reviewer_revision_reruns_stage_cleanly() -> None:
    plan = DispatchPlan(
        agents=(
            AgentSpec(id="writer", role="Writer", goal="Draft asset", logical_model="general"),
            AgentSpec(id="critic", role="Critic", goal="Review asset", logical_model="general"),
            AgentSpec(id="final", role="Final", goal="Finish", logical_model="general"),
        ),
        steps=(
            DispatchStep(
                id="storyboard",
                agent="writer",
                task="Draft storyboard.",
                reviewer="critic",
                reviewer_retries=1,
                requires_user_review=True,
                token_budget=100,
            ),
            DispatchStep(
                id="final_response",
                agent="final",
                task="Finish after approved storyboard.",
                depends_on=("storyboard",),
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        total_token_budget=300,
    )
    repository = InMemoryArtifactRepository()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(
            (
                '{"verdict":"revise","feedback":"补齐缺失镜头。"}',
                '{"verdict":"approve"}',
            )
        ),
        plan,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )

    events = [event async for event in runtime.run(_context(request="先出分镜图，再剪辑成片"))]
    approval = next(event for event in events if event.kind is EventKind.APPROVAL_REQUESTED)
    checkpoint = next(
        event.checkpoint
        for event in reversed(events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert checkpoint is not None
    assert any(
        event.kind is EventKind.REVIEW_COMPLETED
        and event.payload.get("verdict") == "revise"
        for event in events
    )
    assert checkpoint.state["retries"] == {"storyboard": 1}
    stored_artifacts = tuple(
        event.artifact
        for event in events
        if event.kind is EventKind.ARTIFACT_CREATED and event.artifact is not None
    )
    restored_factory = CapturingFactory()
    restored = CrewDispatchRuntime(
        ReviewAwareGateway(('{"verdict":"approve"}',)),
        plan,
        crew_factory=restored_factory,
        artifact_repository=repository,
    )
    await restored.restore_checkpoint(checkpoint)

    restored_events = [
        event
        async for event in restored.run(
            _context(
                checkpoint=checkpoint,
                artifacts=stored_artifacts,
                request="先出分镜图，再剪辑成片",
                routing_decision={
                    "artifact_review_feedback": {
                        "stage_id": "storyboard",
                        "artifact_id": cast(str, approval.payload["artifact_id"]),
                        "feedback": "分镜节奏不对，重新生成。",
                    }
                },
            )
        )
    ]

    assert any(
        event.kind is EventKind.STEP_STARTED
        and event.step_id == "storyboard"
        and event.payload["attempt"] == 1
        for event in restored_events
    )
    assert "分镜节奏不对" in restored_factory.generation.prompts[0]
    refreshed_checkpoint = next(
        event.checkpoint
        for event in reversed(restored_events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert refreshed_checkpoint is not None
    assert refreshed_checkpoint.state["retries"] == {"storyboard": 0}
    assert any(event.kind is EventKind.APPROVAL_REQUESTED for event in restored_events)
    await CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    ).restore_checkpoint(refreshed_checkpoint)


def test_direct_multimedia_generation_prompt_includes_user_review_feedback() -> None:
    step = DispatchStep(
        id="character_model_sheet",
        agent="multimedia_generator",
        task="生成角色 Character Model Sheet 图片。",
        token_budget=100,
    )

    prompt = _direct_multimedia_generation_prompt(
        _context(request="生成女主角定妆设定表"),
        step,
        (),
        "服装和脸型不一致，按原角色设定重新生成。",
    )

    assert "用户审核退回意见" in prompt
    assert "服装和脸型不一致" in prompt


def test_direct_multimedia_generation_prompt_constrains_character_model_sheet() -> None:
    step = DispatchStep(
        id="character_model_sheet",
        agent="multimedia_generator",
        task="生成 Character Model Sheet 形式的角色参考设定表图片。",
        token_budget=100,
    )

    prompt = _direct_multimedia_generation_prompt(
        _context(request="生成女主角角色参考设定表，风格参考定妆照，不要太细节"),
        step,
        (),
    )

    assert "角色定妆照" in prompt
    assert "一张图只包含一个角色" in prompt
    assert "不要把多个角色放在同一张设定表" in prompt
    assert "不要过度堆叠小物件" in prompt
    assert "重复近景头像" in prompt
    assert "与角色设定无关的食物" in prompt


@pytest.mark.parametrize(
    ("task_text", "expected_kind"),
    [
        ("给我做一张图片版设定板。", "image"),
        ("根据剧情以Character Model Sheet的形式生成角色参考设定表。", "image"),
        ("根据这段视频剧情生成 Character Model Sheet 形式的角色参考设定表。", "image"),
        (
            (
                "基于刚才剧本，只生成 Character Model Sheet 形式的角色参考设定表和角色服装设定板图片，"
                "不要生成视频，不要剪辑成片。"
            ),
            "image",
        ),
        ("生成角色设定表。", "image"),
        ("根据剧本为每个角色生成角色参考图。", "image"),
        ("根据剧本为每个人物生成定妆图。", "image"),
        ("根据剧本生成男女主角同框合照。", "image"),
        ("出一张赛博朋克产品概念图。", "image"),
        ("生成三张可下载表情包贴纸。", "image"),
        ("做一张商品 3D 渲染图。", "image"),
        ("把这个故事做成 8 秒动画短片成片。", "video"),
        ("根据分镜剪辑成片。", "video"),
        ("先生成角色参考设定表、服装设定板和分镜图，最终剪辑成片。", "video"),
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


async def test_dispatch_tool_step_timeout_retries_with_compact_recovery_prompt() -> None:
    artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="researcher",
        content={"text": "source material for a long report " * 500},
    )
    factory = StepTimeoutOnceFactory()
    runtime = CrewDispatchRuntime(
        DocumentToolGateway(),
        _one_step_tool_plan(tools=("document.generate_docx",)),
        crew_factory=factory,
        capability_gateway=ReplaySafeDocumentCapabilities(),
    )

    events = [event async for event in runtime.run(_context(artifacts=(artifact,)))]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    assert factory.generation.calls == 2
    retry = next(event for event in events if event.kind is EventKind.STEP_RETRYING)
    assert retry.actor == "writer"
    assert retry.reason == "step execution timed out; retrying with compact recovery"
    assert retry.payload["attempt"] == 2
    assert retry.payload["strategy"] == "compact_retry"
    assert retry.payload["input_policy"] == "compact_source_previews"
    assert retry.payload["error_code"] == "crew.step_timeout"
    assert "compact_retry" in factory.generation.prompts[1]
    assert len(factory.generation.prompts[1].encode("utf-8")) < len(
        factory.generation.prompts[0].encode("utf-8")
    )


async def test_dispatch_step_timeout_after_model_call_drops_stale_attempt_ledger() -> None:
    artifact = Artifact(
        id=uuid4(),
        type="text",
        producer="researcher",
        content={"text": "large source context " * 500},
    )
    factory = StepTimeoutAfterModelCallFactory()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        _one_step_plan(),
        crew_factory=factory,
    )

    events = [event async for event in runtime.run(_context(artifacts=(artifact,)))]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    assert factory.generation.calls == 2
    assert any(
        event.kind is EventKind.STEP_RETRYING
        and event.reason == "step execution timed out; retrying with compact recovery"
        for event in events
    )
    assert not any(
        event.reason is not None and "model request changed after checkpoint" in event.reason
        for event in events
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


async def test_dispatch_step_sanitized_empty_model_response_retries_before_failing() -> None:
    gateway = ControlCharsThenSuccessGateway()
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


async def test_dispatch_step_gateway_empty_response_error_retries_before_failing() -> None:
    gateway = EmptyErrorThenSuccessGateway()
    repository = InMemoryArtifactRepository()
    runtime = CrewDispatchRuntime(
        gateway,
        _one_step_plan(),
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )

    events = [event async for event in runtime.run(_context())]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    assert gateway.calls == 2
    retry = next(event for event in events if event.kind is EventKind.STEP_RETRYING)
    assert retry.actor == "writer"
    assert retry.reason == "model returned empty response; retrying with explicit output request"
    assert retry.payload["strategy"] == "empty_response_retry"
    assert retry.payload["error_code"] == "model.empty_response"
    assert retry.payload["logical_models"] == "qwen"
    assert retry.payload["deployments"] == "qwen_1"
    checkpoint = next(
        event.checkpoint
        for event in reversed(events)
        if event.kind is EventKind.CHECKPOINT_SAVED and event.checkpoint is not None
    )
    assert checkpoint is not None
    model_states = cast(Mapping[str, Mapping[str, object]], checkpoint.state["models"])
    assert len(model_states) == 1
    assert next(iter(model_states.values()))["status"] == "succeeded"
    restored_gateway = EmptyErrorThenSuccessGateway()
    restored = CrewDispatchRuntime(
        restored_gateway,
        _one_step_plan(),
        crew_factory=CapturingFactory(),
        artifact_repository=repository,
    )
    await restored.restore_checkpoint(checkpoint)
    stored_artifacts = tuple(
        event.artifact
        for event in events
        if event.kind is EventKind.ARTIFACT_CREATED and event.artifact is not None
    )
    restored_events = [
        event
        async for event in restored.run(
            _context(checkpoint=checkpoint, artifacts=stored_artifacts)
        )
    ]
    assert [event.kind for event in restored_events] == [EventKind.RUNTIME_COMPLETED]
    assert restored_gateway.calls == 0
    await CrewDispatchRuntime(
        EmptyErrorThenSuccessGateway(),
        _one_step_plan(),
        crew_factory=CapturingFactory(),
    ).restore_checkpoint(checkpoint)


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


async def test_reviewer_prompt_adds_character_sheet_acceptance_criteria() -> None:
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="multimedia_generator",
                role="Multimedia Generator",
                goal="Generate reviewed media",
                logical_model="general",
            ),
            AgentSpec(id="critic", role="critic", goal="Review", logical_model="general"),
        ),
        steps=(
            DispatchStep(
                id="character_model_sheet",
                agent="multimedia_generator",
                task="Generate Character Model Sheet.",
                reviewer="critic",
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        total_token_budget=200,
    )
    factory = CapturingFactory()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        crew_factory=factory,
    )

    events = [
        event
        async for event in runtime.run(
            _context(request="为男女主生成角色参考设定表，风格全是写实，不要太细节也不要太简化")
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    reviewer_prompt = next(prompt for prompt in factory.generation.prompts if "REVIEWER" in prompt)
    assert "角色参考设定表审核标准" in reviewer_prompt
    assert "至少应有 2 张独立角色图片" in reviewer_prompt
    assert "一张图片只允许一个角色" in reviewer_prompt
    assert "同一人物身份必须一致" in reviewer_prompt
    assert "同一画风" in reviewer_prompt
    assert "全写实" in reviewer_prompt


async def test_reviewer_prompt_adds_source_character_targets_to_acceptance_criteria() -> None:
    script = Artifact(
        id=uuid4(),
        type="script",
        producer="copywriter",
        content={
            "text": (
                "## 女主：苏念（26岁）\n"
                "- 外貌：黑长直，浅粉针织衫。\n\n"
                "## 男主：陆沉（29岁）\n"
                "- 外貌：短黑发，灰色西装。\n\n"
                "## 闺蜜：林小鹿（25岁）\n"
                "- 外貌：短发，牛仔外套。"
            )
        },
    )
    plan = DispatchPlan(
        agents=(
            AgentSpec(
                id="multimedia_generator",
                role="Multimedia Generator",
                goal="Generate reviewed media",
                logical_model="general",
            ),
            AgentSpec(id="critic", role="critic", goal="Review", logical_model="general"),
        ),
        steps=(
            DispatchStep(
                id="character_model_sheet",
                agent="multimedia_generator",
                task="Generate Character Model Sheet.",
                reviewer="critic",
                final_synthesizer=True,
                token_budget=100,
            ),
        ),
        total_token_budget=200,
    )
    factory = CapturingFactory()
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(),
        plan,
        crew_factory=factory,
    )

    events = [
        event
        async for event in runtime.run(
            _context(
                request="为每个角色生成角色参考设定表，风格全是写实",
                artifacts=(script,),
            )
        )
    ]

    assert events[-1].kind is EventKind.RUNTIME_COMPLETED
    reviewer_prompt = next(prompt for prompt in factory.generation.prompts if "REVIEWER" in prompt)
    assert "至少应有 3 张独立角色图片" in reviewer_prompt
    assert "女主" in reviewer_prompt
    assert "男主" in reviewer_prompt
    assert "闺蜜" in reviewer_prompt
    assert "中等复杂度" in reviewer_prompt


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


async def test_reviewer_chinese_consensus_rejection_requests_step_revision() -> None:
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(
            (
                (
                    "# Skeptic 审查结论\n\n"
                    "## [CONSENSUS] 不通过——现有交付物为残缺品，拒绝放行\n\n"
                    "核心问题：产物被截断，剧本不完整。需要退回重新生成完整剧本。"
                ),
                '{"verdict":"approve"}',
            )
        ),
        _reviewed_plan_with_retry_budget(),
        crew_factory=CapturingFactory(),
    )

    events = [event async for event in runtime.run(_context())]

    revisions = [
        event
        for event in events
        if event.kind is EventKind.REVIEW_COMPLETED
        and event.actor == "critic"
        and event.payload.get("verdict") == "revise"
    ]
    assert len(revisions) == 1
    assert "产物被截断" in str(revisions[0].payload["feedback"])
    retry = next(
        event
        for event in events
        if event.kind is EventKind.STEP_RETRYING
        and event.actor == "writer"
        and event.reason == "review requested revision"
    )
    assert retry.payload["attempt"] == 2
    assert events[-1].kind is EventKind.RUNTIME_COMPLETED


async def test_reviewer_plain_chinese_rejection_requests_step_revision() -> None:
    runtime = CrewDispatchRuntime(
        ReviewAwareGateway(
            (
                "未通过，需要重新生成角色参考图。主定妆照和表情图不像同一个人。",
                '{"verdict":"approve"}',
            )
        ),
        _reviewed_plan_with_retry_budget(),
        crew_factory=CapturingFactory(),
    )

    events = [event async for event in runtime.run(_context())]

    revisions = [
        event
        for event in events
        if event.kind is EventKind.REVIEW_COMPLETED
        and event.actor == "critic"
        and event.payload.get("verdict") == "revise"
    ]
    assert len(revisions) == 1
    assert "不像同一个人" in str(revisions[0].payload["feedback"])
    skipped = [
        event
        for event in events
        if event.kind is EventKind.REVIEW_COMPLETED
        and event.actor == "critic"
        and event.payload.get("review_status") == "skipped"
    ]
    assert skipped == []
    assert events[-1].kind is EventKind.RUNTIME_COMPLETED


def test_plain_chinese_review_rejection_fallback_does_not_require_marker() -> None:
    fallback = _fallback_review_response_from_text(
        "未通过，需要重新生成角色参考图。主定妆照和表情图不像同一个人。"
    )

    assert fallback is not None
    verdict, feedback = fallback
    assert verdict == "revise"
    assert feedback is not None
    assert "不像同一个人" in feedback


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


def test_artifact_review_items_payload_exposes_each_generated_file_for_review() -> None:
    artifact = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="character_designer",
        content={
            "result": {
                "artifacts": (
                    {
                        "storage_key": "tenant/run/artifact/male.png",
                        "mime_type": "image/png",
                        "filename": "male-lead-model-sheet.png",
                        "sha256": "a" * 64,
                        "title": "男主角色参考设定表",
                    },
                    {
                        "storage_key": "tenant/run/artifact/female.png",
                        "mime_type": "image/png",
                        "filename": "female-lead-model-sheet.png",
                        "sha256": "b" * 64,
                        "title": "女主角色参考设定表",
                    },
                ),
            },
        },
    )

    payload = _artifact_review_items_payload(artifact)

    assert payload == (
        {
            "id": f"{artifact.id}:1",
            "artifact_id": str(artifact.id),
            "kind": "tool_result",
            "storage_key": "tenant/run/artifact/male.png",
            "mime_type": "image/png",
            "filename": "male-lead-model-sheet.png",
            "sha256": "a" * 64,
            "title": "男主角色参考设定表",
        },
        {
            "id": f"{artifact.id}:2",
            "artifact_id": str(artifact.id),
            "kind": "tool_result",
            "storage_key": "tenant/run/artifact/female.png",
            "mime_type": "image/png",
            "filename": "female-lead-model-sheet.png",
            "sha256": "b" * 64,
            "title": "女主角色参考设定表",
        },
    )


def test_artifact_review_items_payload_includes_failed_multimedia_items() -> None:
    artifact = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="asset_generator",
        content={
            "result": {
                "artifacts": (
                    {
                        "kind": "image",
                        "label": "角色锁定资产：女主苏清月",
                        "title": "角色锁定资产：女主苏清月",
                        "generation_error": "image generation timed out",
                        "visual_review": {
                            "passed": False,
                            "summary": "生成失败，需单项重试。",
                            "issues": ("image generation timed out",),
                        },
                    },
                    {
                        "kind": "image",
                        "storage_key": "tenant/run/scene.png",
                        "mime_type": "image/png",
                        "filename": "scene.png",
                        "sha256": "c" * 64,
                        "label": "场景资产",
                        "title": "场景资产",
                    },
                ),
            },
        },
    )

    payload = _artifact_review_items_payload(artifact)

    assert payload[0] == {
        "id": f"{artifact.id}:1",
        "artifact_id": str(artifact.id),
        "kind": "image",
        "title": "角色锁定资产：女主苏清月",
        "label": "角色锁定资产：女主苏清月",
        "generation_error": "image generation timed out",
        "visual_review_summary": "生成失败，需单项重试。",
    }
    assert payload[1]["id"] == f"{artifact.id}:2"
    assert payload[1]["title"] == "场景资产"
    assert payload[1]["storage_key"] == "tenant/run/scene.png"


def test_artifact_review_feedback_text_includes_rejected_file_items() -> None:
    artifact_id = uuid4()
    feedback = _artifact_review_feedback_from_routing(
        {
            "artifact_review_feedback": {
                "stage_id": "character_model_sheet",
                "artifact_id": str(artifact_id),
                "feedback": "部分角色设定图需要重做。",
                "review_items": (
                    {
                        "id": f"{artifact_id}:2",
                        "artifact_id": str(artifact_id),
                        "filename": "female-lead-model-sheet.png",
                        "sha256": "b" * 64,
                        "title": "女主角色参考设定表",
                        "feedback": "女主没有按设定生成单人参考表。",
                    },
                ),
            },
        }
    )

    assert feedback is not None
    text = _artifact_review_feedback_text(feedback)
    assert "部分角色设定图需要重做" in text
    assert "female-lead-model-sheet.png" in text
    assert "女主没有按设定生成单人参考表" in text


def test_usable_file_artifacts_payload_exposes_generated_file_handles() -> None:
    artifact = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="multimedia_generator",
        content={
            "result": {
                "file": {
                    "storage_key": "tenant/run/artifact/shot.mp4",
                    "mime_type": "video/mp4",
                    "filename": "shot.mp4",
                    "download_url": "/api/v1/admin/runs/run/artifacts/artifact/download",
                    "artifact_id": "artifact-001",
                },
                "metadata": {
                    "storage_key": "tenant/run/artifact/shot.mp4",
                    "mime_type": "video/mp4",
                    "filename": "shot.mp4",
                },
                "artifacts": (
                    {
                        "file": {
                            "storage_key": "tenant/run/artifact/storyboard.png",
                            "mime_type": "image/png",
                            "filename": "storyboard.png",
                        },
                    },
                ),
            },
        },
    )

    payload = _usable_file_artifacts_payload((artifact,))

    assert payload == (
        {
            "source_artifact_id": str(artifact.id),
            "source_producer": "multimedia_generator",
            "storage_key": "tenant/run/artifact/shot.mp4",
            "mime_type": "video/mp4",
            "filename": "shot.mp4",
            "artifact_id": "artifact-001",
            "download_url": "/api/v1/admin/runs/run/artifacts/artifact/download",
        },
        {
            "source_artifact_id": str(artifact.id),
            "source_producer": "multimedia_generator",
            "storage_key": "tenant/run/artifact/storyboard.png",
            "mime_type": "image/png",
            "filename": "storyboard.png",
        },
    )


def test_compose_video_arguments_use_upstream_file_handles() -> None:
    source = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="multimedia_generator",
        content={
            "result": {
                "file": {
                    "storage_key": (
                        "00000000-0000-4000-8000-000000000001/run/artifact/"
                        "kling_kling-v3-omni-video-generation.mp4"
                    ),
                    "mime_type": "video/mp4",
                    "filename": "kling_kling-v3-omni-video-generation.mp4",
                    "artifact_id": "video-artifact",
                    "download_url": "/api/v1/admin/runs/run/artifacts/video-artifact/download",
                }
            }
        },
    )
    arguments: Mapping[str, JsonValue] = {
        "title": "test cut",
        "filename": "test-cut.mp4",
        "clips": (
            {
                "storage_key": str(source.id),
                "mime_type": "video/mp4",
                "filename": "guessed.mp4",
                "duration_seconds": 5,
            },
        ),
    }

    normalized = _normalize_compose_video_arguments_with_sources(arguments, (source,))
    clips = cast(tuple[Mapping[str, JsonValue], ...], normalized["clips"])

    assert clips[0]["storage_key"] == (
        "00000000-0000-4000-8000-000000000001/run/artifact/"
        "kling_kling-v3-omni-video-generation.mp4"
    )
    assert clips[0]["mime_type"] == "video/mp4"
    assert clips[0]["filename"] == "kling_kling-v3-omni-video-generation.mp4"
    assert clips[0]["duration_seconds"] == 5


def test_direct_compose_video_arguments_resolves_file_handles_from_lineage_pool() -> None:
    tool_artifact = Artifact(
        id=uuid4(),
        type="tool_result",
        producer="multimedia_generator",
        content={
            "result": {
                "artifacts": (
                    {
                        "artifact_id": "source-video-artifact",
                        "download_url": "/api/v1/admin/runs/run/artifacts/source-video-artifact/download",
                        "filename": "shot-001.mp4",
                        "mime_type": "video/mp4",
                        "storage_key": "tenant/run/source/shot-001.mp4",
                    },
                ),
            }
        },
        source_ids=(),
    )
    dependency_output = Artifact(
        id=uuid4(),
        type="text",
        producer="multimedia_generator",
        content={"text": "Generated downloadable artifact shot-001.mp4 (video/mp4)."},
        source_ids=(str(tool_artifact.id),),
    )
    step = DispatchStep(
        id="compose",
        agent="video_compositor",
        task="将上游镜头剪辑成最终 5 秒 MP4",
        tools=("compose_video",),
        final_synthesizer=True,
        token_budget=100,
    )

    arguments = _direct_compose_video_arguments(
        step,
        (dependency_output,),
        available_artifacts=(dependency_output, tool_artifact),
    )

    assert arguments is not None
    clips = arguments["clips"]
    assert isinstance(clips, tuple)
    first_clip = cast(Mapping[str, JsonValue], clips[0])
    assert first_clip["storage_key"] == "tenant/run/source/shot-001.mp4"
    assert first_clip["mime_type"] == "video/mp4"
    assert first_clip["filename"] == "shot-001.mp4"
