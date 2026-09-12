from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from agent_hub.domain.runs import RunStatus, TaskMode
from agent_hub.routing.types import (
    EXECUTABLE_MODES,
    RiskLevel,
    RouteAssessment,
    RouteDecision,
    RouteSource,
)
from agent_hub.runs.repository import RunRecord, _status_can_seed_conversation_mode
from agent_hub.runs.service import (
    HermesMemoryInjection,
    HermesRunAdvice,
    HermesRunOutcome,
    HermesSkippedMemory,
    RunService,
    _local_main_agent_auto_mode,
    _local_schedule_proposal,
)
from agent_hub.runtime.defaults import UnavailableRuntime
from agent_hub.runtime.registry import RuntimeRegistry


class RecordingQueue:
    def __init__(self) -> None:
        self.enqueued: list[UUID] = []

    async def enqueue_run(self, run_id: UUID, *, idempotency_key: str) -> None:
        del idempotency_key
        self.enqueued.append(run_id)


class WaitingRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def route(self, task_text: object) -> RouteDecision:
        del task_text
        self.calls += 1
        return RouteDecision(
            mode=None,
            needs_user_choice=True,
            status="waiting_user_mode",
            assessments=(),
            clarification_reason="classification_unavailable",
            options=EXECUTABLE_MODES,
            decision_token="safe-decision-token-abcdefghijklmnopqrstuvwxyz1234",
            version=1,
            risk=RiskLevel.LOW,
            requires_approval=False,
            permissions_still_apply=True,
        )


class UserChoiceRouter:
    async def route(self, task_text: object) -> RouteDecision:
        del task_text
        return RouteDecision(
            mode=None,
            needs_user_choice=True,
            status="waiting_user_mode",
            assessments=(),
            clarification_reason="routing_requires_user_choice",
            options=EXECUTABLE_MODES,
            decision_token="safe-decision-token-abcdefghijklmnopqrstuvwxyz1234",
            version=1,
            risk=RiskLevel.LOW,
            requires_approval=False,
            permissions_still_apply=True,
        )


class ReadyDispatchRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def route(self, task_text: object) -> RouteDecision:
        del task_text
        self.calls += 1
        return RouteDecision(
            mode=TaskMode.DISPATCH,
            needs_user_choice=False,
            status="ready",
            assessments=(
                RouteAssessment(
                    mode=TaskMode.DISPATCH,
                    confidence=0.91,
                    reason="misclassified as executable planning",
                    roles=("planner",),
                    estimated_seconds=120,
                    estimated_cost_usd=Decimal("0.10"),
                    risk=RiskLevel.LOW,
                    source=RouteSource.CLASSIFIER,
                    logical_model="router",
                    deployment_id="router-test",
                    provider_id="test",
                ),
            ),
            options=(),
            decision_token=None,
            version=1,
            risk=RiskLevel.LOW,
            requires_approval=False,
            permissions_still_apply=True,
        )


class ConversationModeRepository:
    def __init__(self, previous_mode: TaskMode | None) -> None:
        self.previous_mode = previous_mode
        self.created: list[dict[str, object]] = []

    async def latest_resolved_mode_for_conversation(
        self,
        *,
        tenant_id: UUID,
        actor_id: UUID,
        conversation_id: str,
    ) -> TaskMode | None:
        del tenant_id, actor_id, conversation_id
        return self.previous_mode

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
    def __init__(self, advice: HermesRunAdvice | None) -> None:
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
    ) -> HermesRunAdvice | None:
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

    async def record_outcome(self, outcome: HermesRunOutcome) -> None:
        del outcome


class SlowHermesAdvisor(RecordingHermesAdvisor):
    async def advise(self, **kwargs: object) -> HermesRunAdvice | None:
        await asyncio.sleep(2)
        return None


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        (RunStatus.QUEUED, True),
        (RunStatus.PLANNING, True),
        (RunStatus.RUNNING, True),
        (RunStatus.RETRYING, True),
        (RunStatus.SYNTHESIZING, True),
        (RunStatus.COMPLETED, True),
        (RunStatus.FAILED, False),
        (RunStatus.WAITING_USER_MODE, False),
        (RunStatus.WAITING_APPROVAL, False),
        (RunStatus.PAUSED, False),
        (RunStatus.CANCELLED, False),
    ),
)
def test_conversation_mode_seed_status_boundary(status: RunStatus, expected: bool) -> None:
    assert _status_can_seed_conversation_mode(status) is expected


async def test_auto_submission_reuses_previous_mode_for_same_conversation_without_reasking() -> None:
    repository = ConversationModeRepository(TaskMode.HYBRID)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.HYBRID),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="预算是多少",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-continuation",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.HYBRID
    assert submitted.clarification_reason is None
    assert router.calls == 0
    assert repository.created[0]["routing_decision"] == {
        "reason": "conversation_mode_continuation",
        "main_agent_selected_mode": "hybrid",
        "mode_source": "previous_conversation_run",
        "selected_agent_ids": [],
        "workflow_id": None,
        "allow_workflow_adjustment": False,
        "workflow_adjustment_policy": "strict_preset",
        "conversation_id": "conv-1",
        "reference_conversation_id": None,
        "attachment_ids": [],
    }


async def test_submit_extracts_explicit_file_reference_into_routing_context() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=WaitingRouter(),
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="请结合 @file:handoff.md 继续任务",
        mode=TaskMode.DIRECT,
        skip_evolution_proposal=True,
    )

    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, Mapping)
    assert submitted.status is RunStatus.QUEUED
    assert routing["requested_files"] == "handoff.md"


async def test_script_request_records_long_lived_media_pipeline_plan_without_slots() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.HYBRID),)),
        router=WaitingRouter(),
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="先生成一个短剧剧本，后续我可能要生成角色参考设定表、分镜图并剪辑成片。",
        mode=TaskMode.HYBRID,
        conversation_id="conv-video-plan",
        idempotency_key="idem-video-plan",
    )

    assert submitted.status is RunStatus.QUEUED
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    plan = routing.get("media_pipeline_plan")
    assert isinstance(plan, dict)
    assert plan["status"] == "planned"
    assert plan["execution_slots"] == []
    assert [stage["id"] for stage in plan["stages"]] == [
        "script",
        "character_model_sheet",
        "costume_sheet",
        "scene_prop_assets",
        "storyboard",
        "shot_videos",
        "edit_decision_list",
        "compose_video",
    ]
    assert plan["stages"][0]["status"] == "completed"
    assert all(stage["status"] == "planned" for stage in plan["stages"][1:])


async def test_character_sheet_from_story_context_does_not_create_media_pipeline_plan() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DISPATCH),)),
        router=WaitingRouter(),
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="根据这个剧情生成 Character Model Sheet 形式的角色参考设定表。",
        mode=TaskMode.DISPATCH,
        conversation_id="conv-character-sheet-only",
        idempotency_key="idem-character-sheet-only",
    )

    assert submitted.status is RunStatus.QUEUED
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert "media_pipeline_plan" not in routing


async def test_character_sheet_followup_from_script_context_does_not_create_media_pipeline_plan() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DISPATCH),)),
        router=WaitingRouter(),
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message=(
            "基于刚才剧本，只生成 Character Model Sheet 形式的角色参考设定表和角色服装设定板图片，"
            "不要生成视频，不要剪辑成片。"
        ),
        mode=TaskMode.DISPATCH,
        conversation_id="conv-character-sheet-followup",
        idempotency_key="idem-character-sheet-followup",
    )

    assert submitted.status is RunStatus.QUEUED
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert "media_pipeline_plan" not in routing


async def test_character_makeup_reference_without_concrete_script_records_script_first_plan() -> None:
    repository = ConversationModeRepository(TaskMode.DIRECT)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DISPATCH),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="根据剧本为每个角色生成定妆参考图。",
        mode=TaskMode.AUTO,
        conversation_id="conv-character-makeup-reference",
        idempotency_key="idem-character-makeup-reference",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DISPATCH
    assert router.calls == 0
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "current_artifact_delivery_request"
    plan = routing.get("media_pipeline_plan")
    assert isinstance(plan, dict)
    assert plan["source"] == "unresolved_script_reference"
    assert [stage["id"] for stage in plan["stages"][:2]] == ["script", "character_model_sheet"]
    assert plan["stages"][0]["status"] == "planned"
    assert plan["stages"][1]["status"] == "planned"


async def test_auto_reuses_previous_mode_when_discussion_is_context() -> None:
    repository = ConversationModeRepository(TaskMode.HYBRID)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.HYBRID),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="继续刚刚的方案，用上一轮讨论结论补充执行细节",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-continuation-discussion-word",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.HYBRID
    assert router.calls == 0
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "conversation_mode_continuation"


async def test_auto_reuses_previous_mode_when_mixed_model_is_context() -> None:
    repository = ConversationModeRepository(TaskMode.HYBRID)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.HYBRID),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="继续解释刚刚说的混合模型为什么会被识别错",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-continuation-mixed-model-word",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.HYBRID
    assert router.calls == 0
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "conversation_mode_continuation"


async def test_auto_submission_does_not_reuse_previous_direct_mode_for_current_media_delivery() -> None:
    repository = ConversationModeRepository(TaskMode.DIRECT)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DISPATCH),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="满意，按刚刚方案生成图片",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-current-media-delivery",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DISPATCH
    assert router.calls == 0
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "current_artifact_delivery_request"
    assert routing["main_agent_selected_mode"] == "dispatch"
    assert routing["mode_source"] == "current_user_request"


async def test_auto_submission_routes_character_model_sheet_as_current_media_delivery() -> None:
    repository = ConversationModeRepository(TaskMode.DIRECT)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DISPATCH),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="根据剧情以Character Model Sheet的形式生成角色参考设定表",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-current-character-model-sheet-delivery",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DISPATCH
    assert router.calls == 0
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "current_artifact_delivery_request"
    assert routing["main_agent_selected_mode"] == "dispatch"
    assert routing["mode_source"] == "current_user_request"


async def test_auto_submission_does_not_reuse_previous_discuss_mode_for_current_office_delivery() -> None:
    repository = ConversationModeRepository(TaskMode.DISCUSS)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DISPATCH),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="确认，生成一份 PPT 作为最终文件",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-current-office-delivery",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DISPATCH
    assert router.calls == 0
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "current_artifact_delivery_request"
    assert routing["main_agent_selected_mode"] == "dispatch"
    assert routing["mode_source"] == "current_user_request"


async def test_auto_submission_switches_mode_when_user_explicitly_requests_it() -> None:
    repository = ConversationModeRepository(TaskMode.HYBRID)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DISCUSS),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="这轮切换到讨论模式",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-mode-switch",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DISCUSS
    assert router.calls == 0
    assert repository.created[0]["routing_decision"] == {
        "reason": "conversation_mode_switch",
        "main_agent_selected_mode": "discuss",
        "mode_source": "explicit_user_request",
        "selected_agent_ids": [],
        "workflow_id": None,
        "allow_workflow_adjustment": False,
        "workflow_adjustment_policy": "strict_preset",
        "conversation_id": "conv-1",
        "reference_conversation_id": None,
        "attachment_ids": [],
    }


async def test_auto_submission_does_not_reuse_previous_mode_when_user_requests_new_conversation() -> None:
    repository = ConversationModeRepository(TaskMode.HYBRID)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.HYBRID),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="换个话题，帮我看一个新问题",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-new-conversation",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DIRECT
    assert submitted.clarification_reason is None
    assert router.calls == 1


async def test_auto_submission_queues_local_direct_when_router_cannot_classify() -> None:
    repository = ConversationModeRepository(None)
    router = WaitingRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="为什么刚才任务停住了",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-auto-direct-fallback",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DIRECT
    assert submitted.clarification_reason is None
    assert router.calls == 1
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "main_agent_local_resolution"
    assert routing["main_agent_selected_mode"] == "direct"
    assert routing["router_clarification_reason"] == "classification_unavailable"


async def test_submit_rejects_runtime_unbounded_message_before_creating_run() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=WaitingRouter(),
        task_queue=RecordingQueue(),
    )

    with pytest.raises(ValueError, match="at most 16000 characters"):
        await service.submit(
            tenant_id=uuid4(),
            actor_id=uuid4(),
            message="长文档" * 6000,
            mode=TaskMode.AUTO,
            conversation_id="conv-long-doc",
            idempotency_key="idem-long-doc",
        )

    assert repository.created == []


async def test_submit_strips_padding_before_routing_and_persistence() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=WaitingRouter(),
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message=" \n 为什么刚才会失败 \t ",
        mode=TaskMode.AUTO,
        conversation_id="conv-trim",
        idempotency_key="idem-trim",
    )

    assert submitted.status is RunStatus.QUEUED
    assert repository.created[0]["request"] == "为什么刚才会失败"


@pytest.mark.parametrize(
    "message",
    (
        "是不是没办法看到这个 skill，然后 agent 是不是读不了内部文件？我想安装一些新的 skill，agent 能给我查找并下载安装吗？",
        "我上传了几个 skill 压缩包，用自动模式想让它安装，根本没有安装，第二次交互又找不到压缩包里的内容，这是为什么？",
        "为什么普通长文档里出现计划、提醒、执行这些词，就会被识别成计划任务？帮我分析一下原因。",
        "这个交互有问题：取消以后没有退回普通对话，后续消息还是像计划流程一样卡住，怎么解决？",
        "我想让你帮我分析一下这段经历。今天领导提醒我执行新的交接计划，但我实际想讨论沟通策略。",
    ),
)
def test_interactive_support_requests_with_planning_words_stay_direct(message: str) -> None:
    assert _local_main_agent_auto_mode(message, ()) is TaskMode.DIRECT


async def test_router_unavailable_keeps_skill_file_visibility_question_as_normal_chat() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=None,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message=(
            "是不是没办法看到这个 skill，然后 agent 是不是读不了内部文件？"
            "我想安装一些新的 skill，agent 能给我查找并下载安装吗？"
        ),
        mode=TaskMode.AUTO,
        conversation_id="conv-skill-file-question",
        idempotency_key="idem-skill-file-question",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DIRECT
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "main_agent_local_resolution"
    assert routing["main_agent_selected_mode"] == "direct"


async def test_router_ready_dispatch_is_overridden_for_interactive_support_question() -> None:
    repository = ConversationModeRepository(None)
    router = ReadyDispatchRouter()
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=router,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message=(
            "为什么普通交互里出现计划两个字就会创建计划？"
            "这个有问题，取消后也没有回到正常对话。"
        ),
        mode=TaskMode.AUTO,
        conversation_id="conv-router-misclass",
        idempotency_key="idem-router-misclass",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DIRECT
    assert router.calls == 1
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["router_selected_mode"] == "dispatch"
    assert routing["main_agent_selected_mode"] == "direct"
    assert routing["main_agent_adjusted"] is True


@pytest.mark.parametrize(
    ("message", "expected_mode", "expected_reason"),
    (
        ("请用讨论模式评审这个方案的优缺点", TaskMode.DISCUSS, "conversation_mode_switch"),
        ("组织多角色讨论，对比两个技术方案", TaskMode.DISCUSS, "main_agent_local_resolution"),
        ("生成男女主每人一张 Character Model Sheet 角色参考设定表图片", TaskMode.DISPATCH, "current_artifact_delivery_request"),
        ("完整流程：先讨论剧本问题，再执行角色参考图生成", TaskMode.HYBRID, "main_agent_local_resolution"),
    ),
)
async def test_auto_mode_keeps_explicit_non_direct_workflows(
    message: str,
    expected_mode: TaskMode,
    expected_reason: str,
) -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(expected_mode),)),
        router=None,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message=message,
        mode=TaskMode.AUTO,
        conversation_id="conv-non-direct-boundary",
        idempotency_key=f"idem-non-direct-{expected_mode.value}",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is expected_mode
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == expected_reason
    assert routing["main_agent_selected_mode"] == expected_mode.value


async def test_auto_submission_waits_when_router_requires_user_choice() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=UserChoiceRouter(),
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="ambiguous workflow",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-router-user-choice",
    )

    assert submitted.status is RunStatus.WAITING_USER_MODE
    assert submitted.mode is None
    assert submitted.decision_token == "safe-decision-token-abcdefghijklmnopqrstuvwxyz1234"
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "routing_requires_user_choice"
    assert "main_agent_selected_mode" not in routing


async def test_auto_submission_uses_hermes_before_local_direct_router_fallback() -> None:
    repository = ConversationModeRepository(None)
    advisor = RecordingHermesAdvisor(
        HermesRunAdvice(
            recommended_mode=TaskMode.DISPATCH,
            confidence=0.86,
            reasons=("matched previous execution pattern",),
            recommended_skills=("script-review",),
            requires_approval=False,
        )
    )
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DISPATCH),)),
        router=None,
        task_queue=RecordingQueue(),
        hermes_advisor=advisor,
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="short video script",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-hermes-before-direct-fallback",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DISPATCH
    assert len(advisor.calls) == 1
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "hermes_recommendation"


async def test_auto_submission_records_hermes_injected_memory_payload() -> None:
    repository = ConversationModeRepository(None)
    advisor = RecordingHermesAdvisor(
        HermesRunAdvice(
            recommended_mode=TaskMode.DISPATCH,
            confidence=0.86,
            reasons=("matched previous execution pattern",),
            recommended_skills=("script-review",),
            requires_approval=False,
            injected_memories=(
                HermesMemoryInjection(
                    id="hermes_confirmed_review",
                    summary="reviewer 超时时先压缩上下文再分块审查。",
                    memory_type="error_handling",
                    target="reviewer",
                    score=0.91,
                    reason="命中 reviewer 超时处理经验",
                ),
            ),
            skipped_memories=(
                HermesSkippedMemory(
                    id="hermes_old_direct",
                    summary="旧 direct 模式观察。",
                    reason="当前任务相关性不足",
                    score=0.42,
                ),
            ),
        )
    )
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DISPATCH),)),
        router=None,
        task_queue=RecordingQueue(),
        hermes_advisor=advisor,
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="short video script",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-hermes-memory-payload",
    )

    assert submitted.status is RunStatus.QUEUED
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    hermes = routing["hermes"]
    assert isinstance(hermes, dict)
    assert hermes["injected_memories"] == [
        {
            "id": "hermes_confirmed_review",
            "summary": "reviewer 超时时先压缩上下文再分块审查。",
            "memory_type": "error_handling",
            "target": "reviewer",
            "score": 0.91,
            "reason": "命中 reviewer 超时处理经验",
        }
    ]
    assert hermes["skipped_memories"][0]["reason"] == "当前任务相关性不足"


async def test_hermes_advice_timeout_does_not_block_auto_submission() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=None,
        task_queue=RecordingQueue(),
        hermes_advisor=SlowHermesAdvisor(None),
    )

    started = time.monotonic()
    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="hello",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        idempotency_key="idem-hermes-timeout",
    )
    elapsed = time.monotonic() - started

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DIRECT
    assert elapsed < 1.5


async def test_declined_evolution_proposal_can_continue_through_auto_mode() -> None:
    repository = ConversationModeRepository(None)
    service = RunService(
        repository,  # type: ignore[arg-type]
        runtime_registry=RuntimeRegistry((UnavailableRuntime(TaskMode.DIRECT),)),
        router=None,
        task_queue=RecordingQueue(),
    )

    submitted = await service.submit(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="请进化 darwin-skill，做多轮迭代",
        mode=TaskMode.AUTO,
        conversation_id="conv-1",
        skip_evolution_proposal=True,
        idempotency_key="idem-declined-evolution-continue",
    )

    assert submitted.status is RunStatus.QUEUED
    assert submitted.mode is TaskMode.DIRECT
    assert submitted.evolution_proposal is None
    routing = repository.created[0]["routing_decision"]
    assert isinstance(routing, dict)
    assert routing["reason"] == "main_agent_local_resolution"
    assert routing["main_agent_selected_mode"] == "direct"
    assert routing["skip_evolution_proposal"] is True


def test_long_background_story_with_schedule_words_does_not_create_schedule_proposal() -> None:
    message = (
        "我想让你帮我分析一下这段经历。"
        "2026年9月12日领导在群里通知大家执行新的排班提醒规则，"
        "今天领导提醒我执行新的交接计划，"
        "但我实际想讨论的是沟通策略、情绪压力和后续怎么处理。"
        "下面是很长的背景材料：" + "工作沟通细节。" * 80
    )

    proposal = _local_schedule_proposal(
        message=message,
        mode=TaskMode.DIRECT,
        workflow_id=None,
    )

    assert proposal is None


def test_background_analysis_with_remind_me_phrase_does_not_create_schedule_proposal() -> None:
    message = (
        "我想让你帮我分析一下这段经历。"
        "今天同事提醒我执行交接计划，但这只是背景，"
        "我需要的是沟通建议和问题复盘。"
        + "更多背景。" * 120
    )

    proposal = _local_schedule_proposal(
        message=message,
        mode=TaskMode.AUTO,
        workflow_id=None,
    )

    assert proposal is None


def test_explicit_reminder_request_still_creates_schedule_proposal() -> None:
    proposal = _local_schedule_proposal(
        message="明天8点提醒我填写日报",
        mode=TaskMode.DIRECT,
        workflow_id=None,
    )

    assert proposal is not None
    assert proposal.kind == "one_time"
    assert "08:00" in proposal.summary
