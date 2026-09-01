from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self, cast
from uuid import uuid4

import pytest

from agent_hub.domain.runs import RunStatus, TaskMode
from agent_hub.hermes.advisor import (
    PersistentHermesRunAdvisor,
    _cognitive_candidate_payload_from_outcome,
    _outcome_learning_payload,
    _runtime_lesson_summary,
)
from agent_hub.runs.service import HermesRunOutcome


@dataclass(slots=True)
class FakeRow:
    payload: dict[str, object]


class FakeResult:
    def __init__(self, rows: list[FakeRow]) -> None:
        self._rows = rows

    def scalar_one_or_none(self) -> FakeRow | None:
        return self._rows[0] if self._rows else None

    def scalars(self) -> list[FakeRow]:
        return self._rows


class FakeSession:
    def __init__(self, result_sets: list[list[FakeRow]]) -> None:
        self._result_sets = result_sets

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    async def execute(self, statement: object) -> FakeResult:
        del statement
        if not self._result_sets:
            return FakeResult([])
        return FakeResult(self._result_sets.pop(0))


class FakeSessionFactory:
    def __init__(self, result_sets: list[list[FakeRow]]) -> None:
        self._result_sets = result_sets

    def __call__(self) -> FakeSession:
        return FakeSession(self._result_sets)


@pytest.mark.asyncio
async def test_runtime_advice_ignores_confirmed_scheduler_observations() -> None:
    scheduler_lesson = {
        "id": "hermes_scheduler_capacity",
        "category": "scheduler",
        "outcome": "failure",
        "lesson": "Run failed with mode=hybrid. Scheduler notices: trigger=model_capacity_pressure.",
        "summary": "调度观察：capacity pressure should not become ordinary conversation advice.",
        "tags": ["planning", "hybrid", "model_capacity_pressure"],
        "weight": 10,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": str(uuid4()),
        "conversation_id": "conv-scheduler-only",
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(scheduler_lesson)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="planning task needs a routing suggestion",
        mode=TaskMode.AUTO,
        agent_ids=(),
        workflow_id=None,
    )

    assert advice is None


@pytest.mark.asyncio
async def test_runtime_advice_can_use_confirmed_conversation_lessons() -> None:
    actor_id = uuid4()
    conversation_lesson = {
        "id": "hermes_conversation_review",
        "user_id": str(actor_id),
        "memory_scope": "user",
        "category": "conversation",
        "outcome": "success",
        "lesson": "Use group chat when debate review is required.",
        "summary": "Learned success pattern: debate review.",
        "tags": ["debate", "review"],
        "weight": 10,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": None,
        "conversation_id": "conv-review",
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(conversation_lesson)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="please run a debate review",
        mode=TaskMode.AUTO,
        agent_ids=(),
        workflow_id=None,
    )

    assert advice is not None
    assert advice.recommended_mode is TaskMode.DISCUSS


@pytest.mark.asyncio
async def test_runtime_advice_does_not_use_other_users_conversation_lessons() -> None:
    actor_id = uuid4()
    other_user_id = uuid4()
    conversation_lesson = {
        "id": "hermes_other_user_review",
        "user_id": str(other_user_id),
        "memory_scope": "user",
        "category": "conversation",
        "outcome": "success",
        "lesson": "Use group chat when debate review is required.",
        "summary": "Learned success pattern: debate review.",
        "tags": ["debate", "review"],
        "weight": 10,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": None,
        "conversation_id": "conv-review",
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(conversation_lesson)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="please run a debate review",
        mode=TaskMode.AUTO,
        agent_ids=(),
        workflow_id=None,
    )

    assert advice is None


@pytest.mark.asyncio
async def test_runtime_advice_can_use_root_scoped_conversation_lessons() -> None:
    actor_id = uuid4()
    conversation_lesson = {
        "id": "hermes_root_review",
        "user_id": str(uuid4()),
        "memory_scope": "root",
        "category": "conversation",
        "outcome": "success",
        "lesson": "Use group chat when debate review is required.",
        "summary": "Learned success pattern: debate review.",
        "tags": ["debate", "review"],
        "weight": 10,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": None,
        "conversation_id": "conv-review",
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(conversation_lesson)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="please run a debate review",
        mode=TaskMode.AUTO,
        agent_ids=(),
        workflow_id=None,
    )

    assert advice is not None
    assert advice.injected_memories[0].id == "hermes_root_review"


@pytest.mark.asyncio
async def test_runtime_advice_confirmed_conversation_lesson_not_starved_by_scheduler_rows() -> None:
    actor_id = uuid4()
    confirmed_lesson = {
        "id": "hermes_confirmed_review_starvation",
        "user_id": str(actor_id),
        "memory_scope": "user",
        "category": "conversation",
        "outcome": "success",
        "lesson": "Use group chat when debate review is required.",
        "summary": "Learned success pattern: debate review.",
        "tags": ["debate", "review"],
        "weight": 10,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": None,
        "conversation_id": "conv-review",
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    scheduler_rows = [
        FakeRow(
            {
                "id": f"hermes_scheduler_{index}",
                "category": "scheduler",
                "outcome": "failure",
                "lesson": "Run failed with mode=hybrid, workflow=no-workflow.",
                "tags": ["hybrid", "scheduler"],
                "weight": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "confirmed_at": None,
            }
        )
        for index in range(250)
    ]
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            scheduler_rows + [FakeRow(confirmed_lesson)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="please run a debate review",
        mode=TaskMode.AUTO,
        agent_ids=(),
        workflow_id=None,
    )

    assert advice is not None
    assert advice.injected_memories[0].id == "hermes_confirmed_review_starvation"


@pytest.mark.asyncio
async def test_runtime_advice_injects_cross_mode_project_rule_when_relevant() -> None:
    actor_id = uuid4()
    lesson = {
        "id": "hermes_ui_drawer_rule",
        "user_id": str(actor_id),
        "memory_scope": "user",
        "category": "conversation",
        "outcome": "success",
        "lesson": "调度卡片应默认显示摘要，详情放抽屉，点击遮罩关闭。",
        "user_summary": "调度卡片默认只显示摘要，详情放抽屉。",
        "tags": ["调度卡片", "抽屉", "ui"],
        "weight": 9,
        "source_mode": "discuss",
        "applies_to_modes": ["dispatch", "direct", "hybrid"],
        "memory_type": "ui_rule",
        "target": "frontend",
        "confidence": 0.88,
        "noise_risk": 0.1,
        "created_at": datetime.now(UTC).isoformat(),
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(lesson)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="修改调度卡片 UI，详情用抽屉展示",
        mode=TaskMode.DISPATCH,
        agent_ids=("frontend",),
        workflow_id=None,
    )

    assert advice is not None
    assert advice.injected_memories[0].id == "hermes_ui_drawer_rule"
    assert advice.injected_memories[0].target == "frontend"


@pytest.mark.asyncio
async def test_runtime_advice_injects_confirmed_cognitive_experience() -> None:
    actor_id = uuid4()
    experience_id = uuid4()
    cognitive_experience = {
        "id": str(experience_id),
        "user_id": str(actor_id),
        "memory_scope": "user",
        "kind": "error_handling",
        "status": "confirmed",
        "summary": "分享链接多行文本需要在角色规划前统一校验。",
        "lesson": "TaskContext 允许换行，但 RolePlanningRequest 曾经拒绝换行。",
        "strategy": "处理分享链接或多行输入时，保留可读换行并拒绝隐藏控制字符。",
        "confidence": 0.86,
        "evidence": [
            {
                "source_type": "run",
                "source_id": "runtime-control-char",
                "note": "多行 URL 输入触发 task control character error。",
            }
        ],
        "contradictions": [],
        "source_run_ids": ["runtime-control-char"],
        "source_memory_ids": [],
        "tags": ["分享链接", "多行", "control"],
        "applies_to_modes": ["hybrid", "dispatch"],
        "applies_to_agents": ["main_agent"],
        "use_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "active_for_runtime": True,
        "last_used_at": None,
        "last_verified_at": None,
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "storage_kind": "hermes",
        "resource_id": f"cognitive_experience:{experience_id}",
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [],
            [FakeRow(cognitive_experience)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="帮我排查分享链接后多行 control 字符失败",
        mode=TaskMode.HYBRID,
        agent_ids=("main_agent",),
        workflow_id=None,
    )

    assert advice is not None
    assert advice.injected_memories[0].id.startswith("cognitive_experience:")
    assert advice.injected_memories[0].memory_type == "error_handling"
    assert advice.injected_memories[0].target == "main_agent"
    assert "分享链接" in advice.injected_memories[0].summary


@pytest.mark.asyncio
async def test_runtime_advice_does_not_inject_cognitive_experience_from_short_common_tag() -> None:
    actor_id = uuid4()
    experience_id = uuid4()
    cognitive_experience = {
        "id": str(experience_id),
        "user_id": str(actor_id),
        "memory_scope": "user",
        "kind": "error_handling",
        "status": "confirmed",
        "summary": "Go release jobs require the release agent.",
        "lesson": "Go module release failures should be handled by the release workflow.",
        "strategy": "Use the release workflow only when the request is explicitly about Go module releases.",
        "confidence": 0.9,
        "evidence": [
            {
                "source_type": "run",
                "source_id": "go-release-run",
                "note": "Go release workflow failed until release_agent handled it.",
            }
        ],
        "contradictions": [],
        "source_run_ids": ["go-release-run"],
        "source_memory_ids": [],
        "tags": ["go"],
        "applies_to_modes": ["dispatch"],
        "applies_to_agents": ["release_agent"],
        "use_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "active_for_runtime": True,
        "last_used_at": None,
        "last_verified_at": None,
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "storage_kind": "hermes",
        "resource_id": f"cognitive_experience:{experience_id}",
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [],
            [FakeRow(cognitive_experience)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="ongoing investigation of the dashboard layout",
        mode=TaskMode.DIRECT,
        agent_ids=("frontend",),
        workflow_id=None,
    )

    assert advice is None


@pytest.mark.asyncio
async def test_runtime_advice_ignores_unconfirmed_cognitive_experience() -> None:
    actor_id = uuid4()
    cognitive_experience = {
        "id": str(uuid4()),
        "user_id": str(actor_id),
        "memory_scope": "user",
        "kind": "preference",
        "status": "candidate",
        "summary": "候选经验不能直接影响运行时。",
        "lesson": "候选经验必须由用户确认。",
        "strategy": "等待确认。",
        "confidence": 0.9,
        "tags": ["候选经验"],
        "applies_to_modes": ["hybrid"],
        "applies_to_agents": ["main_agent"],
        "active_for_runtime": False,
        "resource_id": "cognitive_experience:candidate",
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [],
            [FakeRow(cognitive_experience)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="继续处理候选经验相关任务",
        mode=TaskMode.HYBRID,
        agent_ids=("main_agent",),
        workflow_id=None,
    )

    assert advice is None


@pytest.mark.asyncio
async def test_runtime_advice_skips_same_mode_low_quality_noise() -> None:
    actor_id = uuid4()
    lesson = {
        "id": "hermes_noise",
        "user_id": str(actor_id),
        "memory_scope": "user",
        "category": "conversation",
        "outcome": "neutral",
        "lesson": "这个任务成功了。",
        "user_summary": "这个任务成功了。",
        "tags": ["direct"],
        "weight": 10,
        "source_mode": "direct",
        "memory_type": "temporary_state",
        "target": "main_agent",
        "confidence": 0.3,
        "noise_risk": 0.9,
        "created_at": datetime.now(UTC).isoformat(),
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(lesson)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="direct 模式继续处理这个任务",
        mode=TaskMode.DIRECT,
        agent_ids=(),
        workflow_id=None,
    )

    assert advice is None


@pytest.mark.asyncio
async def test_runtime_advice_ignores_confirmed_runtime_observations() -> None:
    runtime_observation = {
        "id": "hermes_run_observation",
        "category": "scheduler",
        "outcome": "success",
        "lesson": "Run completed with mode=hybrid, workflow=no-workflow.",
        "user_summary": "本次运行观察记录了一个成功经验：no-workflow 工作流以 hybrid 模式成功完成。",
        "tags": ["completed", "hybrid", "no-workflow"],
        "weight": 10,
        "source_mode": "hybrid",
        "memory_type": "runtime_observation",
        "target": "scheduler",
        "confidence": 0.9,
        "noise_risk": 0.1,
        "created_at": datetime.now(UTC).isoformat(),
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(runtime_observation)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="继续用 hybrid 模式处理 no-workflow 任务",
        mode=TaskMode.HYBRID,
        agent_ids=(),
        workflow_id="no-workflow",
    )

    assert advice is None


@pytest.mark.asyncio
async def test_runtime_advice_ignores_legacy_conversation_shaped_runtime_observations() -> None:
    legacy_runtime_observation = {
        "id": "legacy_runtime_observation",
        "category": "conversation",
        "outcome": "success",
        "lesson": "Run completed with mode=hybrid, workflow=no-workflow.",
        "tags": ["completed", "hybrid", "no-workflow"],
        "weight": 10,
        "source_mode": "hybrid",
        "memory_type": "conversation_advice",
        "target": "main_agent",
        "confidence": 0.9,
        "noise_risk": 0.1,
        "created_at": datetime.now(UTC).isoformat(),
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(legacy_runtime_observation)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="继续用 hybrid 模式处理 no-workflow 任务",
        mode=TaskMode.HYBRID,
        agent_ids=(),
        workflow_id="no-workflow",
    )

    assert advice is None


@pytest.mark.asyncio
async def test_runtime_advice_ignores_legacy_scheduler_notice_observations() -> None:
    legacy_scheduler_observation = {
        "id": "hermes_run_notice_legacy",
        "category": "conversation",
        "outcome": "failure",
        "lesson": (
            "Run failed with mode=hybrid, workflow=quality-review. "
            "Scheduler notices: trigger=model_capacity_pressure."
        ),
        "user_summary": "本次对话学习记录了一个失败教训：quality-review 工作流以 hybrid 模式运行失败。",
        "tags": ["failed", "hybrid", "quality-review", "model_capacity_pressure"],
        "weight": 10,
        "source_mode": "hybrid",
        "memory_type": "conversation_advice",
        "target": "main_agent",
        "confidence": 0.8,
        "noise_risk": 0.2,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": str(uuid4()),
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(legacy_scheduler_observation)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=uuid4(),
        message="quality-review 又出现 model capacity pressure，继续处理",
        mode=TaskMode.HYBRID,
        agent_ids=(),
        workflow_id="quality-review",
    )

    assert advice is None


@pytest.mark.asyncio
async def test_runtime_advice_records_conflicting_memory_as_skipped() -> None:
    actor_id = uuid4()
    lesson = {
        "id": "hermes_hybrid_preference",
        "user_id": str(actor_id),
        "memory_scope": "user",
        "category": "conversation",
        "outcome": "success",
        "lesson": "大任务优先使用混合模式。",
        "user_summary": "大任务优先使用混合模式。",
        "tags": ["大任务", "hybrid"],
        "weight": 8,
        "source_mode": "hybrid",
        "memory_type": "scheduling_rule",
        "target": "scheduler",
        "confidence": 0.8,
        "noise_risk": 0.1,
        "created_at": datetime.now(UTC).isoformat(),
        "confirmed_at": datetime.now(UTC).isoformat(),
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(lesson)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="先跑直连模式，不要混合，处理这个大任务",
        mode=TaskMode.DIRECT,
        agent_ids=(),
        workflow_id=None,
    )

    assert advice is not None
    assert advice.injected_memories == ()
    assert advice.skipped_memories[0].id == "hermes_hybrid_preference"
    assert advice.skipped_memories[0].reason == "当前用户指令覆盖这条记忆"


def test_runtime_lesson_summary_localizes_scheduler_outcomes_for_users() -> None:
    assert (
        _runtime_lesson_summary("Run failed with mode=hybrid, workflow=quality-review.")
        == "quality-review 工作流以 hybrid 模式运行失败。"
    )
    assert (
        _runtime_lesson_summary(
            "Run completed with mode=dispatch, workflow=short-video-dispatch. "
            "Scheduler notices: trigger=model_capacity_pressure."
        )
        == "short-video-dispatch 工作流以 dispatch 模式成功完成。 已记录调度告警。"
    )


def test_runtime_outcome_without_scheduler_notice_creates_scheduler_observation() -> None:
    run_id = uuid4()
    actor_id = uuid4()
    payload = _outcome_learning_payload(
        HermesRunOutcome(
            tenant_id=uuid4(),
            actor_id=actor_id,
            run_id=run_id,
            status=RunStatus.COMPLETED,
            mode=TaskMode.HYBRID,
            workflow_id=None,
            conversation_id="conv-dialog",
            agent_ids=("moderator", "domain_expert"),
        ),
        lesson_id="hermes_run_unit",
    )

    assert payload["category"] == "scheduler"
    assert payload["user_id"] == str(actor_id)
    assert payload["memory_scope"] == "user"
    assert payload["memory_type"] == "runtime_observation"
    assert payload["target"] == "scheduler"
    assert payload["conversation_id"] == "conv-dialog"
    assert payload["run_id"] == str(run_id)
    assert payload["confirmed_at"] is None
    assert payload["user_summary"] == (
        "本次运行观察记录了一个成功经验：no-workflow 工作流以 hybrid 模式成功完成。"
    )


def test_runtime_outcome_with_scheduler_notice_creates_scheduler_observation() -> None:
    payload = _outcome_learning_payload(
        HermesRunOutcome(
            tenant_id=uuid4(),
            actor_id=uuid4(),
            run_id=uuid4(),
            status=RunStatus.FAILED,
            mode=TaskMode.DISPATCH,
            workflow_id="short-video-dispatch",
            conversation_id="conv-scheduler",
            agent_ids=("planner",),
            scheduler_notices=(
                {
                    "trigger": "model_capacity_pressure",
                    "action": "reschedule_or_reassign_model",
                    "severity": "warning",
                    "source_kind": "step.failed",
                    "actor": "planner",
                },
            ),
        ),
        lesson_id="hermes_run_notice",
    )

    assert payload["category"] == "scheduler"
    assert str(payload["user_summary"]).startswith(
        "本次调度观察提醒：short-video-dispatch 工作流以 dispatch 模式运行失败"
    )


def test_failed_runtime_outcome_with_notice_creates_cognitive_candidate() -> None:
    run_id = uuid4()
    actor_id = uuid4()
    payload = _cognitive_candidate_payload_from_outcome(
        HermesRunOutcome(
            tenant_id=uuid4(),
            actor_id=actor_id,
            run_id=run_id,
            status=RunStatus.FAILED,
            mode=TaskMode.HYBRID,
            workflow_id="quality-review",
            conversation_id="conv-timeout",
            agent_ids=("quality_reviewer",),
            scheduler_notices=(
                {
                    "trigger": "empty_model_response",
                    "action": "retry_fallback_or_reassign_model",
                    "severity": "warning",
                    "source_kind": "runtime.failed",
                    "actor": "quality_reviewer",
                },
            ),
        )
    )

    assert payload is not None
    assert payload["kind"] == "error_handling"
    assert payload["status"] == "candidate"
    assert payload["active_for_runtime"] is False
    assert payload["source_run_ids"] == [str(run_id)]
    assert payload["user_id"] == str(actor_id)
    assert payload["memory_scope"] == "user"
    assert payload["applies_to_modes"] == ["hybrid"]
    assert "quality_reviewer" in cast(list[str], payload["applies_to_agents"])
    assert str(payload["resource_id"]).startswith("cognitive_experience:")


@pytest.mark.asyncio
async def test_runtime_advice_does_not_inject_user_scoped_cognitive_experience_from_other_user() -> None:
    actor_id = uuid4()
    other_user_id = uuid4()
    experience_id = uuid4()
    cognitive_experience = {
        "id": str(experience_id),
        "user_id": str(other_user_id),
        "memory_scope": "user",
        "kind": "communication_style",
        "status": "confirmed",
        "summary": "另一个用户偏好冗长解释。",
        "lesson": "当用户提到 cognitive-smoke 时输出长篇背景。",
        "strategy": "输出长篇背景。",
        "confidence": 0.9,
        "evidence": [{"source_type": "feedback", "source_id": "other-user", "note": "explicit"}],
        "contradictions": [],
        "source_run_ids": [],
        "source_memory_ids": [],
        "tags": ["cognitive-smoke"],
        "applies_to_modes": ["direct"],
        "applies_to_agents": ["main_agent"],
        "use_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "active_for_runtime": True,
        "last_used_at": None,
        "last_verified_at": None,
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "storage_kind": "hermes",
        "resource_id": f"cognitive_experience:{experience_id}",
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [],
            [FakeRow(cognitive_experience)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="请做 cognitive-smoke 生产烟测",
        mode=TaskMode.DIRECT,
        agent_ids=("main_agent",),
        workflow_id=None,
    )

    assert advice is None


@pytest.mark.asyncio
async def test_runtime_advice_can_inject_root_scoped_cognitive_experience_for_any_user() -> None:
    actor_id = uuid4()
    owner_id = uuid4()
    experience_id = uuid4()
    cognitive_experience = {
        "id": str(experience_id),
        "user_id": str(owner_id),
        "memory_scope": "root",
        "kind": "communication_style",
        "status": "confirmed",
        "summary": "根经验：生产烟测先给结论。",
        "lesson": "当用户提到 cognitive-smoke 时，回答应先给结论。",
        "strategy": "先输出结论，再给证据。",
        "confidence": 0.9,
        "evidence": [{"source_type": "feedback", "source_id": "root-policy", "note": "admin confirmed"}],
        "contradictions": [],
        "source_run_ids": [],
        "source_memory_ids": [],
        "tags": ["cognitive-smoke"],
        "applies_to_modes": ["direct"],
        "applies_to_agents": ["main_agent"],
        "use_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "active_for_runtime": True,
        "last_used_at": None,
        "last_verified_at": None,
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "storage_kind": "hermes",
        "resource_id": f"cognitive_experience:{experience_id}",
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [],
            [FakeRow(cognitive_experience)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=uuid4(),
        actor_id=actor_id,
        message="请做 cognitive-smoke 生产烟测",
        mode=TaskMode.DIRECT,
        agent_ids=("main_agent",),
        workflow_id=None,
    )

    assert advice is not None
    assert advice.injected_memories[0].summary == "根经验：生产烟测先给结论。"


@pytest.mark.asyncio
async def test_runtime_advice_injects_relevant_cognitive_state_without_hermes_lesson() -> None:
    actor_id = uuid4()
    tenant_id = uuid4()
    now = datetime.now(UTC).isoformat()
    world_state: dict[str, object] = {
        "id": "world:cubeagent.project",
        "tenant_id": str(tenant_id),
        "user_id": str(actor_id),
        "memory_scope": "user",
        "scope": "cubeagent.project",
        "facts": ["CubeAgent 是纯对话 Agent，不直接修改代码 harness。"],
        "open_items": ["持续学习系统需要接入运行时注入。"],
        "future_events": ["harness 改造走独立项目。"],
        "last_verified_at": now,
        "evidence": [{"source_type": "handoff", "source_id": "HANDOFF", "note": "project boundary"}],
        "created_at": now,
        "updated_at": now,
    }
    skill_candidate: dict[str, object] = {
        "id": str(uuid4()),
        "tenant_id": str(tenant_id),
        "user_id": str(actor_id),
        "memory_scope": "user",
        "name": "runtime-context-injection",
        "purpose": "持续学习系统命中相关经验后，将少量可信上下文注入运行时。",
        "steps": ["检索相关 cognitive context。", "只注入可信且相关的少量条目。"],
        "required_inputs": ["hybrid"],
        "output_contract": "返回运行时可使用的上下文摘要。",
        "confidence": 0.84,
        "evidence": [{"source_type": "experience", "source_id": "exp-context", "note": "worked"}],
        "contradictions": [],
        "use_count": 2,
        "success_count": 2,
        "failure_count": 0,
        "last_used_at": None,
        "last_verified_at": now,
        "version": 1,
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [],
            [],
            [],
            [],
            [FakeRow(world_state)],
            [FakeRow(skill_candidate)],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=tenant_id,
        actor_id=actor_id,
        message="继续持续学习系统，注意不要改 harness",
        mode=TaskMode.HYBRID,
        agent_ids=("main_agent",),
        workflow_id=None,
    )

    assert advice is not None
    assert {item.memory_type for item in advice.injected_memories} >= {"world_state", "skill"}
    assert any("不要改 harness" in item.summary or "harness" in item.summary for item in advice.injected_memories)


@pytest.mark.asyncio
async def test_runtime_advice_bounds_combined_hermes_and_cognitive_context() -> None:
    actor_id = uuid4()
    tenant_id = uuid4()
    now = datetime.now(UTC).isoformat()
    hermes_lesson = {
        "id": "hermes_context_limit",
        "user_id": str(actor_id),
        "memory_scope": "user",
        "category": "conversation",
        "outcome": "success",
        "lesson": "持续学习系统运行时只注入少量高相关上下文。",
        "user_summary": "运行时只注入少量高相关上下文。",
        "tags": ["持续学习", "上下文"],
        "weight": 10,
        "source_mode": "hybrid",
        "applies_to_modes": ["hybrid"],
        "memory_type": "workflow_strategy",
        "target": "main_agent",
        "confidence": 0.9,
        "noise_risk": 0.05,
        "created_at": now,
        "confirmed_at": now,
    }
    cognitive_rows: list[dict[str, object]] = [
        {
            "id": f"world:cognitive:{index}",
            "tenant_id": str(tenant_id),
            "user_id": str(actor_id),
            "memory_scope": "user",
            "scope": "cubeagent.cognitive",
            "facts": [f"持续学习上下文注入规则 {index}。"],
            "open_items": [],
            "future_events": [],
            "last_verified_at": now,
            "evidence": [{"source_type": "test", "source_id": f"world-{index}", "note": "bounded"}],
            "created_at": now,
            "updated_at": now,
        }
        for index in range(5)
    ]
    session_factory = FakeSessionFactory(
        [
            [],
            [FakeRow({"hermes_policy": "suggest"})],
            [FakeRow(hermes_lesson)],
            [],
            [],
            [],
            [FakeRow(row) for row in cognitive_rows],
            [],
        ]
    )
    advisor = PersistentHermesRunAdvisor(session_factory)  # type: ignore[arg-type]

    advice = await advisor.advise(
        tenant_id=tenant_id,
        actor_id=actor_id,
        message="持续学习系统上下文注入继续处理",
        mode=TaskMode.HYBRID,
        agent_ids=("main_agent",),
        workflow_id=None,
    )

    assert advice is not None
    assert len(advice.injected_memories) == 3
    assert advice.injected_memories[0].id == "hermes_context_limit"


def test_successful_runtime_outcome_without_notice_does_not_create_cognitive_candidate() -> None:
    payload = _cognitive_candidate_payload_from_outcome(
        HermesRunOutcome(
            tenant_id=uuid4(),
            actor_id=uuid4(),
            run_id=uuid4(),
            status=RunStatus.COMPLETED,
            mode=TaskMode.DIRECT,
            workflow_id=None,
            conversation_id="conv-success",
            agent_ids=("main_agent",),
        )
    )

    assert payload is None
