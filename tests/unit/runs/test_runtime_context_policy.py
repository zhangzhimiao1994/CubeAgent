from __future__ import annotations

from uuid import uuid4

from agent_hub.domain.runs import TaskMode
from agent_hub.runs.repository import ConversationContextItem
from agent_hub.runs.service import (
    _conversation_history_artifact,
    _conversation_history_token_budget,
    _runtime_timeout_seconds,
    _runtime_token_budget,
)


def test_runtime_timeout_policy_uses_configured_production_window_for_dispatch() -> None:
    assert _runtime_timeout_seconds(TaskMode.DISPATCH, configured_seconds=300.0) == 300.0


def test_runtime_timeout_policy_clamps_to_runtime_contract_limit() -> None:
    assert _runtime_timeout_seconds(TaskMode.HYBRID, configured_seconds=7200.0) == 3600.0


def test_runtime_token_budget_policy_uses_configured_complex_budget() -> None:
    assert _runtime_token_budget(TaskMode.HYBRID, configured_tokens=1_000_000) == 1_000_000


def test_runtime_token_budget_policy_clamps_to_runtime_contract_limit() -> None:
    assert _runtime_token_budget(TaskMode.DISPATCH, configured_tokens=99_000_000) == 10_000_000


def test_conversation_history_budget_uses_main_agent_context_window() -> None:
    assert (
        _conversation_history_token_budget(
            runtime_token_budget=1_000_000,
            main_agent_context_window_tokens=4096,
        )
        == 1024
    )


def test_conversation_history_stays_full_when_inside_budget() -> None:
    run_id = uuid4()
    artifact = _conversation_history_artifact(
        run_id=run_id,
        conversation_id="conv-short",
        current_request="continue",
        context_items=(
            ConversationContextItem(
                run_id=uuid4(),
                request="first request",
                artifacts=(
                    {
                        "producer": "main_agent",
                        "content": {"text": "first answer"},
                    },
                ),
            ),
        ),
        history_token_budget=4096,
    )

    assert artifact is not None
    assert artifact.producer == "conversation_history"
    assert artifact.content["context_policy"] == "full_history"
    text = artifact.content["text"]
    assert isinstance(text, str)
    assert "first request" in text
    assert "first answer" in text

    repeated = _conversation_history_artifact(
        run_id=run_id,
        conversation_id="conv-short",
        current_request="continue",
        context_items=(
            ConversationContextItem(
                run_id=uuid4(),
                request="first request",
                artifacts=(
                    {
                        "producer": "main_agent",
                        "content": {"text": "first answer"},
                    },
                ),
            ),
        ),
        history_token_budget=4096,
    )
    assert repeated is not None
    assert repeated.id == artifact.id


def test_conversation_history_includes_media_pipeline_plan() -> None:
    artifact = _conversation_history_artifact(
        run_id=uuid4(),
        conversation_id="conv-video-plan",
        current_request="继续生成分镜图",
        context_items=(
            ConversationContextItem(
                run_id=uuid4(),
                request="先生成一个短剧剧本，后续可能要生成角色设定和剪辑成片。",
                artifacts=(),
                routing_decision={
                    "media_pipeline_plan": {
                        "plan_id": "media-plan-001",
                        "status": "planned",
                        "source": "script_request",
                        "summary": "短剧生产计划：先定角色，再做分镜，最后剪辑成片。",
                        "stages": [
                            {"id": "script", "status": "completed"},
                            {"id": "character_model_sheet", "status": "planned"},
                            {"id": "storyboard", "status": "planned"},
                            {"id": "compose_video", "status": "planned"},
                        ],
                        "approved_artifacts": [
                            {
                                "stage_id": "character_model_sheet",
                                "artifact_id": "asset-character-001",
                            }
                        ],
                        "storage_key": "secret/path/must/not/leak",
                    }
                },
            ),
        ),
        history_token_budget=4096,
    )

    assert artifact is not None
    text = artifact.content["text"]
    assert isinstance(text, str)
    assert "MEDIA_PIPELINE_PLAN" in text
    assert "media-plan-001" in text
    assert "character_model_sheet:planned" in text
    assert "asset-character-001" in text
    assert "secret/path/must/not/leak" not in text


def test_conversation_history_includes_media_pipeline_rejected_artifact_feedback() -> None:
    artifact = _conversation_history_artifact(
        run_id=uuid4(),
        conversation_id="conv-video-plan",
        current_request="重新生成角色参考设定表",
        context_items=(
            ConversationContextItem(
                run_id=uuid4(),
                request="生成角色参考设定表，审核后再生成视频。",
                artifacts=(),
                routing_decision={
                    "media_pipeline_plan": {
                        "plan_id": "media-plan-001",
                        "status": "planned",
                        "stages": [
                            {"id": "character_model_sheet", "status": "planned"},
                            {"id": "storyboard", "status": "planned"},
                        ],
                        "rejected_artifacts": [
                            {
                                "stage_id": "character_model_sheet",
                                "artifact_id": "asset-character-bad",
                                "feedback": "角色脸型和服装不一致，退回重新生成。",
                            }
                        ],
                    },
                    "storage_key": "secret/path/must/not/leak",
                },
            ),
        ),
        history_token_budget=4096,
    )

    assert artifact is not None
    text = artifact.content["text"]
    assert isinstance(text, str)
    assert "rejected_artifacts=character_model_sheet:asset-character-bad" in text
    assert "角色脸型和服装不一致" in text
    assert "secret/path/must/not/leak" not in text


def test_conversation_history_is_auto_compacted_when_over_model_budget() -> None:
    old_noise = "old implementation detail " * 2000
    latest_decision = "latest important conclusion: use framework-level context compression"

    artifact = _conversation_history_artifact(
        run_id=uuid4(),
        conversation_id="conv-long",
        current_request="continue the work",
        context_items=(
            ConversationContextItem(
                run_id=uuid4(),
                request=old_noise,
                artifacts=(
                    {
                        "producer": "main_agent",
                        "content": {"text": old_noise},
                    },
                ),
            ),
            ConversationContextItem(
                run_id=uuid4(),
                request="what was the final decision?",
                artifacts=(
                    {
                        "producer": "main_agent",
                        "content": {"text": latest_decision},
                    },
                ),
            ),
        ),
        history_token_budget=256,
    )

    assert artifact is not None
    assert artifact.producer == "conversation_history_compacted"
    assert artifact.content["context_policy"] == "auto_compacted"
    original_tokens = artifact.content["original_estimated_tokens"]
    history_budget = artifact.content["history_token_budget"]
    text = artifact.content["text"]
    assert type(original_tokens) is int
    assert type(history_budget) is int
    assert isinstance(text, str)
    assert original_tokens > history_budget
    assert latest_decision in text


def test_conversation_history_compaction_preserves_origin_goal_anchor() -> None:
    first_goal = "初始目标：完成 Agent Hub，并且所有高风险操作都必须审批。"
    items = [
        ConversationContextItem(
            run_id=uuid4(),
            request=first_goal,
            artifacts=(
                {
                    "producer": "main_agent",
                    "content": {"text": "长期约束：服务器增量部署，GitHub 全量推送。"},
                },
            ),
        )
    ]
    items.extend(
        ConversationContextItem(
            run_id=uuid4(),
            request=f"中间讨论 {index} " * 30,
            artifacts=(
                {
                    "producer": "main_agent",
                    "content": {"text": f"中间结果 {index} " * 30},
                },
            ),
        )
        for index in range(12)
    )
    latest_decision = "最新结论：上下文压缩属于对话框架，不属于进化模块。"
    items.append(
        ConversationContextItem(
            run_id=uuid4(),
            request="确认长期记忆归属",
            artifacts=(
                {
                    "producer": "main_agent",
                    "content": {"text": latest_decision},
                },
            ),
        )
    )

    artifact = _conversation_history_artifact(
        run_id=uuid4(),
        conversation_id="conv-framework-memory",
        current_request="继续当前任务",
        context_items=tuple(items),
        history_token_budget=128,
    )

    assert artifact is not None
    assert artifact.content["context_policy"] == "auto_compacted"
    text = artifact.content["text"]
    assert isinstance(text, str)
    assert first_goal in text
    assert "服务器增量部署" in text
    assert latest_decision in text

def test_conversation_history_compaction_preserves_latest_request_without_artifacts() -> None:
    first_goal = "初始目标：完成 Agent Hub，并且所有高风险操作都必须审批。"
    latest_decision = "最新结论：上下文压缩属于对话框架，不属于进化模块。"
    items = [
        ConversationContextItem(run_id=uuid4(), request=first_goal, artifacts=()),
    ]
    items.extend(
        ConversationContextItem(
            run_id=uuid4(),
            request=f"中间讨论 {index} " * 300,
            artifacts=(),
        )
        for index in range(4)
    )
    items.append(
        ConversationContextItem(run_id=uuid4(), request=latest_decision, artifacts=())
    )

    artifact = _conversation_history_artifact(
        run_id=uuid4(),
        conversation_id="conv-framework-memory-requests-only",
        current_request="继续当前任务",
        context_items=tuple(items),
        history_token_budget=128,
    )

    assert artifact is not None
    assert artifact.content["context_policy"] == "auto_compacted"
    text = artifact.content["text"]
    assert isinstance(text, str)
    assert first_goal in text
    assert latest_decision in text
