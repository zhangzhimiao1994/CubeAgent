from __future__ import annotations

from agent_hub.runtime.hermes_context import hermes_memory_context_text


def test_hermes_memory_context_ignores_scheduler_observations() -> None:
    text = hermes_memory_context_text(
        {
            "hermes": {
                "injected_memories": [
                    {
                        "summary": "no-workflow 工作流以 hybrid 模式成功完成。",
                        "memory_type": "runtime_observation",
                        "target": "scheduler",
                        "reason": "legacy routing decision already had this item",
                    }
                ]
            }
        }
    )

    assert text == ""


def test_hermes_memory_context_ignores_legacy_runtime_conversation_items() -> None:
    text = hermes_memory_context_text(
        {
            "hermes": {
                "injected_memories": [
                    {
                        "summary": "Run completed with mode=hybrid, workflow=no-workflow.",
                        "memory_type": "conversation_advice",
                        "target": "main_agent",
                        "reason": "legacy routing decision already had this item",
                    }
                ]
            }
        }
    )

    assert text == ""


def test_hermes_memory_context_keeps_confirmed_conversation_guidance() -> None:
    text = hermes_memory_context_text(
        {
            "hermes": {
                "injected_memories": [
                    {
                        "summary": "用户希望调度卡片默认显示摘要，点击卡片再展开详情。",
                        "memory_type": "ui_rule",
                        "target": "main_agent",
                        "reason": "confirmed Hermes+ memory matched this task",
                    }
                ]
            }
        }
    )

    assert "用户希望调度卡片默认显示摘要" in text
    assert "runtime_observation" not in text
