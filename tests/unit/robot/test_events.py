from __future__ import annotations

from agent_hub.robot.events import extract_assistant_text, text_delta


def test_extract_assistant_text_prefers_artifact_output() -> None:
    events = (
        {"kind": "model.started", "payload": {}},
        {
            "kind": "artifact.created",
            "payload": {"output": "你好，今天心情怎么样？"},
            "artifact": {"content": {"text": "你好，今天心情怎么样？"}, "producer": "main_agent"},
        },
        {"kind": "runtime.completed", "payload": {"summary": "你好，今天心情怎么样？"}},
    )

    assert extract_assistant_text(events) == "你好，今天心情怎么样？"


def test_extract_assistant_text_skips_conversation_history() -> None:
    events = (
        {
            "kind": "artifact.created",
            "artifact": {
                "producer": "conversation_history",
                "content": {"text": "第 1 轮用户：昨天吃了面"},
            },
        },
        {
            "kind": "artifact.created",
            "payload": {"output": "面很好吃。"},
            "artifact": {"producer": "main_agent", "content": {"text": "面很好吃。"}},
        },
    )

    assert extract_assistant_text(events) == "面很好吃。"


def test_text_delta_emits_only_new_suffix() -> None:
    assert text_delta("", "你好") == "你好"
    assert text_delta("你好", "你好啊") == "啊"
    assert text_delta("你好", "你好") == ""
