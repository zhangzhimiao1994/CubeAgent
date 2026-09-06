from __future__ import annotations

from agent_hub.config.schema import PlatformConfig
from agent_hub.runs.temporary_agents import (
    _CAPABILITIES,
    _choose_recommended_model,
    _infer_capability,
)


def test_temporary_agent_policy_prefers_code_capable_model_for_engineering_gap() -> None:
    config = PlatformConfig.model_validate(
        {
            "models": {
                "general": {
                    "deployments": [
                        {
                            "provider": "deepseek",
                            "model": "deepseek-v4-flash",
                            "credential_ref": "secret://deepseek",
                            "quota_scope_id": "general",
                            "capabilities": ["text"],
                        }
                    ]
                },
                "claude_code": {
                    "deployments": [
                        {
                            "provider": "anthropic-compatible",
                            "model": "claude-sonnet-5",
                            "credential_ref": "secret://claude",
                            "quota_scope_id": "claude-code",
                            "capabilities": ["text", "tool_calling", "structured_output"],
                        }
                    ]
                },
            },
            "agents": [],
        }
    )

    engineering = next(
        spec for spec in _CAPABILITIES if spec.capability == "software_engineering"
    )

    assert _choose_recommended_model(config, engineering) == "claude_code"


def test_temporary_agent_policy_returns_none_when_no_text_model_exists() -> None:
    config = PlatformConfig.model_validate(
        {
            "models": {
                "vision_only": {
                    "deployments": [
                        {
                            "provider": "minimax",
                            "model": "abab-vision",
                            "credential_ref": "secret://vision",
                            "quota_scope_id": "vision",
                            "capabilities": ["vision"],
                        }
                    ]
                }
            },
            "agents": [],
        }
    )
    copywriting = next(spec for spec in _CAPABILITIES if spec.capability == "copywriting")

    assert _choose_recommended_model(config, copywriting) is None


def test_temporary_agent_policy_does_not_block_builtin_delivery_tools() -> None:
    builtin_delivery_requests = (
        "生成一份 DOCX 文档，标题为《魔方 Agent 验收记录》，正文包含三条项目检查结论，并把 DOCX 作为最终可下载产物。",
        "制作一份 PPTX 演示稿，标题为季度复盘，最终给我可下载文件。",
        "第二轮同一个会话：直接调用系统里的多媒体子 Agent，生成一张极简蓝色方块测试图片，最终给我可下载图片。",
    )

    for message in builtin_delivery_requests:
        assert _infer_capability(message) is None


def test_temporary_agent_policy_still_detects_real_copywriting_gap() -> None:
    proposal = _infer_capability("帮我写 3 个广告标题和品牌文案。")

    assert proposal is not None
    assert proposal.capability == "copywriting"
