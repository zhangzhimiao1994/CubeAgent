from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

from agent_hub.cognitive.types import ReflectionRecord


def reflect_from_feedback(
    *,
    tenant_id: UUID,
    user_id: UUID,
    source_run_id: str,
    user_feedback: str,
    outcome: str,
    now: Callable[[], datetime],
) -> ReflectionRecord:
    feedback = " ".join(user_feedback.strip().split())
    lowered = feedback.casefold()
    timestamp = now()
    negative = outcome == "negative" or any(
        token in lowered for token in ("不是", "不对", "失败", "甩锅", "不要", "错误")
    )
    positive = outcome == "positive" or any(
        token in lowered for token in ("可以", "满意", "成功", "继续按这个方式", "不错")
    )
    if negative:
        trigger = "user_correction" if any(token in lowered for token in ("不是", "不对", "甩锅", "错误")) else "user_rejection"
        causal_analysis = "用户反馈表明当前处理方式没有承担足够的问题诊断和修复责任。"
        counterfactual = "如果重新处理，应先解决已发现的问题，实在无法解决时再报告阻塞原因。"
        positive_patterns: tuple[str, ...] = ()
        negative_patterns: tuple[str, ...] = ("把可处理的问题交回给用户", "没有先做 in-scope 修复")
        confidence = 0.68
    elif positive:
        trigger = "user_satisfaction" if any(token in lowered for token in ("可以", "满意", "不错")) else "user_confirmation"
        causal_analysis = "用户反馈表明当前执行方式符合其偏好或任务目标。"
        counterfactual = ""
        positive_patterns = ("保持当前执行策略", "后续同类任务可复用该处理方式")
        negative_patterns = ()
        confidence = 0.64
    else:
        trigger = "manual_feedback"
        causal_analysis = "用户提供了中性反馈，可作为后续判断的弱证据。"
        counterfactual = ""
        positive_patterns = ()
        negative_patterns = ()
        confidence = 0.5

    return ReflectionRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=user_id,
        source_run_id=source_run_id,
        trigger=trigger,
        outcome=outcome,
        causal_analysis=causal_analysis,
        counterfactual=counterfactual,
        positive_patterns=positive_patterns,
        negative_patterns=negative_patterns,
        proposed_experience_ids=(),
        confidence=confidence,
        created_at=timestamp,
    )
