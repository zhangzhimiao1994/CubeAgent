from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

from agent_hub.cognitive.types import OutcomeAssessmentRecord, OutcomeVerdict, ReflectionRecord


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


class ReflectionEngine:
    def __init__(self, *, now: Callable[[], datetime]) -> None:
        self._now = now

    def reflect_from_outcome(self, assessment: OutcomeAssessmentRecord) -> ReflectionRecord:
        if assessment.verdict is OutcomeVerdict.SUCCESS:
            trigger = "outcome_success"
            causal_analysis = "运行结果有可见输出且没有未恢复的失败事件，说明当前策略在该场景下有效。"
            counterfactual = ""
            positive_patterns: tuple[str, ...] = ("保留本次有效执行路径", "未来相似场景优先复用已验证策略")
            negative_patterns: tuple[str, ...] = ()
            confidence = 0.62
        elif assessment.verdict is OutcomeVerdict.PARTIAL:
            trigger = "outcome_partial"
            causal_analysis = "运行产生了可用输出，但过程存在失败、降级或证据缺口。"
            counterfactual = "如果重新处理，应先缩小任务范围、压缩输入，并补充验证步骤后再继续。"
            positive_patterns = ("保留可用输出路径",)
            negative_patterns = ("未完全恢复失败事件", "结果证据不足")
            confidence = 0.58
        elif assessment.verdict is OutcomeVerdict.FAILURE:
            trigger = "outcome_failure"
            causal_analysis = "运行以失败终止，当前执行路径、模型选择或输入规模没有满足任务目标。"
            counterfactual = "如果重新处理，应先定位失败层级，压缩或拆分任务，再切换更可靠策略重试。"
            positive_patterns = ()
            negative_patterns = ("终止失败未恢复", "执行策略需要调整")
            confidence = 0.66
        else:
            trigger = "outcome_insufficient_evidence"
            causal_analysis = "运行证据不足，不能形成稳定经验。"
            counterfactual = "如果重新处理，应补充可验证输出、事件证据或成功标准。"
            positive_patterns = ()
            negative_patterns = ("缺少可验证结果",)
            confidence = 0.42

        return ReflectionRecord(
            id=uuid4(),
            tenant_id=assessment.tenant_id,
            user_id=assessment.user_id,
            memory_scope=assessment.memory_scope,
            source_run_id=assessment.source_run_id,
            trigger=trigger,
            outcome=assessment.verdict.value,
            causal_analysis=causal_analysis,
            counterfactual=counterfactual,
            positive_patterns=positive_patterns,
            negative_patterns=negative_patterns,
            proposed_experience_ids=(),
            confidence=confidence,
            created_at=self._now(),
        )
