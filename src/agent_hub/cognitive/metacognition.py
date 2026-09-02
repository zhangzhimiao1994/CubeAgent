from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CognitiveGateLevel(StrEnum):
    LIGHT = "light"
    ADVANCED = "advanced"


class MetacognitionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: CognitiveGateLevel
    actions: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    context_budget: int = Field(ge=1)


class MetacognitionService:
    def __init__(self, *, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or (lambda: datetime.now(UTC))

    def assess(
        self,
        *,
        request: str,
        mode: str,
        uncertainty_signals: tuple[str, ...] = (),
        conflict_count: int = 0,
        last_verified_at: datetime | None = None,
    ) -> MetacognitionDecision:
        reasons: list[str] = []
        if mode in {"hybrid", "dispatch", "auto"}:
            reasons.append("multi_agent_mode")
        if len(request) >= 48 or any(token in request for token in ("系统", "三阶段", "测试", "审查", "提交", "CI")):
            reasons.append("complex_request")
        if uncertainty_signals:
            reasons.append("uncertainty_signals")
        if conflict_count > 0:
            reasons.append("cognitive_conflicts")
        if self._is_stale(last_verified_at):
            reasons.append("stale_context")
        if not reasons:
            return MetacognitionDecision(
                level=CognitiveGateLevel.LIGHT,
                actions=("answer_directly",),
                context_budget=3,
            )
        actions = ["retrieve_memory", "verify_state"]
        if "multi_agent_mode" in reasons or "complex_request" in reasons:
            actions.append("delegate_agent")
        if "cognitive_conflicts" in reasons:
            actions.append("resolve_conflicts")
        return MetacognitionDecision(
            level=CognitiveGateLevel.ADVANCED,
            actions=tuple(dict.fromkeys(actions)),
            reasons=tuple(reasons),
            context_budget=8,
        )

    def _is_stale(self, last_verified_at: datetime | None) -> bool:
        if last_verified_at is None:
            return True
        return (self._now() - last_verified_at).days >= 30
