from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from agent_hub.cognitive.types import BeliefRecord, ExperienceStatus, StrategyStatus


class ConflictResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class ConflictResolutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ConflictResolutionStatus
    winning_record_id: str | None
    conflicting_record_ids: tuple[str, ...]
    reason: str


class ConflictResolutionEngine:
    def resolve_beliefs(
        self,
        records: tuple[BeliefRecord, ...],
        *,
        minimum_confidence_gap: float = 0.18,
    ) -> ConflictResolutionDecision:
        active = tuple(record for record in records if record.status not in {"deprecated", "rejected"})
        if not active:
            return ConflictResolutionDecision(
                status=ConflictResolutionStatus.UNRESOLVED,
                winning_record_id=None,
                conflicting_record_ids=(),
                reason="no active belief evidence",
            )
        sorted_records = sorted(active, key=lambda item: item.confidence, reverse=True)
        winner = sorted_records[0]
        conflicting_ids = tuple(str(record.id) for record in sorted_records)
        if len(sorted_records) == 1:
            return ConflictResolutionDecision(
                status=ConflictResolutionStatus.RESOLVED,
                winning_record_id=str(winner.id),
                conflicting_record_ids=conflicting_ids,
                reason="single active belief",
            )
        runner_up = sorted_records[1]
        if winner.confidence - runner_up.confidence < minimum_confidence_gap:
            return ConflictResolutionDecision(
                status=ConflictResolutionStatus.UNRESOLVED,
                winning_record_id=None,
                conflicting_record_ids=conflicting_ids,
                reason="competing beliefs have similar confidence",
            )
        return ConflictResolutionDecision(
            status=ConflictResolutionStatus.RESOLVED,
            winning_record_id=str(winner.id),
            conflicting_record_ids=conflicting_ids,
            reason="highest confidence belief is sufficiently stronger",
        )


class ConfidenceCalibrationService:
    def calibrate(
        self,
        *,
        confidence: float,
        success_count: int,
        failure_count: int,
        contradiction_count: int,
        last_verified_at: datetime | None,
        now: datetime | None = None,
    ) -> float:
        timestamp = now or datetime.now(UTC)
        delta = (success_count * 0.03) - (failure_count * 0.05) - (contradiction_count * 0.07)
        if last_verified_at is None:
            delta -= 0.05
        else:
            age_days = max(0, (timestamp - last_verified_at).days)
            if age_days >= 90:
                delta -= 0.08
            elif age_days >= 30:
                delta -= 0.03
        return max(0.0, min(1.0, confidence + delta))


class AntiLearningService:
    def experience_status(
        self,
        *,
        current_status: ExperienceStatus,
        confidence: float,
        success_count: int,
        failure_count: int,
    ) -> ExperienceStatus:
        if current_status is ExperienceStatus.REJECTED:
            return ExperienceStatus.REJECTED
        if confidence <= 0.24:
            return ExperienceStatus.DEPRECATED
        if failure_count > success_count and confidence < 0.45:
            return ExperienceStatus.DEPRECATED
        return current_status

    def strategy_status(
        self,
        *,
        current_status: StrategyStatus,
        confidence: float,
        success_count: int,
        failure_count: int,
    ) -> StrategyStatus:
        if current_status is StrategyStatus.REJECTED:
            return StrategyStatus.REJECTED
        if confidence <= 0.24:
            return StrategyStatus.DEPRECATED
        if failure_count > success_count and confidence < 0.45:
            return StrategyStatus.CONTESTED
        return current_status


def conflict_record_ids(records: tuple[BeliefRecord, ...]) -> tuple[UUID, ...]:
    return tuple(record.id for record in records)
