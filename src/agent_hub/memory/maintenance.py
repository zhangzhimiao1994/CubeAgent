from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from agent_hub.memory.repository import InMemoryMemoryRepository
from agent_hub.memory.service import MemoryService
from agent_hub.memory.types import (
    MemoryAuditEvent,
    MemoryCategory,
    MemoryLayer,
    MemoryMaintenanceResult,
    MemoryRecord,
    MemoryRetentionAction,
    MemoryRetentionDecision,
    MemoryRetentionPolicy,
    MemorySummaryPeriod,
    MemoryTier,
)


class MemoryMaintenanceService:
    """Bounded memory maintenance with dry-run decisions and explicit audits."""

    def __init__(
        self,
        repository: InMemoryMemoryRepository | None = None,
        *,
        policy: MemoryRetentionPolicy | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository or InMemoryMemoryRepository()
        self._policy = policy or MemoryRetentionPolicy()
        self._now = now or (lambda: datetime.now(UTC))
        self._memory_service = MemoryService(self._repository, now=self._now)

    def evaluate(self, record: MemoryRecord) -> MemoryRetentionDecision:
        score = self._retention_score(record)
        if record.locked or record.layer is MemoryLayer.CORE:
            return MemoryRetentionDecision(
                memory_id=record.id,
                action=MemoryRetentionAction.KEEP,
                score=max(score, self._policy.min_retention_score),
                reason="protected_core_or_locked_memory",
                protected=True,
            )

        now = self._now()
        if record.deleted_at is not None:
            age_days = (now - record.deleted_at).days
            if age_days >= self._policy.tombstone_purge_days:
                return MemoryRetentionDecision(
                    memory_id=record.id,
                    action=MemoryRetentionAction.PURGE,
                    score=score,
                    reason="tombstone_retention_window_elapsed",
                )
            return MemoryRetentionDecision(
                memory_id=record.id,
                action=MemoryRetentionAction.KEEP,
                score=score,
                reason="recent_tombstone_retained_for_auditability",
            )

        if record.archived_at is not None:
            age_days = (now - record.archived_at).days
            if age_days >= self._policy.archive_purge_days:
                return MemoryRetentionDecision(
                    memory_id=record.id,
                    action=MemoryRetentionAction.PURGE,
                    score=score,
                    reason="archive_retention_window_elapsed",
                )
            return MemoryRetentionDecision(
                memory_id=record.id,
                action=MemoryRetentionAction.KEEP,
                score=score,
                reason="archived_memory_retained",
            )

        if self._memory_service.classify_tier(record) is MemoryTier.COLD:
            age_days = (now - self._last_activity(record)).days
            if age_days >= self._policy.cold_archive_days:
                return MemoryRetentionDecision(
                    memory_id=record.id,
                    action=MemoryRetentionAction.ARCHIVE,
                    score=score,
                    reason="cold_memory_archive_window_elapsed",
                )

        if score < self._policy.min_retention_score:
            return MemoryRetentionDecision(
                memory_id=record.id,
                action=MemoryRetentionAction.TOMBSTONE,
                score=score,
                reason="low_retention_score",
            )

        if self._memory_service.classify_tier(record) is MemoryTier.COLD:
            return MemoryRetentionDecision(
                memory_id=record.id,
                action=MemoryRetentionAction.COOL_DOWN,
                score=score,
                reason="cold_memory_kept_in_working_set",
            )

        return MemoryRetentionDecision(
            memory_id=record.id,
            action=MemoryRetentionAction.KEEP,
            score=score,
            reason="retention_score_sufficient",
        )

    async def maintain(self, *, apply: bool = False) -> MemoryMaintenanceResult:
        records = await self._repository.list_all()
        compression_groups = self._compression_groups(records)
        compressed_source_ids = {record.id for group in compression_groups for record in group}
        compression_decisions = tuple(
            MemoryRetentionDecision(
                memory_id=record.id,
                action=MemoryRetentionAction.COMPRESS,
                score=self._retention_score(record),
                reason="compression_candidate_group",
            )
            for group in compression_groups
            for record in group
        )
        lifecycle_decisions = tuple(
            self.evaluate(record) for record in records if record.id not in compressed_source_ids
        )
        decisions = self._enforce_active_record_limits(compression_decisions + lifecycle_decisions, records)
        if not apply:
            return MemoryMaintenanceResult(decisions=decisions)

        compressed = 0
        archived = 0
        tombstoned = 0
        purged = 0
        cooled_down = 0
        for group in compression_groups:
            archived_in_group = await self._compress_group(group)
            if archived_in_group:
                archived += archived_in_group
                compressed += 1

        for decision in decisions:
            if decision.action is MemoryRetentionAction.COMPRESS:
                continue
            record = await self._repository.get(decision.memory_id)
            if record is None or decision.protected or record.locked or record.layer is MemoryLayer.CORE:
                continue
            if decision.action in {
                MemoryRetentionAction.ARCHIVE,
                MemoryRetentionAction.TOMBSTONE,
                MemoryRetentionAction.PURGE,
            } and not self._destructive_decision_still_valid(record, decision):
                continue
            if decision.action is MemoryRetentionAction.ARCHIVE:
                await self._archive(record, reason=decision.reason)
                archived += 1
            elif decision.action is MemoryRetentionAction.TOMBSTONE:
                await self._tombstone(record, reason=decision.reason)
                tombstoned += 1
            elif decision.action is MemoryRetentionAction.PURGE:
                if await self._repository.delete(record.id):
                    await self._audit("memory.maintenance_purged", record, "purged", decision.reason)
                    purged += 1
            elif decision.action is MemoryRetentionAction.COOL_DOWN:
                cooled = record.model_copy(
                    update={
                        "heat": max(0.0, record.heat - 0.05),
                        "updated_at": self._now(),
                    }
                )
                await self._repository.upsert(cooled)
                await self._audit("memory.maintenance_cooled_down", cooled, "cooled_down", decision.reason)
                cooled_down += 1

        return MemoryMaintenanceResult(
            decisions=decisions,
            compressed=compressed,
            archived=archived,
            tombstoned=tombstoned,
            purged=purged,
            cooled_down=cooled_down,
        )

    def _enforce_active_record_limits(
        self,
        decisions: tuple[MemoryRetentionDecision, ...],
        records: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryRetentionDecision, ...]:
        decision_by_id = {decision.memory_id: decision for decision in decisions}
        records_by_owner: dict[tuple[object, object], list[MemoryRecord]] = {}
        for record in records:
            decision = decision_by_id.get(record.id)
            if (
                record.deleted_at is not None
                or record.archived_at is not None
                or record.locked
                or record.layer is MemoryLayer.CORE
                or (decision is not None and decision.action is MemoryRetentionAction.COMPRESS)
            ):
                continue
            records_by_owner.setdefault((record.tenant_id, record.user_id), []).append(record)

        replacements: dict[object, MemoryRetentionDecision] = {}
        for owner_records in records_by_owner.values():
            overflow = len(owner_records) - self._policy.max_active_records_per_user
            if overflow <= 0:
                continue
            ranked = sorted(owner_records, key=lambda record: (self._retention_score(record), record.heat, record.updated_at))
            for record in ranked[:overflow]:
                score = self._retention_score(record)
                action = (
                    MemoryRetentionAction.TOMBSTONE
                    if score < self._policy.min_retention_score
                    else MemoryRetentionAction.ARCHIVE
                )
                replacements[record.id] = MemoryRetentionDecision(
                    memory_id=record.id,
                    action=action,
                    score=score,
                    reason="active_working_set_limit_exceeded",
                )

        if not replacements:
            return decisions
        return tuple(replacements.get(decision.memory_id, decision) for decision in decisions)

    def _compression_groups(self, records: tuple[MemoryRecord, ...]) -> tuple[tuple[MemoryRecord, ...], ...]:
        grouped: dict[
            tuple[object, object, str | None, str | None, MemoryCategory],
            list[MemoryRecord],
        ] = {}
        for record in records:
            if not self._compressible(record):
                continue
            key = (
                record.tenant_id,
                record.user_id,
                record.project_id,
                record.conversation_id,
                record.category,
            )
            grouped.setdefault(key, []).append(record)
        groups: list[tuple[MemoryRecord, ...]] = []
        for group in grouped.values():
            if len(group) < self._policy.compress_after_source_count:
                continue
            group.sort(key=lambda record: record.created_at)
            groups.append(tuple(group))
        return tuple(groups)

    def _compressible(self, record: MemoryRecord) -> bool:
        now = self._now()
        return (
            not record.locked
            and record.layer is not MemoryLayer.CORE
            and record.category is not MemoryCategory.SUMMARY
            and record.summary_period is MemorySummaryPeriod.NONE
            and record.deleted_at is None
            and record.archived_at is None
            and (record.expires_at is None or record.expires_at > now)
        )

    async def _compress_group(self, records: tuple[MemoryRecord, ...]) -> int:
        current_records: list[MemoryRecord] = []
        for record in records:
            current = await self._repository.get(record.id)
            if current is not None and self._compressible(current):
                current_records.append(current)
        if len(current_records) < self._policy.compress_after_source_count:
            return 0
        records = tuple(current_records)
        now = self._now()
        summary = MemoryRecord(
            id=uuid4(),
            tenant_id=records[0].tenant_id,
            user_id=records[0].user_id,
            layer=MemoryLayer.EPISODIC,
            category=MemoryCategory.SUMMARY,
            text=_summary_text(records),
            confidence=min(0.95, max(record.confidence for record in records)),
            created_at=now,
            updated_at=now,
            heat=0.8,
            source_memory_ids=tuple(record.id for record in records),
            locked=True,
            project_id=records[0].project_id,
            conversation_id=records[0].conversation_id,
            summary_period=MemorySummaryPeriod.DAY,
            metadata={"source_count": str(len(records)), "maintenance": "compressed"},
        )
        await self._repository.upsert(summary)
        await self._audit(
            "memory.maintenance_compressed",
            summary,
            "compressed",
            f"source_count={len(records)}",
        )
        archived = 0
        for record in records:
            archived_record = record.model_copy(
                update={
                    "archived_at": now,
                    "archive_reason": f"consolidated_into:{summary.id}",
                    "updated_at": now,
                }
            )
            await self._repository.upsert(archived_record)
            await self._audit(
                "memory.maintenance_archived",
                archived_record,
                "archived",
                archived_record.archive_reason,
            )
            archived += 1
        return archived

    async def _archive(self, record: MemoryRecord, *, reason: str) -> MemoryRecord:
        now = self._now()
        archived = record.model_copy(
            update={
                "archived_at": now,
                "archive_reason": reason,
                "updated_at": now,
            }
        )
        await self._repository.upsert(archived)
        await self._audit("memory.maintenance_archived", archived, "archived", reason)
        return archived

    async def _tombstone(self, record: MemoryRecord, *, reason: str) -> MemoryRecord:
        now = self._now()
        tombstoned = record.model_copy(
            update={
                "deleted_at": now,
                "tombstone_reason": reason,
                "updated_at": now,
            }
        )
        await self._repository.upsert(tombstoned)
        await self._audit("memory.maintenance_tombstoned", tombstoned, "tombstoned", reason)
        return tombstoned

    async def _audit(self, kind: str, record: MemoryRecord, status: str, reason: str | None) -> None:
        await self._repository.append_audit(
            MemoryAuditEvent(
                kind=kind,
                tenant_id=record.tenant_id,
                user_id=record.user_id,
                memory_id=record.id,
                status=status,
                reason=reason,
                created_at=self._now(),
            )
        )

    def _destructive_decision_still_valid(
        self,
        record: MemoryRecord,
        decision: MemoryRetentionDecision,
    ) -> bool:
        if decision.reason == "active_working_set_limit_exceeded":
            return record.deleted_at is None and record.archived_at is None
        current_decision = self.evaluate(record)
        return current_decision.action is decision.action and not current_decision.protected

    def _retention_score(self, record: MemoryRecord) -> float:
        score = (record.confidence * 0.45) + (record.heat * 0.30) + (min(record.recall_count, 20) / 20 * 0.15)
        if record.source_memory_ids:
            score += 0.05
        if record.locked or record.layer is MemoryLayer.CORE:
            score += 0.20
        if record.metadata.get("status") == "candidate":
            score -= 0.20
        if record.metadata.get("conflict") or record.metadata.get("contradictions"):
            score -= 0.25
        return min(1.0, max(0.0, score))

    @staticmethod
    def _last_activity(record: MemoryRecord) -> datetime:
        return record.last_recalled_at or record.updated_at


def _summary_text(records: tuple[MemoryRecord, ...]) -> str:
    text = "; ".join(record.text for record in records)
    return f"memory consolidation summary: {text}"[:4096]
