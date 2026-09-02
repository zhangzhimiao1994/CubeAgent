from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from agent_hub.memory import MemoryMaintenanceService as ExportedMemoryMaintenanceService
from agent_hub.memory.maintenance import MemoryMaintenanceService
from agent_hub.memory.repository import InMemoryMemoryRepository
from agent_hub.memory.types import (
    MemoryAuditEvent,
    MemoryCategory,
    MemoryLayer,
    MemoryRecord,
    MemoryRetentionAction,
    MemoryRetentionPolicy,
    MemorySummaryPeriod,
)

TENANT_ID = UUID("11111111-aaaa-4aaa-8aaa-111111111111")
USER_ID = UUID("33333333-1111-4111-8111-333333333333")


def memory_record(
    text: str,
    *,
    now: datetime,
    layer: MemoryLayer = MemoryLayer.EPISODIC,
    category: MemoryCategory = MemoryCategory.OTHER,
    confidence: float = 0.5,
    heat: float = 0.5,
    created_days_ago: int | None = None,
    updated_days_ago: int = 0,
    recall_count: int = 0,
    locked: bool = False,
    metadata: dict[str, str] | None = None,
    deleted_days_ago: int | None = None,
    archived_days_ago: int | None = None,
) -> MemoryRecord:
    effective_created_days_ago = (
        max(updated_days_ago, deleted_days_ago or 0, archived_days_ago or 0)
        if created_days_ago is None
        else max(created_days_ago, updated_days_ago)
    )
    created_at = now - timedelta(days=effective_created_days_ago)
    updated_at = now - timedelta(days=updated_days_ago)
    return MemoryRecord(
        id=UUID(int=abs(hash(text)) % (2**128)),
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        layer=layer,
        category=category,
        text=text,
        confidence=confidence,
        created_at=created_at,
        updated_at=updated_at,
        heat=heat,
        recall_count=recall_count,
        locked=locked,
        metadata=metadata or {},
        deleted_at=None if deleted_days_ago is None else now - timedelta(days=deleted_days_ago),
        tombstone_reason=None if deleted_days_ago is None else "retention_expired",
        archived_at=None if archived_days_ago is None else now - timedelta(days=archived_days_ago),
        archive_reason=None if archived_days_ago is None else "cold_storage",
    )


def test_retention_policy_defaults_are_bounded() -> None:
    policy = MemoryRetentionPolicy()

    assert policy.stale_candidate_days == 30
    assert policy.cold_archive_days == 180
    assert policy.tombstone_purge_days == 90
    assert policy.archive_purge_days == 365
    assert policy.min_retention_score == 0.22
    assert policy.compress_after_source_count == 3
    assert policy.max_active_records_per_user == 1000

    with pytest.raises(ValueError):
        MemoryRetentionPolicy(min_retention_score=1.5)


def test_package_exports_memory_maintenance_service() -> None:
    assert ExportedMemoryMaintenanceService is MemoryMaintenanceService


async def test_repository_delete_physically_removes_record_but_keeps_audit_history() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    repository = InMemoryMemoryRepository()
    record = memory_record("old tombstone", now=now, deleted_days_ago=120)
    await repository.upsert(record)
    await repository.append_audit(
        MemoryAuditEvent(
            kind="memory.forgotten",
            tenant_id=TENANT_ID,
            user_id=USER_ID,
            memory_id=record.id,
            status="forgotten",
            reason="user_request",
            created_at=now,
        )
    )

    assert await repository.delete(record.id) is True
    assert await repository.get(record.id) is None
    assert len(await repository.audit_events()) == 1
    assert await repository.delete(record.id) is False


async def test_maintenance_keeps_locked_and_core_memories() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    service = MemoryMaintenanceService(now=lambda: now)
    locked = memory_record("locked user preference", now=now, locked=True, updated_days_ago=400)
    core = memory_record(
        "root operating rule",
        now=now,
        layer=MemoryLayer.CORE,
        confidence=0.2,
        heat=0.0,
        updated_days_ago=400,
    )

    assert service.evaluate(locked).action is MemoryRetentionAction.KEEP
    assert service.evaluate(locked).protected is True
    assert service.evaluate(core).action is MemoryRetentionAction.KEEP
    assert service.evaluate(core).protected is True


async def test_maintenance_archives_cold_records_and_purges_old_tombstones() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    repository = InMemoryMemoryRepository()
    cold = memory_record(
        "cold but maybe useful",
        now=now,
        confidence=0.45,
        heat=0.1,
        updated_days_ago=200,
    )
    tombstone = memory_record("old tombstone", now=now, deleted_days_ago=120)
    await repository.upsert(cold)
    await repository.upsert(tombstone)
    service = MemoryMaintenanceService(repository, now=lambda: now)

    dry_run = await service.maintain(apply=False)
    assert [decision.action for decision in dry_run.decisions] == [
        MemoryRetentionAction.ARCHIVE,
        MemoryRetentionAction.PURGE,
    ]
    assert await repository.get(cold.id) == cold
    assert await repository.get(tombstone.id) == tombstone

    applied = await service.maintain(apply=True)
    assert applied.archived == 1
    assert applied.purged == 1
    assert (await repository.get(cold.id)).archived_at == now  # type: ignore[union-attr]
    assert await repository.get(tombstone.id) is None
    audit_kinds = [event.kind for event in await repository.audit_events()]
    assert audit_kinds == ["memory.maintenance_archived", "memory.maintenance_purged"]


async def test_low_value_candidate_is_tombstoned_before_physical_delete() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    repository = InMemoryMemoryRepository()
    candidate = memory_record(
        "unverified noisy candidate",
        now=now,
        confidence=0.12,
        heat=0.0,
        updated_days_ago=45,
        metadata={"status": "candidate", "contradictions": "2"},
    )
    await repository.upsert(candidate)
    service = MemoryMaintenanceService(repository, now=lambda: now)

    decision = service.evaluate(candidate)
    assert decision.action is MemoryRetentionAction.TOMBSTONE
    assert decision.score < MemoryRetentionPolicy().min_retention_score

    applied = await service.maintain(apply=True)
    assert applied.tombstoned == 1
    tombstoned = await repository.get(candidate.id)
    assert tombstoned is not None
    assert tombstoned.deleted_at == now
    assert tombstoned.tombstone_reason == "low_retention_score"


async def test_maintenance_compresses_repeated_active_memories_before_archiving_sources() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    repository = InMemoryMemoryRepository()
    source_records = tuple(
        memory_record(
            f"用户偏好第 {index} 条：调度卡片需要摘要。",
            now=now,
            category=MemoryCategory.PREFERENCE,
            confidence=0.78,
            heat=0.62,
            updated_days_ago=index,
        ).model_copy(update={"project_id": "cubeagent", "conversation_id": "conv-1"})
        for index in range(3)
    )
    locked = memory_record(
        "锁定核心偏好不参与压缩。",
        now=now,
        category=MemoryCategory.PREFERENCE,
        locked=True,
    ).model_copy(update={"project_id": "cubeagent", "conversation_id": "conv-1"})
    for record in (*source_records, locked):
        await repository.upsert(record)
    service = MemoryMaintenanceService(repository, now=lambda: now)

    dry_run = await service.maintain(apply=False)
    assert [decision.action for decision in dry_run.decisions].count(MemoryRetentionAction.COMPRESS) == 3
    assert await repository.list_all() == (*source_records, locked)

    applied = await service.maintain(apply=True)

    assert applied.compressed == 1
    assert applied.archived == 3
    records = await repository.list_all()
    summaries = [record for record in records if record.category is MemoryCategory.SUMMARY]
    assert len(summaries) == 1
    assert summaries[0].summary_period is MemorySummaryPeriod.DAY
    assert set(summaries[0].source_memory_ids) == {record.id for record in source_records}
    assert summaries[0].locked is True
    assert (await repository.get(locked.id)).archived_at is None  # type: ignore[union-attr]
    for source in source_records:
        archived = await repository.get(source.id)
        assert archived is not None
        assert archived.archived_at == now
        assert archived.archive_reason == f"consolidated_into:{summaries[0].id}"
    assert [event.kind for event in await repository.audit_events()].count("memory.maintenance_compressed") == 1


async def test_maintenance_enforces_active_record_limit_per_user_without_cross_user_mutation() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    repository = InMemoryMemoryRepository()
    policy = MemoryRetentionPolicy(max_active_records_per_user=2, compress_after_source_count=10)
    for index in range(3):
        await repository.upsert(
            memory_record(
                f"user one active memory {index}",
                now=now,
                confidence=0.55 + (index * 0.05),
                heat=0.2 + (index * 0.1),
            )
        )
    other_user_id = UUID("44444444-1111-4111-8111-333333333333")
    other_user_record = memory_record("other user memory remains active", now=now).model_copy(
        update={"user_id": other_user_id}
    )
    await repository.upsert(other_user_record)
    service = MemoryMaintenanceService(repository, policy=policy, now=lambda: now)

    applied = await service.maintain(apply=True)

    assert applied.archived == 1
    user_records = await repository.list_for_user(TENANT_ID, USER_ID)
    assert sum(1 for record in user_records if record.archived_at is None and record.deleted_at is None) == 2
    assert (await repository.get(other_user_record.id)).archived_at is None  # type: ignore[union-attr]


async def test_maintenance_rechecks_protection_before_applying_destructive_decision() -> None:
    class LockingRepository(InMemoryMemoryRepository):
        def __init__(self, *, now: datetime) -> None:
            super().__init__()
            self._now = now
            self._locked_once = False

        async def get(self, memory_id: UUID) -> MemoryRecord | None:
            record = await super().get(memory_id)
            if record is not None and not self._locked_once:
                self._locked_once = True
                locked = record.model_copy(update={"locked": True, "updated_at": self._now})
                await super().upsert(locked)
                return locked
            return record

    now = datetime(2026, 9, 2, tzinfo=UTC)
    repository = LockingRepository(now=now)
    record = memory_record("old tombstone becomes protected", now=now, deleted_days_ago=120)
    await repository.upsert(record)
    service = MemoryMaintenanceService(repository, now=lambda: now)

    applied = await service.maintain(apply=True)

    assert applied.purged == 0
    retained = await repository.get(record.id)
    assert retained is not None
    assert retained.locked is True
