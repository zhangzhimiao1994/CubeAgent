from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from agent_hub.memory.service import MemoryService
from agent_hub.memory.types import MemoryCategory, MemorySummaryPeriod, MemoryTier

TENANT_A = UUID("11111111-aaaa-4aaa-8aaa-111111111111")
USER_1 = UUID("33333333-1111-4111-8111-333333333333")


def memory_service_with_clock() -> tuple[MemoryService, list[datetime]]:
    now = [datetime(2026, 8, 6, tzinfo=UTC)]
    return MemoryService(now=lambda: now[0]), now


async def test_memory_tier_classifies_hot_warm_cold_and_archive() -> None:
    service, _ = memory_service_with_clock()
    hot = await service.add_candidate(
        tenant_id=TENANT_A,
        user_id=USER_1,
        text="hot memory",
    )
    warm = await service.add_candidate(
        tenant_id=TENANT_A,
        user_id=USER_1,
        text="warm memory",
    )
    cold = await service.add_candidate(
        tenant_id=TENANT_A,
        user_id=USER_1,
        text="cold memory",
    )
    assert hot.record is not None and warm.record is not None and cold.record is not None
    archived = await service.archive(cold.record.id, tenant_id=TENANT_A, user_id=USER_1, reason="manual_archive")

    assert service.classify_tier(hot.record.model_copy(update={"heat": 0.86})) is MemoryTier.HOT
    assert service.classify_tier(warm.record.model_copy(update={"heat": 0.45})) is MemoryTier.WARM
    assert service.classify_tier(cold.record.model_copy(update={"heat": 0.12})) is MemoryTier.COLD
    assert service.classify_tier(archived) is MemoryTier.ARCHIVE


async def test_archive_excludes_memory_from_search_without_user_forget_tombstone() -> None:
    service, _ = memory_service_with_clock()
    added = await service.add_candidate(
        tenant_id=TENANT_A,
        user_id=USER_1,
        text="archive keeps audit history",
    )
    assert added.record is not None

    archived = await service.archive(
        added.record.id,
        tenant_id=TENANT_A,
        user_id=USER_1,
        reason="cold_storage",
    )

    assert archived.archived_at is not None
    assert archived.archive_reason == "cold_storage"
    assert archived.deleted_at is None
    assert archived.tombstone_reason is None
    assert await service.search(tenant_id=TENANT_A, user_id=USER_1, query="audit history") == ()
    assert await service.inspect(tenant_id=TENANT_A, user_id=USER_1) == ()
    assert (await service.inspect(tenant_id=TENANT_A, user_id=USER_1, include_archived=True))[0].id == archived.id
    assert (await service.audit_events(tenant_id=TENANT_A))[-1].kind == "memory.archived"


async def test_consolidation_records_source_ids_and_can_archive_sources() -> None:
    service, _ = memory_service_with_clock()
    first = await service.add_candidate(
        tenant_id=TENANT_A,
        user_id=USER_1,
        text="first detail to consolidate",
        category=MemoryCategory.FACT,
        project_id="cube-agent",
    )
    second = await service.add_candidate(
        tenant_id=TENANT_A,
        user_id=USER_1,
        text="second detail to consolidate",
        category=MemoryCategory.LESSON,
        project_id="cube-agent",
    )
    assert first.record is not None and second.record is not None

    summary = await service.consolidate(
        tenant_id=TENANT_A,
        user_id=USER_1,
        period=MemorySummaryPeriod.DAY,
        project_id="cube-agent",
        archive_sources=True,
    )

    assert summary.source_memory_ids == (first.record.id, second.record.id)
    archived_sources = await service.inspect(
        tenant_id=TENANT_A,
        user_id=USER_1,
        include_archived=True,
    )
    archived_by_id = {record.id: record for record in archived_sources}
    assert archived_by_id[first.record.id].archived_at is not None
    assert archived_by_id[second.record.id].archived_at is not None
    results = await service.search(tenant_id=TENANT_A, user_id=USER_1, query="first detail")
    assert all(record.id not in {first.record.id, second.record.id} for record in results)
