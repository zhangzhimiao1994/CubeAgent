from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from agent_hub.cognitive.hierarchy import WorkingSetBuilder
from agent_hub.memory.types import MemoryCategory, MemoryLayer, MemoryRecord


def test_working_set_excludes_archived_memories_and_respects_limit() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    hot = _memory("reviewer timeout recovery", layer=MemoryLayer.EPISODIC, heat=0.9, now=now)
    warm = _memory("CubeAgent project boundary", layer=MemoryLayer.CORE, heat=0.4, now=now)
    archived = _memory("old noisy detail", layer=MemoryLayer.EPISODIC, heat=1.0, now=now).model_copy(
        update={"archived_at": now}
    )

    selected = WorkingSetBuilder().build(
        request="CubeAgent reviewer timeout",
        memories=(archived, warm, hot),
        limit=2,
    )

    assert [item.id for item in selected] == [hot.id, warm.id]


def test_working_set_prefers_working_layer_and_request_relevance() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    working = _memory("current task needs CI verification", layer=MemoryLayer.WORKING, heat=0.5, now=now)
    hot_irrelevant = _memory("travel planning detail", layer=MemoryLayer.EPISODIC, heat=0.95, now=now)
    relevant_core = _memory("CI must pass before push is considered complete", layer=MemoryLayer.CORE, heat=0.5, now=now)

    selected = WorkingSetBuilder().build(
        request="检查 CI verification",
        memories=(hot_irrelevant, relevant_core, working),
        limit=3,
    )

    assert selected[0].id == working.id
    assert selected[1].id == relevant_core.id
    assert selected[2].id == hot_irrelevant.id


def _memory(text: str, *, layer: MemoryLayer, heat: float, now: datetime) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        layer=layer,
        category=MemoryCategory.OTHER,
        text=text,
        confidence=0.8,
        created_at=now,
        updated_at=now,
        heat=heat,
    )
