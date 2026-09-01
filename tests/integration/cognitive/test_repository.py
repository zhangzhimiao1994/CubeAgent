from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent_hub.cognitive.repository import PersistentCognitiveRecordRepository
from agent_hub.cognitive.types import BeliefRecord, CognitiveEvidence, CognitiveMemoryScope
from agent_hub.db.models import AdminResourceRow
from agent_hub.db.session import build_database
from tests.integration.conftest import _clean_database


@pytest.fixture
async def cognitive_session_factory(
    database_url: str,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    database = build_database(database_url)
    try:
        await _clean_database(database.session_factory)
        yield database.session_factory
    finally:
        await _clean_database(database.session_factory)
        await database.dispose()


async def test_persistent_cognitive_record_repository_scopes_beliefs_by_user_and_root(
    cognitive_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime.now(UTC)
    tenant_id = uuid4()
    owner_user_id = uuid4()
    other_user_id = uuid4()
    repository = PersistentCognitiveRecordRepository(cognitive_session_factory)
    user_belief = BeliefRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=owner_user_id,
        memory_scope=CognitiveMemoryScope.USER,
        subject="user.confirmation_style",
        claim="用户偏好少确认。",
        confidence=0.74,
        evidence=(CognitiveEvidence(source_type="feedback", source_id="msg-1", note="explicit preference"),),
        contradictions=(),
        status="active",
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )
    root_belief = BeliefRecord(
        id=uuid4(),
        tenant_id=tenant_id,
        user_id=owner_user_id,
        memory_scope=CognitiveMemoryScope.ROOT,
        subject="project.boundary",
        claim="CubeAgent 不实现 harness 代码执行。",
        confidence=0.91,
        evidence=(CognitiveEvidence(source_type="handoff", source_id="HANDOFF", note="project boundary"),),
        contradictions=(),
        status="active",
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )

    await repository.upsert(user_belief)
    await repository.upsert(root_belief)

    owner_records = await repository.list_for_user(BeliefRecord, tenant_id=tenant_id, user_id=owner_user_id)
    other_records = await repository.list_for_user(BeliefRecord, tenant_id=tenant_id, user_id=other_user_id)

    assert [item.id for item in owner_records] == [user_belief.id, root_belief.id]
    assert [item.id for item in other_records] == [root_belief.id]

    async with cognitive_session_factory() as session:
        rows = (
            await session.execute(
                select(AdminResourceRow)
                .where(AdminResourceRow.tenant_id == tenant_id)
                .where(AdminResourceRow.kind == "hermes")
                .where(AdminResourceRow.resource_id.like("cognitive_belief:%"))
            )
        ).scalars().all()

    assert len(rows) == 2
    assert {row.payload["record_type"] for row in rows} == {"belief"}

    async with cognitive_session_factory() as session, session.begin():
        await session.execute(delete(AdminResourceRow).where(AdminResourceRow.tenant_id == tenant_id))
