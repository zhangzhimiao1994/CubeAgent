from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TypeVar, cast
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent_hub.cognitive.types import (
    BeliefRecord,
    CognitiveMemoryScope,
    ExperienceRecord,
    ReflectionRecord,
    RelationshipStateRecord,
    SkillCandidateRecord,
    StrategyRecord,
    WorldStateRecord,
)
from agent_hub.db.models import AdminResourceRow

_HERMES_KIND = "hermes"
_EXPERIENCE_PREFIX = "cognitive_experience:"
_RECORD_STORAGE: dict[type[object], tuple[str, str]] = {
    ReflectionRecord: ("cognitive_reflection:", "reflection"),
    BeliefRecord: ("cognitive_belief:", "belief"),
    RelationshipStateRecord: ("cognitive_relationship:", "relationship"),
    WorldStateRecord: ("cognitive_world:", "world_state"),
    SkillCandidateRecord: ("cognitive_skill:", "skill_candidate"),
    StrategyRecord: ("cognitive_strategy:", "strategy"),
}

CognitiveRecord = (
    ReflectionRecord
    | BeliefRecord
    | RelationshipStateRecord
    | WorldStateRecord
    | SkillCandidateRecord
    | StrategyRecord
)
CognitiveRecordT = TypeVar("CognitiveRecordT", bound=CognitiveRecord)


class ExperienceRepositoryError(RuntimeError):
    pass


class InMemoryCognitiveRecordRepository:
    def __init__(self, records: Iterable[CognitiveRecord] = ()) -> None:
        self._records: dict[type[object], dict[str, CognitiveRecord]] = {}
        for record in records:
            self._bucket(type(record))[_record_key(record)] = record

    async def upsert(self, record: CognitiveRecordT) -> CognitiveRecordT:
        self._bucket(type(record))[_record_key(record)] = record
        return record

    async def get(
        self, record_type: type[CognitiveRecordT], record_id: str | UUID
    ) -> CognitiveRecordT | None:
        record = self._bucket(record_type).get(str(record_id))
        if record is None:
            return None
        return cast(CognitiveRecordT, record)

    async def list_for_user(
        self,
        record_type: type[CognitiveRecordT],
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> tuple[CognitiveRecordT, ...]:
        records = (
            record
            for record in self._bucket(record_type).values()
            if record.tenant_id == tenant_id
            and (record.user_id == user_id or record.memory_scope is CognitiveMemoryScope.ROOT)
        )
        return tuple(
            cast(
                CognitiveRecordT,
                record,
            )
            for record in sorted(records, key=lambda item: item.created_at)
        )

    async def delete(
        self,
        record_type: type[CognitiveRecordT],
        record_id: str | UUID,
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> bool:
        bucket = self._bucket(record_type)
        record = bucket.get(str(record_id))
        if record is None:
            return False
        if record.tenant_id != tenant_id or record.user_id != user_id:
            raise PermissionError("cognitive record is not visible to caller")
        del bucket[str(record_id)]
        return True

    def _bucket(self, record_type: type[object]) -> dict[str, CognitiveRecord]:
        return self._records.setdefault(record_type, {})


class InMemoryExperienceRepository:
    def __init__(self, records: Iterable[ExperienceRecord] = ()) -> None:
        self._records = {record.id: record for record in records}

    async def upsert(self, record: ExperienceRecord) -> ExperienceRecord:
        self._records[record.id] = record
        return record

    async def get(self, record_id: UUID) -> ExperienceRecord | None:
        return self._records.get(record_id)

    async def list_for_user(self, tenant_id: UUID, user_id: UUID) -> tuple[ExperienceRecord, ...]:
        return tuple(
            sorted(
                (
                    record
                    for record in self._records.values()
                    if record.tenant_id == tenant_id
                    and (
                        record.user_id == user_id
                        or record.memory_scope is CognitiveMemoryScope.ROOT
                    )
                ),
                key=lambda item: item.created_at,
            )
        )

    async def delete(self, record_id: UUID, *, tenant_id: UUID, user_id: UUID) -> bool:
        record = self._records.get(record_id)
        if record is None:
            return False
        if record.tenant_id != tenant_id or record.user_id != user_id:
            raise PermissionError("experience is not visible to caller")
        del self._records[record_id]
        return True


class PersistentCognitiveRecordRepository:
    """Persist non-experience cognitive records in the existing Hermes bucket."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert(self, record: CognitiveRecordT) -> CognitiveRecordT:
        prefix, record_type_name = _record_storage(type(record))
        resource_id = f"{prefix}{_record_key(record)}"
        payload = {
            **record.model_dump(mode="json"),
            "record_type": record_type_name,
            "resource_id": resource_id,
            "storage_kind": _HERMES_KIND,
        }
        statement = (
            insert(AdminResourceRow)
            .values(
                id=uuid4(),
                tenant_id=record.tenant_id,
                kind=_HERMES_KIND,
                resource_id=resource_id,
                payload=payload,
            )
            .on_conflict_do_update(
                index_elements=[
                    AdminResourceRow.tenant_id,
                    AdminResourceRow.kind,
                    AdminResourceRow.resource_id,
                ],
                set_={"payload": payload},
            )
        )
        async with self._session_factory() as session, session.begin():
            await session.execute(statement)
        return record

    async def get(
        self, record_type: type[CognitiveRecordT], record_id: str | UUID
    ) -> CognitiveRecordT | None:
        prefix, _record_type_name = _record_storage(record_type)
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.kind == _HERMES_KIND)
                    .where(AdminResourceRow.resource_id == f"{prefix}{record_id}")
                )
            ).scalar_one_or_none()
        return None if row is None else _cognitive_record_from_payload(record_type, dict(row.payload))

    async def list_for_user(
        self,
        record_type: type[CognitiveRecordT],
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> tuple[CognitiveRecordT, ...]:
        prefix, _record_type_name = _record_storage(record_type)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == tenant_id)
                    .where(AdminResourceRow.kind == _HERMES_KIND)
                    .where(AdminResourceRow.resource_id.like(f"{prefix}%"))
                    .order_by(AdminResourceRow.created_at)
                )
            ).scalars()
            records = [
                _cognitive_record_from_payload(record_type, dict(row.payload))
                for row in rows
            ]
        return tuple(
            record
            for record in records
            if record.user_id == user_id or record.memory_scope is CognitiveMemoryScope.ROOT
        )

    async def delete(
        self,
        record_type: type[CognitiveRecordT],
        record_id: str | UUID,
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> bool:
        existing = await self.get(record_type, record_id)
        if existing is None:
            return False
        if existing.tenant_id != tenant_id or existing.user_id != user_id:
            raise PermissionError("cognitive record is not visible to caller")
        prefix, _record_type_name = _record_storage(record_type)
        async with self._session_factory() as session, session.begin():
            await session.execute(
                delete(AdminResourceRow)
                .where(AdminResourceRow.tenant_id == tenant_id)
                .where(AdminResourceRow.kind == _HERMES_KIND)
                .where(AdminResourceRow.resource_id == f"{prefix}{record_id}")
            )
        return True


class PersistentExperienceRepository:
    """Store cognitive experiences in the existing Hermes resource bucket.

    The current `agent_hub_admin_resources.kind` check constraint does not allow
    new cognitive-specific kinds. The first slice therefore stores records under
    `kind='hermes'` with `resource_id='cognitive_experience:<uuid>'`.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert(self, record: ExperienceRecord) -> ExperienceRecord:
        resource_id = _resource_id(record.id)
        payload = {
            **record.model_dump(mode="json"),
            "active_for_runtime": record.active_for_runtime,
            "resource_id": resource_id,
            "storage_kind": _HERMES_KIND,
        }
        statement = (
            insert(AdminResourceRow)
            .values(
                id=uuid4(),
                tenant_id=record.tenant_id,
                kind=_HERMES_KIND,
                resource_id=resource_id,
                payload=payload,
            )
            .on_conflict_do_update(
                index_elements=[
                    AdminResourceRow.tenant_id,
                    AdminResourceRow.kind,
                    AdminResourceRow.resource_id,
                ],
                set_={"payload": payload},
            )
        )
        async with self._session_factory() as session, session.begin():
            await session.execute(statement)
        return record

    async def get(self, record_id: UUID) -> ExperienceRecord | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.kind == _HERMES_KIND)
                    .where(AdminResourceRow.resource_id == _resource_id(record_id))
                )
            ).scalar_one_or_none()
        return None if row is None else _record_from_payload(dict(row.payload))

    async def list_for_user(self, tenant_id: UUID, user_id: UUID) -> tuple[ExperienceRecord, ...]:
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == tenant_id)
                    .where(AdminResourceRow.kind == _HERMES_KIND)
                    .where(AdminResourceRow.resource_id.like(f"{_EXPERIENCE_PREFIX}%"))
                    .order_by(AdminResourceRow.created_at)
                )
            ).scalars()
            records = [_record_from_payload(dict(row.payload)) for row in rows]
        return tuple(
            record
            for record in records
            if record.user_id == user_id or record.memory_scope is CognitiveMemoryScope.ROOT
        )

    async def delete(self, record_id: UUID, *, tenant_id: UUID, user_id: UUID) -> bool:
        existing = await self.get(record_id)
        if existing is None:
            return False
        if existing.tenant_id != tenant_id or existing.user_id != user_id:
            raise PermissionError("experience is not visible to caller")
        async with self._session_factory() as session, session.begin():
            await session.execute(
                delete(AdminResourceRow)
                .where(AdminResourceRow.tenant_id == tenant_id)
                .where(AdminResourceRow.kind == _HERMES_KIND)
                .where(AdminResourceRow.resource_id == _resource_id(record_id))
            )
        return True


def _resource_id(record_id: UUID) -> str:
    return f"{_EXPERIENCE_PREFIX}{record_id}"


def _record_key(record: CognitiveRecord) -> str:
    return str(record.id)


def _record_storage(record_type: type[object]) -> tuple[str, str]:
    storage = _RECORD_STORAGE.get(record_type)
    if storage is None:
        raise TypeError("unsupported cognitive record type")
    return storage


def _cognitive_record_from_payload[RecordT: CognitiveRecord](
    record_type: type[RecordT], payload: dict[str, object]
) -> RecordT:
    fields = getattr(record_type, "model_fields", None)
    if not isinstance(fields, dict):
        raise TypeError("unsupported cognitive record type")
    normalized = {key: value for key, value in payload.items() if key in fields}
    try:
        return cast(RecordT, record_type.model_validate_json(json.dumps(normalized)))
    except ValueError as error:
        raise ExperienceRepositoryError("stored cognitive record is invalid") from error


def _record_from_payload(payload: dict[str, object]) -> ExperienceRecord:
    record_fields = set(ExperienceRecord.model_fields)
    normalized = {key: value for key, value in payload.items() if key in record_fields}
    try:
        return ExperienceRecord.model_validate_json(json.dumps(normalized))
    except ValueError as error:
        raise ExperienceRepositoryError("stored cognitive experience is invalid") from error
