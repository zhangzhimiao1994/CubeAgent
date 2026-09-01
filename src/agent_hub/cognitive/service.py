from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from agent_hub.cognitive.types import (
    CognitiveEvidence,
    CognitiveMemoryScope,
    ExperienceKind,
    ExperienceRecord,
    ExperienceStatus,
)


class ExperienceRepository(Protocol):
    async def upsert(self, record: ExperienceRecord) -> ExperienceRecord: ...

    async def get(self, record_id: UUID) -> ExperienceRecord | None: ...

    async def list_for_user(self, tenant_id: UUID, user_id: UUID) -> tuple[ExperienceRecord, ...]: ...

    async def delete(self, record_id: UUID, *, tenant_id: UUID, user_id: UUID) -> bool: ...


class ExperienceNotFound(LookupError):
    pass


class ExperienceService:
    def __init__(
        self,
        repository: ExperienceRepository,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._now = now or (lambda: datetime.now(UTC))

    async def create_candidate(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        kind: ExperienceKind,
        summary: str,
        lesson: str,
        strategy: str,
        evidence: tuple[CognitiveEvidence, ...],
        memory_scope: CognitiveMemoryScope = CognitiveMemoryScope.USER,
        confidence: float = 0.62,
        contradictions: tuple[CognitiveEvidence, ...] = (),
        source_run_ids: tuple[str, ...] = (),
        source_memory_ids: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        applies_to_modes: tuple[str, ...] = (),
        applies_to_agents: tuple[str, ...] = (),
    ) -> ExperienceRecord:
        timestamp = self._now()
        record = ExperienceRecord(
            id=uuid4(),
            tenant_id=tenant_id,
            user_id=user_id,
            memory_scope=memory_scope,
            kind=kind,
            status=ExperienceStatus.CANDIDATE,
            summary=summary,
            lesson=lesson,
            strategy=strategy,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            source_run_ids=source_run_ids,
            source_memory_ids=source_memory_ids,
            tags=tags,
            applies_to_modes=applies_to_modes,
            applies_to_agents=applies_to_agents,
            use_count=0,
            success_count=0,
            failure_count=0,
            last_used_at=None,
            last_verified_at=timestamp,
            version=1,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return await self._repository.upsert(record)

    async def confirm(self, record_id: UUID, *, tenant_id: UUID, user_id: UUID) -> ExperienceRecord:
        record = await self._owned(record_id, tenant_id, user_id)
        updated = record.model_copy(
            update={"status": ExperienceStatus.CONFIRMED, "updated_at": self._now()}
        )
        return await self._repository.upsert(updated)

    async def reject(self, record_id: UUID, *, tenant_id: UUID, user_id: UUID) -> ExperienceRecord:
        record = await self._owned(record_id, tenant_id, user_id)
        updated = record.model_copy(
            update={"status": ExperienceStatus.REJECTED, "updated_at": self._now()}
        )
        return await self._repository.upsert(updated)

    async def delete(self, record_id: UUID, *, tenant_id: UUID, user_id: UUID) -> bool:
        return await self._repository.delete(record_id, tenant_id=tenant_id, user_id=user_id)

    async def list_records(self, *, tenant_id: UUID, user_id: UUID) -> tuple[ExperienceRecord, ...]:
        return await self._repository.list_for_user(tenant_id, user_id)

    async def record_use_outcome(
        self,
        record_id: UUID,
        *,
        tenant_id: UUID,
        user_id: UUID,
        succeeded: bool,
        evidence: CognitiveEvidence,
    ) -> ExperienceRecord:
        record = await self._owned(record_id, tenant_id, user_id)
        timestamp = self._now()
        use_count = record.use_count + 1
        success_count = record.success_count + (1 if succeeded else 0)
        failure_count = record.failure_count + (0 if succeeded else 1)
        confidence_delta = 0.05 if succeeded else -0.08
        updated = record.model_copy(
            update={
                "confidence": max(0.0, min(1.0, record.confidence + confidence_delta)),
                "evidence": (*record.evidence, evidence) if succeeded else record.evidence,
                "contradictions": record.contradictions
                if succeeded
                else (*record.contradictions, evidence),
                "use_count": use_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "last_used_at": timestamp,
                "last_verified_at": timestamp,
                "updated_at": timestamp,
            }
        )
        return await self._repository.upsert(updated)

    async def _owned(self, record_id: UUID, tenant_id: UUID, user_id: UUID) -> ExperienceRecord:
        record = await self._repository.get(record_id)
        if record is None:
            raise ExperienceNotFound("experience not found")
        if record.tenant_id != tenant_id or record.user_id != user_id:
            raise PermissionError("experience is not visible to caller")
        return record
