from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, overload
from uuid import UUID, uuid4

from agent_hub.cognitive.types import (
    BeliefRecord,
    CognitiveEvidence,
    CognitiveMemoryScope,
    ExperienceKind,
    ExperienceRecord,
    ExperienceStatus,
    SkillCandidateRecord,
)


class ExperienceRepository(Protocol):
    async def upsert(self, record: ExperienceRecord) -> ExperienceRecord: ...

    async def get(self, record_id: UUID) -> ExperienceRecord | None: ...

    async def list_for_user(self, tenant_id: UUID, user_id: UUID) -> tuple[ExperienceRecord, ...]: ...

    async def delete(self, record_id: UUID, *, tenant_id: UUID, user_id: UUID) -> bool: ...


class ExperienceNotFound(LookupError):
    pass


class CognitiveRecordNotFound(LookupError):
    pass


class CognitiveStateRepository(Protocol):
    @overload
    async def upsert(self, record: BeliefRecord) -> BeliefRecord: ...

    @overload
    async def upsert(self, record: SkillCandidateRecord) -> SkillCandidateRecord: ...

    @overload
    async def get(
        self, record_type: type[BeliefRecord], record_id: str | UUID
    ) -> BeliefRecord | None: ...

    @overload
    async def get(
        self, record_type: type[SkillCandidateRecord], record_id: str | UUID
    ) -> SkillCandidateRecord | None: ...

    async def list_for_user(
        self,
        record_type: type[BeliefRecord],
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> tuple[BeliefRecord, ...]: ...


def _bounded_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


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


class CognitiveStateService:
    def __init__(
        self,
        repository: CognitiveStateRepository,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._now = now or (lambda: datetime.now(UTC))

    async def record_belief_observation(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        subject: str,
        claim: str,
        evidence: CognitiveEvidence,
        supported: bool,
        memory_scope: CognitiveMemoryScope = CognitiveMemoryScope.USER,
    ) -> BeliefRecord:
        timestamp = self._now()
        existing = await self._find_owned_belief(
            tenant_id=tenant_id,
            user_id=user_id,
            subject=subject,
            claim=claim,
            memory_scope=memory_scope,
        )
        if existing is None:
            record = BeliefRecord(
                id=uuid4(),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_scope=memory_scope,
                subject=subject,
                claim=claim,
                confidence=0.58 if supported else 0.42,
                evidence=(evidence,) if supported else (),
                contradictions=() if supported else (evidence,),
                status="active" if supported else "contested",
                last_verified_at=timestamp,
                created_at=timestamp,
                updated_at=timestamp,
            )
            return await self._repository.upsert(record)

        confidence = _bounded_confidence(existing.confidence + (0.07 if supported else -0.12))
        updated = existing.model_copy(
            update={
                "confidence": confidence,
                "evidence": (*existing.evidence, evidence) if supported else existing.evidence,
                "contradictions": existing.contradictions
                if supported
                else (*existing.contradictions, evidence),
                "status": self._belief_status(confidence=confidence, supported=supported),
                "last_verified_at": timestamp,
                "version": existing.version + 1,
                "updated_at": timestamp,
            }
        )
        return await self._repository.upsert(updated)

    async def record_skill_use_outcome(
        self,
        skill_id: UUID,
        *,
        tenant_id: UUID,
        user_id: UUID,
        succeeded: bool,
        evidence: CognitiveEvidence,
    ) -> SkillCandidateRecord:
        skill = await self._repository.get(SkillCandidateRecord, skill_id)
        if skill is None:
            raise CognitiveRecordNotFound("skill candidate not found")
        if skill.tenant_id != tenant_id or skill.user_id != user_id:
            raise PermissionError("skill candidate is not visible to caller")

        timestamp = self._now()
        use_count = skill.use_count + 1
        success_count = skill.success_count + (1 if succeeded else 0)
        failure_count = skill.failure_count + (0 if succeeded else 1)
        confidence = _bounded_confidence(skill.confidence + (0.05 if succeeded else -0.08))
        updated = skill.model_copy(
            update={
                "confidence": confidence,
                "evidence": (*skill.evidence, evidence) if succeeded else skill.evidence,
                "contradictions": skill.contradictions
                if succeeded
                else (*skill.contradictions, evidence),
                "use_count": use_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "last_used_at": timestamp,
                "last_verified_at": timestamp,
                "version": skill.version + 1,
                "status": self._skill_status(
                    confidence=confidence,
                    success_count=success_count,
                    failure_count=failure_count,
                    current_status=skill.status,
                ),
                "updated_at": timestamp,
            }
        )
        return await self._repository.upsert(updated)

    async def _find_owned_belief(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        subject: str,
        claim: str,
        memory_scope: CognitiveMemoryScope,
    ) -> BeliefRecord | None:
        records = await self._repository.list_for_user(
            BeliefRecord,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return next(
            (
                record
                for record in records
                if record.user_id == user_id
                and record.memory_scope is memory_scope
                and record.subject == subject
                and record.claim == claim
            ),
            None,
        )

    @staticmethod
    def _belief_status(*, confidence: float, supported: bool) -> str:
        if confidence <= 0.24:
            return "deprecated"
        if not supported or confidence < 0.55:
            return "contested"
        return "active"

    @staticmethod
    def _skill_status(
        *,
        confidence: float,
        success_count: int,
        failure_count: int,
        current_status: str,
    ) -> str:
        if confidence <= 0.24:
            return "deprecated"
        if failure_count > success_count and confidence < 0.45:
            return "contested"
        if success_count >= 2 and confidence >= 0.72:
            return "active"
        return current_status
