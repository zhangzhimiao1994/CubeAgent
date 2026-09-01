from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, overload
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from agent_hub.cognitive.types import (
    BeliefRecord,
    CognitiveEvidence,
    CognitiveMemoryScope,
    ExperienceKind,
    ExperienceRecord,
    ExperienceStatus,
    RelationshipStateRecord,
    SkillCandidateRecord,
    StrategyRecord,
    StrategyStatus,
    WorldStateRecord,
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
    async def upsert(self, record: StrategyRecord) -> StrategyRecord: ...

    @overload
    async def upsert(self, record: RelationshipStateRecord) -> RelationshipStateRecord: ...

    @overload
    async def upsert(self, record: WorldStateRecord) -> WorldStateRecord: ...

    @overload
    async def get(
        self, record_type: type[BeliefRecord], record_id: str | UUID
    ) -> BeliefRecord | None: ...

    @overload
    async def get(
        self, record_type: type[SkillCandidateRecord], record_id: str | UUID
    ) -> SkillCandidateRecord | None: ...

    @overload
    async def get(
        self, record_type: type[StrategyRecord], record_id: str | UUID
    ) -> StrategyRecord | None: ...

    @overload
    async def get(
        self, record_type: type[RelationshipStateRecord], record_id: str | UUID
    ) -> RelationshipStateRecord | None: ...

    @overload
    async def get(
        self, record_type: type[WorldStateRecord], record_id: str | UUID
    ) -> WorldStateRecord | None: ...

    @overload
    async def list_for_user(
        self,
        record_type: type[BeliefRecord],
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> tuple[BeliefRecord, ...]: ...

    @overload
    async def list_for_user(
        self,
        record_type: type[SkillCandidateRecord],
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> tuple[SkillCandidateRecord, ...]: ...

    @overload
    async def list_for_user(
        self,
        record_type: type[StrategyRecord],
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> tuple[StrategyRecord, ...]: ...


def _bounded_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _stable_record_id(kind: str, tenant_id: UUID, user_id: UUID, memory_scope: CognitiveMemoryScope, scope: str) -> str:
    return f"{kind}:{uuid5(NAMESPACE_URL, f'agent-hub:{kind}:{tenant_id}:{user_id}:{memory_scope.value}:{scope}')}"


def _merge_unique(existing: tuple[str, ...], incoming: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for item in (*existing, *incoming):
        if item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def _remove_items(existing: tuple[str, ...], completed: tuple[str, ...]) -> tuple[str, ...]:
    completed_set = set(completed)
    return tuple(item for item in existing if item not in completed_set)


def _merge_evidence(
    existing: tuple[CognitiveEvidence, ...],
    incoming: tuple[CognitiveEvidence, ...],
) -> tuple[CognitiveEvidence, ...]:
    seen: set[tuple[str, str, str]] = set()
    result: list[CognitiveEvidence] = []
    for item in (*existing, *incoming):
        key = (item.source_type, item.source_id, item.note)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)


def _matches_request(value: str, request: str) -> bool:
    haystack = value.casefold()
    for token in _request_tokens(request):
        if token in haystack:
            return True
    return False


def _request_tokens(request: str) -> tuple[str, ...]:
    normalized = request.casefold()
    separators = ",.;:!?，。；：！？、()（）[]【】{}<>《》\n\t\r"
    for separator in separators:
        normalized = normalized.replace(separator, " ")
    return tuple(token for token in normalized.split(" ") if len(token) >= 2)


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

    async def create_strategy_candidate(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        name: str,
        context: str,
        strategy: str,
        rationale: str,
        evidence: tuple[CognitiveEvidence, ...],
        memory_scope: CognitiveMemoryScope = CognitiveMemoryScope.USER,
        confidence: float = 0.62,
        contradictions: tuple[CognitiveEvidence, ...] = (),
        tags: tuple[str, ...] = (),
        applies_to_modes: tuple[str, ...] = (),
        applies_to_agents: tuple[str, ...] = (),
    ) -> StrategyRecord:
        timestamp = self._now()
        record = StrategyRecord(
            id=uuid5(
                NAMESPACE_URL,
                f"agent-hub:strategy:{tenant_id}:{user_id}:{memory_scope.value}:{name}",
            ),
            tenant_id=tenant_id,
            user_id=user_id,
            memory_scope=memory_scope,
            name=name,
            context=context,
            strategy=strategy,
            rationale=rationale,
            status=StrategyStatus.CANDIDATE,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            tags=tags,
            applies_to_modes=applies_to_modes,
            applies_to_agents=applies_to_agents,
            last_verified_at=timestamp,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return await self._repository.upsert(record)

    async def confirm_strategy(
        self,
        strategy_id: UUID,
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> StrategyRecord:
        strategy = await self._owned_strategy(strategy_id, tenant_id=tenant_id, user_id=user_id)
        timestamp = self._now()
        updated = strategy.model_copy(
            update={
                "status": StrategyStatus.ACTIVE,
                "last_verified_at": timestamp,
                "updated_at": timestamp,
            }
        )
        return await self._repository.upsert(updated)

    async def reject_strategy(
        self,
        strategy_id: UUID,
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> StrategyRecord:
        strategy = await self._owned_strategy(strategy_id, tenant_id=tenant_id, user_id=user_id)
        updated = strategy.model_copy(
            update={"status": StrategyStatus.REJECTED, "updated_at": self._now()}
        )
        return await self._repository.upsert(updated)

    async def select_strategies_for_task(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        request: str,
        mode: str,
        limit: int = 3,
        include_candidates: bool = False,
    ) -> tuple[StrategyRecord, ...]:
        records = await self._repository.list_for_user(
            StrategyRecord,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        scored: list[tuple[float, StrategyRecord]] = []
        for strategy in records:
            if not self._strategy_selectable(
                strategy,
                mode=mode,
                include_candidates=include_candidates,
            ):
                continue
            score = self._strategy_score(strategy, request=request)
            if score <= 0:
                continue
            scored.append((score, strategy))
        scored.sort(key=lambda item: (-item[0], -item[1].confidence, item[1].created_at))
        return tuple(strategy for _score, strategy in scored[: max(0, limit)])

    async def record_strategy_use_outcome(
        self,
        strategy_id: UUID,
        *,
        tenant_id: UUID,
        user_id: UUID,
        succeeded: bool,
        evidence: CognitiveEvidence,
    ) -> StrategyRecord:
        strategy = await self._owned_strategy(strategy_id, tenant_id=tenant_id, user_id=user_id)
        timestamp = self._now()
        use_count = strategy.use_count + 1
        success_count = strategy.success_count + (1 if succeeded else 0)
        failure_count = strategy.failure_count + (0 if succeeded else 1)
        confidence = _bounded_confidence(strategy.confidence + (0.05 if succeeded else -0.08))
        updated = strategy.model_copy(
            update={
                "confidence": confidence,
                "evidence": (*strategy.evidence, evidence) if succeeded else strategy.evidence,
                "contradictions": strategy.contradictions
                if succeeded
                else (*strategy.contradictions, evidence),
                "use_count": use_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "last_used_at": timestamp,
                "last_verified_at": timestamp,
                "version": strategy.version + 1,
                "status": self._strategy_status(
                    confidence=confidence,
                    success_count=success_count,
                    failure_count=failure_count,
                    current_status=strategy.status,
                ),
                "updated_at": timestamp,
            }
        )
        return await self._repository.upsert(updated)

    async def update_relationship_state(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        preferred_language: str | None = None,
        preferred_confirmation_style: str | None = None,
        shared_milestones: tuple[str, ...] = (),
        recent_friction_points: tuple[str, ...] = (),
        familiarity_delta: float = 0.0,
        memory_scope: CognitiveMemoryScope = CognitiveMemoryScope.USER,
    ) -> RelationshipStateRecord:
        record_id = _stable_record_id("relationship", tenant_id, user_id, memory_scope, "default")
        timestamp = self._now()
        existing = await self._repository.get(RelationshipStateRecord, record_id)
        if existing is None:
            record = RelationshipStateRecord(
                id=record_id,
                tenant_id=tenant_id,
                user_id=user_id,
                memory_scope=memory_scope,
                familiarity=_bounded_confidence(0.2 + familiarity_delta),
                preferred_language=preferred_language or "zh-CN",
                preferred_confirmation_style=preferred_confirmation_style or "minimal",
                shared_milestones=_merge_unique((), shared_milestones),
                recent_friction_points=_merge_unique((), recent_friction_points),
                last_interaction_at=timestamp,
                created_at=timestamp,
                updated_at=timestamp,
            )
            return await self._repository.upsert(record)
        if existing.tenant_id != tenant_id or existing.user_id != user_id:
            raise PermissionError("relationship state is not visible to caller")
        updated = existing.model_copy(
            update={
                "familiarity": _bounded_confidence(existing.familiarity + familiarity_delta),
                "preferred_language": preferred_language or existing.preferred_language,
                "preferred_confirmation_style": preferred_confirmation_style
                or existing.preferred_confirmation_style,
                "shared_milestones": _merge_unique(
                    existing.shared_milestones,
                    shared_milestones,
                ),
                "recent_friction_points": _merge_unique(
                    existing.recent_friction_points,
                    recent_friction_points,
                ),
                "last_interaction_at": timestamp,
                "updated_at": timestamp,
            }
        )
        return await self._repository.upsert(updated)

    async def update_world_state(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        scope: str,
        facts: tuple[str, ...] = (),
        open_items: tuple[str, ...] = (),
        completed_items: tuple[str, ...] = (),
        future_events: tuple[str, ...] = (),
        evidence: CognitiveEvidence | None = None,
        memory_scope: CognitiveMemoryScope = CognitiveMemoryScope.USER,
    ) -> WorldStateRecord:
        record_id = _stable_record_id("world", tenant_id, user_id, memory_scope, scope)
        timestamp = self._now()
        existing = await self._repository.get(WorldStateRecord, record_id)
        if existing is None:
            record = WorldStateRecord(
                id=record_id,
                tenant_id=tenant_id,
                user_id=user_id,
                memory_scope=memory_scope,
                scope=scope,
                facts=_merge_unique((), facts),
                open_items=_merge_unique((), open_items),
                future_events=_merge_unique((), future_events),
                last_verified_at=timestamp if evidence is not None else None,
                evidence=(evidence,) if evidence is not None else (),
                created_at=timestamp,
                updated_at=timestamp,
            )
            return await self._repository.upsert(record)
        if existing.tenant_id != tenant_id or existing.user_id != user_id:
            raise PermissionError("world state is not visible to caller")
        remaining_open_items = _remove_items(existing.open_items, completed_items)
        updated = existing.model_copy(
            update={
                "facts": _merge_unique(existing.facts, facts),
                "open_items": _merge_unique(remaining_open_items, open_items),
                "future_events": _merge_unique(existing.future_events, future_events),
                "last_verified_at": timestamp if evidence is not None else existing.last_verified_at,
                "evidence": (*existing.evidence, evidence)
                if evidence is not None
                else existing.evidence,
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

    async def _owned_strategy(
        self,
        strategy_id: UUID,
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> StrategyRecord:
        strategy = await self._repository.get(StrategyRecord, strategy_id)
        if strategy is None:
            raise CognitiveRecordNotFound("strategy not found")
        if strategy.tenant_id != tenant_id or strategy.user_id != user_id:
            raise PermissionError("strategy is not visible to caller")
        return strategy

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

    @staticmethod
    def _strategy_selectable(
        strategy: StrategyRecord,
        *,
        mode: str,
        include_candidates: bool,
    ) -> bool:
        if strategy.status in {
            StrategyStatus.REJECTED,
            StrategyStatus.DEPRECATED,
            StrategyStatus.CONTESTED,
        }:
            return False
        if strategy.status is StrategyStatus.CANDIDATE and not include_candidates:
            return False
        if strategy.confidence < 0.6:
            return False
        return not strategy.applies_to_modes or mode in strategy.applies_to_modes

    @staticmethod
    def _strategy_score(strategy: StrategyRecord, *, request: str) -> float:
        searchable = (
            strategy.name,
            strategy.context,
            strategy.strategy,
            strategy.rationale,
            *strategy.tags,
        )
        matches = sum(1 for value in searchable if _matches_request(value, request))
        if matches == 0:
            return 0.0
        return float(matches) + strategy.confidence

    @staticmethod
    def _strategy_status(
        *,
        confidence: float,
        success_count: int,
        failure_count: int,
        current_status: StrategyStatus,
    ) -> StrategyStatus:
        if confidence <= 0.24:
            return StrategyStatus.DEPRECATED
        if failure_count > success_count and confidence < 0.45:
            return StrategyStatus.CONTESTED
        return current_status


class SkillPromotionNotReady(RuntimeError):
    pass


class SkillPromotionService:
    def __init__(
        self,
        experience_repository: ExperienceRepository,
        cognitive_repository: CognitiveStateRepository,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._experience_repository = experience_repository
        self._cognitive_repository = cognitive_repository
        self._now = now or (lambda: datetime.now(UTC))

    async def promote_from_experiences(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        experience_ids: tuple[UUID, ...],
        name: str,
        purpose: str,
        output_contract: str,
        required_inputs: tuple[str, ...] = (),
        memory_scope: CognitiveMemoryScope = CognitiveMemoryScope.USER,
        minimum_successes: int = 2,
        minimum_confidence: float = 0.65,
    ) -> SkillCandidateRecord:
        if not experience_ids:
            raise SkillPromotionNotReady("at least one experience is required")
        experiences = await self._owned_experiences(
            tenant_id=tenant_id,
            user_id=user_id,
            experience_ids=experience_ids,
        )
        self._validate_ready(
            experiences,
            minimum_successes=minimum_successes,
            minimum_confidence=minimum_confidence,
        )

        timestamp = self._now()
        promotion_evidence = tuple(
            CognitiveEvidence(
                source_type="experience",
                source_id=str(experience.id),
                note=experience.summary,
            )
            for experience in experiences
        )
        confidence = _bounded_confidence(
            sum(experience.confidence for experience in experiences) / len(experiences)
        )
        steps = _merge_unique((), tuple(experience.strategy for experience in experiences))
        existing = await self._find_existing_skill(
            tenant_id=tenant_id,
            user_id=user_id,
            name=name,
            memory_scope=memory_scope,
        )
        if existing is None:
            record = SkillCandidateRecord(
                id=uuid5(
                    NAMESPACE_URL,
                    f"agent-hub:skill:{tenant_id}:{user_id}:{memory_scope.value}:{name}",
                ),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_scope=memory_scope,
                name=name,
                purpose=purpose,
                steps=steps,
                required_inputs=required_inputs,
                output_contract=output_contract,
                confidence=confidence,
                evidence=promotion_evidence,
                status="candidate",
                created_at=timestamp,
                updated_at=timestamp,
            )
            return await self._cognitive_repository.upsert(record)

        updated = existing.model_copy(
            update={
                "purpose": purpose,
                "steps": _merge_unique(existing.steps, steps),
                "required_inputs": _merge_unique(existing.required_inputs, required_inputs),
                "output_contract": output_contract,
                "confidence": max(existing.confidence, confidence),
                "evidence": _merge_evidence(existing.evidence, promotion_evidence),
                "version": existing.version + 1,
                "last_verified_at": timestamp,
                "updated_at": timestamp,
            }
        )
        return await self._cognitive_repository.upsert(updated)

    async def _owned_experiences(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        experience_ids: tuple[UUID, ...],
    ) -> tuple[ExperienceRecord, ...]:
        records = await self._experience_repository.list_for_user(tenant_id, user_id)
        by_id = {record.id: record for record in records}
        selected: list[ExperienceRecord] = []
        for experience_id in experience_ids:
            record = by_id.get(experience_id)
            if record is None:
                raise SkillPromotionNotReady("experience is not available for promotion")
            if record.tenant_id != tenant_id or record.user_id != user_id:
                raise PermissionError("experience is not visible to caller")
            selected.append(record)
        return tuple(selected)

    @staticmethod
    def _validate_ready(
        experiences: tuple[ExperienceRecord, ...],
        *,
        minimum_successes: int,
        minimum_confidence: float,
    ) -> None:
        if any(not experience.active_for_runtime for experience in experiences):
            raise SkillPromotionNotReady("only confirmed or active experiences can be promoted")
        success_count = sum(experience.success_count for experience in experiences)
        failure_count = sum(experience.failure_count for experience in experiences)
        if success_count < minimum_successes:
            raise SkillPromotionNotReady("not enough successful uses to promote skill")
        if failure_count > success_count:
            raise SkillPromotionNotReady("experience failures outweigh successful uses")
        confidence = sum(experience.confidence for experience in experiences) / len(experiences)
        if confidence < minimum_confidence:
            raise SkillPromotionNotReady("experience confidence is too low for promotion")

    async def _find_existing_skill(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        name: str,
        memory_scope: CognitiveMemoryScope,
    ) -> SkillCandidateRecord | None:
        skills = await self._cognitive_repository.list_for_user(
            SkillCandidateRecord,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return next(
            (
                skill
                for skill in skills
                if skill.user_id == user_id
                and skill.memory_scope is memory_scope
                and skill.name == name
            ),
            None,
        )
