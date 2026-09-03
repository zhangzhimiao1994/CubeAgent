from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from agent_hub.cognitive.reflection import ReflectionEngine
from agent_hub.cognitive.service import CognitiveStateService, ExperienceService
from agent_hub.cognitive.types import (
    CognitiveEvidence,
    ExperienceKind,
    ExperienceRecord,
    OutcomeAssessmentRecord,
    OutcomeVerdict,
    ReflectionRecord,
)
from agent_hub.cognitive.verifier import OutcomeAssessment, OutcomeVerifier
from agent_hub.domain.runs import RunStatus, TaskMode

_LOGGER = logging.getLogger(__name__)


class RunEvidenceRepository(Protocol):
    async def events(self, tenant_id: UUID, run_id: UUID) -> tuple[dict[str, object], ...]: ...

    async def artifacts(self, tenant_id: UUID, run_id: UUID) -> tuple[dict[str, object], ...]: ...


class CognitiveLearningPipelineProtocol(Protocol):
    async def process_terminal_run(
        self,
        *,
        tenant_id: UUID,
        run_id: UUID,
        status: RunStatus,
        mode: TaskMode | None,
        routing_decision: dict[str, object],
        actor_id: UUID | None = None,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class OutcomeCriticResult:
    assessment: OutcomeAssessment
    record: OutcomeAssessmentRecord

    @property
    def verdict(self) -> OutcomeVerdict:
        return self.record.verdict


@dataclass(frozen=True, slots=True)
class CognitiveLearningResult:
    outcome: OutcomeCriticResult
    reflection: ReflectionRecord | None
    experience: ExperienceRecord | None


class OutcomeCritic:
    def __init__(
        self,
        cognitive_service: CognitiveStateService | None = None,
        verifier: OutcomeVerifier | None = None,
    ) -> None:
        self._cognitive_service = cognitive_service
        self._verifier = verifier or OutcomeVerifier()

    def assess(
        self,
        *,
        status: RunStatus,
        events: Sequence[Mapping[str, object]],
        artifacts: Sequence[Mapping[str, object]],
    ) -> tuple[OutcomeVerdict, float, str, tuple[str, ...], bool]:
        assessment = self._verifier.assess(
            terminal_status=status.value,
            events=events,
            artifacts=artifacts,
        )
        return (
            assessment.verdict,
            assessment.confidence,
            _outcome_note(
                verdict=assessment.verdict,
                fallback=assessment.failure_or_gap_reason or _default_note(assessment.verdict),
                events=events,
            ),
            assessment.evidence,
            assessment.learnable,
        )

    async def assess_and_record(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        run_id: UUID,
        status: RunStatus,
        events: Sequence[Mapping[str, object]],
        artifacts: Sequence[Mapping[str, object]],
    ) -> OutcomeCriticResult:
        if self._cognitive_service is None:
            raise RuntimeError("cognitive service is required to record outcome assessments")
        assessment = self._verifier.assess(
            terminal_status=status.value,
            events=events,
            artifacts=artifacts,
        )
        record = await self._cognitive_service.record_outcome_assessment(
            tenant_id=tenant_id,
            user_id=user_id,
            source_run_id=str(run_id),
            target_type="run",
            target_id=str(run_id),
            verdict=assessment.verdict,
            note=_outcome_note(
                verdict=assessment.verdict,
                fallback=assessment.failure_or_gap_reason or _default_note(assessment.verdict),
                events=events,
            ),
            evidence=_evidence_from_notes(run_id=run_id, notes=assessment.evidence),
            confidence_delta=_confidence_delta(assessment.verdict, assessment.confidence),
        )
        return OutcomeCriticResult(assessment=assessment, record=record)


class CognitiveLearningPipeline:
    def __init__(
        self,
        *,
        experience_service: ExperienceService,
        cognitive_service: CognitiveStateService,
        run_repository: RunEvidenceRepository | None = None,
        outcome_critic: OutcomeCritic | None = None,
        reflection_engine: ReflectionEngine | None = None,
    ) -> None:
        self._experience_service = experience_service
        self._cognitive_service = cognitive_service
        self._run_repository = run_repository
        self._outcome_critic = outcome_critic or OutcomeCritic(cognitive_service)
        self._reflection_engine = reflection_engine

    async def process_terminal_run(
        self,
        *,
        tenant_id: UUID,
        run_id: UUID,
        status: RunStatus,
        mode: TaskMode | None,
        routing_decision: dict[str, object],
        actor_id: UUID | None = None,
        user_id: UUID | None = None,
        events: Sequence[dict[str, object]] | None = None,
        artifacts: Sequence[dict[str, object]] | None = None,
    ) -> CognitiveLearningResult | None:
        effective_user_id = actor_id or user_id
        if effective_user_id is None:
            return None
        terminal_events = (
            tuple(events)
            if events is not None
            else await self._events_from_repository(tenant_id=tenant_id, run_id=run_id)
        )
        terminal_artifacts = (
            tuple(artifacts)
            if artifacts is not None
            else await self._artifacts_from_repository(tenant_id=tenant_id, run_id=run_id)
        )
        outcome = await self._outcome_critic.assess_and_record(
            tenant_id=tenant_id,
            user_id=effective_user_id,
            run_id=run_id,
            status=status,
            events=terminal_events,
            artifacts=terminal_artifacts,
        )
        if not outcome.assessment.learnable:
            return CognitiveLearningResult(outcome=outcome, reflection=None, experience=None)
        reflection_engine = self._reflection_engine or ReflectionEngine(
            now=lambda: outcome.record.created_at
        )
        reflection = await self._cognitive_service.record_reflection(
            reflection_engine.reflect_from_outcome(outcome.record)
        )
        succeeded = outcome.record.verdict is OutcomeVerdict.SUCCESS
        usage_evidence = CognitiveEvidence(
            source_type="run_outcome",
            source_id=str(run_id),
            note=outcome.record.verdict.value,
        )
        await self._record_used_cognitive_items(
            tenant_id=tenant_id,
            user_id=effective_user_id,
            routing_decision=routing_decision,
            succeeded=succeeded,
            evidence=usage_evidence,
        )
        experience = None
        if self._should_create_candidate(verdict=outcome.record.verdict, routing_decision=routing_decision):
            experience = await self._experience_service.create_candidate(
                tenant_id=tenant_id,
                user_id=effective_user_id,
                kind=ExperienceKind.ERROR_HANDLING
                if outcome.record.verdict is not OutcomeVerdict.SUCCESS
                else ExperienceKind.WORKFLOW_STRATEGY,
                summary=_candidate_summary(outcome.record.verdict, mode),
                lesson=_candidate_lesson(outcome.record.verdict, outcome.record.note),
                strategy=_candidate_strategy(outcome.record.verdict),
                evidence=(usage_evidence,),
                source_run_ids=(str(run_id),),
                tags=("runtime_outcome", outcome.record.verdict.value),
                applies_to_modes=() if mode is None else (mode.value,),
                confidence=0.56 if outcome.record.verdict is not OutcomeVerdict.SUCCESS else 0.52,
            )
        return CognitiveLearningResult(outcome=outcome, reflection=reflection, experience=experience)

    async def _events_from_repository(self, *, tenant_id: UUID, run_id: UUID) -> tuple[dict[str, object], ...]:
        if self._run_repository is None:
            return ()
        return await self._run_repository.events(tenant_id, run_id)

    async def _artifacts_from_repository(self, *, tenant_id: UUID, run_id: UUID) -> tuple[dict[str, object], ...]:
        if self._run_repository is None:
            return ()
        return await self._run_repository.artifacts(tenant_id, run_id)

    async def _record_used_cognitive_items(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        routing_decision: dict[str, object],
        succeeded: bool,
        evidence: CognitiveEvidence,
    ) -> None:
        for experience_id in _experience_uuid_values(routing_decision):
            try:
                await self._experience_service.record_use_outcome(
                    experience_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    succeeded=succeeded,
                    evidence=evidence,
                )
            except Exception as error:  # noqa: BLE001 - learning must not affect runtime outcome.
                _LOGGER.info(
                    "cognitive_experience_outcome_update_skipped id=%s error_type=%s",
                    experience_id,
                    type(error).__name__,
                )
        for strategy_id in _uuid_values(routing_decision, ("used_strategy_ids", "selected_strategy_ids")):
            try:
                await self._cognitive_service.record_strategy_use_outcome(
                    strategy_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    succeeded=succeeded,
                    evidence=evidence,
                )
            except Exception as error:  # noqa: BLE001 - learning must not affect runtime outcome.
                _LOGGER.info(
                    "cognitive_strategy_outcome_update_skipped id=%s error_type=%s",
                    strategy_id,
                    type(error).__name__,
                )

    @staticmethod
    def _should_create_candidate(
        *,
        verdict: OutcomeVerdict,
        routing_decision: dict[str, object],
    ) -> bool:
        if verdict in {OutcomeVerdict.FAILURE, OutcomeVerdict.PARTIAL}:
            return True
        return bool(_experience_uuid_values(routing_decision)) or any(
            key in routing_decision
            for key in ("used_experience_ids", "injected_experience_ids", "used_strategy_ids")
        )


class CognitiveLearningTerminalHook:
    def __init__(self, pipeline: CognitiveLearningPipelineProtocol) -> None:
        self._pipeline = pipeline
        self._tasks: set[asyncio.Task[object]] = set()

    @property
    def pending_count(self) -> int:
        return len(self._tasks)

    async def __call__(
        self,
        *,
        tenant_id: UUID,
        actor_id: UUID | None,
        run_id: UUID,
        status: RunStatus,
        mode: TaskMode | None,
        routing_decision: dict[str, object],
    ) -> None:
        task = asyncio.create_task(
            self._run(
                tenant_id=tenant_id,
                actor_id=actor_id,
                run_id=run_id,
                status=status,
                mode=mode,
                routing_decision=routing_decision,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        if not self._tasks:
            return
        await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _run(
        self,
        *,
        tenant_id: UUID,
        actor_id: UUID | None,
        run_id: UUID,
        status: RunStatus,
        mode: TaskMode | None,
        routing_decision: dict[str, object],
    ) -> None:
        try:
            await self._pipeline.process_terminal_run(
                tenant_id=tenant_id,
                actor_id=actor_id,
                run_id=run_id,
                status=status,
                mode=mode,
                routing_decision=routing_decision,
            )
        except Exception as error:
            _LOGGER.exception(
                "cognitive_learning_pipeline_failed run_id=%s error_type=%s",
                run_id,
                type(error).__name__,
            )


def _evidence_from_notes(*, run_id: UUID, notes: tuple[str, ...]) -> tuple[CognitiveEvidence, ...]:
    if not notes:
        return (CognitiveEvidence(source_type="run", source_id=str(run_id), note="terminal_status"),)
    return tuple(
        CognitiveEvidence(source_type="run_event", source_id=f"{run_id}:{index}", note=note[:512])
        for index, note in enumerate(notes, start=1)
    )


def _uuid_values(payload: dict[str, object], keys: tuple[str, ...]) -> tuple[UUID, ...]:
    values: list[UUID] = []
    for key in keys:
        raw = payload.get(key)
        if not isinstance(raw, list | tuple):
            continue
        for item in raw:
            if not isinstance(item, str):
                continue
            try:
                values.append(UUID(item))
            except ValueError:
                continue
    return tuple(dict.fromkeys(values))


def _experience_uuid_values(payload: dict[str, object]) -> tuple[UUID, ...]:
    values = list(_uuid_values(payload, ("used_experience_ids", "injected_experience_ids")))
    hermes = payload.get("hermes")
    if isinstance(hermes, dict):
        raw_items = hermes.get("injected_memories")
        if isinstance(raw_items, list | tuple):
            for raw in raw_items:
                if not isinstance(raw, dict):
                    continue
                raw_id = raw.get("id")
                if not isinstance(raw_id, str):
                    continue
                prefix = "cognitive_experience:"
                if not raw_id.startswith(prefix):
                    continue
                try:
                    values.append(UUID(raw_id.removeprefix(prefix)))
                except ValueError:
                    continue
    return tuple(dict.fromkeys(values))


def _confidence_delta(verdict: OutcomeVerdict, confidence: float) -> float:
    if verdict is OutcomeVerdict.SUCCESS:
        return min(0.12, confidence * 0.1)
    if verdict is OutcomeVerdict.PARTIAL:
        return -0.03
    if verdict is OutcomeVerdict.FAILURE:
        return -0.1
    return 0.0


def _candidate_summary(verdict: OutcomeVerdict, mode: TaskMode | None) -> str:
    mode_label = "unknown" if mode is None else mode.value
    return f"{mode_label} 模式运行结果：{verdict.value}"


def _candidate_lesson(verdict: OutcomeVerdict, note: str) -> str:
    if verdict is OutcomeVerdict.SUCCESS:
        return f"本次运行有可见输出，可作为弱正向经验候选：{note}"
    if verdict is OutcomeVerdict.PARTIAL:
        return f"本次运行部分成功，需要保留可用路径并修正失败点：{note}"
    return f"本次运行失败，需要记录失败原因并避免重复：{note}"


def _outcome_note(
    *,
    verdict: OutcomeVerdict,
    fallback: str,
    events: Sequence[Mapping[str, object]],
) -> str:
    if verdict not in {OutcomeVerdict.FAILURE, OutcomeVerdict.PARTIAL}:
        return fallback[:512]
    for event in events:
        message = str(event.get("message") or event.get("reason") or "").strip()
        if message:
            return message[:512]
    return fallback[:512]


def _candidate_strategy(verdict: OutcomeVerdict) -> str:
    if verdict is OutcomeVerdict.SUCCESS:
        return "下次遇到相似任务时，可以优先复用本次被验证的路径，但仍需检查当前上下文是否一致。"
    if verdict is OutcomeVerdict.PARTIAL:
        return "下次遇到相似任务时，先压缩输入、拆分高风险步骤，并补充结果验证后再推进。"
    return "下次遇到相似任务时，先定位失败层级，压缩或拆分任务，必要时切换更可靠模型或策略后重试。"


def _default_note(verdict: OutcomeVerdict) -> str:
    if verdict is OutcomeVerdict.SUCCESS:
        return "completed with visible output"
    if verdict is OutcomeVerdict.PARTIAL:
        return "completed with recoverable failure evidence"
    if verdict is OutcomeVerdict.FAILURE:
        return "terminal runtime failure"
    return "insufficient evidence"
