"""Persistent Hermes advice for run submission and bounded outcome learning."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent_hub.cognitive.context_router import CognitiveContextRouteResult, route_cognitive_context
from agent_hub.cognitive.repository import (
    ExperienceRepositoryError,
    PersistentCognitiveRecordRepository,
)
from agent_hub.cognitive.types import (
    BeliefRecord,
    RelationshipStateRecord,
    SkillCandidateRecord,
    WorldStateRecord,
)
from agent_hub.db.models import AdminResourceRow
from agent_hub.domain.runs import RunStatus, TaskMode
from agent_hub.hermes.runtime_observation import is_runtime_observation_lesson
from agent_hub.runs.service import (
    HermesMemoryInjection,
    HermesRunAdvice,
    HermesRunOutcome,
    HermesSkippedMemory,
)

_INJECTABLE_MEMORY_TYPES = {
    "user_preference",
    "project_fact",
    "ui_rule",
    "error_handling",
    "scheduling_rule",
}
_LOW_QUALITY_PHRASES = (
    "这个任务成功了",
    "任务成功了",
    "出错了",
    "失败了",
    "worked",
    "failed",
)


class PersistentHermesRunAdvisor:
    """Use stored Hermes lessons as bounded runtime advice.

    Hermes never stores the raw user request here. Outcome learning records only
    mode, workflow, selected role IDs, and terminal status, so it can improve
    future routing without memorizing secrets from prompts.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._cognitive_repository = PersistentCognitiveRecordRepository(session_factory)

    async def advise(
        self,
        *,
        tenant_id: UUID,
        actor_id: UUID,
        message: str,
        mode: TaskMode,
        agent_ids: tuple[str, ...],
        workflow_id: str | None,
    ) -> HermesRunAdvice | None:
        if not await self._enabled(tenant_id):
            return None
        policy = await self._main_agent_hermes_policy(tenant_id)
        if policy in {"off", "observe"}:
            return None
        lessons = [
            lesson
            for lesson in await self._lessons(tenant_id, actor_id)
            if _lesson_is_conversation_advice(lesson) and _lesson_visible_to_actor(lesson, actor_id)
        ]
        lowered = message.casefold()
        confirmed = [lesson for lesson in lessons if _lesson_is_confirmed(lesson)]

        injected: list[tuple[float, int, HermesMemoryInjection, dict[str, object]]] = []
        skipped: list[HermesSkippedMemory] = []
        conflict_skipped = False
        for lesson in confirmed:
            if not _lesson_matches(lowered, lesson, workflow_id, mode=mode, agent_ids=agent_ids):
                continue
            score = _lesson_relevance_score(
                lowered,
                lesson,
                mode=mode,
                agent_ids=agent_ids,
                workflow_id=workflow_id,
            )
            summary = _lesson_user_summary(lesson)
            lesson_id = _lesson_id(lesson)
            noise_reason = _lesson_noise_reason(lesson)
            if noise_reason is not None:
                skipped.append(
                    HermesSkippedMemory(
                        id=lesson_id,
                        summary=summary,
                        reason=noise_reason,
                        score=score,
                    )
                )
                continue
            if _lesson_conflicts_with_request(lowered, lesson):
                conflict_skipped = True
                skipped.append(
                    HermesSkippedMemory(
                        id=lesson_id,
                        summary=summary,
                        reason="当前用户指令覆盖这条记忆",
                        score=score,
                    )
                )
                continue
            if score >= 0.65:
                injected.append(
                    (
                        score,
                        _lesson_weight(lesson),
                        HermesMemoryInjection(
                            id=lesson_id,
                            summary=summary,
                            memory_type=_lesson_memory_type(lesson),
                            target=_lesson_target(lesson),
                            score=round(score, 2),
                            reason=_lesson_injection_reason(lesson, score),
                        ),
                        lesson,
                    )
                )
            elif score >= 0.45:
                skipped.append(
                    HermesSkippedMemory(
                        id=lesson_id,
                        summary=summary,
                        reason="当前任务相关性不足",
                        score=round(score, 2),
                    )
                )

        injected.sort(key=lambda item: (item[0], item[1]), reverse=True)
        cognitive_context = await self._cognitive_context(
            tenant_id=tenant_id,
            actor_id=actor_id,
            message=message,
            mode=mode,
            agent_ids=agent_ids,
        )
        selected_injections = [item[2] for item in injected[:3]]
        selected_ids = {item.id for item in selected_injections}
        for item in cognitive_context.selected:
            if len(selected_injections) >= 3:
                break
            if item.id in selected_ids:
                continue
            selected_injections.append(
                HermesMemoryInjection(
                    id=item.id,
                    summary=item.summary,
                    memory_type=item.source,
                    target=item.target,
                    score=item.score,
                    reason=item.reason,
                )
            )
            selected_ids.add(item.id)
        selected_skipped = tuple(
            [
                *skipped,
                *(
                    HermesSkippedMemory(
                        id=item.id,
                        summary=item.summary,
                        reason=item.reason,
                        score=item.score,
                    )
                    for item in cognitive_context.skipped
                ),
            ][:5]
        )
        if not selected_injections and not conflict_skipped:
            return None
        best = injected[0][3] if injected else confirmed[0] if confirmed else None
        recommended_mode = _recommended_mode(best, lowered) if best is not None else _safe_mode(mode)
        confidence = max(
            [item[0] for item in injected[:1]]
            + [item.score for item in cognitive_context.selected[:1]]
            + [0.5]
        )
        reasons = []
        if best is not None:
            reasons.append(f"Hermes matched stored lesson {_lesson_id(best)}")
        if cognitive_context.selected:
            reasons.append(f"Hermes matched cognitive context {cognitive_context.selected[0].id}")
        return HermesRunAdvice(
            recommended_mode=recommended_mode,
            confidence=confidence,
            reasons=tuple(reasons),
            recommended_skills=(),
            requires_approval=policy in {"suggest", "confirm_before_apply"} or confidence < 0.75,
            injected_memories=tuple(selected_injections),
            skipped_memories=selected_skipped,
        )

    async def record_outcome(self, outcome: HermesRunOutcome) -> None:
        if outcome.status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return
        if not await self._enabled(outcome.tenant_id):
            return
        lesson_id = f"hermes_run_{uuid4().hex}"
        payload = _outcome_learning_payload(outcome, lesson_id=lesson_id)
        await self._upsert(outcome.tenant_id, lesson_id, payload)
        cognitive_payload = _cognitive_candidate_payload_from_outcome(outcome)
        if cognitive_payload is not None:
            await self._upsert(
                outcome.tenant_id,
                str(cognitive_payload["resource_id"]),
                cognitive_payload,
            )

    async def _enabled(self, tenant_id: UUID) -> bool:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == tenant_id)
                    .where(AdminResourceRow.kind == "setting")
                    .where(AdminResourceRow.resource_id == "system")
                )
            ).scalar_one_or_none()
        if row is None:
            return True
        return dict(row.payload).get("hermes_enabled", True) is True

    async def _main_agent_hermes_policy(self, tenant_id: UUID) -> str:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == tenant_id)
                    .where(AdminResourceRow.kind == "main_agent")
                    .where(AdminResourceRow.resource_id == "default")
                )
            ).scalar_one_or_none()
        if row is None:
            return "observe"
        policy = dict(row.payload).get("hermes_policy")
        if policy in {"off", "observe", "suggest", "confirm_before_apply"}:
            return str(policy)
        return "observe"

    async def _lessons(self, tenant_id: UUID, actor_id: UUID) -> list[dict[str, object]]:
        async with self._session_factory() as session:
            conversation_rows = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == tenant_id)
                    .where(AdminResourceRow.kind == "hermes")
                    .where(AdminResourceRow.resource_id.not_like("cognitive_experience:%"))
                    .where(
                        or_(
                            AdminResourceRow.payload["category"].as_string() == "conversation",
                            AdminResourceRow.payload["category"].as_string().is_(None),
                        )
                    )
                    .where(
                        or_(
                            AdminResourceRow.payload["memory_scope"].as_string() == "root",
                            AdminResourceRow.payload["user_id"].as_string() == str(actor_id),
                        )
                    )
                    .where(AdminResourceRow.payload["confirmed_at"].as_string().is_not(None))
                    .order_by(AdminResourceRow.created_at.desc())
                    .limit(200)
                )
            ).scalars()
            cognitive_rows = (
                await session.execute(
                    select(AdminResourceRow)
                    .where(AdminResourceRow.tenant_id == tenant_id)
                    .where(AdminResourceRow.kind == "hermes")
                    .where(AdminResourceRow.resource_id.like("cognitive_experience:%"))
                    .where(
                        or_(
                            AdminResourceRow.payload["memory_scope"].as_string() == "root",
                            AdminResourceRow.payload["user_id"].as_string() == str(actor_id),
                        )
                    )
                    .where(
                        AdminResourceRow.payload["status"].as_string().in_(("confirmed", "active"))
                    )
                    .where(AdminResourceRow.payload["active_for_runtime"].as_boolean().is_(True))
                    .order_by(AdminResourceRow.created_at.desc())
                    .limit(200)
                )
            ).scalars()
            return [dict(row.payload) for row in conversation_rows] + [
                dict(row.payload) for row in cognitive_rows
            ]

    async def _upsert(self, tenant_id: UUID, resource_id: str, payload: dict[str, object]) -> None:
        statement = (
            insert(AdminResourceRow)
            .values(
                id=uuid4(),
                tenant_id=tenant_id,
                kind="hermes",
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

    async def _cognitive_context(
        self,
        *,
        tenant_id: UUID,
        actor_id: UUID,
        message: str,
        mode: TaskMode,
        agent_ids: tuple[str, ...],
    ) -> CognitiveContextRouteResult:
        try:
            beliefs = await self._cognitive_repository.list_for_user(
                BeliefRecord,
                tenant_id=tenant_id,
                user_id=actor_id,
            )
            relationships = await self._cognitive_repository.list_for_user(
                RelationshipStateRecord,
                tenant_id=tenant_id,
                user_id=actor_id,
            )
            worlds = await self._cognitive_repository.list_for_user(
                WorldStateRecord,
                tenant_id=tenant_id,
                user_id=actor_id,
            )
            skills = await self._cognitive_repository.list_for_user(
                SkillCandidateRecord,
                tenant_id=tenant_id,
                user_id=actor_id,
            )
        except ExperienceRepositoryError:
            return route_cognitive_context(
                request=message,
                mode=mode.value,
                agent_ids=agent_ids,
                limit=3,
            )
        return route_cognitive_context(
            request=message,
            mode=mode.value,
            agent_ids=agent_ids,
            beliefs=beliefs,
            relationship_states=relationships,
            world_states=worlds,
            skill_candidates=skills,
            limit=3,
        )


def _lesson_is_conversation_advice(lesson: dict[str, object]) -> bool:
    category = lesson.get("category")
    return category in {None, "conversation"} and not _lesson_is_runtime_observation(lesson)


def _lesson_is_runtime_observation(lesson: dict[str, object]) -> bool:
    return is_runtime_observation_lesson(lesson.get("lesson"))


def _outcome_learning_payload(
    outcome: HermesRunOutcome,
    *,
    lesson_id: str,
) -> dict[str, object]:
    mode = "unknown" if outcome.mode is None else outcome.mode.value
    workflow = outcome.workflow_id or "no-workflow"
    conversation_id = outcome.conversation_id or "unknown-conversation"
    status = outcome.status.value
    scheduler_notices = _safe_scheduler_notices(outcome.scheduler_notices)
    if scheduler_notices:
        lesson = _scheduler_notice_lesson(
            status=status,
            mode=mode,
            workflow=workflow,
            notices=scheduler_notices,
        )
        tags = _unique_tags(
            [
                status,
                mode,
                workflow,
                *outcome.agent_ids[:8],
                *_scheduler_notice_tags(scheduler_notices),
            ]
        )
        weight = min(10, 5 + len(scheduler_notices))
        summary = (
            f"调度观察：Hermes learned from conversation {conversation_id}: "
            f"{lesson} Tags: {', '.join(tags) or 'none'}. Weight: {weight}."
        )
        user_summary = f"本次调度观察提醒：{_runtime_lesson_summary(lesson)}"
        category = "scheduler"
        memory_type = "scheduler_observation"
        target = "scheduler"
        confidence = 0.65
        noise_risk = 0.25
    else:
        lesson = f"Run {status} with mode={mode}, workflow={workflow}."
        tags = _unique_tags([status, mode, workflow, *outcome.agent_ids[:8]])
        weight = 4 if outcome.status is RunStatus.COMPLETED else 2
        summary = (
            f"Hermes learned from conversation {conversation_id}: "
            f"{lesson} Tags: {', '.join(tags) or 'none'}. Weight: {weight}."
        )
        label = "成功经验" if outcome.status is RunStatus.COMPLETED else "失败教训"
        user_summary = f"本次运行观察记录了一个{label}：{_runtime_lesson_summary(lesson)}"
        category = "scheduler"
        memory_type = "runtime_observation"
        target = "scheduler"
        confidence = 0.6 if outcome.status is RunStatus.COMPLETED else 0.45
        noise_risk = 0.35 if outcome.status is RunStatus.COMPLETED else 0.55
    return {
        "id": lesson_id,
        "user_id": str(outcome.actor_id),
        "memory_scope": "user",
        "category": category,
        "outcome": "success" if outcome.status is RunStatus.COMPLETED else "failure",
        "lesson": lesson,
        "summary": summary,
        "user_summary": user_summary,
        "tags": tags,
        "weight": weight,
        "source_mode": mode,
        "applies_to_modes": [] if mode == "unknown" else [mode],
        "memory_type": memory_type,
        "target": target,
        "confidence": confidence,
        "noise_risk": noise_risk,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": str(outcome.run_id),
        "conversation_id": outcome.conversation_id,
        "confirmed_at": None,
    }


def _cognitive_candidate_payload_from_outcome(
    outcome: HermesRunOutcome,
) -> dict[str, object] | None:
    if outcome.status not in {RunStatus.FAILED, RunStatus.CANCELLED}:
        return None
    scheduler_notices = _safe_scheduler_notices(outcome.scheduler_notices)
    if not scheduler_notices:
        return None
    experience_id = str(uuid4())
    mode = "unknown" if outcome.mode is None else outcome.mode.value
    workflow = outcome.workflow_id or "no-workflow"
    triggers = _unique_tags(
        [
            notice.get("trigger", "")
            for notice in scheduler_notices
            if notice.get("trigger")
        ]
    )
    actions = _unique_tags(
        [
            notice.get("action", "")
            for notice in scheduler_notices
            if notice.get("action")
        ]
    )
    actor_tags = _unique_tags([actor for actor in outcome.agent_ids[:8] if actor])
    tags = _unique_tags([outcome.status.value, mode, workflow, *triggers, *actions, *actor_tags])
    summary_trigger = triggers[0] if triggers else "runtime_failure"
    summary = _compact_sentence(
        f"{workflow} 工作流在 {mode} 模式出现 {summary_trigger}，需要先按失败信号调整执行策略。",
        limit=220,
    )
    lesson = _compact_sentence(
        _scheduler_notice_lesson(
            status=outcome.status.value,
            mode=mode,
            workflow=workflow,
            notices=scheduler_notices,
        ),
        limit=900,
    )
    strategy = _compact_sentence(
        "后续遇到相似失败信号时，先保留已有输出并重试；若仍失败，再压缩输入、切换更快或更稳模型、"
        "拆分大步骤，最后才跳过非关键审查步骤。",
        limit=360,
    )
    now = datetime.now(UTC).isoformat()
    return {
        "id": experience_id,
        "user_id": str(outcome.actor_id),
        "memory_scope": "user",
        "kind": "error_handling",
        "status": "candidate",
        "summary": summary,
        "lesson": lesson,
        "strategy": strategy,
        "confidence": 0.58,
        "evidence": [
            {
                "source_type": "run",
                "source_id": str(outcome.run_id),
                "note": _compact_sentence(
                    f"{outcome.status.value} outcome with scheduler notices: {', '.join(triggers or actions)}",
                    limit=512,
                ),
            }
        ],
        "contradictions": [],
        "source_run_ids": [str(outcome.run_id)],
        "source_memory_ids": [],
        "tags": tags,
        "applies_to_modes": [] if mode == "unknown" else [mode],
        "applies_to_agents": actor_tags,
        "use_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "active_for_runtime": False,
        "last_used_at": None,
        "last_verified_at": None,
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "storage_kind": "hermes",
        "resource_id": f"cognitive_experience:{experience_id}",
    }


def _lesson_matches(
    lowered_message: str,
    lesson: dict[str, object],
    workflow_id: str | None,
    *,
    mode: TaskMode,
    agent_ids: tuple[str, ...],
) -> bool:
    if _is_cognitive_experience_payload(lesson):
        return _cognitive_lesson_matches(
            lowered_message,
            lesson,
            workflow_id,
            mode=mode,
            agent_ids=agent_ids,
        )
    tags = lesson.get("tags")
    if isinstance(tags, list) and any(
        isinstance(tag, str) and _meaningful_tag_matches(tag, lowered_message) for tag in tags
    ):
        return True
    if workflow_id and isinstance(tags, list) and workflow_id in tags:
        return True
    text = lesson.get("lesson")
    if not isinstance(text, str):
        return False
    return any(word and len(word) >= 4 and word in lowered_message for word in text.lower().split())


def _cognitive_lesson_matches(
    lowered_message: str,
    lesson: dict[str, object],
    workflow_id: str | None,
    *,
    mode: TaskMode,
    agent_ids: tuple[str, ...],
) -> bool:
    if not _lesson_has_evidence(lesson):
        return False
    applies = _string_list(lesson.get("applies_to_modes"))
    if applies and mode.value not in applies:
        return False
    assigned_agents = set(agent_ids)
    applicable_agents = set(_string_list(lesson.get("applies_to_agents")))
    if assigned_agents and applicable_agents and applicable_agents.isdisjoint(assigned_agents):
        return False
    tags = _string_list(lesson.get("tags"))
    if workflow_id and workflow_id in tags:
        return True
    return any(_meaningful_tag_matches(tag, lowered_message) for tag in tags)


def _lesson_has_evidence(lesson: dict[str, object]) -> bool:
    evidence = lesson.get("evidence")
    return isinstance(evidence, list | tuple) and any(isinstance(item, dict) for item in evidence)


def _meaningful_tag_matches(tag: str, lowered_message: str) -> bool:
    normalized = tag.casefold().strip()
    if len(normalized) < 2:
        return False
    if normalized.isascii() and len(normalized) < 3:
        return False
    return normalized in lowered_message


def _lesson_id(lesson: dict[str, object]) -> str:
    resource_id = lesson.get("resource_id")
    if _is_cognitive_experience_payload(lesson) and isinstance(resource_id, str) and resource_id:
        return resource_id
    value = lesson.get("id")
    return value if isinstance(value, str) and value else "unknown"


def _lesson_user_summary(lesson: dict[str, object]) -> str:
    for key in ("user_summary", "summary", "lesson"):
        value = lesson.get(key)
        if isinstance(value, str) and value.strip():
            return _compact_sentence(value, limit=220)
    return "Hermes+ 记忆"


def _lesson_memory_type(lesson: dict[str, object]) -> str:
    if _is_cognitive_experience_payload(lesson):
        value = lesson.get("kind")
        return value if isinstance(value, str) and value else "experience"
    value = lesson.get("memory_type")
    return value if isinstance(value, str) and value else "conversation_advice"


def _lesson_target(lesson: dict[str, object]) -> str:
    if _is_cognitive_experience_payload(lesson):
        agents = _string_list(lesson.get("applies_to_agents"))
        if agents:
            return agents[0]
    value = lesson.get("target")
    return value if isinstance(value, str) and value else "main_agent"


def _float_or_default(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _lesson_noise_reason(lesson: dict[str, object]) -> str | None:
    confidence = _float_or_default(lesson.get("confidence"), 0.7)
    noise = _float_or_default(lesson.get("noise_risk"), 0.0)
    text = f"{lesson.get('lesson', '')} {lesson.get('user_summary', '')}"
    if confidence < 0.45:
        return "置信度不足"
    if noise >= 0.7:
        return "噪音风险过高"
    if any(phrase in text.casefold() for phrase in _LOW_QUALITY_PHRASES):
        return "记忆过于泛化"
    if _lesson_memory_type(lesson) in {"temporary_state", "single_run_state"}:
        return "临时运行状态不参与注入"
    return None


def _lesson_relevance_score(
    lowered_message: str,
    lesson: dict[str, object],
    *,
    mode: TaskMode,
    agent_ids: tuple[str, ...],
    workflow_id: str | None,
) -> float:
    score = 0.0
    tags = _string_list(lesson.get("tags"))
    if _lesson_matches(lowered_message, lesson, workflow_id, mode=mode, agent_ids=agent_ids):
        score += 0.35
    if any(_meaningful_tag_matches(tag, lowered_message) for tag in tags):
        score += 0.2
    if workflow_id and workflow_id in tags:
        score += 0.1
    agent_id_set = set(agent_ids)
    if any(tag in agent_id_set for tag in tags):
        score += 0.1
    if _lesson_target(lesson) in agent_id_set:
        score += 0.1
    applies = _string_list(lesson.get("applies_to_modes"))
    if mode.value in applies:
        score += 0.12
    elif _lesson_memory_type(lesson) in _INJECTABLE_MEMORY_TYPES:
        score += 0.08
    score += min(0.12, _lesson_weight(lesson) / 100)
    score += min(0.08, _float_or_default(lesson.get("confidence"), 0.7) / 10)
    score -= min(0.2, _float_or_default(lesson.get("noise_risk"), 0.0) / 2)
    return max(0.0, min(1.0, score))


def _lesson_conflicts_with_request(lowered_message: str, lesson: dict[str, object]) -> bool:
    tags = " ".join(_string_list(lesson.get("tags")))
    text = f"{lesson.get('lesson', '')} {tags}".casefold()
    direct_requested = any(token in lowered_message for token in ("直连", "direct", "不要混合", "不混合"))
    hybrid_suggested = any(token in text for token in ("hybrid", "混合"))
    return direct_requested and hybrid_suggested


def _lesson_injection_reason(lesson: dict[str, object], score: float) -> str:
    raw = lesson.get("injection_reason")
    if isinstance(raw, str) and raw.strip():
        return _compact_sentence(raw, limit=160)
    return f"相关性评分 {score:.2f}，命中 Hermes+ 已确认记忆"


def _lesson_is_confirmed(lesson: dict[str, object]) -> bool:
    if _is_cognitive_experience_payload(lesson):
        return lesson.get("status") in {"confirmed", "active"} and lesson.get("active_for_runtime") is True
    confirmed_at = lesson.get("confirmed_at")
    return isinstance(confirmed_at, str) and bool(confirmed_at.strip())


def _lesson_visible_to_actor(lesson: dict[str, object], actor_id: UUID) -> bool:
    memory_scope = lesson.get("memory_scope")
    if memory_scope == "root":
        return True
    owner_user_id = lesson.get("user_id")
    return memory_scope == "user" and owner_user_id == str(actor_id)


def _is_cognitive_experience_payload(lesson: dict[str, object]) -> bool:
    resource_id = lesson.get("resource_id")
    return isinstance(resource_id, str) and resource_id.startswith("cognitive_experience:")


def _lesson_weight(lesson: dict[str, object]) -> int:
    value = lesson.get("weight", 1)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 1


def _recommended_mode(lesson: dict[str, object], lowered_message: str) -> TaskMode:
    raw_tags = lesson.get("tags")
    tags = [tag.lower() for tag in raw_tags if isinstance(tag, str)] if isinstance(raw_tags, list) else []
    haystack = " ".join([lowered_message, str(lesson.get("lesson", "")).lower(), *tags])
    if any(token in haystack for token in ("hybrid", "混合")):
        return TaskMode.HYBRID
    if any(token in haystack for token in ("discuss", "discussion", "group_chat", "debate", "review", "讨论")):
        return TaskMode.DISCUSS
    if any(token in haystack for token in ("direct", "直接")):
        return TaskMode.DIRECT
    return TaskMode.DISPATCH


def _safe_mode(mode: TaskMode) -> TaskMode:
    if mode is TaskMode.AUTO:
        return TaskMode.DISPATCH
    return mode


def _safe_scheduler_notices(
    notices: tuple[dict[str, object], ...],
) -> tuple[dict[str, str], ...]:
    safe: list[dict[str, str]] = []
    allowed_keys = ("trigger", "action", "severity", "source_kind", "actor")
    for notice in notices[:4]:
        item: dict[str, str] = {}
        for key in allowed_keys:
            value = notice.get(key)
            if isinstance(value, str) and value.strip():
                item[key] = value.strip()[:96]
        if item:
            safe.append(item)
    return tuple(safe)


def _scheduler_notice_lesson(
    *,
    status: str,
    mode: str,
    workflow: str,
    notices: tuple[dict[str, str], ...],
) -> str:
    notice_text = "; ".join(
        ", ".join(f"{key}={value}" for key, value in notice.items()) for notice in notices
    )
    return f"Run {status} with mode={mode}, workflow={workflow}. Scheduler notices: {notice_text}."


def _scheduler_notice_tags(notices: tuple[dict[str, str], ...]) -> list[str]:
    tags: list[str] = []
    for notice in notices:
        for key in ("trigger", "action", "severity", "source_kind", "actor"):
            value = notice.get(key)
            if value:
                tags.append(value)
    return tags


def _unique_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        cleaned = tag.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= 16:
            break
    return result


def _compact_sentence(value: str, *, limit: int = 96) -> str:
    cleaned = " ".join(value.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}..."


def _runtime_lesson_summary(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    prefix = "Run "
    if not cleaned.startswith(prefix):
        return _compact_sentence(cleaned)
    head = cleaned.split(". Scheduler notices:", 1)[0]
    parts = head.removeprefix(prefix).split(" with mode=", 1)
    if len(parts) != 2:
        return _compact_sentence(cleaned)
    status, rest = parts
    mode, _, workflow = rest.partition(", workflow=")
    workflow = workflow.removesuffix(".")
    status_label = {
        "completed": "成功完成",
        "failed": "运行失败",
        "cancelled": "被取消",
    }.get(status, status)
    summary = f"{workflow} 工作流以 {mode} 模式{status_label}。"
    if ". Scheduler notices:" in cleaned:
        summary = f"{summary} 已记录调度告警。"
    return summary
