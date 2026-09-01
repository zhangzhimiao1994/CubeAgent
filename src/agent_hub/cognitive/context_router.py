from __future__ import annotations

import re
from dataclasses import dataclass

from agent_hub.cognitive.router import route_experiences
from agent_hub.cognitive.types import (
    BeliefRecord,
    ExperienceRecord,
    RelationshipStateRecord,
    SkillCandidateRecord,
    WorldStateRecord,
)
from agent_hub.memory.types import MemoryLayer, MemoryRecord


@dataclass(frozen=True, slots=True)
class RoutedCognitiveContext:
    id: str
    source: str
    summary: str
    target: str
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class SkippedCognitiveContext:
    id: str
    source: str
    summary: str
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class CognitiveContextRouteResult:
    selected: tuple[RoutedCognitiveContext, ...]
    skipped: tuple[SkippedCognitiveContext, ...]


def route_cognitive_context(
    *,
    request: str,
    mode: str,
    agent_ids: tuple[str, ...],
    memories: tuple[MemoryRecord, ...] = (),
    experiences: tuple[ExperienceRecord, ...] = (),
    beliefs: tuple[BeliefRecord, ...] = (),
    relationship_states: tuple[RelationshipStateRecord, ...] = (),
    world_states: tuple[WorldStateRecord, ...] = (),
    skill_candidates: tuple[SkillCandidateRecord, ...] = (),
    limit: int = 5,
) -> CognitiveContextRouteResult:
    if limit < 1 or limit > 20:
        raise ValueError("context route limit must be between 1 and 20")
    request_terms = _terms(request)
    selected: list[RoutedCognitiveContext] = []
    skipped: list[SkippedCognitiveContext] = []

    for memory in memories:
        routed, skip = _route_memory(memory, request_terms)
        _append_route(selected, skipped, routed, skip)

    experience_result = route_experiences(
        request=request,
        mode=mode,
        agent_ids=agent_ids,
        experiences=experiences,
    )
    selected.extend(
        RoutedCognitiveContext(
            id=item.id,
            source="experience",
            summary=item.summary,
            target=item.target,
            score=item.score,
            reason=item.reason,
        )
        for item in experience_result.selected
    )
    skipped.extend(
        SkippedCognitiveContext(
            id=item.id,
            source="experience",
            summary=item.summary,
            score=item.score,
            reason=item.reason,
        )
        for item in experience_result.skipped
    )

    for belief in beliefs:
        routed, skip = _route_belief(belief, request_terms)
        _append_route(selected, skipped, routed, skip)
    for relationship in relationship_states:
        routed, skip = _route_relationship(relationship, request_terms)
        _append_route(selected, skipped, routed, skip)
    for world in world_states:
        routed, skip = _route_world_state(world, request_terms)
        _append_route(selected, skipped, routed, skip)
    for skill in skill_candidates:
        routed, skip = _route_skill(skill, request_terms, mode, set(agent_ids))
        _append_route(selected, skipped, routed, skip)

    selected.sort(key=lambda item: item.score, reverse=True)
    skipped.sort(key=lambda item: item.score, reverse=True)
    return CognitiveContextRouteResult(
        selected=tuple(selected[:limit]),
        skipped=tuple(skipped[:10]),
    )


def _append_route(
    selected: list[RoutedCognitiveContext],
    skipped: list[SkippedCognitiveContext],
    routed: RoutedCognitiveContext | None,
    skip: SkippedCognitiveContext | None,
) -> None:
    if routed is not None:
        selected.append(routed)
    if skip is not None:
        skipped.append(skip)


def _route_memory(
    memory: MemoryRecord,
    request_terms: set[str],
) -> tuple[RoutedCognitiveContext | None, SkippedCognitiveContext | None]:
    if not memory.active:
        return None, _skip(
            id_=f"memory:{memory.id}",
            source="memory",
            summary=memory.text,
            score=0.0,
            reason="记忆已删除或过期",
        )
    if memory.confidence < 0.45:
        return None, _skip(
            id_=f"memory:{memory.id}",
            source="memory",
            summary=memory.text,
            score=memory.confidence,
            reason="记忆置信度不足",
        )
    score = _text_score(request_terms, memory.text)
    score += min(0.16, memory.confidence * 0.16)
    score += min(0.14, memory.heat * 0.14)
    if memory.layer is MemoryLayer.CORE:
        score += 0.1
    if score < 0.45:
        return None, _skip(
            id_=f"memory:{memory.id}",
            source="memory",
            summary=memory.text,
            score=score,
            reason="当前任务相关性不足",
        )
    return _route(
        id_=f"memory:{memory.id}",
        source="memory",
        summary=memory.text,
        target="main_agent",
        score=score,
        reason=f"命中长期记忆，置信度 {memory.confidence:.2f}",
    ), None


def _route_belief(
    belief: BeliefRecord,
    request_terms: set[str],
) -> tuple[RoutedCognitiveContext | None, SkippedCognitiveContext | None]:
    summary = f"{belief.subject}: {belief.claim}"
    if belief.confidence < 0.45:
        return None, _skip(
            id_=f"belief:{belief.id}",
            source="belief",
            summary=summary,
            score=belief.confidence,
            reason="信念置信度不足",
        )
    if belief.status not in {"active", "confirmed"}:
        return None, _skip(
            id_=f"belief:{belief.id}",
            source="belief",
            summary=summary,
            score=belief.confidence,
            reason="信念状态未激活",
        )
    score = _text_score(request_terms, summary)
    score += min(0.18, belief.confidence * 0.18)
    score += min(0.08, len(belief.evidence) * 0.04)
    score -= min(0.18, len(belief.contradictions) * 0.08)
    if score < 0.34:
        return None, _skip(
            id_=f"belief:{belief.id}",
            source="belief",
            summary=summary,
            score=score,
            reason="当前任务相关性不足",
        )
    return _route(
        id_=f"belief:{belief.id}",
        source="belief",
        summary=summary,
        target="main_agent",
        score=score,
        reason=f"命中信念模型，置信度 {belief.confidence:.2f}",
    ), None


def _route_relationship(
    relationship: RelationshipStateRecord,
    request_terms: set[str],
) -> tuple[RoutedCognitiveContext | None, SkippedCognitiveContext | None]:
    summary = "；".join(
        (
            f"语言={relationship.preferred_language}",
            f"确认方式={relationship.preferred_confirmation_style}",
            *_bounded_items(relationship.shared_milestones, 2),
            *_bounded_items(relationship.recent_friction_points, 2),
        )
    )
    score = _text_score(request_terms, summary)
    score += min(0.12, relationship.familiarity * 0.12)
    if relationship.familiarity >= 0.5:
        score += 0.2
    if any(term in {"确认", "语言", "风格", "偏好"} for term in request_terms):
        score += 0.15
    if score < 0.4:
        return None, _skip(
            id_=relationship.id,
            source="relationship",
            summary=summary,
            score=score,
            reason="当前任务相关性不足",
        )
    return _route(
        id_=relationship.id,
        source="relationship",
        summary=summary,
        target="main_agent",
        score=score,
        reason=f"命中关系模型，熟悉度 {relationship.familiarity:.2f}",
    ), None


def _route_world_state(
    world: WorldStateRecord,
    request_terms: set[str],
) -> tuple[RoutedCognitiveContext | None, SkippedCognitiveContext | None]:
    summary = "；".join(
        (
            f"范围={world.scope}",
            *_bounded_items(world.facts, 3),
            *_bounded_items(world.open_items, 2),
            *_bounded_items(world.future_events, 2),
        )
    )
    score = _text_score(request_terms, summary)
    score += min(0.1, len(world.evidence) * 0.04)
    if world.last_verified_at is not None:
        score += 0.06
    if score < 0.25:
        return None, _skip(
            id_=world.id,
            source="world_state",
            summary=summary,
            score=score,
            reason="当前任务相关性不足",
        )
    return _route(
        id_=world.id,
        source="world_state",
        summary=summary,
        target="main_agent",
        score=score,
        reason="命中世界状态",
    ), None


def _route_skill(
    skill: SkillCandidateRecord,
    request_terms: set[str],
    mode: str,
    agent_ids: set[str],
) -> tuple[RoutedCognitiveContext | None, SkippedCognitiveContext | None]:
    summary = "；".join((skill.name, skill.purpose, *_bounded_items(skill.steps, 4)))
    if skill.confidence < 0.45:
        return None, _skip(
            id_=f"skill:{skill.id}",
            source="skill",
            summary=summary,
            score=skill.confidence,
            reason="技能候选置信度不足",
        )
    if skill.status not in {"candidate", "active"}:
        return None, _skip(
            id_=f"skill:{skill.id}",
            source="skill",
            summary=summary,
            score=skill.confidence,
            reason="技能候选状态未激活",
        )
    score = _text_score(request_terms, summary)
    score += min(0.16, skill.confidence * 0.16)
    if skill.use_count:
        score += min(0.1, skill.success_count / max(1, skill.use_count) * 0.1)
        score -= min(0.16, skill.failure_count / max(1, skill.use_count) * 0.16)
    if mode in skill.required_inputs:
        score += 0.04
    if agent_ids and any(agent in summary for agent in agent_ids):
        score += 0.08
    if score < 0.45:
        return None, _skip(
            id_=f"skill:{skill.id}",
            source="skill",
            summary=summary,
            score=score,
            reason="当前任务相关性不足",
        )
    return _route(
        id_=f"skill:{skill.id}",
        source="skill",
        summary=summary,
        target="main_agent",
        score=score,
        reason=f"命中技能库候选，置信度 {skill.confidence:.2f}",
    ), None


def _route(
    *,
    id_: str,
    source: str,
    summary: str,
    target: str,
    score: float,
    reason: str,
) -> RoutedCognitiveContext:
    return RoutedCognitiveContext(
        id=id_,
        source=source,
        summary=_compact(summary),
        target=target,
        score=round(max(0.0, min(1.0, score)), 2),
        reason=reason,
    )


def _skip(
    *,
    id_: str,
    source: str,
    summary: str,
    score: float,
    reason: str,
) -> SkippedCognitiveContext:
    return SkippedCognitiveContext(
        id=id_,
        source=source,
        summary=_compact(summary),
        score=round(max(0.0, min(1.0, score)), 2),
        reason=reason,
    )


def _text_score(request_terms: set[str], text: str) -> float:
    text_terms = _terms(text)
    if not request_terms or not text_terms:
        return 0.0
    overlap = request_terms & text_terms
    if not overlap:
        return 0.0
    return min(0.56, 0.16 * len(overlap))


def _terms(text: str) -> set[str]:
    lowered = text.casefold()
    terms = set(re.findall(r"[a-z0-9_]+", lowered))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    for run in cjk_runs:
        if len(run) == 1:
            terms.add(run)
            continue
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _bounded_items(items: tuple[str, ...], limit: int) -> tuple[str, ...]:
    return items[:limit]


def _compact(value: str, *, limit: int = 240) -> str:
    cleaned = " ".join(value.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}…"
