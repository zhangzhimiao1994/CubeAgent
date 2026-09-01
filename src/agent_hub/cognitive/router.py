from __future__ import annotations

import re
from dataclasses import dataclass

from agent_hub.cognitive.types import ExperienceRecord


@dataclass(frozen=True, slots=True)
class RoutedExperience:
    id: str
    summary: str
    memory_type: str
    target: str
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class SkippedExperience:
    id: str
    summary: str
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class ExperienceRouteResult:
    selected: tuple[RoutedExperience, ...]
    skipped: tuple[SkippedExperience, ...]


def route_experiences(
    *,
    request: str,
    mode: str,
    agent_ids: tuple[str, ...],
    experiences: tuple[ExperienceRecord, ...],
) -> ExperienceRouteResult:
    selected: list[tuple[float, RoutedExperience]] = []
    skipped: list[SkippedExperience] = []
    request_text = request.casefold()
    request_terms = _terms(request_text)
    agent_id_set = set(agent_ids)

    for experience in experiences:
        score = _score_experience(experience, request_terms, request_text, mode, agent_id_set)
        if not experience.active_for_runtime:
            skipped.append(_skipped(experience, score, "经验尚未确认"))
            continue
        if _conflicts_with_request(experience, request_text):
            skipped.append(_skipped(experience, score, "当前用户指令覆盖这条经验"))
            continue
        if experience.confidence < 0.45:
            skipped.append(_skipped(experience, score, "置信度不足"))
            continue
        if score >= 0.70:
            selected.append(
                (
                    score,
                    RoutedExperience(
                        id=f"cognitive_experience:{experience.id}",
                        summary=experience.summary,
                        memory_type=experience.kind.value,
                        target=experience.applies_to_agents[0]
                        if experience.applies_to_agents
                        else "main_agent",
                        score=round(score, 2),
                        reason=_selection_reason(experience, score),
                    ),
                )
            )
        elif score >= 0.45:
            skipped.append(_skipped(experience, score, "当前任务相关性不足"))

    selected.sort(key=lambda item: item[0], reverse=True)
    skipped.sort(key=lambda item: item.score, reverse=True)
    return ExperienceRouteResult(
        selected=tuple(item for _, item in selected[:3]),
        skipped=tuple(skipped[:5]),
    )


def _score_experience(
    experience: ExperienceRecord,
    request_terms: set[str],
    request_text: str,
    mode: str,
    agent_ids: set[str],
) -> float:
    tags = set(experience.tags)
    score = 0.0
    if tags & request_terms:
        score += min(0.32, 0.16 * len(tags & request_terms))
    if any(tag.casefold() in request_text for tag in tags if len(tag) >= 2):
        score += 0.16
    if mode in experience.applies_to_modes:
        score += 0.18
    if agent_ids & set(experience.applies_to_agents):
        score += 0.15
    score += min(0.2, experience.confidence * 0.2)
    score += min(0.08, len(experience.evidence) * 0.04)
    if experience.use_count:
        score += min(0.08, experience.success_count / max(1, experience.use_count) * 0.08)
        score -= min(0.12, experience.failure_count / max(1, experience.use_count) * 0.12)
    if experience.contradictions:
        score -= min(0.2, len(experience.contradictions) * 0.08)
    return max(0.0, min(1.0, score))


def _conflicts_with_request(experience: ExperienceRecord, request_text: str) -> bool:
    experience_text = " ".join((experience.summary, experience.lesson, experience.strategy, *experience.tags)).casefold()
    direct_requested = any(token in request_text for token in ("直连", "direct", "不要 hybrid", "不要混合", "不混合"))
    hybrid_suggested = any(token in experience_text for token in ("hybrid", "混合"))
    return direct_requested and hybrid_suggested


def _selection_reason(experience: ExperienceRecord, score: float) -> str:
    return f"命中已确认经验，相关性评分 {score:.2f}，置信度 {experience.confidence:.2f}。"


def _skipped(experience: ExperienceRecord, score: float, reason: str) -> SkippedExperience:
    return SkippedExperience(
        id=f"cognitive_experience:{experience.id}",
        summary=experience.summary,
        score=round(score, 2),
        reason=reason,
    )


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_\u4e00-\u9fff]+", text.casefold()))
