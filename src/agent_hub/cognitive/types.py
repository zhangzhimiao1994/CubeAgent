from __future__ import annotations

import unicodedata
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperienceKind(StrEnum):
    USER_PREFERENCE = "user_preference"
    PROJECT_FACT = "project_fact"
    WORKFLOW_STRATEGY = "workflow_strategy"
    ERROR_HANDLING = "error_handling"
    UI_RULE = "ui_rule"
    COMMUNICATION_STYLE = "communication_style"
    TOOLING_STRATEGY = "tooling_strategy"
    DOMAIN_PATTERN = "domain_pattern"


class ExperienceStatus(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class CognitiveEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: str = Field(min_length=1, max_length=48)
    source_id: str = Field(min_length=1, max_length=128)
    note: str = Field(min_length=1, max_length=512)

    @field_validator("source_type", "source_id", "note")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return _bounded_printable(value)


class ExperienceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    tenant_id: UUID
    user_id: UUID
    kind: ExperienceKind
    status: ExperienceStatus
    summary: str = Field(min_length=1, max_length=240)
    lesson: str = Field(min_length=1, max_length=1200)
    strategy: str = Field(min_length=1, max_length=1200)
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[CognitiveEvidence, ...] = ()
    contradictions: tuple[CognitiveEvidence, ...] = ()
    source_run_ids: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    applies_to_modes: tuple[str, ...] = ()
    applies_to_agents: tuple[str, ...] = ()
    use_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    last_used_at: datetime | None
    last_verified_at: datetime | None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    @property
    def active_for_runtime(self) -> bool:
        return self.status in {ExperienceStatus.CONFIRMED, ExperienceStatus.ACTIVE}

    @field_validator("summary", "lesson", "strategy")
    @classmethod
    def clean_body_text(cls, value: str) -> str:
        return _bounded_printable(value)

    @field_validator("tags", "applies_to_modes", "applies_to_agents", "source_run_ids", "source_memory_ids")
    @classmethod
    def clean_string_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []
        for item in value:
            cleaned = _bounded_printable(item)
            if len(cleaned) > 128:
                raise ValueError("cognitive identifiers must be bounded")
            if cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return tuple(result)

    @model_validator(mode="after")
    def validate_counts_and_times(self) -> ExperienceRecord:
        if self.success_count + self.failure_count > self.use_count:
            raise ValueError("success and failure counts cannot exceed use count")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        return self


class ReflectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    tenant_id: UUID
    user_id: UUID
    source_run_id: str
    trigger: str = Field(min_length=1, max_length=64)
    outcome: str = Field(min_length=1, max_length=32)
    causal_analysis: str = Field(min_length=1, max_length=1200)
    counterfactual: str = Field(default="", max_length=1200)
    positive_patterns: tuple[str, ...] = ()
    negative_patterns: tuple[str, ...] = ()
    proposed_experience_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    created_at: datetime

    @field_validator("source_run_id", "trigger", "outcome", "causal_analysis", "counterfactual")
    @classmethod
    def clean_reflection_text(cls, value: str) -> str:
        return _bounded_printable(value, allow_empty=True)


class BeliefRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    tenant_id: UUID
    user_id: UUID
    subject: str = Field(min_length=1, max_length=160)
    claim: str = Field(min_length=1, max_length=512)
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[CognitiveEvidence, ...] = ()
    contradictions: tuple[CognitiveEvidence, ...] = ()
    status: str = Field(min_length=1, max_length=32)
    last_verified_at: datetime | None
    created_at: datetime
    updated_at: datetime


class RelationshipStateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=160)
    tenant_id: UUID
    user_id: UUID
    familiarity: float = Field(ge=0, le=1)
    preferred_language: str = Field(default="zh-CN", min_length=1, max_length=32)
    preferred_confirmation_style: str = Field(default="minimal", min_length=1, max_length=64)
    shared_milestones: tuple[str, ...] = ()
    recent_friction_points: tuple[str, ...] = ()
    last_interaction_at: datetime | None
    created_at: datetime
    updated_at: datetime


class WorldStateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=160)
    tenant_id: UUID
    user_id: UUID
    scope: str = Field(min_length=1, max_length=160)
    facts: tuple[str, ...] = ()
    open_items: tuple[str, ...] = ()
    future_events: tuple[str, ...] = ()
    last_verified_at: datetime | None
    evidence: tuple[CognitiveEvidence, ...] = ()
    created_at: datetime
    updated_at: datetime


class SkillCandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    tenant_id: UUID
    user_id: UUID
    name: str = Field(min_length=1, max_length=128)
    purpose: str = Field(min_length=1, max_length=512)
    steps: tuple[str, ...] = Field(min_length=1)
    required_inputs: tuple[str, ...] = ()
    output_contract: str = Field(min_length=1, max_length=512)
    evidence: tuple[CognitiveEvidence, ...] = ()
    use_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    version: int = Field(ge=1)
    status: str = Field(min_length=1, max_length=32)
    created_at: datetime
    updated_at: datetime


def _bounded_printable(value: str, *, allow_empty: bool = False) -> str:
    if value != value.strip():
        raise ValueError("cognitive text must be unpadded")
    if not allow_empty and not value:
        raise ValueError("cognitive text must be non-empty")
    if any((ord(ch) < 32 and ch not in "\n\t") or ord(ch) == 127 or unicodedata.category(ch) == "Cf" for ch in value):
        raise ValueError("cognitive text must be printable")
    return value


def validate_cognitive_printable_text(value: str, *, allow_empty: bool = False) -> str:
    return _bounded_printable(value, allow_empty=allow_empty)
