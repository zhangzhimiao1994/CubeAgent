from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryLayer(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    CORE = "core"


class MemoryTier(StrEnum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVE = "archive"


class MemoryCategory(StrEnum):
    PREFERENCE = "preference"
    FACT = "fact"
    TASK = "task"
    SUMMARY = "summary"
    DECISION = "decision"
    LESSON = "lesson"
    OTHER = "other"


class MemorySummaryPeriod(StrEnum):
    NONE = "none"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class MemoryAddStatus(StrEnum):
    STORED = "stored"
    DEDUPLICATED = "deduplicated"
    REJECTED_SENSITIVE = "rejected_sensitive"
    REJECTED_PROMPT_LIKE = "rejected_prompt_like"
    REJECTED_UNCONFIRMED_CORE = "rejected_unconfirmed_core"


class MemoryRetentionAction(StrEnum):
    KEEP = "keep"
    COMPRESS = "compress"
    COOL_DOWN = "cool_down"
    ARCHIVE = "archive"
    TOMBSTONE = "tombstone"
    PURGE = "purge"


class MemoryRetentionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stale_candidate_days: int = Field(default=30, ge=1, le=3650)
    cold_archive_days: int = Field(default=180, ge=1, le=3650)
    tombstone_purge_days: int = Field(default=90, ge=1, le=3650)
    archive_purge_days: int = Field(default=365, ge=1, le=3650)
    min_retention_score: float = Field(default=0.22, ge=0, le=1)
    compress_after_source_count: int = Field(default=3, ge=2, le=100)
    max_active_records_per_user: int = Field(default=1000, ge=1, le=100000)


class MemoryRetentionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    memory_id: UUID
    action: MemoryRetentionAction
    score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=256)
    protected: bool = False


class MemoryMaintenanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    decisions: tuple[MemoryRetentionDecision, ...]
    compressed: int = Field(default=0, ge=0)
    archived: int = Field(default=0, ge=0)
    tombstoned: int = Field(default=0, ge=0)
    purged: int = Field(default=0, ge=0)
    cooled_down: int = Field(default=0, ge=0)


class MemoryAuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: str
    tenant_id: UUID
    user_id: UUID
    memory_id: UUID | None = None
    status: str
    reason: str | None = None
    created_at: datetime


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: UUID
    tenant_id: UUID
    user_id: UUID
    layer: MemoryLayer
    category: MemoryCategory
    text: str = Field(min_length=1, max_length=4096)
    confidence: float = Field(ge=0, le=1)
    source_run_id: UUID | None = None
    source_event_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    heat: float = Field(default=0.5, ge=0, le=1)
    last_recalled_at: datetime | None = None
    recall_count: int = Field(default=0, ge=0)
    source_memory_ids: tuple[UUID, ...] = ()
    locked: bool = False
    project_id: str | None = Field(default=None, min_length=1, max_length=128)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    summary_period: MemorySummaryPeriod = MemorySummaryPeriod.NONE
    metadata: dict[str, str] = Field(default_factory=dict, max_length=16)
    expires_at: datetime | None = None
    deleted_at: datetime | None = None
    tombstone_reason: str | None = Field(default=None, max_length=256)
    archived_at: datetime | None = None
    archive_reason: str | None = Field(default=None, max_length=256)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if value != value.strip() or any(ord(ch) < 32 and ch not in "\n\t" for ch in value):
            raise ValueError("memory text must be non-empty unpadded printable text")
        return value

    @field_validator("project_id", "conversation_id")
    @classmethod
    def validate_scope_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.strip() or any(ch.isspace() for ch in value):
            raise ValueError("memory scope identifiers must be unpadded non-whitespace strings")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if (
                key != key.strip()
                or item != item.strip()
                or not key
                or len(key) > 64
                or len(item) > 256
                or any(ord(ch) < 32 for ch in key + item)
            ):
                raise ValueError("memory metadata must contain bounded printable strings")
        return value

    @field_validator("created_at", "updated_at", "last_recalled_at", "expires_at", "deleted_at", "archived_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("memory timestamps must be timezone aware")
        return value

    @model_validator(mode="after")
    def validate_times(self) -> MemoryRecord:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        if self.archived_at is not None and self.archived_at < self.created_at:
            raise ValueError("archived_at cannot be before created_at")
        return self

    @property
    def active(self) -> bool:
        now = datetime.now(UTC)
        return (
            self.deleted_at is None
            and self.archived_at is None
            and (self.expires_at is None or self.expires_at > now)
        )


class MemoryAddResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: MemoryAddStatus
    record: MemoryRecord | None = None
    reason: str | None = None
