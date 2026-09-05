from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class CompanionStateMirror(BaseModel):
    persona_id: str = "default"
    relationship_level: int = Field(default=0, ge=0, le=100)
    mood: str = "neutral"
    scene: str = "idle_chat"
    last_user_topics: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LocalSession:
    def __init__(self) -> None:
        self.session_id: str | None = None
        self.turn_id: str | None = None
        self.reply_id: str | None = None
        self.pending_transcript: str = ""
        self.companion = CompanionStateMirror()

    def apply_cloud_state(self, payload: dict[str, object]) -> None:
        self.companion = CompanionStateMirror.model_validate(
            {**self.companion.model_dump(mode="python"), **payload}
        )
