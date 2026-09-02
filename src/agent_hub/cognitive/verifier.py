from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from agent_hub.cognitive.types import OutcomeVerdict


class OutcomeAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: OutcomeVerdict
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[str, ...] = ()
    failure_or_gap_reason: str = ""
    learnable: bool


class OutcomeVerifier:
    def assess(
        self,
        *,
        terminal_status: str,
        events: Sequence[Mapping[str, object]] = (),
        artifacts: Sequence[Mapping[str, object]] = (),
    ) -> OutcomeAssessment:
        normalized_status = terminal_status.casefold()
        event_kinds = tuple(str(event.get("kind", "")).casefold() for event in events)
        visible_output = self._has_visible_output(events=events, artifacts=artifacts)
        failure_events = tuple(kind for kind in event_kinds if "failed" in kind or "error" in kind)
        if normalized_status in {"failed", "cancelled"} or any(kind == "runtime.failed" for kind in event_kinds):
            return OutcomeAssessment(
                verdict=OutcomeVerdict.FAILURE,
                confidence=0.86,
                evidence=event_kinds,
                failure_or_gap_reason="terminal runtime failure",
                learnable=True,
            )
        if normalized_status == "completed" and failure_events and visible_output:
            return OutcomeAssessment(
                verdict=OutcomeVerdict.PARTIAL,
                confidence=0.72,
                evidence=event_kinds,
                failure_or_gap_reason="failure recovered with visible output",
                learnable=True,
            )
        if normalized_status == "completed" and visible_output:
            return OutcomeAssessment(
                verdict=OutcomeVerdict.SUCCESS,
                confidence=0.8,
                evidence=event_kinds,
                learnable=True,
            )
        return OutcomeAssessment(
            verdict=OutcomeVerdict.INSUFFICIENT_EVIDENCE,
            confidence=0.42,
            evidence=event_kinds,
            failure_or_gap_reason="no visible output evidence",
            learnable=False,
        )

    @staticmethod
    def _has_visible_output(
        *,
        events: Sequence[Mapping[str, object]],
        artifacts: Sequence[Mapping[str, object]],
    ) -> bool:
        if any(_artifact_has_text(artifact) for artifact in artifacts):
            return True
        return any(
            str(event.get("kind", "")).casefold() == "message.created"
            and bool(str(event.get("message", "")).strip())
            for event in events
        ) or any(_event_artifact_has_text(event) for event in events)


def _artifact_has_text(artifact: Mapping[str, object]) -> bool:
    if str(artifact.get("text", "")).strip():
        return True
    content = artifact.get("content")
    return isinstance(content, Mapping) and bool(str(content.get("text", "")).strip())


def _event_artifact_has_text(event: Mapping[str, object]) -> bool:
    artifact = event.get("artifact")
    return isinstance(artifact, Mapping) and _artifact_has_text(artifact)
