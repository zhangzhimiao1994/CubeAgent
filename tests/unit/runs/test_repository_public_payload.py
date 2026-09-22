from __future__ import annotations

from uuid import uuid4

from agent_hub.runs.repository import (
    _checkpoint_required_artifact_ids,
    _event_with_failure_diagnostic,
    _public_artifact_payload,
    _public_event_payload,
)
from agent_hub.domain.runs import TaskMode
from agent_hub.runtime.contracts import EventKind, RunEvent, RuntimeCheckpoint


def test_public_artifact_payload_filters_generated_file_storage_key_recursively() -> None:
    payload: dict[str, object] = {
        "id": "artifact-1",
        "content": {
            "result": {
                "file": {
                    "filename": "report.docx",
                    "storage_key": "tenant/run/artifact/report.docx",
                    "download_url": "/api/v1/admin/runs/run/artifacts/artifact/download",
                }
            }
        },
    }

    sanitized = _public_artifact_payload(payload)

    assert sanitized == {
        "id": "artifact-1",
        "content": {
            "result": {
                "file": {
                    "filename": "report.docx",
                    "download_url": "/api/v1/admin/runs/run/artifacts/artifact/download",
                }
            }
        },
    }


def test_public_event_payload_filters_generated_file_storage_key_recursively() -> None:
    payload: dict[str, object] = {
        "type": "artifact.created",
        "artifact": {
            "content": {
                "file": {
                    "filename": "deck.pptx",
                    "storage_key": "tenant/run/artifact/deck.pptx",
                }
            }
        },
    }

    sanitized = _public_event_payload(payload)

    assert sanitized == {
        "type": "artifact.created",
        "artifact": {"content": {"file": {"filename": "deck.pptx"}}},
    }


def test_failed_run_event_is_enriched_with_public_diagnostics() -> None:
    event = RunEvent(
        kind=EventKind.RUNTIME_FAILED,
        sequence=1,
        run_id=uuid4(),
        reason="model gateway failed: model response failed (status=502)",
    )

    enriched = _event_with_failure_diagnostic(event)

    assert enriched.payload["error_summary"] == "model gateway failed: model response failed (status=502)"
    assert enriched.payload["error_code"] == "model.provider_unavailable"
    assert enriched.payload["error_stage"] == "model_provider"
    assert enriched.payload["retryable"] is True
    assert _public_event_payload(enriched.to_payload())["payload"] == enriched.payload


def test_failed_event_keeps_existing_diagnostic_payload() -> None:
    event = RunEvent(
        kind=EventKind.STEP_FAILED,
        sequence=1,
        run_id=uuid4(),
        actor="planner",
        step_id="plan",
        reason="step execution failed",
        payload={"error_code": "custom.error", "summary": "custom"},
    )

    enriched = _event_with_failure_diagnostic(event)

    assert enriched.payload == {"error_code": "custom.error", "summary": "custom"}


def test_checkpoint_required_artifact_ids_prefers_registry_roots() -> None:
    root_id = uuid4()
    external_source_id = uuid4()
    checkpoint = RuntimeCheckpoint(
        id=uuid4(),
        runtime_type="crew",
        runtime_version="7",
        run_id=uuid4(),
        tenant_id=uuid4(),
        mode=TaskMode.DISPATCH,
        state={
            "artifact_registry": {
                str(root_id): "root-sha",
                str(external_source_id): "external-source-sha",
            },
            "artifact_registry_roots": (str(root_id),),
        },
    )

    assert _checkpoint_required_artifact_ids(checkpoint) == (root_id,)


def test_checkpoint_required_artifact_ids_keeps_legacy_registry_behavior_without_roots() -> None:
    root_id = uuid4()
    checkpoint = RuntimeCheckpoint(
        id=uuid4(),
        runtime_type="crew",
        runtime_version="6",
        run_id=uuid4(),
        tenant_id=uuid4(),
        mode=TaskMode.DISPATCH,
        state={"artifact_registry": {str(root_id): "root-sha"}},
    )

    assert _checkpoint_required_artifact_ids(checkpoint) == (root_id,)
