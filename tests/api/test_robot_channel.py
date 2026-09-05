from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent_hub.app import create_app
from agent_hub.domain.runs import RunStatus, TaskMode
from agent_hub.runs.service import SubmittedRun
from agent_hub.settings import Settings


class RobotRunService:
    def __init__(self, *, auto_complete: bool = True) -> None:
        self.auto_complete = auto_complete
        self.submit_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[UUID] = []
        self.cancelled: set[UUID] = set()
        self._events: dict[UUID, list[dict[str, object]]] = {}
        self._status: dict[UUID, RunStatus] = {}

    def queue_completion(self, run_id: UUID, answer: str) -> None:
        self._events[run_id] = [
            {
                "kind": "artifact.created",
                "payload": {"output": answer, "producer": "direct_runtime"},
                "artifact": {
                    "producer": "main_agent",
                    "content": {"text": answer},
                },
            },
            {"kind": "runtime.completed", "payload": {}},
        ]
        self._status[run_id] = RunStatus.COMPLETED

    async def submit(
        self,
        *,
        tenant_id: UUID,
        actor_id: UUID,
        message: str,
        mode: TaskMode,
        attachment_ids: tuple[str, ...] = (),
        conversation_id: str | None = None,
        channel_context: dict[str, str] | None = None,
        vibe_coding: bool = False,
        skip_evolution_proposal: bool = False,
        idempotency_key: str | None = None,
        direct_model: str | None = None,
    ) -> SubmittedRun:
        del tenant_id, actor_id, attachment_ids, vibe_coding
        del skip_evolution_proposal, idempotency_key, direct_model
        run_id = uuid4()
        self.submit_calls.append(
            {
                "message": message,
                "mode": mode,
                "conversation_id": conversation_id,
                "channel_context": channel_context,
            }
        )
        if self.auto_complete:
            self.queue_completion(run_id, "你好，我在。")
        else:
            self._events[run_id] = []
            self._status[run_id] = RunStatus.RUNNING
        return SubmittedRun(
            id=run_id,
            tenant_id=UUID("00000000-0000-4000-8000-000000000001"),
            status=RunStatus.QUEUED,
            mode=mode,
            decision_token=None,
            version=1,
        )

    async def events(self, tenant_id: UUID, run_id: UUID) -> tuple[dict[str, object], ...]:
        del tenant_id
        if run_id in self.cancelled:
            return ()
        return tuple(self._events.get(run_id, []))

    async def get(self, tenant_id: UUID, run_id: UUID) -> object:
        del tenant_id
        status = (
            RunStatus.CANCELLED if run_id in self.cancelled else self._status.get(run_id)
        )
        return type("Record", (), {"status": status or RunStatus.RUNNING})()

    async def cancel(self, tenant_id: UUID, run_id: UUID) -> object:
        del tenant_id
        self.cancel_calls.append(run_id)
        self.cancelled.add(run_id)
        self._status[run_id] = RunStatus.CANCELLED
        return type("Record", (), {"status": RunStatus.CANCELLED})()


async def _ok_probe() -> None:
    return None


def _app(run_service: RobotRunService) -> FastAPI:
    return create_app(
        settings=Settings.model_construct(environment="test"),
        auth_service=object(),
        rate_limiter=object(),
        config_service=object(),
        admin_resource_service=object(),
        user_admin_service=object(),
        run_service=run_service,
        database_probe=_ok_probe,
        redis_probe=_ok_probe,
    )


def _register(client: TestClient, device_id: str = "pi-01") -> str:
    response = client.post("/api/robot/v1/devices/register", json={"device_id": device_id})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["device_id"] == device_id
    token = payload["device_token"]
    assert isinstance(token, str) and token
    return token


def _until(websocket: Any, message_type: str, *, limit: int = 8) -> dict[str, Any]:
    for _ in range(limit):
        message = websocket.receive_json()
        if message.get("type") == message_type:
            return message
    raise AssertionError(f"did not receive {message_type}")


def test_robot_ws_rejects_missing_token() -> None:
    with TestClient(_app(RobotRunService())) as client:
        try:
            with client.websocket_connect("/api/robot/v1/ws") as websocket:
                message = websocket.receive()
                assert message["type"] == "websocket.close"
                assert message["code"] == 4401
        except WebSocketDisconnect as error:
            assert error.code == 4401


def test_robot_register_and_ws_hello_then_stream_reply() -> None:
    run_service = RobotRunService()
    with TestClient(_app(run_service)) as client:
        token = _register(client)
        with client.websocket_connect(
            "/api/robot/v1/ws",
            headers={"X-Device-Token": token},
        ) as websocket:
            hello = websocket.receive_json()
            assert hello["type"] == "hello.ok"
            assert hello["device_id"] == "pi-01"
            websocket.send_json(
                {"type": "final_transcript", "text": "你好", "turn_id": "turn-1"}
            )
            delta = _until(websocket, "text_delta")
            assert delta["text"]
            final = _until(websocket, "final")
            assert final["text"]

    assert len(run_service.submit_calls) == 1
    call = run_service.submit_calls[0]
    assert call["mode"] is TaskMode.DIRECT
    assert str(call["conversation_id"]).startswith("ch-robot-")
    context = call["channel_context"]
    assert isinstance(context, Mapping)
    assert context["source_channel"] == "robot"


def test_robot_barge_in_cancels_in_flight_run() -> None:
    run_service = RobotRunService(auto_complete=False)
    with TestClient(_app(run_service)) as client:
        token = _register(client, device_id="pi-02")
        with client.websocket_connect(f"/api/robot/v1/ws?device_token={token}") as websocket:
            hello = websocket.receive_json()
            assert hello["type"] == "hello.ok"
            websocket.send_json({"type": "utterance.end", "text": "讲个故事", "turn_id": "t1"})
            first = websocket.receive_json()
            assert first["type"] in {"text_delta", "state"}
            websocket.send_json({"type": "barge_in", "turn_id": "t1"})
            messages = [first]
            for _ in range(6):
                messages.append(websocket.receive_json())
                if any(item.get("type") == "cancelled" for item in messages):
                    break
            assert any(item.get("type") == "cancelled" for item in messages)

    assert run_service.cancel_calls


def test_robot_conversation_id_stable_across_turns() -> None:
    run_service = RobotRunService()
    with TestClient(_app(run_service)) as client:
        token = _register(client, device_id="pi-stable")
        with client.websocket_connect(
            "/api/robot/v1/ws",
            headers={"X-Device-Token": token},
        ) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "final_transcript", "text": "第一句", "turn_id": "a"})
            _until(websocket, "final")
            websocket.send_json({"type": "final_transcript", "text": "第二句", "turn_id": "b"})
            _until(websocket, "final")

    assert len(run_service.submit_calls) == 2
    first_id = run_service.submit_calls[0]["conversation_id"]
    second_id = run_service.submit_calls[1]["conversation_id"]
    assert first_id == second_id
    assert str(first_id).startswith("ch-robot-")


def test_create_app_exposes_robot_route_metadata() -> None:
    application = _app(RobotRunService())
    paths = {getattr(route, "path", "") for route in application.router.routes}
    assert "/api/robot/v1/devices/register" in paths
    assert "/api/robot/v1/ws" in paths
