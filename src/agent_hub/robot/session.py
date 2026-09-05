from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from agent_hub.channels.base import Channel, ConversationType, InboundMessage
from agent_hub.channels.directives import ChannelDirectiveError
from agent_hub.channels.submitter import RunServiceInboundSubmitter
from agent_hub.domain.runs import RunStatus
from agent_hub.robot.events import (
    event_failed,
    event_is_terminal,
    extract_assistant_text,
    text_delta,
)
from agent_hub.robot.tokens import DeviceCredentials

_LOGGER = logging.getLogger(__name__)
_POLL_SECONDS = 0.05
_TRANSCRIPT_TYPES = frozenset({"final_transcript", "utterance.end"})
_TERMINAL_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.WAITING_USER_MODE,
        RunStatus.WAITING_APPROVAL,
    }
)


class RobotRunControl(Protocol):
    async def events(self, tenant_id: UUID, run_id: UUID) -> tuple[Mapping[str, object], ...]: ...

    async def get(self, tenant_id: UUID, run_id: UUID) -> object: ...

    async def cancel(self, tenant_id: UUID, run_id: UUID) -> object: ...


class RobotChannelSession:
    """One authenticated Pi WebSocket: submit DIRECT runs and stream answer text."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        device: DeviceCredentials,
        submitter: RunServiceInboundSubmitter,
        run_service: RobotRunControl,
        tenant_id: UUID,
        tenant_external_id: str,
    ) -> None:
        self._websocket = websocket
        self._device = device
        self._submitter = submitter
        self._run_service = run_service
        self._tenant_id = tenant_id
        self._tenant_external_id = tenant_external_id
        self._session_id = uuid4().hex
        self._active_run_id: UUID | None = None
        self._last_run_id: UUID | None = None
        self._active_turn_id: str | None = None
        self._emitted_text = ""
        self._terminal_sent = False

    async def run(self) -> None:
        await self._send(
            {
                "type": "hello.ok",
                "device_id": self._device.device_id,
                "session_id": self._session_id,
                "state": "listening",
            }
        )
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(
                        self._websocket.receive_json(),
                        timeout=_POLL_SECONDS,
                    )
                except TimeoutError:
                    await self._poll_active_run()
                    continue
                if isinstance(payload, Mapping):
                    await self._handle_inbound(payload)
        except WebSocketDisconnect:
            await self._cancel_active(send_cancelled=False)
        except Exception:
            _LOGGER.exception("robot_ws_session_failed device_id=%s", self._device.device_id)
            await self._send_error("robot session failed")
            await self._cancel_active(send_cancelled=False)

    async def _handle_inbound(self, payload: Mapping[str, object]) -> None:
        message_type = str(payload.get("type") or "")
        if message_type == "ping":
            await self._send({"type": "pong"})
            return
        if message_type == "hello":
            await self._send(
                {
                    "type": "hello.ok",
                    "device_id": self._device.device_id,
                    "session_id": self._session_id,
                    "state": "listening",
                }
            )
            return
        if message_type == "barge_in":
            turn_id = _optional_str(payload.get("turn_id")) or self._active_turn_id
            await self._cancel_active(send_cancelled=True, turn_id=turn_id)
            return
        if message_type in _TRANSCRIPT_TYPES:
            await self._submit_transcript(payload)
            return

    async def _submit_transcript(self, payload: Mapping[str, object]) -> None:
        text = _transcript_text(payload)
        turn_id = _optional_str(payload.get("turn_id")) or uuid4().hex
        if not text:
            await self._send_error("empty transcript", turn_id=turn_id)
            return
        await self._cancel_active(send_cancelled=False)
        self._active_turn_id = turn_id
        self._emitted_text = ""
        self._terminal_sent = False
        await self._send({"type": "state", "state": "thinking", "turn_id": turn_id})
        inbound = InboundMessage(
            channel=Channel.ROBOT,
            tenant_external_id=self._tenant_external_id,
            sender_external_id=self._device.device_id,
            conversation_external_id=self._device.device_id,
            message_id=turn_id,
            event_id=turn_id,
            conversation_type=ConversationType.PRIVATE,
            text=text,
            mentions_bot=True,
            received_at=datetime.now(tz=UTC),
        )
        try:
            run_id = await self._submitter.submit(
                inbound,
                idempotency_key=f"robot:{self._device.device_id}:{turn_id}",
            )
        except ChannelDirectiveError as error:
            await self._send_error(str(error) or "submit rejected", turn_id=turn_id)
            return
        except Exception:
            _LOGGER.exception("robot_run_submit_failed device_id=%s", self._device.device_id)
            await self._send_error("failed to submit run", turn_id=turn_id)
            return
        self._active_run_id = run_id
        self._last_run_id = run_id
        await self._poll_active_run()

    async def _poll_active_run(self) -> None:
        run_id = self._active_run_id
        turn_id = self._active_turn_id
        if run_id is None or self._terminal_sent:
            return
        try:
            events = await self._run_service.events(self._tenant_id, run_id)
        except Exception:
            _LOGGER.exception("robot_run_events_failed run_id=%s", run_id)
            await self._send_error("failed to read run events", turn_id=turn_id)
            self._terminal_sent = True
            self._active_run_id = None
            return
        current = extract_assistant_text(events)
        delta = text_delta(self._emitted_text, current)
        if delta:
            if not self._emitted_text:
                await self._send({"type": "state", "state": "speaking", "turn_id": turn_id})
            await self._send({"type": "text_delta", "text": delta, "turn_id": turn_id})
            self._emitted_text = current
        if any(event_failed(event) for event in events):
            await self._send_error(current or "run failed", turn_id=turn_id)
            self._terminal_sent = True
            self._active_run_id = None
            return
        if any(event_is_terminal(event) for event in events) or await self._run_finished(run_id):
            await self._send({"type": "final", "text": current, "turn_id": turn_id})
            await self._send({"type": "state", "state": "listening", "turn_id": turn_id})
            self._terminal_sent = True
            self._active_run_id = None

    async def _run_finished(self, run_id: UUID) -> bool:
        try:
            record = await self._run_service.get(self._tenant_id, run_id)
        except Exception:  # noqa: BLE001 - run lookup is best-effort for stream completion
            return False
        status = getattr(record, "status", None)
        return status in _TERMINAL_STATUSES

    async def _cancel_active(
        self,
        *,
        send_cancelled: bool,
        turn_id: str | None = None,
    ) -> None:
        run_id = self._active_run_id or self._last_run_id
        active_turn = turn_id or self._active_turn_id
        self._active_run_id = None
        self._last_run_id = None
        self._terminal_sent = True
        if run_id is not None:
            try:
                await self._run_service.cancel(self._tenant_id, run_id)
            except Exception:
                _LOGGER.exception("robot_run_cancel_failed run_id=%s", run_id)
        if send_cancelled:
            await self._send({"type": "cancelled", "turn_id": active_turn})
            await self._send({"type": "state", "state": "listening", "turn_id": active_turn})
        self._active_turn_id = None
        self._emitted_text = ""

    async def _send_error(self, message: str, *, turn_id: str | None = None) -> None:
        await self._send(
            {
                "type": "error",
                "message": message,
                "turn_id": turn_id or self._active_turn_id,
            }
        )
        await self._send(
            {
                "type": "state",
                "state": "listening",
                "turn_id": turn_id or self._active_turn_id,
            }
        )

    async def _send(self, payload: Mapping[str, object]) -> None:
        await self._websocket.send_json(dict(payload))


def _transcript_text(payload: Mapping[str, object]) -> str:
    for key in ("text", "transcript"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = payload.get("payload")
    if isinstance(nested, Mapping):
        for key in ("text", "transcript"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["RobotChannelSession", "RobotRunControl"]
