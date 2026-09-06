from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from robot_runtime.bridge.protocol import Envelope
from robot_runtime.config import RuntimeConfig

MessageHandler = Callable[[Envelope], Awaitable[None]]


class BridgeClient(Protocol):
    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def send(self, envelope: Envelope) -> None: ...

    def on_message(self, handler: MessageHandler) -> None: ...


class LoggingBridgeClient:
    """Dev stub that records outbound frames until WebSocket is configured."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._handler: MessageHandler | None = None
        self.sent: list[Envelope] = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send(self, envelope: Envelope) -> None:
        self.sent.append(envelope)

    def on_message(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def wait_closed(self) -> None:
        await asyncio.Event().wait()

    async def inject_for_tests(self, envelope: Envelope) -> None:
        if self._handler is not None:
            await self._handler(envelope)

    def dumps(self, envelope: Envelope) -> str:
        return json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False)


class WebsocketBridgeClient:
    """Authenticated CubeAgent /api/robot/v1/ws client. No on-device AI."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._handler: MessageHandler | None = None
        self._connection: Any = None
        self._receiver: asyncio.Task[None] | None = None
        self._closed = asyncio.Event()
        self._closed.set()

    async def connect(self) -> None:
        import websockets

        self._closed.clear()
        self._connection = await websockets.connect(
            _ws_url_with_token(self._config.cloud_ws_url, self._config.device_token),
            additional_headers=_token_headers(self._config.device_token),
        )
        self._receiver = asyncio.create_task(self._receive_loop())

    async def close(self) -> None:
        if self._receiver is not None:
            self._receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver
            self._receiver = None
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        self._closed.set()

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def send(self, envelope: Envelope) -> None:
        if self._connection is None:
            raise RuntimeError("bridge is not connected")
        await self._connection.send(
            json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False)
        )

    def on_message(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def _receive_loop(self) -> None:
        connection = self._connection
        try:
            if connection is None:
                return
            async for raw in connection:
                if self._handler is None:
                    continue
                data = json.loads(raw) if isinstance(raw, str | bytes) else raw
                if isinstance(data, dict):
                    await self._handler(envelope_from_cloud(data))
        except Exception:
            pass
        finally:
            self._closed.set()


def envelope_from_cloud(data: dict[str, Any]) -> Envelope:
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    merged = dict(payload)
    for key in (
        "text",
        "state",
        "message",
        "device_id",
        "session_id",
        "audio",
        "format",
        "mime_type",
    ):
        if key in data and key not in merged:
            merged[key] = data[key]
    turn_id = data.get("turn_id")
    reply_id = data.get("reply_id")
    return Envelope(
        type=str(data.get("type") or "error"),
        turn_id=turn_id if isinstance(turn_id, str) else None,
        reply_id=reply_id if isinstance(reply_id, str) else None,
        payload=merged,
    )


def _ws_url_with_token(url: str, token: str) -> str:
    if not token:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("device_token", token)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _token_headers(token: str) -> list[tuple[str, str]]:
    return [("X-Device-Token", token)] if token else []

