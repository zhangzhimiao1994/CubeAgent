#!/usr/bin/env python3
"""Live L0–L5(/L8) smoke for CubeAgent /api/robot/v1. Stdlib only.

Example (from a machine that can reach the host):

  python3 scripts/robot-live-smoke.py --base-url http://103.236.93.62:32020

Cursor cloud egress is often blocked by security groups; a timeout is not a
product-logic failure. Do not put secrets in arguments or commit output tokens
to git.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

_DEFAULT_BASE = "http://103.236.93.62:32020"
_SNIPPET = 240


class SmokeError(Exception):
    """One failed layer; already printed as FAIL."""


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    failures = 0
    try:
        _run(args)
    except SmokeError:
        failures = 1
    except KeyboardInterrupt:
        _fail("interrupted", "KeyboardInterrupt")
        failures = 1
    return 1 if failures else 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=_DEFAULT_BASE, help="CubeAgent origin")
    parser.add_argument("--device-id", default="debug-pi-smoke")
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--ws-timeout", type=float, default=30.0)
    parser.add_argument(
        "--audio",
        action="store_true",
        help="L5: send tiny pcm16 + audio.end (debug text if STT is off)",
    )
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="L8: send barge_in after the first text_delta",
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> None:
    base = args.base_url.rstrip("/")
    print(f"base-url={base}")
    print("note: Cursor cloud IPs may be blocked; run this on a host that can open the URL")
    _layer_health(base, args.http_timeout)
    token = _layer_register(base, args.device_id, args.http_timeout)
    ws_url = _ws_url(base, token)
    _layer_text(
        ws_url,
        args.ws_timeout,
        barge_in=args.barge_in and not args.audio,
    )
    if args.audio:
        _layer_audio(ws_url, args.ws_timeout, barge_in=args.barge_in)
    _pass("summary", "live smoke finished")


def _layer_health(base: str, timeout: float) -> None:
    live_ok = True
    for path in ("/health", "/health/live", "/health/ready"):
        url = base + path
        try:
            status, body = _http("GET", url, timeout=timeout)
        except Exception as error:
            _fail(
                f"L0 {path}",
                f"unreachable: {_err(error)} "
                "(security group / firewall may block this client IP)",
            )
            raise SmokeError("L0") from error
        snippet = _snippet(body)
        if status == 200:
            _pass(f"L0 {path}", f"{status} {snippet}")
            continue
        _fail(f"L0 {path}", f"{status} {snippet}")
        live_ok = False
    if not live_ok:
        raise SmokeError("L0")


def _layer_register(base: str, device_id: str, timeout: float) -> str:
    url = base + "/api/robot/v1/devices/register"
    payload = json.dumps({"device_id": device_id}).encode("utf-8")
    try:
        status, body = _http(
            "POST",
            url,
            timeout=timeout,
            data=payload,
            headers={"content-type": "application/json"},
        )
    except Exception as error:
        _fail("L2 register", f"unreachable: {_err(error)}")
        raise SmokeError("L2") from error
    if status != 200:
        _fail("L2 register", f"{status} {_snippet(body)}")
        raise SmokeError("L2")
    try:
        parsed = json.loads(body)
        token = parsed["device_token"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        _fail("L2 register", f"bad body {_snippet(body)}")
        raise SmokeError("L2") from error
    if not isinstance(token, str) or not token:
        _fail("L2 register", "empty device_token")
        raise SmokeError("L2")
    _pass("L2 register", f"device_id={device_id} token={token[:12]}…")
    return token


def _layer_text(ws_url: str, timeout: float, *, barge_in: bool) -> None:
    turn_id = "smoke-text-1"
    outbound = {
        "type": "final_transcript",
        "text": "你好，请用一句话介绍你自己",
        "turn_id": turn_id,
    }
    _ws_turn(
        ws_url,
        timeout,
        label="L3 text",
        first=outbound,
        barge_in=barge_in,
        turn_id=turn_id,
        success_types=frozenset({"cancelled"} if barge_in else {"final"}),
    )


def _layer_audio(ws_url: str, timeout: float, *, barge_in: bool) -> None:
    turn_id = "smoke-audio-1"
    pcm = b"\x00\x00" * 1600
    audio = base64.b64encode(pcm).decode("ascii")
    chunk = {
        "type": "audio_chunk",
        "turn_id": turn_id,
        "audio": audio,
        "format": "pcm16",
        "sample_rate_hz": 16000,
    }
    end = {
        "type": "audio.end",
        "turn_id": turn_id,
        "audio": audio,
        "format": "pcm16",
        "sample_rate_hz": 16000,
        "text": "live smoke 你好",
    }
    _ws_turn(
        ws_url,
        timeout,
        label="L5 audio",
        first=chunk,
        extra=[end],
        barge_in=barge_in,
        turn_id=turn_id,
        success_types=frozenset({"cancelled"} if barge_in else {"final"}),
    )


def _ws_turn(
    ws_url: str,
    timeout: float,
    *,
    label: str,
    first: dict[str, Any],
    extra: list[dict[str, Any]] | None = None,
    barge_in: bool,
    turn_id: str,
    success_types: frozenset[str],
) -> None:
    deadline = time.monotonic() + timeout
    try:
        sock = _ws_connect(ws_url, timeout=max(1.0, timeout / 3))
    except Exception as error:
        _fail(label, f"ws connect: {_err(error)}")
        raise SmokeError(label) from error
    saw_hello = False
    saw_delta = False
    saw_audio = False
    barged = False
    terminal: str | None = None
    try:
        hello = _ws_recv_json(sock, deadline)
        print(f"  ws <- {_event_line(hello)}")
        if hello.get("type") == "hello.ok":
            saw_hello = True
        else:
            _fail(label, f"expected hello.ok, got {hello.get('type')!r}")
            raise SmokeError(label)
        _ws_send_json(sock, first)
        for item in extra or ():
            _ws_send_json(sock, item)
        while time.monotonic() < deadline:
            event = _ws_recv_json(sock, deadline)
            print(f"  ws <- {_event_line(event)}")
            kind = str(event.get("type") or "")
            if kind == "text_delta":
                saw_delta = True
                if barge_in and not barged:
                    _ws_send_json(sock, {"type": "barge_in", "turn_id": turn_id})
                    barged = True
                    print("  ws -> barge_in")
            if kind == "audio_delta":
                saw_audio = True
            if kind == "error":
                _fail(label, str(event.get("message") or "error"))
                raise SmokeError(label)
            if kind in {"final", "cancelled", "audio.final"}:
                terminal = kind
                if kind in success_types:
                    break
        else:
            _fail(label, f"timeout {timeout:.0f}s waiting for {sorted(success_types)}")
            raise SmokeError(label)
    finally:
        _ws_close(sock)
    if barge_in and terminal != "cancelled":
        _fail(label, f"expected cancelled after barge_in, got {terminal!r}")
        raise SmokeError(label)
    extra_note = "hello.ok"
    if saw_delta:
        extra_note += " text_delta"
    if saw_audio:
        extra_note += " audio_delta"
    if terminal:
        extra_note += f" {terminal}"
    if saw_hello:
        _pass(label, extra_note)


def _http(
    method: str,
    url: str,
    *,
    timeout: float,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str]:
    request = Request(url, data=data, method=method, headers=headers or {})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), body
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return int(error.code), body


def _ws_url(base: str, token: str) -> str:
    parsed = urlparse(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"device_token": token})
    return urlunparse((scheme, parsed.netloc, "/api/robot/v1/ws", "", query, ""))


def _ws_connect(url: str, *, timeout: float) -> ssl.SSLSocket | socket.socket:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    secure = parsed.scheme == "wss"
    port = parsed.port or (443 if secure else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    raw = socket.create_connection((host, port), timeout=timeout)
    raw.settimeout(timeout)
    sock: ssl.SSLSocket | socket.socket = raw
    if secure:
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(raw, server_hostname=host)
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {parsed.netloc}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    header = _read_http_headers(sock)
    if " 101 " not in header.split("\r\n", 1)[0]:
        raise OSError(f"websocket upgrade rejected: {_snippet(header)}")
    expected = _accept_key(key)
    if expected.lower() not in header.lower():
        raise OSError("websocket accept mismatch")
    return sock


def _read_http_headers(sock: socket.socket) -> str:
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise OSError("connection closed during websocket handshake")
        buf += chunk
        if len(buf) > 65536:
            raise OSError("handshake headers too large")
    return buf.decode("iso-8859-1", errors="replace")


def _accept_key(key: str) -> str:
    digest = hashlib.sha1(
        (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def _ws_send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    mask = os.urandom(4)
    header = bytearray()
    header.append(0x81)
    length = len(data)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    header.extend(mask)
    sock.sendall(bytes(header) + _mask(data, mask))


def _ws_recv_json(sock: socket.socket, deadline: float) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("websocket receive timed out")
    sock.settimeout(remaining)
    raw = _ws_recv_text(sock)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("websocket frame was not a JSON object")
    return parsed


def _ws_recv_text(sock: socket.socket) -> str:
    pieces: list[bytes] = []
    while True:
        opcode, payload, fin = _ws_read_frame(sock)
        if opcode == 0x8:
            raise OSError("websocket closed")
        if opcode == 0x9:
            _ws_send_pong(sock, payload)
            continue
        if opcode == 0xA:
            continue
        if opcode not in {0x0, 0x1}:
            raise OSError(f"unsupported websocket opcode {opcode}")
        pieces.append(payload)
        if fin:
            return b"".join(pieces).decode("utf-8")


def _ws_read_frame(sock: socket.socket) -> tuple[int, bytes, bool]:
    header = _recv_exact(sock, 2)
    fin = (header[0] & 0x80) != 0
    opcode = header[0] & 0x0F
    masked = (header[1] & 0x80) != 0
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    mask = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, length)
    if masked:
        payload = _mask(payload, mask)
    return opcode, payload, fin


def _ws_send_pong(sock: socket.socket, payload: bytes) -> None:
    mask = os.urandom(4)
    header = bytearray([0x8A, 0x80 | len(payload)])
    header.extend(mask)
    sock.sendall(bytes(header) + _mask(payload, mask))


def _ws_close(sock: socket.socket) -> None:
    try:
        mask = os.urandom(4)
        sock.sendall(bytes([0x88, 0x80]) + mask)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise OSError("websocket closed")
        buf.extend(chunk)
    return bytes(buf)


def _mask(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % 4] for i, b in enumerate(data))


def _event_line(event: dict[str, Any]) -> str:
    kind = str(event.get("type") or "?")
    if kind == "audio_delta":
        audio = event.get("audio")
        n = len(audio) if isinstance(audio, str) else 0
        return f"{kind} format={event.get('format')} mime={event.get('mime_type')} b64={n}"
    parts = [kind]
    for key in ("state", "text", "message", "device_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{key}={_snippet(value, 80)}")
    return " ".join(parts)


def _snippet(text: str, limit: int = _SNIPPET) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _err(error: BaseException) -> str:
    if isinstance(error, URLError):
        return str(error.reason or error)
    return str(error)


def _pass(layer: str, detail: str) -> None:
    print(f"PASS {layer}: {detail}")


def _fail(layer: str, detail: str) -> None:
    print(f"FAIL {layer}: {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
