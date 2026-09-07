from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "robot-live-smoke.py"


def _mod() -> object:
    spec = importlib.util.spec_from_file_location("robot_live_smoke", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ws_url_maps_http_origin_to_ws_with_token() -> None:
    mod = _mod()
    url = mod._ws_url("http://103.236.93.62:32020/", "pi.token")  # type: ignore[attr-defined]
    assert url.startswith("ws://103.236.93.62:32020/api/robot/v1/ws?")
    assert "device_token=pi.token" in url


def test_ws_url_maps_https_origin_to_wss() -> None:
    mod = _mod()
    url = mod._ws_url("https://agent.example.com", "tok")  # type: ignore[attr-defined]
    assert url.startswith("wss://agent.example.com/api/robot/v1/ws?")
