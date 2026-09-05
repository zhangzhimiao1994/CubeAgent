from __future__ import annotations

from agent_hub.robot.tokens import DeviceTokenService


def test_register_issues_stable_token_for_same_device() -> None:
    tokens = DeviceTokenService("test-signing-key")

    first = tokens.register("pi-01")
    second = tokens.register("pi-01")

    assert first.device_id == "pi-01"
    assert first.device_token == second.device_token
    assert tokens.authenticate(first.device_token) == first
    assert tokens.authenticate("wrong-token") is None


def test_different_devices_get_different_tokens() -> None:
    tokens = DeviceTokenService("test-signing-key")

    left = tokens.register("pi-01")
    right = tokens.register("pi-02")

    assert left.device_token != right.device_token
    assert tokens.authenticate(left.device_token).device_id == "pi-01"
    assert tokens.authenticate(right.device_token).device_id == "pi-02"
