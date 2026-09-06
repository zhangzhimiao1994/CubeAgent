from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from hashlib import sha256

_DEVICE_ID = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
_HMAC_HEX_LEN = 64


@dataclass(frozen=True, slots=True)
class DeviceCredentials:
    device_id: str
    device_token: str


class DeviceTokenService:
    """Stateless HMAC device tokens derived from the JWT signing key."""

    def __init__(self, signing_key: str) -> None:
        self._key = signing_key.encode("utf-8")

    def register(self, device_id: str) -> DeviceCredentials:
        cleaned = device_id.strip()
        if _DEVICE_ID.fullmatch(cleaned) is None:
            raise ValueError("device_id is invalid")
        return DeviceCredentials(device_id=cleaned, device_token=self._issue(cleaned))

    def authenticate(self, token: str | None) -> DeviceCredentials | None:
        if token is None or "." not in token:
            return None
        device_id, digest = token.rsplit(".", 1)
        if (
            _DEVICE_ID.fullmatch(device_id) is None
            or len(digest) != _HMAC_HEX_LEN
            or not hmac.compare_digest(digest, self._digest(device_id))
        ):
            return None
        return DeviceCredentials(device_id=device_id, device_token=token)

    def _issue(self, device_id: str) -> str:
        return f"{device_id}.{self._digest(device_id)}"

    def _digest(self, device_id: str) -> str:
        return hmac.new(self._key, device_id.encode("utf-8"), sha256).hexdigest()


__all__ = ["DeviceCredentials", "DeviceTokenService"]
