from __future__ import annotations

from enum import StrEnum


class RouteTarget(StrEnum):
    CLOUD = "cloud"


class EdgeCloudRouter:
    """Pi is frontend-only: always send turns to the agent backend."""

    def choose(self, *, text_len: int = 0, needs_tools: bool = False) -> RouteTarget:
        del text_len, needs_tools
        return RouteTarget.CLOUD
