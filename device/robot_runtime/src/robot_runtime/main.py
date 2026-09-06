from __future__ import annotations

import asyncio

from robot_runtime.config import RuntimeConfig
from robot_runtime.runtime import RobotRuntime


async def _amain() -> None:
    runtime = RobotRuntime(RuntimeConfig())
    await runtime.run_forever()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
