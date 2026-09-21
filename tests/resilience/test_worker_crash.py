from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

from agent_hub.runtime.worker import LocalRunQueue, _run, run_worker_loop


def test_worker_entrypoint_can_be_cancelled_cleanly() -> None:
    async def scenario() -> None:
        task = asyncio.create_task(_run())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())


def test_worker_publishes_pending_outbox_and_executes_enqueued_runs() -> None:
    async def scenario() -> None:
        run_id = uuid4()
        queue = LocalRunQueue()
        service = RecordingRunService(queue, run_id)
        stop = asyncio.Event()

        await run_worker_loop(
            service,
            queue,
            stop=stop,
            poll_interval_seconds=0.01,
            batch_limit=10,
            max_idle_polls=1,
        )

        assert service.published_limits == [10, 10]
        assert service.executed_run_ids == [run_id]

    asyncio.run(scenario())


def test_worker_recovers_after_publish_pending_failure() -> None:
    async def scenario() -> None:
        run_id = uuid4()
        queue = LocalRunQueue()
        service = FlakyPublishRunService(queue, run_id)
        stop = asyncio.Event()

        await run_worker_loop(
            service,
            queue,
            stop=stop,
            poll_interval_seconds=0.01,
            batch_limit=10,
            max_idle_polls=2,
        )

        assert service.published_limits == [10, 10, 10, 10]
        assert service.executed_run_ids == [run_id]

    asyncio.run(scenario())


def test_worker_requeues_orphaned_queued_runs_before_idle_exit() -> None:
    async def scenario() -> None:
        run_id = uuid4()
        queue = LocalRunQueue()
        service = OrphanRecoveryRunService(queue, run_id)
        stop = asyncio.Event()

        await run_worker_loop(
            service,
            queue,
            stop=stop,
            poll_interval_seconds=0.01,
            batch_limit=10,
            max_idle_polls=1,
        )

        assert service.recovery_limits == [10, 10]
        assert service.published_limits == [10, 10]
        assert service.executed_run_ids == [run_id]

    asyncio.run(scenario())


class RecordingRunService:
    def __init__(self, queue: LocalRunQueue, run_id: UUID) -> None:
        self._queue = queue
        self._run_id = run_id
        self.published_limits: list[int] = []
        self.recovery_limits: list[int] = []
        self.executed_run_ids: list[UUID] = []

    async def recover_worker_orphans(self, limit: int) -> int:
        self.recovery_limits.append(limit)
        return 0

    async def publish_pending(self, limit: int) -> int:
        self.published_limits.append(limit)
        if self.published_limits != [limit]:
            return 0
        await self._queue.enqueue_run(self._run_id, idempotency_key="outbox-1")
        return 1

    async def execute(self, run_id: UUID) -> None:
        self.executed_run_ids.append(run_id)


class FlakyPublishRunService:
    def __init__(self, queue: LocalRunQueue, run_id: UUID) -> None:
        self._queue = queue
        self._run_id = run_id
        self.published_limits: list[int] = []
        self.recovery_limits: list[int] = []
        self.executed_run_ids: list[UUID] = []

    async def recover_worker_orphans(self, limit: int) -> int:
        self.recovery_limits.append(limit)
        return 0

    async def publish_pending(self, limit: int) -> int:
        self.published_limits.append(limit)
        if len(self.published_limits) == 1:
            raise RuntimeError("transient database failure")
        if len(self.published_limits) == 2:
            await self._queue.enqueue_run(self._run_id, idempotency_key="outbox-1")
            return 1
        return 0

    async def execute(self, run_id: UUID) -> None:
        self.executed_run_ids.append(run_id)


class OrphanRecoveryRunService:
    def __init__(self, queue: LocalRunQueue, run_id: UUID) -> None:
        self._queue = queue
        self._run_id = run_id
        self.recovery_limits: list[int] = []
        self.published_limits: list[int] = []
        self.executed_run_ids: list[UUID] = []

    async def recover_worker_orphans(self, limit: int) -> int:
        self.recovery_limits.append(limit)
        if self.recovery_limits != [limit]:
            return 0
        await self._queue.enqueue_run(self._run_id, idempotency_key="orphan-recovery-1")
        return 1

    async def publish_pending(self, limit: int) -> int:
        self.published_limits.append(limit)
        return 0

    async def execute(self, run_id: UUID) -> None:
        self.executed_run_ids.append(run_id)
