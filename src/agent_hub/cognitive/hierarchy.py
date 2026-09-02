from __future__ import annotations

import re

from agent_hub.memory.types import MemoryLayer, MemoryRecord


class WorkingSetBuilder:
    def build(
        self,
        *,
        request: str,
        memories: tuple[MemoryRecord, ...],
        limit: int = 8,
    ) -> tuple[MemoryRecord, ...]:
        if limit < 1 or limit > 100:
            raise ValueError("working set limit must be between 1 and 100")
        scored: list[tuple[float, MemoryRecord]] = []
        for memory in memories:
            if memory.deleted_at is not None or memory.archived_at is not None:
                continue
            score = self._score(memory, request=request)
            scored.append((score, memory))
        scored.sort(key=lambda item: (-item[0], -item[1].heat, -item[1].confidence, item[1].created_at))
        return tuple(memory for _score, memory in scored[:limit])

    @staticmethod
    def _score(memory: MemoryRecord, *, request: str) -> float:
        request_terms = _terms(request)
        memory_terms = _terms(memory.text)
        relevance = len(request_terms & memory_terms)
        layer_boost = {
            MemoryLayer.WORKING: 3.0,
            MemoryLayer.CORE: 1.5,
            MemoryLayer.EPISODIC: 0.5,
        }[memory.layer]
        return float(relevance * 2) + layer_boost + memory.heat + memory.confidence


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.casefold()))
