# Cognitive Experience Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conservative Cognitive Experience Layer that turns meaningful feedback and outcomes into confirmed, evidence-backed experience that can improve future conversation-agent decisions.

**Architecture:** Add a new `agent_hub.cognitive` package with typed models, service logic, routing, and repository boundaries. Reuse existing Hermes+/Memory/runtime injection paths instead of introducing a vector database or harness behavior.

**Tech Stack:** Python 3.12, Pydantic, SQLAlchemy/AdminResourceRow-compatible persistence, pytest, FastAPI, React, TypeScript, Zod, Vitest.

---

## File Map

- Create `src/agent_hub/cognitive/types.py`
  - Pydantic/domain types for experiences, reflections, beliefs, relationship state, world state, skill candidates, routing decisions.
- Create `src/agent_hub/cognitive/repository.py`
  - In-memory repository plus a persistent adapter using `AdminResourceRow.kind = "hermes"` with `cognitive_*` resource-id prefixes.
- Create `src/agent_hub/cognitive/service.py`
  - Candidate creation, confirmation, rejection, deletion, confidence updates, usage outcome recording.
- Create `src/agent_hub/cognitive/reflection.py`
  - Deterministic reflection extraction from explicit feedback and terminal outcomes.
- Create `src/agent_hub/cognitive/router.py`
  - Retrieval, quality gating, conflict detection, selected/skipped explanation.
- Modify `src/agent_hub/runs/service.py`
  - Call the cognitive service after terminal outcomes and before runtime submission where safe.
- Modify `src/agent_hub/hermes/advisor.py`
  - Read confirmed active experiences as an additional advice source through a typed adapter and convert them to the existing `HermesMemoryInjection` shape.
- Modify `src/agent_hub/runtime/hermes_context.py`
  - Render confirmed experience guidance using the existing bounded `injected_memories` prompt strategy.
- Modify `src/agent_hub/api/routers/admin.py`
  - Add admin endpoints for cognitive records and confirmation actions.
- Modify `web/src/api/client.ts`
  - Add schemas and API methods for cognitive records.
- Modify `web/src/pages/HermesPage.tsx`
  - Add compact experience/reflection views without a new top-level navigation module.
- Modify `web/src/pages/RunsPage.tsx`
  - Show compact experience injection summary in the existing dispatch/process drawer.
- Modify `web/src/styles.css`
  - Add compact row/card/drawer styles if existing Hermes styles are insufficient.
- Add tests under:
  - `tests/unit/cognitive/`
  - `tests/unit/hermes/test_advisor.py`
  - `tests/unit/runs/test_conversation_mode.py`
  - `tests/unit/runtime/test_direct_prompt.py`
  - `tests/api/test_admin_resources.py`
  - `web/src/pages/OperationalPages.test.tsx`

---

## Execution Order and Sub-Agent Ownership

Run these tasks mostly in sequence because API/runtime shape depends on the domain model:

| Order | Owner role | Files | Dependency |
|---|---|---|---|
| 1 | domain worker | `src/agent_hub/cognitive/types.py`, `tests/unit/cognitive/test_types.py` | none |
| 2 | service worker | `src/agent_hub/cognitive/repository.py`, `src/agent_hub/cognitive/service.py`, service tests | Task 1 |
| 3 | router worker | `src/agent_hub/cognitive/router.py`, router tests | Task 1 |
| 4 | integration worker | `src/agent_hub/hermes/advisor.py`, `src/agent_hub/runs/service.py`, runtime context tests | Tasks 1-3 |
| 5 | API/UI worker | `src/agent_hub/api/routers/admin.py`, `web/src/api/client.ts`, `web/src/pages/HermesPage.tsx` | Tasks 1-4 |
| 6 | verifier | whole repo checks, deployment, CI, handoff | Tasks 1-5 |

All workers must remember they are not alone in the codebase and must not revert edits made by others. Reviewers should fix in-scope issues instead of handing them back when the fix is clear.

## Hard Negative Constraints

Do not use the old Evolution implementation for this feature:

- Do not import or call `agent_hub.evolution_hooks`.
- Do not write `AdminResourceRow.kind = "evolution"`.
- Do not call or create `/api/v1/admin/evolution-runs`.
- Do not generate `routing_decision.source = "evolution"`.
- Do not route cognitive skill candidates into old Evolution round execution.

Use `AdminResourceRow.kind = "hermes"` with namespaced `resource_id` values instead:

- `cognitive_experience:<uuid>`
- `cognitive_reflection:<uuid>`
- `cognitive_belief:<uuid>`
- `cognitive_relationship:<user_id>`
- `cognitive_world:<scope_id>`
- `cognitive_skill:<uuid>`

First-slice confirmed experiences are converted into the existing `HermesMemoryInjection` compatible payload:

```json
{
  "id": "cognitive_experience:<uuid>",
  "summary": "reviewer 超时时先压缩上下文再分块审查。",
  "memory_type": "error_handling",
  "target": "quality_reviewer",
  "score": 0.86,
  "reason": "命中已确认经验，当前任务包含 reviewer 超时风险。"
}
```

Belief, relationship, world-state, and skill-library advanced behavior is first represented by typed records and read/list APIs only. Automatic mutation of SOUL/persona/security/tool permission state is out of scope.

### Task 1: Cognitive Domain Types

**Files:**
- Create: `src/agent_hub/cognitive/__init__.py`
- Create: `src/agent_hub/cognitive/types.py`
- Test: `tests/unit/cognitive/test_types.py`

- [ ] **Step 1: Write failing model validation tests**

Create `tests/unit/cognitive/test_types.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_hub.cognitive.types import CognitiveEvidence, ExperienceKind, ExperienceRecord, ExperienceStatus


def test_experience_requires_bounded_confidence_and_evidence() -> None:
    now = datetime.now(UTC)
    record = ExperienceRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        kind=ExperienceKind.ERROR_HANDLING,
        status=ExperienceStatus.CANDIDATE,
        summary="reviewer 超时时先压缩上下文再分块审查。",
        lesson="大输入导致 reviewer 超时。",
        strategy="先压缩输入，再拆分审查任务。",
        confidence=0.72,
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="reviewer timeout"),),
        contradictions=(),
        source_run_ids=("run-1",),
        source_memory_ids=(),
        tags=("reviewer", "timeout"),
        applies_to_modes=("hybrid", "dispatch"),
        applies_to_agents=("quality_reviewer",),
        use_count=0,
        success_count=0,
        failure_count=0,
        last_used_at=None,
        last_verified_at=now,
        version=1,
        created_at=now,
        updated_at=now,
    )

    assert record.summary == "reviewer 超时时先压缩上下文再分块审查。"
    assert record.active_for_runtime is False


def test_experience_rejects_unbounded_confidence() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError):
        ExperienceRecord(
            id=uuid4(),
            tenant_id=uuid4(),
            user_id=uuid4(),
            kind=ExperienceKind.USER_PREFERENCE,
            status=ExperienceStatus.ACTIVE,
            summary="用户偏好简洁回答。",
            lesson="用户多次要求精简。",
            strategy="先给结论，再给必要证据。",
            confidence=1.4,
            evidence=(),
            contradictions=(),
            source_run_ids=(),
            source_memory_ids=(),
            tags=("communication",),
            applies_to_modes=(),
            applies_to_agents=(),
            use_count=0,
            success_count=0,
            failure_count=0,
            last_used_at=None,
            last_verified_at=now,
            version=1,
            created_at=now,
            updated_at=now,
        )
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\cognitive\test_types.py -q
```

Expected: fails because `agent_hub.cognitive.types` does not exist.

- [ ] **Step 3: Implement the domain types**

Create `src/agent_hub/cognitive/__init__.py`:

```python
"""Cognitive experience layer for durable learning and bounded future guidance."""
```

Create `src/agent_hub/cognitive/types.py` with enums and Pydantic models:

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperienceKind(StrEnum):
    USER_PREFERENCE = "user_preference"
    PROJECT_FACT = "project_fact"
    WORKFLOW_STRATEGY = "workflow_strategy"
    ERROR_HANDLING = "error_handling"
    UI_RULE = "ui_rule"
    COMMUNICATION_STYLE = "communication_style"
    TOOLING_STRATEGY = "tooling_strategy"
    DOMAIN_PATTERN = "domain_pattern"


class ExperienceStatus(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class CognitiveEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_type: str = Field(min_length=1, max_length=48)
    source_id: str = Field(min_length=1, max_length=128)
    note: str = Field(min_length=1, max_length=512)


class ExperienceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: UUID
    tenant_id: UUID
    user_id: UUID
    kind: ExperienceKind
    status: ExperienceStatus
    summary: str = Field(min_length=1, max_length=240)
    lesson: str = Field(min_length=1, max_length=1200)
    strategy: str = Field(min_length=1, max_length=1200)
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[CognitiveEvidence, ...] = ()
    contradictions: tuple[CognitiveEvidence, ...] = ()
    source_run_ids: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    applies_to_modes: tuple[str, ...] = ()
    applies_to_agents: tuple[str, ...] = ()
    use_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    last_used_at: datetime | None
    last_verified_at: datetime | None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    @property
    def active_for_runtime(self) -> bool:
        return self.status in {ExperienceStatus.CONFIRMED, ExperienceStatus.ACTIVE}

    @field_validator("summary", "lesson", "strategy")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        if value != value.strip() or any(ord(ch) < 32 and ch not in "\n\t" for ch in value):
            raise ValueError("cognitive text must be bounded printable text")
        return value

    @field_validator("tags", "applies_to_modes", "applies_to_agents", "source_run_ids", "source_memory_ids")
    @classmethod
    def _clean_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if not item or item != item.strip() or len(item) > 128:
                raise ValueError("cognitive identifiers must be bounded non-empty strings")
        return value

    @model_validator(mode="after")
    def _validate_counts_and_times(self) -> ExperienceRecord:
        if self.success_count + self.failure_count > self.use_count:
            raise ValueError("success and failure counts cannot exceed use count")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        return self
```

- [ ] **Step 4: Run the model tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\cognitive\test_types.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/agent_hub/cognitive/__init__.py src/agent_hub/cognitive/types.py tests/unit/cognitive/test_types.py
git -c user.name=zhangzhimiao -c user.email=41898282+zhangzhimiao1994@users.noreply.github.com commit -m "feat: add cognitive experience types"
```

---

### Task 2: Experience Store, Persistence, and Service

**Files:**
- Create: `src/agent_hub/cognitive/repository.py`
- Create: `src/agent_hub/cognitive/service.py`
- Test: `tests/unit/cognitive/test_service.py`

- [ ] **Step 1: Write failing service tests**

Create `tests/unit/cognitive/test_service.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_hub.cognitive.repository import InMemoryExperienceRepository
from agent_hub.cognitive.service import ExperienceService
from agent_hub.cognitive.types import CognitiveEvidence, ExperienceKind, ExperienceStatus


@pytest.mark.asyncio
async def test_create_candidate_experience_is_not_runtime_active() -> None:
    service = ExperienceService(InMemoryExperienceRepository(), now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()

    record = await service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.ERROR_HANDLING,
        summary="reviewer 超时时先压缩上下文再分块审查。",
        lesson="大输入会让 reviewer 步骤超时。",
        strategy="先压缩输入，再拆分审查。",
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="timeout"),),
        tags=("reviewer", "timeout"),
        applies_to_modes=("dispatch", "hybrid"),
        applies_to_agents=("quality_reviewer",),
    )

    assert record.status is ExperienceStatus.CANDIDATE
    assert record.active_for_runtime is False


@pytest.mark.asyncio
async def test_confirm_experience_makes_it_runtime_active() -> None:
    repository = InMemoryExperienceRepository()
    service = ExperienceService(repository, now=lambda: datetime.now(UTC))
    tenant_id = uuid4()
    user_id = uuid4()
    record = await service.create_candidate(
        tenant_id=tenant_id,
        user_id=user_id,
        kind=ExperienceKind.USER_PREFERENCE,
        summary="用户偏好先给结论。",
        lesson="用户多次要求先给结论。",
        strategy="回答先给结论，再给关键证据。",
        evidence=(CognitiveEvidence(source_type="feedback", source_id="fb-1", note="explicit confirmation"),),
    )

    confirmed = await service.confirm(record.id, tenant_id=tenant_id, user_id=user_id)

    assert confirmed.status is ExperienceStatus.CONFIRMED
    assert confirmed.active_for_runtime is True
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\cognitive\test_service.py -q
```

Expected: fails because repository/service do not exist.

- [ ] **Step 3: Implement repository and service**

Create `src/agent_hub/cognitive/repository.py` with async in-memory `upsert`, `get`, `list_for_user`, and `delete`.

Also add a persistent repository class that stores records in `AdminResourceRow` using `kind="hermes"` and `resource_id` prefixes. Do not add new `kind` values in this task.

Create `src/agent_hub/cognitive/service.py` with:

- `create_candidate()`
- `confirm()`
- `reject()`
- `record_use_outcome()`
- `list_records()`

Use immutable updates through `record.model_copy(update=...)`.

- [ ] **Step 4: Run service tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\cognitive\test_types.py tests\unit\cognitive\test_service.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/agent_hub/cognitive/repository.py src/agent_hub/cognitive/service.py tests/unit/cognitive/test_service.py
git -c user.name=zhangzhimiao -c user.email=41898282+zhangzhimiao1994@users.noreply.github.com commit -m "feat: add cognitive experience store"
```

---

### Task 3: Reflection Engine

**Files:**
- Modify: `src/agent_hub/cognitive/types.py`
- Create: `src/agent_hub/cognitive/reflection.py`
- Test: `tests/unit/cognitive/test_reflection.py`

- [ ] **Step 1: Write failing reflection tests**

Create `tests/unit/cognitive/test_reflection.py`:

```python
from __future__ import annotations

from agent_hub.cognitive.reflection import reflect_from_feedback


def test_user_correction_creates_counterfactual_reflection() -> None:
    reflection = reflect_from_feedback(
        tenant_id=uuid4(),
        user_id=uuid4(),
        source_run_id="run-1",
        user_feedback="不是让你甩锅给我，发现问题应该先解决。",
        outcome="negative",
        now=lambda: datetime.now(UTC),
    )

    assert reflection.trigger == "user_correction"
    assert "先解决" in reflection.counterfactual
    assert reflection.confidence >= 0.6


def test_user_satisfaction_creates_positive_pattern() -> None:
    reflection = reflect_from_feedback(
        tenant_id=uuid4(),
        user_id=uuid4(),
        source_run_id="run-2",
        user_feedback="这样可以，继续按这个方式处理。",
        outcome="positive",
        now=lambda: datetime.now(UTC),
    )

    assert reflection.trigger == "user_satisfaction"
    assert reflection.positive_patterns
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\cognitive\test_reflection.py -q
```

Expected: fails because reflection module does not exist.

- [ ] **Step 3: Implement deterministic reflection**

Add `ReflectionRecord` to `types.py` and `reflect_from_feedback()` to `reflection.py`.

Rules:

- correction/rejection wording creates `trigger="user_correction"` or `user_rejection`
- satisfaction/approval wording creates `trigger="user_satisfaction"` or `user_confirmation`
- negative feedback must include a counterfactual string
- positive feedback must include positive patterns
- output is bounded and does not include secrets
- function input includes `tenant_id`, `user_id`, and a `now` callable so tests can verify isolation and timestamps

- [ ] **Step 4: Run reflection tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\cognitive\test_reflection.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/agent_hub/cognitive/types.py src/agent_hub/cognitive/reflection.py tests/unit/cognitive/test_reflection.py
git -c user.name=zhangzhimiao -c user.email=41898282+zhangzhimiao1994@users.noreply.github.com commit -m "feat: add cognitive reflection engine"
```

---

### Task 4: Experience Router and Bounded Injection

**Files:**
- Create: `src/agent_hub/cognitive/router.py`
- Modify: `src/agent_hub/runtime/hermes_context.py`
- Test: `tests/unit/cognitive/test_router.py`
- Test: `tests/unit/runtime/test_hermes_context.py`

- [ ] **Step 1: Write failing router tests**

Create `tests/unit/cognitive/test_router.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from agent_hub.cognitive.router import route_experiences
from agent_hub.cognitive.types import CognitiveEvidence, ExperienceKind, ExperienceRecord, ExperienceStatus


def _experience(summary: str, *, status: ExperienceStatus = ExperienceStatus.CONFIRMED) -> ExperienceRecord:
    now = datetime.now(UTC)
    return ExperienceRecord(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        kind=ExperienceKind.ERROR_HANDLING,
        status=status,
        summary=summary,
        lesson="reviewer timeout",
        strategy="compress then split",
        confidence=0.86,
        evidence=(CognitiveEvidence(source_type="run", source_id="run-1", note="timeout"),),
        contradictions=(),
        source_run_ids=("run-1",),
        source_memory_ids=(),
        tags=("reviewer", "timeout", "审查"),
        applies_to_modes=("dispatch", "hybrid"),
        applies_to_agents=("quality_reviewer",),
        use_count=2,
        success_count=2,
        failure_count=0,
        last_used_at=None,
        last_verified_at=now,
        version=1,
        created_at=now,
        updated_at=now,
    )


def test_router_selects_relevant_confirmed_experience() -> None:
    result = route_experiences(
        request="审查输出时 reviewer 又超时了",
        mode="dispatch",
        agent_ids=("quality_reviewer",),
        experiences=(_experience("reviewer 超时时先压缩上下文再分块审查。"),),
    )

    assert result.selected[0].summary == "reviewer 超时时先压缩上下文再分块审查。"
    assert result.selected[0].score >= 0.7


def test_router_skips_unconfirmed_experience() -> None:
    result = route_experiences(
        request="审查输出时 reviewer 又超时了",
        mode="dispatch",
        agent_ids=("quality_reviewer",),
        experiences=(_experience("候选经验", status=ExperienceStatus.CANDIDATE),),
    )

    assert result.selected == ()
    assert result.skipped[0].reason == "经验尚未确认"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\cognitive\test_router.py -q
```

Expected: fails because router does not exist.

- [ ] **Step 3: Implement router**

Create `route_experiences()` that:

- filters non-active experiences
- scores tags, mode, agent id, confidence, evidence count, success/failure ratio
- skips contradictory or low-confidence items with reasons
- returns top 3 selected and up to 5 skipped

- [ ] **Step 4: Keep prompt context rendering compatible**

Convert selected confirmed experiences into the existing `routing_decision.hermes.injected_memories` list. `hermes_memory_context_text()` should not need a new top-level payload shape in the first slice.

The renderer must:

- keep the existing 900-byte bound
- ignore unconfirmed or scheduler/runtime observation items
- include summary, type, target, and reason from the compatible injection item

- [ ] **Step 5: Run router and context tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\cognitive\test_router.py tests\unit\runtime\test_hermes_context.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/agent_hub/cognitive/router.py src/agent_hub/runtime/hermes_context.py tests/unit/cognitive/test_router.py tests/unit/runtime/test_hermes_context.py
git -c user.name=zhangzhimiao -c user.email=41898282+zhangzhimiao1994@users.noreply.github.com commit -m "feat: route cognitive experiences into runtime context"
```

---

### Task 5: Admin API and Hermes UI

**Files:**
- Modify: `src/agent_hub/api/routers/admin.py`
- Modify: `web/src/api/client.ts`
- Modify: `web/src/pages/HermesPage.tsx`
- Modify: `web/src/styles.css`
- Test: `tests/api/test_admin_resources.py`
- Test: `web/src/pages/OperationalPages.test.tsx`

- [ ] **Step 1: Write failing API tests**

Add tests that verify:

- `GET /api/v1/admin/cognitive/experiences` lists records
- `POST /api/v1/admin/cognitive/experiences/{id}/confirm` confirms a candidate
- unconfirmed candidates are not returned as runtime-active context
- persistent records are stored under `AdminResourceRow.kind = "hermes"` with `resource_id` starting `cognitive_experience:`

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\api\test_admin_resources.py -k cognitive -q
```

Expected: fails because the endpoints do not exist.

- [ ] **Step 2: Implement API schemas and endpoints**

In `admin.py`, add response/request models and endpoints:

- `GET /cognitive/experiences`
- `POST /cognitive/experiences/{experience_id}/confirm`
- `POST /cognitive/experiences/{experience_id}/reject`
- `DELETE /cognitive/experiences/{experience_id}`

Use permissions:

- read: `hermes:read`
- confirm/reject/delete: `hermes:write`

- [ ] **Step 3: Write failing UI tests**

Add `OperationalPages.test.tsx` assertions:

- Hermes page shows `经验候选`
- Chinese one-sentence summary is visible
- confirming calls the new confirm endpoint
- the run drawer shows compact `经验注入：N 条`
- detail drawer opens by clicking the row and closes by backdrop click

Run:

```powershell
npm --prefix web test -- OperationalPages.test.tsx -t "经验"
```

Expected: fails because UI does not fetch or render cognitive experience records.

- [ ] **Step 4: Implement frontend schemas and UI**

In `client.ts`, add `CognitiveExperienceSchema` and methods.

In `HermesPage.tsx`, add a compact view inside the existing Hermes page rather than a new top-level module.

In `RunsPage.tsx`, reuse existing Hermes memory row styling where possible.

- [ ] **Step 5: Run API and UI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\api\test_admin_resources.py -k cognitive -q
npm --prefix web test -- OperationalPages.test.tsx -t "经验"
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/agent_hub/api/routers/admin.py web/src/api/client.ts web/src/pages/HermesPage.tsx web/src/pages/RunsPage.tsx web/src/styles.css tests/api/test_admin_resources.py web/src/pages/OperationalPages.test.tsx
git -c user.name=zhangzhimiao -c user.email=41898282+zhangzhimiao1994@users.noreply.github.com commit -m "feat: expose cognitive experiences in Hermes"
```

---

### Task 6: Run Integration, Verification, Deploy, and Handoff

**Files:**
- Modify: `src/agent_hub/runs/service.py`
- Modify: `src/agent_hub/hermes/advisor.py`
- Modify: `src/agent_hub/api/routers/admin.py`
- Modify: `HANDOFF.md` locally only
- Test: `tests/unit/runs/test_conversation_mode.py`
- Test: `tests/unit/hermes/test_advisor.py`

- [ ] **Step 1: Write failing run integration tests**

Add tests verifying:

- completed run outcome can create a candidate experience when meaningful evidence exists
- failed reviewer timeout creates an error-handling candidate
- normal successful direct Q&A does not create an experience
- cognitive service failure does not fail run completion

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\runs\test_conversation_mode.py -k cognitive -q
```

Expected: fails because run service does not call cognitive service.

- [ ] **Step 2: Wire run service**

Add a `CognitiveAdvisorProtocol` or service protocol to `RunService`.

Call it:

- after terminal run outcome is known
- before Hermes runtime advice when retrieving active experiences
- inside safe timeout/error guards

Use the same resilience pattern as `_safe_hermes_advice()` and `_safe_record_hermes_outcome()`.

- [ ] **Step 3: Verify persistent storage adapter**

Confirm the persistent repository from Task 2 stores all cognitive records under existing `AdminResourceRow.kind = "hermes"` using `cognitive_*` resource-id prefixes. Do not add destructive migrations in this slice.

- [ ] **Step 4: Run focused backend tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\cognitive tests\unit\hermes\test_advisor.py tests\unit\runs\test_conversation_mode.py tests\unit\runtime\test_hermes_context.py tests\api\test_admin_resources.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Run broad local verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy --strict src tests
.\.venv\Scripts\python.exe -m pytest tests\unit tests\api tests\contracts -q -p no:cacheprovider
npm --prefix web run lint
npm --prefix web test -- --run
npm --prefix web run build
git diff --check
```

Expected: all checks pass. Existing Vite large chunk warning is acceptable only if unchanged.

- [ ] **Step 6: Deploy before GitHub push if runtime behavior changed**

For runtime-affecting implementation, deploy to `prod-web-01` before pushing GitHub:

```text
Host: prod-web-01
Current pointer: /opt/agent-hub/current
Release dir pattern: /opt/agent-hub/releases/YYYYMMDD-<sha>-cognitive-experience
```

Production probe must verify:

- `/health/live`
- `/health/ready`
- create candidate experience
- confirm candidate
- submit a run that retrieves the confirmed experience
- inspect run detail for selected/skipped experience explanation

- [ ] **Step 7: Push and check GitHub**

Push only to:

```text
https://github.com/zhangzhimiao1994/CubeAgent
```

Then check the triggered GitHub run. If it fails, retrieve details, fix locally, verify, commit, push, and repeat.

- [ ] **Step 8: Update local handoff**

Append to local `HANDOFF.md`:

- commit hash
- production release path
- verification commands
- production probe result
- remaining risks
- explicit reminder that this repository still has no harness/Vibe Coding execution

Do not commit `HANDOFF.md`.
