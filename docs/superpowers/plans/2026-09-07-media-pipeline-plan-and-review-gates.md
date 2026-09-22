# Media Pipeline Plan And Review Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve long-lived AI video production plans without reserving execution slots, and pause downstream media generation at user-review gates.

**Architecture:** Store `media_pipeline_plan` as safe routing metadata and project it into conversation history. Add `DispatchStep.requires_user_review` so runtime can checkpoint and request `runtime_artifact_review` before dependent stages run. Approving the review requeues the same run from checkpoint.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, SQLAlchemy async, existing Agent Hub runtime contracts.

**Spec:** `docs/superpowers/specs/2026-09-07-ai-video-production-pipeline-design.md`

## Global Constraints

- A long-lived media plan is not an execution slot and must not reserve model capacity.
- Script generation can complete normally while recording the reusable plan.
- Review gates apply only to intermediate artifacts that feed later generation or composition.
- Approval must reuse `RunStatus.WAITING_APPROVAL` without mixing runtime artifact reviews with capability approval fingerprints.
- Existing `compose_video` behavior remains unchanged.

---

### Task 1: Long-Lived Plan Context

**Files:**
- Modify: `src/agent_hub/runs/repository.py`
- Modify: `src/agent_hub/runs/service.py`
- Test: `tests/unit/runs/test_runtime_context_policy.py`

**Interfaces:**
- Consumes: `ConversationContextItem`.
- Produces: `routing_decision` on conversation context items and `MEDIA_PIPELINE_PLAN` history lines.

- [x] **Step 1: Write the failing test**

```python
def test_conversation_history_includes_media_pipeline_plan() -> None:
    ...
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -c "...test_conversation_history_includes_media_pipeline_plan()..."`
Expected: FAIL because `ConversationContextItem` lacks `routing_decision`.

- [x] **Step 3: Write minimal implementation**

Add `routing_decision` to `ConversationContextItem` and render safe plan fields into history.

- [x] **Step 4: Run test to verify it passes**

Expected: PASS.

### Task 2: Script Request Plan Recording

**Files:**
- Modify: `src/agent_hub/runs/service.py`
- Test: `tests/unit/runs/test_conversation_mode.py`

**Interfaces:**
- Consumes: submit request text.
- Produces: `media_pipeline_plan` with empty `execution_slots`.

- [x] **Step 1: Write the failing test**

```python
async def test_script_request_records_long_lived_media_pipeline_plan_without_slots() -> None:
    ...
```

- [x] **Step 2: Run test to verify it fails**

Expected: FAIL because no plan is recorded.

- [x] **Step 3: Write minimal implementation**

Add `_media_pipeline_plan_for_request()` and merge it into submit routing metadata.

- [x] **Step 4: Run test to verify it passes**

Expected: PASS.

### Task 3: Runtime Review Gate

**Files:**
- Modify: `src/agent_hub/runtime/crew/plan.py`
- Modify: `src/agent_hub/runtime/crew/adapter.py`
- Modify: `src/agent_hub/runtime/defaults.py`
- Test: `tests/unit/runtime/crew/test_plan.py`
- Test: `tests/unit/runtime/crew/test_adapter_failure_reason.py`
- Test: `tests/unit/runtime/test_configured_runtime.py`

**Interfaces:**
- Consumes: `DispatchStep.requires_user_review`.
- Produces: `approval.requested` event with `approval_kind=runtime_artifact_review`.

- [x] **Step 1: Write failing tests**

```python
def test_step_user_review_gate_round_trips_and_defaults_to_false() -> None:
    ...

async def test_user_review_gate_requests_approval_before_downstream_step() -> None:
    ...
```

- [x] **Step 2: Run tests to verify they fail**

Expected: schema rejects `requires_user_review`; runtime emits no approval.

- [x] **Step 3: Write minimal implementation**

Add the step field, mark intermediate media steps, save checkpoint, emit approval, and stop runtime.

- [x] **Step 4: Run tests to verify they pass**

Expected: PASS.

### Task 4: Approval Resume API

**Files:**
- Modify: `src/agent_hub/api/routers/runs.py`
- Modify: `src/agent_hub/runs/service.py`
- Modify: `src/agent_hub/runs/repository.py`
- Test: `tests/api/test_runs_api.py`
- Test: `tests/unit/runs/test_temporary_agent.py`
- Test: `tests/unit/runs/test_terminal_hooks.py`

**Interfaces:**
- Consumes: `POST /api/v1/runs/{run_id}/artifact-reviews/{approval_id}/approve`.
- Produces: queued run with `approved_artifacts` metadata and cleared pending approval fields.

- [x] **Step 1: Write failing tests**

```python
def test_approve_artifact_review_queues_waiting_run_safely() -> None:
    ...

async def test_user_can_approve_runtime_artifact_review_and_continue() -> None:
    ...
```

- [x] **Step 2: Run tests to verify they fail**

Expected: endpoint and service method are missing.

- [x] **Step 3: Write minimal implementation**

Add route, service method, and repository update/requeue method.

- [x] **Step 4: Run tests to verify they pass**

Expected: PASS.

### Task 5: Verification And Handoff

**Files:**
- Modify: `HANDOFF.md`

**Interfaces:**
- Consumes: completed implementation and test output.
- Produces: current-state handoff.

- [x] **Step 1: Run focused tests**

Run the modified unit/API tests.

- [x] **Step 2: Run static checks**

Run `ruff check` on modified Python files.

- [x] **Step 3: Update handoff**

Record changes, verification, risks, and next steps.

- [x] **Step 4: Commit and push**

Commit the branch, push to GitHub, and check PR checks.
