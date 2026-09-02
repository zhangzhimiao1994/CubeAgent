# Cognitive Memory Maintenance V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep long-running Agent memory accurate, condensed, and bounded instead of letting memory volume grow without control.

**Architecture:** Add a policy-driven memory maintenance layer on top of the existing MemoryService, not a second memory framework. The layer scores records, prefers compression/archive before physical deletion, protects root/core/locked records, and exposes deterministic maintenance decisions that can be tested and later scheduled.

**Tech Stack:** Python 3.12, Pydantic, existing `agent_hub.memory` and `agent_hub.cognitive` modules, pytest, ruff, mypy.

---

### Task 1: Memory retention decision model

**Files:**
- Modify: `src/agent_hub/memory/types.py`
- Test: `tests/unit/memory/test_maintenance.py`

- [x] Add `MemoryRetentionAction` with `keep`, `compress`, `cool_down`, `archive`, `tombstone`, and `purge`.
- [x] Add `MemoryRetentionPolicy` with bounded defaults:
  - `stale_candidate_days=30`
  - `cold_archive_days=180`
  - `tombstone_purge_days=90`
  - `archive_purge_days=365`
  - `min_retention_score=0.22`
  - `compress_after_source_count=3`
  - `max_active_records_per_user=1000`
- [x] Add `MemoryRetentionDecision` containing `memory_id`, `action`, `score`, `reason`, `protected`.
- [x] Test strict validation and sane defaults.

### Task 2: Repository physical delete support

**Files:**
- Modify: `src/agent_hub/memory/repository.py`
- Test: `tests/unit/memory/test_maintenance.py`

- [x] Add async `delete(memory_id: UUID) -> bool`.
- [x] Verify physical delete removes records but not audit history.
- [x] Use this only for old tombstones/archives, never active records.

### Task 3: Policy-driven maintenance service

**Files:**
- Create: `src/agent_hub/memory/maintenance.py`
- Modify: `src/agent_hub/memory/__init__.py`
- Test: `tests/unit/memory/test_maintenance.py`

- [x] Implement `MemoryMaintenanceService.evaluate(record, policy)`:
  - return `keep` for locked/core records;
  - return `purge` for tombstones older than `tombstone_purge_days`;
  - return `purge` for archives older than `archive_purge_days` if not locked/core;
  - return `archive` for old cold records;
  - return `tombstone` for very low retention score;
  - return `cool_down` for stale but still useful active records;
  - return `keep` otherwise.
- [x] Implement retention score:
  - starts from confidence;
  - increases with recall count, heat, source links, confirmed/root/core signal;
  - decreases with age, low heat, contradiction metadata, and candidate status metadata.
- [x] Implement `maintain(apply=False)` dry-run by default, and `maintain(apply=True)` that applies safe actions.
- [x] Add audit events for applied archive/tombstone/purge decisions.

### Task 4: Compression-first consolidation

**Files:**
- Modify: `src/agent_hub/memory/maintenance.py`
- Modify: `src/agent_hub/memory/service.py` only if existing `consolidate()` needs a safe wrapper.
- Test: `tests/unit/memory/test_maintenance.py`

- [x] Group repeated or related normal memories by tenant/user/project/conversation/category.
- [x] When a group has at least `compress_after_source_count` active normal memories, create or request a summary memory before archiving sources.
- [x] Keep the summary source ids and archive original records with `archive_reason=consolidated_into:<summary_id>`.
- [x] Never compress locked/core records without explicit user confirmation.

### Task 5: Context injection budget guard

**Files:**
- Modify: `src/agent_hub/context/*` or `src/agent_hub/cognitive/context_router.py` after locating the active router.
- Test: `tests/unit/cognitive/test_context_router.py`

- [x] Ensure Hot memories/experiences rank before Warm.
- [x] Ensure Cold is retrieved only on strong relevance.
- [x] Ensure Archive/Tombstone never inject into ordinary runtime context.
- [x] Add per-source and total context budget tests.

### Task 6: Conflict and confidence maintenance integration

**Files:**
- Modify: `src/agent_hub/cognitive/governance.py`
- Test: `tests/unit/cognitive/test_governance.py`

- [x] Apply `ConfidenceCalibrationService` to stale beliefs, experiences, strategies, and skill candidates.
- [x] Keep conflicting records as `UNRESOLVED` unless evidence gap is sufficient.
- [x] Ensure important inferred records remain candidates until verified by outcome or user confirmation.

### Task 7: End-to-end learning and injection test

**Files:**
- Test: `tests/unit/cognitive/test_pipeline.py`

- [x] Generate a candidate experience from an interaction.
- [x] Confirm or activate it.
- [x] Run context routing and verify it is injected only when relevant.
- [x] Record a failed outcome and verify confidence drops.
- [x] Run maintenance and verify repeated raw memories are condensed or archived.

### Task 8: Documentation and handoff

**Files:**
- Modify: `README.md`
- Modify local-only: `HANDOFF.md`

- [x] Document memory lifecycle: Candidate → Active/Hot → Warm → Cold → Archive/Tombstone → Purge.
- [x] Document deletion policy: compress first, archive second, purge only after retention windows or explicit deletion.
- [x] Update local handoff with verification, risks, and remaining UI work.
