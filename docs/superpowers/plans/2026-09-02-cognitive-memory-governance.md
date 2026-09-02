# Cognitive Memory Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long-running CubeAgent memory become more reliable and compact over time, without increasing prompt size or mixing noisy memories into decisions.

**Architecture:** Extend the existing Cognitive Layer instead of replacing Hermes/Memory. Add Strategy Library behavior, governance, metacognition, reflection, outcome verification, conflict, calibration, decay, and hierarchical summary services around the existing repositories and router. Runtime integration is limited to bounded advice and terminal learning hooks.

**Tech Stack:** Python 3.13, Pydantic models, SQLAlchemy-backed `AdminResourceRow` storage via existing repositories, pytest, ruff, mypy.

---

### Task 1: Strategy Library primitives inside CognitiveStateService

**Files:**
- Modify: `src/agent_hub/cognitive/types.py`
- Modify: `src/agent_hub/cognitive/repository.py`
- Modify: `src/agent_hub/cognitive/service.py`
- Modify: `src/agent_hub/cognitive/__init__.py`
- Test: `tests/unit/cognitive/test_types.py`
- Test: `tests/unit/cognitive/test_repository.py`
- Test: `tests/unit/cognitive/test_service.py`

- [x] Write failing tests for strategy records, repository visibility, strategy candidate selection, confirmation, and outcome-based confidence changes.
- [x] Run the new tests and verify missing imports/classes fail.
- [x] Implement `StrategyRecord` and repository persistence mapping under `cognitive_strategy:`.
- [x] Implement Strategy Library methods on `CognitiveStateService` for candidate creation, confirm/reject, task selection, and use outcome updates. Do not expose a separate `StrategyService` abstraction.
- [x] Run targeted cognitive tests.
- [x] Commit as `feat: add cognitive strategy governance`.

### Task 2: Outcome verification, conflict resolution, confidence calibration, and anti-learning

**Files:**
- Create: `src/agent_hub/cognitive/governance.py`
- Create: `src/agent_hub/cognitive/verifier.py`
- Modify: `src/agent_hub/cognitive/service.py`
- Modify: `src/agent_hub/cognitive/__init__.py`
- Test: `tests/unit/cognitive/test_governance.py`
- Test: `tests/unit/cognitive/test_verifier.py`
- Test: `tests/unit/cognitive/test_service.py`

- [x] Write failing tests for outcome verdicts (`success`, `partial`, `failure`, `insufficient_evidence`), unresolved conflicts, supported winners, confidence calibration, and automatic degradation of repeatedly failing Experience/Strategy records.
- [x] Run the new tests and verify missing module/service failures.
- [x] Implement `OutcomeVerifier`, `ConflictResolutionEngine`, `ConfidenceCalibrationService`, `AntiLearningService`, and memory-tier scoring types.
- [x] Wire anti-learning helpers into existing experience/strategy outcome update behavior.
- [x] Run targeted governance/service tests.
- [x] Commit as `feat: add cognitive governance controls`.

### Task 3: Metacognition, working set, and hierarchical memory

**Files:**
- Create: `src/agent_hub/cognitive/metacognition.py`
- Create: `src/agent_hub/cognitive/hierarchy.py`
- Modify: `src/agent_hub/context/builder.py`
- Modify: `src/agent_hub/cognitive/__init__.py`
- Test: `tests/unit/cognitive/test_metacognition.py`
- Test: `tests/unit/cognitive/test_hierarchy.py`
- Test: `tests/unit/context/test_builder.py`
- Test: `tests/unit/memory/test_governance.py`

- [x] Write failing tests for simple-task lightweight gate, complex-task advanced gate, stale/conflict detection, bounded working set, hot/warm/cold/archive tiering, and hierarchical summary consolidation.
- [x] Run the new tests and verify missing module/function failures.
- [x] Implement `MetacognitionService`, `WorkingSetBuilder`, memory tiering, archive, and source-linked consolidation.
- [x] Extend context construction to keep working memory bounded ahead of long-term memory and skip archived memory in active retrieval.
- [x] Run targeted cognitive/memory/context tests plus ruff, strict mypy, and diff check.
- [x] Commit as `feat: add cognitive metacognition and memory hierarchy`.

### Task 4: Reflection pipeline, Outcome Critic, and runtime terminal hook

**Files:**
- Modify: `src/agent_hub/cognitive/reflection.py`
- Create: `src/agent_hub/cognitive/pipeline.py`
- Modify: `src/agent_hub/runtime/worker.py`
- Modify: `src/agent_hub/app.py`
- Test: `tests/unit/cognitive/test_reflection.py`
- Test: `tests/unit/cognitive/test_pipeline.py`
- Test: `tests/unit/runs/test_terminal_hooks.py`

- [x] Write failing tests for success reflection, failure reflection, outcome critic verdict persistence, candidate experience creation, nested artifact output detection, strategy outcome calibration, and terminal hook non-blocking behavior.
- [x] Run the new tests and verify current reflection is too narrow or pipeline missing.
- [x] Implement `ReflectionEngine`, `OutcomeCritic`, and `CognitiveLearningPipeline`.
- [x] Register the pipeline as an additional terminal hook in app and worker wiring.
- [x] Run targeted cognitive, verifier, worker-wiring, and run-hook tests plus ruff, strict mypy, and diff check.
- [x] Commit as `feat: connect cognitive reflection pipeline`.

### Task 5: API, verification, and handoff

**Files:**
- Modify: `src/agent_hub/api/routers/admin.py`
- Modify: `README.md`
- Modify local-only: `HANDOFF.md`
- Test: `tests/api/test_admin_resources.py`

- [x] Write failing API tests for listing strategies/reflections, confirming/rejecting strategies, and exposing governance metadata.
- [x] Implement minimal admin endpoints without adding UI dependency.
- [x] Run full backend verification: cognitive tests, Hermes advisor tests, admin API tests, ruff, mypy.
- [x] Ensure `HANDOFF.md` is local-only and not staged.
- [x] Push to `https://github.com/zhangzhimiao1994/CubeAgent`.
- [x] Check GitHub Actions until success.
- [x] Skip production deployment for this slice per user instruction.
