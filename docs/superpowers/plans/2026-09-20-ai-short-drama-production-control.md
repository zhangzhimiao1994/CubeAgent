# AI Short Drama Production Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-control layer that separates character identity from costumes/looks, preserves scene continuity, injects director/producer guidance into asset prompts, and blocks or retries assets that drift from the intended character/state.

**Architecture:** Add a small `agent_hub.runtime.production` module that owns structured production concepts and prompt sections. Existing crew/runtime code remains the caller: it extracts script context, asks production helpers for character identity/look/continuity plans, passes metadata into multimedia generation, and uses existing visual review/retry paths.

**Tech Stack:** Python 3.12, Pydantic/dataclass-style immutable models, existing runtime contracts, existing multimedia generation/review provider interfaces, pytest-compatible unit tests plus direct compile/assertion checks when pytest hangs.

**Spec:** `docs/superpowers/specs/2026-09-20-ai-short-drama-production-control-design.md`

## Global Constraints

- Do not create a second agent framework, model router, memory system, task system, or multimedia router.
- Character Identity defines "who this person is"; Look / Costume defines "what this person is wearing now".
- Character generation must compose `Character Identity + Look / Costume + Pose + Scene + Shot Prompt`.
- Costume reference images may transfer clothing silhouette, color, material, texture, accessories, and wearing method only.
- Consecutive scenes inherit the previous look unless the script indicates wardrobe/state change.
- Rejecting one identity/look/storyboard/video item retries only the affected output and dependent downstream outputs.
- Human-visible artifacts must show enough metadata to tell which Character ID, Look ID, and production category they represent.
- The MVP ships deterministic tests/fake validation first and can use vision-model identity validation through an adapter.

## Review Focus

- Multi-role scripts: every important character gets a separate Character ID and one role-lock asset.
- Costume changes: a scene with a time jump or explicit wardrobe change switches Look ID without rewriting identity.
- Continuous scenes: location changes in the same time span keep the same Look ID.
- Costume reference leakage: reference-image prompts forbid inheriting the model face/body/hair/age.
- Partial retry: a failed look or identity asset preserves passed artifacts and regenerates only failed labels.

---

### Task 1: Production Models And Prompt Builder

**Files:**
- Create: `src/agent_hub/runtime/production/__init__.py`
- Create: `src/agent_hub/runtime/production/schemas.py`
- Create: `src/agent_hub/runtime/production/prompts.py`
- Test: `tests/unit/runtime/production/test_prompts.py`

**Interfaces:**
- Produces: `CharacterIdentity`, `CharacterLook`, `SceneCharacterState`, `ProductionDirection`, `IdentityValidationResult`
- Produces: `build_identity_lock_prompt(identity, look=None, pose=None, scene=None, shot=None) -> str`
- Produces: `production_metadata_for_label(label: str, prompt: str) -> dict[str, str]`

- [ ] **Step 1: Write failing tests for identity/look separation**

```python
from agent_hub.runtime.production import CharacterIdentity, CharacterLook, build_identity_lock_prompt


def test_identity_lock_prioritizes_face_before_costume():
    identity = CharacterIdentity(
        character_id="CHAR_LXM_001",
        display_name="林小满",
        role_type="女主",
        identity_prompt="26岁甜品店店长，圆杏眼，柔和下颌线，黑色低马尾",
        identity_traits=("圆杏眼", "柔和下颌线", "黑色低马尾"),
        forbidden_drift=("换脸", "变年龄", "换发际线"),
        master_reference_artifact_ids=(),
        embedding_refs=(),
    )
    look = CharacterLook(
        look_id="LOOK_001",
        character_id="CHAR_LXM_001",
        name="甜品店工作服",
        scene_applicability=("EP01_SC01",),
        costume_traits=("浅粉围裙", "白衬衫"),
        accessories=("胸牌",),
        hair_makeup_variations=("低马尾保持",),
        forbidden_identity_changes=("不要继承服装参考模特的脸",),
    )

    prompt = build_identity_lock_prompt(identity, look=look)

    assert prompt.index("IDENTITY LOCK") < prompt.index("LOOK / COSTUME")
    assert "CHAR_LXM_001" in prompt
    assert "只允许修改服装" in prompt
    assert "不要继承服装参考图中的脸" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/runtime/production/test_prompts.py::test_identity_lock_prioritizes_face_before_costume -v`

Expected: import failure because the production module does not exist yet.

- [ ] **Step 3: Implement immutable schema models and prompt builder**

Create frozen dataclasses with tuple fields and a prompt builder that emits sections in this order: `CHARACTER_ID`, `IDENTITY LOCK`, `LOOK / COSTUME`, `POSE`, `SCENE`, `SHOT`, `NEGATIVE_DRIFT_RULES`, `QC_EXPECTATIONS`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/unit/runtime/production/test_prompts.py -v`

Expected: all Task 1 tests pass, or if pytest hangs in this repo, run direct Python assertions importing the new module.

### Task 2: Script-Derived Production Plan

**Files:**
- Create: `src/agent_hub/runtime/production/planner.py`
- Test: `tests/unit/runtime/production/test_planner.py`

**Interfaces:**
- Consumes: `CharacterIdentity`, `CharacterLook`, `SceneCharacterState`
- Produces: `ProductionPlan(character_identities, looks, scene_states, direction)`
- Produces: `build_production_plan(script_text: str, *, request_text: str = "") -> ProductionPlan`

- [ ] **Step 1: Write failing tests for roles and continuity**

```python
from agent_hub.runtime.production import build_production_plan


def test_two_characters_get_separate_ids_and_assets():
    plan = build_production_plan("女主苏清月，医生。男主林渊，外卖员。第一场：雨夜相遇。")

    assert [item.display_name for item in plan.character_identities] == ["苏清月", "林渊"]
    assert len({item.character_id for item in plan.character_identities}) == 2
    assert all(not item.character_id.startswith("LOOK_") for item in plan.character_identities)


def test_time_jump_switches_look_without_changing_identity():
    plan = build_production_plan(
        "EP01_SC01：苏清月穿白大褂值夜班。"
        "EP01_SC02：同一夜，她继续追查。"
        "EP01_SC03：第二天回家，她换成居家服。"
    )

    states = [state for state in plan.scene_states if state.character_id == "CHAR_SQY_001"]
    assert states[0].look_id == states[1].look_id
    assert states[2].look_id != states[1].look_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/runtime/production/test_planner.py -v`

Expected: import failure or empty plan failure.

- [ ] **Step 3: Implement deterministic heuristic planner**

Extract Chinese role/name pairs such as `女主苏清月`, `男主林渊`, `医生苏清月`, and explicit names near role descriptions. Generate stable IDs from initials plus index, generate default identity traits from nearby snippets, generate default `LOOK_001`, and switch looks on markers like `第二天`, `回家`, `换成`, `晚宴`, `睡衣`, `雨夜`, `受伤`, `战斗`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/unit/runtime/production/test_planner.py -v`

Expected: all Task 2 tests pass, or direct assertions pass if pytest hangs.

### Task 3: Integrate Production Prompt Sections Into Asset Generation

**Files:**
- Modify: `src/agent_hub/runtime/crew/adapter.py`
- Test: `tests/unit/runtime/crew/test_adapter_failure_reason.py`
- Test: `tests/unit/runtime/crew/test_tool_contracts.py`

**Interfaces:**
- Consumes: `build_production_plan`
- Consumes: `build_identity_lock_prompt`
- Produces: full-production asset prompts containing `Character ID`, `Look ID`, `Identity Lock`, director/producer notes, and clean-board constraints.

- [ ] **Step 1: Add failing adapter tests**

Add assertions that `_direct_full_production_asset_prompts(...)` for a script containing `苏清月` and `林渊` emits:

```python
assert "Character ID" in prompt or "CHAR_" in prompt
assert "Identity Lock" in prompt or "IDENTITY LOCK" in prompt
assert "Look ID" in prompt or "LOOK_" in prompt
assert "导演/制片" in prompt or "Production Direction" in prompt
assert "不得继承服装参考图中的脸" in prompt
```

- [ ] **Step 2: Run the focused adapter tests and verify failure**

Run: `pytest tests/unit/runtime/crew/test_adapter_failure_reason.py -v`

Expected: missing production sections.

- [ ] **Step 3: Inject production plan into per-character and support asset prompts**

Build a production plan from `context.request`, `step.task`, and structural source text. For per-character assets, prepend the matching identity lock section. For support assets, prepend director/producer/art/continuity rules that say support assets must respect existing Character IDs and Look IDs.

- [ ] **Step 4: Run focused adapter tests**

Run: `pytest tests/unit/runtime/crew/test_adapter_failure_reason.py tests/unit/runtime/crew/test_tool_contracts.py -v`

Expected: all prompt contract assertions pass, or direct assertions pass if pytest hangs.

### Task 4: Artifact Metadata And Retry Preservation

**Files:**
- Modify: `src/agent_hub/capabilities/runtime.py`
- Modify: `src/agent_hub/api/routers/admin.py`
- Test: `tests/unit/capabilities/test_runtime_gateway.py`
- Test: `web/src/components/ArtifactFileCard.test.tsx`

**Interfaces:**
- Consumes: `production_metadata_for_label(label, prompt)`
- Produces: multimedia result items with `production_metadata`
- Produces: admin/UI expanded artifacts that surface `character_id`, `look_id`, `production_category`, and failed review state.

- [ ] **Step 1: Add failing metadata tests**

Assert a generated item labeled `角色锁定资产：苏清月` with a prompt containing `CHAR_SQY_001` and `LOOK_001` includes:

```python
assert item["production_metadata"]["character_id"] == "CHAR_SQY_001"
assert item["production_metadata"]["look_id"] == "LOOK_001"
assert item["production_metadata"]["production_category"] == "character_identity"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/unit/capabilities/test_runtime_gateway.py -v`

Expected: missing `production_metadata`.

- [ ] **Step 3: Store production metadata in result items and preserved artifacts**

Pass `generation_prompt` and label through the metadata extractor inside `_multimedia_artifact_result(...)`. Ensure `_preserved_multimedia_result_item(...)` keeps `production_metadata` and `visual_review`.

- [ ] **Step 4: Run focused runtime/UI tests**

Run: `pytest tests/unit/capabilities/test_runtime_gateway.py -v`

Run: `npm test -- --run web/src/components/ArtifactFileCard.test.tsx`

Expected: runtime metadata assertions pass and UI renders metadata/failure state.

### Task 5: Identity Validation Adapter Boundary

**Files:**
- Create: `src/agent_hub/runtime/production/validators.py`
- Modify: `src/agent_hub/app.py`
- Test: `tests/unit/runtime/production/test_validators.py`
- Test: `tests/unit/test_app_wiring.py`

**Interfaces:**
- Produces: `RuntimeIdentityValidator.review_image_identity(...)`
- Produces: fake deterministic validator for tests
- Produces: vision-model prompt text that compares generated asset to master identity when references exist.

- [ ] **Step 1: Add failing validator tests**

```python
from agent_hub.runtime.production import DeterministicIdentityValidator


def test_low_identity_score_blocks_asset():
    validator = DeterministicIdentityValidator(scores={"artifact-1": 0.41})
    result = validator.validate("CHAR_LXM_001", "artifact-1", threshold=0.78)

    assert not result.passed
    assert result.identity_score == 0.41
    assert result.issues
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/unit/runtime/production/test_validators.py -v`

Expected: validator module missing.

- [ ] **Step 3: Implement adapter and wire prompt text**

Keep this non-blocking unless `master_reference_artifact_ids` or configured provider references exist. When unavailable, return `identity_score=None`, `passed=False` only for explicit mismatch signals from visual review; otherwise mark provider `not_configured`.

- [ ] **Step 4: Run validator/app wiring tests**

Run: `pytest tests/unit/runtime/production/test_validators.py tests/unit/test_app_wiring.py -v`

Expected: validator and app prompt wiring assertions pass.

### Task 6: Video/Storyboard Continuity And Frame QC Hooks

**Files:**
- Modify: `src/agent_hub/runtime/crew/adapter.py`
- Modify: `src/agent_hub/capabilities/runtime.py`
- Test: `tests/unit/runtime/crew/test_adapter_failure_reason.py`
- Test: `tests/unit/capabilities/test_runtime_gateway.py`

**Interfaces:**
- Consumes: `SceneCharacterState`
- Produces: storyboard/video prompts that include scene state, look continuity, and sampled-frame QC expectations.

- [ ] **Step 1: Add failing prompt tests**

Assert storyboard/video prompts include `Scene Character State`, `继承上一场造型`, `只重试失败镜头`, and `抽帧检测身份/服装/黑帧/静音/字幕`.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/unit/runtime/crew/test_adapter_failure_reason.py -v`

Expected: missing video continuity/QC wording.

- [ ] **Step 3: Add continuity/QC sections to storyboard and video prompts**

Include director rhythm, look inheritance, shot purpose, and frame-sampling QC requirements without changing existing multimedia provider contracts.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/unit/runtime/crew/test_adapter_failure_reason.py tests/unit/capabilities/test_runtime_gateway.py -v`

Expected: assertions pass or direct assertions pass if pytest hangs.

### Task 7: Verification, Deployment, Handoff

**Files:**
- Modify: `E:/code_x/mofangagent/HANDOFF.md`
- Use existing deployment scripts/hot-patch flow

**Interfaces:**
- Produces: deployable production hot patch or release package
- Produces: production smoke evidence for script -> assets -> review -> partial retry -> storyboard/video prompt readiness

- [ ] **Step 1: Run local compile and targeted assertions**

Run local Python compile for touched Python files. If pytest hangs, run direct Python assertion scripts that import the new production module and call prompt builders.

- [ ] **Step 2: Deploy to production only after local checks**

Copy changed runtime files to the current production release or package a release, restart `agent-hub-api` and `agent-hub-worker`, and verify `/health/ready`.

- [ ] **Step 3: Production smoke**

Start a short-drama asset run from a high-wu urban cultivation script. Pass criteria: character IDs and look IDs appear in prompts/metadata, role assets are one per character, visual review can reject bad identity/look assets, and reject/retry regenerates only failed labels.

- [ ] **Step 4: Update handoff**

Record current branch, deployment result, verification snapshot, remaining risks, and next work. Do not include command logs.
