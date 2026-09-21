# AI Short Drama Production Control System Design

## Goal

Build a production-control layer for the existing AI short drama asset and video pipeline so the agent behaves like a director, producer, continuity supervisor, art director, editor, and QC reviewer rather than a plain image/video generator.

The system must keep character identity stable across scenes, looks, costumes, poses, expressions, camera angles, and video segments while still allowing costume and state changes that are justified by the script.

## Problem Statement

The current asset workflow can generate assets and route them through review, but user testing exposed several quality failures:

- A character asset can look like a valid model sheet while representing the wrong identity, such as changing a medical trainee into a silver-haired fantasy character.
- Character identity and costume are implicitly bound together, so changing clothing can cause face, body, age, or role drift.
- Different characters can converge toward the same generated face or styling.
- Scene changes are not clearly separated from look changes; the model may change outfits without script justification or keep one outfit when the plot needs a different state.
- Asset images can contain wrong text, garbled labels, irrelevant props, or modules whose content does not match the role.
- Passing visual review can still mean only "format looks right", not "directorially correct for this script".
- Video generation needs rhythm, continuity, shot purpose, performance, sound, edit timing, and QC, not only storyboard stills.

## Scope

This design covers the production-control system for AI short drama generation:

- Character Identity System
- Look / Costume System
- Continuity System
- Production Direction System
- Identity Validator
- Asset, storyboard, video, and edit prompt integration
- Human approval and retry behavior
- Workbench/API state surfaces needed by the existing conversation agent

This design does not require a fully accurate biometric face-recognition provider in the first implementation. The MVP must define the adapter boundary and ship a deterministic/fake provider for tests, with an optional real embedding provider behind the same interface.

## Architecture

Add an independent business module under the existing CubeAgent runtime rather than building a second agent framework.

Recommended module boundary:

```text
agent_hub/runtime/production/
  character_identity.py
  looks.py
  continuity.py
  direction.py
  validators.py
  prompts.py
  schemas.py
```

The existing conversation and crew layers continue to decide when an asset/video workflow is needed. They call this production module to produce structured plans, prompts, validation requests, and retry decisions. The module does not own model routing, memory, task scheduling, tools, or artifact storage; those remain in existing CubeAgent services.

## Core Concepts

### Character ID

Each story character receives a stable `Character ID`, for example `CHAR_LXM_001`.

The Character ID is long-lived within a project and must be used in every related asset, look, storyboard shot, video segment, and QC report.

Character IDs must not be derived only from display names because names can repeat. The display name is metadata; the ID is the stable key.

### Character Identity Reference

Identity defines "who this person is".

Identity includes:

- face shape
- feature proportions
- eye shape and spacing
- nose shape
- mouth shape
- jawline
- skin tone
- age features
- hairline
- base hairstyle characteristics
- body proportions
- recognizable marks or distinguishing features

Identity does not include changeable clothing as a defining field.

The first identity package for a character should include:

- front face
- left and right 45 degree views
- side profile
- half body
- full body
- necessary expression references
- structured textual identity description
- master identity artifact IDs
- optional embedding/vector references

### Look / Costume

Look defines "what this character is wearing now".

Each `Character ID` can own many `Look ID`s:

```text
CHAR_LXM_001
  IDENTITY_MASTER
  LOOK_001: dessert shop uniform
  LOOK_002: home clothes
  LOOK_003: outdoor clothes
  LOOK_004: evening dress
```

Look includes:

- outfit pieces
- shoes
- jewelry
- bag
- hat
- makeup or hair styling variations
- wet, injured, battle-damaged, work, sleep, or event state
- scene applicability
- allowed accessories
- forbidden identity changes

Look never redefines the face, age, body, or core identity.

### Scene Character State

Each scene stores a state mapping:

```text
EP01_SC01 -> CHAR_LXM_001 + LOOK_001
EP01_SC02 -> CHAR_LXM_001 + LOOK_001
EP01_SC03 -> CHAR_LXM_001 + LOOK_002
EP02_SC01 -> CHAR_LXM_001 + LOOK_003
```

The continuity system decides whether a scene inherits the previous look or switches look.

Scene changes do not automatically mean identity changes or outfit changes.

### Production Direction

Production direction is the director/producer layer that controls the whole video, not just assets.

It produces:

- episode target: duration, platform, pacing, visual rhythm, budget risk
- director statement: emotional arc, shot language, visual priorities
- producer plan: approvals, cost, retry policy, high-cost gates
- continuity notes: character state, props, wounds, wetness, hairstyle, time jumps
- art direction: color, style, asset cleanliness, prop/scene specificity
- edit direction: shot length, transitions, BGM, SFX, subtitles, cut points
- QC criteria: identity, continuity, scene, text, audio, video, copyright, facts when applicable

## Data Model

The module should expose Pydantic/dataclass models first. Persistent storage can then use existing artifact metadata, run state, and later database tables/migrations if needed.

Minimum models:

```python
class CharacterIdentity:
    character_id: str
    display_name: str
    role_type: str | None
    identity_prompt: str
    identity_traits: tuple[str, ...]
    forbidden_drift: tuple[str, ...]
    master_reference_artifact_ids: tuple[str, ...]
    embedding_refs: tuple[str, ...]

class CharacterLook:
    look_id: str
    character_id: str
    name: str
    scene_applicability: tuple[str, ...]
    costume_traits: tuple[str, ...]
    accessories: tuple[str, ...]
    hair_makeup_variations: tuple[str, ...]
    forbidden_identity_changes: tuple[str, ...]

class SceneCharacterState:
    scene_id: str
    character_id: str
    look_id: str
    continuity_reason: str
    inherited_from_scene_id: str | None

class ProductionDirection:
    project_id: str | None
    target_duration_seconds: int
    director_statement: str
    producer_constraints: tuple[str, ...]
    rhythm_beats: tuple[RhythmBeat, ...]
    continuity_rules: tuple[str, ...]
    qc_rules: tuple[str, ...]

class IdentityValidationResult:
    character_id: str
    artifact_id: str
    identity_score: float | None
    threshold: float
    passed: bool
    issues: tuple[str, ...]
    provider: str
```

## Prompt Composition

All character-involved generation must use:

```text
Character Identity + Look / Costume + Pose + Scene + Shot Prompt
```

It must not use "previous scene image edited into next outfit" as the default strategy, because repeated edits accumulate identity drift.

Every generation involving a character must include an Identity Lock section:

```text
IDENTITY LOCK:
Use Character ID CHAR_LXM_001.
Keep the same face shape, feature proportions, eye spacing, eye shape, nose, mouth, jawline, skin tone, age features, hairline, body proportions, and recognizable marks from the master identity reference.
Only change the current Look fields: clothing, shoes, jewelry, bag, hat, scene-specific wetness/damage/makeup.
Do not inherit the face, hair, age, body, or skin tone from costume reference images.
```

Priority order:

```text
Character Identity / Face
> Character Body
> Hairstyle base
> Costume / Look
> Pose
> Scene
```

The prompt builder should produce explicit sections:

- `CHARACTER_ID`
- `IDENTITY_REFERENCE`
- `LOOK_ID`
- `LOOK_RULES`
- `POSE`
- `SCENE`
- `SHOT`
- `NEGATIVE_DRIFT_RULES`
- `QC_EXPECTATIONS`

## Costume Reference Images

If a user provides a costume reference image containing another model, only extract:

- clothing silhouette
- color
- material
- texture
- accessories
- wearing method

The system must explicitly forbid inheriting:

- face
- hair
- skin tone
- age
- body identity
- model ethnicity or personal identity

This must be present in both generation prompts and visual review prompts.

## Continuity Rules

The continuity system receives script scenes and decides whether each character state should inherit or switch look.

Default:

- consecutive scenes inherit the previous look.

Create or switch a Look only when the script indicates:

- changing clothes
- next day / time jump
- arriving home
- leaving home
- attending an event
- sleep / bath / medical treatment / disguise
- rain, injury, battle damage, or other state transformation
- explicit director or user instruction

Scene changes alone do not imply a look change.

The continuity result must be saved as structured data so later storyboard/video generation can reuse it without re-parsing chat history.

## Production Direction Workflow

The AI short drama pipeline should become:

```text
Script
-> Production Direction
-> Character Identity Plan
-> Look / Costume Plan
-> Scene Character State / Continuity
-> Asset Plan
-> Identity References
-> Look Assets
-> Scene / Prop / VFX / Action / Camera Assets
-> Storyboard
-> Shot Video Generation
-> Edit Plan
-> Preview Render
-> Video QC
-> Human Approval
-> Final Render
```

The conversation agent may trigger or approve these stages, but it must not store the project state only in chat.

## Approval Gates

MVP approval gates:

1. Script approval.
2. Character identity approval.
3. Look / costume approval.
4. Full asset package approval.
5. Storyboard approval.
6. Preview/final video approval.

If an upstream item changes:

- Changing identity invalidates all looks, character assets, storyboards, and video involving that character.
- Changing one look invalidates only assets/scenes/shots using that look.
- Changing continuity invalidates affected scene prompts and downstream video, not unrelated identity references.
- Rejecting one asset should retry only that asset and its dependent downstream outputs.

## Identity Validation

Every generated character image or video frame sample must be validated against the character's master identity reference.

MVP:

- Add an `IdentityValidator` interface.
- Provide a fake deterministic validator for CI.
- Provide a vision-model validator that returns structured identity score and issues.
- Store score, threshold, provider, and issues in artifact metadata.

Future provider:

- Face detection.
- Face embedding comparison.
- Multi-frame sampling for videos.
- Cross-character similarity check to prevent two characters from becoming too similar.

Default thresholds:

- image identity pass: `0.78`
- video frame sampled identity pass: `0.72`
- cross-character collision warning: `0.82`

The exact numeric thresholds must be configuration values, not hardcoded constants hidden in prompt text.

If identity validation fails:

- the artifact is not considered approved;
- the system retries only that character/look/shot;
- retry prompt must start from the master identity reference, not the failed output;
- after retry exhaustion, the result goes to human review as Identity Correction.

## Visual Review And QC

Visual review must check both form and content:

- The model sheet has the required modules.
- Each module actually matches the character anchor.
- Expression changes are the same face in different emotions.
- Clothing display matches the role and current look.
- Look variants are justified by scenes.
- Props belong to the role or script.
- Text labels are short and correct; key label typos or garbled labels fail.
- Background is clean for identity/look assets.
- Costume references do not leak model identity.

Video QC adds:

- sampled-frame identity validation;
- look continuity across shots;
- prop continuity;
- wounds, wetness, hair, makeup, and damage state continuity;
- shot rhythm vs director statement;
- cut timing, subtitle timing, and BGM/SFX balance;
- repeated/frozen/black frames;
- scene/shot mismatch;
- generated text artifacts.

## Integration Points

Initial integration should touch these existing areas:

- `src/agent_hub/runtime/crew/adapter.py`
  - direct asset prompt specs;
  - artifact labels;
  - retry prompts;
  - approval review feedback.

- `src/agent_hub/capabilities/runtime.py`
  - generated artifact metadata;
  - visual review results;
  - retry loops for rejected generated images.

- `src/agent_hub/app.py`
  - production visual reviewer;
  - identity validation model gateway adapter.

- `src/agent_hub/runs/service.py` and `src/agent_hub/runs/repository.py`
  - run state and review item feedback;
  - preserving structured production state across interruptions.

- `web/src/components/ArtifactFileCard.tsx` and run/workbench pages
  - show Character ID, Look ID, identity score, and failed/needs-regeneration state clearly.

## API Surface

Expose internal service functions first; public tools can wrap them later.

Minimum operations:

- `create_character_identity_plan(script)`
- `create_character_identity_assets(character_id)`
- `create_or_update_look(character_id, scene_context)`
- `resolve_scene_character_states(script_scenes)`
- `build_character_generation_prompt(character_id, look_id, pose, scene, shot)`
- `validate_character_identity(artifact, character_id)`
- `validate_cross_character_distinctness(character_ids)`
- `regenerate_identity_failed_asset(artifact_id)`
- `regenerate_look_failed_asset(artifact_id)`

Conversation operations should map natural language to these, such as:

- "这个角色换成晚宴服，但脸不要变。"
- "第 6 镜头衣服不对，只重生这个镜头。"
- "下一场是同一天，沿用上一场造型。"
- "这个参考图只取衣服，不要取模特脸。"

## Testing Strategy

Unit tests:

- Character ID generation is stable and unique.
- Identity and Look are separate structures.
- Look changes do not rewrite identity traits.
- Scene continuity inherits looks across continuous scenes.
- Explicit time jumps or wardrobe changes create/switch looks.
- Prompt builder orders identity before look/costume.
- Costume reference prompts forbid inheriting model identity.
- Visual review prompt rejects module-content mismatch and key label typos.
- Identity validation failure prevents artifact approval.

Contract tests:

- Generated artifact metadata includes `character_id`, `look_id`, and `identity_validation` when applicable.
- Approval review items expose failed identity/continuity states.
- Retry payload preserves passed assets and regenerates only failed character/look/shot items.

Integration tests:

- Script with two characters creates two identities and distinct looks.
- Consecutive scenes inherit look until a script time jump.
- Rejected identity asset retries only the affected character.
- Rejected look asset retries only that look, not the identity master.

Golden tests:

- A known script yields a stable production direction, identity plan, look plan, and continuity map.

Media tests:

- Fake image/video providers produce artifacts with deterministic identity scores.
- Video frame sampling flags identity drift in at least one sampled frame.

E2E smoke:

- Input a short script.
- Generate identity references.
- Generate look assets.
- Generate storyboard prompts.
- Generate at least one character shot prompt from identity + look + pose + scene + shot.
- Reject one failed look and verify only that look is retried.

## Migration Strategy

Phase 1: Schema and prompt control

- Add models and prompt builder.
- Add unit tests for identity/look separation and continuity.
- Add metadata to generated asset artifacts where labels indicate character assets.

Phase 2: Identity and look assets

- Generate identity assets before look/costume assets.
- Store master identity artifact IDs.
- Generate look assets from identity plus look definition.

Phase 3: Continuity integration

- Parse scenes into scene character states.
- Use continuity output for storyboard/video prompts.
- Preserve continuity state through approvals and retries.

Phase 4: Validation

- Add identity validator adapter and fake provider.
- Add visual-model identity validation.
- Add video frame sampling hooks.

Phase 5: Workbench and production direction

- Display identity/look/continuity/QC state in the UI.
- Add director/producer/edit/QC summaries to project status.

## Acceptance Criteria

- The system creates a distinct Character ID for each important character.
- Character identity and costume/look are separate data structures.
- A character can have multiple Look IDs without changing identity.
- Scene continuity decides whether a look is inherited or changed.
- Character generation prompts start from identity, not previous scene output.
- Costume reference images cannot transfer model face/body identity.
- Artifact metadata records character/look identity when applicable.
- Identity validation can block failed character assets.
- Rejection of one character/look item retries only affected outputs.
- Visual review fails assets whose model sheet modules do not match role anchors.
- The UI or admin payload makes failed identity/look assets visibly distinct from approved assets.

## Risks And Open Questions

- Real face embedding may be unavailable or costly for some providers. The interface must support a fake/test provider and a vision-model provider first.
- Some anime/stylized generations may lack reliable face detection. For those, the validator should fall back to vision-model structured comparison and stricter visual review.
- Identity validation thresholds need calibration with real generated samples.
- Existing artifact metadata may not be enough for long-lived projects; a DB-backed production project table may be needed after MVP.
- Prompt length is already bounded, so identity/look summaries must be compact and structured.
- The workbench needs clear UI state so rejected assets are not mistaken for final approved outputs.
