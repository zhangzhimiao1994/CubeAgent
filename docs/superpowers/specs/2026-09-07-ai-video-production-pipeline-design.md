# AI Video Production Pipeline Design

## Goal

Add a structured AI video production pipeline that turns a high-level user brief into a consistent long-form video by planning, generating, validating, and editing short model-generated assets. The pipeline builds on the existing `compose_video` capability instead of asking a video model to generate a long video in one pass.

## Problem

Current video generation models usually have short duration limits and weak continuity across separate generations. Long-form output therefore needs a production workflow:

- Plan the story before generation.
- Define stable character and scene references before shot generation.
- Generate each shot independently with fixed references.
- Retry only failed or inconsistent shots.
- Compose accepted shot clips into a final downloadable MP4.

The core risk is identity drift: the same character can change face, body shape, age, outfit, or visual style between shots. Text prompts alone are not a reliable control surface. The workflow must create explicit visual reference assets and make every downstream prompt refer to those asset IDs.

## Scope

This design covers the next pipeline layer above `compose_video`:

- Script generation.
- Character bible generation.
- Character Model Sheet asset generation.
- Costume Sheet asset generation.
- Scene and prop asset generation.
- Storyboard generation.
- Shot-level video generation requests.
- Shot selection and retry metadata.
- Final edit decision list generation.
- Final composition through `compose_video`.

This design does not require a timeline editor UI, manual trim UI, non-linear multi-track editing, lip-sync, voice cloning, background music generation, or provider-specific reference-image implementation details in the first implementation. Those can be added after the structured data model is stable.

## Pipeline

The pipeline has seven ordered stages.

### 1. Script

The script stage converts the user brief into a structured production script.

Outputs:

- `script_id`
- `title`
- `logline`
- `target_duration_seconds`
- `aspect_ratio`
- `style_profile`
- `characters`
- `scenes`
- `beats`
- `dialogue`
- `constraints`

Every reusable entity gets a stable ID:

- Characters use IDs such as `char_hero_001`.
- Scenes use IDs such as `scene_rooftop_001`.
- Props use IDs such as `prop_red_umbrella_001`.
- Shots use IDs such as `shot_0001`.

The IDs become the binding layer between text, images, video clips, and final editing metadata.

### 2. Character Bible

The character bible defines identity before image generation.

Each character entry includes:

- `character_id`
- `display_name`
- `age_range`
- `body_type`
- `face_shape`
- `hair`
- `eyes`
- `skin_tone`
- `signature_features`
- `personality`
- `movement_style`
- `voice_notes`
- `negative_identity_constraints`
- `continuity_locked_fields`

`continuity_locked_fields` identifies attributes that must not change between shots. Examples include hair color, age range, face shape, eye color, height impression, dominant costume colors, signature accessory, scars, tattoos, or silhouette.

### 3. Character Model Sheet

Character Model Sheet is the primary visual reference asset for identity consistency. It is generated after the character bible and before storyboards or videos.

Required views:

- Front full body.
- Side full body.
- Back full body.
- Three-quarter full body.
- Head close-up.
- Neutral expression close-up.
- Expression row: neutral, happy, angry, surprised, sad.
- Hand reference or distinctive gesture reference when hands are important.
- Height and silhouette guide.
- Signature feature callouts.

Required metadata:

- `asset_id`
- `character_id`
- `asset_kind`: `character_model_sheet`
- `storage_key`
- `mime_type`
- `prompt`
- `negative_prompt`
- `locked_traits`
- `allowed_variations`
- `forbidden_variations`
- `quality_notes`

The model sheet is not just a presentation image. It is an authoritative reference that every character-dependent storyboard and video prompt must cite by `asset_id`.

### 4. Costume Sheet

Costume Sheet separates identity from wardrobe continuity.

Each costume entry includes:

- `costume_id`
- `character_id`
- `costume_name`
- `story_context`
- `front_view_asset_id`
- `back_view_asset_id`
- `detail_asset_ids`
- `materials`
- `primary_colors`
- `secondary_colors`
- `accessories`
- `locked_costume_fields`
- `allowed_damage_or_weathering`
- `forbidden_costume_changes`

For a first implementation, one combined costume board image per costume is acceptable if it contains front, back, material swatches, accessory callouts, and color labels. Later versions can split those into separate assets.

### 5. Scene And Prop Assets

Scene and prop references control continuity outside the character.

Scene assets include:

- `scene_id`
- `asset_kind`: `scene_reference`
- `location_name`
- `time_of_day`
- `lighting`
- `palette`
- `layout_notes`
- `continuity_constraints`
- `storage_key`

Prop assets include:

- `prop_id`
- `asset_kind`: `prop_reference`
- `owner_character_id`
- `shape`
- `material`
- `colors`
- `scale_notes`
- `continuity_constraints`
- `storage_key`

### 6. Storyboard

The storyboard stage creates one still image per shot before video generation.

Each storyboard shot includes:

- `shot_id`
- `sequence_index`
- `duration_seconds`
- `scene_id`
- `character_ids`
- `costume_ids`
- `prop_ids`
- `storyboard_asset_id`
- `camera`
- `action`
- `dialogue`
- `subtitle_text`
- `shot_prompt`
- `negative_prompt`
- `reference_asset_ids`
- `continuity_requirements`
- `acceptance_checks`

`reference_asset_ids` must include the relevant Character Model Sheet and Costume Sheet assets for all visible characters. It also includes scene and prop assets when those entities are visible.

### 7. Shot Video And Final Edit

The shot generation stage sends one generation request per shot.

Each shot video record includes:

- `shot_id`
- `attempt_index`
- `video_asset_id`
- `source_storyboard_asset_id`
- `reference_asset_ids`
- `model`
- `prompt`
- `negative_prompt`
- `duration_seconds`
- `status`
- `review_notes`
- `selected`

The final edit stage builds an edit decision list:

- `edit_id`
- `title`
- `aspect_ratio`
- `ordered_clips`
- `subtitle_track`
- `audio_notes`
- `transitions`
- `final_output_filename`

For the first implementation, `ordered_clips` maps directly to `compose_video.clips`. Transitions, subtitles, BGM, and voiceover remain structured metadata until dedicated rendering capabilities exist.

## Architecture

The implementation should introduce a pipeline planner layer separate from the existing ffmpeg composer.

Recommended modules:

- `agent_hub.video.production.schema`
  - Strict Pydantic models for scripts, characters, model sheets, costume sheets, storyboard shots, shot videos, and edit decisions.
- `agent_hub.video.production.prompts`
  - Prompt builders that convert structured records into text prompts for image and video generation.
- `agent_hub.video.production.planner`
  - Pure planning functions that derive the next stage request from the previous stage outputs.
- `RuntimeCapabilityGateway`
  - New built-in tools that expose the pipeline stages to agents.
- `role_catalog` and `role_planner`
  - New roles for producer, character designer, storyboard artist, shot generator, and compositor.

The existing `agent_hub.video.composer.VideoComposer` remains the low-level media implementation. It should not learn about scripts, characters, storyboards, or provider prompts.

## Dispatch Model

This workflow should be scheduled as an ordered stage graph, not as a flat set of parallel creative roles.

The default dispatch behavior can parallelize many producer-style roles, but video production has hard dependencies:

1. Script must exist before character, scene, storyboard, or shot planning.
2. Character bible must exist before Character Model Sheet and Costume Sheet prompts.
3. Character Model Sheets and Costume Sheets must exist before character-visible storyboard frames.
4. Storyboard frames must exist before shot video generation.
5. Selected shot video artifacts must exist before final edit composition.

The implementation should therefore add a video-production-specific dispatch planner or a dedicated production workflow builder that emits staged `DispatchPlan` records. Parallelism is allowed only inside a stage after dependencies are satisfied. For example, multiple character sheets can be generated in parallel after the character bible exists, and multiple shot videos can be generated in parallel after storyboard validation passes.

## Runtime Tool Model

The first implementation should expose pipeline steps as explicit tools rather than a single opaque long-running tool.

Recommended built-ins:

- `plan_video_script`
- `plan_character_assets`
- `plan_storyboard`
- `plan_shot_videos`
- `compose_video`

`compose_video` already exists. The new tools can initially produce structured JSON artifacts and prompt packages without calling provider-specific image or video APIs directly. This keeps the orchestration testable and lets existing multimodal generation capabilities produce the actual images and videos.

Tool outputs should validate against `agent_hub.video.production.schema` before being stored. Invalid stage outputs should fail early with a stable runtime error instead of letting downstream roles infer missing IDs from prose.

Later versions can add provider-backed tools:

- `generate_character_model_sheet`
- `generate_costume_sheet`
- `generate_storyboard_image`
- `generate_shot_video`

Those provider-backed tools should depend on a stable schema first.

## Agent Roles

Recommended roles:

- `video_producer`
  - Owns script, beats, target duration, shot list, and final continuity decisions.
- `character_designer`
  - Owns Character Model Sheets, Costume Sheets, identity constraints, and reference prompt locks.
- `storyboard_artist`
  - Owns storyboard frames, camera language, and per-shot visual prompts.
- `shot_video_generator`
  - Owns shot-level video requests and retry metadata.
- `video_compositor`
  - Owns final ordered clip assembly through `compose_video`.

The roles should communicate through structured artifacts, not free-form chat summaries. The final compositor should consume selected shot video artifacts and an edit decision list.

## Prompt Locking

Every generated visual prompt must include a prompt lock section.

Character prompt lock fields:

- `character_id`
- `model_sheet_asset_id`
- `costume_sheet_asset_id`
- `locked_identity_text`
- `locked_costume_text`
- `negative_identity_text`
- `reference_priority`

Rules:

- Character identity references outrank shot-specific style embellishments.
- Costume locks outrank scene mood changes.
- A shot prompt must not introduce a conflicting age, face, hair, outfit color, or body type.
- If a user asks for a visible character change, the pipeline should create a new costume or character variant record instead of silently mutating the base character.

## Validation

Validation should run between stages.

Script validation:

- Target duration is positive.
- Shot durations sum to the requested target duration within an allowed tolerance.
- Every referenced character, scene, and prop ID exists.

Character validation:

- Every character has a model sheet before storyboard generation.
- Every visible character has at least one costume sheet.
- Locked traits are non-empty for primary characters.

Storyboard validation:

- Every shot has one storyboard asset.
- Every visible character references a model sheet and costume sheet.
- Every `duration_seconds` fits the target provider limits.
- Every shot has acceptance checks.

Shot video validation:

- Every selected shot has a generated `video/mp4` artifact.
- Shot attempts preserve the source `shot_id`.
- Rejected attempts are retained with review notes.

Edit validation:

- Ordered clips are non-empty.
- Ordered clips point to generated file storage keys.
- Final duration estimate matches selected shots.
- `compose_video` inputs stay within clip count limits.

## Failure And Retry Behavior

Failures should be localized to a stage.

- If a Character Model Sheet is poor, regenerate the model sheet before storyboards.
- If a storyboard frame violates identity or costume constraints, regenerate the storyboard frame before video.
- If a video shot drifts from the storyboard, retry only that shot.
- If final composition fails, keep selected clips and edit decision metadata intact.

Each retry creates a new attempt record and never overwrites the previous asset. The selected attempt is explicit.

## Data And Storage

For the first implementation, generated media assets should continue using `GeneratedFileStore`.

Structured pipeline records can be stored as generated JSON artifacts initially:

- `script.json`
- `character-bible.json`
- `character-model-sheets.json`
- `costume-sheets.json`
- `storyboard.json`
- `shot-videos.json`
- `edit-decision-list.json`

This avoids a database migration in the first implementation. If the workflow becomes interactive or long-running across sessions, promote these records to database tables later.

## First Implementation Boundary

The MVP should produce a structured production package and final MP4 when source video/image artifacts are available.

Included:

- Structured schemas.
- Prompt builders.
- Planning tools for script, character assets, storyboard, shot videos, and edit decision list.
- Role planner updates.
- Tests for prompt lock propagation.
- Tests for Character Model Sheet requirements.
- Tests that `compose_video` can consume selected shot outputs.

Excluded:

- Provider-specific reference image upload APIs.
- Automatic visual similarity scoring.
- Timeline UI.
- Subtitle burn-in.
- Audio mixing.
- Lip-sync.

## Tests

Unit tests should cover:

- Character records require stable `character_id`.
- Character Model Sheet records require front, side, back, three-quarter, head close-up, expression row, and locked traits.
- Costume Sheet records bind to a known `character_id`.
- Storyboard shots fail validation when visible characters lack model sheet or costume sheet references.
- Prompt builders include model sheet and costume sheet asset IDs in every character-visible shot prompt.
- Shot video plans preserve `shot_id`, `storyboard_asset_id`, reference assets, duration, and acceptance checks.
- Edit decision lists map selected shot videos into `compose_video` clip inputs in order.
- Planner selects production roles for long-video requests and still selects the simple compositor for explicit merge-only requests.

## Rollback

This pipeline should be added on top of the existing PR branch and should not alter the low-level `compose_video` contract unless a later implementation plan explicitly requires it.

Current rollback checkpoint:

- Commit: `30bb721b008feca4c6b07cf03375cf0141f377b2`
- Branch/tag: `checkpoint-before-ai-video-editing-20260906-2026`

If only the production pipeline is rejected, revert the pipeline commits and keep the existing `compose_video` PR intact.
