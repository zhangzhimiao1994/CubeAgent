# AI Video Editing Compose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal `compose_video` built-in tool that merges generated image/video artifacts into one downloadable MP4.

**Architecture:** Put ffmpeg process orchestration in `agent_hub.video.composer`, expose it through `RuntimeCapabilityGateway`, and wire a dedicated `video_compositor` role plus Crew tool schema to call it. The existing `video_editor` role remains for edit-plan-only work. The final MP4 uses the existing `GeneratedFileStore` file contract so the API/UI artifact path stays unchanged.

**Tech Stack:** Python 3.12, FastAPI runtime capability gateway, ffmpeg CLI, pytest, ruff, mypy, native systemd and Docker deployment manifests.

**Spec:** `docs/superpowers/specs/2026-09-06-ai-video-editing-design.md`

## Global Constraints

- Tool name is `compose_video`.
- Use `GeneratedFileStore.resolve_for`; do not accept arbitrary source paths.
- First version supports `video/mp4`, `image/png`, `image/jpeg`, and `image/webp`.
- Clip count is 1 to 32.
- Image durations are 1 to 10 seconds.
- Aspect ratio is `original`, `16:9`, or `9:16`.
- Output MIME type is `video/mp4`.
- No database migration.
- Production dependency is system `ffmpeg`.
- `compose_video` must run off the async event loop.
- `original` still normalizes all clips to a common compatibility canvas before concat.

---

### Task 1: Composer Unit

**Files:**
- Create: `src/agent_hub/video/__init__.py`
- Create: `src/agent_hub/video/composer.py`
- Test: `tests/unit/video/test_composer.py`

**Interfaces:**
- Produces: `VideoClipInput(storage_key: str, path: Path, mime_type: str, duration_seconds: int | None, filename: str | None)`
- Produces: `VideoComposeRequest(title: str, clips: tuple[VideoClipInput, ...], output_filename: str, aspect_ratio: str, image_duration_seconds: int)`
- Produces: `VideoComposer.compose(request: VideoComposeRequest, output_dir: Path) -> Path`
- Produces: `VideoCompositionError(RuntimeError)`

- [ ] **Step 1: Write the failing tests**

```python
def test_composer_requires_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_hub.video.composer.shutil.which", lambda name: None)
    request = VideoComposeRequest(
        title="Trial",
        clips=(VideoClipInput(storage_key="k", path=tmp_path / "clip.mp4", mime_type="video/mp4"),),
        output_filename="trial.mp4",
    )
    with pytest.raises(VideoCompositionError, match="ffmpeg is not installed"):
        VideoComposer().compose(request, tmp_path)

def test_composer_rejects_unsupported_mime_type(tmp_path):
    request = VideoComposeRequest(
        title="Trial",
        clips=(VideoClipInput(storage_key="k", path=tmp_path / "clip.gif", mime_type="image/gif"),),
        output_filename="trial.mp4",
    )
    with pytest.raises(VideoCompositionError, match="unsupported clip MIME type"):
        VideoComposer(ffmpeg_binary="ffmpeg").compose(request, tmp_path)
```

- [ ] **Step 2: Run red tests**

Run: `$env:PYTHONPATH=(Resolve-Path src).Path; E:\code_x\mofangagent\.venv\Scripts\python.exe -m pytest tests\unit\video\test_composer.py -q`

Expected: fail because `agent_hub.video.composer` does not exist.

- [ ] **Step 3: Implement composer validation and ffmpeg invocation**

Create dataclasses, validation, temp workspace creation, ffmpeg argv calls, and bounded timeout. Use concat demuxer after normalizing each input into a temporary MP4 segment.

- [ ] **Step 4: Run green tests**

Run the same pytest command and make the tests pass.

### Task 2: Runtime Capability Gateway

**Files:**
- Modify: `src/agent_hub/capabilities/runtime.py`
- Test: `tests/unit/capabilities/test_runtime_gateway.py`

**Interfaces:**
- Consumes: `VideoComposer.compose(request, output_dir) -> Path`
- Produces: runtime tool result with `artifact_id`, `file`, `metadata`, `summary`, and `presentation`.

- [ ] **Step 1: Write failing runtime tests**

Add tests that instantiate `RuntimeCapabilityGateway(generated_artifact_dir=tmp_path / "generated", video_composer=FakeVideoComposer())`, store two source files with `GeneratedFileStore`, call `compose_video`, and assert the returned MP4 metadata is stored under the same run.

- [ ] **Step 2: Run red tests**

Run: `$env:PYTHONPATH=(Resolve-Path src).Path; E:\code_x\mofangagent\.venv\Scripts\python.exe -m pytest tests\unit\capabilities\test_runtime_gateway.py -k "compose_video" -q`

Expected: fail because `compose_video` is unknown.

- [ ] **Step 3: Implement gateway support**

Add `_COMPOSE_VIDEO_TOOL = "compose_video"`, include it in replay-safe built-ins, accept `video_composer` injection, parse arguments, resolve source files through `GeneratedFileStore.resolve_for`, call composer, store output bytes as MP4, and return a final attachment.

- [ ] **Step 4: Run green tests**

Run the same targeted pytest command.

### Task 3: Runtime Planning And Tool Schema

**Files:**
- Modify: `src/agent_hub/runtime/role_catalog.py`
- Modify: `src/agent_hub/runtime/role_planner.py`
- Modify: `src/agent_hub/runtime/crew/adapter.py`
- Modify: `src/agent_hub/runtime/defaults.py`
- Test: `tests/unit/runtime/test_role_planner.py`
- Test: `tests/unit/runtime/crew/test_tool_contracts.py`

**Interfaces:**
- Consumes: `compose_video` runtime capability.
- Produces: `video_compositor` role with `("read_context", "compose_video")` for explicit merge/edit delivery requests.
- Preserves: `video_editor` role without `compose_video` for edit-plan-only requests.
- Produces: strict Crew tool schema for `compose_video`.

- [ ] **Step 1: Write failing planner/schema tests**

Add tests asserting “把这些子 Agent 生成的视频和图片合成一个30秒竖屏 MP4” selects `video_compositor` with `compose_video`, ordinary edit-plan requests do not get `compose_video`, English merge requests are detected, and `_tool_definitions(("compose_video",))` exposes required `title` and `clips`.

- [ ] **Step 2: Run red tests**

Run: `$env:PYTHONPATH=(Resolve-Path src).Path; E:\code_x\mofangagent\.venv\Scripts\python.exe -m pytest tests\unit\runtime\test_role_planner.py tests\unit\runtime\crew\test_tool_contracts.py -k "compose_video or video_edit" -q`

Expected: fail because `compose_video` is not planned/exposed.

- [ ] **Step 3: Implement role and schema wiring**

Add a dispatch `video_compositor` role with the tool, add compose-video request detection terms, add `compose_video` to final attachment tool enforcement, and define strict tool parameters.

- [ ] **Step 4: Run green tests**

Run the same targeted pytest command.

### Task 4: Deployment Dependency Declarations

**Files:**
- Modify: `Dockerfile`
- Modify: `deploy/native/install-packages.sh`
- Test: `tests/unit/test_video_editing_dependencies.py`

**Interfaces:**
- Produces: native and Docker runtime install paths that include `ffmpeg`, with dnf handled through a separate best-effort install so base packages are not blocked by repository differences.

- [ ] **Step 1: Write failing dependency tests**

Add tests that read both files and assert `ffmpeg` is included in the runtime package installation path.

- [ ] **Step 2: Run red tests**

Run: `$env:PYTHONPATH=(Resolve-Path src).Path; E:\code_x\mofangagent\.venv\Scripts\python.exe -m pytest tests\unit\test_video_editing_dependencies.py -q`

Expected: fail because dependency declarations do not include `ffmpeg`.

- [ ] **Step 3: Add dependency declarations**

Add `ffmpeg` to native apt package lists and Docker runtime layer installation. For dnf hosts, install base packages first, then try `ffmpeg` and `ffmpeg-free` separately with a clear error if no enabled repository provides either package.

- [ ] **Step 4: Run green tests**

Run the same targeted pytest command.

### Task 5: Verification And Handoff

**Files:**
- Modify: `HANDOFF.md` in the main checkout after merging status back.

**Interfaces:**
- Consumes: all tasks above.
- Produces: verified branch state and handoff note.

- [ ] **Step 1: Run focused backend verification**

Run: `$env:PYTHONPATH=(Resolve-Path src).Path; E:\code_x\mofangagent\.venv\Scripts\python.exe -m pytest tests\unit\video\test_composer.py tests\unit\capabilities\test_runtime_gateway.py tests\unit\runtime\test_role_planner.py tests\unit\runtime\crew\test_tool_contracts.py tests\unit\test_video_editing_dependencies.py -k "compose_video or video_edit or composer" -q`

- [ ] **Step 2: Run static checks**

Run: `$env:PYTHONPATH=(Resolve-Path src).Path; E:\code_x\mofangagent\.venv\Scripts\python.exe -m ruff check src tests/unit/video/test_composer.py tests/unit/capabilities/test_runtime_gateway.py tests/unit/runtime/test_role_planner.py tests/unit/runtime/crew/test_tool_contracts.py tests/unit/test_video_editing_dependencies.py`

Run: `$env:PYTHONPATH=(Resolve-Path src).Path; E:\code_x\mofangagent\.venv\Scripts\python.exe -m mypy --strict src tests`

- [ ] **Step 3: Commit**

Commit implementation on `codex/ai-video-editing-compose` after verification.

- [ ] **Step 4: Update handoff**

Record changed files, tests, dependency requirement, and deployment note.

## Self-Review

- Spec coverage: composer, runtime gateway, role wiring, Crew schema, deployment dependencies, verification, and rollback are all mapped to tasks.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: `VideoClipInput`, `VideoComposeRequest`, `VideoComposer.compose(request, output_dir)`, `compose_video`, and `GeneratedFileStore.resolve_for` are named consistently across tasks.
