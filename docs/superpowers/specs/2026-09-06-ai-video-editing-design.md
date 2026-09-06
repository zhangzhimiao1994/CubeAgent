# AI Video Editing Design

## Goal

Add a first production-ready video composition capability so Agent Hub can turn generated image/video artifacts into a downloadable MP4. This is the first step toward long-form video workflows where short model-generated clips are stitched into a longer deliverable.

## Scope

The first version is deliberately narrow:

- Provide one built-in runtime tool named `compose_video`.
- Let a dedicated `video_compositor` role call `compose_video`; keep `video_editor` for edit-plan-only work.
- Accept existing generated artifact metadata as clip inputs.
- Support MP4 video inputs for concatenation.
- Support image inputs by turning each image into a fixed-duration segment.
- Store the final MP4 through `GeneratedFileStore` and return the same public file contract used by DOCX/PPTX/ZIP/multimedia tools.
- Require `ffmpeg` and fail with a stable `RuntimeCapabilityError` when it is unavailable.

The first version does not include a timeline UI, manual trimming UI, BGM mixing, subtitle rendering, transitions, multi-track audio mixing, or provider-specific video generation changes.

## Architecture

`src/agent_hub/video/composer.py` owns the small ffmpeg wrapper. It validates clip metadata, receives already-resolved source paths, builds a temporary ffmpeg concat workspace under a caller-provided output directory, and writes one MP4 file. It does not know about agents or runs.

`RuntimeCapabilityGateway` exposes the built-in `compose_video` tool. It resolves run-scoped generated artifact storage keys, validates MIME claims against source filenames, calls the composer off the event loop, stores the MP4 with a new artifact id, and returns a final attachment result. The Crew adapter exposes the tool schema and treats it as a required final attachment tool for the dedicated composition role, preventing text-only “I edited it” responses.

Deployment declares `ffmpeg` in Docker and native runtime paths so production has the same dependency as CI/container installs. Native apt hosts install `ffmpeg` with the base package set. Native dnf hosts attempt `ffmpeg` and `ffmpeg-free` separately and emit a clear repository/Docker-mode message if neither package is available. When a host lacks `ffmpeg`, the capability remains visible but execution fails with a clear stable message.

## Tool Contract

Tool name: `compose_video`

Required fields:

- `title`: human-readable title.
- `clips`: ordered array of clip objects.

Optional fields:

- `filename`: safe MP4 filename.
- `aspect_ratio`: `original`, `16:9`, or `9:16`; default `original`.
- `image_duration_seconds`: default duration for image clips; allowed range `1` to `10`.
- `presentation`: `step_detail` or `final_attachment`; default `final_attachment`.

Clip object:

- `storage_key`: required generated file storage key.
- `filename`: optional display filename, used only for diagnostics.
- `mime_type`: required; allowed image MIME types and `video/mp4`.
- `duration_seconds`: optional image duration override, allowed range `1` to `10`.

Output:

- `artifact_id`
- `file`
- `metadata`
- `summary`
- `presentation`

The `file`/`metadata` object uses the existing generated artifact contract with `filename`, `mime_type`, `size_bytes`, `sha256`, `storage_key`, and `download_url`.

## Validation And Safety

- Resolve every source file through `GeneratedFileStore.resolve_for`; never accept arbitrary filesystem paths.
- Only support MIME types already allowed by generated artifact storage.
- Reject caller-supplied MIME types that do not match the generated artifact filename extension.
- Reject empty clip lists and cap clips at 32.
- Reject unsupported aspect ratios and unsafe filenames.
- Normalize every segment to a common H.264/yuv420p MP4 canvas before concatenation. `9:16` uses 1080x1920, `16:9` uses 1920x1080, and `original` uses a conservative 1280x720 compatibility canvas while preserving source aspect ratio with padding.
- Use a caller-provided temporary output directory for ffmpeg staging so the returned MP4 path stays readable until the gateway stores it.
- Invoke ffmpeg with argv lists, not shell-constructed command strings.
- Bound ffmpeg execution with a timeout.
- Surface stable errors without leaking local paths or secrets.

## Tests

Unit tests cover:

- `compose_video` availability and replay-safety.
- Runtime gateway stores an MP4 final artifact when the composer succeeds.
- Runtime gateway rejects missing generated artifact store.
- Runtime gateway passes only storage-key-resolved paths into the composer.
- Composer fails clearly when `ffmpeg` is missing.
- Crew tool schema exposes the strict `compose_video` contract.
- Final attachment enforcement includes `compose_video`.
- Role planner selects `video_compositor` with `compose_video` for explicit editing/merge delivery requests while keeping ordinary `video_editor` requests tool-free.
- Native/Docker install manifests include `ffmpeg`, with dnf handled outside the base package list.

## Rollback

Rollback target is the pre-trial checkpoint:

- Commit: `30bb721b008feca4c6b07cf03375cf0141f377b2`
- Branch/tag: `checkpoint-before-ai-video-editing-20260906-2026`

No database migration is required for this feature.
