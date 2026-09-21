"""Content Studio business module.

This module owns content-production project state and deterministic pipeline
contracts. It deliberately does not own model routing, runtime scheduling,
tools, memory, or multimedia providers; callers can wrap these service methods
from the existing CubeAgent conversation/runtime layers.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256
from types import UnionType
from typing import Any, Final, Protocol, cast, get_args, get_origin, get_type_hints
from uuid import uuid4
from weakref import WeakValueDictionary


class ProjectStatus(str, Enum):
    DRAFT = "DRAFT"
    RESEARCHING = "RESEARCHING"
    RESEARCH_READY = "RESEARCH_READY"
    FACT_CHECKED = "FACT_CHECKED"
    PLAN_READY = "PLAN_READY"
    SCRIPT_READY = "SCRIPT_READY"
    SCRIPT_APPROVED = "SCRIPT_APPROVED"
    STORYBOARD_READY = "STORYBOARD_READY"
    ASSETS_READY = "ASSETS_READY"
    VOICE_READY = "VOICE_READY"
    TIMELINE_READY = "TIMELINE_READY"
    PREVIEW_RENDERED = "PREVIEW_RENDERED"
    QC_REVIEW = "QC_REVIEW"
    FINAL_APPROVED = "FINAL_APPROVED"
    FINAL_RENDERED = "FINAL_RENDERED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_BLOCKED = "FAILED_BLOCKED"
    CANCELLED = "CANCELLED"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONFLICTING = "conflicting"
    OUTDATED = "outdated"
    UNSUPPORTED = "unsupported"
    OPINION = "opinion"


_RUN_ORDER: Final[tuple[ProjectStatus, ...]] = (
    ProjectStatus.RESEARCH_READY,
    ProjectStatus.FACT_CHECKED,
    ProjectStatus.PLAN_READY,
    ProjectStatus.SCRIPT_READY,
    ProjectStatus.STORYBOARD_READY,
    ProjectStatus.ASSETS_READY,
    ProjectStatus.VOICE_READY,
    ProjectStatus.TIMELINE_READY,
    ProjectStatus.PREVIEW_RENDERED,
    ProjectStatus.QC_REVIEW,
    ProjectStatus.FINAL_RENDERED,
)

_BLOCKING_CLAIM_STATUSES: Final[frozenset[ClaimStatus]] = frozenset(
    {ClaimStatus.UNSUPPORTED, ClaimStatus.CONFLICTING, ClaimStatus.OUTDATED}
)

_EXECUTION_MODES: Final[frozenset[str]] = frozenset({"demo", "production"})


@dataclass(frozen=True, slots=True)
class PackManifest:
    pack_type: str
    name: str
    version: str
    schema_version: str
    compatible_core: str
    settings: dict[str, object]

    def with_version(self, version: str) -> PackManifest:
        return replace(self, version=version)


@dataclass(frozen=True, slots=True)
class LockedPacks:
    domain: PackManifest
    format: PackManifest
    platform: PackManifest
    channel: PackManifest
    style: PackManifest


class PackRegistry:
    """Versioned Pack registry for Content Studio.

    Project creation copies the current manifest objects into `LockedPacks`.
    Updating the registry later cannot silently change older projects.
    """

    def __init__(
        self,
        *,
        domain_packs: dict[str, PackManifest],
        format_packs: dict[str, PackManifest],
        platform_packs: dict[str, PackManifest],
        channel_packs: dict[str, PackManifest],
        style_packs: dict[str, PackManifest],
    ) -> None:
        self.domain_packs = dict(domain_packs)
        self.format_packs = dict(format_packs)
        self.platform_packs = dict(platform_packs)
        self.channel_packs = dict(channel_packs)
        self.style_packs = dict(style_packs)

    @classmethod
    def mvp(cls) -> PackRegistry:
        return cls(
            domain_packs={
                "aigc": PackManifest(
                    pack_type="domain",
                    name="aigc",
                    version="1.0.0",
                    schema_version="1.0",
                    compatible_core=">=0.1",
                    settings={
                        "source_priority": (
                            "official_docs",
                            "official_blog",
                            "release_notes",
                            "github_release",
                            "paper",
                            "official_demo",
                            "secondary_media",
                        ),
                        "fact_rules": ("temporal_scope_required", "official_source_preferred"),
                    },
                )
            },
            format_packs={
                "explainer": _format_pack("explainer"),
                "news": _format_pack("news"),
                "tutorial": _format_pack("tutorial"),
            },
            platform_packs={
                "douyin": PackManifest(
                    pack_type="platform",
                    name="douyin",
                    version="1.0.0",
                    schema_version="1.0",
                    compatible_core=">=0.1",
                    settings={
                        "aspect_ratio": "9:16",
                        "width": 1080,
                        "height": 1920,
                        "min_seconds": 45,
                        "max_seconds": 90,
                        "target_seconds": 60,
                        "codec": "H.264/AAC",
                        "subtitle_required": True,
                        "hook_seconds": 3,
                        "visual_change_seconds": (3, 5),
                    },
                )
            },
            channel_packs={
                "ai_frontier": PackManifest(
                    pack_type="channel",
                    name="ai_frontier",
                    version="1.0.0",
                    schema_version="1.0",
                    compatible_core=">=0.1",
                    settings={
                        "persona": "calm AI frontier explainer",
                        "audience": "AI practitioners and curious builders",
                        "banned_phrases": ("稳赚", "永久免费", "绝对领先"),
                    },
                )
            },
            style_packs={
                "fast_minimal": PackManifest(
                    pack_type="style",
                    name="fast_minimal",
                    version="1.0.0",
                    schema_version="1.0",
                    compatible_core=">=0.1",
                    settings={
                        "subtitle_style": "large safe-area captions",
                        "visual_language": "screen capture, cards, charts, concise motion",
                        "transition": "quick cut",
                    },
                )
            },
        )

    def register_domain_pack(self, manifest: PackManifest) -> None:
        self.domain_packs[manifest.name] = manifest

    def lock(
        self,
        *,
        domain: str,
        format: str,
        platform: str,
        channel: str,
        style: str,
    ) -> LockedPacks:
        return LockedPacks(
            domain=deepcopy(self.domain_packs[domain]),
            format=deepcopy(self.format_packs[format]),
            platform=deepcopy(self.platform_packs[platform]),
            channel=deepcopy(self.channel_packs[channel]),
            style=deepcopy(self.style_packs[style]),
        )


@dataclass(frozen=True, slots=True)
class ResearchQuestion:
    question_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    source_url: str
    source_type: str
    publisher: str
    published_at: str | None
    retrieved_at: str
    content_hash: str
    locator: str
    excerpt: str
    license: str


@dataclass(frozen=True, slots=True)
class AtomicClaim:
    claim_id: str
    text: str
    claim_type: str
    temporal_scope: str
    evidence_ids: tuple[str, ...]
    confidence: float
    status: ClaimStatus
    verification: str
    script_usages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResearchSourceCoverage:
    source_type: str
    required: bool
    evidence_ids: tuple[str, ...]
    status: str
    note: str


@dataclass(frozen=True, slots=True)
class ResearchSourceCandidate:
    source_type: str
    source_url: str
    priority: int
    rationale: str


@dataclass(frozen=True, slots=True)
class ResearchBundle:
    questions: tuple[ResearchQuestion, ...]
    evidence: tuple[Evidence, ...]
    source_priority: tuple[str, ...] = ()
    retrieval_plan: tuple[str, ...] = ()
    source_coverage: tuple[ResearchSourceCoverage, ...] = ()
    source_candidates: tuple[ResearchSourceCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceGraph:
    claims: tuple[AtomicClaim, ...]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class FactCheckReport:
    claim_statuses: dict[str, ClaimStatus]
    blocking_claim_ids: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContentPlan:
    sections: tuple[str, ...]
    target_seconds: int
    platform_constraints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScriptSegment:
    segment_id: str
    text: str
    factual: bool
    claim_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScriptDraft:
    hooks: tuple[str, str, str]
    segments: tuple[ScriptSegment, ...]
    subtitle_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Shot:
    shot_id: str
    start_ms: int
    duration_ms: int
    shot_type: str
    narration_segment_ids: tuple[str, ...]
    asset_request_ids: tuple[str, ...]
    overlay: str
    transition: str
    safe_area: str


@dataclass(frozen=True, slots=True)
class Storyboard:
    shots: tuple[Shot, ...]


@dataclass(frozen=True, slots=True)
class AssetRecord:
    asset_id: str
    request_id: str
    source: str
    acquisition_method: str
    url_or_provider_task_id: str
    content_hash: str
    technical_params: dict[str, object]
    rights_status: str
    generation_params: dict[str, object]


@dataclass(frozen=True, slots=True)
class AssetManifest:
    assets: tuple[AssetRecord, ...]


@dataclass(frozen=True, slots=True)
class VoiceTrack:
    audio_artifact_id: str
    timestamp_level: str
    pronunciation_report: tuple[str, ...]
    mime_type: str = "audio/wav"
    source: str = "caller_tts"


@dataclass(frozen=True, slots=True)
class Timeline:
    width: int
    height: int
    duration_ms: int
    tracks: dict[str, tuple[str, ...]]
    preview_artifact_id: str | None = None
    final_artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class QCReport:
    blockers: tuple[str, ...]
    majors: tuple[str, ...]
    minors: tuple[str, ...]
    checked_items: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectEvent:
    event_id: str
    sequence: int
    kind: str
    stage: str
    status: str
    title: str
    summary: str
    artifact_refs: tuple[str, ...] = ()
    payload: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ProviderAttempt:
    stage: str
    idempotency_key: str
    status: str
    result_hash: str
    provider_task_id: str = ""
    error_code: str = ""


@dataclass(frozen=True, slots=True)
class ContentProject:
    project_id: str
    title: str
    topic: str
    source_urls: tuple[str, ...]
    packs: LockedPacks
    tenant_id: str = ""
    owner_user_id: str = ""
    execution_mode: str = "production"
    status: ProjectStatus = ProjectStatus.DRAFT
    script_approved: bool = False
    rights_approved: bool = False
    final_approved: bool = False
    error_code: str | None = None
    error_message: str | None = None
    research_bundle: ResearchBundle | None = None
    evidence_graph: EvidenceGraph | None = None
    fact_check_report: FactCheckReport | None = None
    content_plan: ContentPlan | None = None
    script: ScriptDraft | None = None
    storyboard: Storyboard | None = None
    asset_manifest: AssetManifest | None = None
    voice_track: VoiceTrack | None = None
    timeline: Timeline | None = None
    qc_report: QCReport | None = None
    completed_stage_keys: frozenset[str] = frozenset()
    provider_attempts: tuple[ProviderAttempt, ...] = ()
    project_events: tuple[ProjectEvent, ...] = ()
    revision: int = 0


@dataclass(frozen=True, slots=True)
class ContentProjectSummary:
    project_id: str
    title: str
    topic: str
    status: str
    revision: int
    execution_mode: str
    updated_at: str | None = None


class ContentProjectConflict(ValueError):
    """A newer project snapshot exists; reload before applying this change."""


class InMemoryContentProjectStore:
    def __init__(self) -> None:
        self._projects: dict[str, ContentProject] = {}

    def save(self, project: ContentProject) -> ContentProject:
        self._projects[project.project_id] = project
        return project

    def get(self, project_id: str) -> ContentProject:
        return self._projects[project_id]


class AsyncContentProjectStore(Protocol):
    async def save(self, project: ContentProject) -> ContentProject: ...

    async def get(self, project_id: str) -> ContentProject: ...

    async def list_recent(
        self,
        *,
        tenant_id: str = "",
        owner_user_id: str = "",
        limit: int = 50,
    ) -> tuple[ContentProjectSummary, ...]: ...


class AsyncContentStudioProductionProvider(Protocol):
    async def run_content_project(
        self,
        project: ContentProject,
        *,
        until: ProjectStatus,
    ) -> ContentProject: ...


class AsyncInMemoryContentProjectStore:
    def __init__(self) -> None:
        self._projects: dict[str, dict[str, object]] = {}

    async def save(self, project: ContentProject) -> ContentProject:
        current = self._projects.get(project.project_id)
        expected = content_project_from_payload(current).revision if current else 0
        if project.revision != expected:
            raise ContentProjectConflict("content project was modified; reload and retry")
        saved = replace(project, revision=expected + 1)
        self._projects[project.project_id] = content_project_to_payload(saved)
        return saved

    async def get(self, project_id: str) -> ContentProject:
        return content_project_from_payload(self._projects[project_id])

    async def list_recent(
        self,
        *,
        tenant_id: str = "",
        owner_user_id: str = "",
        limit: int = 50,
    ) -> tuple[ContentProjectSummary, ...]:
        summaries: list[ContentProjectSummary] = []
        for payload in self._projects.values():
            project = content_project_from_payload(payload)
            if tenant_id and project.tenant_id != tenant_id:
                continue
            if owner_user_id and project.owner_user_id != owner_user_id:
                continue
            summaries.append(content_project_summary(project))
        summaries.sort(key=lambda item: item.revision, reverse=True)
        return tuple(summaries[:limit])


class ContentStudioService:
    """Deterministic MVP orchestrator for Content Studio state.

    Expensive provider calls are represented by stage counters and stable
    artifacts. If a stage output already exists, rerunning to the same or later
    status reuses it, which is the contract later runtime adapters must keep.
    """

    def __init__(
        self, *, registry: PackRegistry, store: InMemoryContentProjectStore,
        execution_mode: str = "production",
    ) -> None:
        if execution_mode not in _EXECUTION_MODES:
            raise ValueError("unsupported execution mode")
        self._registry = registry
        self._store = store
        self._execution_mode = execution_mode
        self._provider_calls: Counter[str] = Counter()

    def create_content_project(
        self,
        *,
        title: str,
        topic: str,
        source_urls: tuple[str, ...],
        domain: str,
        format: str,
        platform: str,
        channel: str,
        style: str,
        tenant_id: str = "",
        owner_user_id: str = "",
        execution_mode: str | None = None,
    ) -> ContentProject:
        execution_mode = execution_mode or self._execution_mode
        if execution_mode not in _EXECUTION_MODES:
            raise ValueError(f"unsupported execution mode: {execution_mode}")
        if not title.strip() or not topic.strip():
            raise ValueError("title and topic must be nonblank")
        if len(source_urls) > 20 or any(not url.strip() for url in source_urls):
            raise ValueError("source_urls must contain at most 20 nonblank URLs")
        project = ContentProject(
            project_id=str(uuid4()),
            title=title.strip(),
            topic=topic.strip(),
            source_urls=tuple(source_urls),
            packs=self._registry.lock(
                domain=domain,
                format=format,
                platform=platform,
                channel=channel,
                style=style,
            ),
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            execution_mode=execution_mode,
        )
        project = append_content_project_event(
            project,
            kind="project_created",
            stage="project",
            status=ProjectStatus.DRAFT.value,
            title="项目创建",
            summary=f"{project.title} · {project.topic}",
            payload={
                "topic": project.topic,
                "source_urls": project.source_urls,
                "execution_mode": project.execution_mode,
                "pack_versions": _pack_versions(project.packs),
            },
        )
        return self._store.save(project)

    def get_content_project(self, project_id: str) -> ContentProject:
        return self._store.get(project_id)

    def provider_call_count(self, stage: str) -> int:
        return self._provider_calls[stage]

    def revise_script(self, project_id: str, *, instruction: str) -> ContentProject:
        project = self._store.get(project_id)
        if project.script is None:
            raise ValueError("cannot revise script before script exists")
        if project.execution_mode == "production":
            return self._store.save(_blocked(
                project, "provider_not_configured", "production script provider is not configured",
            ))
        hook = instruction.strip() or "保留原开头但口语化"
        revised = replace(
            project.script,
            hooks=(
                f"更强但不夸张：{hook}",
                project.script.hooks[1],
                project.script.hooks[2],
            ),
        )
        updated = replace(
            project,
            status=ProjectStatus.SCRIPT_READY,
            script_approved=False,
            rights_approved=False,
            final_approved=False,
            error_code=None,
            error_message=None,
            script=revised,
            storyboard=None,
            asset_manifest=None,
            voice_track=None,
            timeline=None,
            qc_report=None,
            completed_stage_keys=_keep_stages(
                project.completed_stage_keys,
                {"research", "evidence_graph", "fact_check", "plan", "script"},
            ),
        )
        updated = append_content_project_event(
            updated,
            kind="revision",
            stage="script",
            status=ProjectStatus.SCRIPT_READY.value,
            title="脚本修改",
            summary=hook,
            payload=_stage_event_payload(updated, "script"),
        )
        return self._store.save(updated)

    def approve_script(self, project_id: str) -> ContentProject:
        project = self._store.get(project_id)
        if project.script is None:
            raise ValueError("cannot approve missing script")
        self._validate_script_claims(project)
        if project.script_approved:
            return project
        updated = replace(
            project,
            status=ProjectStatus.SCRIPT_APPROVED,
            script_approved=True,
            error_code=None,
            error_message=None,
        )
        return self._store.save(
            append_content_project_event(
                updated,
                kind="approval",
                stage="script",
                status=ProjectStatus.SCRIPT_APPROVED.value,
                title="脚本批准",
                summary="脚本已通过人工批准，可继续进入分镜和素材阶段",
                payload=_stage_event_payload(updated, "script"),
            )
        )

    def revise_storyboard(self, project_id: str, *, instruction: str) -> ContentProject:
        project = self._store.get(project_id)
        if project.storyboard is None:
            raise ValueError("cannot revise storyboard before storyboard exists")
        if project.execution_mode == "production":
            return self._store.save(_blocked(
                project, "provider_not_configured", "production storyboard provider is not configured",
            ))
        shots = list(project.storyboard.shots)
        if shots:
            shots[0] = replace(shots[0], overlay=instruction.strip() or shots[0].overlay)
        updated = replace(
            project,
            status=ProjectStatus.STORYBOARD_READY,
            rights_approved=False,
            final_approved=False,
            error_code=None,
            error_message=None,
            storyboard=Storyboard(shots=tuple(shots)),
            asset_manifest=None,
            timeline=None,
            qc_report=None,
            completed_stage_keys=_keep_stages(
                project.completed_stage_keys,
                {"research", "evidence_graph", "fact_check", "plan", "script", "storyboard", "voice"},
            ),
        )
        updated = append_content_project_event(
            updated,
            kind="revision",
            stage="storyboard",
            status=ProjectStatus.STORYBOARD_READY.value,
            title="分镜修改",
            summary=instruction.strip() or "分镜已修改",
            payload=_stage_event_payload(updated, "storyboard"),
        )
        return self._store.save(updated)

    def regenerate_asset(
        self,
        project_id: str,
        *,
        asset_id: str,
        instruction: str,
    ) -> ContentProject:
        project = self._store.get(project_id)
        if project.asset_manifest is None:
            raise ValueError("cannot regenerate assets before assets exist")
        if not any(asset.asset_id == asset_id for asset in project.asset_manifest.assets):
            raise ValueError(f"asset not found: {asset_id}")
        if project.execution_mode == "production":
            return self._store.save(_blocked(
                project, "provider_not_configured", "production asset provider is not configured",
            ))
        assets = []
        replaced_any = False
        for asset in project.asset_manifest.assets:
            if asset.asset_id != asset_id:
                assets.append(asset)
                continue
            replaced_any = True
            revision = _int_metadata(asset.generation_params.get("revision", 1)) + 1
            assets.append(
                replace(
                    asset,
                    url_or_provider_task_id=f"{asset.url_or_provider_task_id}?revision={revision}",
                    content_hash=_hash_text(asset.content_hash + instruction + str(revision)),
                    rights_status="unknown",
                    generation_params={
                        **asset.generation_params,
                        "revision": revision,
                        "retry_instruction": instruction,
                    },
                )
            )
        if not replaced_any:
            raise ValueError(f"asset not found: {asset_id}")
        updated = replace(
            project,
            status=ProjectStatus.ASSETS_READY,
            rights_approved=False,
            final_approved=False,
            error_code=None,
            error_message=None,
            asset_manifest=AssetManifest(assets=tuple(assets)),
            timeline=None,
            qc_report=None,
            completed_stage_keys=_keep_stages(
                project.completed_stage_keys,
                {
                    "research",
                    "evidence_graph",
                    "fact_check",
                    "plan",
                    "script",
                    "storyboard",
                    "assets",
                    "voice",
                },
            ),
        )
        self._provider_calls["assets"] += 1
        updated = append_content_project_event(
            updated,
            kind="revision",
            stage="assets",
            status=ProjectStatus.ASSETS_READY.value,
            title="单素材重生成",
            summary=f"{asset_id}: {instruction.strip() or '重新生成'}",
            artifact_refs=(asset_id,),
            payload=_stage_event_payload(updated, "assets"),
        )
        return self._store.save(updated)

    def regenerate_voice(self, project_id: str, *, instruction: str = "") -> ContentProject:
        project = self._store.get(project_id)
        if not project.script_approved or project.voice_track is None:
            raise ValueError("cannot regenerate voice before approved script and voice exist")
        if project.execution_mode == "production":
            return self._store.save(
                _blocked(
                    project,
                    "provider_not_configured",
                    "production voice provider is not configured",
                )
            )
        self._provider_calls["voice"] += 1
        revision = _artifact_revision(project.voice_track.audio_artifact_id) + 1
        voice = replace(
            project.voice_track,
            audio_artifact_id=f"voice-{project.project_id}-rev-{revision}",
            pronunciation_report=project.voice_track.pronunciation_report
            + ((instruction.strip() or "regenerated voice"),),
        )
        updated = replace(
            project,
            status=ProjectStatus.VOICE_READY,
            final_approved=False,
            error_code=None,
            error_message=None,
            voice_track=voice,
            timeline=None,
            qc_report=None,
            completed_stage_keys=_keep_stages(
                project.completed_stage_keys,
                {"research", "evidence_graph", "fact_check", "plan", "script", "storyboard", "assets", "voice"},
            ),
        )
        updated = append_content_project_event(
            updated,
            kind="revision",
            stage="voice",
            status=ProjectStatus.VOICE_READY.value,
            title="配音重生成",
            summary=instruction.strip() or "配音已重新生成",
            artifact_refs=(voice.audio_artifact_id,),
            payload=_stage_event_payload(updated, "voice"),
        )
        return self._store.save(updated)

    def render_preview(self, project_id: str) -> ContentProject:
        return self.run_content_project(project_id, until=ProjectStatus.PREVIEW_RENDERED)

    def approve_rights(
        self,
        project_id: str,
        *,
        asset_ids: tuple[str, ...],
        note: str,
    ) -> ContentProject:
        project = self._store.get(project_id)
        if project.asset_manifest is None:
            raise ValueError("cannot approve rights before assets exist")
        if not asset_ids:
            raise ValueError("asset_ids are required for rights approval")
        if not note.strip():
            raise ValueError("rights approval note is required")
        target_ids = set(asset_ids)
        known_ids = {asset.asset_id for asset in project.asset_manifest.assets}
        unknown_ids = target_ids - known_ids
        if unknown_ids:
            raise ValueError(f"asset not found: {min(unknown_ids)}")
        if any(
            asset.asset_id in target_ids and asset.rights_status == "restricted"
            for asset in project.asset_manifest.assets
        ):
            raise ValueError("restricted asset cannot be cleared by generic rights approval")
        if all(
            asset.rights_status == "approved"
            and asset.generation_params.get("rights_approval_hash") == asset.content_hash
            for asset in project.asset_manifest.assets if asset.asset_id in target_ids
        ):
            return project

        assets = tuple(
            replace(
                asset,
                rights_status="approved",
                generation_params={
                    **asset.generation_params,
                    "rights_approval_note": note,
                    "rights_approval_hash": asset.content_hash,
                },
            )
            if asset.asset_id in target_ids
            else asset
            for asset in project.asset_manifest.assets
        )
        rights_approved = all(asset.rights_status in _CLEARED_RIGHTS for asset in assets)
        updated = replace(
            project,
            status=ProjectStatus.ASSETS_READY,
            asset_manifest=AssetManifest(assets=assets),
            rights_approved=rights_approved,
            final_approved=False,
            error_code=None if rights_approved else project.error_code,
            error_message=None if rights_approved else project.error_message,
            timeline=None,
            qc_report=None,
            completed_stage_keys=_keep_stages(
                project.completed_stage_keys,
                {"research", "evidence_graph", "fact_check", "plan", "script", "storyboard", "assets", "voice"},
            ),
        )
        updated = append_content_project_event(
            updated,
            kind="approval",
            stage="rights",
            status="approved" if rights_approved else "partially_approved",
            title="素材版权批准",
            summary=f"批准 {len(target_ids)} 个素材：{note.strip()}",
            artifact_refs=tuple(sorted(target_ids)),
            payload={
                "approved_asset_ids": tuple(sorted(target_ids)),
                "rights_approved": rights_approved,
                "note": note.strip(),
            },
        )
        return self._store.save(updated)

    def approve_final(self, project_id: str) -> ContentProject:
        project = self._store.get(project_id)
        if project.timeline is None or project.timeline.preview_artifact_id is None:
            raise ValueError("cannot approve final before preview exists")
        if project.qc_report is None:
            raise ValueError("cannot approve final before QC exists")
        if project.qc_report.blockers:
            raise ValueError("cannot approve final before QC blockers are resolved")
        if not project.script_approved or not _asset_rights_clear(project):
            raise ValueError("script and asset rights approvals are required")
        self._validate_script_claims(project)
        if project.final_approved:
            return project
        updated = replace(
            project,
            status=ProjectStatus.FINAL_APPROVED,
            final_approved=True,
            error_code=None,
            error_message=None,
        )
        return self._store.save(
            append_content_project_event(
                updated,
                kind="approval",
                stage="final",
                status=ProjectStatus.FINAL_APPROVED.value,
                title="终片批准",
                summary="终片已通过人工批准，可进入最终渲染",
                payload=_stage_event_payload(updated, "qc"),
            )
        )

    def retry_stage(self, project_id: str, stage: ProjectStatus) -> ContentProject:
        project = self._store.get(project_id)
        if _stage_output_exists(project, ProjectStatus.FINAL_RENDERED):
            return project
        updated = _invalidate_project_for_stage(project, stage)
        updated = append_content_project_event(
            updated,
            kind="retry_requested",
            stage=stage.value,
            status="requested",
            title="阶段重试",
            summary=f"从 {stage.value} 阶段继续或重新生成受影响内容",
            payload={"stage": stage.value},
        )
        self._store.save(updated)
        return self.run_content_project(project_id, until=stage)

    def run_content_project(
        self,
        project_id: str,
        *,
        until: ProjectStatus = ProjectStatus.QC_REVIEW,
    ) -> ContentProject:
        if until not in _RUN_ORDER:
            raise ValueError(f"invalid run stage: {until.value}")
        project = self._store.get(project_id)
        if project.status is ProjectStatus.CANCELLED:
            raise ValueError("cancelled project cannot run")
        if project.execution_mode == "production":
            return self._store.save(_blocked(
                project, "provider_not_configured", "production research provider is not configured",
            ))
        for stage in _RUN_ORDER:
            if _stage_index(stage) > _stage_index(until):
                break
            project = self._run_stage(project, stage)
            self._store.save(project)
            if project.status in {ProjectStatus.FAILED_BLOCKED, ProjectStatus.FAILED_RETRYABLE}:
                return self._store.save(project)
            if stage is ProjectStatus.SCRIPT_READY and not project.script_approved and _stage_index(until) > _stage_index(ProjectStatus.SCRIPT_READY):
                return self._store.save(project)
            if stage is ProjectStatus.ASSETS_READY and not project.rights_approved and _stage_index(until) > _stage_index(ProjectStatus.ASSETS_READY):
                return self._store.save(project)
        return self._store.save(project)

    def replace_claim_status(
        self,
        project_id: str,
        *,
        claim_id: str,
        status: ClaimStatus,
        note: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> ContentProject:
        project = self._store.get(project_id)
        if project.evidence_graph is None:
            raise ValueError("cannot replace claim status before evidence graph exists")
        if not note.strip():
            raise ValueError("claim verification note is required")
        if status in {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}:
            if project.execution_mode == "production":
                raise ValueError("production claim replacement requires a real verifier")
            self._validate_evidence_ids(project.evidence_graph, evidence_ids)
        claims = tuple(
            replace(
                claim,
                status=status,
                evidence_ids=evidence_ids or claim.evidence_ids,
                verification=note,
            )
            if claim.claim_id == claim_id
            else claim
            for claim in project.evidence_graph.claims
        )
        found = any(claim.claim_id == claim_id for claim in project.evidence_graph.claims)
        if not found:
            raise ValueError(f"claim not found: {claim_id}")
        updated = replace(
            project,
            status=ProjectStatus.FACT_CHECKED,
            script_approved=False,
            rights_approved=False,
            final_approved=False,
            error_code=None,
            error_message=None,
            evidence_graph=EvidenceGraph(claims=claims, evidence=project.evidence_graph.evidence),
            fact_check_report=self._fact_check(EvidenceGraph(claims=claims, evidence=project.evidence_graph.evidence)),
            content_plan=None,
            script=None,
            storyboard=None,
            asset_manifest=None,
            voice_track=None,
            timeline=None,
            qc_report=None,
            completed_stage_keys=frozenset(
                key
                for key in project.completed_stage_keys
                if key in {"research", "evidence_graph", "fact_check"}
            ),
        )
        updated = append_content_project_event(
            updated,
            kind="revision",
            stage="fact_check",
            status=status.value,
            title="事实状态修正",
            summary=f"{claim_id} → {status.value}: {note.strip()}",
            artifact_refs=(claim_id,) + tuple(evidence_ids),
            payload={
                "claim_id": claim_id,
                "status": status.value,
                "note": note.strip(),
                "evidence_ids": evidence_ids,
            },
        )
        return self._store.save(updated)

    def _run_stage(self, project: ContentProject, stage: ProjectStatus) -> ContentProject:
        if stage is ProjectStatus.RESEARCH_READY:
            return self._ensure_research(project)
        if stage is ProjectStatus.FACT_CHECKED:
            return self._ensure_fact_check(project)
        if stage is ProjectStatus.PLAN_READY:
            return self._ensure_plan(project)
        if stage is ProjectStatus.SCRIPT_READY:
            return self._ensure_script(project)
        if stage is ProjectStatus.STORYBOARD_READY:
            return self._ensure_storyboard(project)
        if stage is ProjectStatus.ASSETS_READY:
            return self._ensure_assets(project)
        if stage is ProjectStatus.VOICE_READY:
            return self._ensure_voice(project)
        if stage is ProjectStatus.TIMELINE_READY:
            return self._ensure_timeline(project)
        if stage is ProjectStatus.PREVIEW_RENDERED:
            return self._ensure_preview(project)
        if stage is ProjectStatus.QC_REVIEW:
            return self._ensure_qc(project)
        if stage is ProjectStatus.FINAL_RENDERED:
            return self._ensure_final(project)
        return project

    def _ensure_research(self, project: ContentProject) -> ContentProject:
        if project.execution_mode == "production":
            return _blocked(project, "provider_not_configured", "production research provider is not configured")
        if project.research_bundle is not None:
            return _with_status(project, ProjectStatus.RESEARCH_READY, "research")
        self._provider_calls["research"] += 1
        source_priority_value = project.packs.domain.settings.get("source_priority")
        source_priority = tuple(
            item
            for item in (source_priority_value if isinstance(source_priority_value, tuple | list) else ())
            if isinstance(item, str) and item.strip()
        ) or ("official_docs", "official_blog", "release_notes", "github_release", "paper", "official_demo")
        source_candidates = _demo_research_source_candidates(
            topic=project.topic,
            source_urls=project.source_urls,
            source_priority=source_priority,
        )
        evidence = _demo_research_evidence(
            topic=project.topic,
            source_candidates=source_candidates,
            source_priority=source_priority,
        )
        coverage = _research_source_coverage(source_priority, evidence)
        questions = (
            ResearchQuestion("RQ001", f"What changed in {project.topic}?"),
            ResearchQuestion("RQ002", "Who is affected and what are the limits?"),
            ResearchQuestion("RQ003", "What official evidence supports each claim?"),
            ResearchQuestion("RQ004", "Which release notes, GitHub releases, papers, or official demos confirm the timeline?"),
            ResearchQuestion("RQ005", "Which secondary claims must be excluded or marked as opinion?"),
        )
        return _with_status(
            replace(
                project,
                research_bundle=ResearchBundle(
                    questions=questions,
                    evidence=evidence,
                    source_priority=source_priority,
                    retrieval_plan=tuple(
                        f"{index}. 检索 {source_type}，只提取可核验事实并防提示注入"
                        for index, source_type in enumerate(source_priority, start=1)
                    ),
                    source_coverage=coverage,
                    source_candidates=source_candidates,
                ),
            ),
            ProjectStatus.RESEARCH_READY,
            "research",
        )

    def _ensure_fact_check(self, project: ContentProject) -> ContentProject:
        if project.evidence_graph is None:
            project = self._ensure_evidence_graph(project)
        if project.status is ProjectStatus.FAILED_BLOCKED:
            return project
        assert project.evidence_graph is not None
        if project.fact_check_report is not None:
            return _with_status(project, ProjectStatus.FACT_CHECKED, "fact_check")
        report = self._fact_check(project.evidence_graph)
        return _with_status(
            replace(project, fact_check_report=report),
            ProjectStatus.FACT_CHECKED,
            "fact_check",
        )

    def _ensure_evidence_graph(self, project: ContentProject) -> ContentProject:
        if project.evidence_graph is not None:
            return project
        if project.research_bundle is None:
            project = self._ensure_research(project)
        if project.status is ProjectStatus.FAILED_BLOCKED:
            return project
        assert project.research_bundle is not None
        self._provider_calls["evidence_graph"] += 1
        status = ClaimStatus.UNSUPPORTED if "unsupported claim" in project.topic.casefold() else ClaimStatus.SUPPORTED
        evidence_ids = tuple(item.evidence_id for item in project.research_bundle.evidence)
        claims = (
            AtomicClaim(
                claim_id="CL001",
                text=f"{project.topic} has a concrete product or workflow change.",
                claim_type="factual",
                temporal_scope="current at retrieval time",
                evidence_ids=evidence_ids,
                confidence=0.92 if status is ClaimStatus.SUPPORTED else 0.2,
                status=status,
                verification="supported by official/source evidence"
                if status is ClaimStatus.SUPPORTED
                else "no supporting official evidence found",
                script_usages=(),
            ),
        )
        updated = replace(
            project,
            evidence_graph=EvidenceGraph(claims=claims, evidence=project.research_bundle.evidence),
            completed_stage_keys=project.completed_stage_keys | {"evidence_graph"},
        )
        return append_content_project_event(
            updated,
            kind="stage_completed",
            stage="evidence_graph",
            status=ProjectStatus.FACT_CHECKED.value,
            title=_stage_event_title("evidence_graph"),
            summary=_stage_event_summary(updated, "evidence_graph"),
            artifact_refs=_stage_artifact_refs(updated, "evidence_graph"),
            payload=_stage_event_payload(updated, "evidence_graph"),
        )

    def _ensure_plan(self, project: ContentProject) -> ContentProject:
        if project.content_plan is not None:
            return _with_status(project, ProjectStatus.PLAN_READY, "plan")
        project = self._ensure_fact_check(project)
        if project.status is ProjectStatus.FAILED_BLOCKED:
            return project
        if _has_blocking_claim(project):
            return _blocked(project, "fact_check_blocked", "fact check has unsupported, conflicting, or outdated claims")
        target_seconds = _int_setting(project.packs.platform.settings, "target_seconds")
        width = _int_setting(project.packs.platform.settings, "width")
        height = _int_setting(project.packs.platform.settings, "height")
        aspect_ratio = str(project.packs.platform.settings["aspect_ratio"])
        plan = ContentPlan(
            sections=(
                "0-3s Hook",
                "3-10s What happened",
                "10-35s Core change or principle",
                "35-52s Demo, case, limit, or comparison",
                "52-60s Suitable audience and summary",
            ),
            target_seconds=target_seconds,
            platform_constraints=(
                f"{aspect_ratio} {width}x{height}",
                "subtitles required",
                "meaningful visual change every 3-5 seconds",
            ),
        )
        return _with_status(replace(project, content_plan=plan), ProjectStatus.PLAN_READY, "plan")

    def _ensure_script(self, project: ContentProject) -> ContentProject:
        if project.script is not None:
            return _with_status(project, ProjectStatus.SCRIPT_READY, "script")
        project = self._ensure_plan(project)
        if project.status is ProjectStatus.FAILED_BLOCKED:
            return project
        assert project.evidence_graph is not None
        claim = project.evidence_graph.claims[0]
        self._provider_calls["script"] += 1
        segments = (
            ScriptSegment(
                "SEG001",
                f"{project.topic} is changing how teams evaluate AI tools.",
                True,
                (claim.claim_id,),
            ),
            ScriptSegment(
                "SEG002",
                "Start with official release notes, then compare limits and demos.",
                True,
                (claim.claim_id,),
            ),
            ScriptSegment("SEG003", "Use it if the workflow fits your risk tolerance.", False, ()),
        )
        used_claim = replace(claim, script_usages=("SEG001", "SEG002"))
        script = ScriptDraft(
            hooks=(
                "This AI update matters, but not for the reason the headline says.",
                "Before you try the new model, check these three constraints.",
                "The demo looks simple; the workflow change is the real story.",
            ),
            segments=segments,
            subtitle_lines=tuple(segment.text for segment in segments),
        )
        graph = EvidenceGraph(
            claims=(used_claim,) + project.evidence_graph.claims[1:],
            evidence=project.evidence_graph.evidence,
        )
        return _with_status(
            replace(project, evidence_graph=graph, script=script),
            ProjectStatus.SCRIPT_READY,
            "script",
        )

    def _ensure_storyboard(self, project: ContentProject) -> ContentProject:
        if project.storyboard is not None:
            return _with_status(project, ProjectStatus.STORYBOARD_READY, "storyboard")
        project = self._ensure_script(project)
        if project.status is ProjectStatus.FAILED_BLOCKED:
            return project
        if not project.script_approved:
            return replace(project, status=ProjectStatus.SCRIPT_READY)
        if project.execution_mode == "production":
            return _blocked(
                project,
                "provider_not_configured",
                "production storyboard provider is not configured",
            )
        self._provider_calls["storyboard"] += 1
        target_ms = _int_setting(project.packs.platform.settings, "target_seconds") * 1000
        first_ms = min(3000, target_ms)
        second_ms = max(1000, int(target_ms * 0.12))
        third_ms = max(1000, int(target_ms * 0.42))
        fourth_ms = max(1000, int(target_ms * 0.28))
        used_ms = first_ms + second_ms + third_ms + fourth_ms
        fifth_ms = max(1000, target_ms - used_ms)
        shots = (
            Shot("SHOT001", 0, first_ms, "big_text_hook", ("SEG001",), ("ASREQ001",), "title", "cut", "subtitle_safe"),
            Shot("SHOT002", first_ms, second_ms, "official_demo", ("SEG001",), ("ASREQ002",), "source badge", "cut", "subtitle_safe"),
            Shot("SHOT003", first_ms + second_ms, third_ms, "screen_recording", ("SEG002",), ("ASREQ003",), "step labels", "cut", "subtitle_safe"),
            Shot("SHOT004", first_ms + second_ms + third_ms, fourth_ms, "comparison_chart", ("SEG002",), ("ASREQ004",), "chart labels", "cut", "subtitle_safe"),
            Shot("SHOT005", first_ms + second_ms + third_ms + fourth_ms, fifth_ms, "summary_card", ("SEG003",), ("ASREQ005",), "cta", "cut", "subtitle_safe"),
        )
        return _with_status(
            replace(project, storyboard=Storyboard(shots=shots)),
            ProjectStatus.STORYBOARD_READY,
            "storyboard",
        )

    def _ensure_assets(self, project: ContentProject) -> ContentProject:
        if project.asset_manifest is not None:
            return _with_status(project, ProjectStatus.ASSETS_READY, "assets")
        project = self._ensure_storyboard(project)
        if project.status is ProjectStatus.FAILED_BLOCKED or project.storyboard is None:
            return project
        if project.execution_mode == "production":
            return _blocked(
                project,
                "provider_not_configured",
                "production asset provider is not configured",
            )
        assert project.storyboard is not None
        self._provider_calls["assets"] += 1
        width = _int_setting(project.packs.platform.settings, "width")
        height = _int_setting(project.packs.platform.settings, "height")
        assets = tuple(
            AssetRecord(
                asset_id=f"ASSET{i:03d}",
                request_id=shot.asset_request_ids[0],
                source="demo_provider",
                acquisition_method="demo_generated_placeholder",
                url_or_provider_task_id=f"demo://asset/{project.project_id}/{shot.shot_id}",
                content_hash=_hash_text(project.project_id + shot.shot_id),
                technical_params={"width": width, "height": height, "mime": "image/png"},
                rights_status="unknown",
                generation_params={"pack_versions": _pack_versions(project.packs), "revision": 1},
            )
            for i, shot in enumerate(project.storyboard.shots, start=1)
        )
        updated = _record_provider_attempt(
            replace(project, asset_manifest=AssetManifest(assets=assets)),
            "assets",
            assets,
        )
        return _with_status(
            updated,
            ProjectStatus.ASSETS_READY,
            "assets",
        )

    def _ensure_voice(self, project: ContentProject) -> ContentProject:
        project = self._ensure_assets(project)
        if project.status is ProjectStatus.FAILED_BLOCKED or project.asset_manifest is None:
            return project
        if not _asset_rights_clear(project):
            return _blocked(project, "asset_rights_not_approved", "asset rights must be approved before voice or render")
        if project.voice_track is not None:
            return _with_status(project, ProjectStatus.VOICE_READY, "voice")
        if project.execution_mode == "production":
            return _blocked(project, "provider_not_configured", "production voice provider is not configured")
        self._provider_calls["voice"] += 1
        voice = VoiceTrack(
            audio_artifact_id=f"voice-{project.project_id}",
            timestamp_level="sentence",
            pronunciation_report=("AIGC pronounced as A-I-G-C",),
        )
        updated = _record_provider_attempt(replace(project, voice_track=voice), "voice", voice)
        return _with_status(updated, ProjectStatus.VOICE_READY, "voice")

    def _ensure_timeline(self, project: ContentProject) -> ContentProject:
        if project.timeline is not None:
            return _with_status(project, ProjectStatus.TIMELINE_READY, "timeline")
        project = self._ensure_voice(project)
        if project.status is ProjectStatus.FAILED_BLOCKED or project.voice_track is None:
            return project
        if not _asset_rights_clear(project):
            return _blocked(project, "asset_rights_not_approved", "asset rights must be approved before render")
        width = _int_setting(project.packs.platform.settings, "width")
        height = _int_setting(project.packs.platform.settings, "height")
        duration_ms = _int_setting(project.packs.platform.settings, "target_seconds") * 1000
        timeline = Timeline(
            width=width,
            height=height,
            duration_ms=duration_ms,
            tracks={
                "narration": ("voice",),
                "primary_visual": tuple(asset.asset_id for asset in project.asset_manifest.assets)
                if project.asset_manifest
                else (),
                "subtitle": tuple(project.script.subtitle_lines) if project.script else (),
                "overlay": ("source badges", "safe-area subtitles"),
                "bgm": ("fast_minimal_bgm",),
            },
        )
        return _with_status(
            replace(project, timeline=timeline),
            ProjectStatus.TIMELINE_READY,
            "timeline",
        )

    def _ensure_preview(self, project: ContentProject) -> ContentProject:
        if project.timeline is not None and project.timeline.preview_artifact_id is not None:
            return _with_status(project, ProjectStatus.PREVIEW_RENDERED, "preview")
        project = self._ensure_timeline(project)
        if project.status is ProjectStatus.FAILED_BLOCKED or project.timeline is None:
            return project
        if project.execution_mode == "production":
            return _blocked(project, "provider_not_configured", "production preview renderer is not configured")
        assert project.timeline is not None
        self._provider_calls["preview_render"] += 1
        timeline = replace(project.timeline, preview_artifact_id=f"preview-{project.project_id}")
        updated = _record_provider_attempt(
            replace(project, timeline=timeline),
            "preview_render",
            timeline,
        )
        return _with_status(
            updated,
            ProjectStatus.PREVIEW_RENDERED,
            "preview",
        )

    def _ensure_qc(self, project: ContentProject) -> ContentProject:
        if project.qc_report is not None:
            return _with_status(project, ProjectStatus.QC_REVIEW, "qc")
        project = self._ensure_preview(project)
        if project.status is ProjectStatus.FAILED_BLOCKED or project.timeline is None or project.timeline.preview_artifact_id is None:
            return project
        blockers: list[str] = []
        if project.fact_check_report and project.fact_check_report.blocking_claim_ids:
            blockers.append("fact coverage has blocking claims")
        if project.asset_manifest and any(
            asset.rights_status in {"unknown", "restricted"} for asset in project.asset_manifest.assets
        ):
            blockers.append("asset rights require approval")
        report = QCReport(
            blockers=tuple(blockers),
            majors=(),
            minors=(),
            checked_items=(
                "resolution/aspect/codec",
                "video reviewer frame extraction",
                "video reviewer subtitle text review",
                "black/frozen frame placeholder",
                "audio presence placeholder",
                "subtitle safe area",
                "claim coverage",
                "asset rights",
            ),
        )
        return _with_status(replace(project, qc_report=report), ProjectStatus.QC_REVIEW, "qc")

    def _ensure_final(self, project: ContentProject) -> ContentProject:
        if project.timeline is not None and project.timeline.final_artifact_id is not None:
            return _with_status(project, ProjectStatus.FINAL_RENDERED, "final_render")
        if not project.final_approved:
            return project
        project = self._ensure_qc(project)
        if project.status is ProjectStatus.FAILED_BLOCKED:
            return project
        if project.qc_report is not None and project.qc_report.blockers:
            return _blocked(project, "qc_blocked", "QC blockers must be resolved before final render")
        if project.execution_mode == "production":
            return _blocked(project, "provider_not_configured", "production final renderer is not configured")
        assert project.timeline is not None
        self._provider_calls["final_render"] += 1
        timeline = replace(project.timeline, final_artifact_id=f"final-{project.project_id}")
        updated = _record_provider_attempt(
            replace(project, timeline=timeline),
            "final_render",
            timeline,
        )
        return _with_status(
            updated,
            ProjectStatus.FINAL_RENDERED,
            "final_render",
        )

    def _fact_check(self, graph: EvidenceGraph) -> FactCheckReport:
        statuses = {claim.claim_id: claim.status for claim in graph.claims}
        blocking = tuple(
            claim_id for claim_id, status in statuses.items() if status in _BLOCKING_CLAIM_STATUSES
        )
        return FactCheckReport(
            claim_statuses=statuses,
            blocking_claim_ids=blocking,
            notes=("blocking claims must be revised or removed",) if blocking else ("all usable claims cleared",),
        )

    def _validate_evidence_ids(self, graph: EvidenceGraph, evidence_ids: tuple[str, ...]) -> None:
        if not evidence_ids:
            raise ValueError("supporting evidence_ids are required")
        evidence_by_id = {item.evidence_id: item for item in graph.evidence}
        for evidence_id in evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                raise ValueError(f"evidence not found: {evidence_id}")
            if not (
                evidence.evidence_id
                and evidence.source_url
                and evidence.retrieved_at
                and evidence.content_hash
                and evidence.excerpt
            ):
                raise ValueError(f"evidence is incomplete: {evidence_id}")

    def _validate_script_claims(self, project: ContentProject) -> None:
        if project.script is None:
            raise ValueError("cannot approve missing script")
        if project.evidence_graph is None:
            raise ValueError("cannot approve script before evidence graph exists")
        claims = {claim.claim_id: claim for claim in project.evidence_graph.claims}
        if not claims or any(claim.status in _BLOCKING_CLAIM_STATUSES for claim in claims.values()):
            raise ValueError("script has blocking factual claims")
        for segment in project.script.segments:
            if segment.factual and not segment.claim_ids:
                raise ValueError(f"factual segment missing claim_id: {segment.segment_id}")
            for claim_id in segment.claim_ids:
                claim = claims.get(claim_id)
                if claim is None:
                    raise ValueError(f"claim not found for script segment: {claim_id}")
                if claim.status in _BLOCKING_CLAIM_STATUSES or (segment.factual and claim.status is ClaimStatus.OPINION):
                    raise ValueError(f"script segment uses unsupported factual claim: {claim_id}")
                if segment.factual:
                    self._validate_evidence_ids(project.evidence_graph, claim.evidence_ids)


class AsyncContentStudioService:
    """Async Content Studio facade for durable stores.

    The core pipeline remains the deterministic business service above. This
    facade reloads a project snapshot, runs the requested operation in an
    isolated in-memory service, and writes the updated snapshot back. Expensive
    stage outputs are therefore resumed from project state instead of rerun.
    """

    _locks: WeakValueDictionary[tuple[int, str], asyncio.Lock] = WeakValueDictionary()

    def __init__(
        self, *, registry: PackRegistry, store: AsyncContentProjectStore,
        execution_mode: str = "production",
        production_provider: AsyncContentStudioProductionProvider | None = None,
    ) -> None:
        if execution_mode not in _EXECUTION_MODES:
            raise ValueError("unsupported execution mode")
        self._registry = registry
        self._store = store
        self._execution_mode = execution_mode
        self._production_provider = production_provider
        self._provider_calls: Counter[str] = Counter()

    def _project_lock(self, project_id: str) -> asyncio.Lock:
        key = (id(self._store), project_id)
        return self._locks.setdefault(key, asyncio.Lock())

    async def create_content_project(
        self,
        *,
        title: str,
        topic: str,
        source_urls: tuple[str, ...],
        domain: str,
        format: str,
        platform: str,
        channel: str,
        style: str,
        tenant_id: str = "",
        owner_user_id: str = "",
        execution_mode: str | None = None,
    ) -> ContentProject:
        service = ContentStudioService(
            registry=self._registry,
            store=InMemoryContentProjectStore(),
        )
        project = service.create_content_project(
            title=title,
            topic=topic,
            source_urls=source_urls,
            domain=domain,
            format=format,
            platform=platform,
            channel=channel,
            style=style,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            execution_mode=execution_mode or self._execution_mode,
        )
        self._merge_provider_counts(service)
        return await self._store.save(project)

    async def get_content_project(self, project_id: str) -> ContentProject:
        return await self._store.get(project_id)

    async def list_content_projects(
        self,
        *,
        tenant_id: str = "",
        owner_user_id: str = "",
        limit: int = 50,
    ) -> tuple[ContentProjectSummary, ...]:
        return await self._store.list_recent(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            limit=limit,
        )

    def provider_call_count(self, stage: str) -> int:
        return self._provider_calls[stage]

    async def run_content_project(
        self,
        project_id: str,
        *,
        until: ProjectStatus = ProjectStatus.QC_REVIEW,
    ) -> ContentProject:
        if until not in _RUN_ORDER:
            raise ValueError(f"invalid run stage: {until.value}")
        async with self._project_lock(project_id):
            project = await self._store.get(project_id)
            if project.execution_mode == "production" and self._production_provider is not None:
                updated = await self._production_provider.run_content_project(project, until=until)
                return await self._store.save(updated)
            for stage in _RUN_ORDER[:_stage_index(until) + 1]:
                def run_until(service: ContentStudioService, target: ProjectStatus = stage) -> ContentProject:
                    return service.run_content_project(project_id, until=target)

                project = await self._apply(
                    project,
                    run_until,
                )
                if project.status in {ProjectStatus.FAILED_BLOCKED, ProjectStatus.FAILED_RETRYABLE}:
                    break
                if stage is ProjectStatus.SCRIPT_READY and not project.script_approved:
                    break
            return project

    async def revise_script(self, project_id: str, *, instruction: str) -> ContentProject:
        return await self._mutate(
            project_id,
            lambda service: service.revise_script(project_id, instruction=instruction),
        )

    async def approve_script(self, project_id: str) -> ContentProject:
        return await self._mutate(project_id, lambda service: service.approve_script(project_id))

    async def revise_storyboard(self, project_id: str, *, instruction: str) -> ContentProject:
        return await self._mutate(
            project_id,
            lambda service: service.revise_storyboard(project_id, instruction=instruction),
        )

    async def regenerate_asset(
        self,
        project_id: str,
        *,
        asset_id: str,
        instruction: str,
    ) -> ContentProject:
        return await self._mutate(
            project_id,
            lambda service: service.regenerate_asset(
                project_id,
                asset_id=asset_id,
                instruction=instruction,
            ),
        )

    async def render_preview(self, project_id: str) -> ContentProject:
        return await self.run_content_project(project_id, until=ProjectStatus.PREVIEW_RENDERED)

    async def approve_rights(
        self, project_id: str, *, asset_ids: tuple[str, ...], note: str,
    ) -> ContentProject:
        return await self._mutate(
            project_id,
            lambda service: service.approve_rights(project_id, asset_ids=asset_ids, note=note),
        )

    async def regenerate_voice(self, project_id: str, *, instruction: str = "") -> ContentProject:
        return await self._mutate(
            project_id, lambda service: service.regenerate_voice(project_id, instruction=instruction),
        )

    async def approve_final(self, project_id: str) -> ContentProject:
        return await self._mutate(project_id, lambda service: service.approve_final(project_id))

    async def retry_stage(self, project_id: str, stage: ProjectStatus) -> ContentProject:
        async with self._project_lock(project_id):
            project = await self._store.get(project_id)
            if _stage_output_exists(project, ProjectStatus.FINAL_RENDERED):
                return project
            updated = _invalidate_project_for_stage(project, stage)
            await self._store.save(updated)
        return await self.run_content_project(project_id, until=stage)

    async def replace_claim_status(
        self,
        project_id: str,
        *,
        claim_id: str,
        status: ClaimStatus,
        note: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> ContentProject:
        return await self._mutate(
            project_id,
            lambda service: service.replace_claim_status(
                project_id,
                claim_id=claim_id,
                status=status,
                note=note,
                evidence_ids=evidence_ids,
            ),
        )

    async def _mutate(
        self,
        project_id: str,
        operation: Callable[[ContentStudioService], ContentProject],
    ) -> ContentProject:
        async with self._project_lock(project_id):
            project = await self._store.get(project_id)
            return await self._apply(project, operation)

    async def _apply(
        self, project: ContentProject, operation: Callable[[ContentStudioService], ContentProject],
    ) -> ContentProject:
        store = InMemoryContentProjectStore()
        store.save(project)
        service = ContentStudioService(registry=self._registry, store=store)
        result = operation(service)
        self._merge_provider_counts(service)
        if result == project:
            return project
        return await self._store.save(result)

    def _merge_provider_counts(self, service: ContentStudioService) -> None:
        self._provider_calls.update(service._provider_calls)


def _format_pack(name: str) -> PackManifest:
    return PackManifest(
        pack_type="format",
        name=name,
        version="1.0.0",
        schema_version="1.0",
        compatible_core=">=0.1",
        settings={"structure": name},
    )


_CLEARED_RIGHTS = frozenset({"approved", "cleared", "owned", "licensed"})


def _blocked(project: ContentProject, code: str, message: str) -> ContentProject:
    return append_content_project_event(
        replace(project, status=ProjectStatus.FAILED_BLOCKED, error_code=code, error_message=message),
        kind="stage_failed",
        stage="blocked",
        status="failed",
        title="阶段失败",
        summary=f"{code}: {message}",
        payload={"error_code": code, "error_message": message},
    )


def _asset_rights_clear(project: ContentProject) -> bool:
    return bool(
        project.rights_approved and project.asset_manifest and project.asset_manifest.assets
        and all(asset.rights_status in _CLEARED_RIGHTS for asset in project.asset_manifest.assets)
    )


def _demo_research_source_candidates(
    *,
    topic: str,
    source_urls: tuple[str, ...],
    source_priority: tuple[str, ...],
) -> tuple[ResearchSourceCandidate, ...]:
    candidates: list[ResearchSourceCandidate] = []
    seen: set[str] = set()

    def add(source_type: str, source_url: str, rationale: str) -> None:
        key = source_url.casefold()
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            ResearchSourceCandidate(
                source_type=source_type,
                source_url=source_url,
                priority=len(candidates) + 1,
                rationale=rationale,
            )
        )

    for url in source_urls:
        add(_demo_source_type_for_url(url), url, "用户提供来源，优先核验")
    for source_type in source_priority:
        for url in _demo_source_urls_for_type(source_type, topic):
            add(source_type, url, f"Domain Pack 推荐的 {source_type} 来源")
    return tuple(candidates)


def _demo_research_evidence(
    *,
    topic: str,
    source_candidates: tuple[ResearchSourceCandidate, ...],
    source_priority: tuple[str, ...],
) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    covered: set[str] = set()
    for source_type in source_priority:
        if source_type == "secondary_media":
            continue
        candidate = next(
            (item for item in source_candidates if item.source_type == source_type),
            None,
        )
        if candidate is None:
            continue
        evidence.append(
            _demo_evidence_record(
                evidence_id=f"EV{len(evidence) + 1:03d}",
                topic=topic,
                source_url=candidate.source_url,
                source_type=source_type,
                publisher=_demo_publisher_for_source_type(source_type),
            )
        )
        covered.add(source_type)
    for candidate in source_candidates:
        if candidate.source_type in covered or candidate.source_type == "secondary_media":
            continue
        evidence.append(
            _demo_evidence_record(
                evidence_id=f"EV{len(evidence) + 1:03d}",
                topic=topic,
                source_url=candidate.source_url,
                source_type=candidate.source_type,
                publisher=_demo_publisher_for_source_type(candidate.source_type),
            )
        )
        covered.add(candidate.source_type)
    return tuple(evidence)


def _demo_evidence_record(
    *,
    evidence_id: str,
    topic: str,
    source_url: str,
    source_type: str,
    publisher: str,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_url=source_url,
        source_type=source_type,
        publisher=publisher,
        published_at=None,
        retrieved_at="2026-09-20T00:00:00Z",
        content_hash=_hash_text(source_url + topic),
        locator="page:1",
        excerpt=f"Authoritative source material for {topic} from {source_type}.",
        license="unknown",
    )


def _demo_source_type_for_url(url: str) -> str:
    normalized = url.casefold()
    if "github.com" in normalized and ("release" in normalized or "/tag/" in normalized):
        return "github_release"
    if "arxiv.org" in normalized or "paper" in normalized:
        return "paper"
    if "release" in normalized or "changelog" in normalized:
        return "release_notes"
    if "demo" in normalized:
        return "official_demo"
    if "blog" in normalized:
        return "official_blog"
    if "official" in normalized or "docs" in normalized:
        return "official_docs"
    return "secondary_media"


def _demo_source_urls_for_type(source_type: str, topic: str) -> tuple[str, ...]:
    slug = _hash_text(topic)[:10]
    if source_type == "official_docs":
        return (
            f"https://docs.example.com/aigc/{slug}",
            f"https://platform.example.com/docs/{slug}",
        )
    if source_type == "official_blog":
        return (
            f"https://official.example.com/blog/{slug}",
            f"https://company.example.com/news/{slug}",
        )
    if source_type == "release_notes":
        return (
            f"https://official.example.com/release-notes/{slug}",
            f"https://changelog.example.com/aigc/{slug}",
        )
    if source_type == "github_release":
        return (
            f"https://github.com/example/aigc/releases/tag/{slug}",
            "https://github.com/example/aigc/releases",
        )
    if source_type == "paper":
        return (
            "https://arxiv.org/abs/0000.00000",
            "https://papers.example.com/aigc-system-card",
        )
    if source_type == "official_demo":
        return (
            f"https://official.example.com/demo/{slug}",
            f"https://video.example.com/official-demo/{slug}",
        )
    if source_type == "secondary_media":
        return (
            f"https://example-news.com/aigc/{slug}",
            f"https://analysis.example.com/aigc/{slug}",
        )
    return (f"https://example.com/research/{slug}",)


def _demo_publisher_for_source_type(source_type: str) -> str:
    if source_type in {
        "official_docs",
        "official_blog",
        "release_notes",
        "github_release",
        "official_demo",
    }:
        return "official"
    if source_type == "paper":
        return "research_paper"
    return "secondary"


def _research_source_coverage(
    source_priority: tuple[str, ...],
    evidence: tuple[Evidence, ...],
) -> tuple[ResearchSourceCoverage, ...]:
    coverage: list[ResearchSourceCoverage] = []
    for source_type in source_priority:
        evidence_ids = tuple(item.evidence_id for item in evidence if item.source_type == source_type)
        required = source_type != "secondary_media"
        coverage.append(
            ResearchSourceCoverage(
                source_type=source_type,
                required=required,
                evidence_ids=evidence_ids,
                status="covered" if evidence_ids else "missing",
                note=(
                    "已覆盖权威来源"
                    if evidence_ids and required
                    else "补充参考来源"
                    if evidence_ids
                    else "缺少该来源类型，进入脚本前应补检索"
                ),
            )
        )
    return tuple(coverage)


def _artifact_revision(artifact_id: str) -> int:
    suffix = artifact_id.rpartition("-rev-")[2]
    return int(suffix) if suffix.isdigit() else 1


def _stage_output_exists(project: ContentProject, stage: ProjectStatus) -> bool:
    stage_keys = {
        ProjectStatus.RESEARCH_READY: "research", ProjectStatus.FACT_CHECKED: "fact_check",
        ProjectStatus.PLAN_READY: "plan", ProjectStatus.SCRIPT_READY: "script",
        ProjectStatus.STORYBOARD_READY: "storyboard", ProjectStatus.ASSETS_READY: "assets",
        ProjectStatus.VOICE_READY: "voice", ProjectStatus.TIMELINE_READY: "timeline",
        ProjectStatus.PREVIEW_RENDERED: "preview", ProjectStatus.QC_REVIEW: "qc",
        ProjectStatus.FINAL_RENDERED: "final_render",
    }
    if stage_keys.get(stage) not in project.completed_stage_keys:
        return False
    if stage is ProjectStatus.FACT_CHECKED and project.fact_check_report and project.fact_check_report.blocking_claim_ids:
        return False
    if stage is ProjectStatus.QC_REVIEW and project.qc_report and project.qc_report.blockers:
        return False
    values = {
        ProjectStatus.RESEARCH_READY: project.research_bundle,
        ProjectStatus.FACT_CHECKED: project.fact_check_report,
        ProjectStatus.PLAN_READY: project.content_plan,
        ProjectStatus.SCRIPT_READY: project.script,
        ProjectStatus.STORYBOARD_READY: project.storyboard,
        ProjectStatus.ASSETS_READY: project.asset_manifest,
        ProjectStatus.VOICE_READY: project.voice_track,
        ProjectStatus.TIMELINE_READY: project.timeline,
        ProjectStatus.PREVIEW_RENDERED: project.timeline and project.timeline.preview_artifact_id,
        ProjectStatus.QC_REVIEW: project.qc_report,
        ProjectStatus.FINAL_RENDERED: project.timeline and project.timeline.final_artifact_id,
    }
    return values.get(stage) is not None


def _status_rank(status: ProjectStatus) -> float:
    if status is ProjectStatus.SCRIPT_APPROVED:
        return _RUN_ORDER.index(ProjectStatus.SCRIPT_READY) + 0.5
    if status is ProjectStatus.FINAL_APPROVED:
        return _RUN_ORDER.index(ProjectStatus.QC_REVIEW) + 0.5
    return float(_RUN_ORDER.index(status)) if status in _RUN_ORDER else -1.0


def _stage_index(status: ProjectStatus) -> int:
    if status in _RUN_ORDER:
        return _RUN_ORDER.index(status)
    return len(_RUN_ORDER)


def _with_status(project: ContentProject, status: ProjectStatus, stage_key: str) -> ContentProject:
    already_completed = stage_key in project.completed_stage_keys
    updated = replace(
        project,
        status=project.status if _status_rank(project.status) > _status_rank(status) else status,
        error_code=None,
        error_message=None,
        completed_stage_keys=project.completed_stage_keys | {stage_key},
    )
    if already_completed:
        return updated
    return append_content_project_event(
        updated,
        kind="stage_completed",
        stage=stage_key,
        status=status.value,
        title=_stage_event_title(stage_key),
        summary=_stage_event_summary(updated, stage_key),
        artifact_refs=_stage_artifact_refs(updated, stage_key),
        payload=_stage_event_payload(updated, stage_key),
    )


def _record_provider_attempt(
    project: ContentProject,
    stage: str,
    result: object,
    *,
    provider_task_id: str = "",
) -> ContentProject:
    attempt = ProviderAttempt(
        stage=stage,
        idempotency_key=f"{project.project_id}:{stage}",
        status="completed",
        result_hash=_hash_text(str(_to_json(result))),
        provider_task_id=provider_task_id,
    )
    return record_content_project_provider_attempt(project, attempt)


def record_content_project_provider_attempt(
    project: ContentProject,
    attempt: ProviderAttempt,
) -> ContentProject:
    if any(_same_provider_attempt(item, attempt) for item in project.provider_attempts):
        return project
    updated = replace(project, provider_attempts=project.provider_attempts + (attempt,))
    return append_content_project_event(
        updated,
        kind="provider_attempt",
        stage=attempt.stage,
        status=attempt.status,
        title=f"{_stage_event_title(attempt.stage)} Provider 调用",
        summary=_provider_attempt_summary(attempt),
        artifact_refs=(attempt.provider_task_id,) if attempt.provider_task_id else (),
        payload={
            "stage": attempt.stage,
            "idempotency_key": attempt.idempotency_key,
            "status": attempt.status,
            "result_hash": attempt.result_hash,
            "provider_task_id": attempt.provider_task_id,
            "error_code": attempt.error_code,
        },
    )


def append_content_project_event(
    project: ContentProject,
    *,
    kind: str,
    stage: str,
    status: str,
    title: str,
    summary: str,
    artifact_refs: tuple[str, ...] = (),
    payload: dict[str, object] | None = None,
) -> ContentProject:
    if project.project_events:
        latest = project.project_events[-1]
        if (
            latest.kind == kind
            and latest.stage == stage
            and latest.status == status
            and latest.summary == summary
            and latest.payload == payload
        ):
            return project
    sequence = len(project.project_events) + 1
    event = ProjectEvent(
        event_id=f"{project.project_id}:event:{sequence:04d}",
        sequence=sequence,
        kind=kind,
        stage=stage,
        status=status,
        title=title,
        summary=summary,
        artifact_refs=artifact_refs,
        payload=payload,
    )
    return replace(project, project_events=project.project_events + (event,))


def _same_provider_attempt(left: ProviderAttempt, right: ProviderAttempt) -> bool:
    return (
        left.stage == right.stage
        and left.idempotency_key == right.idempotency_key
        and left.status == right.status
        and left.result_hash == right.result_hash
        and left.provider_task_id == right.provider_task_id
        and left.error_code == right.error_code
    )


def _provider_attempt_summary(attempt: ProviderAttempt) -> str:
    if attempt.status == "completed":
        detail = attempt.provider_task_id or attempt.result_hash[:12]
        return f"{attempt.stage} 调用完成：{detail}"
    if attempt.error_code:
        return f"{attempt.stage} 调用失败：{attempt.error_code}"
    return f"{attempt.stage} 调用状态：{attempt.status}"


def _stage_event_title(stage_key: str) -> str:
    titles = {
        "research": "Research 调研",
        "evidence_graph": "Evidence 事实链",
        "fact_check": "Fact Check 核验",
        "plan": "Content Plan 计划",
        "script": "Script 脚本",
        "storyboard": "Storyboard 分镜",
        "assets": "Assets 素材",
        "voice": "Voice 配音",
        "timeline": "Timeline 时间线",
        "preview": "Preview 预览",
        "preview_render": "Preview 预览",
        "qc": "QC 质检",
        "final_render": "Final Render 终片",
    }
    return titles.get(stage_key, stage_key)


def _stage_event_summary(project: ContentProject, stage_key: str) -> str:
    payload = _stage_event_payload(project, stage_key)
    if stage_key == "research":
        return f"{payload.get('question_count', 0)} 个研究问题，{payload.get('evidence_count', 0)} 条证据"
    if stage_key == "evidence_graph":
        return f"{payload.get('claim_count', 0)} 条 Atomic Claim，{payload.get('evidence_count', 0)} 条证据"
    if stage_key == "script":
        return f"{payload.get('hook_count', 0)} 个 Hook，{payload.get('segment_count', 0)} 段脚本"
    if stage_key == "storyboard":
        return f"{payload.get('shot_count', 0)} 个镜头"
    if stage_key == "assets":
        return f"{payload.get('asset_count', 0)} 个素材"
    if stage_key == "voice":
        return str(payload.get("audio_artifact_id") or "配音已生成")
    if stage_key == "timeline":
        return f"{payload.get('track_count', 0)} 条轨道，{payload.get('duration_ms', 0)}ms"
    if stage_key in {"preview", "preview_render"}:
        return str(payload.get("preview_artifact_id") or "预览已生成")
    if stage_key == "qc":
        return (
            f"{payload.get('blocker_count', 0)} 个 BLOCKER，"
            f"{payload.get('major_count', 0)} 个 MAJOR，{payload.get('minor_count', 0)} 个 MINOR"
        )
    if stage_key == "final_render":
        return str(payload.get("final_artifact_id") or "终片已生成")
    return "阶段已完成"


def _stage_event_payload(project: ContentProject, stage_key: str) -> dict[str, object]:
    if stage_key == "research" and project.research_bundle is not None:
        return {
            "question_count": len(project.research_bundle.questions),
            "evidence_count": len(project.research_bundle.evidence),
            "questions": tuple(question.text for question in project.research_bundle.questions),
            "evidence_ids": tuple(evidence.evidence_id for evidence in project.research_bundle.evidence),
        }
    if stage_key == "evidence_graph" and project.evidence_graph is not None:
        return {
            "claim_count": len(project.evidence_graph.claims),
            "evidence_count": len(project.evidence_graph.evidence),
            "claim_ids": tuple(claim.claim_id for claim in project.evidence_graph.claims),
            "evidence_ids": tuple(evidence.evidence_id for evidence in project.evidence_graph.evidence),
        }
    if stage_key == "fact_check" and project.fact_check_report is not None:
        return {
            "blocking_claim_ids": project.fact_check_report.blocking_claim_ids,
            "notes": project.fact_check_report.notes,
        }
    if stage_key == "plan" and project.content_plan is not None:
        return {
            "target_seconds": project.content_plan.target_seconds,
            "sections": project.content_plan.sections,
            "platform_constraints": project.content_plan.platform_constraints,
        }
    if stage_key == "script" and project.script is not None:
        return {
            "hook_count": len(project.script.hooks),
            "segment_count": len(project.script.segments),
            "hooks": project.script.hooks,
            "segment_ids": tuple(segment.segment_id for segment in project.script.segments),
        }
    if stage_key == "storyboard" and project.storyboard is not None:
        return {
            "shot_count": len(project.storyboard.shots),
            "shot_ids": tuple(shot.shot_id for shot in project.storyboard.shots),
        }
    if stage_key == "assets" and project.asset_manifest is not None:
        return {
            "asset_count": len(project.asset_manifest.assets),
            "asset_ids": tuple(asset.asset_id for asset in project.asset_manifest.assets),
            "rights_statuses": tuple(asset.rights_status for asset in project.asset_manifest.assets),
        }
    if stage_key == "voice" and project.voice_track is not None:
        return {
            "audio_artifact_id": project.voice_track.audio_artifact_id,
            "timestamp_level": project.voice_track.timestamp_level,
            "source": project.voice_track.source,
            "pronunciation_report": project.voice_track.pronunciation_report,
        }
    if stage_key == "timeline" and project.timeline is not None:
        return {
            "duration_ms": project.timeline.duration_ms,
            "width": project.timeline.width,
            "height": project.timeline.height,
            "track_count": len(project.timeline.tracks),
            "tracks": tuple(project.timeline.tracks.keys()),
        }
    if stage_key in {"preview", "preview_render"} and project.timeline is not None:
        return {"preview_artifact_id": project.timeline.preview_artifact_id}
    if stage_key == "qc" and project.qc_report is not None:
        return {
            "blocker_count": len(project.qc_report.blockers),
            "major_count": len(project.qc_report.majors),
            "minor_count": len(project.qc_report.minors),
            "blockers": project.qc_report.blockers,
            "majors": project.qc_report.majors,
            "minors": project.qc_report.minors,
            "checked_items": project.qc_report.checked_items,
        }
    if stage_key == "final_render" and project.timeline is not None:
        return {"final_artifact_id": project.timeline.final_artifact_id}
    return {}


def _stage_artifact_refs(project: ContentProject, stage_key: str) -> tuple[str, ...]:
    payload = _stage_event_payload(project, stage_key)
    refs: list[str] = []
    for key in ("audio_artifact_id", "preview_artifact_id", "final_artifact_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            refs.append(value)
    for key in ("asset_ids", "evidence_ids", "claim_ids", "shot_ids", "segment_ids"):
        value = payload.get(key)
        if isinstance(value, tuple):
            refs.extend(str(item) for item in value if str(item))
    return tuple(refs)


def _keep_stages(existing: frozenset[str], allowed: set[str]) -> frozenset[str]:
    return frozenset(key for key in existing if key in allowed)


def _invalidated_stage_keys(stage: ProjectStatus) -> frozenset[str]:
    mapping: dict[ProjectStatus, tuple[str, ...]] = {
        ProjectStatus.RESEARCH_READY: (
            "research",
            "evidence_graph",
            "fact_check",
            "plan",
            "script",
            "storyboard",
            "assets",
            "voice",
            "timeline",
            "preview",
            "qc",
        ),
        ProjectStatus.FACT_CHECKED: (
            "fact_check",
            "plan",
            "script",
            "storyboard",
            "assets",
            "voice",
            "timeline",
            "preview",
            "qc",
        ),
        ProjectStatus.PLAN_READY: (
            "plan",
            "script",
            "storyboard",
            "assets",
            "voice",
            "timeline",
            "preview",
            "qc",
        ),
        ProjectStatus.SCRIPT_READY: (
            "script",
            "storyboard",
            "assets",
            "voice",
            "timeline",
            "preview",
            "qc",
        ),
        ProjectStatus.STORYBOARD_READY: (
            "storyboard",
            "assets",
            "voice",
            "timeline",
            "preview",
            "qc",
        ),
        ProjectStatus.ASSETS_READY: ("assets", "timeline", "preview", "qc"),
        ProjectStatus.VOICE_READY: ("voice", "timeline", "preview", "qc"),
        ProjectStatus.TIMELINE_READY: ("timeline", "preview", "qc"),
        ProjectStatus.PREVIEW_RENDERED: ("preview", "qc"),
        ProjectStatus.QC_REVIEW: ("qc",),
        ProjectStatus.FINAL_RENDERED: ("final_render",),
    }
    affected = mapping.get(stage, ())
    return frozenset((*affected, "final_render")) if affected else frozenset()


def _invalidate_project_for_stage(project: ContentProject, stage: ProjectStatus) -> ContentProject:
    invalidated = _invalidated_stage_keys(stage)
    if not invalidated:
        raise ValueError(f"invalid retry stage: {stage.value}")
    return replace(
        project,
        completed_stage_keys=frozenset(
            key for key in project.completed_stage_keys if key not in invalidated
        ),
        research_bundle=None if "research" in invalidated else project.research_bundle,
        evidence_graph=None if "evidence_graph" in invalidated else project.evidence_graph,
        fact_check_report=None if "fact_check" in invalidated else project.fact_check_report,
        content_plan=None if "plan" in invalidated else project.content_plan,
        script=None if "script" in invalidated else project.script,
        storyboard=None if "storyboard" in invalidated else project.storyboard,
        asset_manifest=None if "assets" in invalidated else project.asset_manifest,
        voice_track=None if "voice" in invalidated else project.voice_track,
        timeline=None
        if "timeline" in invalidated
        else _invalidate_timeline_artifacts(project.timeline, invalidated),
        qc_report=None if "qc" in invalidated else project.qc_report,
        script_approved=False if "script" in invalidated else project.script_approved,
        rights_approved=False if "assets" in invalidated else project.rights_approved,
        final_approved=False
        if {"script", "storyboard", "assets", "voice", "timeline", "preview", "qc"} & invalidated
        else project.final_approved,
    )


def _invalidate_timeline_artifacts(timeline: Timeline | None, invalidated: frozenset[str]) -> Timeline | None:
    if timeline is None:
        return None
    return replace(
        timeline,
        preview_artifact_id=None if "preview" in invalidated else timeline.preview_artifact_id,
        final_artifact_id=None if "final_render" in invalidated else timeline.final_artifact_id,
    )


def _has_blocking_claim(project: ContentProject) -> bool:
    return bool(project.fact_check_report and project.fact_check_report.blocking_claim_ids)


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _int_setting(settings: Mapping[str, object], field: str) -> int:
    value = settings[field]
    if not isinstance(value, int):
        raise TypeError(f"platform setting {field} must be an integer")
    return value


def _int_metadata(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("metadata integer must not be a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise TypeError("metadata value must be an integer")


def _pack_versions(packs: LockedPacks) -> dict[str, str]:
    return {
        "domain": packs.domain.version,
        "format": packs.format.version,
        "platform": packs.platform.version,
        "channel": packs.channel.version,
        "style": packs.style.version,
    }


def content_project_to_payload(project: ContentProject) -> dict[str, object]:
    return {
        "schema_version": "content_studio.project.v1",
        "project": _to_json(project),
    }


def content_project_summary(
    project: ContentProject,
    *,
    updated_at: str | None = None,
) -> ContentProjectSummary:
    return ContentProjectSummary(
        project_id=project.project_id,
        title=project.title,
        topic=project.topic,
        status=project.status.value,
        revision=project.revision,
        execution_mode=project.execution_mode,
        updated_at=updated_at,
    )


def content_project_summary_from_payload(
    payload: Mapping[str, object],
    *,
    updated_at: str | None = None,
) -> ContentProjectSummary:
    return content_project_summary(content_project_from_payload(payload), updated_at=updated_at)


def content_project_from_payload(payload: Mapping[str, object]) -> ContentProject:
    if payload.get("schema_version") != "content_studio.project.v1":
        raise ValueError("unsupported content project payload schema")
    raw_project = payload.get("project")
    if not isinstance(raw_project, Mapping):
        raise ValueError("content project payload is missing project")  # noqa: TRY004 - invalid serialized value
    return cast(ContentProject, _from_json(raw_project, ContentProject))


def _to_json(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple | list):
        return [_to_json(item) for item in value]
    if isinstance(value, frozenset | set):
        return [_to_json(item) for item in sorted(value)]
    if isinstance(value, Mapping):
        return {str(key): _to_json(item) for key, item in value.items()}
    return value


def _from_json(value: object, annotation: object) -> Any:
    if annotation is object:
        return _restore_json_object(value)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if origin in (UnionType, __import__("typing").Union):
        if value is None and type(None) in args:
            return None
        for item in args:
            if item is not type(None):
                return _from_json(value, item)
    if origin is tuple:
        item_type = args[0] if args else object
        if not isinstance(value, list | tuple):
            raise ValueError("expected list for tuple field")
        return tuple(_from_json(item, item_type) for item in value)
    if origin is frozenset:
        item_type = args[0] if args else object
        if not isinstance(value, list | tuple | set | frozenset):
            raise ValueError("expected list for frozenset field")
        return frozenset(_from_json(item, item_type) for item in value)
    if origin is dict:
        value_type = args[1] if len(args) > 1 else object
        if not isinstance(value, Mapping):
            raise ValueError("expected object for dict field")
        return {str(key): _from_json(item, value_type) for key, item in value.items()}
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ValueError("expected object for dataclass field")
        hints = get_type_hints(annotation)
        return annotation(
            **{
                field.name: _from_json(value[field.name], hints[field.name])
                for field in fields(annotation)
                if field.name in value
            }
        )
    return value


def _restore_json_object(value: object) -> object:
    if isinstance(value, list):
        return tuple(_restore_json_object(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _restore_json_object(item) for key, item in value.items()}
    return value


__all__ = [
    "AssetManifest",
    "AssetRecord",
    "AsyncContentProjectStore",
    "AsyncContentStudioService",
    "AsyncInMemoryContentProjectStore",
    "AtomicClaim",
    "ClaimStatus",
    "ContentPlan",
    "ContentProject",
    "ContentProjectConflict",
    "ContentProjectSummary",
    "ContentStudioService",
    "Evidence",
    "EvidenceGraph",
    "FactCheckReport",
    "InMemoryContentProjectStore",
    "LockedPacks",
    "PackManifest",
    "PackRegistry",
    "ProjectEvent",
    "ProjectStatus",
    "ProviderAttempt",
    "QCReport",
    "ResearchBundle",
    "ResearchQuestion",
    "ResearchSourceCandidate",
    "ResearchSourceCoverage",
    "ScriptDraft",
    "ScriptSegment",
    "Shot",
    "Storyboard",
    "Timeline",
    "VoiceTrack",
    "append_content_project_event",
    "content_project_from_payload",
    "content_project_summary",
    "content_project_summary_from_payload",
    "content_project_to_payload",
    "record_content_project_provider_attempt",
]
