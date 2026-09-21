# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent_hub.content_studio import (
    AsyncContentStudioService,
    AsyncInMemoryContentProjectStore,
    AtomicClaim,
    ClaimStatus,
    ContentPlan,
    ContentProjectConflict,
    ContentStudioService,
    Evidence,
    EvidenceGraph,
    FactCheckReport,
    InMemoryContentProjectStore,
    PackRegistry,
    ProjectStatus,
    ProviderAttempt,
    QCReport,
    ResearchBundle,
    ResearchQuestion,
    ScriptDraft,
    ScriptSegment,
    Timeline,
    content_project_from_payload,
    content_project_to_payload,
    record_content_project_provider_attempt,
)


def test_content_project_payload_round_trips_locked_state_and_artifacts() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="AIGC explainer",
        topic="Explain a supported AIGC release",
        source_urls=("https://example.com/official-release",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    project = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)

    restored = content_project_from_payload(content_project_to_payload(project))

    assert restored == project
    assert restored.packs.domain.version == "1.0.0"
    assert restored.asset_manifest is not None
    assert restored.asset_manifest.assets[0].rights_status == "unknown"


async def test_async_content_studio_service_resumes_from_persisted_project_without_regenerating() -> None:
    store = AsyncInMemoryContentProjectStore()
    service = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    project = await service.create_content_project(
        title="AIGC tutorial",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="tutorial",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    await service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    await service.approve_script(project.project_id)
    project = await service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert project.asset_manifest is not None
    project = await service.approve_rights(project.project_id, asset_ids=tuple(
        asset.asset_id for asset in project.asset_manifest.assets
    ), note="Fixture sources reviewed")
    first_assets = project.asset_manifest

    restarted = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    resumed = await restarted.run_content_project(project.project_id, until=ProjectStatus.QC_REVIEW)

    assert resumed.status is ProjectStatus.QC_REVIEW
    assert resumed.asset_manifest == first_assets
    assert restarted.provider_call_count("assets") == 0


async def test_async_content_studio_service_uses_production_provider_for_deep_research_and_plain_script() -> None:
    store = AsyncInMemoryContentProjectStore()
    provider = FakeProductionContentProvider()
    service = AsyncContentStudioService(
        registry=PackRegistry.mvp(),
        store=store,
        production_provider=provider,
    )
    project = await service.create_content_project(
        title="AIGC 科普视频",
        topic="做一条 60 秒 AIGC 抖音科普视频",
        source_urls=("https://openai.com/index/gpt-5/",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
    )

    ready = await service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)

    assert ready.status is ProjectStatus.SCRIPT_READY
    assert provider.calls == [ProjectStatus.SCRIPT_READY]
    assert ready.research_bundle is not None
    assert len(ready.research_bundle.questions) >= 3
    assert len(ready.research_bundle.evidence) >= 2
    assert ready.evidence_graph is not None
    assert len(ready.evidence_graph.claims) >= 2
    assert ready.fact_check_report is not None
    assert ready.fact_check_report.blocking_claim_ids == ()
    assert ready.script is not None
    assert len(ready.script.hooks) == 3
    assert all("Claim" not in segment.text for segment in ready.script.segments)
    assert all(segment.claim_ids for segment in ready.script.segments if segment.factual)


async def test_provider_attempt_ledger_persists_successful_expensive_stages() -> None:
    store = AsyncInMemoryContentProjectStore()
    service = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    project = await _create_demo(service)
    await service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    await service.approve_script(project.project_id)
    generated = await service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)

    asset_attempts = tuple(
        attempt for attempt in generated.provider_attempts if attempt.stage == "assets"
    )
    assert len(asset_attempts) == 1
    assert asset_attempts[0].status == "completed"
    assert asset_attempts[0].idempotency_key == f"{project.project_id}:assets"
    assert asset_attempts[0].result_hash

    restarted = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    resumed = await restarted.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)

    assert tuple(attempt for attempt in resumed.provider_attempts if attempt.stage == "assets") == asset_attempts
    assert restarted.provider_call_count("assets") == 0


def test_provider_attempt_ledger_keeps_failed_then_successful_history() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Attempt history",
        topic="AIGC attempt history",
        source_urls=(),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    failed = ProviderAttempt(
        stage="voice",
        idempotency_key=f"{project.project_id}:voice",
        status="failed",
        result_hash="",
        error_code="tts_timeout",
    )
    succeeded = ProviderAttempt(
        stage="voice",
        idempotency_key=f"{project.project_id}:voice",
        status="completed",
        result_hash="voice-result",
        provider_task_id="voice.mp3",
    )

    with_failure = record_content_project_provider_attempt(project, failed)
    with_success = record_content_project_provider_attempt(with_failure, succeeded)
    deduped = record_content_project_provider_attempt(with_success, succeeded)

    assert tuple(attempt.status for attempt in deduped.provider_attempts if attempt.stage == "voice") == (
        "failed",
        "completed",
    )
    assert tuple(event.status for event in deduped.project_events if event.kind == "provider_attempt") == (
        "failed",
        "completed",
    )
    assert any("tts_timeout" in event.summary for event in deduped.project_events)


async def test_retry_stage_regenerates_existing_preview_without_restarting_assets() -> None:
    store = AsyncInMemoryContentProjectStore()
    service = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    project = await _create_demo(service)
    await service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    await service.approve_script(project.project_id)
    project = await service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert project.asset_manifest is not None
    await service.approve_rights(
        project.project_id,
        asset_ids=tuple(asset.asset_id for asset in project.asset_manifest.assets),
        note="assets reviewed",
    )
    rendered = await service.run_content_project(project.project_id, until=ProjectStatus.PREVIEW_RENDERED)
    assert rendered.timeline is not None
    assert rendered.timeline.preview_artifact_id is not None
    assert service.provider_call_count("assets") == 1
    assert service.provider_call_count("preview_render") == 1

    retried = await service.retry_stage(project.project_id, ProjectStatus.PREVIEW_RENDERED)

    assert retried.timeline is not None
    assert retried.timeline.preview_artifact_id is not None
    assert service.provider_call_count("assets") == 1
    assert service.provider_call_count("preview_render") == 2


async def test_async_retry_stage_uses_production_provider_after_invalidating_preview() -> None:
    store = AsyncInMemoryContentProjectStore()
    provider = RetryPreviewProductionProvider()
    service = AsyncContentStudioService(
        registry=PackRegistry.mvp(),
        store=store,
        production_provider=provider,
    )
    project = await service.create_content_project(
        title="AIGC preview",
        topic="AIGC preview retry",
        source_urls=(),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
    )
    timeline = Timeline(
        width=1080,
        height=1920,
        duration_ms=60_000,
        tracks={"primary_visual": ("ASSET001",)},
        preview_artifact_id="old-preview.mp4",
    )
    await store.save(
        replace(
            project,
            status=ProjectStatus.QC_REVIEW,
            completed_stage_keys=frozenset({"timeline", "preview", "qc"}),
            timeline=timeline,
            qc_report=QCReport(blockers=(), majors=(), minors=(), checked_items=("old qc",)),
        )
    )

    retried = await service.retry_stage(project.project_id, ProjectStatus.PREVIEW_RENDERED)

    assert provider.calls == [ProjectStatus.PREVIEW_RENDERED]
    assert retried.timeline is not None
    assert retried.timeline.preview_artifact_id == "production-preview-1.mp4"
    assert retried.qc_report is None


async def test_async_content_studio_service_persists_blocked_claim_fix_across_restart() -> None:
    store = AsyncInMemoryContentProjectStore()
    service = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    project = await service.create_content_project(
        title="Unsafe AIGC explainer",
        topic="unsupported claim: every model is permanently free",
        source_urls=("https://example.com/blog",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    blocked = await service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    assert blocked.status is ProjectStatus.FAILED_BLOCKED
    assert blocked.fact_check_report is not None

    restarted = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    fixed = await restarted.replace_claim_status(
        project.project_id,
        claim_id=blocked.fact_check_report.blocking_claim_ids[0],
        status=ClaimStatus.SUPPORTED,
        note="manual official source added",
        evidence_ids=("EV001",),
    )
    resumed = await restarted.run_content_project(fixed.project_id, until=ProjectStatus.SCRIPT_READY)

    assert resumed.status is ProjectStatus.SCRIPT_READY
    assert resumed.script is not None
    assert resumed.fact_check_report is not None
    assert resumed.fact_check_report.blocking_claim_ids == ()


async def _create_demo(service: AsyncContentStudioService):
    return await service.create_content_project(
        title="Persistent demo", topic="Explain a supported release",
        source_urls=("https://example.com/docs",), domain="aigc", format="explainer",
        platform="douyin", channel="ai_frontier", style="fast_minimal", execution_mode="demo",
    )


class FakeProductionContentProvider:
    def __init__(self) -> None:
        self.calls: list[ProjectStatus] = []

    async def run_content_project(self, project, *, until: ProjectStatus):
        self.calls.append(until)
        evidence = (
            Evidence(
                evidence_id="EV001",
                source_url="https://openai.com/index/gpt-5/",
                source_type="official_blog",
                publisher="OpenAI",
                published_at="2026-08-01T00:00:00Z",
                retrieved_at="2026-09-21T00:00:00Z",
                content_hash="hash-openai",
                locator="section:overview",
                excerpt="GPT-5 release material describes model and product changes.",
                license="source_terms",
            ),
            Evidence(
                evidence_id="EV002",
                source_url="https://github.com/openai/openai-python/releases",
                source_type="github_release",
                publisher="GitHub",
                published_at=None,
                retrieved_at="2026-09-21T00:00:00Z",
                content_hash="hash-github",
                locator="release:list",
                excerpt="OpenAI SDK releases document API integration updates.",
                license="source_terms",
            ),
        )
        claims = (
            AtomicClaim(
                claim_id="CL001",
                text="AIGC tools are moving from single prompts toward workflow integration.",
                claim_type="factual",
                temporal_scope="current at retrieval time",
                evidence_ids=("EV001", "EV002"),
                confidence=0.86,
                status=ClaimStatus.SUPPORTED,
                verification="verified against official release and SDK evidence",
                script_usages=("SEG001", "SEG002"),
            ),
            AtomicClaim(
                claim_id="CL002",
                text="Teams still need to check costs, permissions, and review steps before production use.",
                claim_type="factual",
                temporal_scope="current at retrieval time",
                evidence_ids=("EV001",),
                confidence=0.82,
                status=ClaimStatus.SUPPORTED,
                verification="verified against official release caveats",
                script_usages=("SEG003",),
            ),
        )
        return replace(
            project,
            status=ProjectStatus.SCRIPT_READY,
            research_bundle=ResearchBundle(
                questions=(
                    ResearchQuestion("RQ001", "AIGC 最近发生了什么变化？"),
                    ResearchQuestion("RQ002", "普通创作者为什么要关心？"),
                    ResearchQuestion("RQ003", "有哪些限制不能夸大？"),
                ),
                evidence=evidence,
            ),
            evidence_graph=EvidenceGraph(claims=claims, evidence=evidence),
            fact_check_report=FactCheckReport(
                claim_statuses={claim.claim_id: claim.status for claim in claims},
                blocking_claim_ids=(),
                notes=("all usable claims cleared",),
            ),
            content_plan=ContentPlan(
                sections=("0-3s Hook", "3-10s 发生了什么", "10-35s 原理", "35-52s 案例", "52-60s 总结"),
                target_seconds=60,
                platform_constraints=("9:16 1080x1920", "subtitles required"),
            ),
            script=ScriptDraft(
                hooks=(
                    "AIGC 不是换个聊天框这么简单。",
                    "一分钟看懂 AIGC 真正在改变什么。",
                    "别先追热点，先看它能不能进你的工作流。",
                ),
                segments=(
                    ScriptSegment("SEG001", "AIGC 正在从一次性问答，变成能接进工作流的工具。", True, ("CL001",)),
                    ScriptSegment("SEG002", "你可以把它理解成：以前是找人帮你写一句话，现在是让一套流程帮你改稿、做图、检查结果。", True, ("CL001",)),
                    ScriptSegment("SEG003", "但别急着全自动上线，成本、版权和人工审核，仍然要先看清楚。", True, ("CL002",)),
                ),
                subtitle_lines=(
                    "AIGC 正在从一次性问答，变成能接进工作流的工具。",
                    "以前是帮你写一句话，现在是帮你跑一套流程。",
                    "成本、版权和人工审核，仍然要先看清楚。",
                ),
            ),
            completed_stage_keys=frozenset({"research", "evidence_graph", "fact_check", "plan", "script"}),
        )


class RetryPreviewProductionProvider:
    def __init__(self) -> None:
        self.calls: list[ProjectStatus] = []

    async def run_content_project(self, project, *, until: ProjectStatus):
        self.calls.append(until)
        assert project.timeline is not None
        timeline = replace(project.timeline, preview_artifact_id=f"production-preview-{len(self.calls)}.mp4")
        return replace(
            project,
            timeline=timeline,
            qc_report=None,
            status=until,
            completed_stage_keys=project.completed_stage_keys | {"preview"},
        )


async def test_stale_save_cannot_overwrite_a_newer_project_revision() -> None:
    store = AsyncInMemoryContentProjectStore()
    service = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    created = await _create_demo(service)
    saved = await store.save(replace(created, title="New title"))
    assert saved.revision == created.revision + 1
    with pytest.raises(ContentProjectConflict):
        await store.save(replace(created, title="Stale title"))
    assert (await store.get(created.project_id)).title == "New title"


async def test_two_service_instances_reuse_successful_stages_when_run_concurrently() -> None:
    store = AsyncInMemoryContentProjectStore()
    first = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    second = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    created = await _create_demo(first)
    await asyncio.gather(
        first.run_content_project(created.project_id, until=ProjectStatus.SCRIPT_READY),
        second.run_content_project(created.project_id, until=ProjectStatus.SCRIPT_READY),
    )
    result = await store.get(created.project_id)
    assert result.script is not None
    assert first.provider_call_count("research") + second.provider_call_count("research") == 1
    assert first.provider_call_count("script") + second.provider_call_count("script") == 1


async def test_successful_stages_persist_when_a_later_checkpoint_fails() -> None:
    class FailOnceStore(AsyncInMemoryContentProjectStore):
        failed = False

        async def save(self, project):
            if project.status is ProjectStatus.PLAN_READY and not self.failed:
                self.failed = True
                raise OSError("simulated checkpoint failure")
            return await super().save(project)

    store = FailOnceStore()
    first = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    created = await _create_demo(first)
    with pytest.raises(OSError, match="checkpoint"):
        await first.run_content_project(created.project_id, until=ProjectStatus.SCRIPT_READY)
    saved = await store.get(created.project_id)
    assert saved.status is ProjectStatus.FACT_CHECKED
    assert saved.research_bundle is not None
    assert saved.fact_check_report is not None
    restarted = AsyncContentStudioService(registry=PackRegistry.mvp(), store=store)
    resumed = await restarted.run_content_project(created.project_id, until=ProjectStatus.SCRIPT_READY)
    assert resumed.status is ProjectStatus.SCRIPT_READY
    assert restarted.provider_call_count("research") == 0


async def test_legacy_payload_uses_secure_defaults_for_new_fields() -> None:
    service = AsyncContentStudioService(registry=PackRegistry.mvp(), store=AsyncInMemoryContentProjectStore())
    created = await _create_demo(service)
    payload = content_project_to_payload(created)
    raw_project = payload["project"]
    assert isinstance(raw_project, dict)
    for key in ("revision", "tenant_id", "owner_user_id", "execution_mode", "script_approved", "rights_approved", "final_approved"):
        raw_project.pop(key, None)
    restored = content_project_from_payload(payload)
    assert restored.execution_mode == "production"
    assert restored.revision == 0
    assert restored.tenant_id == ""
    assert restored.owner_user_id == ""
    assert not restored.script_approved
    assert not restored.rights_approved
    assert not restored.final_approved
