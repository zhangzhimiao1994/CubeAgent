# mypy: disable-error-code="index, union-attr"

from __future__ import annotations

from dataclasses import replace

import pytest

from agent_hub.content_studio import (
    AssetManifest,
    ClaimStatus,
    ContentStudioService,
    InMemoryContentProjectStore,
    PackRegistry,
    ProjectStatus,
)


def test_content_project_locks_pack_versions_and_runs_to_qc_with_fake_provider() -> None:
    registry = PackRegistry.mvp()
    store = InMemoryContentProjectStore()
    service = ContentStudioService(registry=registry, store=store)

    project = service.create_content_project(
        title="AIGC release explainer",
        topic="Explain the new AIGC model release",
        source_urls=("https://example.com/official-release",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    registry.register_domain_pack(registry.domain_packs["aigc"].with_version("2099.1.0"))

    script_ready = service.run_content_project(project.project_id, until=ProjectStatus.QC_REVIEW)
    assert script_ready.status is ProjectStatus.SCRIPT_READY
    assert script_ready.storyboard is None

    service.approve_script(project.project_id)
    assets = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert assets.asset_manifest is not None
    service.approve_rights(project.project_id, asset_ids=tuple(
        asset.asset_id for asset in assets.asset_manifest.assets
    ), note="Test fixture rights reviewed")
    result = service.run_content_project(project.project_id, until=ProjectStatus.QC_REVIEW)

    assert result.packs.domain.version == "1.0.0"
    assert result.status is ProjectStatus.QC_REVIEW
    assert result.research_bundle is not None
    assert result.evidence_graph is not None
    assert result.fact_check_report is not None
    assert all(claim.status is ClaimStatus.SUPPORTED for claim in result.evidence_graph.claims)
    assert result.script is not None
    assert len(result.script.hooks) == 3
    assert all(segment.claim_ids for segment in result.script.segments if segment.factual)
    assert result.storyboard is not None
    assert all(shot.asset_request_ids for shot in result.storyboard.shots)
    assert result.asset_manifest is not None
    assert all(asset.rights_status == "approved" for asset in result.asset_manifest.assets)
    assert result.timeline is not None
    assert result.timeline.width == 1080
    assert result.timeline.height == 1920
    assert result.qc_report is not None
    assert not result.qc_report.blockers


def test_unsupported_claim_blocks_before_script_and_can_resume_after_evidence_fix() -> None:
    registry = PackRegistry.mvp()
    service = ContentStudioService(registry=registry, store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Unsafe AIGC explainer",
        topic="unsupported claim: the model is free forever",
        source_urls=("https://example.com/blog",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )

    blocked = service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)

    assert blocked.status is ProjectStatus.FAILED_BLOCKED
    assert blocked.fact_check_report is not None
    assert blocked.fact_check_report.blocking_claim_ids
    assert blocked.script is None
    first_research_count = service.provider_call_count("research")

    service.replace_claim_status(
        project.project_id,
        claim_id=blocked.fact_check_report.blocking_claim_ids[0],
        status=ClaimStatus.SUPPORTED,
        note="manual official source added",
        evidence_ids=("EV001",),
    )
    resumed = service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)

    assert resumed.status is ProjectStatus.SCRIPT_READY
    assert resumed.script is not None
    assert service.provider_call_count("research") == first_research_count


def test_stage_rerun_is_idempotent_and_does_not_repeat_expensive_generation() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="AIGC tutorial",
        topic="Teach users how to compare model release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="tutorial",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )

    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    first_asset_calls = service.provider_call_count("assets")
    first_project = service.get_content_project(project.project_id)

    service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    second_project = service.get_content_project(project.project_id)

    assert first_project.asset_manifest is not None
    assert second_project.asset_manifest is not None
    assert first_project.asset_manifest == second_project.asset_manifest
    assert service.provider_call_count("assets") == first_asset_calls


def test_research_bundle_records_authoritative_source_coverage() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="AIGC research depth",
        topic="Explain a current AIGC release with official evidence",
        source_urls=("https://docs.example.com/aigc/release",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )

    researched = service.run_content_project(project.project_id, until=ProjectStatus.RESEARCH_READY)

    assert researched.research_bundle is not None
    source_types = {evidence.source_type for evidence in researched.research_bundle.evidence}
    assert {
        "official_docs",
        "official_blog",
        "release_notes",
        "github_release",
        "paper",
        "official_demo",
    }.issubset(source_types)
    coverage = {
        item.source_type: item.status for item in researched.research_bundle.source_coverage
    }
    assert coverage["official_docs"] == "covered"
    assert coverage["github_release"] == "covered"
    assert researched.research_bundle.retrieval_plan
    assert len(researched.research_bundle.source_candidates) >= 12


def test_run_stops_at_assets_until_rights_are_approved() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Stage separated project",
        topic="Explain why staged production needs reviews",
        source_urls=("https://docs.example.com/staged-production",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )

    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    staged = service.run_content_project(project.project_id, until=ProjectStatus.QC_REVIEW)

    assert staged.status is ProjectStatus.ASSETS_READY
    assert staged.asset_manifest is not None
    assert staged.voice_track is None
    assert staged.timeline is None
    assert staged.qc_report is None


def test_dialog_operations_revise_retry_and_approve_without_full_rerun() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="AIGC news",
        topic="Summarize the official AIGC launch notes",
        source_urls=("https://example.com/official-launch",),
        domain="aigc",
        format="news",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    project = service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    project = service.approve_script(project.project_id)
    project = service.run_content_project(project.project_id, until=ProjectStatus.QC_REVIEW)
    first_asset_calls = service.provider_call_count("assets")
    assert project.asset_manifest is not None
    target_asset_id = project.asset_manifest.assets[1].asset_id

    revised = service.revise_script(project.project_id, instruction="换一个更强但不夸张的开头")

    assert revised.status is ProjectStatus.SCRIPT_READY
    assert revised.script is not None
    assert revised.script.hooks[0].startswith("更强但不夸张")
    assert revised.storyboard is None
    assert revised.asset_manifest is None
    assert service.provider_call_count("research") == 1

    approved = service.approve_script(project.project_id)
    assert approved.status is ProjectStatus.SCRIPT_APPROVED

    regenerated = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert service.provider_call_count("assets") == first_asset_calls + 1
    assert regenerated.asset_manifest is not None
    replaced = service.regenerate_asset(
        project.project_id,
        asset_id=regenerated.asset_manifest.assets[1].asset_id,
        instruction="第 2 个镜头不要数字人，换成产品录屏",
    )

    assert replaced.asset_manifest is not None
    assert replaced.asset_manifest.assets[1].content_hash != regenerated.asset_manifest.assets[1].content_hash
    assert replaced.asset_manifest.assets[0] == regenerated.asset_manifest.assets[0]
    assert regenerated.asset_manifest.assets[1].asset_id == replaced.asset_manifest.assets[1].asset_id
    assert target_asset_id == replaced.asset_manifest.assets[1].asset_id

    service.approve_rights(project.project_id, asset_ids=tuple(
        asset.asset_id for asset in replaced.asset_manifest.assets
    ), note="Revised fixtures reviewed")
    preview = service.render_preview(project.project_id)
    assert preview.status is ProjectStatus.PREVIEW_RENDERED
    assert preview.timeline is not None
    assert preview.timeline.preview_artifact_id is not None

    service.run_content_project(project.project_id, until=ProjectStatus.QC_REVIEW)
    final_ready = service.approve_final(project.project_id)
    assert final_ready.status is ProjectStatus.FINAL_APPROVED


def test_script_approval_is_required_before_any_media_generation() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="AIGC guarded flow",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )

    blocked = service.run_content_project(project.project_id, until=ProjectStatus.PREVIEW_RENDERED)

    assert blocked.status is ProjectStatus.SCRIPT_READY
    assert blocked.script is not None
    assert blocked.script_approved is False
    assert blocked.storyboard is None
    assert blocked.asset_manifest is None
    assert blocked.voice_track is None
    assert blocked.timeline is None
    assert service.provider_call_count("storyboard") == 0
    assert service.provider_call_count("assets") == 0
    assert service.provider_call_count("voice") == 0
    current = service.get_content_project(project.project_id)
    assert current.status is ProjectStatus.SCRIPT_READY


def test_approve_script_only_approves_existing_script_without_generating_it() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="AIGC guarded approval",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )

    with pytest.raises(ValueError, match="script"):
        service.approve_script(project.project_id)

    current = service.get_content_project(project.project_id)
    assert current.script is None
    assert current.status is ProjectStatus.DRAFT
    assert service.provider_call_count("script") == 0


def test_approve_script_rejects_missing_or_unsupported_factual_claims() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Script claim gate",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    script_ready = service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    assert script_ready.script is not None
    assert script_ready.evidence_graph is not None
    unsupported_graph = replace(
        script_ready.evidence_graph,
        claims=(
            replace(script_ready.evidence_graph.claims[0], status=ClaimStatus.OPINION),
        ),
    )
    service._store.save(replace(script_ready, evidence_graph=unsupported_graph))

    with pytest.raises(ValueError, match="factual"):
        service.approve_script(project.project_id)

    missing_claim_segment = replace(script_ready.script.segments[0], claim_ids=("missing-claim",))
    service._store.save(
        replace(
            script_ready,
            script=replace(
                script_ready.script,
                segments=(missing_claim_segment, *script_ready.script.segments[1:]),
            ),
        )
    )
    with pytest.raises(ValueError, match="claim"):
        service.approve_script(project.project_id)


def test_production_mode_blocks_unconfigured_research_without_fake_outputs() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Production video",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
    )

    blocked = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)

    assert project.execution_mode == "production"
    assert blocked.status is ProjectStatus.FAILED_BLOCKED
    assert blocked.error_code == "provider_not_configured"
    assert blocked.research_bundle is None
    assert blocked.evidence_graph is None
    assert blocked.script is None
    assert blocked.storyboard is None
    assert blocked.asset_manifest is None
    assert service.provider_call_count("research") == 0
    assert service.provider_call_count("assets") == 0
    assert "fake" not in repr(blocked)


def test_claim_status_replacement_requires_valid_evidence_and_outdated_blocks_production() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Claim evidence gate",
        topic="unsupported claim: the model is free forever",
        source_urls=("https://example.com/blog",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    blocked = service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    assert blocked.fact_check_report is not None

    with pytest.raises(ValueError, match="evidence"):
        service.replace_claim_status(
            project.project_id,
            claim_id=blocked.fact_check_report.blocking_claim_ids[0],
            status=ClaimStatus.SUPPORTED,
            note="manual note without evidence",
        )

    service.replace_claim_status(
        project.project_id,
        claim_id=blocked.fact_check_report.blocking_claim_ids[0],
        status=ClaimStatus.OUTDATED,
        note="official source shows this is outdated",
        evidence_ids=("EV001",),
    )
    still_blocked = service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    assert still_blocked.status is ProjectStatus.FAILED_BLOCKED
    assert still_blocked.script is None


def test_production_cannot_manually_replace_claim_status_without_real_verifier() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Production claim gate",
        topic="unsupported claim: the model is free forever",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    blocked = service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    assert blocked.fact_check_report is not None
    service._store.save(replace(blocked, execution_mode="production"))

    with pytest.raises(ValueError, match="verifier"):
        service.replace_claim_status(
            project.project_id,
            claim_id=blocked.fact_check_report.blocking_claim_ids[0],
            status=ClaimStatus.SUPPORTED,
            note="human reviewed",
            evidence_ids=("EV001",),
        )


def test_asset_rights_approval_is_asset_scoped_and_restricted_assets_stay_blocked() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Rights scoped flow",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    assets_ready = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert assets_ready.asset_manifest is not None
    assert {asset.rights_status for asset in assets_ready.asset_manifest.assets} == {"unknown"}

    blocked = service.run_content_project(project.project_id, until=ProjectStatus.VOICE_READY)
    assert blocked.status is ProjectStatus.ASSETS_READY
    assert blocked.error_code is None
    assert blocked.voice_track is None

    first_asset = assets_ready.asset_manifest.assets[0]
    partially_approved = service.approve_rights(
        project.project_id,
        asset_ids=(first_asset.asset_id,),
        note="source reviewed",
    )
    assert partially_approved.rights_approved is False
    still_blocked = service.run_content_project(project.project_id, until=ProjectStatus.VOICE_READY)
    assert still_blocked.status is ProjectStatus.ASSETS_READY
    assert still_blocked.voice_track is None

    all_ids = tuple(asset.asset_id for asset in assets_ready.asset_manifest.assets)
    approved = service.approve_rights(project.project_id, asset_ids=all_ids, note="all reviewed")
    assert approved.rights_approved is True
    voice_ready = service.run_content_project(project.project_id, until=ProjectStatus.VOICE_READY)
    assert voice_ready.status is ProjectStatus.VOICE_READY
    assert voice_ready.voice_track is not None

    restricted_assets = tuple(
        replace(asset, rights_status="restricted") if index == 0 else asset
        for index, asset in enumerate(voice_ready.asset_manifest.assets)
    )
    service._store.save(
        replace(
            voice_ready,
            status=ProjectStatus.ASSETS_READY,
            asset_manifest=AssetManifest(assets=restricted_assets),
            rights_approved=False,
            voice_track=None,
            timeline=None,
            qc_report=None,
        )
    )
    with pytest.raises(ValueError, match="restricted"):
        service.approve_rights(project.project_id, asset_ids=all_ids, note="generic approval")


def test_pack_lock_deep_copies_mutable_manifest_settings() -> None:
    registry = PackRegistry.mvp()
    service = ContentStudioService(registry=registry, store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Pack immutability",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )

    registry.platform_packs["douyin"].settings["target_seconds"] = 45
    registry.domain_packs["aigc"].settings["source_priority"] = ("mutated",)
    locked = service.get_content_project(project.project_id)

    assert locked.packs.platform.settings["target_seconds"] == 60
    assert locked.packs.domain.settings["source_priority"][0] == "official_docs"


def test_plan_storyboard_and_timeline_use_locked_platform_settings() -> None:
    registry = PackRegistry.mvp()
    registry.platform_packs["douyin"].settings.update({
        **registry.platform_packs["douyin"].settings,
        "width": 720,
        "height": 1280,
        "target_seconds": 45,
        "aspect_ratio": "9:16",
    })
    service = ContentStudioService(registry=registry, store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Platform settings",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    assets_ready = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    service.approve_rights(
        project.project_id,
        asset_ids=tuple(asset.asset_id for asset in assets_ready.asset_manifest.assets),
        note="all reviewed",
    )
    timeline_ready = service.run_content_project(project.project_id, until=ProjectStatus.TIMELINE_READY)

    assert timeline_ready.content_plan is not None
    assert timeline_ready.content_plan.target_seconds == 45
    assert timeline_ready.timeline is not None
    assert timeline_ready.timeline.width == 720
    assert timeline_ready.timeline.height == 1280
    assert timeline_ready.timeline.duration_ms == 45000


def test_final_approval_requires_existing_clean_preview_and_final_render_is_separate() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Final approval flow",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)

    with pytest.raises(ValueError, match="preview"):
        service.approve_final(project.project_id)

    assets_ready = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    service.approve_rights(
        project.project_id,
        asset_ids=tuple(asset.asset_id for asset in assets_ready.asset_manifest.assets),
        note="all reviewed",
    )
    preview = service.render_preview(project.project_id)
    with pytest.raises(ValueError, match="QC"):
        service.approve_final(project.project_id)
    assert service.provider_call_count("qc") == 0
    service.run_content_project(project.project_id, until=ProjectStatus.QC_REVIEW)
    approved = service.approve_final(project.project_id)

    assert preview.status is ProjectStatus.PREVIEW_RENDERED
    assert approved.status is ProjectStatus.FINAL_APPROVED
    assert approved.final_approved is True
    assert approved.timeline is not None
    assert approved.timeline.preview_artifact_id is not None
    assert approved.timeline.final_artifact_id is None

    rendered = service.run_content_project(project.project_id, until=ProjectStatus.FINAL_RENDERED)
    assert rendered.status is ProjectStatus.FINAL_RENDERED
    assert rendered.timeline is not None
    assert rendered.timeline.final_artifact_id is not None


def test_storyboard_and_asset_changes_revoke_affected_approvals_but_keep_voice_when_safe() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore())
    project = service.create_content_project(
        title="Revision flow",
        topic="Explain official AIGC release notes",
        source_urls=("https://example.com/docs",),
        domain="aigc",
        format="explainer",
        platform="douyin",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    assets_ready = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    service.approve_rights(
        project.project_id,
        asset_ids=tuple(asset.asset_id for asset in assets_ready.asset_manifest.assets),
        note="all reviewed",
    )
    voice_ready = service.run_content_project(project.project_id, until=ProjectStatus.VOICE_READY)
    assert voice_ready.voice_track is not None

    storyboard_revised = service.revise_storyboard(project.project_id, instruction="更换第二镜头画面")
    assert storyboard_revised.rights_approved is False
    assert storyboard_revised.final_approved is False
    assert storyboard_revised.voice_track == voice_ready.voice_track
    assert storyboard_revised.timeline is None

    regenerated = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert regenerated.asset_manifest is not None
    asset_revised = service.regenerate_asset(
        project.project_id,
        asset_id=regenerated.asset_manifest.assets[0].asset_id,
        instruction="换成产品录屏",
    )
    assert asset_revised.rights_approved is False
    assert asset_revised.voice_track == voice_ready.voice_track

    voice_regenerated = service.regenerate_voice(project.project_id, instruction="语速更稳")
    assert voice_regenerated.voice_track is not None
    assert voice_regenerated.voice_track != voice_ready.voice_track
    assert voice_regenerated.timeline is None


def test_unknown_asset_retry_does_not_generate_a_missing_asset_manifest() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore(), execution_mode="demo")
    project = service.create_content_project(
        title="Missing asset", topic="Supported release", source_urls=("https://example.com/docs",),
        domain="aigc", format="explainer", platform="douyin", channel="ai_frontier", style="fast_minimal",
    )
    with pytest.raises(ValueError, match="assets"):
        service.regenerate_asset(project.project_id, asset_id="missing", instruction="Retry")
    assert service.provider_call_count("research") == 0
    assert service.get_content_project(project.project_id) == project


@pytest.mark.parametrize("operation", ["revise_script", "revise_storyboard", "regenerate_asset"])
def test_production_edits_cannot_emit_demo_artifacts_even_with_existing_outputs(operation: str) -> None:
    store = InMemoryContentProjectStore()
    service = ContentStudioService(registry=PackRegistry.mvp(), store=store, execution_mode="demo")
    project = service.create_content_project(
        title="Existing production outputs", topic="Supported release", source_urls=("https://example.com/docs",),
        domain="aigc", format="explainer", platform="douyin", channel="ai_frontier", style="fast_minimal",
    )
    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    original = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    store.save(replace(original, execution_mode="production"))
    assert original.asset_manifest is not None
    if operation == "regenerate_asset":
        result = service.regenerate_asset(project.project_id, asset_id=original.asset_manifest.assets[0].asset_id, instruction="Retry")
    else:
        result = getattr(service, operation)(project.project_id, instruction="Revise")
    assert result.status is ProjectStatus.FAILED_BLOCKED
    assert result.error_code == "provider_not_configured"
    assert result.script == original.script
    assert result.storyboard == original.storyboard
    assert result.asset_manifest == original.asset_manifest


def test_repeat_run_and_approvals_do_not_regress_completed_status() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore(), execution_mode="demo")
    project = service.create_content_project(
        title="Idempotent approvals", topic="Supported release", source_urls=("https://example.com/docs",),
        domain="aigc", format="explainer", platform="douyin", channel="ai_frontier", style="fast_minimal",
    )
    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    project = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert project.asset_manifest is not None
    ids = tuple(asset.asset_id for asset in project.asset_manifest.assets)
    service.approve_rights(project.project_id, asset_ids=ids, note="Fixture reviewed")
    service.run_content_project(project.project_id, until=ProjectStatus.QC_REVIEW)
    service.approve_final(project.project_id)
    final = service.run_content_project(project.project_id, until=ProjectStatus.FINAL_RENDERED)
    assert service.run_content_project(project.project_id, until=ProjectStatus.RESEARCH_READY) == final
    assert service.approve_script(project.project_id) == final
    assert service.approve_rights(project.project_id, asset_ids=ids, note="Same approval") == final
    assert service.approve_final(project.project_id) == final
    assert service.retry_stage(project.project_id, ProjectStatus.ASSETS_READY) == final


def test_cached_voice_does_not_clear_new_asset_rights_gate() -> None:
    service = ContentStudioService(registry=PackRegistry.mvp(), store=InMemoryContentProjectStore(), execution_mode="demo")
    project = service.create_content_project(
        title="Rights changed", topic="Supported release", source_urls=("https://example.com/docs",),
        domain="aigc", format="explainer", platform="douyin", channel="ai_frontier", style="fast_minimal",
    )
    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    assets = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert assets.asset_manifest is not None
    ids = tuple(asset.asset_id for asset in assets.asset_manifest.assets)
    service.approve_rights(project.project_id, asset_ids=ids, note="Fixture reviewed")
    voiced = service.run_content_project(project.project_id, until=ProjectStatus.VOICE_READY)
    service.regenerate_asset(project.project_id, asset_id=ids[0], instruction="Replace first visual")
    blocked = service.run_content_project(project.project_id, until=ProjectStatus.VOICE_READY)
    assert blocked.error_code is None
    assert blocked.status is ProjectStatus.ASSETS_READY
    assert blocked.voice_track == voiced.voice_track
    assert service.provider_call_count("voice") == 1


def test_retry_rechecks_failed_qc_without_regenerating_successful_assets() -> None:
    store = InMemoryContentProjectStore()
    service = ContentStudioService(registry=PackRegistry.mvp(), store=store, execution_mode="demo")
    project = service.create_content_project(
        title="Retry failed QC", topic="Supported release", source_urls=("https://example.com/docs",),
        domain="aigc", format="explainer", platform="douyin", channel="ai_frontier", style="fast_minimal",
    )
    service.run_content_project(project.project_id, until=ProjectStatus.SCRIPT_READY)
    service.approve_script(project.project_id)
    assets = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert assets.asset_manifest is not None
    service.approve_rights(project.project_id, asset_ids=tuple(
        asset.asset_id for asset in assets.asset_manifest.assets
    ), note="Fixture reviewed")
    ready = service.run_content_project(project.project_id, until=ProjectStatus.QC_REVIEW)
    assert ready.qc_report is not None
    failed_report = replace(ready.qc_report, blockers=("simulated transient check failure",))
    store.save(replace(ready, qc_report=failed_report, status=ProjectStatus.FAILED_RETRYABLE))
    retried = service.retry_stage(project.project_id, ProjectStatus.QC_REVIEW)
    assert retried.qc_report != failed_report
    assert retried.status is ProjectStatus.QC_REVIEW
    assert retried.asset_manifest == ready.asset_manifest
    assert retried.voice_track == ready.voice_track
    assert retried.timeline == ready.timeline
    assert service.provider_call_count("assets") == 1
