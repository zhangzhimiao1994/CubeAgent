# mypy: disable-error-code="index, attr-defined"

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_hub.content_studio import (
    ContentStudioService,
    InMemoryContentProjectStore,
    ProjectStatus,
)
from agent_hub.content_studio.packs import (
    CORE_VERSION,
    PackManifestError,
    bundled_pack_manifest_dir,
    load_pack_registry,
)


def test_bundled_pack_registry_loads_mvp_packs_and_manifest_only_platform_extension() -> None:
    registry = load_pack_registry(bundled_pack_manifest_dir())

    assert set(registry.domain_packs) == {"aigc"}
    assert set(registry.format_packs) == {"explainer", "news", "tutorial"}
    assert {"douyin", "xiaohongshu"} <= set(registry.platform_packs)
    assert set(registry.channel_packs) == {"ai_frontier"}
    assert set(registry.style_packs) == {"fast_minimal"}

    locked = registry.lock(
        domain="aigc",
        format="explainer",
        platform="xiaohongshu",
        channel="ai_frontier",
        style="fast_minimal",
    )

    assert locked.platform.name == "xiaohongshu"
    assert locked.platform.settings["aspect_ratio"] == "3:4"
    allowed_hosts = locked.domain.settings["allowed_hosts"]
    assert isinstance(allowed_hosts, list)
    assert set(allowed_hosts) >= {
        "openai.com",
        "github.com",
        "arxiv.org",
        "huggingface.co",
        "anthropic.com",
        "ai.google.dev",
        "deepmind.google",
        "microsoft.com",
        "azure.microsoft.com",
        "github.blog",
        "nvidia.com",
        "stability.ai",
        "runwayml.com",
        "minimax.io",
        "qwenlm.github.io",
        "alibabacloud.com",
        "cloud.tencent.com",
        "baidu.com",
    }
    source_types = locked.domain.settings["source_types"]
    assert isinstance(source_types, dict)
    assert source_types["github.blog"] == "official_blog"
    assert source_types["arxiv.org"] == "paper"
    assert locked.domain.settings["default_license"] == "source_terms"


def test_manifest_only_platform_runs_through_timeline_with_pack_settings() -> None:
    registry = load_pack_registry(bundled_pack_manifest_dir())
    service = ContentStudioService(registry=registry, store=InMemoryContentProjectStore())

    project = service.create_content_project(
        title="AIGC explainer",
        topic="Explain an official launch",
        source_urls=("https://example.com/official-launch",),
        domain="aigc",
        format="explainer",
        platform="xiaohongshu",
        channel="ai_frontier",
        style="fast_minimal",
        execution_mode="demo",
    )
    script_ready = service.run_content_project(project.project_id)
    assert script_ready.status.name == "SCRIPT_READY"

    service.approve_script(project.project_id)
    rights_ready = service.run_content_project(project.project_id, until=ProjectStatus.ASSETS_READY)
    assert rights_ready.status.name == "ASSETS_READY"
    assert rights_ready.asset_manifest is not None
    blocked = service.run_content_project(project.project_id, until=ProjectStatus.TIMELINE_READY)
    assert blocked.status.name == "ASSETS_READY"
    assert blocked.error_code is None
    assert blocked.timeline is None
    rights_ready = service.approve_rights(
        project.project_id,
        asset_ids=tuple(asset.asset_id for asset in rights_ready.asset_manifest.assets),
        note="reviewed",
    )
    assert rights_ready.rights_approved is True
    result = service.run_content_project(project.project_id, until=ProjectStatus.TIMELINE_READY)

    assert result.content_plan is not None
    assert result.content_plan.target_seconds == 45
    assert "3:4 1080x1440" in result.content_plan.platform_constraints
    assert result.storyboard is not None
    assert sum(shot.duration_ms for shot in result.storyboard.shots) == 45_000
    assert result.timeline is not None
    assert result.timeline.width == 1080
    assert result.timeline.height == 1440
    assert result.timeline.duration_ms == 45_000


def test_loaded_registry_uses_normal_pack_registry_lock_method() -> None:
    registry = load_pack_registry(bundled_pack_manifest_dir())

    assert registry.lock.__self__ is registry
    assert type(registry).lock is registry.lock.__func__


def test_pack_manifest_validation_rejects_incompatible_core(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "packs"
    manifest_dir.mkdir()
    manifest = _valid_manifest(pack_type="domain", name="future_domain")
    manifest["compatible_core"] = ">=9.0.0"
    (manifest_dir / "future-domain.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(PackManifestError, match="compatible_core"):
        load_pack_registry(manifest_dir, core_version="0.1.0")


def test_pack_manifest_validation_rejects_malformed_manifest(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "packs"
    manifest_dir.mkdir()
    manifest = _valid_manifest(pack_type="platform", name="bad_platform")
    manifest["settings"]["width"] = "1080"
    (manifest_dir / "bad-platform.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(PackManifestError, match="settings.width"):
        load_pack_registry(manifest_dir)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"min_seconds": 70, "target_seconds": 60, "max_seconds": 90}, "min_seconds"),
        ({"aspect_ratio": "1:1"}, "aspect_ratio"),
        ({"visual_change_seconds": [7, 3]}, "visual_change_seconds"),
        ({"hook_seconds": 60, "target_seconds": 60}, "hook_seconds"),
    ),
)
def test_pack_manifest_validation_rejects_invalid_platform_constraints(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    manifest_dir = tmp_path / "packs"
    manifest_dir.mkdir()
    for pack_type in ("domain", "format", "channel", "style"):
        (manifest_dir / f"{pack_type}.json").write_text(
            json.dumps(_valid_manifest(pack_type=pack_type, name=pack_type), ensure_ascii=False),
            encoding="utf-8",
        )
    manifest = _valid_manifest(pack_type="platform", name="bad_platform")
    settings = manifest["settings"]
    assert isinstance(settings, dict)
    settings.update(updates)
    (manifest_dir / "platform.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(PackManifestError, match=message):
        load_pack_registry(manifest_dir)


def _valid_manifest(*, pack_type: str, name: str) -> dict[str, object]:
    settings_by_type: dict[str, dict[str, object]] = {
        "domain": {
            "source_priority": ["official_docs"],
            "fact_rules": ["temporal_scope_required"],
            "allowed_hosts": ["openai.com"],
            "official_hosts": ["openai.com"],
            "source_types": {"openai.com": "official_docs"},
            "default_license": "source_terms",
        },
        "format": {"structure": "explainer"},
        "platform": {
            "aspect_ratio": "9:16",
            "width": 1080,
            "height": 1920,
            "min_seconds": 45,
            "max_seconds": 90,
            "target_seconds": 60,
            "codec": "H.264/AAC",
            "subtitle_required": True,
            "hook_seconds": 3,
            "visual_change_seconds": [3, 5],
        },
        "channel": {
            "persona": "calm AI frontier explainer",
            "audience": "AI builders",
            "banned_phrases": ["稳赚"],
        },
        "style": {
            "subtitle_style": "large safe-area captions",
            "visual_language": "screen capture",
            "transition": "quick cut",
        },
    }
    return {
        "manifest": "content_studio.pack.v1",
        "pack_type": pack_type,
        "name": name,
        "version": "1.0.0",
        "schema_version": "1.0",
        "compatible_core": f">={CORE_VERSION},<1.0.0",
        "settings": settings_by_type[pack_type],
    }
