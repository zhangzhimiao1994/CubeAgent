from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from agent_hub.app import create_app
from agent_hub.auth.models import AuthenticatedPrincipal, InvalidCredentials, Role
from agent_hub.content_studio import (
    AssetManifest,
    AsyncContentStudioService,
    AsyncInMemoryContentProjectStore,
    ClaimStatus,
    ContentProject,
    ContentProjectSummary,
    ContentStudioService,
    InMemoryContentProjectStore,
    PackRegistry,
    ProjectStatus,
    VoiceTrack,
    content_project_summary,
)


class StubAuthService:
    def __init__(self) -> None:
        self.principal = AuthenticatedPrincipal(uuid4(), uuid4(), Role.ADMIN)

    def authenticate_token(self, token: str) -> AuthenticatedPrincipal:
        if token != "valid-token":
            raise InvalidCredentials("invalid")
        return self.principal

    def become(self, principal: AuthenticatedPrincipal) -> None:
        self.principal = principal


def test_content_studio_project_api_create_run_and_revise_script(tmp_path: Path) -> None:
    service = PrincipalScopedContentStudioService(media_dir=tmp_path)
    auth = StubAuthService()
    client = TestClient(
        create_app(
            auth_service=auth,
            content_studio_service=service,
        )
    )
    headers = {"Authorization": "Bearer valid-token"}

    created = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={
            "title": "AIGC explainer",
            "topic": "Explain official AIGC launch notes",
            "source_urls": ["https://example.com/official-launch"],
        },
    )
    assert created.status_code == 201
    project_id = created.json()["project_id"]

    ran = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "SCRIPT_READY"},
    )
    assert ran.status_code == 200
    assert ran.json()["status"] == "SCRIPT_READY"
    assert ran.json()["script"]["hooks"]

    revised = client.post(
        f"/api/v1/content-studio/projects/{project_id}/revise-script",
        headers=headers,
        json={"instruction": "换一个更强但不夸张的开头"},
    )
    assert revised.status_code == 200
    assert revised.json()["status"] == "SCRIPT_READY"
    assert revised.json()["script"]["hooks"][0].startswith("更强但不夸张")

    approved = client.post(
        f"/api/v1/content-studio/projects/{project_id}/approve-script",
        headers=headers,
        json={"revision": revised.json()["revision"]},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "SCRIPT_APPROVED"

    storyboard_ready = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "STORYBOARD_READY"},
    )
    assert storyboard_ready.status_code == 200

    storyboard = client.post(
        f"/api/v1/content-studio/projects/{project_id}/revise-storyboard",
        headers=headers,
        json={"instruction": "第一个镜头改成官方 Demo 录屏"},
    )
    assert storyboard.status_code == 200
    assert storyboard.json()["status"] == "STORYBOARD_READY"
    assert storyboard.json()["storyboard"]["shots"][0]["overlay"] == "第一个镜头改成官方 Demo 录屏"

    assets = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "ASSETS_READY"},
    )
    assert assets.status_code == 200
    asset_id = assets.json()["asset_manifest"]["assets"][0]["asset_id"]

    regenerated = client.post(
        f"/api/v1/content-studio/projects/{project_id}/regenerate-asset",
        headers=headers,
        json={"asset_id": asset_id, "instruction": "只重做这个素材"},
    )
    assert regenerated.status_code == 200
    assert regenerated.json()["status"] == "ASSETS_READY"
    assert regenerated.json()["asset_manifest"]["assets"][0]["asset_id"] == asset_id

    rights = client.post(
        f"/api/v1/content-studio/projects/{project_id}/approve-rights",
        headers=headers,
        json={
            "revision": regenerated.json()["revision"],
            "asset_ids": [
                asset["asset_id"] for asset in regenerated.json()["asset_manifest"]["assets"]
            ],
            "note": "人工确认素材版权可用于本视频。",
        },
    )
    assert rights.status_code == 200
    assert rights.json()["asset_manifest"]["assets"][0]["rights_status"] == "approved"

    preview = client.post(
        f"/api/v1/content-studio/projects/{project_id}/render-preview",
        headers=headers,
    )
    assert preview.status_code == 200
    assert preview.json()["status"] == "PREVIEW_RENDERED"
    assert preview.json()["timeline"]["preview_artifact_id"]

    preview_download = client.get(
        f"/api/v1/content-studio/projects/{project_id}/media/preview/download",
        headers=headers,
    )
    assert preview_download.status_code == 200
    assert preview_download.headers["content-type"].startswith("video/mp4")
    assert preview_download.content

    final = client.post(
        f"/api/v1/content-studio/projects/{project_id}/approve-final",
        headers=headers,
        json={"revision": preview.json()["revision"]},
    )
    assert final.status_code == 409

    qc = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "QC_REVIEW"},
    )
    assert qc.status_code == 200

    final = client.post(
        f"/api/v1/content-studio/projects/{project_id}/approve-final",
        headers=headers,
        json={"revision": qc.json()["revision"]},
    )
    assert final.status_code == 200
    assert final.json()["status"] == "FINAL_APPROVED"


def test_content_studio_project_api_unblocks_claim_and_continues() -> None:
    service = AsyncContentStudioService(
        registry=PackRegistry.mvp(),
        store=AsyncInMemoryContentProjectStore(),
        execution_mode="demo",
    )
    client = TestClient(
        create_app(
            auth_service=StubAuthService(),
            content_studio_service=service,
        )
    )
    headers = {"Authorization": "Bearer valid-token"}

    created = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={
            "title": "Blocked explainer",
            "topic": "unsupported claim about an AIGC launch",
            "source_urls": ["https://example.com/official-launch"],
        },
    )
    assert created.status_code == 201
    project_id = created.json()["project_id"]

    blocked = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "SCRIPT_READY"},
    )
    assert blocked.status_code == 200
    assert blocked.json()["status"] == "FAILED_BLOCKED"

    unblocked = client.post(
        f"/api/v1/content-studio/projects/{project_id}/claims/CL001",
        headers=headers,
        json={
            "status": "supported",
            "note": "人工补充官方证据后可使用。",
            "evidence_ids": ["EV001"],
        },
    )
    assert unblocked.status_code == 200
    assert unblocked.json()["status"] == "FACT_CHECKED"

    continued = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "SCRIPT_READY"},
    )
    assert continued.status_code == 200
    assert continued.json()["status"] == "SCRIPT_READY"
    assert continued.json()["script"]["segments"][0]["claim_ids"] == ["CL001"]


def test_content_studio_claim_support_requires_valid_evidence() -> None:
    service = AsyncContentStudioService(
        registry=PackRegistry.mvp(),
        store=AsyncInMemoryContentProjectStore(),
        execution_mode="demo",
    )
    client = TestClient(create_app(auth_service=StubAuthService(), content_studio_service=service))
    headers = {"Authorization": "Bearer valid-token"}

    created = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={
            "title": "Blocked explainer",
            "topic": "unsupported claim about an AIGC launch",
            "source_urls": ["https://example.com/official-launch"],
        },
    )
    project_id = created.json()["project_id"]
    client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "SCRIPT_READY"},
    )

    missing_evidence = client.post(
        f"/api/v1/content-studio/projects/{project_id}/claims/CL001",
        headers=headers,
        json={"status": "supported", "note": "只有人工备注，没有有效证据。"},
    )
    assert missing_evidence.status_code == 409


def test_content_studio_api_scopes_projects_to_principal_user_and_tenant() -> None:
    service = PrincipalScopedContentStudioService()
    auth = StubAuthService()
    owner = auth.principal
    same_tenant_other_user = AuthenticatedPrincipal(uuid4(), owner.tenant_id, Role.ADMIN)
    other_tenant = AuthenticatedPrincipal(uuid4(), uuid4(), Role.ADMIN)
    client = TestClient(create_app(auth_service=auth, content_studio_service=service))
    headers = {"Authorization": "Bearer valid-token"}

    created = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={"title": "Scoped project", "topic": "Show tenant isolation"},
    )
    assert created.status_code == 201
    project_id = created.json()["project_id"]
    assert created.json()["tenant_id"] == str(owner.tenant_id)
    assert created.json()["owner_user_id"] == str(owner.user_id)

    auth.become(same_tenant_other_user)
    same_tenant = client.get(f"/api/v1/content-studio/projects/{project_id}", headers=headers)
    assert same_tenant.status_code == 404

    auth.become(other_tenant)
    cross_tenant = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "SCRIPT_READY"},
    )
    assert cross_tenant.status_code == 404


def test_content_studio_api_lists_only_current_users_projects() -> None:
    service = PrincipalScopedContentStudioService()
    auth = StubAuthService()
    owner = auth.principal
    same_tenant_other_user = AuthenticatedPrincipal(uuid4(), owner.tenant_id, Role.ADMIN)
    client = TestClient(create_app(auth_service=auth, content_studio_service=service))
    headers = {"Authorization": "Bearer valid-token"}

    owned = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={"title": "Owned project", "topic": "Show my history"},
    )
    assert owned.status_code == 201

    auth.become(same_tenant_other_user)
    other = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={"title": "Other project", "topic": "Should stay hidden"},
    )
    assert other.status_code == 201

    auth.become(owner)
    listed = client.get("/api/v1/content-studio/projects", headers=headers)

    assert listed.status_code == 200
    assert [item["project_id"] for item in listed.json()["projects"]] == [owned.json()["project_id"]]
    assert listed.json()["projects"][0]["title"] == "Owned project"
    assert listed.json()["projects"][0]["status"] == "DRAFT"


def test_content_studio_api_rejects_stale_revision_for_approval() -> None:
    service = PrincipalScopedContentStudioService()
    auth = StubAuthService()
    client = TestClient(create_app(auth_service=auth, content_studio_service=service))
    headers = {"Authorization": "Bearer valid-token"}

    created = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={"title": "Revisioned project", "topic": "Show CAS"},
    )
    project_id = created.json()["project_id"]
    ran = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "SCRIPT_READY"},
    )
    assert ran.status_code == 200

    stale = client.post(
        f"/api/v1/content-studio/projects/{project_id}/approve-script",
        headers=headers,
        json={"revision": created.json()["revision"]},
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "content_project_conflict"


def test_content_studio_api_validates_trimmed_input_and_unknown_pack() -> None:
    service = PrincipalScopedContentStudioService()
    client = TestClient(create_app(auth_service=StubAuthService(), content_studio_service=service))
    headers = {"Authorization": "Bearer valid-token"}

    blank_title = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={"title": "  ", "topic": "usable topic"},
    )
    assert blank_title.status_code == 422

    unknown_pack = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={"title": "Pack test", "topic": "usable topic", "domain": "missing-pack"},
    )
    assert unknown_pack.status_code == 422
    assert unknown_pack.json()["error"]["code"] == "content_studio_pack_not_found"


def test_content_studio_api_regenerates_voice_with_instruction() -> None:
    service = PrincipalScopedContentStudioService()
    auth = StubAuthService()
    client = TestClient(create_app(auth_service=auth, content_studio_service=service))
    headers = {"Authorization": "Bearer valid-token"}

    created = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={"title": "Voice project", "topic": "Show voice regeneration"},
    )
    project_id = created.json()["project_id"]
    script = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "SCRIPT_READY"},
    )
    approved = client.post(
        f"/api/v1/content-studio/projects/{project_id}/approve-script",
        headers=headers,
        json={"revision": script.json()["revision"]},
    )
    voice_ready = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "VOICE_READY"},
    )
    assert approved.status_code == 200
    assert voice_ready.status_code == 200

    voice = client.post(
        f"/api/v1/content-studio/projects/{project_id}/regenerate-voice",
        headers=headers,
        json={"revision": voice_ready.json()["revision"], "instruction": "女声，语速稍慢"},
    )

    assert voice.status_code == 200
    assert voice.json()["status"] == "VOICE_READY"
    assert "regen" in voice.json()["voice_track"]["audio_artifact_id"]


def test_content_studio_api_rejects_unapproved_script_before_asset_stages() -> None:
    service = PrincipalScopedContentStudioService()
    auth = StubAuthService()
    client = TestClient(create_app(auth_service=auth, content_studio_service=service))
    headers = {"Authorization": "Bearer valid-token"}

    created = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={"title": "Gate project", "topic": "Show stage gates"},
    )
    project_id = created.json()["project_id"]
    script = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "SCRIPT_READY"},
    )
    assert script.status_code == 200

    assets = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "ASSETS_READY"},
    )
    assert assets.status_code == 409


def test_content_studio_api_rejects_preview_until_asset_rights_are_approved() -> None:
    service = PrincipalScopedContentStudioService()
    auth = StubAuthService()
    client = TestClient(create_app(auth_service=auth, content_studio_service=service))
    headers = {"Authorization": "Bearer valid-token"}

    created = client.post(
        "/api/v1/content-studio/projects",
        headers=headers,
        json={"title": "Rights project", "topic": "Show rights gate"},
    )
    project_id = created.json()["project_id"]
    script = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "SCRIPT_READY"},
    )
    approved = client.post(
        f"/api/v1/content-studio/projects/{project_id}/approve-script",
        headers=headers,
        json={"revision": script.json()["revision"]},
    )
    assets = client.post(
        f"/api/v1/content-studio/projects/{project_id}/run",
        headers=headers,
        json={"until": "ASSETS_READY"},
    )
    assert approved.status_code == 200
    assert assets.status_code == 200

    preview = client.post(
        f"/api/v1/content-studio/projects/{project_id}/render-preview",
        headers=headers,
    )
    assert preview.status_code == 409


def test_content_studio_api_returns_provider_blocked_when_generation_provider_is_missing() -> None:
    service = AsyncContentStudioService(
        registry=PackRegistry.mvp(),
        store=AsyncInMemoryContentProjectStore(),
        execution_mode="production",
    )
    client = TestClient(
        create_app(
            auth_service=StubAuthService(),
            content_studio_service=service,
        )
    )
    created = client.post(
        "/api/v1/content-studio/projects",
        headers={"Authorization": "Bearer valid-token"},
        json={"title": "Provider project", "topic": "Should not fake production"},
    )
    assert created.status_code == 201

    response = client.post(
        f"/api/v1/content-studio/projects/{created.json()['project_id']}/run",
        headers={"Authorization": "Bearer valid-token"},
        json={"until": "SCRIPT_READY"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "FAILED_BLOCKED"
    assert response.json()["research_bundle"] is None
    assert response.json()["script"] is None
    assert response.json()["asset_manifest"] is None


class PrincipalScopedContentStudioService:
    def __init__(self, media_dir: Path | None = None) -> None:
        self._registry = PackRegistry.mvp()
        self._projects: dict[str, ContentProject] = {}
        self._scopes: dict[str, tuple[str, str]] = {}
        self._revisions: dict[str, int] = {}
        self._media_dir = media_dir

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
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        service = ContentStudioService(
            registry=self._registry,
            store=InMemoryContentProjectStore(),
            execution_mode="demo",
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
            tenant_id=str(tenant_id),
            owner_user_id=str(owner_user_id),
        )
        project = replace(project, revision=1)
        self._projects[project.project_id] = project
        self._scopes[project.project_id] = (str(tenant_id), str(owner_user_id))
        self._revisions[project.project_id] = 1
        return project

    async def get_content_project(
        self,
        project_id: str,
        *,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        self._assert_scope(project_id, tenant_id, owner_user_id)
        return self._projects[project_id]

    async def list_content_projects(
        self,
        *,
        tenant_id: object,
        owner_user_id: object,
        limit: int = 50,
    ) -> tuple[ContentProjectSummary, ...]:
        summaries = [
            content_project_summary(project)
            for project_id, project in self._projects.items()
            if self._scopes.get(project_id) == (str(tenant_id), str(owner_user_id))
        ]
        summaries.sort(key=lambda summary: summary.revision, reverse=True)
        return tuple(summaries[:limit])

    async def run_content_project(
        self,
        project_id: str,
        *,
        until: ProjectStatus = ProjectStatus.QC_REVIEW,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        if until in {
            ProjectStatus.STORYBOARD_READY,
            ProjectStatus.ASSETS_READY,
            ProjectStatus.VOICE_READY,
            ProjectStatus.TIMELINE_READY,
            ProjectStatus.PREVIEW_RENDERED,
            ProjectStatus.QC_REVIEW,
        }:
            current = await self.get_content_project(
                project_id,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
            )
            if current.status is not ProjectStatus.SCRIPT_APPROVED and _stage_before(
                current.status, ProjectStatus.STORYBOARD_READY
            ):
                raise ValueError("script must be approved before asset stages")
        return self._mutate(
            project_id,
            tenant_id,
            owner_user_id,
            lambda service: service.run_content_project(project_id, until=until),
        )

    async def revise_script(
        self,
        project_id: str,
        *,
        instruction: str,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        return self._mutate(
            project_id,
            tenant_id,
            owner_user_id,
            lambda service: service.revise_script(project_id, instruction=instruction),
        )

    async def approve_script(
        self,
        project_id: str,
        *,
        revision: int,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        self._assert_revision(project_id, revision)
        return self._mutate(
            project_id,
            tenant_id,
            owner_user_id,
            lambda service: service.approve_script(project_id),
        )

    async def revise_storyboard(
        self,
        project_id: str,
        *,
        instruction: str,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        return self._mutate(
            project_id,
            tenant_id,
            owner_user_id,
            lambda service: service.revise_storyboard(project_id, instruction=instruction),
        )

    async def regenerate_asset(
        self,
        project_id: str,
        *,
        asset_id: str,
        instruction: str,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        return self._mutate(
            project_id,
            tenant_id,
            owner_user_id,
            lambda service: service.regenerate_asset(
                project_id, asset_id=asset_id, instruction=instruction
            ),
        )

    async def approve_rights(
        self,
        project_id: str,
        *,
        asset_ids: tuple[str, ...],
        note: str,
        revision: int,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        self._assert_scope(project_id, tenant_id, owner_user_id)
        self._assert_revision(project_id, revision)
        project = self._projects[project_id]
        if project.asset_manifest is None:
            raise ValueError("cannot approve rights before assets exist")
        requested = set(asset_ids)
        assets = tuple(
            replace(
                asset,
                rights_status="approved",
                generation_params={**asset.generation_params, "rights_note": note},
            )
            if asset.asset_id in requested
            else asset
            for asset in project.asset_manifest.assets
        )
        if {asset.asset_id for asset in assets if asset.asset_id in requested} != requested:
            raise ValueError("asset not found")
        return self._save(
            replace(
                project,
                asset_manifest=AssetManifest(assets=assets),
                rights_approved=True,
            )
        )

    async def regenerate_voice(
        self,
        project_id: str,
        *,
        instruction: str,
        revision: int,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        self._assert_scope(project_id, tenant_id, owner_user_id)
        self._assert_revision(project_id, revision)
        project = self._projects[project_id]
        voice = VoiceTrack(
            audio_artifact_id=f"voice-{project.project_id}-regen",
            timestamp_level="sentence",
            pronunciation_report=(instruction,),
        )
        return self._save(replace(project, status=ProjectStatus.VOICE_READY, voice_track=voice))

    async def render_preview(
        self,
        project_id: str,
        *,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        project = await self.get_content_project(
            project_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        if project.asset_manifest is not None and any(
            asset.rights_status not in {"approved", "cleared", "owned", "licensed"}
            for asset in project.asset_manifest.assets
        ):
            raise ValueError("asset rights must be approved before preview")
        rendered = self._mutate(
            project_id,
            tenant_id,
            owner_user_id,
            lambda service: service.render_preview(project_id),
        )
        if self._media_dir is None or rendered.timeline is None:
            return rendered
        media_path = self._media_dir / f"{project_id}-preview.mp4"
        media_path.write_bytes(b"fake preview mp4")
        return self._save(
            replace(
                rendered,
                timeline=replace(rendered.timeline, preview_artifact_id=str(media_path)),
            )
        )

    async def approve_final(
        self,
        project_id: str,
        *,
        revision: int,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        project = await self.get_content_project(
            project_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        if project.status is not ProjectStatus.QC_REVIEW:
            raise ValueError("clean QC review is required before final approval")
        if project.qc_report is not None and project.qc_report.blockers:
            raise ValueError("clean QC review is required before final approval")
        self._assert_revision(project_id, revision)
        approved = self._mutate(
            project_id,
            tenant_id,
            owner_user_id,
            lambda service: service.approve_final(project_id),
        )
        if self._media_dir is None or approved.timeline is None:
            return approved
        media_path = self._media_dir / f"{project_id}-final.mp4"
        media_path.write_bytes(b"fake final mp4")
        return self._save(
            replace(
                approved,
                timeline=replace(approved.timeline, final_artifact_id=str(media_path)),
            )
        )

    async def retry_stage(
        self,
        project_id: str,
        stage: ProjectStatus,
        *,
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        return self._mutate(
            project_id,
            tenant_id,
            owner_user_id,
            lambda service: service.retry_stage(project_id, stage),
        )

    async def replace_claim_status(
        self,
        project_id: str,
        *,
        claim_id: str,
        status: ClaimStatus,
        note: str,
        evidence_ids: tuple[str, ...] = (),
        tenant_id: object,
        owner_user_id: object,
    ) -> ContentProject:
        return self._mutate(
            project_id,
            tenant_id,
            owner_user_id,
            lambda service: service.replace_claim_status(
                project_id,
                claim_id=claim_id,
                status=status,
                note=note,
                evidence_ids=evidence_ids,
            ),
        )

    def revision_for(self, project_id: str) -> int:
        return self._revisions[project_id]

    def _mutate(
        self,
        project_id: str,
        tenant_id: object,
        owner_user_id: object,
        operation: object,
    ) -> ContentProject:
        self._assert_scope(project_id, tenant_id, owner_user_id)
        store = InMemoryContentProjectStore()
        store.save(self._projects[project_id])
        service = ContentStudioService(
            registry=self._registry,
            store=store,
            execution_mode="demo",
        )
        return self._save(operation(service))  # type: ignore[operator]

    def _save(self, project: ContentProject) -> ContentProject:
        revision = self._revisions[project.project_id] + 1
        project = replace(project, revision=revision)
        self._projects[project.project_id] = project
        self._revisions[project.project_id] = revision
        return project

    def _assert_scope(self, project_id: str, tenant_id: object, owner_user_id: object) -> None:
        if self._scopes.get(project_id) != (str(tenant_id), str(owner_user_id)):
            raise KeyError(project_id)

    def _assert_revision(self, project_id: str, revision: int) -> None:
        if self._revisions.get(project_id) != revision:
            raise ValueError("content project revision changed")


def _stage_before(current: ProjectStatus, target: ProjectStatus) -> bool:
    order = (
        ProjectStatus.DRAFT,
        ProjectStatus.RESEARCH_READY,
        ProjectStatus.FACT_CHECKED,
        ProjectStatus.PLAN_READY,
        ProjectStatus.SCRIPT_READY,
        ProjectStatus.SCRIPT_APPROVED,
        ProjectStatus.STORYBOARD_READY,
        ProjectStatus.ASSETS_READY,
        ProjectStatus.VOICE_READY,
        ProjectStatus.TIMELINE_READY,
        ProjectStatus.PREVIEW_RENDERED,
        ProjectStatus.QC_REVIEW,
        ProjectStatus.FINAL_APPROVED,
    )
    return order.index(current) < order.index(target)
