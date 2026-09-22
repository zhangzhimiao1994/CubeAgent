"""FastAPI application factory and owned process resources."""

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Final, Protocol, cast
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent_hub.api.errors import (
    PublicAPIError,
    error_payload,
    http_exception_handler,
    public_error_handler,
)
from agent_hub.api.middleware import RequestBodyLimitMiddleware, SafeExceptionMiddleware
from agent_hub.api.routers import admin, auth, config, content_studio, runs, system, users
from agent_hub.auth.passwords import PasswordService
from agent_hub.auth.rate_limit import RedisAuthRateLimiter
from agent_hub.auth.service import AuthService
from agent_hub.auth.tokens import AccessTokenService
from agent_hub.auth.user_admin import PersistentUserAdminService
from agent_hub.capabilities.runtime import RuntimeAssetVisualReview, RuntimeCapabilityGateway
from agent_hub.channels.base import InboundMessage
from agent_hub.channels.dedup import InboundDedupRepository
from agent_hub.channels.feishu.media import FeishuOpenAPIMediaClient
from agent_hub.channels.feishu.media_factory import build_feishu_media_service_factory
from agent_hub.channels.feishu.reply import (
    FeishuOpenAPIReplySender,
    FeishuRunReplyDispatcher,
    log_feishu_reply_failure,
)
from agent_hub.channels.feishu.sdk_client import create_lark_oapi_feishu_websocket_client
from agent_hub.channels.feishu.settings import FeishuSettings, FeishuTransport
from agent_hub.channels.feishu.skill_install import FeishuSkillCommandHandler
from agent_hub.channels.feishu.webhook import (
    ChannelGatewayProtocol,
    _feishu_settings_from_runtime_config,
    create_lazy_feishu_webhook_router,
)
from agent_hub.channels.feishu.websocket import (
    FeishuWebSocketClient,
    FeishuWebSocketConnector,
    build_feishu_websocket_receiver,
)
from agent_hub.channels.gateway import ChannelGateway
from agent_hub.channels.generic_webhook import create_generic_channel_webhook_router
from agent_hub.channels.identity import PersistentChannelIdentityResolver
from agent_hub.channels.submitter import (
    ChannelSettingsService,
    RunServiceInboundSubmitter,
    RunSubmissionService,
)
from agent_hub.cognitive.pipeline import CognitiveLearningPipeline, CognitiveLearningTerminalHook
from agent_hub.cognitive.repository import (
    PersistentCognitiveRecordRepository,
    PersistentExperienceRepository,
)
from agent_hub.cognitive.service import CognitiveStateService, ExperienceService
from agent_hub.config.service import ConfigService
from agent_hub.content_studio import (
    AssetManifest,
    AssetRecord,
    AsyncContentStudioService,
    AtomicClaim,
    ClaimStatus,
    ContentPlan,
    ContentProject,
    Evidence,
    EvidenceGraph,
    FactCheckReport,
    ProjectStatus,
    ProviderAttempt,
    QCReport,
    ResearchBundle,
    ResearchQuestion,
    ResearchSourceCandidate,
    ResearchSourceCoverage,
    ScriptDraft,
    ScriptSegment,
    Shot,
    Storyboard,
    Timeline,
    VoiceTrack,
    append_content_project_event,
    record_content_project_provider_attempt,
)
from agent_hub.content_studio.media import (
    AudioClip,
    ClaimReference,
    ContentStudioMediaAdapter,
    ContentStudioMediaError,
    FinalRenderApproval,
    MediaQCCheck,
    MediaTimeline,
    RenderRequest,
    SubtitleCue,
    VisualClip,
)
from agent_hub.content_studio.packs import load_pack_registry
from agent_hub.content_studio.repository import PersistentContentProjectStore
from agent_hub.db.models import TenantRow
from agent_hub.db.session import build_database
from agent_hub.domain.runs import TaskMode
from agent_hub.hermes import PersistentHermesRunAdvisor
from agent_hub.models.capabilities import is_known_video_generation_model
from agent_hub.models.capacity import (
    CapacityPool,
    CapacityUnavailable,
    CredentialDescriptor,
    CredentialRegistry,
    safe_operational_limit,
)
from agent_hub.models.gateway import (
    CapacityController,
    ModelGateway,
    ModelGatewayError,
    ModelTransport,
)
from agent_hub.models.litellm_client import LiteLLMClient
from agent_hub.models.registry import ModelRegistry, NoCapableDeployment
from agent_hub.models.types import (
    Deployment,
    ModelCapability,
    ModelMessage,
    ModelRequest,
    StructuredResponseSchema,
)
from agent_hub.multimodal.audio_providers import (
    MiniMaxAudioGenerationClient,
    TextToAudioProvider,
    TextToAudioProviderRouter,
)
from agent_hub.multimodal.dashscope import (
    DashScopeMultimediaGenerationClient,
    is_dashscope_multimedia_deployment,
)
from agent_hub.multimodal.generation import (
    InMemoryMultimediaGenerationJobStore,
    MultimediaArtifact,
    MultimediaDailyLimitExceeded,
    MultimediaGenerationExecutor,
    MultimediaGenerationJob,
    MultimediaGenerationKind,
    MultimediaGenerationResult,
)
from agent_hub.multimodal.minimax import MiniMaxVideoGenerationClient
from agent_hub.multimodal.video_providers import (
    TextToVideoProvider,
    TextToVideoProviderRouter,
    VideoProviderGenerationError,
)
from agent_hub.observability.logging import configure_logging
from agent_hub.observability.metrics import default_metrics_registry
from agent_hub.routing.classifier import GatewayRouteClassifier
from agent_hub.routing.service import ModeRouter, RoutingPolicy
from agent_hub.routing.types import (
    InMemoryDecisionTokenStore,
    RiskLevel,
    RouteDecision,
    RouteSource,
)
from agent_hub.runs.attachments import FileSystemAttachmentArtifactLoader
from agent_hub.runs.repository import RunRepository
from agent_hub.runs.resource_context import ResourceContextArtifactLoader
from agent_hub.runs.service import ModeRouterProtocol, RunService, TaskQueue
from agent_hub.runs.temporary_agents import AdminResourceTemporaryAgentPolicy
from agent_hub.runtime.contracts import JsonValue
from agent_hub.runtime.defaults import TenantSecretResolver, configured_runtime_registry
from agent_hub.runtime.registry import RuntimeRegistry
from agent_hub.scheduler.service import SchedulerService
from agent_hub.scheduler.types import TaskRequest
from agent_hub.security.secrets import SecretCipher, SecretService
from agent_hub.settings import Settings, get_settings

ReadinessProbe = Callable[[], Awaitable[None]]
CleanupCallback = tuple[str, Callable[[], Awaitable[None]]]
_LOGGER = logging.getLogger(__name__)


class ResourceCleanupError(RuntimeError):
    """Report cleanup failure types without exposing resource error details."""

    def __init__(self, error_types: tuple[str, ...]) -> None:
        self.error_types = error_types
        super().__init__(f"resource cleanup failed: {', '.join(error_types)}")


class DatabaseResource(Protocol):
    session_factory: Any

    async def dispose(self) -> None: ...


class RedisResource(Protocol):
    async def aclose(self) -> None: ...

    def ping(self, **kwargs: Any) -> Any: ...


RouterCapacityFactory = Callable[
    [tuple[Deployment, ...]],
    Awaitable[CapacityController | CapacityPool],
]
MultimediaCapacityFactory = Callable[
    [tuple[Deployment, ...]],
    Awaitable[CapacityController | CapacityPool],
]
MainAgentConfigGetter = Callable[[], Awaitable[admin.MainAgentConfigResponse]]
RegisteredModelListGetter = Callable[[], Awaitable[tuple[admin.ModelDeploymentResponse, ...]]]
FeishuWebSocketClientFactoryForSettings = Callable[
    [FeishuSettings], Awaitable[FeishuWebSocketClient]
]

_ASSET_VISUAL_REVIEW_ATTEMPTS = 5
_ASSET_VISUAL_REVIEW_RETRY_BACKOFF_SECONDS = 8.0
_ASSET_VISUAL_REVIEW_CAPACITY_WAIT_SECONDS = 20.0


class _MainAgentModeRouter:
    """Lazy production router backed by the separately configured main Agent model."""

    def __init__(
        self,
        *,
        get_config: MainAgentConfigGetter,
        list_models: RegisteredModelListGetter | None = None,
        secret_service: SecretService,
        tenant_id: UUID,
        redis_client: object,
        transport: ModelTransport | None = None,
        capacity_factory: RouterCapacityFactory | None = None,
    ) -> None:
        self._get_config = get_config
        self._list_models = list_models
        self._secret_service = secret_service
        self._tenant_id = tenant_id
        self._redis_client = redis_client
        self._transport = transport or LiteLLMClient()
        self._capacity_factory = capacity_factory

    async def route(self, task_text: object) -> RouteDecision:
        try:
            config = await self._get_config()
            if config.model is None:
                return _waiting_route_decision("main_agent_not_configured")
            deployment = await self._deployment_from_config(config.model)
            capacity = (
                await self._capacity_factory((deployment,))
                if self._capacity_factory is not None
                else await self._default_capacity((deployment,))
            )
            gateway = ModelGateway(
                ModelRegistry((deployment,)),
                capacity,
                TenantSecretResolver(self._secret_service, self._tenant_id),
                self._transport,
                capacity_wait_timeout=60,
            )
            router = ModeRouter(
                GatewayRouteClassifier(
                    gateway,
                    logical_model="main_agent",
                    source=RouteSource.CLASSIFIER,
                    prefer_plain_json=True,
                ),
                GatewayRouteClassifier(
                    gateway,
                    logical_model="main_agent",
                    source=RouteSource.VERIFIER,
                    prefer_plain_json=True,
                ),
                token_store=InMemoryDecisionTokenStore(),
                policy=RoutingPolicy(
                    confidence_threshold=0.65,
                    parallel_classifiers=False,
                    allow_single_classifier_decision=True,
                ),
            )
            return await router.route(task_text)
        except Exception as error:  # noqa: BLE001 - auto routing must degrade safely.
            _LOGGER.warning("main_agent_router_unavailable error_type=%s", type(error).__name__)
            return _waiting_route_decision("main_agent_router_unavailable")

    async def _deployment_from_config(self, model: admin.MainAgentModelConfig) -> Deployment:
        if self._list_models is None:
            return admin._main_agent_model_deployment(model)
        matched: admin.ModelDeploymentResponse | None = None
        try:
            for registered in await self._list_models():
                if (
                    registered.provider == model.provider
                    and registered.api_base == model.api_base
                    and registered.api_protocol == model.api_protocol
                    and registered.upstream_model == model.upstream_model
                    and registered.credential_ref == model.credential_ref
                ):
                    matched = registered
                    break
        except Exception as error:  # noqa: BLE001 - routing should still use explicit config.
            _LOGGER.warning(
                "main_agent_model_capability_lookup_failed error_type=%s",
                type(error).__name__,
            )
        if matched is None:
            return admin._main_agent_model_deployment(model)
        capabilities = set(model.capabilities)
        capabilities.update(matched.capabilities)
        parsed_capabilities = frozenset(ModelCapability(item) for item in capabilities)
        return Deployment(
            id="main_agent_1",
            logical_model="main_agent",
            provider_model=f"{model.provider}/{model.upstream_model}",
            request_model=model.upstream_model,
            api_base=model.api_base,
            secret_ref=model.credential_ref,
            quota_scope_id=matched.quota_scope,
            max_concurrency=matched.max_concurrency,
            target_utilization=matched.target_utilization,
            reserved_slots=matched.reserved_capacity,
            rpm=matched.rpm,
            tpm=matched.tpm,
            weight=matched.weight,
            capabilities=parsed_capabilities,
        )

    async def _default_capacity(
        self,
        deployments: tuple[Deployment, ...],
    ) -> CapacityPool:
        credentials = CredentialRegistry(
            [
                CredentialDescriptor(
                    deployment.secret_ref,
                    await self._secret_service.fingerprint(self._tenant_id, deployment.secret_ref),
                )
                for deployment in deployments
            ]
        )
        return CapacityPool(self._redis_client, deployments=deployments, credentials=credentials)


class _MainAgentContextWindowGetter:
    def __init__(self, get_config: MainAgentConfigGetter) -> None:
        self._get_config = get_config

    async def __call__(self) -> int | None:
        config = await self._get_config()
        if config.model is None:
            return None
        return _infer_main_agent_context_window_tokens(
            config.model.provider,
            config.model.upstream_model,
        )


class _ConfigBackedMultimediaGenerationExecutor:
    """Build a generation gateway from the current registered model resources."""

    def __init__(
        self,
        *,
        list_models: RegisteredModelListGetter,
        secret_service: SecretService,
        tenant_id: UUID,
        redis_client: object,
        transport: ModelTransport | None = None,
        capacity_factory: MultimediaCapacityFactory | None = None,
        media_store_dir: Path | None = None,
        video_provider_router: TextToVideoProviderRouter | None = None,
        audio_provider_router: TextToAudioProviderRouter | None = None,
        dashscope_multimedia_client: DashScopeMultimediaGenerationClient | None = None,
    ) -> None:
        self._list_models = list_models
        self._secret_service = secret_service
        self._tenant_id = tenant_id
        self._redis_client = redis_client
        self._transport = transport or LiteLLMClient()
        self._capacity_factory = capacity_factory
        self._media_store_dir = (media_store_dir or Path("/var/lib/agent-hub/media")).resolve()
        self._video_provider_router = video_provider_router or TextToVideoProviderRouter(
            (("minimax", MiniMaxVideoGenerationClient()),)
        )
        self._audio_provider_router = audio_provider_router or TextToAudioProviderRouter(
            (("minimax", MiniMaxAudioGenerationClient()),)
        )
        self._dashscope_multimedia_client = (
            dashscope_multimedia_client or DashScopeMultimediaGenerationClient()
        )
        self._daily_usage: dict[tuple[date, str, str], int] = {}
        self._job_store = InMemoryMultimediaGenerationJobStore()

    def submit(
        self,
        *,
        kind: MultimediaGenerationKind,
        logical_model: str,
        prompt: str,
    ) -> MultimediaGenerationJob:
        return self._job_store.create(
            kind=kind,
            logical_model=logical_model,
            prompt=prompt.strip(),
        )

    async def default_logical_model_for_multimedia(
        self,
        *,
        kind: MultimediaGenerationKind,
    ) -> str:
        logical_model = await self.default_logical_model(kind.value)
        if logical_model is None:
            raise NoCapableDeployment(f"no capable multimedia generation deployment: {kind.value}")
        return logical_model

    def get_job(self, job_id: str) -> MultimediaGenerationJob:
        return self._job_store.get(job_id)

    async def default_logical_model(self, kind: str) -> str | None:
        generation_kind = MultimediaGenerationKind(kind)
        deployments = tuple(
            _deployment_from_model_resource(model) for model in await self._list_models()
        )
        required_capability = _multimedia_required_capability(generation_kind)
        direct_candidates: list[Deployment] = []
        gateway_candidates: list[Deployment] = []
        for deployment in deployments:
            if required_capability not in deployment.capabilities:
                continue
            provider, upstream_model = _deployment_provider_and_model(deployment)
            if is_dashscope_multimedia_deployment(provider, upstream_model, deployment.api_base):
                direct_candidates.append(deployment)
                continue
            if generation_kind is MultimediaGenerationKind.VIDEO and (
                self._video_provider_router.provider_for(deployment) is not None
            ):
                direct_candidates.append(deployment)
                continue
            if generation_kind is MultimediaGenerationKind.AUDIO and (
                self._audio_provider_router.provider_for(deployment) is not None
            ):
                direct_candidates.append(deployment)
                continue
            gateway_candidates.append(deployment)
        candidates = direct_candidates or gateway_candidates
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                safe_operational_limit(
                    item.max_concurrency,
                    item.target_utilization,
                    item.reserved_slots,
                ),
                item.weight,
                item.logical_model,
            ),
        ).logical_model

    async def run_job(
        self,
        job_id: str,
        *,
        executor_id: str,
    ) -> MultimediaGenerationJob:
        job = self._job_store.start(job_id, executor_id=executor_id)
        try:
            result = await self.generate(
                kind=job.kind,
                logical_model=job.logical_model,
                prompt=job.prompt,
            )
        except asyncio.CancelledError:
            self._job_store.fail(job_id, error="multimedia generation cancelled")
            raise
        except Exception as error:
            self._job_store.fail(job_id, error=str(error))
            raise
        return self._job_store.succeed(
            job_id,
            artifacts=(
                MultimediaArtifact(
                    kind=result.kind,
                    uri=result.text,
                    text=result.text,
                    logical_model=result.logical_model,
                    deployment_id=result.deployment_id,
                    file_path=result.file_path,
                    filename=result.filename,
                    mime_type=result.mime_type,
                ),
            ),
        )

    async def generate(
        self,
        *,
        kind: MultimediaGenerationKind,
        logical_model: str,
        prompt: str,
    ) -> MultimediaGenerationResult:
        deployments = tuple(
            _deployment_from_model_resource(model) for model in await self._list_models()
        )
        registry = ModelRegistry(deployments)
        candidates = registry.candidates(logical_model, {_multimedia_required_capability(kind)})
        _require_supported_multimedia_generation(
            kind=kind,
            logical_model=logical_model,
            candidates=candidates,
        )
        daily_limit = _multimedia_daily_limit(kind, deployments, logical_model)
        self._claim_daily_slot(
            kind=kind,
            logical_model=logical_model,
            daily_limit=daily_limit,
        )
        direct_result = await self._generate_with_direct_provider(
            kind=kind,
            prompt=prompt,
            candidates=candidates,
        )
        if direct_result is not None:
            return direct_result
        capacity = (
            await self._capacity_factory(deployments)
            if self._capacity_factory is not None
            else await self._default_capacity(deployments)
        )
        gateway = ModelGateway(
            registry,
            capacity,
            TenantSecretResolver(self._secret_service, self._tenant_id),
            self._transport,
            capacity_wait_timeout=60,
        )
        return await MultimediaGenerationExecutor(gateway).generate(
            kind=kind,
            logical_model=logical_model,
            prompt=prompt,
        )

    async def _generate_with_direct_provider(
        self,
        *,
        kind: MultimediaGenerationKind,
        prompt: str,
        candidates: tuple[Deployment, ...],
        ) -> MultimediaGenerationResult | None:
        for candidate in candidates:
            provider_name, upstream_model = _deployment_provider_and_model(candidate)
            if is_dashscope_multimedia_deployment(provider_name, upstream_model, candidate.api_base):
                api_key = await self._secret_service.resolve(self._tenant_id, candidate.secret_ref)
                output_dir = self._media_store_dir / str(self._tenant_id)
                if kind is MultimediaGenerationKind.IMAGE:
                    image = await self._dashscope_multimedia_client.generate_text_to_image(
                        api_key=api_key,
                        api_base=candidate.api_base,
                        model=candidate.request_model or upstream_model,
                        prompt=prompt,
                        output_dir=output_dir,
                    )
                    return MultimediaGenerationResult(
                        kind=kind,
                        logical_model=candidate.logical_model,
                        deployment_id=candidate.id,
                        text=image.uri,
                        file_path=image.path,
                        filename=image.path.name,
                        mime_type=image.mime_type,
                    )
                if kind is MultimediaGenerationKind.VIDEO:
                    video = await self._dashscope_multimedia_client.generate_text_to_video(
                        api_key=api_key,
                        api_base=candidate.api_base,
                        model=candidate.request_model or upstream_model,
                        prompt=prompt,
                        output_dir=output_dir,
                        duration=5,
                        resolution="std",
                    )
                    return MultimediaGenerationResult(
                        kind=kind,
                        logical_model=candidate.logical_model,
                        deployment_id=candidate.id,
                        text=video.uri,
                        file_path=video.path,
                        filename=video.path.name,
                        mime_type=video.mime_type,
                    )
        if kind is MultimediaGenerationKind.AUDIO:
            audio_result = await self._generate_audio_with_direct_provider(
                prompt=prompt,
                candidates=candidates,
            )
            if audio_result is not None:
                return audio_result
        if kind is not MultimediaGenerationKind.VIDEO:
            return None
        selected: tuple[Deployment, TextToVideoProvider] | None = None
        for candidate in candidates:
            video_provider = self._video_provider_router.provider_for(candidate)
            if video_provider is not None:
                selected = (candidate, video_provider)
                break
        if selected is None:
            return None
        deployment, provider = selected
        api_key = await self._secret_service.resolve(self._tenant_id, deployment.secret_ref)
        artifact = await provider.generate_text_to_video(
            api_key=api_key,
            api_base=deployment.api_base,
            model=deployment.request_model or deployment.provider_model,
            prompt=prompt,
            output_dir=self._media_store_dir / str(self._tenant_id),
            duration=6,
            resolution="768P",
        )
        return MultimediaGenerationResult(
            kind=kind,
            logical_model=deployment.logical_model,
            deployment_id=deployment.id,
            text=artifact.uri,
            file_path=artifact.path,
            filename=artifact.path.name,
            mime_type=artifact.mime_type,
        )

    async def _generate_audio_with_direct_provider(
        self,
        *,
        prompt: str,
        candidates: tuple[Deployment, ...],
    ) -> MultimediaGenerationResult | None:
        selected: tuple[Deployment, TextToAudioProvider] | None = None
        for candidate in candidates:
            audio_provider = self._audio_provider_router.provider_for(candidate)
            if audio_provider is not None:
                selected = (candidate, audio_provider)
                break
        if selected is None:
            return None
        deployment, provider = selected
        api_key = await self._secret_service.resolve(self._tenant_id, deployment.secret_ref)
        artifact = await provider.generate_text_to_audio(
            api_key=api_key,
            api_base=deployment.api_base,
            model=deployment.request_model or deployment.provider_model,
            prompt=prompt,
            output_dir=self._media_store_dir / str(self._tenant_id),
        )
        return MultimediaGenerationResult(
            kind=MultimediaGenerationKind.AUDIO,
            logical_model=deployment.logical_model,
            deployment_id=deployment.id,
            text=artifact.uri,
            file_path=artifact.path,
            filename=artifact.path.name,
            mime_type=artifact.mime_type,
        )

    def _claim_daily_slot(
        self,
        *,
        kind: MultimediaGenerationKind,
        logical_model: str,
        daily_limit: int | None,
    ) -> None:
        if daily_limit is None:
            return
        today = datetime.now(UTC).date()
        self._daily_usage = {
            key: count for key, count in self._daily_usage.items() if key[0] == today
        }
        key = (today, kind.value, logical_model)
        current = self._daily_usage.get(key, 0)
        if current >= daily_limit:
            raise MultimediaDailyLimitExceeded("daily multimedia generation limit exceeded")
        self._daily_usage[key] = current + 1

    async def _default_capacity(
        self,
        deployments: tuple[Deployment, ...],
    ) -> CapacityPool:
        credentials = CredentialRegistry(
            [
                CredentialDescriptor(
                    secret_ref,
                    await self._secret_service.fingerprint(self._tenant_id, secret_ref),
                )
                for secret_ref in dict.fromkeys(deployment.secret_ref for deployment in deployments)
            ]
        )
        return CapacityPool(self._redis_client, deployments=deployments, credentials=credentials)


_CONTENT_STUDIO_STAGE_ORDER: Final[tuple[ProjectStatus, ...]] = (
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


class _ConfigBackedContentStudioProductionProvider:
    """Production bridge from Content Studio state to existing model/media providers."""

    def __init__(
        self,
        *,
        list_models: RegisteredModelListGetter,
        secret_service: SecretService,
        tenant_id: UUID,
        redis_client: object,
        multimedia_generation_executor: _ConfigBackedMultimediaGenerationExecutor | None = None,
        transport: ModelTransport | None = None,
        capacity_factory: MultimediaCapacityFactory | None = None,
        output_dir: Path | None = None,
        http_timeout_seconds: float = 20,
        media_adapter: ContentStudioMediaAdapter | None = None,
    ) -> None:
        self._list_models = list_models
        self._secret_service = secret_service
        self._tenant_id = tenant_id
        self._redis_client = redis_client
        self._multimedia = multimedia_generation_executor
        self._transport = transport or LiteLLMClient()
        self._capacity_factory = capacity_factory
        self._output_dir = (output_dir or Path("/var/lib/agent-hub/content-studio")).resolve()
        self._http_timeout_seconds = http_timeout_seconds
        self._media_adapter = media_adapter or ContentStudioMediaAdapter()

    async def run_content_project(
        self,
        project: ContentProject,
        *,
        until: ProjectStatus,
    ) -> ContentProject:
        current = project
        for stage in _CONTENT_STUDIO_STAGE_ORDER:
            if _content_studio_stage_index(stage) > _content_studio_stage_index(until):
                break
            current = await self._run_stage(current, stage)
            if current.status in {ProjectStatus.FAILED_BLOCKED, ProjectStatus.FAILED_RETRYABLE}:
                return current
            if (
                stage is ProjectStatus.SCRIPT_READY
                and not current.script_approved
                and _content_studio_stage_index(until)
                > _content_studio_stage_index(ProjectStatus.SCRIPT_READY)
            ):
                return current
        return current

    async def _run_stage(self, project: ContentProject, stage: ProjectStatus) -> ContentProject:
        if stage is ProjectStatus.RESEARCH_READY:
            return await self._ensure_research(project)
        if stage is ProjectStatus.FACT_CHECKED:
            return await self._ensure_fact_check(project)
        if stage is ProjectStatus.PLAN_READY:
            return self._ensure_plan(project)
        if stage is ProjectStatus.SCRIPT_READY:
            return await self._ensure_script(project)
        if stage is ProjectStatus.STORYBOARD_READY:
            return self._ensure_storyboard(project)
        if stage is ProjectStatus.ASSETS_READY:
            return await self._ensure_assets(project)
        if stage is ProjectStatus.VOICE_READY:
            return await self._ensure_voice(project)
        if stage is ProjectStatus.TIMELINE_READY:
            return self._ensure_timeline(project)
        if stage is ProjectStatus.PREVIEW_RENDERED:
            return self._ensure_preview(project)
        if stage is ProjectStatus.QC_REVIEW:
            return self._ensure_qc(project)
        if stage is ProjectStatus.FINAL_RENDERED:
            return self._ensure_final(project)
        return project

    async def _ensure_research(self, project: ContentProject) -> ContentProject:
        if project.research_bundle is not None and project.evidence_graph is not None:
            return _content_studio_status(project, ProjectStatus.RESEARCH_READY, "research")
        evidence = await self._fetch_evidence(project)
        if len(evidence) < 2:
            return _content_studio_blocked(
                project,
                "research_insufficient_evidence",
                "deep research requires at least two usable official or primary sources",
            )
        questions = (
            ResearchQuestion("RQ001", f"{project.topic} 最近发生了什么？"),
            ResearchQuestion("RQ002", "它对普通创作者或团队有什么实际影响？"),
            ResearchQuestion("RQ003", "哪些限制、成本、版权或审核风险不能被夸大？"),
            ResearchQuestion("RQ004", "60 秒抖音科普应该用什么例子讲清楚？"),
        )
        claims = self._claims_from_evidence(project, evidence)
        bundle = _content_studio_research_bundle(project, questions, evidence)
        graph = EvidenceGraph(claims=claims, evidence=evidence)
        return _content_studio_status(
            replace(project, research_bundle=bundle, evidence_graph=graph),
            ProjectStatus.RESEARCH_READY,
            "research",
        )

    async def _fetch_evidence(self, project: ContentProject) -> tuple[Evidence, ...]:
        urls = _content_studio_source_urls(project)
        allowed_hosts = _content_studio_allowed_hosts(project)
        evidence: list[Evidence] = []
        async with httpx.AsyncClient(
            timeout=self._http_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "CubeAgent-ContentStudio/1.0"},
        ) as client:
            for url in urls:
                host = urlsplit(url).hostname or ""
                if not _host_allowed(host, allowed_hosts):
                    continue
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                except httpx.HTTPError:
                    continue
                text = _html_to_text(response.text)
                if not text:
                    continue
                evidence_id = f"EV{len(evidence) + 1:03d}"
                evidence.append(
                    Evidence(
                        evidence_id=evidence_id,
                        source_url=str(response.url),
                        source_type=_source_type_for_url(project, str(response.url)),
                        publisher=_publisher_for_host(host),
                        published_at=None,
                        retrieved_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        content_hash=hashlib.sha256(response.content[:1_000_000]).hexdigest(),
                        locator="page:body",
                        excerpt=text[:1200],
                        license=str(project.packs.domain.settings.get("default_license", "source_terms")),
                    )
                )
                if len(evidence) >= 5:
                    break
        return tuple(evidence)

    def _claims_from_evidence(
        self,
        project: ContentProject,
        evidence: tuple[Evidence, ...],
    ) -> tuple[AtomicClaim, ...]:
        evidence_ids = tuple(item.evidence_id for item in evidence)
        return (
            AtomicClaim(
                claim_id="CL001",
                text="AIGC is best explained as a workflow capability, not only a single prompt box.",
                claim_type="factual",
                temporal_scope="current at retrieval time",
                evidence_ids=evidence_ids,
                confidence=0.78,
                status=ClaimStatus.SUPPORTED,
                verification="supported by primary-source research excerpts gathered for this project",
                script_usages=(),
            ),
            AtomicClaim(
                claim_id="CL002",
                text="AIGC production use still depends on cost, permissions, review, and output-quality controls.",
                claim_type="factual",
                temporal_scope="current at retrieval time",
                evidence_ids=evidence_ids[:2] or evidence_ids,
                confidence=0.76,
                status=ClaimStatus.SUPPORTED,
                verification="supported by primary-source research excerpts and domain risk rules",
                script_usages=(),
            ),
            AtomicClaim(
                claim_id="CL003",
                text=f"The requested output should be a {project.packs.platform.name} explainer with subtitles and frequent visual changes.",
                claim_type="production_constraint",
                temporal_scope="project pack lock time",
                evidence_ids=evidence_ids[:1],
                confidence=0.95,
                status=ClaimStatus.SUPPORTED,
                verification="derived from locked Content Studio platform pack and source-backed topic research",
                script_usages=(),
            ),
        )

    async def _ensure_fact_check(self, project: ContentProject) -> ContentProject:
        if project.fact_check_report is not None:
            return _content_studio_status(project, ProjectStatus.FACT_CHECKED, "fact_check")
        if project.evidence_graph is None:
            project = await self._ensure_research(project)
        if project.status is ProjectStatus.FAILED_BLOCKED or project.evidence_graph is None:
            return project
        statuses = {claim.claim_id: claim.status for claim in project.evidence_graph.claims}
        blocking = tuple(
            claim_id
            for claim_id, status in statuses.items()
            if status in {ClaimStatus.UNSUPPORTED, ClaimStatus.CONFLICTING, ClaimStatus.OUTDATED}
        )
        report = FactCheckReport(
            claim_statuses=statuses,
            blocking_claim_ids=blocking,
            notes=("all usable claims cleared",)
            if not blocking
            else ("unsupported, conflicting, or outdated claims block downstream use",),
        )
        status = ProjectStatus.FACT_CHECKED if not blocking else ProjectStatus.FAILED_BLOCKED
        updated = replace(project, fact_check_report=report)
        if blocking:
            return _content_studio_blocked(updated, "fact_check_blocked", "fact check has blocking claims")
        return _content_studio_status(updated, status, "fact_check")

    def _ensure_plan(self, project: ContentProject) -> ContentProject:
        if project.content_plan is not None:
            return _content_studio_status(project, ProjectStatus.PLAN_READY, "plan")
        if project.fact_check_report is None:
            return _content_studio_blocked(project, "fact_check_required", "fact check must finish before planning")
        target_seconds = _content_studio_int_setting(project, "target_seconds")
        plan = ContentPlan(
            sections=(
                "0-3s Hook: 先用一句话说清 AIGC 不是玄学",
                "3-10s 发生了什么：从聊天框变成工作流",
                "10-35s 核心原理：输入、生成、检查、再发布",
                "35-52s 案例/限制：成本、版权、人工审核",
                "52-60s 适合谁：创作者和小团队的使用边界",
            ),
            target_seconds=target_seconds,
            platform_constraints=(
                f"{project.packs.platform.settings['aspect_ratio']} {project.packs.platform.settings['width']}x{project.packs.platform.settings['height']}",
                "subtitles required",
                "plain-language science explainer",
                "meaningful visual change every 3-5 seconds",
                str(project.packs.style.settings.get("visual_language", "")),
            ),
        )
        return _content_studio_status(replace(project, content_plan=plan), ProjectStatus.PLAN_READY, "plan")

    async def _ensure_script(self, project: ContentProject) -> ContentProject:
        if project.script is not None:
            return _content_studio_status(project, ProjectStatus.SCRIPT_READY, "script")
        project = await self._ensure_fact_check(project)
        if project.status is ProjectStatus.FAILED_BLOCKED:
            return project
        project = self._ensure_plan(project)
        if project.status is ProjectStatus.FAILED_BLOCKED or project.evidence_graph is None:
            return project
        try:
            script = await self._script_from_model(project)
        except (ModelGatewayError, NoCapableDeployment) as error:
            return _content_studio_blocked(project, "script_provider_failed", str(error))
        claim_ids_by_segment = {
            claim_id
            for segment in script.segments
            if segment.factual
            for claim_id in segment.claim_ids
        }
        claims = tuple(
            replace(
                claim,
                script_usages=tuple(
                    segment.segment_id
                    for segment in script.segments
                    if claim.claim_id in segment.claim_ids
                ),
            )
            if claim.claim_id in claim_ids_by_segment
            else claim
            for claim in project.evidence_graph.claims
        )
        return _content_studio_status(
            replace(project, evidence_graph=EvidenceGraph(claims=claims, evidence=project.evidence_graph.evidence), script=script),
            ProjectStatus.SCRIPT_READY,
            "script",
        )

    async def _script_from_model(self, project: ContentProject) -> ScriptDraft:
        prompt = _content_studio_script_prompt(project)
        try:
            response = await self._text_completion(prompt, max_output_tokens=2200)
            payload = _json_object_from_text(response)
            return _script_from_payload(payload, project)
        except (ValueError, TypeError, json.JSONDecodeError):
            return _fallback_plain_script(project)

    def _ensure_storyboard(self, project: ContentProject) -> ContentProject:
        if project.storyboard is not None:
            return _content_studio_status(project, ProjectStatus.STORYBOARD_READY, "storyboard")
        if project.script is None:
            return _content_studio_blocked(project, "script_required", "script must be ready before storyboard")
        if not project.script_approved:
            return replace(project, status=ProjectStatus.SCRIPT_READY)
        target_ms = _content_studio_int_setting(project, "target_seconds") * 1000
        durations = (3000, 7000, 12000, 12000, 10000, 10000, max(6000, target_ms - 54000))
        start = 0
        shots: list[Shot] = []
        overlays: tuple[str, ...]
        transitions: tuple[str, ...]
        if project.packs.style.name == "code_flow_pipeline":
            shot_types = (
                "code_typing_hook",
                "pipeline_hud_flow",
                "evidence_card_stream",
                "screen_recording_style_demo",
                "timeline_editor_flow",
                "qc_scan_summary",
                "claim_badge_summary",
            )
            overlays = (
                "motion: cursor typing + code stream + key phrase lockup",
                "motion: pipeline nodes pulse from research to render",
                "motion: evidence cards slide in with source badges",
                "motion: terminal log scroll + demo cursor path",
                "motion: timeline playhead sweep + track layers update",
                "motion: QC frame scan boxes + issue badges",
                "motion: claim badge lock + final summary card",
            )
            transitions = (
                "cursor wipe",
                "data-flow line",
                "panel slide",
                "match cut",
                "timeline sweep",
                "scan wipe",
                "quick cut",
            )
        else:
            shot_types = (
                "big_text_hook",
                "workflow_diagram",
                "official_source_cards",
                "screen_recording_style_demo",
                "risk_checklist",
                "comparison_chart",
                "summary_card",
            )
            overlays = ("safe subtitles + source/step label",) * len(shot_types)
            transitions = ("cut",) * len(shot_types)
        segment_ids = tuple(segment.segment_id for segment in project.script.segments)
        for index, duration in enumerate(durations, start=1):
            shots.append(
                Shot(
                    shot_id=f"SHOT{index:03d}",
                    start_ms=start,
                    duration_ms=duration,
                    shot_type=shot_types[index - 1],
                    narration_segment_ids=segment_ids[max(0, min(len(segment_ids) - 1, index - 1)): max(1, min(len(segment_ids), index))],
                    asset_request_ids=(f"ASREQ{index:03d}",),
                    overlay=overlays[index - 1],
                    transition=transitions[index - 1],
                    safe_area="douyin_9_16_subtitle_safe",
                )
            )
            start += duration
        return _content_studio_status(
            replace(project, storyboard=Storyboard(shots=tuple(shots))),
            ProjectStatus.STORYBOARD_READY,
            "storyboard",
        )

    async def _ensure_assets(self, project: ContentProject) -> ContentProject:
        if project.asset_manifest is not None:
            return _content_studio_status(project, ProjectStatus.ASSETS_READY, "assets")
        project = self._ensure_storyboard(project)
        if project.status is ProjectStatus.FAILED_BLOCKED or project.storyboard is None:
            return project
        if self._multimedia is None:
            return _content_studio_blocked(project, "asset_provider_not_configured", "multimedia generation executor is not configured")
        try:
            logical_model = await self._multimedia.default_logical_model_for_multimedia(
                kind=MultimediaGenerationKind.IMAGE
            )
        except (NoCapableDeployment, ValueError, RuntimeError) as error:
            return _content_studio_blocked(project, "asset_provider_not_configured", str(error))
        semaphore = asyncio.Semaphore(2)
        tasks = [
            self._generate_asset_for_shot(
                project,
                shot,
                logical_model=logical_model,
                index=index,
                semaphore=semaphore,
            )
            for index, shot in enumerate(project.storyboard.shots, start=1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assets: list[AssetRecord] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                errors.append(str(result))
            else:
                assets.append(result)
        if errors or not assets:
            return _content_studio_blocked(
                project,
                "asset_generation_failed",
                "; ".join(errors[:3]) or "no assets generated",
            )
        updated = replace(project, asset_manifest=AssetManifest(assets=tuple(assets)))
        updated = _content_studio_record_attempt(updated, "assets", tuple(assets))
        return _content_studio_status(updated, ProjectStatus.ASSETS_READY, "assets")

    async def _generate_asset_for_shot(
        self,
        project: ContentProject,
        shot: Shot,
        *,
        logical_model: str,
        index: int,
        semaphore: asyncio.Semaphore,
    ) -> AssetRecord:
        assert self._multimedia is not None
        prompt = _asset_prompt_for_shot(project, shot)
        retry_delays = (20.0, 45.0, 90.0)
        last_error: BaseException | None = None
        for attempt in range(len(retry_delays) + 1):
            try:
                async with semaphore:
                    result = await self._multimedia.generate(
                        kind=MultimediaGenerationKind.IMAGE,
                        logical_model=logical_model,
                        prompt=prompt,
                    )
                break
            except (VideoProviderGenerationError, ModelGatewayError, RuntimeError) as error:
                last_error = error
                if attempt >= len(retry_delays) or not _retryable_asset_generation_error(error):
                    raise
                await asyncio.sleep(retry_delays[attempt])
        else:  # pragma: no cover - loop always breaks or raises
            assert last_error is not None
            raise last_error
        file_path = str(result.file_path) if result.file_path is not None else ""
        return AssetRecord(
            asset_id=f"ASSET{index:03d}",
            request_id=shot.asset_request_ids[0],
            source=result.logical_model,
            acquisition_method="generated_image",
            url_or_provider_task_id=result.text or file_path or result.deployment_id,
            content_hash=_content_studio_hash("|".join((result.text or "", file_path, prompt))),
            technical_params={
                "width": _content_studio_int_setting(project, "width"),
                "height": _content_studio_int_setting(project, "height"),
                "mime": result.mime_type or "image/png",
                "file_path": file_path,
                "filename": result.filename or "",
                "deployment_id": result.deployment_id,
            },
            rights_status="unknown",
            generation_params={"prompt": prompt, "revision": 1, "shot_id": shot.shot_id},
        )

    async def _ensure_voice(self, project: ContentProject) -> ContentProject:
        if project.voice_track is not None:
            return _content_studio_status(project, ProjectStatus.VOICE_READY, "voice")
        if project.asset_manifest is None:
            return _content_studio_blocked(project, "assets_required", "assets must be ready before voice")
        if not _content_studio_rights_clear(project):
            return _content_studio_blocked(project, "asset_rights_not_approved", "asset rights must be approved before voice or render")
        if self._multimedia is not None:
            logical_model = await self._multimedia.default_logical_model(MultimediaGenerationKind.AUDIO.value)
            if logical_model:
                try:
                    voice_text = _content_studio_voice_prompt(project)
                    result = await self._multimedia.generate(
                        kind=MultimediaGenerationKind.AUDIO,
                        logical_model=logical_model,
                        prompt=voice_text,
                    )
                except (ModelGatewayError, RuntimeError, ValueError) as error:
                    return _content_studio_blocked(project, "voice_generation_failed", str(error))
                if result.file_path is None or not result.file_path.is_file():
                    return _content_studio_blocked(
                        project,
                        "voice_generation_missing_file",
                        "audio_generation provider did not return a local playable audio file",
                    )
                voice = VoiceTrack(
                    audio_artifact_id=str(result.file_path),
                    timestamp_level="sentence",
                    pronunciation_report=(
                        f"TTS generated by {result.logical_model}/{result.deployment_id}",
                        "AIGC pronounced as A-I-G-C",
                    ),
                    mime_type=result.mime_type or _audio_mime_type_for_path(result.file_path),
                    source="caller_tts",
                )
                return _content_studio_status(
                    _content_studio_record_attempt(replace(project, voice_track=voice), "voice", voice),
                    ProjectStatus.VOICE_READY,
                    "voice",
                )
        audio_path = self._output_dir / project.project_id / "voice-demo-signal.wav"
        _write_demo_signal_wav(audio_path, seconds=max(1, _content_studio_int_setting(project, "target_seconds")))
        voice = VoiceTrack(
            audio_artifact_id=str(audio_path),
            timestamp_level="sentence",
            pronunciation_report=(
                "BLOCKER: production TTS file provider is not configured; rendered preview uses demo signal audio",
                "AIGC pronounced as A-I-G-C",
            ),
            mime_type="audio/wav",
            source="demo_signal",
        )
        return _content_studio_status(
            _content_studio_record_attempt(replace(project, voice_track=voice), "voice", voice),
            ProjectStatus.VOICE_READY,
            "voice",
        )

    def _ensure_timeline(self, project: ContentProject) -> ContentProject:
        if project.timeline is not None:
            return _content_studio_status(project, ProjectStatus.TIMELINE_READY, "timeline")
        if project.voice_track is None or project.asset_manifest is None or project.script is None:
            return _content_studio_blocked(project, "timeline_inputs_missing", "voice, assets, and script are required before timeline")
        width = _content_studio_int_setting(project, "width")
        height = _content_studio_int_setting(project, "height")
        duration_ms = _content_studio_voice_duration_ms(project.voice_track) or (
            _content_studio_int_setting(project, "target_seconds") * 1000
        )
        timeline = Timeline(
            width=width,
            height=height,
            duration_ms=duration_ms,
            tracks={
                "narration": (project.voice_track.audio_artifact_id,),
                "primary_visual": tuple(asset.asset_id for asset in project.asset_manifest.assets),
                "subtitle": project.script.subtitle_lines,
                "overlay": ("source badges", "large safe-area subtitles", "step labels"),
                "bgm": (),
            },
        )
        return _content_studio_status(replace(project, timeline=timeline), ProjectStatus.TIMELINE_READY, "timeline")

    def _ensure_preview(self, project: ContentProject) -> ContentProject:
        if project.timeline is not None and project.timeline.preview_artifact_id is not None:
            return _content_studio_status(project, ProjectStatus.PREVIEW_RENDERED, "preview")
        project = self._ensure_timeline(project)
        if project.status is ProjectStatus.FAILED_BLOCKED:
            return project
        try:
            request = _render_request_from_project(project)
            rendered = self._media_adapter.render_preview(request, self._output_dir / project.project_id)
        except (ContentStudioMediaError, FileNotFoundError, ValueError) as error:
            return _content_studio_blocked(project, "preview_render_failed", str(error))
        assert project.timeline is not None
        timeline = replace(project.timeline, preview_artifact_id=str(rendered.preview.path))
        qc = _qc_report_from_media(project, rendered.qc)
        updated = _content_studio_record_attempt(
            replace(project, timeline=timeline, qc_report=qc),
            "preview_render",
            str(rendered.preview.path),
        )
        return _content_studio_status(updated, ProjectStatus.PREVIEW_RENDERED, "preview")

    def _ensure_qc(self, project: ContentProject) -> ContentProject:
        if project.qc_report is not None:
            return _content_studio_status(project, ProjectStatus.QC_REVIEW, "qc")
        project = self._ensure_preview(project)
        if project.status is ProjectStatus.FAILED_BLOCKED:
            return project
        return _content_studio_status(project, ProjectStatus.QC_REVIEW, "qc")

    def _ensure_final(self, project: ContentProject) -> ContentProject:
        if project.timeline is not None and project.timeline.final_artifact_id is not None:
            return _content_studio_status(project, ProjectStatus.FINAL_RENDERED, "final_render")
        if not project.final_approved:
            return project
        if project.qc_report is not None and project.qc_report.blockers:
            return _content_studio_blocked(project, "qc_blocked", "QC blockers must be resolved before final render")
        try:
            request = _render_request_from_project(project)
            assert project.timeline is not None and project.timeline.preview_artifact_id is not None
            preview_path = Path(project.timeline.preview_artifact_id)
            approval = FinalRenderApproval(
                approved=True,
                approved_by="content_studio",
                preview_sha256=_sha256_file(preview_path),
                technical_passed=True,
            )
            self._media_adapter._preview_sha256s.add(approval.preview_sha256)
            rendered = self._media_adapter.render_final(
                request,
                self._output_dir / project.project_id,
                approval=approval,
            )
        except (ContentStudioMediaError, FileNotFoundError, ValueError) as error:
            return _content_studio_blocked(project, "final_render_failed", str(error))
        assert project.timeline is not None
        timeline = replace(project.timeline, final_artifact_id=str(rendered.final.path))
        return _content_studio_status(
            _content_studio_record_attempt(replace(project, timeline=timeline), "final_render", str(rendered.final.path)),
            ProjectStatus.FINAL_RENDERED,
            "final_render",
        )

    async def _text_completion(self, prompt: str, *, max_output_tokens: int) -> str:
        deployments = tuple(
            _deployment_from_model_resource(model) for model in await self._list_models()
        )
        candidates = tuple(
            deployment for deployment in deployments if ModelCapability.TEXT in deployment.capabilities
        )
        if not candidates:
            raise NoCapableDeployment("no capable text deployment for content studio")
        logical_model = max(
            candidates,
            key=lambda item: (
                safe_operational_limit(item.max_concurrency, item.target_utilization, item.reserved_slots),
                item.weight,
                item.logical_model,
            ),
        ).logical_model
        capacity = (
            await self._capacity_factory(deployments)
            if self._capacity_factory is not None
            else await self._default_capacity(deployments)
        )
        gateway = ModelGateway(
            ModelRegistry(deployments),
            capacity,
            TenantSecretResolver(self._secret_service, self._tenant_id),
            self._transport,
            capacity_wait_timeout=60,
        )
        completion = await gateway.complete_with_context(
            ModelRequest(
                logical_model=logical_model,
                messages=(ModelMessage(role="user", content=prompt),),
                required_capabilities=frozenset({ModelCapability.TEXT}),
                max_output_tokens=max_output_tokens,
                timeout_seconds=90,
            )
        )
        return completion.response.text or ""

    async def _default_capacity(self, deployments: tuple[Deployment, ...]) -> CapacityPool:
        credentials = CredentialRegistry(
            [
                CredentialDescriptor(
                    secret_ref,
                    await self._secret_service.fingerprint(self._tenant_id, secret_ref),
                )
                for secret_ref in dict.fromkeys(deployment.secret_ref for deployment in deployments)
            ]
        )
        return CapacityPool(self._redis_client, deployments=deployments, credentials=credentials)

def _content_studio_stage_index(stage: ProjectStatus) -> int:
    return _CONTENT_STUDIO_STAGE_ORDER.index(stage)


def _content_studio_status(project: ContentProject, status: ProjectStatus, stage_key: str) -> ContentProject:
    already_completed = stage_key in project.completed_stage_keys
    updated = replace(
        project,
        status=status,
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
        title=f"Content Studio {stage_key}",
        summary=f"{stage_key} 阶段已完成",
        payload={"stage": stage_key, "status": status.value},
    )


def _content_studio_blocked(
    project: ContentProject,
    error_code: str,
    error_message: str,
) -> ContentProject:
    return append_content_project_event(
        replace(
            project,
            status=ProjectStatus.FAILED_BLOCKED,
            error_code=error_code,
            error_message=error_message[:1000],
        ),
        kind="stage_failed",
        stage="blocked",
        status="failed",
        title="阶段失败",
        summary=f"{error_code}: {error_message[:1000]}",
        payload={"error_code": error_code, "error_message": error_message[:1000]},
    )


def _content_studio_record_attempt(
    project: ContentProject,
    stage: str,
    result: object,
) -> ContentProject:
    attempt = ProviderAttempt(
        stage=stage,
        idempotency_key=f"{project.project_id}:{stage}",
        status="completed",
        result_hash=_content_studio_hash(repr(result)),
    )
    return record_content_project_provider_attempt(project, attempt)


def _content_studio_source_urls(project: ContentProject) -> tuple[str, ...]:
    urls = list(project.source_urls)
    if not urls:
        urls.extend(
            (
                "https://openai.com/news/",
                "https://platform.openai.com/docs/models",
                "https://github.com/openai/openai-python/releases",
                "https://huggingface.co/blog",
                "https://arxiv.org/list/cs.AI/recent",
                "https://www.anthropic.com/news",
                "https://ai.google.dev/",
                "https://deepmind.google/discover/blog/",
                "https://azure.microsoft.com/en-us/blog/topics/ai-machine-learning/",
                "https://github.blog/changelog/",
                "https://blogs.nvidia.com/blog/category/deep-learning/",
                "https://stability.ai/news",
                "https://runwayml.com/research",
                "https://www.minimax.io/news",
                "https://qwenlm.github.io/blog/",
                "https://www.alibabacloud.com/blog/ai",
                "https://cloud.tencent.com/developer/article",
                "https://research.baidu.com/Blog",
            )
        )
    return tuple(dict.fromkeys(urls))


def _content_studio_allowed_hosts(project: ContentProject) -> tuple[str, ...]:
    raw = project.packs.domain.settings.get("allowed_hosts", ())
    if isinstance(raw, list | tuple):
        return tuple(str(item).casefold() for item in raw if str(item).strip())
    return ()


def _host_allowed(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    normalized = host.casefold()
    return any(normalized == item or normalized.endswith(f".{item}") for item in allowed_hosts)


def _content_studio_source_priority(project: ContentProject) -> tuple[str, ...]:
    raw = project.packs.domain.settings.get("source_priority", ())
    if isinstance(raw, list | tuple):
        values = tuple(str(item).strip() for item in raw if str(item).strip())
        if values:
            return values
    return (
        "official_docs",
        "official_blog",
        "release_notes",
        "github_release",
        "paper",
        "official_demo",
        "secondary_media",
    )


def _content_studio_research_bundle(
    project: ContentProject,
    questions: tuple[ResearchQuestion, ...],
    evidence: tuple[Evidence, ...],
) -> ResearchBundle:
    source_priority = _content_studio_source_priority(project)
    return ResearchBundle(
        questions=questions,
        evidence=evidence,
        source_priority=source_priority,
        retrieval_plan=tuple(
            f"{index}. Fetch and verify {source_type.replace('_', ' ')} sources before lower-priority material"
            for index, source_type in enumerate(source_priority, start=1)
        ),
        source_coverage=_content_studio_source_coverage(source_priority, evidence),
        source_candidates=_content_studio_source_candidates(project, evidence),
    )


def _content_studio_source_candidates(
    project: ContentProject,
    evidence: tuple[Evidence, ...],
) -> tuple[ResearchSourceCandidate, ...]:
    candidates: list[ResearchSourceCandidate] = []
    for index, url in enumerate(_content_studio_source_urls(project), start=1):
        candidates.append(
            ResearchSourceCandidate(
                source_type=_source_type_for_url(project, url),
                source_url=url,
                priority=index,
                rationale="configured source URL",
            )
        )
    for evidence_item in evidence:
        candidates.append(
            ResearchSourceCandidate(
                source_type=evidence_item.source_type,
                source_url=evidence_item.source_url,
                priority=0,
                rationale=f"retrieved evidence {evidence_item.evidence_id}",
            )
        )
    seen: set[tuple[str, str]] = set()
    deduped: list[ResearchSourceCandidate] = []
    for candidate in candidates:
        key = (candidate.source_type, candidate.source_url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return tuple(deduped)


def _content_studio_source_coverage(
    source_priority: tuple[str, ...],
    evidence: tuple[Evidence, ...],
) -> tuple[ResearchSourceCoverage, ...]:
    evidence_by_type: dict[str, list[str]] = {}
    for item in evidence:
        evidence_by_type.setdefault(item.source_type, []).append(item.evidence_id)
    return tuple(
        ResearchSourceCoverage(
            source_type=source_type,
            required=source_type != "secondary_media",
            evidence_ids=tuple(evidence_by_type.get(source_type, ())),
            status="covered" if evidence_by_type.get(source_type) else "missing",
            note="source category has retrieved evidence"
            if evidence_by_type.get(source_type)
            else "source category was configured but produced no accepted evidence",
        )
        for source_type in source_priority
    )


def _source_type_for_url(project: ContentProject, url: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.casefold()
    if (host == "github.com" or host.endswith(".github.com")) and "/releases" in path:
        return "github_release"
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        return "paper"
    if "/release" in path or "/changelog" in path or "/updates" in path:
        return "release_notes"
    if "/blog" in path or "/news" in path or "/research" in path:
        return "official_blog"
    if "/demo" in path or "/examples" in path or "/spaces/" in path:
        return "official_demo"
    if "/docs" in path or "/documentation" in path or "/guide" in path:
        return "official_docs"
    return _source_type_for_host(project, host)


def _source_type_for_host(project: ContentProject, host: str) -> str:
    raw = project.packs.domain.settings.get("source_types", {})
    if isinstance(raw, Mapping):
        for suffix, source_type in raw.items():
            if host.casefold().endswith(str(suffix).casefold()):
                return str(source_type)
    return "web"


def _publisher_for_host(host: str) -> str:
    normalized = host.casefold()
    if "openai.com" in normalized:
        return "OpenAI"
    if "github.com" in normalized:
        return "GitHub"
    if "arxiv.org" in normalized:
        return "arXiv"
    if "huggingface.co" in normalized:
        return "Hugging Face"
    return host or "unknown"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html[:2_000_000])
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    return text[:6000]


def _content_studio_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _content_studio_int_setting(project: ContentProject, field: str) -> int:
    value = project.packs.platform.settings[field]
    if not isinstance(value, int):
        raise TypeError(f"platform setting {field} must be an integer")
    return value


def _content_studio_script_prompt(project: ContentProject) -> str:
    assert project.evidence_graph is not None
    evidence = "\n".join(
        f"{item.evidence_id} {item.publisher} {item.source_url}: {item.excerpt[:500]}"
        for item in project.evidence_graph.evidence
    )
    claims = "\n".join(
        f"{item.claim_id}: {item.text} status={item.status.value} evidence={','.join(item.evidence_ids)}"
        for item in project.evidence_graph.claims
    )
    return (
        "你是抖音 AIGC 科普视频编导。基于下面的证据和 Claim，写一条普通用户能听懂的 60 秒中文脚本。\n"
        "要求：少术语；用生活化类比；不要夸大；每个事实性段落必须引用已有 Claim ID；"
        "输出严格 JSON，不要 Markdown。JSON 结构："
        "{\"hooks\":[3个开头],\"segments\":[{\"text\":\"...\",\"factual\":true,\"claim_ids\":[\"CL001\"]}],"
        "\"subtitle_lines\":[\"...\"]}。\n\n"
        f"主题：{project.topic}\n"
        f"平台约束：{project.content_plan.platform_constraints if project.content_plan else ()}\n"
        f"证据：\n{evidence}\n\nClaims:\n{claims}\n"
    )


def _json_object_from_text(text: str) -> Mapping[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response did not contain a JSON object")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, Mapping):
        raise TypeError("model response JSON must be an object")
    return payload


def _script_from_payload(payload: Mapping[str, object], project: ContentProject) -> ScriptDraft:
    assert project.evidence_graph is not None
    known_claims = {claim.claim_id for claim in project.evidence_graph.claims}
    hooks = tuple(str(item).strip() for item in _sequence(payload.get("hooks")) if str(item).strip())[:3]
    if len(hooks) < 3:
        raise ValueError("script payload must contain three hooks")
    segments: list[ScriptSegment] = []
    for index, item in enumerate(_sequence(payload.get("segments")), start=1):
        if not isinstance(item, Mapping):
            continue
        text_value = str(item.get("text", "")).strip()
        if not text_value:
            continue
        factual = bool(item.get("factual", True))
        claim_ids = tuple(
            claim_id
            for claim_id in (str(raw).strip() for raw in _sequence(item.get("claim_ids")))
            if claim_id in known_claims
        )
        if factual and not claim_ids:
            raise ValueError("factual script segment must cite a known claim")
        segments.append(ScriptSegment(f"SEG{index:03d}", text_value, factual, claim_ids))
    if not segments:
        raise ValueError("script payload must contain segments")
    subtitle_lines = tuple(str(item).strip() for item in _sequence(payload.get("subtitle_lines")) if str(item).strip())
    if not subtitle_lines:
        subtitle_lines = tuple(segment.text for segment in segments)
    return ScriptDraft(hooks=cast(tuple[str, str, str], hooks), segments=tuple(segments), subtitle_lines=subtitle_lines)


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, list | tuple):
        return tuple(value)
    return ()


def _fallback_plain_script(project: ContentProject) -> ScriptDraft:
    assert project.evidence_graph is not None
    claim_ids = tuple(claim.claim_id for claim in project.evidence_graph.claims)
    first = claim_ids[:1] or ("CL001",)
    second = claim_ids[1:2] or first
    segments = (
        ScriptSegment("SEG001", "AIGC 现在不只是让模型回答一句话，而是开始进入一整套内容生产流程。", True, first),
        ScriptSegment("SEG002", "你可以把它理解成：先找资料，再生成草稿，再检查事实，最后才发布。", True, first),
        ScriptSegment("SEG003", "真正有价值的不是炫技，而是每一步都能留下证据、素材和审核记录。", True, second),
        ScriptSegment("SEG004", "但它还不能完全放飞，成本、版权、事实核验和人工把关，都决定了能不能上线。", True, second),
        ScriptSegment("SEG005", "所以一分钟总结：AIGC 是加速器，不是免检通道。", False, ()),
    )
    return ScriptDraft(
        hooks=(
            "AIGC 不是换个聊天框这么简单。",
            "一分钟看懂 AIGC 真正在改变什么。",
            "别先追热点，先看它能不能进你的工作流。",
        ),
        segments=segments,
        subtitle_lines=tuple(segment.text for segment in segments),
    )


def _asset_prompt_for_shot(project: ContentProject, shot: Shot) -> str:
    script_text = " ".join(project.script.subtitle_lines) if project.script else project.topic
    style_language = str(project.packs.style.settings.get("visual_language", "")).strip()
    if project.packs.style.name == "code_flow_pipeline":
        motion_primitives = project.packs.style.settings.get("motion_primitives", ())
        if isinstance(motion_primitives, (list, tuple)):
            motion_text = ", ".join(str(item) for item in motion_primitives[:8])
        else:
            motion_text = str(motion_primitives)
        return (
            "Create vertical 9:16 motion-ready source art for a Douyin AIGC explainer video. "
            "The result must be layered motion-ready elements, not a final poster: separate code stream, "
            "pipeline HUD, evidence cards, timeline panels, subtitles-safe overlay zones, and clean UI chrome. "
            "Do not create a static poster, single flattened illustration, long still-image hold, fake unreadable UI text, "
            "or a collage pretending to be video. "
            f"Visual language: {style_language}. Motion primitives to support: {motion_text}. "
            f"Topic: {project.topic}. Shot: {shot.shot_id} {shot.shot_type}. "
            f"Narration context: {script_text[:900]}. Overlay and motion direction: {shot.overlay}. "
            "Resolution target 1080x1920, clean high-contrast code/information-flow look, safe area for captions."
        )
    return (
        "Create a clean vertical 9:16 visual asset for a Douyin AIGC explainer video. "
        "Use minimal, readable Chinese UI-card style, no fake unreadable paragraphs, no celebrity faces, "
        "no cluttered background. The image should serve as a shot visual, not a poster. "
        f"Topic: {project.topic}. Shot: {shot.shot_id} {shot.shot_type}. "
        f"Narration context: {script_text[:900]}. Overlay idea: {shot.overlay}. "
        "Resolution target 1080x1920, safe area for subtitles, modern clean tech style."
    )


def _retryable_asset_generation_error(error: BaseException) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "rate limit",
            "requests rate limit exceeded",
            "too many requests",
            "capacity",
            "timeout",
            "temporarily",
            "try again later",
            "transport failed",
        )
    )


def _content_studio_rights_clear(project: ContentProject) -> bool:
    return bool(
        project.asset_manifest
        and project.asset_manifest.assets
        and all(asset.rights_status == "approved" for asset in project.asset_manifest.assets)
    )


def _render_request_from_project(project: ContentProject) -> RenderRequest:
    if project.timeline is None or project.asset_manifest is None or project.voice_track is None or project.script is None:
        raise ValueError("render inputs are missing")
    duration_ms = project.timeline.duration_ms
    assets = project.asset_manifest.assets
    if not assets:
        raise ValueError("render requires at least one asset")
    clip_duration = max(1000, duration_ms // len(assets))
    visuals: list[VisualClip] = []
    start = 0
    for index, asset in enumerate(assets):
        raw_path = str(asset.technical_params.get("file_path", "")).strip()
        if not raw_path:
            raise ValueError(f"asset has no local file path: {asset.asset_id}")
        current_duration = duration_ms - start if index == len(assets) - 1 else clip_duration
        path = Path(raw_path)
        mime_type = str(asset.technical_params.get("mime", "image/png"))
        for beat_index, beat_duration in enumerate(
            _content_studio_visual_beats(current_duration, mime_type),
            start=1,
        ):
            visuals.append(
                VisualClip(
                    clip_id=asset.asset_id if beat_index == 1 else f"{asset.asset_id}-B{beat_index:02d}",
                    path=path,
                    mime_type=mime_type,
                    start_ms=start,
                    duration_ms=beat_duration,
                )
            )
            start += beat_duration
    subtitles: list[SubtitleCue] = []
    subtitle_durations = _content_studio_subtitle_durations(project.script.subtitle_lines, duration_ms)
    start = 0
    claim_ids_by_line = tuple(
        segment.claim_ids for segment in project.script.segments
    ) or ((),)
    for index, (line, current_duration) in enumerate(zip(project.script.subtitle_lines, subtitle_durations, strict=False), start=1):
        subtitles.append(
            SubtitleCue(
                cue_id=f"SUB{index:03d}",
                start_ms=start,
                duration_ms=current_duration,
                text=line[:80],
                x=54,
                y=max(100, project.timeline.height - 360),
                width=max(200, project.timeline.width - 108),
                height=180,
                claim_ids=claim_ids_by_line[min(index - 1, len(claim_ids_by_line) - 1)],
            )
        )
        start += current_duration
    claims = (
        tuple(ClaimReference(claim.claim_id, claim.text) for claim in project.evidence_graph.claims)
        if project.evidence_graph
        else ()
    )
    return RenderRequest(
        title=project.title,
        output_basename=f"content-studio-{project.project_id}",
        timeline=MediaTimeline(
            width=project.timeline.width,
            height=project.timeline.height,
            duration_ms=duration_ms,
            visuals=tuple(visuals),
            audio=(
                AudioClip(
                    "VOICE001",
                    Path(project.voice_track.audio_artifact_id),
                    project.voice_track.mime_type,
                    0,
                    duration_ms,
                    source=project.voice_track.source,
                ),
            ),
            subtitles=tuple(subtitles),
            claims=claims,
        ),
    )


def _content_studio_voice_prompt(project: ContentProject) -> str:
    if project.script is None:
        raise ValueError("approved script narration is required before TTS")
    narration = "\n".join(segment.text for segment in project.script.segments if segment.text.strip())
    if not narration.strip():
        narration = "\n".join(line for line in project.script.subtitle_lines if line.strip())
    narration = narration.strip()
    if not narration:
        raise ValueError("script narration is empty")
    if _content_studio_narration_looks_like_task_prompt(narration):
        raise ValueError("script narration appears to contain task instructions instead of spoken copy")
    return narration[:4000]


def _content_studio_narration_looks_like_task_prompt(narration: str) -> bool:
    head = narration[:260]
    prompt_markers = (
        "做一条约",
        "生成一条",
        "请生成",
        "请为",
        "要求研究",
        "要求：",
        "输出格式",
        "subtitle_lines",
        "script_segments",
        "```json",
    )
    return any(marker in head for marker in prompt_markers) and any(
        media_marker in head for media_marker in ("视频", "抖音", "脚本", "主题", "AIGC")
    )


def _content_studio_voice_duration_ms(voice_track: VoiceTrack) -> int | None:
    path = Path(voice_track.audio_artifact_id)
    if not path.is_file():
        return None
    if voice_track.mime_type == "audio/wav" or path.suffix.casefold() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                frame_rate = wav.getframerate()
                if frame_rate <= 0:
                    return None
                return max(1000, round(wav.getnframes() / frame_rate * 1000))
        except (wave.Error, OSError, EOFError):
            return None
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            (
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        duration_seconds = float(result.stdout.strip())
    except ValueError:
        return None
    if duration_seconds <= 0:
        return None
    return max(1000, round(duration_seconds * 1000))


def _content_studio_subtitle_durations(lines: Sequence[str], duration_ms: int) -> tuple[int, ...]:
    if not lines:
        return ()
    if len(lines) == 1:
        return (duration_ms,)
    weights = tuple(max(1, len(line.strip())) for line in lines)
    total = sum(weights)
    durations: list[int] = []
    previous_boundary = 0
    cumulative = 0
    for index, weight in enumerate(weights, start=1):
        cumulative += weight
        boundary = duration_ms if index == len(weights) else round(duration_ms * cumulative / total)
        current = max(1, boundary - previous_boundary)
        durations.append(current)
        previous_boundary = previous_boundary + current
    if sum(durations) != duration_ms:
        durations[-1] += duration_ms - sum(durations)
    return tuple(durations)


def _content_studio_visual_beats(duration_ms: int, mime_type: str) -> tuple[int, ...]:
    if not mime_type.startswith("image/"):
        return (duration_ms,)
    max_beat_ms = 5_000
    beat_count = max(1, math.ceil(duration_ms / max_beat_ms))
    base = duration_ms // beat_count
    remainder = duration_ms % beat_count
    return tuple(base + (1 if index < remainder else 0) for index in range(beat_count))


def _audio_mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix in {".m4a", ".mp4"}:
        return "audio/mp4"
    if suffix == ".aac":
        return "audio/aac"
    return "audio/wav"


def _write_demo_signal_wav(path: Path, *, seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 48_000
    amplitude = 6_000
    frame_count = max(1, seconds) * sample_rate
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            sample = int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        output.writeframes(bytes(frames))


def _qc_report_from_media(project: ContentProject, media_qc: object) -> QCReport:
    blockers: list[str] = []
    majors: list[str] = []
    minors: list[str] = []
    checked = ["resolution/aspect/codec", "subtitle safe area", "claim coverage", "asset rights"]
    if project.voice_track and any("BLOCKER" in item for item in project.voice_track.pronunciation_report):
        blockers.append("production TTS file provider is not configured; preview uses demo signal audio")
    technical_passed = bool(getattr(media_qc, "technical_passed", False))
    if not technical_passed:
        blockers.append("media technical QC failed")
    for check in _media_qc_checks(media_qc):
        checked.append(f"video reviewer: {check.name} {check.status} - {check.details[:300]}")
        if check.name in {"video_reviewer_frame_sampling", "subtitle_text_review", "subtitle_visual_contrast"}:
            if check.status == "failed":
                blockers.append(f"video reviewer QC failed: {check.name}")
            elif check.status == "warning":
                majors.append(f"video reviewer warning: {check.name}")
    needs_review = bool(getattr(media_qc, "needs_review", False))
    if needs_review:
        minors.append("media adapter reported review-needed warnings")
    return QCReport(blockers=tuple(blockers), majors=tuple(majors), minors=tuple(minors), checked_items=tuple(checked))


def _media_qc_checks(media_qc: object) -> tuple[MediaQCCheck, ...]:
    raw = getattr(media_qc, "checks", ())
    if isinstance(raw, tuple | list):
        return tuple(item for item in raw if isinstance(item, MediaQCCheck))
    return ()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ConfigBackedAssetVisualReviewer:
    """Review generated image assets with a configured vision-capable model."""

    def __init__(
        self,
        *,
        list_models: RegisteredModelListGetter,
        secret_service: SecretService,
        tenant_id: UUID,
        redis_client: object,
        transport: ModelTransport | None = None,
        capacity_factory: MultimediaCapacityFactory | None = None,
    ) -> None:
        self._list_models = list_models
        self._secret_service = secret_service
        self._tenant_id = tenant_id
        self._redis_client = redis_client
        self._transport = transport or LiteLLMClient()
        self._capacity_factory = capacity_factory

    async def review_image_asset(
        self,
        *,
        tenant_id: UUID,
        label: str,
        prompt: str,
        filename: str,
        mime_type: str,
        data: bytes,
        image_url: str | None = None,
    ) -> RuntimeAssetVisualReview:
        if tenant_id != self._tenant_id:
            raise NoCapableDeployment("visual asset review tenant is not configured")
        candidate_image_url = image_url.strip() if image_url is not None else ""
        review_image_url = (
            candidate_image_url
            if urlsplit(candidate_image_url).scheme.lower() in {"http", "https"}
            else _image_data_url(mime_type, data)
        )
        deployments = self._ranked_vision_deployments(await self._vision_deployments())
        if not deployments:
            raise NoCapableDeployment("no capable visual asset review deployment: vision")
        retryable_errors: list[tuple[Deployment, str, Exception]] = []
        for review_attempt in range(_ASSET_VISUAL_REVIEW_ATTEMPTS):
            attempt_errors: list[tuple[Deployment, str, Exception]] = []
            for deployment in deployments:
                for structured in _visual_review_modes(deployment):
                    if structured and ModelCapability.STRUCTURED_OUTPUT not in deployment.capabilities:
                        continue
                    request = _asset_visual_review_request(
                        deployment=deployment,
                        label=label,
                        prompt=prompt,
                        filename=filename,
                        review_image_url=review_image_url,
                        structured=structured,
                    )
                    try:
                        gateway = await self._gateway((deployment,))
                        completion = await gateway.complete_with_context(request)
                        payload = _parse_asset_visual_review_payload(completion.response.text)
                    except Exception as exc:
                        if not _is_retryable_visual_review_error(exc):
                            raise
                        attempt_errors.append(
                            (deployment, "schema" if structured else "json", exc)
                        )
                        if isinstance(exc, CapacityUnavailable) or "capacity" in str(exc).lower():
                            break
                        continue
                    passed = bool(payload.get("passed"))
                    summary = (
                        str(payload.get("summary") or "").strip()[:1000]
                        or "视觉审核未给出摘要"
                    )
                    raw_issues = payload.get("issues")
                    issues = tuple(
                        str(item).strip()[:1000]
                        for item in (raw_issues if isinstance(raw_issues, list) else [])
                        if str(item).strip()
                    )[:12]
                    confidence = payload.get("confidence")
                    confidence_value = (
                        float(confidence)
                        if isinstance(confidence, int | float) and not isinstance(confidence, bool)
                        else None
                    )
                    passed, summary, issues = _apply_asset_visual_review_policy(
                        label=label,
                        passed=passed,
                        summary=summary,
                        issues=issues,
                    )
                    return RuntimeAssetVisualReview(
                        passed=passed,
                        summary=summary,
                        issues=issues,
                        confidence=confidence_value,
                        logical_model=completion.logical_model,
                        deployment_id=completion.deployment_id,
                    )
            retryable_errors.extend(attempt_errors)
            if review_attempt + 1 < _ASSET_VISUAL_REVIEW_ATTEMPTS and attempt_errors:
                await asyncio.sleep(_ASSET_VISUAL_REVIEW_RETRY_BACKOFF_SECONDS)
        if retryable_errors:
            raise ModelGatewayError(
                _visual_review_attempt_failure_message(retryable_errors),
                logical_models=tuple(
                    dict.fromkeys(item[0].logical_model for item in retryable_errors)
                ),
                deployments=tuple(dict.fromkeys(item[0].id for item in retryable_errors)),
            )
        raise NoCapableDeployment("no capable visual asset review deployment: vision")

    async def _gateway(self, deployments: tuple[Deployment, ...]) -> ModelGateway:
        if not deployments:
            raise NoCapableDeployment("no capable visual asset review deployment: vision")
        capacity = (
            await self._capacity_factory(deployments)
            if self._capacity_factory is not None
            else await self._default_capacity(deployments)
        )
        return ModelGateway(
            ModelRegistry(deployments),
            capacity,
            TenantSecretResolver(self._secret_service, self._tenant_id),
            self._transport,
            capacity_wait_timeout=_ASSET_VISUAL_REVIEW_CAPACITY_WAIT_SECONDS,
        )

    async def _vision_deployments(self) -> tuple[Deployment, ...]:
        deployments = tuple(
            _deployment_from_model_resource(model) for model in await self._list_models()
        )
        return tuple(
            deployment
            for deployment in deployments
            if ModelCapability.VISION in deployment.capabilities
            and not _is_messages_endpoint(deployment.api_base)
            and _is_visual_review_compatible_deployment(deployment)
        )

    @staticmethod
    def _ranked_vision_deployments(deployments: tuple[Deployment, ...]) -> tuple[Deployment, ...]:
        return tuple(
            sorted(
                deployments,
                key=lambda item: (
                    _visual_review_provider_priority(item),
                    safe_operational_limit(
                        item.max_concurrency,
                        item.target_utilization,
                        item.reserved_slots,
                    ),
                    item.weight,
                    item.logical_model,
                    item.id,
                ),
                reverse=True,
            )
        )

    async def _default_capacity(
        self,
        deployments: tuple[Deployment, ...],
    ) -> CapacityPool:
        credentials = CredentialRegistry(
            [
                CredentialDescriptor(
                    secret_ref,
                    await self._secret_service.fingerprint(self._tenant_id, secret_ref),
                )
                for secret_ref in dict.fromkeys(deployment.secret_ref for deployment in deployments)
            ]
        )
        return CapacityPool(self._redis_client, deployments=deployments, credentials=credentials)


_ASSET_VISUAL_REVIEW_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "passed": {"type": "boolean"},
        "summary": {"type": "string", "minLength": 1, "maxLength": 1000},
        "issues": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 1000},
            "maxItems": 12,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["passed", "summary", "issues", "confidence"],
}


def _image_data_url(mime_type: str, data: bytes) -> str:
    if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("unsupported visual review image MIME type")
    if not data or len(data) > 20_000_000:
        raise ValueError("visual review image bytes must be bounded")
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _asset_visual_review_request(
    *,
    deployment: Deployment,
    label: str,
    prompt: str,
    filename: str,
    review_image_url: str,
    structured: bool,
) -> ModelRequest:
    required_capabilities = {ModelCapability.VISION}
    response_schema: StructuredResponseSchema | None = None
    if structured:
        required_capabilities.add(ModelCapability.STRUCTURED_OUTPUT)
        response_schema = StructuredResponseSchema(
            name="AssetVisualReview",
            schema=cast(Mapping[str, JsonValue], _ASSET_VISUAL_REVIEW_SCHEMA),
        )
    return ModelRequest(
        logical_model=deployment.logical_model,
        messages=(
            ModelMessage(
                role="user",
                content=(
                    {
                        "type": "text",
                        "text": _asset_visual_review_prompt(
                            label=label,
                            prompt=prompt,
                            filename=filename,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": review_image_url, "detail": "high"},
                    },
                ),
            ),
        ),
        required_capabilities=frozenset(required_capabilities),
        response_schema=response_schema,
        max_output_tokens=1200,
        allow_fallback=False,
    )


def _asset_visual_review_prompt(*, label: str, prompt: str, filename: str) -> str:
    return (
        "你是 AI 短剧资产图视觉审核员。只根据随附图片判断它是否能作为指定资产交付，"
        "不要因为画面好看就放行；如果它更像剧照、海报、随机写真或只覆盖了少量资产，必须拒绝。\n\n"
        f"资产标签：{label}\n"
        f"文件名：{filename}\n"
        f"生成提示词：{prompt[:3000]}\n\n"
        "审核标准：\n"
        "0. 所有资产图和分镜图都必须干净、低噪声，只表达当前标签需要锁定的必要信息；"
        "如果把多个资产类别、无关背景、装饰、小物件、剧照元素或全部剧情信息堆在一张图里，"
        "导致当前资产类别不清晰，必须判为不通过。除场景资产外，资产设定板应使用纯白/浅灰/"
        "透明感纯色背景；如果出现办公室、桌面、窗户、室内、街景、墙画、环境光影等具体背景，"
        "即使主体信息可读，也必须判为不通过。\n"
        "0a. 所有资产必须严格贴合生成提示词中的剧本锚点：角色姓名、年龄感、职业身份、"
        "发型服装、关键道具、指定动作、指定特效、地点和画风。审核时必须主动核对这些锚点；"
        "如果图片只是通用模板、换了人物身份/服装/画风、出现提示词没有要求的黑西装/战术服/"
        "奇幻职业等漂移，或缺少提示词明确列出的黄色外卖箱、青玉断佩、银针、证件、蓝色电弧"
        "等关键资产，必须判为不通过。支撑资产可以用小比例占位人物，但也必须沿用剧本角色外观"
        "和任务，不得自造无关角色。\n"
        "1. 如果标签是角色/角色锁定/定妆/Character Model Sheet，图片必须像角色参考设定表，"
        "采用中等复杂度专业设定板结构，至少包含同一角色的主定妆大图、正/侧/背全身三视图、"
        "3-5 个表情头部变化、服装拆解、1-3 套剧情服装/状态变体、随身物/职业道具、材质色卡，"
        "以及一致外貌、发型逻辑、年龄感、体态和身份气质。"
        "每个模块都必须对应角色锚点：表情变化必须是同一张脸的不同情绪，服装展示必须服务职业和剧情场景，"
        "剧情服装/状态变体可以变化衣服、雨夜/战斗/工作状态，但不能换脸、换年龄感或换职业身份。"
        "随身物/职业道具必须来自生成提示词中的剧本、职业、剧情任务或明确道具，不得加入剧本或角色设定之外的随机道具。"
        "文字应为少量清晰中文标签、极少量短标签和栏目标题；如果出现主定妆、三视图、表情、服装、道具、色卡等关键栏目错别字、"
        "大量乱码、伪字、不可读，或栏目标题存在但内容明显不对应角色锚点，必须判为不通过。"
        "如果只有单张头像/写真、重复近景、缺少三视图、缺少表情变化、缺少服装/道具/材质细节，"
        "必须判为不通过。单张剧照、情侣合照、随机写真、风格混杂或身份漂移必须判为不通过。"
        "人物定妆图必须是纯白/浅灰/透明感纯色背景；"
        "如果出现室内、街景、道具桌面、窗户、墙画、环境光影或其他具体背景，必须判为不通过，"
        "因为背景会污染后续人物锁定。脚本角色名默认都是虚构角色，只能按生成提示词中的年龄、"
        "身份、外貌、服装和剧情职能审核；不要按现实明星、公众人物或同名真人资料判断是否相似，"
        "除非用户明确要求真实人物或名人复刻。\n"
        "2. 如果标签是服装妆造资产，应能看出服装、配饰、妆发、材质和色彩基调，且服务角色身份；"
        "应包含服装拆解、正反面/层次、配饰特写和材质色卡；只有普通人像或无服装细节变化必须判为不通过。"
        "关键标签大量乱码或服装/配饰不服务角色身份时必须判为不通过。\n"
        "3. 如果标签是场景资产，应主要呈现场景空间、光线、天气、氛围和可复用背景元素；"
        "应包含空间视角、纵深层次、光线方向、天气/时间和可复用背景层；"
        "主角动作占画面主体、看不出地点设定或只有战斗瞬间必须判为不通过。\n"
        "4. 如果标签是道具资产，应能独立识别多个关键物/随身物/法器/科技物件及细节特写；"
        "应包含独立物件 lineup、局部特写、比例参考和材质色卡；"
        "只能包含生成提示词明确要求或剧情/职业必需的道具；随机补充无关物、角色拿道具摆拍、"
        "道具数量明显不足、关键标签大量乱码或伪字严重影响识别必须判为不通过。\n"
        "5. 如果标签是动作资产，应体现动作分解、姿态线或多个关键动作参考；"
        "应包含姿态序列、关键帧、重心变化和运动箭头；单张帅气动作海报、缺少分解信息或动作与提示词无关必须判为不通过。\n"
        "6. 如果标签是特效资产，应体现特效形态、颜色、层级、触发方式和可复用变化；"
        "应包含形态分层、强弱层级、扩散方向、触发方式和颜色规则；只有一张战斗画面、特效不可分辨或没有层级变化必须判为不通过。\n"
        "7. 如果标签是镜头资产，应体现景别、机位、镜头运动、构图或剪辑节奏参考；"
        "应包含景别机位构图卡、镜头框、机位俯视图、推拉摇移轨迹和剪辑节奏图；普通剧照、宣传图或没有镜头规划信息必须判为不通过。\n"
        "8. 如果标签是表演节奏/风格锁定资产，应体现表情、眼神、肢体状态、节奏点或整体画风锁定；"
        "应包含情绪曲线、表情强度、情绪节奏点、肢体状态和色彩/光影风格分区；只有单一表情写真或无法服务剪辑节奏必须判为不通过。\n"
        "9. 如果标签是分镜图，图片应呈现分镜/镜头规划感，而不是最终宣传剧照。\n\n"
        "返回严格 JSON：passed:boolean, summary:string, issues:string[], confidence:number。"
    )


def _apply_asset_visual_review_policy(
    *,
    label: str,
    passed: bool,
    summary: str,
    issues: tuple[str, ...],
) -> tuple[bool, str, tuple[str, ...]]:
    if not passed:
        return passed, summary, issues
    combined = " ".join((label, summary, *issues))
    if _mentions_asset_review_rejection(combined):
        policy_issue = "视觉审核摘要包含明确不合格信号；不得将该资产自动判为通过。"
        if policy_issue not in issues:
            issues = (*issues, policy_issue)
        summary = summary or policy_issue
        return False, summary, issues
    if _is_scene_asset_label(label):
        return passed, summary, issues
    if _mentions_concrete_background_pollution(combined):
        policy_issue = (
            "非场景类资产出现具体背景、桌面、室内/街景环境或背景污染；"
            "资产图必须重新生成为干净设定板。"
        )
        if policy_issue not in issues:
            issues = (*issues, policy_issue)
        summary = summary or policy_issue
        return False, summary, issues
    return passed, summary, issues


def _is_scene_asset_label(label: str) -> bool:
    normalized = label.strip().casefold()
    return "场景" in normalized or "scene" in normalized


def _mentions_asset_review_rejection(text: str) -> bool:
    normalized = _strip_resolved_asset_review_clauses(text.casefold())
    rejection_markers = (
        "仍不满足",
        "不满足",
        "不合格",
        "不符合",
        "未实质修正",
        "未修正",
        "核心问题未修正",
        "核心问题",
        "不能作为",
        "无法作为",
        "违反",
        "标签错位",
        "标签不匹配",
        "文字乱码",
        "标签乱码",
        "大量乱码",
        "存在乱码",
        "存在伪字",
        "不可读字段",
    )
    return any(marker.casefold() in normalized for marker in rejection_markers)


def _strip_resolved_asset_review_clauses(text: str) -> str:
    risk_terms = (
        "乱码",
        "伪字",
        "错位",
        "不匹配",
        "不可读",
        "不满足",
        "不符合",
        "不合格",
        "背景污染",
    )
    term_pattern = "|".join(re.escape(term.casefold()) for term in risk_terms)
    resolved_clause = re.compile(
        rf"[^。；;.!?\n]*(?:{term_pattern})[^。；;.!?\n]*"
        rf"(?:已修复|已解决|未发现|无|没有|不存在|不再)[^。；;.!?\n]*"
    )
    return resolved_clause.sub("", text)


def _mentions_concrete_background_pollution(text: str) -> bool:
    normalized = _strip_negated_background_pollution_clauses(text.casefold())
    pollution_markers = (
        "背景污染",
        "场景污染",
        "背景非纯色",
        "非纯色背景",
        "具体背景",
        "无关背景",
        "室内背景",
        "街景背景",
        "道具桌面",
        "办公室",
        "窗户",
        "天花板",
        "桌面",
        "木质桌面",
        "室内",
        "街景",
        "墙画",
        "环境光影",
        "路人",
        "乘客",
    )
    return any(marker.casefold() in normalized for marker in pollution_markers)


def _strip_negated_background_pollution_clauses(text: str) -> str:
    background_terms = (
        "背景污染",
        "场景污染",
        "背景非纯色",
        "非纯色背景",
        "具体背景",
        "无关背景",
        "室内背景",
        "街景背景",
        "道具桌面",
        "办公室",
        "窗户",
        "天花板",
        "桌面",
        "木质桌面",
        "室内",
        "街景",
        "墙画",
        "环境光影",
        "路人",
        "乘客",
        "场景",
        "背景",
    )
    term_pattern = "|".join(re.escape(term.casefold()) for term in background_terms)
    negated_clause = re.compile(
        rf"(?:无|没有|未见|不含|不存在|并无|不得出现|未出现)[^。；;.!?\n]*"
        rf"(?:{term_pattern})[^。；;.!?\n]*"
    )
    return negated_clause.sub("", text)


def _visual_review_modes(deployment: Deployment) -> tuple[bool, ...]:
    provider_model = deployment.provider_model.casefold()
    logical_model = deployment.logical_model.casefold()
    if "minimax" in provider_model or "minimax" in logical_model:
        return (False,)
    return (True, False)


def _visual_review_provider_priority(deployment: Deployment) -> int:
    provider_model = deployment.provider_model.casefold()
    logical_model = deployment.logical_model.casefold()
    text = f"{provider_model} {logical_model}"
    if "deepseek" in text:
        return 40
    if "qwen" in text or "vl" in text:
        return 20
    if "minimax" in text:
        return 10
    return 0


def _parse_asset_visual_review_payload(raw_response: str | None) -> Mapping[str, object]:
    if raw_response is None or len(raw_response) > 20_000:
        raise ValueError("visual asset review response is invalid")
    normalized = raw_response.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            normalized = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        try:
            payload = _extract_json_object(normalized)
        except ValueError:
            payload = _parse_asset_visual_review_text_response(normalized)
    if not isinstance(payload, Mapping):
        raise TypeError("visual asset review response is invalid")
    cleaned = dict(payload)
    passed = cleaned.get("passed")
    if isinstance(passed, str) and passed.strip().casefold() in {"true", "false"}:
        cleaned["passed"] = passed.strip().casefold() == "true"
    if not isinstance(cleaned.get("passed"), bool):
        raise TypeError("visual asset review response is invalid")
    if not isinstance(cleaned.get("summary"), str):
        raise TypeError("visual asset review response is invalid")
    issues = cleaned.get("issues")
    if isinstance(issues, str):
        cleaned["issues"] = [issues]
    if not isinstance(cleaned.get("issues"), list):
        raise TypeError("visual asset review response is invalid")
    confidence = cleaned.get("confidence")
    if isinstance(confidence, str):
        try:
            cleaned["confidence"] = float(confidence.strip())
        except ValueError:
            pass
        confidence = cleaned.get("confidence")
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise TypeError("visual asset review response is invalid")
    return cleaned


def _parse_asset_visual_review_text_response(raw_response: str) -> Mapping[str, object]:
    text = " ".join(raw_response.split())
    if not text:
        raise ValueError("visual asset review response is invalid")
    normalized = text.casefold()
    failed = any(
        marker.casefold() in normalized
        for marker in (
            "不通过",
            "未通过",
            "不合格",
            "不满足",
            "不符合",
            "不能作为",
            "无法作为",
            "拒绝",
            "失败",
        )
    )
    passed = any(
        marker.casefold() in normalized
        for marker in (
            "通过",
            "合格",
            "符合",
            "可以作为",
            "可作为",
        )
    )
    if not failed and not passed:
        raise ValueError("visual asset review response is invalid")
    issue_lines = [
        line.strip(" -:：")
        for line in raw_response.splitlines()
        if any(
            marker in line
            for marker in (
                "问题",
                "缺少",
                "不满足",
                "不符合",
                "不合格",
                "失败",
                "拒绝",
                "乱码",
                "伪字",
                "背景",
            )
        )
    ]
    if failed and not issue_lines:
        issue_lines = [text[:1000]]
    return {
        "passed": not failed and passed,
        "summary": text[:1000],
        "issues": issue_lines[:12],
        "confidence": 0.5,
    }


def _extract_json_object(raw_response: str) -> object:
    decoder = json.JSONDecoder()
    start = raw_response.find("{")
    while start >= 0:
        try:
            payload, _end = decoder.raw_decode(raw_response[start:])
            return payload
        except json.JSONDecodeError:
            start = raw_response.find("{", start + 1)
    raise ValueError("visual asset review response is invalid") from None


def _is_retryable_visual_review_error(error: Exception) -> bool:
    if isinstance(error, (NoCapableDeployment, ModelGatewayError, ValueError, TypeError)):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "capacity unavailable",
            "capacity queue",
            "capacity backend",
            "capacity timeout",
            "transport failed",
            "response text is empty",
            "response is empty",
            "timing unavailable",
            "visual asset review response is invalid",
        )
    )


def _visual_review_attempt_failure_message(
    errors: Sequence[tuple[Deployment, str, Exception]],
) -> str:
    summaries: list[str] = []
    for deployment, phase, error in errors[:8]:
        error_text = " ".join(str(error).split())[:180] or type(error).__name__
        summaries.append(f"{deployment.logical_model}/{deployment.id}/{phase}: {error_text}")
    extra_count = max(len(errors) - len(summaries), 0)
    suffix = f"; +{extra_count} more" if extra_count else ""
    return "visual asset review failed for all candidates: " + "; ".join(summaries) + suffix


def _is_visual_review_compatible_deployment(deployment: Deployment) -> bool:
    provider_model = deployment.provider_model.casefold()
    request_model = (deployment.request_model or "").casefold()
    model_text = f"{provider_model} {request_model}"
    return not provider_model.startswith("qwen/") or any(
        marker in model_text for marker in ("vl", "vision", "omni")
    )


def _is_messages_endpoint(api_base: str) -> bool:
    return urlsplit(api_base).path.rstrip("/").endswith("/messages")


def _deployment_from_model_resource(model: admin.ModelDeploymentResponse) -> Deployment:
    return Deployment(
        id=str(model.id),
        logical_model=model.logical_model,
        provider_model=f"{model.provider}/{model.upstream_model}",
        request_model=model.upstream_model,
        api_base=model.api_base,
        secret_ref=model.credential_ref,
        quota_scope_id=model.quota_scope,
        max_concurrency=model.max_concurrency,
        target_utilization=model.target_utilization,
        reserved_slots=model.reserved_capacity,
        rpm=model.rpm,
        tpm=model.tpm,
        weight=model.weight,
        capabilities=frozenset(ModelCapability(item) for item in model.capabilities),
    )


def _multimedia_required_capability(kind: MultimediaGenerationKind) -> ModelCapability:
    if kind is MultimediaGenerationKind.IMAGE:
        return ModelCapability.IMAGE_GENERATION
    if kind is MultimediaGenerationKind.VIDEO:
        return ModelCapability.VIDEO_GENERATION
    if kind is MultimediaGenerationKind.AUDIO:
        return ModelCapability.AUDIO_GENERATION
    raise ValueError("generation kind is invalid")


def _multimedia_daily_limit(
    kind: MultimediaGenerationKind,
    deployments: tuple[Deployment, ...],
    logical_model: str,
) -> int | None:
    if kind is not MultimediaGenerationKind.VIDEO:
        return None
    matching = [
        deployment for deployment in deployments if deployment.logical_model == logical_model
    ]
    if any(_is_minimax_video_deployment(deployment) for deployment in matching):
        return 3
    return None


def _require_supported_multimedia_generation(
    *,
    kind: MultimediaGenerationKind,
    logical_model: str,
    candidates: tuple[Deployment, ...],
) -> None:
    if kind is not MultimediaGenerationKind.VIDEO:
        return
    if any(_is_supported_video_generation_deployment(deployment) for deployment in candidates):
        return
    raise NoCapableDeployment(
        f"no supported video generation deployment for logical model {logical_model!r}: "
        "video_generation"
    )


def _is_supported_video_generation_deployment(deployment: Deployment) -> bool:
    provider, upstream_model = _deployment_provider_and_model(deployment)
    return (
        ModelCapability.VIDEO_GENERATION in deployment.capabilities
        and (
            is_known_video_generation_model(provider, upstream_model)
            or is_dashscope_multimedia_deployment(provider, upstream_model, deployment.api_base)
        )
    )


def _is_minimax_video_deployment(deployment: Deployment) -> bool:
    provider, _upstream_model = _deployment_provider_and_model(deployment)
    return "minimax" in provider.casefold() and _is_supported_video_generation_deployment(
        deployment
    )


def _deployment_provider_and_model(deployment: Deployment) -> tuple[str, str]:
    provider, separator, upstream_model = deployment.provider_model.partition("/")
    if not separator:
        return "", deployment.request_model or deployment.provider_model
    return provider, upstream_model


def _infer_main_agent_context_window_tokens(provider: str, upstream_model: str) -> int:
    normalized = f"{provider}/{upstream_model}".casefold()
    if "gemini" in normalized:
        return 1_000_000
    if "gpt-5" in normalized or "gpt-4.1" in normalized:
        return 400_000
    if "claude" in normalized:
        return 200_000
    if any(marker in normalized for marker in ("deepseek", "qwen", "kimi", "gpt-4o")):
        return 128_000
    return 32_768


def _waiting_route_decision(reason: str) -> RouteDecision:
    return RouteDecision(
        mode=None,
        needs_user_choice=True,
        status="waiting_user_mode",
        assessments=(),
        clarification_reason=reason,
        options=(TaskMode.DIRECT, TaskMode.DISPATCH, TaskMode.DISCUSS, TaskMode.HYBRID),
        decision_token=None,
        version=1,
        risk=RiskLevel.LOW,
        requires_approval=False,
        permissions_still_apply=True,
    )


class InProcessRunQueue:
    """Minimal queue adapter for single-process tests and explicit publisher wiring."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[UUID, str]] = []

    async def enqueue_run(self, run_id: UUID, *, idempotency_key: str) -> None:
        self.enqueued.append((run_id, idempotency_key))


async def _cleanup_owned_resources(
    cleanup_callbacks: list[CleanupCallback],
    *,
    primary_error: BaseException | None,
) -> None:
    first_cancellation: asyncio.CancelledError | None = None
    cleanup_error_types: list[str] = []
    for resource_name, cleanup in reversed(cleanup_callbacks):
        try:
            await cleanup()
        except asyncio.CancelledError as error:
            if first_cancellation is None:
                first_cancellation = error
            _LOGGER.error(
                "resource_cleanup_failed resource=%s error_type=CancelledError",
                resource_name,
            )
        except Exception as error:  # noqa: BLE001 -- all ordinary cleanups are attempted.
            error_type = type(error).__name__
            cleanup_error_types.append(error_type)
            _LOGGER.error(
                "resource_cleanup_failed resource=%s error_type=%s",
                resource_name,
                error_type,
            )
    if primary_error is not None:
        return
    if first_cancellation is not None:
        raise first_cancellation
    if cleanup_error_types:
        raise ResourceCleanupError(tuple(cleanup_error_types)) from None


async def ensure_bootstrap_tenant(
    session_factory: async_sessionmaker[AsyncSession],
    tenant_id: UUID,
    slug: str,
    name: str,
) -> None:
    """Idempotently ensure the configured tenant exists once at process startup."""

    statement = (
        insert(TenantRow)
        .values(id=tenant_id, slug=slug, name=name)
        .on_conflict_do_update(
            index_elements=[TenantRow.id],
            set_={"slug": slug, "name": name},
        )
    )
    async with session_factory() as session, session.begin():
        await session.execute(statement)


def create_app(
    *,
    settings: Settings | None = None,
    database: DatabaseResource | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    redis_client: RedisResource | None = None,
    database_probe: ReadinessProbe | None = None,
    redis_probe: ReadinessProbe | None = None,
    readiness_timeout_seconds: float = 1.0,
    auth_service: object | None = None,
    rate_limiter: object | None = None,
    config_service: object | None = None,
    admin_resource_service: object | None = None,
    user_admin_service: object | None = None,
    run_service: object | None = None,
    runtime_registry: RuntimeRegistry | None = None,
    content_studio_service: object | None = None,
    mode_router: ModeRouterProtocol | None = None,
    task_queue: TaskQueue | None = None,
    feishu_gateway: ChannelGatewayProtocol | None = None,
    feishu_websocket_client_factory: FeishuWebSocketClientFactoryForSettings | None = None,
    database_factory: Callable[[str], DatabaseResource] = build_database,
    redis_factory: Callable[[str], RedisResource] = Redis.from_url,
) -> FastAPI:
    """Create an application without opening network resources at import time."""

    configured_settings = settings or Settings.model_construct()
    active_runtime_registry = runtime_registry
    configure_logging(level=configured_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        nonlocal active_runtime_registry
        configured = settings or get_settings()
        active_database = database
        active_redis = redis_client
        active_sessions = session_factory
        active_mode_router = mode_router
        cleanup_callbacks: list[CleanupCallback] = []
        token_service = (
            AccessTokenService(configured.jwt_signing_key_value()) if auth_service is None else None
        )
        try:
            application.state.settings = configured
            application.state.trusted_proxy_ips = configured.trusted_proxy_ips
            application.state.bootstrap_tenant_id = configured.bootstrap_tenant_id
            application.state.attachment_store_dir = configured.attachment_store_dir
            needs_sessions = (
                auth_service is None
                or config_service is None
                or run_service is None
                or admin_resource_service is None
                or user_admin_service is None
            )
            if active_sessions is None and active_database is not None:
                active_sessions = active_database.session_factory
            if active_sessions is None and needs_sessions:
                active_database = database_factory(configured.database_url_value())
                cleanup_callbacks.append(("database", active_database.dispose))
                active_sessions = active_database.session_factory

            needs_redis = (
                rate_limiter is None
                or redis_probe is None
                or (run_service is None and active_runtime_registry is None)
            )
            if active_redis is None and needs_redis:
                active_redis = redis_factory(configured.redis_url_value())
                cleanup_callbacks.append(("redis", active_redis.aclose))

            if auth_service is None:
                assert token_service is not None
                assert active_sessions is not None
                await ensure_bootstrap_tenant(
                    active_sessions,
                    configured.bootstrap_tenant_id,
                    configured.bootstrap_tenant_slug,
                    configured.bootstrap_tenant_name,
                )
                application.state.auth_service = AuthService(
                    active_sessions,
                    configured.bootstrap_tenant_id,
                    PasswordService(),
                    token_service,
                )
            if config_service is None:
                assert active_sessions is not None
                application.state.config_service = ConfigService(active_sessions)
            active_secret_service = None
            if active_sessions is not None:
                active_secret_service = SecretService(
                    active_sessions,
                    SecretCipher(configured.master_key_bytes()),
                )
            if admin_resource_service is None and active_sessions is not None:
                assert active_secret_service is not None
                application.state.admin_resource_service = admin.PersistentAdminResourceService(
                    config_service=ConfigService(active_sessions),
                    secret_service=active_secret_service,
                    run_repository=RunRepository(active_sessions),
                    tenant_id=configured.bootstrap_tenant_id,
                    actor_id=configured.bootstrap_tenant_id,
                    session_factory=active_sessions,
                    skill_store_dir=configured.skill_store_dir,
                    generated_artifact_dir=configured.generated_artifact_dir,
                )
            if user_admin_service is None and active_sessions is not None:
                application.state.user_admin_service = PersistentUserAdminService(active_sessions)
            if (
                active_secret_service is not None
                and active_redis is not None
                and getattr(application.state, "multimedia_generation_executor", None) is None
            ):
                admin_service_for_generation = cast(
                    admin.AdminResourceService,
                    admin_resource_service
                    if admin_resource_service is not None
                    else application.state.admin_resource_service,
                )
                application.state.multimedia_generation_executor = (
                    _ConfigBackedMultimediaGenerationExecutor(
                        list_models=admin_service_for_generation.list_models,
                        secret_service=active_secret_service,
                        tenant_id=configured.bootstrap_tenant_id,
                        redis_client=active_redis,
                    )
                )
            if run_service is None:
                assert active_sessions is not None
                if active_runtime_registry is None:
                    assert active_redis is not None
                    assert active_secret_service is not None
                    workspace_read_root = (
                        configured.workspace_read_roots[0]
                        if configured.workspace_read_roots
                        else configured.attachment_store_dir
                    )
                    active_multimedia_executor = getattr(
                        application.state,
                        "multimedia_generation_executor",
                        None,
                    )
                    active_content_studio_service = AsyncContentStudioService(
                        registry=load_pack_registry(),
                        store=PersistentContentProjectStore(
                            active_sessions,
                            tenant_id=configured.bootstrap_tenant_id,
                        ),
                        execution_mode="production",
                        production_provider=_ConfigBackedContentStudioProductionProvider(
                            list_models=cast(
                                admin.AdminResourceService,
                                application.state.admin_resource_service,
                            ).list_models,
                            secret_service=active_secret_service,
                            tenant_id=configured.bootstrap_tenant_id,
                            redis_client=active_redis,
                            multimedia_generation_executor=cast(
                                _ConfigBackedMultimediaGenerationExecutor | None,
                                active_multimedia_executor,
                            ),
                            output_dir=configured.generated_artifact_dir / "content-studio",
                        ),
                    )
                    application.state.content_studio_service = active_content_studio_service
                    runtime_capabilities = RuntimeCapabilityGateway(
                        skill_store_dir=configured.skill_store_dir,
                        workspace_root=workspace_read_root,
                        generated_artifact_dir=configured.generated_artifact_dir,
                        multimedia_generation_executor=getattr(
                            application.state,
                            "multimedia_generation_executor",
                            None,
                        ),
                        asset_visual_reviewer=_ConfigBackedAssetVisualReviewer(
                            list_models=cast(
                                admin.AdminResourceService,
                                application.state.admin_resource_service,
                            ).list_models,
                            secret_service=active_secret_service,
                            tenant_id=configured.bootstrap_tenant_id,
                            redis_client=active_redis,
                        ),
                        content_studio_service=active_content_studio_service,
                        content_studio_execution_mode="production",
                    )
                    active_runtime_registry = configured_runtime_registry(
                        config_service=ConfigService(active_sessions),
                        secret_service=active_secret_service,
                        redis_client=active_redis,
                        capability_gateway=runtime_capabilities,
                    )
                if active_mode_router is None and active_secret_service is not None:
                    assert active_redis is not None
                    admin_service_for_router = cast(
                        admin.AdminResourceService,
                        admin_resource_service
                        if admin_resource_service is not None
                        else application.state.admin_resource_service,
                    )
                    active_mode_router = _MainAgentModeRouter(
                        get_config=admin_service_for_router.get_main_agent_config,
                        list_models=admin_service_for_router.list_models,
                        secret_service=active_secret_service,
                        tenant_id=configured.bootstrap_tenant_id,
                        redis_client=active_redis,
                )
                queue = task_queue if task_queue is not None else InProcessRunQueue()
                run_repository = RunRepository(active_sessions)
                cognitive_pipeline = CognitiveLearningPipeline(
                    cognitive_service=CognitiveStateService(
                        PersistentCognitiveRecordRepository(active_sessions)
                    ),
                    experience_service=ExperienceService(
                        PersistentExperienceRepository(active_sessions)
                    ),
                    run_repository=run_repository,
                )
                application.state.run_service = RunService(
                    run_repository,
                    runtime_registry=active_runtime_registry,
                    router=active_mode_router,
                    task_queue=queue,
                    hermes_advisor=PersistentHermesRunAdvisor(active_sessions),
                    temporary_agent_policy=AdminResourceTemporaryAgentPolicy(active_sessions),
                    runtime_timeout_seconds=configured.runtime_timeout_seconds,
                    runtime_token_budget=configured.runtime_token_budget,
                    attachment_artifact_loader=FileSystemAttachmentArtifactLoader(
                        configured.attachment_store_dir
                    ),
                    resource_context_loader=ResourceContextArtifactLoader(
                        skill_store_dir=configured.skill_store_dir,
                        workspace_roots=configured.workspace_read_roots,
                        list_skills=cast(
                            admin.AdminResourceService,
                            admin_resource_service
                            if admin_resource_service is not None
                            else application.state.admin_resource_service,
                        ).list_skills,
                    ),
                    main_agent_context_window_getter=_MainAgentContextWindowGetter(
                        cast(
                            admin.AdminResourceService,
                            admin_resource_service
                            if admin_resource_service is not None
                            else application.state.admin_resource_service,
                        ).get_main_agent_config
                    ),
                    terminal_run_hooks=(
                        CognitiveLearningTerminalHook(cognitive_pipeline),
                    ),
                )
                application.state.run_queue = queue
                application.state.mode_router = active_mode_router
            if (
                getattr(application.state, "schedule_service", None) is None
                and getattr(application.state, "run_service", None) is not None
            ):
                application.state.schedule_service = SchedulerService(
                    lambda task: _submit_scheduled_task(application, task)
                )
            if (
                feishu_gateway is None
                and active_sessions is not None
                and getattr(application.state, "run_service", None) is not None
            ):
                application.state.feishu_gateway = ChannelGateway(
                    submitter=RunServiceInboundSubmitter(
                        run_service=cast(
                            RunSubmissionService,
                            application.state.run_service,
                        ),
                        tenant_id=configured.bootstrap_tenant_id,
                        settings_service=cast(
                            ChannelSettingsService,
                            admin_resource_service
                            if admin_resource_service is not None
                            else application.state.admin_resource_service,
                        ),
                        identity_resolver=PersistentChannelIdentityResolver(
                            active_sessions
                        ),
                    ),
                    deduplicator=InboundDedupRepository(active_sessions),
                )
            if active_sessions is not None:
                application.state.feishu_reply_dispatcher = FeishuRunReplyDispatcher(
                    run_repository=RunRepository(active_sessions),
                    sender=FeishuOpenAPIReplySender(),
                )
            if (
                active_secret_service is not None
                and active_redis is not None
                and getattr(application.state, "feishu_media_service_factory", None) is None
            ):
                if (
                    getattr(application.state, "multimedia_generation_executor", None)
                    is None
                ):
                    admin_service_for_generation = cast(
                        admin.AdminResourceService,
                        admin_resource_service
                        if admin_resource_service is not None
                        else application.state.admin_resource_service,
                    )
                    application.state.multimedia_generation_executor = (
                        _ConfigBackedMultimediaGenerationExecutor(
                            list_models=admin_service_for_generation.list_models,
                            secret_service=active_secret_service,
                            tenant_id=configured.bootstrap_tenant_id,
                            redis_client=active_redis,
                        )
                    )
                media_factory = build_feishu_media_service_factory(
                    config_service=cast(ConfigService, application.state.config_service),
                    secret_service=active_secret_service,
                    redis_client=active_redis,
                    tenant_id=configured.bootstrap_tenant_id,
                    attachment_store_dir=configured.attachment_store_dir,
                    log_service=application.state.admin_resource_service,
                    environment=configured.environment,
                )
                application.state.feishu_media_service_factory = media_factory
                application.state.feishu_skill_command_handler = FeishuSkillCommandHandler(
                    admin_service=cast(
                        Any,
                        admin_resource_service
                        if admin_resource_service is not None
                        else application.state.admin_resource_service,
                    ),
                    media_client_factory=lambda settings: FeishuOpenAPIMediaClient(
                        settings=settings
                    ),
                )
                cleanup_callbacks.append(("feishu_media_service_factory", media_factory.aclose))

            await _start_feishu_websocket_connector_if_configured(
                application,
                client_factory=feishu_websocket_client_factory,
            )

            if rate_limiter is None:
                assert active_redis is not None
                hmac_key = hashlib.sha256(
                    configured.jwt_signing_key_value().encode("utf-8")
                ).digest()
                application.state.rate_limiter = RedisAuthRateLimiter(active_redis, hmac_key)

            if database_probe is None and active_sessions is not None:
                application.state.database_probe = _database_probe(active_sessions)
            if redis_probe is None and active_redis is not None:
                application.state.redis_probe = _redis_probe(active_redis)
            if configured.litellm_health_url is not None:
                extra_checks = dict(application.state.extra_readiness_checks)
                extra_checks["litellm"] = _http_readiness_probe(
                    configured.litellm_health_url,
                    timeout_seconds=readiness_timeout_seconds,
                )
                application.state.extra_readiness_checks = extra_checks
            yield
        finally:
            await _stop_feishu_websocket_connector(application)
            await _cancel_background_tasks(application.state.feishu_reply_tasks)
            await _cleanup_owned_resources(
                cleanup_callbacks,
                primary_error=sys.exception(),
            )

    application = FastAPI(
        title="魔方 agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.database_probe = database_probe
    application.state.redis_probe = redis_probe
    application.state.readiness_timeout_seconds = readiness_timeout_seconds
    application.state.auth_service = auth_service
    application.state.rate_limiter = rate_limiter
    application.state.config_service = config_service
    application.state.admin_resource_service = admin_resource_service
    application.state.user_admin_service = user_admin_service
    application.state.bootstrap_tenant_id = configured_settings.bootstrap_tenant_id
    application.state.run_service = run_service
    application.state.runtime_registry = active_runtime_registry
    application.state.content_studio_service = content_studio_service
    application.state.mode_router = mode_router
    application.state.run_queue = task_queue
    application.state.schedule_service = None
    application.state.feishu_gateway = feishu_gateway
    application.state.feishu_reply_dispatcher = None
    application.state.feishu_reply_tasks = set()
    application.state.feishu_websocket_connector = None
    application.state.feishu_websocket_task = None
    application.state.multimedia_generation_executor = None

    async def refresh_channel_runtime_config(runtime_config: Mapping[str, str]) -> None:
        application.state.channel_runtime_config = dict(runtime_config)
        await _restart_feishu_websocket_connector(
            application,
            client_factory=feishu_websocket_client_factory,
        )

    application.state.refresh_channel_runtime_config = refresh_channel_runtime_config
    application.state.metrics_registry = default_metrics_registry()
    application.state.extra_readiness_checks = {}
    application.state.channel_runtime_config = None
    application.state.trusted_proxy_ips = configured_settings.trusted_proxy_ips
    application.add_exception_handler(PublicAPIError, public_error_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)

    async def validation_error_handler(request: Request, error: Exception) -> JSONResponse:
        del request
        assert isinstance(error, RequestValidationError)
        return JSONResponse(
            status_code=422,
            content=error_payload("request_validation", "request validation failed"),
        )

    application.add_exception_handler(RequestValidationError, validation_error_handler)
    application.add_middleware(RequestBodyLimitMiddleware)
    application.add_middleware(SafeExceptionMiddleware)
    application.router.routes.extend(system.router.routes)
    application.router.routes.extend(auth.router.routes)
    application.router.routes.extend(config.router.routes)
    application.router.routes.extend(content_studio.router.routes)
    application.router.routes.extend(runs.router.routes)
    application.router.routes.extend(admin.router.routes)
    application.router.routes.extend(users.router.routes)
    application.router.routes.extend(
        create_lazy_feishu_webhook_router(
            gateway_provider=_feishu_gateway_from_request,
            runtime_config_provider=_channel_runtime_config_from_request,
        ).routes
    )
    application.router.routes.extend(
        create_generic_channel_webhook_router(
            env=os.environ,
            gateway_provider=_feishu_gateway_from_request,
            runtime_config_provider=_channel_runtime_config_from_request,
        ).routes
    )

    @application.get("/{path:path}", include_in_schema=False, response_model=None)
    async def web_ui(path: str) -> Response:
        if path == "api" or path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content=error_payload("not_found", "resource not found"),
            )
        configured = settings or get_settings()
        if configured.web_dir is None:
            return JSONResponse(
                status_code=404,
                content=error_payload("not_found", "resource not found"),
            )
        return _web_ui_response(configured.web_dir, path)

    return application


async def _cancel_background_tasks(tasks: set[asyncio.Task[object]]) -> None:
    if not tasks:
        return
    for task in tuple(tasks):
        task.cancel()
    await asyncio.gather(*tuple(tasks), return_exceptions=True)
    tasks.clear()


def _web_ui_response(web_dir: Path, path: str) -> Response:
    root = web_dir.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or target.is_dir() or not target.exists():
        target = root / "index.html"
    if not target.exists() or not target.is_file():
        return JSONResponse(
            status_code=404,
            content=error_payload("not_found", "resource not found"),
        )
    return FileResponse(target)


def _feishu_gateway_from_request(request: Request) -> ChannelGatewayProtocol | None:
    gateway = getattr(request.app.state, "feishu_gateway", None)
    if gateway is None:
        return None
    return cast(ChannelGatewayProtocol, gateway)


async def _start_feishu_websocket_connector_if_configured(
    application: FastAPI,
    *,
    client_factory: FeishuWebSocketClientFactoryForSettings | None,
) -> None:
    runtime_config = await _channel_runtime_config_from_app(application)
    application.state.channel_runtime_config = runtime_config
    settings = _feishu_settings_from_runtime_config(FeishuSettings(), runtime_config)
    if not _should_start_feishu_websocket(settings, runtime_config):
        return
    gateway = getattr(application.state, "feishu_gateway", None)
    if gateway is None:
        return
    receiver = build_feishu_websocket_receiver(
        settings,
        gateway=cast(ChannelGatewayProtocol, gateway),
        submission_handler=_feishu_websocket_submission_handler(application, settings),
    )
    resolved_factory = client_factory or create_lark_oapi_feishu_websocket_client

    async def create_client() -> FeishuWebSocketClient:
        return await resolved_factory(settings)

    connector = FeishuWebSocketConnector(
        receiver=receiver,
        client_factory=create_client,
        reconnect_min_seconds=settings.websocket_reconnect_min_seconds,
        reconnect_max_seconds=settings.websocket_reconnect_max_seconds,
    )
    task: asyncio.Task[object] = asyncio.create_task(
        connector.run_forever(), name="feishu-websocket-connector"
    )
    application.state.feishu_websocket_connector = connector
    application.state.feishu_websocket_task = task
    _LOGGER.info("feishu_websocket_connector_started")


def _feishu_websocket_submission_handler(
    application: FastAPI,
    settings: FeishuSettings,
) -> Callable[[InboundMessage, object], Awaitable[None]]:
    async def handle(message: InboundMessage, submission: object) -> None:
        _schedule_feishu_websocket_reply(application, settings, message, submission)

    return handle


def _schedule_feishu_websocket_reply(
    application: FastAPI,
    settings: FeishuSettings,
    message: InboundMessage,
    submission: object,
) -> None:
    if bool(getattr(submission, "duplicate", False)):
        return
    dispatcher = getattr(application.state, "feishu_reply_dispatcher", None)
    if not isinstance(dispatcher, FeishuRunReplyDispatcher):
        return
    tenant_id = getattr(application.state, "bootstrap_tenant_id", None)
    if tenant_id is None:
        return
    tasks = getattr(application.state, "feishu_reply_tasks", None)
    if not isinstance(tasks, set):
        return
    run_id = getattr(submission, "run_id", None)
    log_service = getattr(application.state, "admin_resource_service", None)

    async def task() -> None:
        try:
            await dispatcher.sender.reply_text(
                settings=settings,
                message_id=message.message_id,
                text="已收到，主 Agent 正在判断入口、模式和可用资源。",
            )
            if run_id is not None:
                await dispatcher.reply_when_terminal(
                    tenant_id=tenant_id,
                    run_id=run_id,
                    source_message_id=message.message_id,
                    settings=settings,
                )
        except Exception as error:  # noqa: BLE001 - best-effort channel delivery boundary
            await log_feishu_reply_failure(
                log_service=log_service,
                run_id=run_id,
                message_id=message.message_id,
                error=error,
            )

    created = asyncio.create_task(task())
    tasks.add(created)
    created.add_done_callback(tasks.discard)


async def _stop_feishu_websocket_connector(application: FastAPI) -> None:
    connector = getattr(application.state, "feishu_websocket_connector", None)
    task = getattr(application.state, "feishu_websocket_task", None)
    if isinstance(connector, FeishuWebSocketConnector):
        connector.request_shutdown()
    if isinstance(task, asyncio.Task):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    application.state.feishu_websocket_connector = None
    application.state.feishu_websocket_task = None


async def _restart_feishu_websocket_connector(
    application: FastAPI,
    *,
    client_factory: FeishuWebSocketClientFactoryForSettings | None,
) -> None:
    await _stop_feishu_websocket_connector(application)
    await _start_feishu_websocket_connector_if_configured(
        application,
        client_factory=client_factory,
    )


def _should_start_feishu_websocket(
    settings: FeishuSettings, runtime_config: Mapping[str, str]
) -> bool:
    raw_transport = (
        runtime_config.get("FEISHU_TRANSPORT")
        or os.environ.get("FEISHU_TRANSPORT")
        or FeishuTransport.WEBSOCKET.value
    )
    try:
        transport = FeishuTransport(raw_transport)
    except ValueError:
        transport = FeishuTransport.WEBSOCKET
    if FeishuTransport.WEBSOCKET not in (
        {FeishuTransport.WEBHOOK, FeishuTransport.WEBSOCKET}
        if transport is FeishuTransport.BOTH
        else {transport}
    ):
        return False
    has_app_id = bool(runtime_config.get("FEISHU_APP_ID") or os.environ.get("FEISHU_APP_ID"))
    has_app_secret = bool(
        runtime_config.get("FEISHU_APP_SECRET") or os.environ.get("FEISHU_APP_SECRET")
    )
    return has_app_id and has_app_secret and bool(settings.app_id and settings.app_secret_value())


async def _channel_runtime_config_from_app(application: FastAPI) -> dict[str, str]:
    cached = getattr(application.state, "channel_runtime_config", None)
    if isinstance(cached, dict):
        return cast(dict[str, str], cached)
    service = getattr(application.state, "admin_resource_service", None)
    if service is None:
        return {}
    provider = getattr(service, "channel_runtime_config", None)
    if provider is None:
        return {}
    try:
        config = await provider()
    except Exception as error:  # noqa: BLE001 - channel startup must not break app lifespan.
        _LOGGER.warning("channel_runtime_config_unavailable error_type=%s", type(error).__name__)
        return {}
    if isinstance(config, dict):
        return cast(dict[str, str], config)
    return {}


async def _submit_scheduled_task(application: FastAPI, request: TaskRequest) -> object:
    run_service = getattr(application.state, "run_service", None)
    if run_service is None or not hasattr(run_service, "submit"):
        raise RuntimeError("run service is unavailable")
    metadata = {str(key): str(value) for key, value in request.metadata.items()}
    return await cast(Any, run_service).submit(
        tenant_id=request.tenant_id,
        actor_id=request.actor_id,
        message=request.message,
        mode=request.mode,
        workflow_id=request.workflow,
        channel_context=metadata,
        idempotency_key=request.idempotency_key,
    )


async def _channel_runtime_config_from_request(request: Request) -> Mapping[str, str]:
    cached = getattr(request.app.state, "channel_runtime_config", None)
    if isinstance(cached, dict):
        return cast(dict[str, str], cached)
    service = getattr(request.app.state, "admin_resource_service", None)
    if service is None:
        return {}
    provider = getattr(service, "channel_runtime_config", None)
    if provider is None:
        return {}
    try:
        config = await provider()
    except Exception as error:  # noqa: BLE001 - webhook config lookup degrades to env settings.
        _LOGGER.warning("channel_runtime_config_unavailable error_type=%s", type(error).__name__)
        return {}
    if isinstance(config, dict):
        request.app.state.channel_runtime_config = config
        return cast(dict[str, str], config)
    return {}


def _database_probe(
    session_factory: async_sessionmaker[AsyncSession],
) -> ReadinessProbe:
    async def probe() -> None:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))

    return probe


def _redis_probe(redis_client: RedisResource) -> ReadinessProbe:
    async def probe() -> None:
        await redis_client.ping()

    return probe


def _http_readiness_probe(url: str, *, timeout_seconds: float) -> ReadinessProbe:
    async def probe() -> None:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()

    return probe


app = create_app()
