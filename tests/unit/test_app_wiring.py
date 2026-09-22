import asyncio
import base64
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self
from uuid import UUID, uuid4

import pytest
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient

import agent_hub.app as app_module
from agent_hub.api.routers.admin import (
    InMemoryAdminResourceService,
    MainAgentConfigResponse,
    MainAgentModelConfig,
    ModelDeploymentResponse,
)
from agent_hub.app import (
    _apply_asset_visual_review_policy,
    _asset_visual_review_prompt,
    _ConfigBackedAssetVisualReviewer,
    _ConfigBackedMultimediaGenerationExecutor,
    _infer_main_agent_context_window_tokens,
    _MainAgentContextWindowGetter,
    _MainAgentModeRouter,
    _parse_asset_visual_review_payload,
    _visual_review_provider_priority,
    _web_ui_response,
    create_app,
)
from agent_hub.auth.models import AuthenticatedPrincipal, InvalidCredentials, Role
from agent_hub.channels.feishu.media import FeishuMediaService
from agent_hub.channels.feishu.media_factory import build_feishu_media_service_factory
from agent_hub.channels.feishu.settings import FeishuSettings
from agent_hub.channels.feishu.websocket import FeishuWebSocketClient
from agent_hub.domain.runs import TaskMode
from agent_hub.models.capacity import CapacityLease, CapacityUnavailable
from agent_hub.models.gateway import CapacityController, ModelGatewayError
from agent_hub.models.litellm_client import ModelTransportError
from agent_hub.models.registry import NoCapableDeployment
from agent_hub.models.types import Deployment, ModelRequest, ModelResponse, TokenUsage
from agent_hub.multimodal.audio_providers import (
    GeneratedAudioArtifact,
    TextToAudioProviderRouter,
)
from agent_hub.multimodal.generation import MultimediaGenerationKind
from agent_hub.multimodal.minimax import MiniMaxGeneratedVideo
from agent_hub.multimodal.video_providers import TextToVideoProviderRouter
from agent_hub.routing.types import RiskLevel
from agent_hub.settings import Settings

TENANT_ID = UUID("00000000-0000-4000-8000-000000000001")


class StubAuthService:
    def authenticate_token(self, token: str) -> AuthenticatedPrincipal:
        if token != "valid-token":
            raise InvalidCredentials("bad token")
        return AuthenticatedPrincipal(uuid4(), TENANT_ID, Role.SUPER_ADMIN)


class StubRateLimiter:
    pass


class StubConfigService:
    async def get_current(self, tenant_id: UUID) -> None:
        assert tenant_id == TENANT_ID


class FakeSession:
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class FakeDatabase:
    def __init__(self) -> None:
        self.session_factory = FakeSession

    async def dispose(self) -> None:
        return None


class FakeRedis:
    async def aclose(self) -> None:
        return None

    async def ping(self, **kwargs: object) -> bool:
        del kwargs
        return True


class BlockingFeishuClient:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def events(self) -> Any:
        self.started.set()
        try:
            await asyncio.Event().wait()
            yield {}
        finally:
            self.cancelled.set()


def valid_settings(
    attachment_store_dir: Path | None = None,
    generated_artifact_dir: Path | None = None,
) -> Settings:
    key = base64.urlsafe_b64encode(b"x" * 32).decode("ascii").rstrip("=")
    values: dict[str, object] = {"jwt_signing_key": "base64url:" + key}
    if attachment_store_dir is not None:
        values["attachment_store_dir"] = attachment_store_dir
    if generated_artifact_dir is not None:
        values["generated_artifact_dir"] = generated_artifact_dir
    return Settings.model_validate(values)


def test_web_ui_rejects_sibling_path_with_shared_prefix(tmp_path: Path) -> None:
    web_root = tmp_path / "web"
    web_root.mkdir()
    (web_root / "index.html").write_text("index", encoding="utf-8")
    sibling = tmp_path / "web-evil"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("secret", encoding="utf-8")

    response = _web_ui_response(web_root, "../web-evil/secret.txt")

    assert isinstance(response, FileResponse)
    assert Path(response.path) == web_root / "index.html"


def test_create_app_mounts_feishu_webhook_on_main_api() -> None:
    application = create_app(
        auth_service=object(),
        rate_limiter=object(),
        config_service=object(),
        run_service=object(),
    )

    paths = {getattr(route, "path", "") for route in application.routes}

    assert "/channels/feishu/events" in paths
    assert application.state.channel_runtime_config is None


def test_create_app_wires_production_feishu_media_service_factory(tmp_path: Path) -> None:
    application = create_app(
        settings=valid_settings(tmp_path),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        auth_service=StubAuthService(),
        rate_limiter=StubRateLimiter(),
        config_service=StubConfigService(),
        admin_resource_service=InMemoryAdminResourceService(),
        user_admin_service=object(),
        run_service=object(),
    )

    with TestClient(application):
        factory = getattr(application.state, "feishu_media_service_factory", None)
        assert callable(factory)
        service = factory(
            FeishuSettings.model_validate(
                {
                    "app_id": "cli_runtime",
                    "app_secret": "secret",
                    "verification_token": "token",
                    "encrypt_key": "encrypt-key",
                    "transport": "webhook",
                }
            )
        )

    assert isinstance(service, FeishuMediaService)


def test_create_app_wires_admin_generated_artifact_store(tmp_path: Path) -> None:
    generated_artifact_dir = tmp_path / "generated"
    application = create_app(
        settings=valid_settings(tmp_path / "attachments", generated_artifact_dir),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        auth_service=StubAuthService(),
        rate_limiter=StubRateLimiter(),
        config_service=StubConfigService(),
        user_admin_service=object(),
        run_service=object(),
    )

    with TestClient(application):
        service = application.state.admin_resource_service

    assert service._generated_file_store._root == generated_artifact_dir.resolve()


def test_create_app_wires_runtime_generated_artifact_store(tmp_path: Path) -> None:
    generated_artifact_dir = tmp_path / "generated"
    application = create_app(
        settings=valid_settings(tmp_path / "attachments", generated_artifact_dir),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        auth_service=StubAuthService(),
        rate_limiter=StubRateLimiter(),
        config_service=StubConfigService(),
        admin_resource_service=InMemoryAdminResourceService(),
        user_admin_service=object(),
    )

    with TestClient(application):
        runtime = application.state.run_service._runtime_registry.get(TaskMode.DISPATCH)

    store = runtime._capability_gateway._generated_file_store
    assert store is not None
    assert store._root == generated_artifact_dir.resolve()



def test_create_app_starts_feishu_websocket_when_runtime_config_enables_it(
    tmp_path: Path,
) -> None:
    client = BlockingFeishuClient()
    created_settings: list[FeishuSettings] = []

    async def client_factory(settings: FeishuSettings) -> FeishuWebSocketClient:
        created_settings.append(settings)
        return client

    admin_service = InMemoryAdminResourceService()
    admin_service.channel_config["feishu"] = {
        "FEISHU_APP_ID": "cli_runtime",
        "FEISHU_APP_SECRET": "secret",
        "FEISHU_TRANSPORT": "websocket",
    }
    application = create_app(
        settings=valid_settings(tmp_path),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        auth_service=StubAuthService(),
        rate_limiter=StubRateLimiter(),
        config_service=StubConfigService(),
        admin_resource_service=admin_service,
        user_admin_service=object(),
        run_service=object(),
        feishu_websocket_client_factory=client_factory,
    )

    with TestClient(application):
        assert client.started.wait(timeout=1)
        connector = getattr(application.state, "feishu_websocket_connector", None)
        task = getattr(application.state, "feishu_websocket_task", None)
        assert connector is not None
        assert task is not None
        assert created_settings[0].app_id == "cli_runtime"

    assert task.done()
    assert client.cancelled.wait(timeout=1)

def test_channel_status_exposes_feishu_websocket_runtime_diagnostics(
    tmp_path: Path,
) -> None:
    client = BlockingFeishuClient()

    async def client_factory(settings: FeishuSettings) -> FeishuWebSocketClient:
        return client

    admin_service = InMemoryAdminResourceService()
    admin_service.channel_config["feishu"] = {
        "FEISHU_APP_ID": "cli_runtime",
        "FEISHU_APP_SECRET": "secret",
        "FEISHU_TRANSPORT": "websocket",
    }
    application = create_app(
        settings=valid_settings(tmp_path),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        auth_service=StubAuthService(),
        rate_limiter=StubRateLimiter(),
        config_service=StubConfigService(),
        admin_resource_service=admin_service,
        user_admin_service=object(),
        run_service=object(),
        feishu_websocket_client_factory=client_factory,
    )

    with TestClient(application) as api:
        assert client.started.wait(timeout=1)
        response = api.get(
            "/api/v1/admin/channels",
            headers={"Authorization": "Bearer valid-token"},
        )

    assert response.status_code == 200
    feishu = next(item for item in response.json() if item["id"] == "feishu")
    assert feishu["runtime"] == {
        "status": "running",
        "ready": True,
        "connection_attempts": 1,
        "reconnects": 0,
        "received_events": 0,
        "submitted_messages": 0,
        "ignored_events": 0,
        "failures": 0,
        "last_error_type": None,
        "last_error_message": None,
    }

def test_feishu_websocket_restarts_when_channel_config_changes(
    tmp_path: Path,
) -> None:
    first_client = BlockingFeishuClient()
    second_client = BlockingFeishuClient()
    clients = [first_client, second_client]
    created_settings: list[FeishuSettings] = []

    async def client_factory(settings: FeishuSettings) -> FeishuWebSocketClient:
        created_settings.append(settings)
        return clients.pop(0)

    admin_service = InMemoryAdminResourceService()
    application = create_app(
        settings=valid_settings(tmp_path),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        auth_service=StubAuthService(),
        rate_limiter=StubRateLimiter(),
        config_service=StubConfigService(),
        admin_resource_service=admin_service,
        user_admin_service=object(),
        run_service=object(),
        feishu_websocket_client_factory=client_factory,
    )

    with TestClient(application) as api:
        assert getattr(application.state, "feishu_websocket_connector", None) is None

        saved = api.post(
            "/api/v1/admin/channels/feishu/config",
            headers={"Authorization": "Bearer valid-token"},
            json={
                "values": {
                    "FEISHU_TRANSPORT": "websocket",
                    "FEISHU_APP_ID": "cli_runtime",
                    "FEISHU_APP_SECRET": "secret",
                }
            },
        )

        assert saved.status_code == 200
        assert first_client.started.wait(timeout=1)
        assert created_settings[-1].app_id == "cli_runtime"
        assert getattr(application.state, "feishu_websocket_connector", None) is not None

        updated = api.post(
            "/api/v1/admin/channels/feishu/config",
            headers={"Authorization": "Bearer valid-token"},
            json={
                "values": {
                    "FEISHU_APP_ID": "cli_runtime_2",
                    "FEISHU_APP_SECRET": "secret-2",
                }
            },
        )

        assert updated.status_code == 200
        assert first_client.cancelled.wait(timeout=1)
        assert second_client.started.wait(timeout=1)
        assert created_settings[-1].app_id == "cli_runtime_2"

        cleared = api.delete(
            "/api/v1/admin/channels/feishu/config",
            headers={"Authorization": "Bearer valid-token"},
        )

        assert cleared.status_code == 200
        assert second_client.cancelled.wait(timeout=1)
        assert getattr(application.state, "feishu_websocket_connector", None) is None


def test_create_app_wires_production_multimedia_generation_executor(tmp_path: Path) -> None:
    application = create_app(
        settings=valid_settings(tmp_path),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        auth_service=StubAuthService(),
        rate_limiter=StubRateLimiter(),
        config_service=StubConfigService(),
        admin_resource_service=InMemoryAdminResourceService(),
        user_admin_service=object(),
        run_service=object(),
    )

    with TestClient(application):
        executor = getattr(application.state, "multimedia_generation_executor", None)

    assert isinstance(executor, _ConfigBackedMultimediaGenerationExecutor)


def test_feishu_media_factory_uses_memory_store_in_development(tmp_path: Path) -> None:
    factory = build_feishu_media_service_factory(
        config_service=StubConfigService(),
        secret_service=FakeSecretService(),
        redis_client=FakeRedis(),
        tenant_id=TENANT_ID,
        attachment_store_dir=tmp_path,
        environment="development",
    )

    assert factory.object_store.store_id == "memory-image-store"


@pytest.mark.asyncio
async def test_multimedia_executor_rejects_video_without_capacity_or_secret_lookup() -> None:
    capacity_calls = 0

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="deepseek",
                api_base="https://api.deepseek.com/v1",
                api_protocol="openai_compatible",
                upstream_model="deepseek-v4-flash",
                logical_model="video_primary",
                capabilities=["text"],
                credential_ref="secret://video-primary",
                quota_scope="deepseek-account",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(
        _deployments: tuple[Deployment, ...],
    ) -> CapacityController:
        nonlocal capacity_calls
        capacity_calls += 1
        return ImmediateCapacity()

    executor = _ConfigBackedMultimediaGenerationExecutor(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        capacity_factory=capacity_factory,
    )

    with pytest.raises(NoCapableDeployment, match="video_generation"):
        await executor.generate(
            kind=MultimediaGenerationKind.VIDEO,
            logical_model="video_primary",
            prompt="make a short launch video",
        )

    assert capacity_calls == 0


@pytest.mark.asyncio
async def test_multimedia_executor_rejects_unknown_video_model_even_if_declared() -> None:
    capacity_calls = 0

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="minimax",
                api_base="https://api.minimax.chat/v1",
                api_protocol="openai_compatible",
                upstream_model="MiniMax-M3",
                logical_model="video_primary",
                capabilities=["text", "video_generation"],
                credential_ref="secret://video-primary",
                quota_scope="minimax-account",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(
        _deployments: tuple[Deployment, ...],
    ) -> CapacityController:
        nonlocal capacity_calls
        capacity_calls += 1
        return ImmediateCapacity()

    executor = _ConfigBackedMultimediaGenerationExecutor(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        capacity_factory=capacity_factory,
    )

    with pytest.raises(NoCapableDeployment, match="supported video generation"):
        await executor.generate(
            kind=MultimediaGenerationKind.VIDEO,
            logical_model="video_primary",
            prompt="make a short launch video",
        )

    assert capacity_calls == 0


@pytest.mark.asyncio
async def test_multimedia_executor_limits_minimax_video_to_three_daily_requests(tmp_path: Path) -> None:
    transport = FakeTransport()
    video_provider = FakeTextToVideoProvider(tmp_path / "limited-output.mp4")

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="minimax",
                api_base="https://api.minimax.chat/v1",
                api_protocol="openai_compatible",
                upstream_model="MiniMax-Hailuo-02",
                logical_model="video_primary",
                capabilities=["text", "video_generation"],
                credential_ref="secret://main-agent",
                quota_scope="minimax-account",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(_deployments: tuple[Deployment, ...]) -> CapacityController:
        return ImmediateCapacity()

    executor = _ConfigBackedMultimediaGenerationExecutor(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
        media_store_dir=tmp_path / "media",
        video_provider_router=TextToVideoProviderRouter((("minimax", video_provider),)),
    )

    for index in range(3):
        await executor.generate(
            kind=MultimediaGenerationKind.VIDEO,
            logical_model="video_primary",
            prompt=f"make minimax video {index}",
        )

    with pytest.raises(RuntimeError, match="daily multimedia generation limit"):
        await executor.generate(
            kind=MultimediaGenerationKind.VIDEO,
            logical_model="video_primary",
            prompt="make minimax video 4",
        )

    assert len(video_provider.calls) == 3
    assert transport.requests == []


@pytest.mark.asyncio
async def test_multimedia_executor_uses_minimax_video_client_for_hailuo_files(tmp_path: Path) -> None:
    transport = FakeTransport()
    video_provider = FakeTextToVideoProvider(tmp_path / "provider-output.mp4")

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="minimax",
                api_base="https://api.minimax.io/v1",
                api_protocol="openai_compatible",
                upstream_model="MiniMax-Hailuo-02",
                logical_model="video_primary",
                capabilities=["video_generation"],
                credential_ref="secret://main-agent",
                quota_scope="minimax-account",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(_deployments: tuple[Deployment, ...]) -> CapacityController:
        return ImmediateCapacity()

    executor = _ConfigBackedMultimediaGenerationExecutor(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
        media_store_dir=tmp_path / "media",
        video_provider_router=TextToVideoProviderRouter((("minimax", video_provider),)),
    )

    result = await executor.generate(
        kind=MultimediaGenerationKind.VIDEO,
        logical_model="video_primary",
        prompt="make a real minimax video",
    )

    assert result.deployment_id
    assert result.text is not None
    assert result.text.startswith("file://")
    assert (tmp_path / "media" / str(TENANT_ID) / "provider-output.mp4").read_bytes() == b"video"
    assert video_provider.calls == [
        {
            "api_key": "sk-live",
            "api_base": "https://api.minimax.io/v1",
            "model": "MiniMax-Hailuo-02",
            "prompt": "make a real minimax video",
            "output_dir": tmp_path / "media" / str(TENANT_ID),
            "duration": 6,
            "resolution": "768P",
        }
    ]
    assert transport.requests == []


@pytest.mark.asyncio
async def test_multimedia_executor_uses_minimax_audio_client_for_tts_files(tmp_path: Path) -> None:
    transport = FakeTransport()
    audio_provider = FakeTextToAudioProvider(tmp_path / "provider-output.mp3")

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="minimax",
                api_base="https://api.minimax.chat/v1",
                api_protocol="openai_compatible",
                upstream_model="speech-2.8-turbo",
                logical_model="audio_primary",
                capabilities=["audio_generation"],
                credential_ref="secret://main-agent",
                quota_scope="minimax-audio-account",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(_deployments: tuple[Deployment, ...]) -> CapacityController:
        return ImmediateCapacity()

    executor = _ConfigBackedMultimediaGenerationExecutor(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
        media_store_dir=tmp_path / "media",
        audio_provider_router=TextToAudioProviderRouter((("minimax", audio_provider),)),
    )

    assert await executor.default_logical_model("audio") == "audio_primary"

    result = await executor.generate(
        kind=MultimediaGenerationKind.AUDIO,
        logical_model="audio_primary",
        prompt="这是一段科普旁白。",
    )

    assert result.deployment_id
    assert result.text is not None
    assert result.text.startswith("file://")
    assert result.file_path == tmp_path / "media" / str(TENANT_ID) / "provider-output.mp3"
    assert result.file_path.read_bytes() == b"audio"
    assert result.mime_type == "audio/mpeg"
    assert audio_provider.calls == [
        {
            "api_key": "sk-live",
            "api_base": "https://api.minimax.chat/v1",
            "model": "speech-2.8-turbo",
            "prompt": "这是一段科普旁白。",
            "output_dir": tmp_path / "media" / str(TENANT_ID),
        }
    ]
    assert transport.requests == []


@pytest.mark.asyncio
async def test_multimedia_executor_uses_dashscope_client_for_kling_image_files(tmp_path: Path) -> None:
    transport = FakeTransport()
    dashscope_provider = FakeDashScopeMultimediaClient(tmp_path / "dashscope-image.png")

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="qwen-token-plan",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_protocol="openai_compatible",
                upstream_model="kling/kling-v3-omni-image-generation",
                logical_model="image_primary",
                capabilities=["image_generation", "video_generation"],
                credential_ref="secret://main-agent",
                quota_scope="dashscope-account",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(_deployments: tuple[Deployment, ...]) -> CapacityController:
        return ImmediateCapacity()

    executor = _ConfigBackedMultimediaGenerationExecutor(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
        media_store_dir=tmp_path / "media",
        dashscope_multimedia_client=dashscope_provider,  # type: ignore[arg-type]
    )

    result = await executor.generate(
        kind=MultimediaGenerationKind.IMAGE,
        logical_model="image_primary",
        prompt="生成一张蓝色方块测试图",
    )

    assert result.deployment_id
    assert result.text is not None
    assert result.text.startswith("file://")
    assert (tmp_path / "media" / str(TENANT_ID) / "dashscope-image.png").read_bytes() == b"image"
    assert dashscope_provider.image_calls == [
        {
            "api_key": "sk-live",
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "kling/kling-v3-omni-image-generation",
            "prompt": "生成一张蓝色方块测试图",
            "output_dir": tmp_path / "media" / str(TENANT_ID),
        }
    ]
    assert dashscope_provider.video_calls == []
    assert transport.requests == []


@pytest.mark.asyncio
async def test_multimedia_executor_uses_dashscope_client_for_kling_video_files(tmp_path: Path) -> None:
    transport = FakeTransport()
    dashscope_provider = FakeDashScopeMultimediaClient(tmp_path / "dashscope-video.mp4")

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="qwen-token-plan",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_protocol="openai_compatible",
                upstream_model="kling/kling-v3-omni-video-generation",
                logical_model="video_primary",
                capabilities=["video_generation"],
                credential_ref="secret://main-agent",
                quota_scope="dashscope-account",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(_deployments: tuple[Deployment, ...]) -> CapacityController:
        return ImmediateCapacity()

    executor = _ConfigBackedMultimediaGenerationExecutor(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
        media_store_dir=tmp_path / "media",
        dashscope_multimedia_client=dashscope_provider,  # type: ignore[arg-type]
    )

    result = await executor.generate(
        kind=MultimediaGenerationKind.VIDEO,
        logical_model="video_primary",
        prompt="生成一段蓝色方块测试视频",
    )

    assert result.deployment_id
    assert result.text is not None
    assert result.text.startswith("file://")
    assert (tmp_path / "media" / str(TENANT_ID) / "dashscope-video.mp4").read_bytes() == b"video"
    assert dashscope_provider.image_calls == []
    assert dashscope_provider.video_calls == [
        {
            "api_key": "sk-live",
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "kling/kling-v3-omni-video-generation",
            "prompt": "生成一段蓝色方块测试视频",
            "output_dir": tmp_path / "media" / str(TENANT_ID),
            "duration": 5,
            "resolution": "std",
        }
    ]
    assert transport.requests == []


@pytest.mark.asyncio
async def test_multimedia_executor_does_not_send_other_video_models_to_minimax(tmp_path: Path) -> None:
    transport = FakeTransport()
    video_provider = FakeTextToVideoProvider(tmp_path / "provider-output.mp4")

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="runway",
                api_base="https://api.runwayml.com/v1",
                api_protocol="openai_compatible",
                upstream_model="gen4-turbo",
                logical_model="video_primary",
                capabilities=["video_generation"],
                credential_ref="secret://main-agent",
                quota_scope="runway-account",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(_deployments: tuple[Deployment, ...]) -> CapacityController:
        return ImmediateCapacity()

    executor = _ConfigBackedMultimediaGenerationExecutor(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
        media_store_dir=tmp_path / "media",
        video_provider_router=TextToVideoProviderRouter((("minimax", video_provider),)),
    )

    with pytest.raises(NoCapableDeployment, match="supported video generation"):
        await executor.generate(
            kind=MultimediaGenerationKind.VIDEO,
            logical_model="video_primary",
            prompt="make a runway video",
        )

    assert video_provider.calls == []
    assert transport.requests == []


class FakeSecretService:
    async def resolve(self, tenant_id: UUID, reference: object) -> str:
        assert tenant_id == TENANT_ID
        assert reference == "secret://main-agent"
        return "sk-live"

    async def fingerprint(self, tenant_id: UUID, reference: object) -> str:
        assert tenant_id == TENANT_ID
        assert reference == "secret://main-agent"
        return "a" * 64


class ImmediateCapacity:
    async def initialize(self) -> None:
        return None

    def validate_configuration(self, deployments: Sequence[Deployment]) -> None:
        assert len(deployments) == 1

    async def acquire(
        self,
        candidates: Sequence[Deployment],
        wait_timeout: float,
        *,
        estimated_tokens: int,
    ) -> CapacityLease:
        assert candidates
        assert wait_timeout > 0
        assert estimated_tokens > 0
        return CapacityLease(
            id=str(uuid4()),
            deployment_id=candidates[0].id,
            quota_scope_id=candidates[0].quota_scope_id,
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            renew_after_seconds=30,
        )

    async def renew(self, lease: CapacityLease) -> CapacityLease | None:
        return lease

    async def release(self, lease: CapacityLease) -> bool:
        del lease
        return True

    async def record_outcome(
        self,
        quota_scope_id: str,
        *,
        status_code: int | None,
        latency_seconds: float,
        succeeded: bool,
    ) -> None:
        del quota_scope_id, status_code, latency_seconds, succeeded


class UnavailableCapacity(ImmediateCapacity):
    async def acquire(
        self,
        candidates: Sequence[Deployment],
        wait_timeout: float,
        *,
        estimated_tokens: int,
    ) -> CapacityLease:
        del candidates, wait_timeout, estimated_tokens
        raise CapacityUnavailable("model capacity unavailable")


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(
        self,
        deployment: Deployment,
        request: ModelRequest,
        api_key: str,
    ) -> ModelResponse:
        del deployment, api_key
        self.requests.append(request)
        return ModelResponse(
            text=(
                '{"mode":"dispatch","confidence":0.92,"reason":"task can be split",'
                '"roles":["writer","reviewer"],"estimated_seconds":30,'
                '"estimated_cost_usd":"0.01","risk":"low"}'
            ),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=6, total_tokens=16),
        )


class FakeVisionReviewTransport:
    def __init__(self, response_text: str | None = None) -> None:
        self.requests: list[ModelRequest] = []
        self.deployments: list[Deployment] = []
        self.response_text = response_text or (
            '{"passed":true,"summary":"角色锁定资产符合要求",'
            '"issues":[],"confidence":0.93}'
        )

    async def complete(
        self,
        deployment: Deployment,
        request: ModelRequest,
        api_key: str,
    ) -> ModelResponse:
        assert api_key == "sk-live"
        self.deployments.append(deployment)
        self.requests.append(request)
        return ModelResponse(
            text=self.response_text,
            usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        )


class SchemaRejectingVisionReviewTransport(FakeVisionReviewTransport):
    async def complete(
        self,
        deployment: Deployment,
        request: ModelRequest,
        api_key: str,
    ) -> ModelResponse:
        if request.response_schema is not None:
            self.deployments.append(deployment)
            self.requests.append(request)
            raise ModelTransportError(
                "model transport failed",
                status_code=400,
                logical_models=(deployment.logical_model,),
                deployments=(deployment.id,),
            )
        return await super().complete(deployment, request, api_key)


def test_parse_asset_visual_review_payload_accepts_common_model_type_drift() -> None:
    payload = _parse_asset_visual_review_payload(
        '{"passed":"false","summary":"不是镜头资产","issues":"缺少景别和机位",'
        '"confidence":"0.72"}'
    )

    assert payload == {
        "passed": False,
        "summary": "不是镜头资产",
        "issues": ["缺少景别和机位"],
        "confidence": 0.72,
    }


def test_parse_asset_visual_review_payload_accepts_plain_text_rejection() -> None:
    payload = _parse_asset_visual_review_payload(
        "结论：不通过。\n问题：画面像电影剧照，缺少角色三视图，且出现具体背景。"
    )

    assert payload["passed"] is False
    summary = payload["summary"]
    assert isinstance(summary, str)
    assert "不通过" in summary
    assert payload["confidence"] == 0.5
    assert payload["issues"] == ["问题：画面像电影剧照，缺少角色三视图，且出现具体背景。"]


def test_visual_review_provider_priority_prefers_deepseek_then_qwen_then_minimax() -> None:
    def deployment(logical_model: str, provider_model: str) -> Deployment:
        return Deployment(
            id=logical_model,
            logical_model=logical_model,
            provider_model=provider_model,
            request_model=provider_model,
            api_base="https://example.com/v1",
            secret_ref="secret://test",
            quota_scope_id=logical_model,
        )

    assert _visual_review_provider_priority(
        deployment("deepseek-mutil", "deepseek/deepseek-v4-flash-vision-exp")
    ) > _visual_review_provider_priority(deployment("qwen-vl", "qwen/qwen-vl-max"))
    assert _visual_review_provider_priority(
        deployment("qwen-vl", "qwen/qwen-vl-max")
    ) > _visual_review_provider_priority(deployment("minimax", "minimax/MiniMax-M3"))


class FakeTextToVideoProvider:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.deployment_id = "video-deployment"
        self.calls: list[dict[str, object]] = []

    async def generate_text_to_video(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        prompt: str,
        output_dir: Path,
        duration: int,
        resolution: str,
    ) -> MiniMaxGeneratedVideo:
        self.calls.append(
            {
                "api_key": api_key,
                "api_base": api_base,
                "model": model,
                "prompt": prompt,
                "output_dir": output_dir,
                "duration": duration,
                "resolution": resolution,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stored = output_dir / self.output_path.name
        stored.write_bytes(b"video")
        return MiniMaxGeneratedVideo(
            path=stored,
            uri=stored.as_uri(),
            provider="minimax",
            model=model,
            task_id="task-1",
            file_id="file-1",
            mime_type="video/mp4",
        )


class FakeTextToAudioProvider:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.calls: list[dict[str, object]] = []

    async def generate_text_to_audio(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        prompt: str,
        output_dir: Path,
    ) -> GeneratedAudioArtifact:
        self.calls.append(
            {
                "api_key": api_key,
                "api_base": api_base,
                "model": model,
                "prompt": prompt,
                "output_dir": output_dir,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stored = output_dir / self.output_path.name
        stored.write_bytes(b"audio")
        return GeneratedAudioArtifact(
            path=stored,
            uri=stored.as_uri(),
            provider="minimax",
            model=model,
            task_id="audio-task-1",
            file_id=None,
            mime_type="audio/mpeg",
        )


class FakeDashScopeMultimediaClient:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.image_calls: list[dict[str, object]] = []
        self.video_calls: list[dict[str, object]] = []

    async def generate_text_to_image(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        prompt: str,
        output_dir: Path,
        aspect_ratio: str = "1:1",
        resolution: str = "1k",
    ) -> MiniMaxGeneratedVideo:
        del aspect_ratio, resolution
        self.image_calls.append(
            {
                "api_key": api_key,
                "api_base": api_base,
                "model": model,
                "prompt": prompt,
                "output_dir": output_dir,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stored = output_dir / self.output_path.name
        stored.write_bytes(b"image")
        return MiniMaxGeneratedVideo(
            path=stored,
            uri=stored.as_uri(),
            provider="dashscope",
            model=model,
            task_id="dashscope-image-1",
            file_id=None,
            mime_type="image/png",
            kind="image",
        )

    async def generate_text_to_video(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        prompt: str,
        output_dir: Path,
        duration: int = 5,
        resolution: str = "std",
    ) -> MiniMaxGeneratedVideo:
        self.video_calls.append(
            {
                "api_key": api_key,
                "api_base": api_base,
                "model": model,
                "prompt": prompt,
                "output_dir": output_dir,
                "duration": duration,
                "resolution": resolution,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stored = output_dir / self.output_path.name
        stored.write_bytes(b"video")
        return MiniMaxGeneratedVideo(
            path=stored,
            uri=stored.as_uri(),
            provider="dashscope",
            model=model,
            task_id="dashscope-video-1",
            file_id=None,
            mime_type="video/mp4",
            kind="video",
        )


@pytest.mark.asyncio
async def test_asset_visual_reviewer_uses_image_url_with_non_messages_vision_model() -> None:
    transport = FakeVisionReviewTransport()

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="anthropic",
                api_base="https://relay.example.com/v1/messages",
                api_protocol="openai_compatible",
                upstream_model="claude-vision",
                logical_model="vision_primary",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-messages",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
            ModelDeploymentResponse(
                provider="openai",
                api_base="https://api.openai.com/v1",
                api_protocol="openai_compatible",
                upstream_model="gpt-4.1",
                logical_model="vision_primary",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-openai",
                max_concurrency=2,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=2,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(deployments: tuple[Deployment, ...]) -> CapacityController:
        assert len(deployments) == 1
        assert deployments[0].quota_scope_id == "vision-openai"
        return ImmediateCapacity()

    reviewer = _ConfigBackedAssetVisualReviewer(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
    )

    review = await reviewer.review_image_asset(
        tenant_id=TENANT_ID,
        label="角色锁定资产：男主",
        prompt="生成男主 Character Model Sheet，不要电影剧照。",
        filename="character.png",
        mime_type="image/png",
        data=b"fake-png",
    )

    assert review.passed is True
    assert review.summary == "角色锁定资产符合要求"
    assert review.confidence == 0.93
    assert [deployment.quota_scope_id for deployment in transport.deployments] == [
        "vision-openai"
    ]
    assert len(transport.requests) == 1
    assert transport.requests[0].required_capabilities == frozenset(
        {"vision", "structured_output"}
    )
    assert transport.requests[0].response_schema is not None
    assert transport.requests[0].response_schema.name == "AssetVisualReview"
    assert transport.requests[0].response_schema.schema["required"] == (
        "passed",
        "summary",
        "issues",
        "confidence",
    )
    review_prompt = "\n".join(str(message.content) for message in transport.requests[0].messages)
    assert "干净、低噪声" in review_prompt
    assert "无关背景、装饰、小物件" in review_prompt
    assert "纯白/浅灰/透明感纯色背景" in review_prompt
    assert "背景会污染后续人物锁定" in review_prompt
    assert "剧本锚点" in review_prompt
    assert "黑西装/战术服" in review_prompt
    assert "黄色外卖箱、青玉断佩、银针、证件、蓝色电弧" in review_prompt
    content = transport.requests[0].messages[0].content
    assert isinstance(content, tuple)
    assert content[1]["type"] == "image_url"
    image_url = content[1]["image_url"]
    assert isinstance(image_url, Mapping)
    assert str(image_url["url"]).startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_asset_visual_reviewer_rejects_non_scene_background_pollution() -> None:
    transport = FakeVisionReviewTransport(
        response_text=(
            '{"passed":true,'
            '"summary":"服装妆造资产覆盖服装和材质，但背景非纯色，顶部出现办公室天花板和窗户，底部出现木质桌面。",'
            '"issues":["画面顶部出现办公室天花板与窗户背景，底部出现木质桌面"],'
            '"confidence":0.72}'
        )
    )

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="openai",
                api_base="https://api.openai.com/v1",
                api_protocol="openai_compatible",
                upstream_model="gpt-4.1",
                logical_model="vision_primary",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-openai",
                max_concurrency=2,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=2,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(deployments: tuple[Deployment, ...]) -> CapacityController:
        assert len(deployments) == 1
        return ImmediateCapacity()

    reviewer = _ConfigBackedAssetVisualReviewer(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
    )

    review = await reviewer.review_image_asset(
        tenant_id=TENANT_ID,
        label="服装妆造资产",
        prompt="生成服装妆造设定板，纯色背景。",
        filename="costume.png",
        mime_type="image/png",
        data=b"fake-png",
    )

    assert review.passed is False
    assert "背景非纯色" in review.summary
    assert any("非场景类资产" in issue for issue in review.issues)


def test_asset_visual_review_policy_ignores_negated_background_pollution() -> None:
    passed, summary, issues = _apply_asset_visual_review_policy(
        label="角色锁定资产：林渊",
        passed=True,
        summary=(
            "图像为标准角色参考设定板。纯白/浅灰背景，"
            "无场景、室内、街景或道具桌面污染；同一男性角色一致呈现三视图。"
        ),
        issues=(),
    )

    assert passed is True
    assert "标准角色参考设定板" in summary
    assert issues == ()


def test_asset_visual_review_policy_rejects_negative_summary_even_when_passed() -> None:
    passed, summary, issues = _apply_asset_visual_review_policy(
        label="道具资产",
        passed=True,
        summary="道具设定板仍不满足要求，核心问题未修正，标签错位且文字乱码。",
        issues=(),
    )

    assert passed is False
    assert "仍不满足" in summary
    assert any("不得将该资产自动判为通过" in issue for issue in issues)


def test_asset_visual_review_policy_keeps_passed_summary_when_prior_issues_are_fixed() -> None:
    passed, summary, issues = _apply_asset_visual_review_policy(
        label="音频字幕资产",
        passed=True,
        summary="前一版的乱码/伪字、角色头像与服装色彩漂移问题已修复，整体贴合资产类别。",
        issues=(),
    )

    assert passed is True
    assert "已修复" in summary
    assert issues == ()


def test_asset_visual_review_prompt_treats_script_characters_as_fictional() -> None:
    prompt = _asset_visual_review_prompt(
        label="角色锁定资产：秦岚",
        prompt="生成秦岚 Character Model Sheet，27岁女，灵纹鉴定师，短发，墨绿色长风衣。",
        filename="qinlan.png",
    )

    assert "脚本角色名默认都是虚构角色" in prompt
    assert "不要按现实明星、公众人物或同名真人资料" in prompt
    assert "用户明确要求真实人物" in prompt


def test_asset_visual_review_prompt_requires_professional_design_sheet_modules() -> None:
    prompt = _asset_visual_review_prompt(
        label="角色锁定资产：苏清月",
        prompt="生成苏清月 Character Model Sheet，白大褂，低马尾，银色胸针。",
        filename="suqingyue.png",
    )

    for required in (
        "主定妆大图",
        "正/侧/背全身三视图",
        "表情头部变化",
        "服装拆解",
        "剧情服装/状态变体",
        "每个模块都必须对应角色锚点",
        "栏目标题存在但内容明显不对应角色锚点",
        "关键栏目错别字",
        "随身物/职业道具",
        "材质色卡",
        "不得加入剧本或角色设定之外的随机道具",
        "少量清晰中文标签",
        "关键标签大量乱码",
        "剧本锚点",
        "换了人物身份/服装/画风",
        "不得自造无关角色",
        "空间视角",
        "独立物件 lineup",
        "姿态序列",
        "形态分层",
        "景别机位构图卡",
        "情绪节奏点",
    ):
        assert required in prompt


@pytest.mark.asyncio
async def test_asset_visual_reviewer_falls_back_to_json_when_schema_is_rejected() -> None:
    transport = SchemaRejectingVisionReviewTransport()

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="openai",
                api_base="https://api.openai.com/v1",
                api_protocol="openai_compatible",
                upstream_model="gpt-4.1",
                logical_model="vision_primary",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-openai",
                max_concurrency=2,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=2,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(deployments: tuple[Deployment, ...]) -> CapacityController:
        assert len(deployments) == 1
        return ImmediateCapacity()

    reviewer = _ConfigBackedAssetVisualReviewer(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
    )

    review = await reviewer.review_image_asset(
        tenant_id=TENANT_ID,
        label="角色锁定资产：男主",
        prompt="生成男主 Character Model Sheet，不要电影剧照。",
        filename="character.png",
        mime_type="image/png",
        data=b"fake-png",
    )

    assert review.passed is True
    assert [request.response_schema is not None for request in transport.requests] == [
        True,
        False,
    ]
    assert transport.requests[1].required_capabilities == frozenset({"vision"})


@pytest.mark.asyncio
async def test_asset_visual_reviewer_uses_json_for_vision_model_without_structured_output() -> None:
    transport = FakeVisionReviewTransport()

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="qwen",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_protocol="openai_compatible",
                upstream_model="qwen-vl",
                logical_model="vision_json",
                capabilities=["vision"],
                credential_ref="secret://main-agent",
                quota_scope="vision-json",
                max_concurrency=2,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=2,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(deployments: tuple[Deployment, ...]) -> CapacityController:
        assert len(deployments) == 1
        assert deployments[0].quota_scope_id == "vision-json"
        return ImmediateCapacity()

    reviewer = _ConfigBackedAssetVisualReviewer(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
    )

    review = await reviewer.review_image_asset(
        tenant_id=TENANT_ID,
        label="角色锁定资产：女主",
        prompt="生成女主 Character Model Sheet，不要混入其他角色。",
        filename="heroine.png",
        mime_type="image/png",
        data=b"fake-png",
    )

    assert review.passed is True
    assert review.logical_model == "vision_json"
    assert [deployment.quota_scope_id for deployment in transport.deployments] == [
        "vision-json"
    ]
    assert len(transport.requests) == 1
    assert transport.requests[0].required_capabilities == frozenset({"vision"})
    assert transport.requests[0].response_schema is None


@pytest.mark.asyncio
async def test_asset_visual_reviewer_falls_back_when_first_vision_deployment_has_no_capacity() -> None:
    transport = FakeVisionReviewTransport()
    capacity_scopes: list[str] = []

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="deepseek",
                api_base="https://api.deepseek.com/v1",
                api_protocol="openai_compatible",
                upstream_model="deepseek-vl",
                logical_model="vision_primary",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-busy",
                max_concurrency=3,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=3,
                saturation_policy="queue_first_then_fallback",
            ),
            ModelDeploymentResponse(
                provider="openai",
                api_base="https://api.openai.com/v1",
                api_protocol="openai_compatible",
                upstream_model="gpt-4.1",
                logical_model="vision_backup",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-backup",
                max_concurrency=2,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=2,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(deployments: tuple[Deployment, ...]) -> CapacityController:
        assert len(deployments) == 1
        capacity_scopes.append(deployments[0].quota_scope_id)
        if deployments[0].quota_scope_id == "vision-busy":
            return UnavailableCapacity()
        return ImmediateCapacity()

    reviewer = _ConfigBackedAssetVisualReviewer(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
    )

    review = await reviewer.review_image_asset(
        tenant_id=TENANT_ID,
        label="角色锁定资产：男主",
        prompt="生成男主 Character Model Sheet，不要电影剧照。",
        filename="character.png",
        mime_type="image/png",
        data=b"fake-png",
    )

    assert review.passed is True
    assert capacity_scopes == ["vision-busy", "vision-backup"]
    assert [deployment.quota_scope_id for deployment in transport.deployments] == [
        "vision-backup"
    ]
    assert review.logical_model == "vision_backup"


@pytest.mark.asyncio
async def test_asset_visual_reviewer_retries_when_all_candidates_temporarily_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeVisionReviewTransport()
    capacity_scopes: list[str] = []
    backup_failures_remaining = 1
    monkeypatch.setattr(app_module, "_ASSET_VISUAL_REVIEW_RETRY_BACKOFF_SECONDS", 0)

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="deepseek",
                api_base="https://api.deepseek.com/v1",
                api_protocol="openai_compatible",
                upstream_model="deepseek-vl",
                logical_model="vision_primary",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-busy",
                max_concurrency=3,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=3,
                saturation_policy="queue_first_then_fallback",
            ),
            ModelDeploymentResponse(
                provider="openai",
                api_base="https://api.openai.com/v1",
                api_protocol="openai_compatible",
                upstream_model="gpt-4.1",
                logical_model="vision_backup",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-backup",
                max_concurrency=2,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=2,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(deployments: tuple[Deployment, ...]) -> CapacityController:
        nonlocal backup_failures_remaining
        assert len(deployments) == 1
        capacity_scopes.append(deployments[0].quota_scope_id)
        if deployments[0].quota_scope_id == "vision-busy":
            return UnavailableCapacity()
        if backup_failures_remaining > 0:
            backup_failures_remaining -= 1
            return UnavailableCapacity()
        return ImmediateCapacity()

    reviewer = _ConfigBackedAssetVisualReviewer(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
    )

    review = await reviewer.review_image_asset(
        tenant_id=TENANT_ID,
        label="角色锁定资产：男主",
        prompt="生成男主 Character Model Sheet，不要电影剧照。",
        filename="character.png",
        mime_type="image/png",
        data=b"fake-png",
    )

    assert review.passed is True
    assert capacity_scopes == [
        "vision-busy",
        "vision-backup",
        "vision-busy",
        "vision-backup",
    ]
    assert [deployment.quota_scope_id for deployment in transport.deployments] == [
        "vision-backup"
    ]
    assert review.logical_model == "vision_backup"


@pytest.mark.asyncio
async def test_asset_visual_reviewer_reports_all_candidate_failures() -> None:
    transport = FakeVisionReviewTransport()

    first_id = uuid4()
    second_id = uuid4()

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="deepseek",
                api_base="https://api.deepseek.com/v1",
                api_protocol="openai_compatible",
                upstream_model="deepseek-vl",
                logical_model="vision_primary",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-busy",
                max_concurrency=3,
                target_utilization=0.8,
                reserved_capacity=0,
                id=first_id,
                effective_slots=3,
                saturation_policy="queue_first_then_fallback",
            ),
            ModelDeploymentResponse(
                provider="openai",
                api_base="https://api.openai.com/v1",
                api_protocol="openai_compatible",
                upstream_model="gpt-4.1",
                logical_model="vision_backup",
                capabilities=["vision", "structured_output"],
                credential_ref="secret://main-agent",
                quota_scope="vision-backup-busy",
                max_concurrency=2,
                target_utilization=0.8,
                reserved_capacity=0,
                id=second_id,
                effective_slots=2,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(deployments: tuple[Deployment, ...]) -> CapacityController:
        del deployments
        return UnavailableCapacity()

    reviewer = _ConfigBackedAssetVisualReviewer(
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
    )

    with pytest.raises(ModelGatewayError) as caught:
        await reviewer.review_image_asset(
            tenant_id=TENANT_ID,
            label="角色锁定资产：男主",
            prompt="生成男主 Character Model Sheet，不要电影剧照。",
            filename="character.png",
            mime_type="image/png",
            data=b"fake-png",
        )

    assert "visual asset review failed for all candidates" in str(caught.value)
    assert "vision_primary" in str(caught.value)
    assert "vision_backup" in str(caught.value)
    assert caught.value.logical_models == ("vision_primary", "vision_backup")
    assert caught.value.deployments == (str(first_id), str(second_id))
    assert transport.requests == []


@pytest.mark.asyncio
async def test_main_agent_mode_router_uses_configured_main_agent_model() -> None:
    transport = FakeTransport()

    async def get_main_agent_config() -> MainAgentConfigResponse:
        return MainAgentConfigResponse(
            model=MainAgentModelConfig(
                provider="deepseek",
                api_base="https://api.deepseek.com/v1",
                api_protocol="openai_compatible",
                upstream_model="deepseek-v4-flash",
                credential_ref="secret://main-agent",
                capabilities=["text", "structured_output"],
            )
        )

    async def capacity_factory(_deployments: tuple[Deployment, ...]) -> CapacityController:
        return ImmediateCapacity()

    router = _MainAgentModeRouter(
        get_config=get_main_agent_config,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
    )

    decision = await router.route("写一个活动策划，需要分工。")

    assert decision.status == "ready"
    assert decision.mode is not None
    assert decision.mode.value == "dispatch"
    assert decision.risk is RiskLevel.LOW
    assert len(transport.requests) == 1
    assert {request.logical_model for request in transport.requests} == {"main_agent"}


@pytest.mark.asyncio
async def test_main_agent_mode_router_inherits_registered_model_capabilities() -> None:
    transport = FakeTransport()

    async def get_main_agent_config() -> MainAgentConfigResponse:
        return MainAgentConfigResponse(
            model=MainAgentModelConfig(
                provider="minimax",
                api_base="https://api.minimax.chat/v1",
                api_protocol="openai_compatible",
                upstream_model="MiniMax-M3",
                credential_ref="secret://main-agent",
                capabilities=["text"],
            )
        )

    async def list_models() -> tuple[ModelDeploymentResponse, ...]:
        return (
            ModelDeploymentResponse(
                provider="minimax",
                api_base="https://api.minimax.chat/v1",
                api_protocol="openai_compatible",
                upstream_model="MiniMax-M3",
                logical_model="minimax",
                capabilities=["text", "structured_output", "tool_calling"],
                credential_ref="secret://main-agent",
                quota_scope="minimax-account",
                max_concurrency=1,
                target_utilization=0.8,
                reserved_capacity=0,
                id=uuid4(),
                effective_slots=1,
                saturation_policy="queue_first_then_fallback",
            ),
        )

    async def capacity_factory(_deployments: tuple[Deployment, ...]) -> CapacityController:
        assert _deployments[0].quota_scope_id == "minimax-account"
        assert _deployments[0].max_concurrency == 1
        return ImmediateCapacity()

    router = _MainAgentModeRouter(
        get_config=get_main_agent_config,
        list_models=list_models,
        secret_service=FakeSecretService(),  # type: ignore[arg-type]
        tenant_id=TENANT_ID,
        redis_client=object(),
        transport=transport,
        capacity_factory=capacity_factory,
    )

    decision = await router.route("写一份活动方案，需要策划和审核。")

    assert decision.status == "ready"
    assert decision.mode is not None
    assert decision.mode.value == "dispatch"
    assert len(transport.requests) == 1


def test_main_agent_context_window_is_inferred_from_model_family() -> None:
    assert _infer_main_agent_context_window_tokens("deepseek", "deepseek-v4-flash") == 128_000
    assert _infer_main_agent_context_window_tokens("anthropic", "claude-sonnet-4") == 200_000
    assert _infer_main_agent_context_window_tokens("openai", "gpt-5") == 400_000
    assert _infer_main_agent_context_window_tokens("custom", "unknown-small") == 32_768


@pytest.mark.asyncio
async def test_main_agent_context_window_getter_reads_configured_model() -> None:
    async def get_main_agent_config() -> MainAgentConfigResponse:
        return MainAgentConfigResponse(
            model=MainAgentModelConfig(
                provider="deepseek",
                api_base="https://api.deepseek.com/v1",
                api_protocol="openai_compatible",
                upstream_model="deepseek-v4-flash",
                credential_ref="secret://main-agent",
                capabilities=["text"],
            )
        )

    assert await _MainAgentContextWindowGetter(get_main_agent_config)() == 128_000
