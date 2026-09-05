from __future__ import annotations

from typing import cast
from uuid import UUID

from fastapi import APIRouter, Header, Request, WebSocket
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_hub.api.errors import PublicAPIError, error_responses
from agent_hub.channels.submitter import RunServiceInboundSubmitter
from agent_hub.robot.session import RobotChannelSession, RobotRunControl
from agent_hub.robot.tokens import DeviceTokenService
from agent_hub.robot.voice.gateway import RobotVoiceGateway


class DeviceRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1, max_length=128)

    @field_validator("device_id")
    @classmethod
    def safe_device_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("device_id is invalid")
        return cleaned


class DeviceRegisterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str
    device_token: str


def create_robot_router() -> APIRouter:
    router = APIRouter(
        prefix="/api/robot/v1",
        tags=["robot"],
        responses=error_responses(401, 405, 422, 500, 503),
    )

    @router.post("/devices/register", response_model=DeviceRegisterResponse)
    async def register_device(
        request: Request,
        body: DeviceRegisterRequest,
    ) -> DeviceRegisterResponse:
        tokens = _device_tokens(request)
        try:
            issued = tokens.register(body.device_id)
        except ValueError as error:
            raise PublicAPIError(422, "invalid_device_id", "device_id is invalid") from error
        return DeviceRegisterResponse(
            device_id=issued.device_id,
            device_token=issued.device_token,
        )

    @router.websocket("/ws")
    async def robot_ws(
        websocket: WebSocket,
        x_device_token: str | None = Header(default=None, alias="X-Device-Token"),
    ) -> None:
        await websocket.accept()
        token = x_device_token or websocket.query_params.get("device_token")
        tokens = getattr(websocket.app.state, "robot_device_tokens", None)
        device = (
            tokens.authenticate(token)
            if isinstance(tokens, DeviceTokenService)
            else None
        )
        if device is None:
            await websocket.close(code=4401)
            return
        submitter = getattr(websocket.app.state, "robot_submitter", None)
        run_service = getattr(websocket.app.state, "run_service", None)
        tenant_id = getattr(websocket.app.state, "bootstrap_tenant_id", None)
        if (
            not isinstance(submitter, RunServiceInboundSubmitter)
            or run_service is None
            or not isinstance(tenant_id, UUID)
        ):
            await websocket.send_json({"type": "error", "message": "robot channel unavailable"})
            await websocket.close(code=1011)
            return
        voice = getattr(websocket.app.state, "robot_voice", None)
        session = RobotChannelSession(
            websocket,
            device=device,
            submitter=submitter,
            run_service=cast(RobotRunControl, run_service),
            tenant_id=tenant_id,
            tenant_external_id=str(tenant_id),
            voice=voice if isinstance(voice, RobotVoiceGateway) else None,
        )
        await session.run()

    return router


def _device_tokens(request: Request) -> DeviceTokenService:
    tokens = getattr(request.app.state, "robot_device_tokens", None)
    if not isinstance(tokens, DeviceTokenService):
        raise PublicAPIError(503, "service_unavailable", "robot channel unavailable")
    return tokens


__all__ = ["create_robot_router"]
