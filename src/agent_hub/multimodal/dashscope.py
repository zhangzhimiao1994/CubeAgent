from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from agent_hub.multimodal.video_providers import (
    GeneratedVideoArtifact,
    VideoProviderGenerationError,
    media_filename_for_model,
    unique_media_path,
)

DashScopeGeneratedMedia = GeneratedVideoArtifact


class DashScopeGenerationError(VideoProviderGenerationError):
    """Safe provider error for DashScope/Bailian multimedia generation failures."""


class DashScopeMultimediaGenerationClient:
    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 60,
        poll_interval_seconds: float = 5,
        max_polls: int = 120,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be nonnegative")
        if max_polls <= 0:
            raise ValueError("max_polls must be positive")
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._max_polls = max_polls

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
    ) -> DashScopeGeneratedMedia:
        api_key = _required_string(api_key, "api_key")
        model = _required_string(model, "model")
        prompt = _required_string(prompt, "prompt")
        output_dir.mkdir(parents=True, exist_ok=True)
        async with self._client() as client:
            task_id = await self._submit_image(
                client,
                api_base=api_base,
                api_key=api_key,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
            media_url = await self._poll_for_image(client, api_base=api_base, api_key=api_key, task_id=task_id)
            stored_path, mime_type = await self._download(
                client,
                media_url=media_url,
                output_dir=output_dir,
                model=model,
                default_suffix=".png",
                default_mime_type="image/png",
            )
        return DashScopeGeneratedMedia(
            path=stored_path,
            uri=stored_path.as_uri(),
            provider="dashscope",
            model=model,
            task_id=task_id,
            file_id=None,
            mime_type=mime_type,
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
    ) -> DashScopeGeneratedMedia:
        api_key = _required_string(api_key, "api_key")
        model = _required_string(model, "model")
        prompt = _required_string(prompt, "prompt")
        output_dir.mkdir(parents=True, exist_ok=True)
        async with self._client() as client:
            task_id = await self._submit_video(
                client,
                api_base=api_base,
                api_key=api_key,
                model=model,
                prompt=prompt,
                duration=duration,
                mode=_video_mode(resolution),
            )
            media_url = await self._poll_for_video(client, api_base=api_base, api_key=api_key, task_id=task_id)
            stored_path, mime_type = await self._download(
                client,
                media_url=media_url,
                output_dir=output_dir,
                model=model,
                default_suffix=".mp4",
                default_mime_type="video/mp4",
            )
        return DashScopeGeneratedMedia(
            path=stored_path,
            uri=stored_path.as_uri(),
            provider="dashscope",
            model=model,
            task_id=task_id,
            file_id=None,
            mime_type=mime_type,
            kind="video",
        )

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout_seconds,
            transport=self._transport,
            follow_redirects=True,
        )

    async def _submit_image(
        self,
        client: httpx.AsyncClient,
        *,
        api_base: str,
        api_key: str,
        model: str,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
    ) -> str:
        response = await client.post(
            _service_url(api_base, "image-generation/generation"),
            headers=_headers(api_key),
            json={
                "model": model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}],
                        }
                    ]
                },
                "parameters": {
                    "n": 1,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "watermark": False,
                },
            },
        )
        payload = _json_object(response, "DashScope image submit failed")
        _raise_for_provider_failure(payload, "DashScope image submit failed")
        return _task_id(payload, "DashScope image submit response missing task_id")

    async def _submit_video(
        self,
        client: httpx.AsyncClient,
        *,
        api_base: str,
        api_key: str,
        model: str,
        prompt: str,
        duration: int,
        mode: str,
    ) -> str:
        response = await client.post(
            _service_url(api_base, "video-generation/video-synthesis"),
            headers=_headers(api_key),
            json={
                "model": model,
                "input": {"prompt": prompt},
                "parameters": {
                    "mode": mode,
                    "aspect_ratio": "16:9",
                    "duration": duration,
                    "audio": False,
                    "watermark": False,
                },
            },
        )
        payload = _json_object(response, "DashScope video submit failed")
        _raise_for_provider_failure(payload, "DashScope video submit failed")
        return _task_id(payload, "DashScope video submit response missing task_id")

    async def _poll_for_image(
        self,
        client: httpx.AsyncClient,
        *,
        api_base: str,
        api_key: str,
        task_id: str,
    ) -> str:
        payload = await self._poll(client, api_base=api_base, api_key=api_key, task_id=task_id)
        output = _mapping(payload.get("output"), "DashScope image query response missing output")
        choices = output.get("choices")
        if not isinstance(choices, list):
            raise DashScopeGenerationError("DashScope image query response missing choices")
        for choice in choices:
            message = _mapping(choice, "DashScope image query response malformed choice").get("message")
            content = _mapping(message, "DashScope image query response missing message").get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                image = _mapping(item, "DashScope image query response malformed content").get("image")
                if isinstance(image, str) and image.strip():
                    return image.strip()
        raise DashScopeGenerationError("DashScope image query response missing image URL")

    async def _poll_for_video(
        self,
        client: httpx.AsyncClient,
        *,
        api_base: str,
        api_key: str,
        task_id: str,
    ) -> str:
        payload = await self._poll(client, api_base=api_base, api_key=api_key, task_id=task_id)
        output = _mapping(payload.get("output"), "DashScope video query response missing output")
        for key in ("video_url", "watermark_video_url"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise DashScopeGenerationError("DashScope video query response missing video URL")

    async def _poll(
        self,
        client: httpx.AsyncClient,
        *,
        api_base: str,
        api_key: str,
        task_id: str,
    ) -> dict[str, Any]:
        for attempt in range(self._max_polls):
            response = await client.get(_task_url(api_base, task_id), headers=_headers(api_key))
            payload = _json_object(response, "DashScope task query failed")
            _raise_for_provider_failure(payload, "DashScope task query failed")
            output = _mapping(payload.get("output"), "DashScope task query response missing output")
            status = str(output.get("task_status") or "").strip().upper()
            if status == "SUCCEEDED":
                return payload
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                code = output.get("code")
                message = output.get("message")
                safe_code = str(code).strip()[:80] if isinstance(code, str) else status
                safe_message = str(message).strip()[:300] if isinstance(message, str) else status
                raise DashScopeGenerationError(
                    f"DashScope task failed: {safe_message}",
                    provider_code=safe_code,
                )
            if attempt < self._max_polls - 1 and self._poll_interval_seconds:
                await asyncio.sleep(self._poll_interval_seconds)
        raise DashScopeGenerationError("DashScope task polling timed out")

    async def _download(
        self,
        client: httpx.AsyncClient,
        *,
        media_url: str,
        output_dir: Path,
        model: str,
        default_suffix: str,
        default_mime_type: str,
    ) -> tuple[Path, str]:
        response = await client.get(media_url)
        if response.status_code >= 400:
            raise DashScopeGenerationError("DashScope media download failed")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
        suffix = _filename_suffix(media_url, content_type) or default_suffix
        target = unique_media_path(output_dir, media_filename_for_model(model, suffix=suffix))
        target.write_bytes(response.content)
        return target, content_type or default_mime_type


def is_dashscope_multimedia_deployment(
    provider: str,
    upstream_model: str,
    api_base: str = "",
) -> bool:
    normalized_provider = provider.casefold()
    normalized_model = upstream_model.casefold()
    normalized_base = api_base.casefold()
    provider_or_base_matches = any(
        marker in normalized_provider or marker in normalized_base
        for marker in ("dashscope", "qwen", "bailian", "aliyun", "alibaba", "maas")
    )
    return (
        provider_or_base_matches
        and any(marker in normalized_model for marker in ("kling", "wan"))
    )


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }


def _service_url(api_base: str, service: str) -> str:
    base = _normalized_root(api_base)
    if "/api/v1/services/aigc/" in base:
        return base
    return f"{base}/api/v1/services/aigc/{service}"


def _task_url(api_base: str, task_id: str) -> str:
    base = _normalized_root(api_base)
    if "/api/v1/services/aigc/" in base:
        base = base.split("/api/v1/services/aigc/", 1)[0]
    return f"{base}/api/v1/tasks/{task_id}"


def _normalized_root(api_base: str) -> str:
    parsed = urlsplit(_required_string(api_base, "api_base"))
    if not parsed.scheme or not parsed.netloc:
        raise DashScopeGenerationError("DashScope api_base is invalid")
    path = parsed.path.rstrip("/")
    if path.endswith("/compatible-mode/v1"):
        path = path[: -len("/compatible-mode/v1")]
    elif path.endswith("/api/v1"):
        path = path[: -len("/api/v1")]
    elif path.endswith("/v1"):
        path = path[: -len("/v1")]
    return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")


def _json_object(response: httpx.Response, message: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code >= 400:
        code = payload.get("code") if isinstance(payload, dict) else None
        detail = payload.get("message") if isinstance(payload, dict) else None
        suffix = f": {str(detail).strip()[:200]}" if isinstance(detail, str) and detail.strip() else ""
        raise DashScopeGenerationError(f"{message}{suffix}", provider_code=str(code or response.status_code)[:80])
    if not isinstance(payload, dict):
        raise DashScopeGenerationError(f"{message}: malformed JSON")
    return payload


def _raise_for_provider_failure(payload: Mapping[str, Any], message: str) -> None:
    code = payload.get("code")
    if isinstance(code, str) and code.strip():
        provider_message = payload.get("message")
        suffix = (
            f": {provider_message.strip()[:200]}"
            if isinstance(provider_message, str) and provider_message.strip()
            else ""
        )
        raise DashScopeGenerationError(f"{message}{suffix}", provider_code=code.strip()[:80])
    output = payload.get("output")
    if isinstance(output, Mapping):
        output_code = output.get("code")
        if isinstance(output_code, str) and output_code.strip():
            output_message = output.get("message")
            suffix = (
                f": {output_message.strip()[:200]}"
                if isinstance(output_message, str) and output_message.strip()
                else ""
            )
            raise DashScopeGenerationError(
                f"{message}{suffix}",
                provider_code=output_code.strip()[:80],
            )


def _task_id(payload: Mapping[str, Any], message: str) -> str:
    output = payload.get("output")
    if isinstance(output, Mapping):
        task_id = output.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            return task_id.strip()
    task_id = payload.get("task_id")
    if isinstance(task_id, str) and task_id.strip():
        return task_id.strip()
    raise DashScopeGenerationError(message)


def _mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DashScopeGenerationError(message)
    return value


def _filename_suffix(media_url: str, content_type: str) -> str | None:
    path_suffix = Path(urlsplit(media_url).path).suffix
    if path_suffix:
        return path_suffix[:16]
    if content_type == "image/png":
        return ".png"
    if content_type in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if content_type == "video/mp4":
        return ".mp4"
    return None


def _required_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DashScopeGenerationError(f"{name} is required")
    return value.strip()


def _video_mode(resolution: str) -> str:
    normalized = resolution.strip().casefold()
    if normalized in {"4k", "pro"}:
        return normalized
    return "std"


__all__ = [
    "DashScopeGeneratedMedia",
    "DashScopeGenerationError",
    "DashScopeMultimediaGenerationClient",
    "is_dashscope_multimedia_deployment",
]
