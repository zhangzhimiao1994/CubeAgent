from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from agent_hub.models.types import Deployment
from agent_hub.multimodal.video_providers import (
    VideoProviderGenerationError,
    media_filename_for_model,
    unique_media_path,
)


@dataclass(frozen=True, slots=True)
class GeneratedAudioArtifact:
    path: Path
    uri: str
    provider: str
    model: str
    task_id: str
    file_id: str | None
    mime_type: str
    kind: str = "audio"


class AudioProviderGenerationError(VideoProviderGenerationError):
    """Safe provider-level audio generation failure."""


class MiniMaxAudioGenerationError(AudioProviderGenerationError):
    """Safe provider error for MiniMax TTS failures."""


class TextToAudioProvider(Protocol):
    async def generate_text_to_audio(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        prompt: str,
        output_dir: Path,
    ) -> GeneratedAudioArtifact: ...


class TextToAudioProviderRouter:
    def __init__(self, providers: tuple[tuple[str, TextToAudioProvider], ...]) -> None:
        self._providers = providers

    def provider_for(self, deployment: Deployment) -> TextToAudioProvider | None:
        provider_name = deployment.provider_model.partition("/")[0].casefold()
        for marker, provider in self._providers:
            if marker.casefold() in provider_name:
                return provider
        return None


class MiniMaxAudioGenerationClient:
    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 60,
        voice_id: str = "male-qn-qingse",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._voice_id = _required_string(voice_id, "voice_id")

    async def generate_text_to_audio(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        prompt: str,
        output_dir: Path,
    ) -> GeneratedAudioArtifact:
        api_key = _required_string(api_key, "api_key")
        model = _required_string(model, "model")
        prompt = _required_string(prompt, "prompt")
        output_dir.mkdir(parents=True, exist_ok=True)
        async with self._client() as client:
            payload = await self._submit(
                client,
                base=_normalized_base(api_base),
                api_key=api_key,
                model=model,
                prompt=prompt,
            )
            stored_path, mime_type = await self._store_audio(
                client,
                payload=payload,
                output_dir=output_dir,
                model=model,
            )
        return GeneratedAudioArtifact(
            path=stored_path,
            uri=stored_path.as_uri(),
            provider="minimax",
            model=model,
            task_id=_trace_id(payload),
            file_id=None,
            mime_type=mime_type,
        )

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout_seconds,
            transport=self._transport,
            follow_redirects=True,
        )

    async def _submit(
        self,
        client: httpx.AsyncClient,
        *,
        base: str,
        api_key: str,
        model: str,
        prompt: str,
    ) -> dict[str, Any]:
        response = await client.post(
            f"{base}/t2a_v2",
            headers=_headers(api_key),
            json={
                "model": model,
                "text": prompt,
                "stream": False,
                "voice_setting": {
                    "voice_id": self._voice_id,
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "format": "mp3",
                    "channel": 1,
                },
            },
        )
        payload = _json_object(response, "MiniMax audio generation failed")
        _raise_for_provider_failure(payload, "MiniMax audio generation failed")
        return payload

    async def _store_audio(
        self,
        client: httpx.AsyncClient,
        *,
        payload: Mapping[str, Any],
        output_dir: Path,
        model: str,
    ) -> tuple[Path, str]:
        data = payload.get("data")
        if not isinstance(data, Mapping):
            data = payload
        audio_hex = data.get("audio")
        if isinstance(audio_hex, str) and audio_hex.strip():
            try:
                audio_bytes = bytes.fromhex(audio_hex.strip())
            except ValueError:
                raise MiniMaxAudioGenerationError("MiniMax audio response has invalid audio hex") from None
            target = unique_media_path(output_dir, media_filename_for_model(model, suffix=".mp3"))
            target.write_bytes(audio_bytes)
            return target, "audio/mpeg"

        audio_url = _first_string(data, ("audio_url", "download_url", "url"))
        if audio_url is not None:
            return await self._download_audio(
                client,
                audio_url=audio_url,
                output_dir=output_dir,
                model=model,
            )
        raise MiniMaxAudioGenerationError("MiniMax audio response missing audio")

    async def _download_audio(
        self,
        client: httpx.AsyncClient,
        *,
        audio_url: str,
        output_dir: Path,
        model: str,
    ) -> tuple[Path, str]:
        response = await client.get(audio_url)
        if response.status_code >= 400:
            raise MiniMaxAudioGenerationError("MiniMax audio download failed")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
        suffix = _filename_suffix(audio_url, content_type) or ".mp3"
        target = unique_media_path(output_dir, media_filename_for_model(model, suffix=suffix))
        target.write_bytes(response.content)
        return target, content_type or _mime_type_for_suffix(suffix)


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _json_object(response: httpx.Response, message: str) -> dict[str, Any]:
    if response.status_code >= 400:
        raise MiniMaxAudioGenerationError(message)
    try:
        payload = response.json()
    except ValueError:
        raise MiniMaxAudioGenerationError(f"{message}: malformed JSON") from None
    if not isinstance(payload, dict):
        raise MiniMaxAudioGenerationError(f"{message}: malformed JSON")
    return payload


def _raise_for_provider_failure(payload: Mapping[str, Any], message: str) -> None:
    base_resp = payload.get("base_resp")
    if isinstance(base_resp, Mapping):
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0, "0"):
            status_msg = base_resp.get("status_msg")
            suffix = f": {status_msg}" if isinstance(status_msg, str) and status_msg else ""
            raise MiniMaxAudioGenerationError(
                f"{message}{suffix}",
                provider_code=str(status_code),
            )
    code = payload.get("code")
    if code not in (None, 0, "0"):
        raise MiniMaxAudioGenerationError(message, provider_code=str(code))


def _trace_id(payload: Mapping[str, Any]) -> str:
    trace_id = payload.get("trace_id")
    if isinstance(trace_id, str) and trace_id.strip():
        return trace_id.strip()
    return "minimax-audio"


def _first_string(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _required_string(value: str, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be nonblank")
    return value.strip()


def _normalized_base(api_base: str) -> str:
    value = _required_string(api_base, "api_base").rstrip("/")
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def _filename_suffix(url: str, content_type: str) -> str | None:
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    if suffix:
        return suffix
    if content_type == "audio/mpeg":
        return ".mp3"
    if content_type == "audio/wav":
        return ".wav"
    if content_type == "audio/mp4":
        return ".m4a"
    if content_type == "audio/aac":
        return ".aac"
    return None


def _mime_type_for_suffix(suffix: str) -> str:
    normalized = suffix.casefold()
    if normalized == ".wav":
        return "audio/wav"
    if normalized in {".m4a", ".mp4"}:
        return "audio/mp4"
    if normalized == ".aac":
        return "audio/aac"
    return "audio/mpeg"


__all__ = [
    "AudioProviderGenerationError",
    "GeneratedAudioArtifact",
    "MiniMaxAudioGenerationClient",
    "MiniMaxAudioGenerationError",
    "TextToAudioProvider",
    "TextToAudioProviderRouter",
]
