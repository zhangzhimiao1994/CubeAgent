from __future__ import annotations

import json

import httpx
import pytest

from agent_hub.multimodal.dashscope import (
    DashScopeMultimediaGenerationClient,
    is_dashscope_multimedia_deployment,
)


@pytest.mark.asyncio
async def test_dashscope_image_client_submits_polls_downloads_and_stores_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/services/aigc/image-generation/generation":
            assert request.headers["X-DashScope-Async"] == "enable"
            payload = json.loads(request.content.decode("utf-8"))
            assert payload == {
                "model": "kling/kling-v3-omni-image-generation",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": "a blue cube"}],
                        }
                    ]
                },
                "parameters": {
                    "n": 1,
                    "aspect_ratio": "1:1",
                    "resolution": "1k",
                    "watermark": False,
                },
            }
            return httpx.Response(
                200,
                json={"output": {"task_status": "PENDING", "task_id": "task-image-1"}},
            )
        if request.url.path == "/api/v1/tasks/task-image-1":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "task_id": "task-image-1",
                        "task_status": "SUCCEEDED",
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {
                                            "type": "image",
                                            "image": "https://media.example/out.png",
                                        }
                                    ]
                                }
                            }
                        ],
                    }
                },
            )
        if request.url.host == "media.example":
            return httpx.Response(200, content=b"png", headers={"content-type": "image/png"})
        return httpx.Response(404)

    client = DashScopeMultimediaGenerationClient(
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0,
    )

    artifact = await client.generate_text_to_image(
        api_key="sk-live",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="kling/kling-v3-omni-image-generation",
        prompt="a blue cube",
        output_dir=tmp_path,
    )

    assert artifact.kind == "image"
    assert artifact.provider == "dashscope"
    assert artifact.task_id == "task-image-1"
    assert artifact.mime_type == "image/png"
    assert artifact.path.read_bytes() == b"png"
    assert artifact.path.name.endswith(".png")
    assert [request.url.path for request in requests] == [
        "/api/v1/services/aigc/image-generation/generation",
        "/api/v1/tasks/task-image-1",
        "/out.png",
    ]


@pytest.mark.asyncio
async def test_dashscope_video_client_submits_polls_downloads_and_stores_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/services/aigc/video-generation/video-synthesis":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload == {
                "model": "kling/kling-v3-omni-video-generation",
                "input": {"prompt": "a cat runs"},
                "parameters": {
                    "mode": "std",
                    "aspect_ratio": "16:9",
                    "duration": 5,
                    "audio": False,
                    "watermark": False,
                },
            }
            return httpx.Response(
                200,
                json={"output": {"task_status": "PENDING", "task_id": "task-video-1"}},
            )
        if request.url.path == "/api/v1/tasks/task-video-1":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "task_id": "task-video-1",
                        "task_status": "SUCCEEDED",
                        "video_url": "https://media.example/out.mp4",
                    }
                },
            )
        if request.url.host == "media.example":
            return httpx.Response(200, content=b"mp4", headers={"content-type": "video/mp4"})
        return httpx.Response(404)

    client = DashScopeMultimediaGenerationClient(
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0,
    )

    artifact = await client.generate_text_to_video(
        api_key="sk-live",
        api_base="https://abc123.cn-beijing.maas.aliyuncs.com/api/v1",
        model="kling/kling-v3-omni-video-generation",
        prompt="a cat runs",
        output_dir=tmp_path,
        duration=5,
        resolution="std",
    )

    assert artifact.kind == "video"
    assert artifact.provider == "dashscope"
    assert artifact.task_id == "task-video-1"
    assert artifact.mime_type == "video/mp4"
    assert artifact.path.read_bytes() == b"mp4"
    assert artifact.path.name.endswith(".mp4")
    assert [request.url.path for request in requests] == [
        "/api/v1/services/aigc/video-generation/video-synthesis",
        "/api/v1/tasks/task-video-1",
        "/out.mp4",
    ]


def test_dashscope_detection_accepts_api_base_marker_for_openai_compatible_provider() -> None:
    assert is_dashscope_multimedia_deployment(
        "openai-compatible",
        "kling/kling-v3-omni-image-generation",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    assert not is_dashscope_multimedia_deployment(
        "openai-compatible",
        "kling/kling-v3-omni-image-generation",
        "https://example.invalid/v1",
    )
