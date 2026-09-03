"""Runtime attachment context loading for uploaded run files."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from agent_hub.runtime.contracts import Artifact

_MAX_TEXT_ATTACHMENT_BYTES = 24_000
_MAX_ARCHIVE_FILES = 80
_TEXT_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/javascript",
    "application/typescript",
    "application/x-python-code",
)


class FileSystemAttachmentArtifactLoader:
    """Convert uploaded attachment metadata into bounded runtime text artifacts."""

    def __init__(self, attachment_store_dir: Path) -> None:
        self._attachment_store_dir = attachment_store_dir

    async def __call__(
        self,
        *,
        tenant_id: UUID,
        attachment_ids: tuple[str, ...],
    ) -> tuple[Artifact, ...]:
        tenant_dir = (self._attachment_store_dir / str(tenant_id)).resolve()
        root = self._attachment_store_dir.resolve()
        if not tenant_dir.is_relative_to(root):
            return ()

        artifacts: list[Artifact] = []
        for attachment_id in attachment_ids:
            metadata = _read_json(tenant_dir / f"{attachment_id}.json")
            if metadata is None or metadata.get("id") != attachment_id:
                continue
            if _expired(metadata.get("expires_at")):
                continue
            text = _attachment_text(
                tenant_dir=tenant_dir,
                attachment_id=attachment_id,
                metadata=metadata,
            )
            artifacts.append(
                Artifact(
                    id=uuid5(
                        NAMESPACE_URL,
                        f"agent-hub:uploaded-attachment:{tenant_id}:{attachment_id}",
                    ),
                    type="text",
                    producer="uploaded_attachment",
                    content={
                        "text": text,
                        "attachment_id": attachment_id,
                        "filename": str(metadata.get("filename") or "attachment.bin"),
                        "kind": str(metadata.get("kind") or "context"),
                        "content_type": str(metadata.get("content_type") or "application/octet-stream"),
                        "trust": "user_uploaded_attachment",
                    },
                )
            )
        return tuple(artifacts)


def _attachment_text(
    *,
    tenant_dir: Path,
    attachment_id: str,
    metadata: dict[str, object],
) -> str:
    filename = str(metadata.get("filename") or "attachment.bin")
    kind = str(metadata.get("kind") or "context")
    content_type = str(metadata.get("content_type") or "application/octet-stream")
    size_bytes = metadata.get("size_bytes")
    lines = [
        "用户本轮上传了附件，以下内容与当前对话消息直接关联。",
        f"附件ID：{attachment_id}",
        f"文件名：{filename}",
        f"附件类型：{kind}",
        f"MIME：{content_type}",
        f"大小：{size_bytes} bytes" if isinstance(size_bytes, int) else "大小：unknown",
    ]

    manifest = _read_json(tenant_dir / f"{attachment_id}.manifest.json")
    if manifest is not None:
        lines.extend(_archive_manifest_lines(manifest))
    elif _is_textual_content_type(content_type):
        content = _read_text_attachment(tenant_dir / f"{attachment_id}.bin")
        if content:
            lines.extend(("文本内容：", content))
    else:
        lines.append("内容摘要：该附件是非文本或二进制文件，当前只注入文件元数据；需要视觉/多模态模型时由后续链路处理。")

    return "\n".join(lines)


def _archive_manifest_lines(manifest: dict[str, object]) -> list[str]:
    archive = manifest.get("archive")
    files = manifest.get("files")
    lines = ["压缩包清单："]
    if isinstance(archive, dict):
        archive_format = archive.get("format")
        extracted = archive.get("extracted")
        lines.append(f"- 格式：{archive_format}; 已解包：{extracted}")
    if not isinstance(files, list) or not files:
        lines.append("- 未解析到安全文件清单。")
        return lines
    for item in files[:_MAX_ARCHIVE_FILES]:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        size = item.get("size_bytes")
        extracted = item.get("extracted")
        if isinstance(path, str):
            lines.append(f"- {path} ({size} bytes, extracted={extracted})")
    if len(files) > _MAX_ARCHIVE_FILES:
        lines.append(f"- 其余 {len(files) - _MAX_ARCHIVE_FILES} 个文件已省略，避免上下文膨胀。")
    return lines


def _read_text_attachment(path: Path) -> str:
    try:
        body = path.read_bytes()
    except OSError:
        return ""
    truncated = len(body) > _MAX_TEXT_ATTACHMENT_BYTES
    body = body[:_MAX_TEXT_ATTACHMENT_BYTES]
    text = body.decode("utf-8", errors="replace").strip()
    if truncated:
        text = f"{text}\n[附件文本已截断，仅保留前 {_MAX_TEXT_ATTACHMENT_BYTES} bytes]"
    return text


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _expired(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        expires_at = datetime.fromisoformat(value)
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC)


def _is_textual_content_type(content_type: str) -> bool:
    normalized = content_type.split(";", 1)[0].strip().lower()
    return any(normalized.startswith(prefix) for prefix in _TEXT_CONTENT_TYPES)
