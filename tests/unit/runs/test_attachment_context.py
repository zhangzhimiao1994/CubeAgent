from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from agent_hub.runs.attachments import FileSystemAttachmentArtifactLoader

TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")


@pytest.mark.asyncio
async def test_attachment_loader_turns_text_file_into_runtime_text_artifact(tmp_path: Path) -> None:
    tenant_dir = tmp_path / str(TENANT_ID)
    tenant_dir.mkdir()
    attachment_id = "att_11111111111111111111111111111111"
    expires_at = datetime.now(UTC) + timedelta(days=1)
    (tenant_dir / f"{attachment_id}.json").write_text(
        json.dumps(
            {
                "id": attachment_id,
                "filename": "notes.txt",
                "kind": "context",
                "content_type": "text/plain",
                "size_bytes": 18,
                "sha256": "abc",
                "expires_at": expires_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    (tenant_dir / f"{attachment_id}.bin").write_text("important context", encoding="utf-8")

    artifacts = await FileSystemAttachmentArtifactLoader(tmp_path)(
        tenant_id=TENANT_ID,
        attachment_ids=(attachment_id,),
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "text"
    assert artifact.producer == "uploaded_attachment"
    assert artifact.content["attachment_id"] == attachment_id
    assert artifact.content["filename"] == "notes.txt"
    assert "important context" in str(artifact.content["text"])


@pytest.mark.asyncio
async def test_attachment_loader_includes_archive_manifest_without_raw_binary(tmp_path: Path) -> None:
    tenant_dir = tmp_path / str(TENANT_ID)
    tenant_dir.mkdir()
    attachment_id = "att_22222222222222222222222222222222"
    expires_at = datetime.now(UTC) + timedelta(days=1)
    (tenant_dir / f"{attachment_id}.json").write_text(
        json.dumps(
            {
                "id": attachment_id,
                "filename": "project.zip",
                "kind": "archive",
                "content_type": "application/zip",
                "size_bytes": 1024,
                "sha256": "def",
                "expires_at": expires_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    (tenant_dir / f"{attachment_id}.bin").write_bytes(b"PK\x03\x04binary")
    (tenant_dir / f"{attachment_id}.manifest.json").write_text(
        json.dumps(
            {
                "archive": {"filename": "project.zip", "extracted": True, "format": "zip"},
                "files": [
                    {"path": "src/main.py", "size_bytes": 42, "extracted": True},
                    {"path": "README.md", "size_bytes": 24, "extracted": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    artifacts = await FileSystemAttachmentArtifactLoader(tmp_path)(
        tenant_id=TENANT_ID,
        attachment_ids=(attachment_id,),
    )

    assert len(artifacts) == 1
    text = str(artifacts[0].content["text"])
    assert "project.zip" in text
    assert "src/main.py" in text
    assert "README.md" in text
    assert "PK" not in text


@pytest.mark.asyncio
async def test_attachment_loader_extracts_docx_body_text(tmp_path: Path) -> None:
    tenant_dir = tmp_path / str(TENANT_ID)
    tenant_dir.mkdir()
    attachment_id = "att_33333333333333333333333333333333"
    expires_at = datetime.now(UTC) + timedelta(days=1)
    (tenant_dir / f"{attachment_id}.json").write_text(
        json.dumps(
            {
                "id": attachment_id,
                "filename": "需求说明.docx",
                "kind": "context",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "size_bytes": 2048,
                "sha256": "ghi",
                "expires_at": expires_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(tenant_dir / f"{attachment_id}.bin", "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>请生成项目方案</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>必须包含测试计划</w:t></w:r></w:p></w:body></w:document>"
            ),
        )

    artifacts = await FileSystemAttachmentArtifactLoader(tmp_path)(
        tenant_id=TENANT_ID,
        attachment_ids=(attachment_id,),
    )

    assert len(artifacts) == 1
    text = str(artifacts[0].content["text"])
    assert "Office 文档正文" in text
    assert "请生成项目方案" in text
    assert "必须包含测试计划" in text
    assert "当前只注入文件元数据" not in text


@pytest.mark.asyncio
async def test_attachment_loader_reports_docx_parse_failure_without_crashing(tmp_path: Path) -> None:
    tenant_dir = tmp_path / str(TENANT_ID)
    tenant_dir.mkdir()
    attachment_id = "att_44444444444444444444444444444444"
    expires_at = datetime.now(UTC) + timedelta(days=1)
    (tenant_dir / f"{attachment_id}.json").write_text(
        json.dumps(
            {
                "id": attachment_id,
                "filename": "坏文件.docx",
                "kind": "context",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "size_bytes": 10,
                "sha256": "bad",
                "expires_at": expires_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    (tenant_dir / f"{attachment_id}.bin").write_bytes(b"not-a-docx")

    artifacts = await FileSystemAttachmentArtifactLoader(tmp_path)(
        tenant_id=TENANT_ID,
        attachment_ids=(attachment_id,),
    )

    assert len(artifacts) == 1
    text = str(artifacts[0].content["text"])
    assert "Word 文档正文解析失败" in text
    assert "文件名：坏文件.docx" in text
