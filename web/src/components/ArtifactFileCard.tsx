import { useEffect, useState } from "react";

import { api, formatApiError } from "../api/client";
import type { RunDetail } from "../api/client";

type ArtifactFile = RunDetail["artifacts"][number] | NonNullable<RunDetail["events"][number]["artifact"]>;

export function hasArtifactDownload(artifact: ArtifactFile | null | undefined): artifact is ArtifactFile & { download_url: string } {
  return typeof artifact?.download_url === "string" && artifact.download_url.trim().length > 0;
}

export function artifactFileName(artifact: ArtifactFile) {
  return artifact.filename?.trim() || artifact.title || artifact.id;
}

function artifactDisplayName(artifact: ArtifactFile) {
  return artifact.title?.trim() || artifactFileName(artifact);
}

function formatFileSize(sizeBytes: number | null | undefined) {
  if (typeof sizeBytes !== "number" || !Number.isFinite(sizeBytes) || sizeBytes < 0) return "";
  if (sizeBytes < 1024) return `${sizeBytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = sizeBytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const precision = value >= 10 || Number.isInteger(value) ? 0 : 1;
  return `${value.toFixed(precision)} ${units[unitIndex]}`;
}

function formatExpiry(expiresAt: string | null | undefined) {
  if (typeof expiresAt !== "string" || expiresAt.trim().length === 0) return "";
  const date = new Date(expiresAt);
  if (Number.isNaN(date.getTime())) return "";
  return `有效至 ${date.toLocaleString("zh-CN", { hour12: false })}`;
}

function visualReviewLabel(artifact: ArtifactFile) {
  const review = artifact.visual_review;
  if (!review?.summary?.trim()) return "";
  return `${review.passed ? "视觉审核通过" : "视觉审核未通过"}：${review.summary.trim()}`;
}

function productionMetadataLabels(artifact: ArtifactFile) {
  const metadata = artifact.production_metadata;
  if (!metadata) return [];
  return [
    metadata.production_category ? `类别 ${metadata.production_category}` : "",
    metadata.character_id ? `Character ${metadata.character_id}` : "",
    metadata.look_id ? `Look ${metadata.look_id}` : "",
  ].filter(Boolean);
}

function downloadLabel(mimeType: string | null | undefined) {
  if (mimeType?.startsWith("image/")) return "下载图片";
  if (mimeType?.startsWith("video/")) return "下载视频";
  if (mimeType?.startsWith("audio/")) return "下载音频";
  return "下载文件";
}

export function ArtifactFileCard({
  artifact,
  compact = false,
}: {
  artifact: ArtifactFile;
  compact?: boolean;
}) {
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const filename = artifactFileName(artifact);
  const displayName = artifactDisplayName(artifact);
  const size = formatFileSize(artifact.size_bytes);
  const mimeType = artifact.mime_type?.trim();
  const checksum = artifact.sha256?.trim();
  const expiry = formatExpiry(artifact.expires_at);
  const reviewMeta = visualReviewLabel(artifact);
  const downloadUrl = artifact.download_url?.trim() || "";
  const hasDownload = downloadUrl.length > 0;
  const generationError =
    artifact.generation_error?.trim() || (!hasDownload ? artifact.text?.trim() || "" : "");
  const meta = [
    filename !== displayName ? filename : "",
    ...productionMetadataLabels(artifact),
    reviewMeta,
    generationError ? `生成失败：${generationError}` : "",
    artifact.kind,
    size,
    mimeType,
    expiry,
  ].filter(Boolean);
  const isImage = mimeType?.startsWith("image/") ?? false;
  const actionLabel = downloadLabel(mimeType);

  useEffect(() => {
    if (!isImage || compact || !hasDownload) {
      setPreviewUrl("");
      return undefined;
    }
    let cancelled = false;
    let objectUrl = "";
    void api
      .downloadGeneratedFile(downloadUrl)
      .then((downloaded) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(downloaded.blob);
        setPreviewUrl(objectUrl);
      })
      .catch(() => {
        if (!cancelled) setPreviewUrl("");
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [compact, downloadUrl, hasDownload, isImage]);

  async function downloadFile(event: React.MouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
    if (!hasDownload) return;
    setDownloading(true);
    setError("");
    try {
      const downloaded = await api.downloadGeneratedFile(downloadUrl);
      const url = URL.createObjectURL(downloaded.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = downloaded.filename || filename;
      anchor.rel = "noopener";
      anchor.style.display = "none";
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
    } catch (caught) {
      setError(formatApiError(caught, "下载失败"));
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className={`artifact-file-card${compact ? " artifact-file-card-compact" : ""}${isImage ? " artifact-file-card-image" : ""}${hasDownload ? "" : " artifact-file-card-missing"}`}>
      {previewUrl ? <img className="artifact-file-preview" src={previewUrl} alt={displayName} /> : null}
      <span className="artifact-file-icon" aria-hidden="true">
        {isImage ? "IMG" : "FILE"}
      </span>
      <div className="artifact-file-main">
        <strong>{displayName}</strong>
        {meta.length > 0 ? (
          <small className="artifact-file-meta">
            {meta.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </small>
        ) : null}
        {checksum ? <small title={checksum}>SHA-256 {checksum.slice(0, 12)}</small> : null}
        {error ? (
          <small className="artifact-file-error" role="alert">
            {error}
          </small>
        ) : null}
      </div>
      {hasDownload ? (
        <button
          type="button"
          className="artifact-file-download"
          onClick={downloadFile}
          disabled={downloading}
          aria-label={`下载 ${filename}`}
        >
          {downloading ? "下载中" : actionLabel}
        </button>
      ) : (
        <span className="artifact-file-download artifact-file-download-disabled">
          待重试
        </span>
      )}
    </div>
  );
}
