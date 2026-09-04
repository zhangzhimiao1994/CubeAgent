import { useState } from "react";

import { api, formatApiError } from "../api/client";
import type { RunDetail } from "../api/client";

type ArtifactFile = RunDetail["artifacts"][number] | NonNullable<RunDetail["events"][number]["artifact"]>;

export function hasArtifactDownload(artifact: ArtifactFile | null | undefined): artifact is ArtifactFile & { download_url: string } {
  return typeof artifact?.download_url === "string" && artifact.download_url.trim().length > 0;
}

export function artifactFileName(artifact: ArtifactFile) {
  return artifact.filename?.trim() || artifact.title || artifact.id;
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

export function ArtifactFileCard({
  artifact,
  compact = false,
}: {
  artifact: ArtifactFile & { download_url: string };
  compact?: boolean;
}) {
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");
  const filename = artifactFileName(artifact);
  const size = formatFileSize(artifact.size_bytes);
  const mimeType = artifact.mime_type?.trim();
  const checksum = artifact.sha256?.trim();
  const expiry = formatExpiry(artifact.expires_at);
  const meta = [artifact.kind, size, mimeType, expiry].filter(Boolean);

  async function downloadFile(event: React.MouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
    setDownloading(true);
    setError("");
    try {
      const downloaded = await api.downloadGeneratedFile(artifact.download_url);
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
    <div className={`artifact-file-card${compact ? " artifact-file-card-compact" : ""}`}>
      <span className="artifact-file-icon" aria-hidden="true">
        FILE
      </span>
      <div className="artifact-file-main">
        <strong>{filename}</strong>
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
      <button
        type="button"
        className="artifact-file-download"
        onClick={downloadFile}
        disabled={downloading}
        aria-label={`下载 ${filename}`}
      >
        {downloading ? "下载中" : "下载"}
      </button>
    </div>
  );
}
