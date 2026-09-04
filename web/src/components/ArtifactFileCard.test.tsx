import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ArtifactFileCard } from "./ArtifactFileCard";
import type { RunDetail } from "../api/client";

const artifact = {
  id: "33333333-3333-4333-8333-333333333333",
  kind: "docx",
  title: "执行报告",
  text: null,
  filename: "run-report.docx",
  mime_type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  size_bytes: 2048,
  sha256: "0f4d0c8d0e4d9d3a0a6a8e2e4b7a6c1d8e9f0a1b2c3d4e5f67890123456789cd",
  download_url:
    "/api/v1/admin/runs/22222222-2222-4222-8222-222222222222/artifacts/33333333-3333-4333-8333-333333333333/download",
} satisfies RunDetail["artifacts"][number] & { download_url: string };

describe("ArtifactFileCard", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    window.sessionStorage.clear();
  });

  it("downloads artifacts through authenticated fetch instead of an unauthenticated link navigation", async () => {
    const user = userEvent.setup();
    window.sessionStorage.setItem("agent_hub_access_token", "owner-token");
    const click = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(new Blob(["docx-bytes"], { type: artifact.mime_type }), {
          status: 200,
          headers: {
            "Content-Type": artifact.mime_type,
            "Content-Disposition": 'attachment; filename="run-report.docx"',
          },
        }),
      ),
    );
    vi.stubGlobal(
      "URL",
      Object.assign(URL, {
        createObjectURL: vi.fn(() => "blob:artifact-download"),
        revokeObjectURL: vi.fn(),
      }),
    );
    vi.spyOn(document.body, "appendChild");
    vi.spyOn(document.body, "removeChild");
    vi.spyOn(document, "createElement").mockImplementation((tagName: string) => {
      const element = document.createElementNS("http://www.w3.org/1999/xhtml", tagName);
      if (tagName.toLowerCase() === "a") {
        Object.defineProperty(element, "click", { value: click });
      }
      return element as HTMLElement;
    });

    render(<ArtifactFileCard artifact={artifact} />);

    await user.click(screen.getByRole("button", { name: /下载 run-report\.docx/ }));

    await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    expect(fetch).toHaveBeenCalledWith(
      artifact.download_url,
      expect.objectContaining({
        credentials: "include",
        headers: expect.objectContaining({ Authorization: "Bearer owner-token" }),
      }),
    );
    expect(URL.createObjectURL).toHaveBeenCalled();
  });
});
