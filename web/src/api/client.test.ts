import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, formatApiError } from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("api client authentication errors", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
  });

  it("clears the remembered session and formats invalid_token as an expired login", async () => {
    window.sessionStorage.setItem("agent_hub_access_token", "expired-token");
    window.sessionStorage.setItem("agent_hub_tenant_id", "tenant-1");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          { error: { code: "invalid_token", message: "invalid access token" } },
          401,
        ),
      ),
    );

    let caught: unknown;
    try {
      await api.createRun({ mode: "direct", message: "hello" });
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(ApiError);
    expect(formatApiError(caught, "消息发送失败")).toBe("登录已失效，请重新登录。");
    expect(window.sessionStorage.getItem("agent_hub_access_token")).toBeNull();
    expect(window.sessionStorage.getItem("agent_hub_tenant_id")).toBeNull();
  });
});
