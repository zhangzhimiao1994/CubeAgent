import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TestApp } from "../app/router";

function jsonResponse(payload: unknown, init: ResponseInit = {}) {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

type TestProject = {
  [key: string]: unknown;
  asset_manifest?: unknown;
  execution_mode: "demo" | "production";
  final_approved: boolean;
  project_id: string;
  qc_report?: unknown;
  revision: number;
  rights_approved: boolean;
  script?: unknown;
  script_approved: boolean;
  status: string;
  title: string;
  topic: string;
};

const contentProject: TestProject = {
  project_id: "project-123",
  revision: 1,
  title: "AIGC 科普项目",
  topic: "做一条 60 秒 AIGC 科普视频",
  status: "SCRIPT_READY",
  script_approved: false,
  rights_approved: false,
  final_approved: false,
  execution_mode: "demo",
  script: {
    title: "AIGC 正在变成软件入口",
    segments: [{ id: "seg-1", text: "前三秒说明发生了什么。" }],
  },
  asset_manifest: {
    assets: [
      {
        asset_id: "asset-a",
        title: "女主定妆资产",
        kind: "image",
        rights_status: "unknown",
      },
      {
        asset_id: "asset-b",
        title: "官方 Demo 截图",
        kind: "image",
        rights_status: "licensed",
      },
    ],
  },
  evidence_graph: {
    evidence: [
      {
        evidence_id: "EV001",
        source_url: "https://official.example/release",
        publisher: "Official",
      },
    ],
  },
  qc_report: {
    blockers: [],
    summary: "演示预览可播放，但未进入正式生产。",
  },
  project_events: [
    {
      event_id: "event-1",
      sequence: 1,
      kind: "stage_completed",
      stage: "research",
      status: "RESEARCH_READY",
      title: "Research 调研",
      summary: "3 个研究问题，2 条证据",
      artifact_refs: ["EV001"],
      payload: { questions: ["AIGC 发生了什么？"] },
    },
    {
      event_id: "event-2",
      sequence: 2,
      kind: "stage_failed",
      stage: "voice",
      status: "failed",
      title: "阶段失败",
      summary: "tts_timeout: qwen-tts 首次调用超时",
      artifact_refs: [],
      payload: { error_code: "tts_timeout" },
    },
    {
      event_id: "event-3",
      sequence: 3,
      kind: "provider_attempt",
      stage: "voice",
      status: "completed",
      title: "Voice Provider 调用",
      summary: "voice 调用完成：voice.mp3",
      artifact_refs: ["voice.mp3"],
      payload: { provider_task_id: "voice.mp3" },
    },
  ],
  provider_attempts: [
    {
      stage: "voice",
      status: "failed",
      idempotency_key: "project-123:voice",
      error_code: "tts_timeout",
    },
  ],
};

const otherProject: TestProject = {
  ...contentProject,
  project_id: "project-456",
  revision: 1,
  title: "另一个 Content 项目",
  topic: "做一条教程视频",
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}

describe("ContentStudioPage", () => {
  const requests: Array<{ body: unknown; method: string; path: string }> = [];
  let visibleProject = contentProject;
  let visibleOtherProject = otherProject;
  let runResponses: Array<Promise<TestProject>> = [];
  let approveRightsResponse: Promise<TestProject> | null = null;
  let project456Response: Promise<TestProject> | null = null;

  beforeEach(() => {
    requests.length = 0;
    visibleProject = contentProject;
    visibleOtherProject = otherProject;
    runResponses = [];
    approveRightsResponse = null;
    project456Response = null;
    window.sessionStorage.setItem("agent_hub_access_token", "owner-token");
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input);
        const method = init?.method ?? "GET";
        const body = init?.body && typeof init.body === "string" ? JSON.parse(init.body) : null;
        requests.push({ path, method, body });
        if (path === "/api/v1/auth/me") {
          return jsonResponse({
            user_id: "11111111-1111-4111-8111-111111111111",
            tenant_id: "33333333-3333-4333-8333-333333333333",
            username: "owner",
            role: "super_admin",
            permissions: ["*"],
          });
        }
        if (path === "/api/v1/content-studio/projects/project-123" && method === "GET") {
          return jsonResponse(visibleProject);
        }
        if (path === "/api/v1/content-studio/projects/project-456" && method === "GET") {
          if (project456Response) return jsonResponse(await project456Response);
          return jsonResponse(visibleOtherProject);
        }
        if (path === "/api/v1/content-studio/projects/project-123/run" && method === "POST") {
          if (runResponses.length) return jsonResponse(await runResponses.shift());
          visibleProject = {
            ...visibleProject,
            revision: visibleProject.revision + 1,
            status: body && typeof body === "object" && "until" in body ? String(body.until) : "QC_REVIEW",
            preview_render: { url: "/preview/project-123.mp4" },
            qc_report: { blockers: [], summary: "QC 通过，可进入终片批准。" },
          };
          return jsonResponse(visibleProject);
        }
        if (path === "/api/v1/content-studio/projects/project-123/approve-rights" && method === "POST") {
          if (approveRightsResponse) return jsonResponse(await approveRightsResponse);
          visibleProject = { ...visibleProject, revision: visibleProject.revision + 1, rights_approved: true };
          return jsonResponse(visibleProject);
        }
        if (path === "/api/v1/content-studio/projects/project-123/approve-script" && method === "POST") {
          visibleProject = { ...visibleProject, revision: visibleProject.revision + 1, script_approved: true };
          return jsonResponse(visibleProject);
        }
        if (path === "/api/v1/content-studio/projects/project-123/approve-final" && method === "POST") {
          visibleProject = { ...visibleProject, revision: visibleProject.revision + 1, final_approved: true };
          return jsonResponse(visibleProject);
        }
        if (path === "/api/v1/content-studio/projects/project-123/claims/CL001" && method === "POST") {
          return jsonResponse({ ...visibleProject, revision: visibleProject.revision + 1 });
        }
        return jsonResponse({ error: { code: "not_found", message: "not found" } }, { status: 404 });
      }),
    );
  });

  afterEach(() => {
    window.sessionStorage.clear();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("restores the active project from the URL and keeps demo approval state visible", async () => {
    render(<TestApp initialPath="/content-studio?project=project-123" />);

    expect(await screen.findByRole("heading", { name: "AIGC 科普项目" })).not.toBeNull();
    expect(requests.some((request) => request.path === "/api/v1/content-studio/projects/project-123")).toBe(true);
    expect(screen.getByText("演示模式")).not.toBeNull();
    expect(screen.getByText(/当前输出仅用于演示验证/)).not.toBeNull();
    expect(screen.getByText("脚本待批准")).not.toBeNull();
    expect(screen.getByText("版权待批准")).not.toBeNull();
    expect(screen.getByText("终片待批准")).not.toBeNull();
    expect(screen.getByText("女主定妆资产")).not.toBeNull();
    expect(screen.getByText("Research 调研")).not.toBeNull();
    expect(screen.getByText("tts_timeout: qwen-tts 首次调用超时")).not.toBeNull();
    expect(screen.getByText("关联产物：voice.mp3")).not.toBeNull();
    expect(screen.getAllByText("查看结构化详情").length).toBeGreaterThan(0);
  });

  it("approves rights only for selected assets and shows a busy state", async () => {
    const user = userEvent.setup();
    const rights = deferred<TestProject>();
    approveRightsResponse = rights.promise;
    render(<TestApp initialPath="/content-studio?project=project-123" />);

    expect(await screen.findByRole("heading", { name: "AIGC 科普项目" })).not.toBeNull();
    await user.click(screen.getByRole("checkbox", { name: /女主定妆资产/ }));
    await user.clear(screen.getByLabelText("版权批准备注"));
    await user.type(screen.getByLabelText("版权批准备注"), "仅批准女主定妆授权。");
    await user.click(screen.getByRole("button", { name: "批准所选版权" }));

    expect((screen.getByRole("button", { name: "版权批准中..." }) as HTMLButtonElement).disabled).toBe(true);
    await waitFor(() =>
      expect(
        requests.find((request) => request.path === "/api/v1/content-studio/projects/project-123/approve-rights"),
      ).toMatchObject({
        method: "POST",
        body: {
          asset_ids: ["asset-a"],
          note: "仅批准女主定妆授权。",
        },
      }),
    );
    rights.resolve({ ...contentProject, revision: 2, rights_approved: true });
    expect(await screen.findByText("版权已批准")).not.toBeNull();
  });

  it("keeps the newest revision visible when refresh returns an older snapshot", async () => {
    const user = userEvent.setup();
    render(<TestApp initialPath="/content-studio?project=project-123" />);

    expect(await screen.findByRole("heading", { name: "AIGC 科普项目" })).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "5. 运行 Video QC" }));
    await waitFor(() => expect(screen.getByText("QC 通过，可进入终片批准。")).not.toBeNull());
    visibleProject = {
      ...contentProject,
      revision: 1,
      status: "RESEARCH_READY",
      title: "较旧的研究结果",
    };
    await user.click(screen.getByRole("button", { name: "刷新项目状态" }));

    await waitFor(() => expect(screen.queryByRole("heading", { name: "较旧的研究结果" })).toBeNull());
    expect(screen.getByRole("heading", { name: "AIGC 科普项目" })).not.toBeNull();
    expect(screen.getByText("QC 通过，可进入终片批准。")).not.toBeNull();
  });

  it("does not let a late result from the previous project replace the active project", async () => {
    const user = userEvent.setup();
    const firstRun = deferred<TestProject>();
    runResponses = [firstRun.promise];
    render(<TestApp initialPath="/content-studio?project=project-123" />);

    expect(await screen.findByRole("heading", { name: "AIGC 科普项目" })).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "1. 运行 Research / Fact Check" }));
    await user.clear(screen.getByLabelText("项目 ID"));
    await user.type(screen.getByLabelText("项目 ID"), "project-456");
    await user.click(screen.getByRole("button", { name: "打开项目" }));

    expect(await screen.findByRole("heading", { name: "另一个 Content 项目" })).not.toBeNull();
    firstRun.resolve({
      ...contentProject,
      revision: 2,
      status: "RESEARCH_READY",
      title: "迟到的旧项目结果",
    });

    await waitFor(() => expect(screen.queryByRole("heading", { name: "迟到的旧项目结果" })).toBeNull());
    expect(screen.getByRole("heading", { name: "另一个 Content 项目" })).not.toBeNull();
  });

  it("does not expose the previous project while a newly selected project is still loading", async () => {
    const user = userEvent.setup();
    const delayedProject = deferred<TestProject>();
    project456Response = delayedProject.promise;
    render(<TestApp initialPath="/content-studio?project=project-123" />);

    expect(await screen.findByRole("heading", { name: "AIGC 科普项目" })).not.toBeNull();
    await user.click(screen.getByRole("checkbox", { name: /女主定妆资产/ }));
    await user.clear(screen.getByLabelText("项目 ID"));
    await user.type(screen.getByLabelText("项目 ID"), "project-456");
    await user.click(screen.getByRole("button", { name: "打开项目" }));

    await waitFor(() => expect(screen.queryByRole("heading", { name: "AIGC 科普项目" })).toBeNull());
    expect((screen.getByRole("button", { name: "批准所选版权" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByText("女主定妆资产")).toBeNull();

    delayedProject.resolve(visibleOtherProject);
    expect(await screen.findByRole("heading", { name: "另一个 Content 项目" })).not.toBeNull();
  });

  it("requires evidence IDs when marking a claim as supported", async () => {
    const user = userEvent.setup();
    render(<TestApp initialPath="/content-studio?project=project-123" />);

    expect(await screen.findByRole("heading", { name: "AIGC 科普项目" })).not.toBeNull();
    expect((screen.getByRole("button", { name: "更新事实状态" }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole("checkbox", { name: /EV001/ }));
    await user.click(screen.getByRole("button", { name: "更新事实状态" }));

    await waitFor(() =>
      expect(
        requests.find((request) => request.path === "/api/v1/content-studio/projects/project-123/claims/CL001"),
      ).toMatchObject({
        method: "POST",
        body: {
          status: "supported",
          note: "人工补充核验后可使用。",
          evidence_ids: ["EV001"],
        },
      }),
    );
  });

  it("requires script approval, rights approval, preview, and blocker-free QC before final approval", async () => {
    const user = userEvent.setup();
    render(<TestApp initialPath="/content-studio?project=project-123" />);

    expect(await screen.findByRole("heading", { name: "AIGC 科普项目" })).not.toBeNull();
    expect((screen.getByRole("button", { name: "批准终片" }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole("button", { name: "批准脚本" }));
    await waitFor(() => expect(screen.getByText("脚本已批准")).not.toBeNull());
    await user.click(screen.getByRole("checkbox", { name: /女主定妆资产/ }));
    await user.click(screen.getByRole("button", { name: "批准所选版权" }));
    await waitFor(() => expect(screen.getByText("版权已批准")).not.toBeNull());
    expect((screen.getByRole("button", { name: "批准终片" }) as HTMLButtonElement).disabled).toBe(true);

    await user.click(screen.getByRole("button", { name: "5. 运行 Video QC" }));
    await waitFor(() => expect((screen.getByRole("button", { name: "批准终片" }) as HTMLButtonElement).disabled).toBe(false));
    await user.click(screen.getByRole("button", { name: "批准终片" }));

    await waitFor(() =>
      expect(
        requests.find((request) => request.path === "/api/v1/content-studio/projects/project-123/approve-final"),
      ).toMatchObject({ method: "POST" }),
    );
    expect(await screen.findByText("终片已批准")).not.toBeNull();
  });
});
