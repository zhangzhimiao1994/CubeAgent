import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { api, formatApiError, type ContentStudioProject } from "../api/client";

const STAGES = [
  "RESEARCH_READY",
  "FACT_CHECKED",
  "SCRIPT_READY",
  "STORYBOARD_READY",
  "ASSETS_READY",
  "TIMELINE_READY",
  "PREVIEW_RENDERED",
  "QC_REVIEW",
] as const;

type Stage = (typeof STAGES)[number];

type AssetSummary = {
  asset_id: string;
  title: string;
  kind: string;
  rights_status: string;
};

type EvidenceSummary = {
  evidence_id: string;
  title: string;
  source_url: string;
};

const contentStudioProjectKey = (projectId: string) => ["content-studio-project", projectId] as const;

export function ContentStudioPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const queryProjectId = useMemo(() => new URLSearchParams(location.search).get("project")?.trim() ?? "", [location.search]);
  const [title, setTitle] = useState("AIGC 科普短视频");
  const [topic, setTopic] = useState("做一条 60 秒 AIGC 科普视频，引用官方发布信息。");
  const [sourceUrls, setSourceUrls] = useState("");
  const [projectId, setProjectId] = useState(queryProjectId);
  const [openProjectInput, setOpenProjectInput] = useState(queryProjectId);
  const [revision, setRevision] = useState("换一个更强但不夸张的开头");
  const [storyboardRevision, setStoryboardRevision] = useState("第一个镜头改成官方 Demo 录屏，不要数字人");
  const [assetId, setAssetId] = useState("");
  const [assetRevision, setAssetRevision] = useState("只重新生成这个素材，保持脚本和其他素材不变");
  const [retryStage, setRetryStage] = useState<Stage>("ASSETS_READY");
  const [claimId, setClaimId] = useState("CL001");
  const [claimStatus, setClaimStatus] = useState("supported");
  const [claimNote, setClaimNote] = useState("人工补充核验后可使用。");
  const [claimEvidenceIdsText, setClaimEvidenceIdsText] = useState("");
  const [rightsNote, setRightsNote] = useState("仅批准已核验且授权明确的素材。");
  const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>([]);

  useEffect(() => {
    setProjectId(queryProjectId);
    setOpenProjectInput(queryProjectId);
    setSelectedAssetIds([]);
    setClaimEvidenceIdsText("");
  }, [queryProjectId]);

  function cacheProject(project: ContentStudioProject) {
    queryClient.setQueryData<ContentStudioProject | undefined>(contentStudioProjectKey(project.project_id), (current) =>
      shouldApplyProjectRevision(current, project) ? project : current,
    );
  }

  function openProject(project: ContentStudioProject) {
    cacheProject(project);
    openProjectId(project.project_id);
  }

  function openProjectId(nextProjectId: string) {
    setProjectId(nextProjectId);
    setSelectedAssetIds([]);
    const params = new URLSearchParams(location.search);
    if (nextProjectId) {
      params.set("project", nextProjectId);
    } else {
      params.delete("project");
    }
    if ((params.get("project") ?? "") !== (new URLSearchParams(location.search).get("project") ?? "")) {
      navigate({ pathname: location.pathname, search: `?${params.toString()}` }, { replace: true });
    }
  }

  const projectQuery = useQuery({
    queryKey: contentStudioProjectKey(projectId),
    queryFn: () => api.getContentStudioProject(projectId),
    enabled: Boolean(projectId),
    refetchOnWindowFocus: false,
    structuralSharing: (oldData, newData) =>
      shouldApplyProjectRevision(oldData as ContentStudioProject | undefined, newData as ContentStudioProject)
        ? newData
        : oldData,
  });
  const visibleProject = projectQuery.data?.project_id === projectId ? projectQuery.data : undefined;
  const activeProjectId = visibleProject?.project_id ?? "";
  const canApproveFinal = projectCanApproveFinal(visibleProject);

  const createProject = useMutation({
    mutationFn: () =>
      api.createContentStudioProject({
        title: title.trim(),
        topic: topic.trim(),
        source_urls: sourceUrls
          .split(/\r?\n/)
          .map((item) => item.trim())
          .filter(Boolean),
      }),
    onSuccess: openProject,
  });

  const runProject = useMutation({
    mutationFn: ({ id, until }: { id: string; until: Stage }) => api.runContentStudioProject(id, until),
    onSuccess: cacheProject,
  });

  const reviseScript = useMutation({
    mutationFn: ({ id, instruction }: { id: string; instruction: string }) => api.reviseContentStudioScript(id, instruction),
    onSuccess: cacheProject,
  });

  const approveScript = useMutation({
    mutationFn: ({ id, revision }: { id: string; revision: number }) => api.approveContentStudioScript(id, revision),
    onSuccess: cacheProject,
  });

  const approveRights = useMutation({
    mutationFn: ({ id, assetIds, note, revision }: { id: string; assetIds: string[]; note: string; revision: number }) =>
      api.approveContentStudioRights(id, {
        asset_ids: assetIds,
        note,
        revision,
      }),
    onSuccess: cacheProject,
  });

  const reviseStoryboard = useMutation({
    mutationFn: ({ id, instruction }: { id: string; instruction: string }) =>
      api.reviseContentStudioStoryboard(id, instruction),
    onSuccess: cacheProject,
  });

  const regenerateAsset = useMutation({
    mutationFn: ({ id, selectedAssetId, instruction }: { id: string; selectedAssetId: string; instruction: string }) =>
      api.regenerateContentStudioAsset(id, selectedAssetId, instruction),
    onSuccess: cacheProject,
  });

  const renderPreview = useMutation({
    mutationFn: (id: string) => api.renderContentStudioPreview(id),
    onSuccess: cacheProject,
  });

  const approveFinal = useMutation({
    mutationFn: ({ id, revision }: { id: string; revision: number }) => api.approveContentStudioFinal(id, revision),
    onSuccess: cacheProject,
  });

  const retryContentStage = useMutation({
    mutationFn: ({ id, stage }: { id: string; stage: Stage }) => api.retryContentStudioStage(id, stage),
    onSuccess: cacheProject,
  });

  const updateClaim = useMutation({
    mutationFn: ({
      id,
      selectedClaimId,
      evidenceIds,
    }: {
      id: string;
      selectedClaimId: string;
      evidenceIds: string[];
    }) =>
      api.updateContentStudioClaim(id, selectedClaimId, {
        status: claimStatus,
        note: claimNote,
        evidence_ids: evidenceIds,
      }),
    onSuccess: cacheProject,
  });

  const assets = useMemo(() => projectAssets(visibleProject), [visibleProject]);
  const evidenceItems = useMemo(() => collectProjectEvidence(visibleProject), [visibleProject]);
  const claimEvidenceIds = useMemo(() => parseEvidenceIdText(claimEvidenceIdsText), [claimEvidenceIdsText]);
  const claimNeedsEvidence = claimStatus === "supported" || claimStatus === "partially_supported";
  const canUpdateClaim = Boolean(activeProjectId && claimId.trim() && (!claimNeedsEvidence || claimEvidenceIds.length > 0));
  const statusSummary = useMemo(() => projectStatusSummary(visibleProject), [visibleProject]);
  const anyActionPending =
    runProject.isPending ||
    reviseScript.isPending ||
    approveScript.isPending ||
    approveRights.isPending ||
    reviseStoryboard.isPending ||
    regenerateAsset.isPending ||
    renderPreview.isPending ||
    approveFinal.isPending ||
    retryContentStage.isPending ||
    updateClaim.isPending;

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!title.trim() || !topic.trim()) return;
    createProject.mutate();
  }

  function toggleAsset(assetId: string) {
    setSelectedAssetIds((current) =>
      current.includes(assetId) ? current.filter((item) => item !== assetId) : [...current, assetId],
    );
  }

  function toggleEvidence(evidenceId: string) {
    setClaimEvidenceIdsText((current) => {
      const ids = parseEvidenceIdText(current);
      const nextIds = ids.includes(evidenceId) ? ids.filter((item) => item !== evidenceId) : [...ids, evidenceId];
      return nextIds.join(", ");
    });
  }

  return (
    <section>
      <p className="eyebrow">Production Studios</p>
      <h2>Content Studio</h2>

      <form className="form-grid schedule-form" aria-label="创建 Content Studio 项目" onSubmit={submit}>
        <label htmlFor="content-title">
          标题
          <input id="content-title" value={title} onChange={(event) => setTitle(event.target.value)} />
        </label>
        <label htmlFor="content-topic">
          主题
          <textarea id="content-topic" rows={4} value={topic} onChange={(event) => setTopic(event.target.value)} />
        </label>
        <label htmlFor="content-sources">
          官方来源
          <textarea
            id="content-sources"
            rows={3}
            value={sourceUrls}
            placeholder="示例：https://openai.com/index/product-release"
            onChange={(event) => setSourceUrls(event.target.value)}
          />
        </label>
        <div className="toolbar">
          <button type="submit" disabled={createProject.isPending}>
            {createProject.isPending ? "创建中..." : "创建项目"}
          </button>
          {STAGES.map((stage) => (
            <button
              key={stage}
              type="button"
              className="secondary-action"
              disabled={!activeProjectId || anyActionPending}
              onClick={() => runProject.mutate({ id: requireProjectId(activeProjectId), until: stage })}
            >
              运行到 {stage}
            </button>
          ))}
        </div>
      </form>

      <form
        className="form-grid schedule-form"
        aria-label="打开 Content Studio 项目"
        onSubmit={(event) => {
          event.preventDefault();
          openProjectId(openProjectInput.trim());
        }}
      >
        <label htmlFor="content-open-project">
          项目 ID
          <input
            id="content-open-project"
            value={openProjectInput}
            placeholder="project_..."
            onChange={(event) => setOpenProjectInput(event.target.value)}
          />
        </label>
        <button type="submit" className="secondary-action" disabled={!openProjectInput.trim()}>
          打开项目
        </button>
      </form>

      {createProject.isError ? <p role="alert">{formatApiError(createProject.error, "项目创建失败")}</p> : null}
      {projectQuery.isError ? <p role="alert">{formatApiError(projectQuery.error, "项目加载失败")}</p> : null}
      {runProject.isError ? <p role="alert">{formatApiError(runProject.error, "项目运行失败")}</p> : null}
      {reviseScript.isError ? <p role="alert">{formatApiError(reviseScript.error, "脚本修改失败")}</p> : null}
      {approveScript.isError ? <p role="alert">{formatApiError(approveScript.error, "脚本批准失败")}</p> : null}
      {approveRights.isError ? <p role="alert">{formatApiError(approveRights.error, "版权批准失败")}</p> : null}
      {reviseStoryboard.isError ? <p role="alert">{formatApiError(reviseStoryboard.error, "分镜修改失败")}</p> : null}
      {regenerateAsset.isError ? <p role="alert">{formatApiError(regenerateAsset.error, "素材重生成失败")}</p> : null}
      {renderPreview.isError ? <p role="alert">{formatApiError(renderPreview.error, "预览渲染失败")}</p> : null}
      {approveFinal.isError ? <p role="alert">{formatApiError(approveFinal.error, "终片批准失败")}</p> : null}
      {retryContentStage.isError ? <p role="alert">{formatApiError(retryContentStage.error, "阶段重试失败")}</p> : null}
      {updateClaim.isError ? <p role="alert">{formatApiError(updateClaim.error, "事实状态更新失败")}</p> : null}

      <section className="section-band content-studio-overview" aria-label="项目概览">
        <p className="eyebrow">Project</p>
        <h3>{visibleProject?.title ?? "尚未创建项目"}</h3>
        <div className="toolbar">
          <span className={visibleProject?.execution_mode === "production" ? "status-pill status-pill-success" : "status-pill"}>
            {visibleProject?.execution_mode === "production" ? "正式生产" : "演示模式"}
          </span>
          {visibleProject?.execution_mode !== "production" ? (
            <span className="content-studio-demo-note">当前输出仅用于演示验证，不能标记为正式交付。</span>
          ) : null}
          <button
            type="button"
            className="secondary-action"
            disabled={!projectId || projectQuery.isFetching}
            onClick={() => projectQuery.refetch()}
          >
            {projectQuery.isFetching ? "刷新中..." : "刷新项目状态"}
          </button>
        </div>
        <dl>
          <dt>Project ID</dt>
          <dd>{visibleProject?.project_id ?? "无"}</dd>
          <dt>状态</dt>
          <dd>{visibleProject?.status ?? "DRAFT"}</dd>
          <dt>当前完成度</dt>
          <dd>{statusSummary}</dd>
        </dl>
        <div className="content-studio-approval-strip" aria-label="审批状态">
          <ApprovalPill approved={Boolean(visibleProject?.script_approved)} approvedText="脚本已批准" pendingText="脚本待批准" />
          <ApprovalPill approved={Boolean(visibleProject?.rights_approved)} approvedText="版权已批准" pendingText="版权待批准" />
          <ApprovalPill approved={Boolean(visibleProject?.final_approved)} approvedText="终片已批准" pendingText="终片待批准" />
        </div>
      </section>

      <form className="form-grid schedule-form" aria-label="修改脚本" onSubmit={(event) => {
        event.preventDefault();
        if (activeProjectId && revision.trim()) {
          reviseScript.mutate({ id: requireProjectId(activeProjectId), instruction: revision });
        }
      }}>
        <label htmlFor="content-revision">
          脚本修改
          <input id="content-revision" value={revision} onChange={(event) => setRevision(event.target.value)} />
        </label>
        <button type="submit" disabled={!activeProjectId || reviseScript.isPending}>
          {reviseScript.isPending ? "修改中..." : "提交脚本修改"}
        </button>
        <button
          type="button"
          className="secondary-action"
          disabled={!activeProjectId || !visibleProject?.script || Boolean(visibleProject?.script_approved) || anyActionPending}
          onClick={() =>
            approveScript.mutate({
              id: requireProjectId(activeProjectId),
              revision: visibleProject?.revision ?? 0,
            })
          }
        >
          {approveScript.isPending ? "脚本批准中..." : "批准脚本"}
        </button>
      </form>

      <section className="section-band" aria-label="版权批准">
        <p className="eyebrow">Rights Review</p>
        <h3>按素材批准版权</h3>
        <p className="content-studio-helper">只勾选已经核验来源和授权状态的素材。这里不会自动整包通过。</p>
        <div className="content-studio-asset-list" role="list" aria-label="素材版权列表">
          {assets.length ? (
            assets.map((asset) => (
              <label key={asset.asset_id} className="content-studio-asset-row">
                <input
                  type="checkbox"
                  checked={selectedAssetIds.includes(asset.asset_id)}
                  onChange={() => toggleAsset(asset.asset_id)}
                />
                <span>
                  <strong>{asset.title}</strong>
                  <small>
                    {asset.asset_id} · {asset.kind} · 权利状态 {asset.rights_status}
                  </small>
                </span>
              </label>
            ))
          ) : (
            <p>暂无可审批素材。</p>
          )}
        </div>
        <label htmlFor="rights-note">
          版权批准备注
          <input id="rights-note" value={rightsNote} onChange={(event) => setRightsNote(event.target.value)} />
        </label>
        <button
          type="button"
          className="secondary-action"
          disabled={
            !activeProjectId ||
            Boolean(visibleProject?.rights_approved) ||
            selectedAssetIds.length === 0 ||
            !rightsNote.trim() ||
            anyActionPending
          }
          onClick={() =>
            approveRights.mutate({
              id: requireProjectId(activeProjectId),
              assetIds: selectedAssetIds,
              note: rightsNote.trim(),
              revision: visibleProject?.revision ?? 0,
            })
          }
        >
          {approveRights.isPending ? "版权批准中..." : "批准所选版权"}
        </button>
      </section>

      <section className="section-band" aria-label="项目操作">
        <p className="eyebrow">Operations</p>
        <h3>局部修改与恢复</h3>
        <div className="form-grid">
          <label htmlFor="storyboard-revision">
            分镜修改
            <input
              id="storyboard-revision"
              value={storyboardRevision}
              onChange={(event) => setStoryboardRevision(event.target.value)}
            />
          </label>
          <label htmlFor="asset-id">
            素材 ID
            <input id="asset-id" value={assetId} onChange={(event) => setAssetId(event.target.value)} />
          </label>
          <label htmlFor="asset-revision">
            素材重生成要求
            <input id="asset-revision" value={assetRevision} onChange={(event) => setAssetRevision(event.target.value)} />
          </label>
          <label htmlFor="retry-stage">
            重试阶段
            <select id="retry-stage" value={retryStage} onChange={(event) => setRetryStage(event.target.value as Stage)}>
              {STAGES.map((stage) => (
                <option key={stage} value={stage}>
                  {stage}
                </option>
              ))}
            </select>
          </label>
          <label htmlFor="claim-id">
            Claim ID
            <input id="claim-id" value={claimId} onChange={(event) => setClaimId(event.target.value)} />
          </label>
          <label htmlFor="claim-status">
            Claim 状态
            <select id="claim-status" value={claimStatus} onChange={(event) => setClaimStatus(event.target.value)}>
              <option value="supported">supported</option>
              <option value="partially_supported">partially_supported</option>
              <option value="conflicting">conflicting</option>
              <option value="outdated">outdated</option>
              <option value="unsupported">unsupported</option>
              <option value="opinion">opinion</option>
            </select>
          </label>
          <label htmlFor="claim-note">
            Claim 说明
            <input id="claim-note" value={claimNote} onChange={(event) => setClaimNote(event.target.value)} />
          </label>
          <label htmlFor="claim-evidence-ids">
            Evidence IDs
            <input
              id="claim-evidence-ids"
              value={claimEvidenceIdsText}
              placeholder="EV001, EV002"
              onChange={(event) => setClaimEvidenceIdsText(event.target.value)}
            />
          </label>
        </div>
        <div className="content-studio-evidence-list" role="list" aria-label="事实证据列表">
          {evidenceItems.length ? (
            evidenceItems.map((evidence) => (
              <label key={evidence.evidence_id} className="content-studio-asset-row">
                <input
                  type="checkbox"
                  checked={claimEvidenceIds.includes(evidence.evidence_id)}
                  onChange={() => toggleEvidence(evidence.evidence_id)}
                />
                <span>
                  <strong>{evidence.evidence_id}</strong>
                  <small>
                    {evidence.title}
                    {evidence.source_url ? ` · ${evidence.source_url}` : ""}
                  </small>
                </span>
              </label>
            ))
          ) : (
            <p>暂无可选择证据。supported / partially_supported 必须填写 Evidence IDs。</p>
          )}
        </div>
        {claimNeedsEvidence && claimEvidenceIds.length === 0 ? (
          <p className="content-studio-helper">supported / partially_supported 必须绑定 Evidence IDs，不能只用说明备注通过核验。</p>
        ) : null}
        <div className="toolbar">
          <button
            type="button"
            className="secondary-action"
            disabled={!activeProjectId || anyActionPending}
            onClick={() => reviseStoryboard.mutate({ id: requireProjectId(activeProjectId), instruction: storyboardRevision })}
          >
            {reviseStoryboard.isPending ? "分镜提交中..." : "提交分镜修改"}
          </button>
          <button
            type="button"
            className="secondary-action"
            disabled={!activeProjectId || !assetId.trim() || anyActionPending}
            onClick={() =>
              regenerateAsset.mutate({
                id: requireProjectId(activeProjectId),
                selectedAssetId: assetId.trim(),
                instruction: assetRevision,
              })
            }
          >
            {regenerateAsset.isPending ? "素材重生成中..." : "重生成单个素材"}
          </button>
          <button
            type="button"
            className="secondary-action"
            disabled={!activeProjectId || anyActionPending}
            onClick={() => retryContentStage.mutate({ id: requireProjectId(activeProjectId), stage: retryStage })}
          >
            {retryContentStage.isPending ? "阶段重试中..." : "重试所选阶段"}
          </button>
          <button
            type="button"
            className="secondary-action"
            disabled={!canUpdateClaim || anyActionPending}
            onClick={() =>
              updateClaim.mutate({
                id: requireProjectId(activeProjectId),
                selectedClaimId: claimId.trim(),
                evidenceIds: claimEvidenceIds,
              })
            }
          >
            {updateClaim.isPending ? "事实更新中..." : "更新事实状态"}
          </button>
          <button
            type="button"
            className="secondary-action"
            disabled={!activeProjectId || anyActionPending}
            onClick={() => renderPreview.mutate(requireProjectId(activeProjectId))}
          >
            {renderPreview.isPending ? "预览渲染中..." : "渲染预览"}
          </button>
          <button
            type="button"
            className="secondary-action"
            disabled={!activeProjectId || !canApproveFinal || Boolean(visibleProject?.final_approved) || anyActionPending}
            onClick={() =>
              approveFinal.mutate({
                id: requireProjectId(activeProjectId),
                revision: visibleProject?.revision ?? 0,
              })
            }
          >
            {approveFinal.isPending ? "终片批准中..." : "批准终片"}
          </button>
          <span className="content-studio-helper">
            终片批准需要脚本批准、版权批准、已有预览和无 BLOCKER 的 QC 报告。
          </span>
        </div>
      </section>

      <div className="resource-list content-studio-grid" aria-label="Content Studio 状态分区">
        <StudioPanel title="Research" value={visibleProject?.research_bundle} />
        <StudioPanel title="Evidence" value={visibleProject?.evidence_graph} />
        <StudioPanel title="Fact Check" value={visibleProject?.fact_check_report} />
        <StudioPanel title="Plan" value={visibleProject?.content_plan} />
        <StudioPanel title="Script" value={visibleProject?.script} />
        <StudioPanel title="Storyboard" value={visibleProject?.storyboard} />
        <StudioPanel title="Assets" value={visibleProject?.asset_manifest} summary={assets.length ? `${assets.length} 个素材待管理` : undefined} />
        <StudioPanel title="Voice" value={visibleProject?.voice_track} />
        <StudioPanel title="Timeline" value={visibleProject?.timeline} />
        <StudioPanel title="QC" value={visibleProject?.qc_report} />
      </div>
    </section>
  );
}

function ApprovalPill({
  approved,
  approvedText,
  pendingText,
}: {
  approved: boolean;
  approvedText: string;
  pendingText: string;
}) {
  return (
    <span className={approved ? "status-pill status-pill-success" : "status-pill status-pill-muted"}>
      {approved ? approvedText : pendingText}
    </span>
  );
}

function StudioPanel({ title, value, summary }: { title: string; value: unknown; summary?: string }) {
  const displaySummary = summary ?? valueSummary(value);
  return (
    <article>
      <p className="eyebrow">{value ? "ready" : "pending"}</p>
      <h3>{title}</h3>
      <p>{displaySummary}</p>
      {value ? (
        <details>
          <summary>查看结构化详情</summary>
          <pre className="json-preview">{JSON.stringify(value, null, 2)}</pre>
        </details>
      ) : null}
    </article>
  );
}

function projectStatusSummary(project: ContentStudioProject | undefined): string {
  if (!project) return "等待创建";
  const ready = [
    project.research_bundle,
    project.evidence_graph,
    project.fact_check_report,
    project.content_plan,
    project.script,
    project.storyboard,
    project.asset_manifest,
    project.voice_track,
    project.timeline,
    project.qc_report,
  ].filter(Boolean).length;
  return `${ready}/10 个阶段已有产物`;
}

function shouldApplyProjectRevision(
  current: ContentStudioProject | undefined,
  incoming: ContentStudioProject,
): boolean {
  if (!current) return true;
  if (current.project_id !== incoming.project_id) return true;
  return incoming.revision > current.revision;
}

function projectCanApproveFinal(project: ContentStudioProject | undefined): boolean {
  if (!project || project.final_approved) return false;
  return Boolean(
    project.script_approved &&
      project.rights_approved &&
      projectHasPreview(project) &&
      qcReportHasNoBlockers(project.qc_report),
  );
}

function projectHasPreview(project: ContentStudioProject): boolean {
  const record = project as Record<string, unknown>;
  return Boolean(
    record.preview_render ??
      record.preview ??
      record.preview_video ??
      record.preview_artifact ??
      record.preview_rendered,
  );
}

function qcReportHasNoBlockers(value: unknown): boolean {
  if (!isRecord(value)) return false;
  const blockers = value.blockers;
  if (Array.isArray(blockers)) return blockers.length === 0;
  const issues = value.issues;
  if (Array.isArray(issues)) {
    return !issues.some((issue) => isRecord(issue) && stringValue(issue.severity)?.toUpperCase() === "BLOCKER");
  }
  return false;
}

function requireProjectId(projectId: string): string {
  if (!projectId) throw new Error("project id is required");
  return projectId;
}

function projectAssets(project: ContentStudioProject | undefined): AssetSummary[] {
  const manifest = project?.asset_manifest;
  if (!isRecord(manifest)) return [];
  const rawAssets = manifest.assets;
  if (!Array.isArray(rawAssets)) return [];
  return rawAssets.flatMap((raw): AssetSummary[] => {
    if (!isRecord(raw)) return [];
    const assetId = stringValue(raw.asset_id) ?? stringValue(raw.id);
    if (!assetId) return [];
    return [
      {
        asset_id: assetId,
        title: stringValue(raw.title) ?? stringValue(raw.name) ?? assetId,
        kind: stringValue(raw.kind) ?? stringValue(raw.type) ?? "asset",
        rights_status: stringValue(raw.rights_status) ?? stringValue(raw.license_status) ?? "unknown",
      },
    ];
  });
}

function collectProjectEvidence(project: ContentStudioProject | undefined): EvidenceSummary[] {
  const graph = project?.evidence_graph;
  if (!isRecord(graph)) return [];
  const rawEvidence = graph.evidence ?? graph.evidences ?? graph.sources;
  if (!Array.isArray(rawEvidence)) return [];
  return rawEvidence.flatMap((raw): EvidenceSummary[] => {
    if (!isRecord(raw)) return [];
    const evidenceId = stringValue(raw.evidence_id) ?? stringValue(raw.id) ?? stringValue(raw.source_id);
    if (!evidenceId) return [];
    const sourceUrl = stringValue(raw.source_url) ?? stringValue(raw.url) ?? "";
    return [
      {
        evidence_id: evidenceId,
        title: stringValue(raw.title) ?? stringValue(raw.publisher) ?? sourceUrl ?? evidenceId,
        source_url: sourceUrl,
      },
    ];
  });
}

function parseEvidenceIdText(value: string): string[] {
  return Array.from(
    new Set(
      value
        .split(/[\s,，;；]+/u)
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  );
}

function valueSummary(value: unknown): string {
  if (!value) return "暂无产物";
  if (isRecord(value)) {
    const title = stringValue(value.title) ?? stringValue(value.summary) ?? stringValue(value.status);
    if (title) return title;
    const keys = Object.keys(value);
    return keys.length ? `${keys.slice(0, 4).join("、")} 等 ${keys.length} 个字段` : "已有结构化产物";
  }
  if (Array.isArray(value)) return `${value.length} 条记录`;
  return String(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
