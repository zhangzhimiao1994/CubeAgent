import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FormEvent, type ReactNode, useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { api, formatApiError, type ContentStudioProject, type ContentStudioProjectSummary } from "../api/client";

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

const PHASE_ACTIONS: Array<{ label: string; until: Stage; description: string }> = [
  { label: "1. 运行 Research / Fact Check", until: "FACT_CHECKED", description: "只完成调研、证据图和事实核验。" },
  { label: "2. 生成 Plan / Script", until: "SCRIPT_READY", description: "生成内容计划、Hook 和脚本，随后等待脚本批准。" },
  { label: "3. 生成 Storyboard / Assets", until: "ASSETS_READY", description: "脚本批准后生成分镜和素材，随后等待版权/素材审核。" },
  { label: "4. 生成 Voice / Timeline / Preview", until: "PREVIEW_RENDERED", description: "版权批准后生成配音、时间线和预览。" },
  { label: "5. 运行 Video QC", until: "QC_REVIEW", description: "对预览做质量检查，终片批准仍需人工确认。" },
];

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

type HistoryItem = {
  id: string;
  sequence: number;
  title: string;
  stage: string;
  status: string;
  summary: string;
  artifactRefs: string[];
  payload?: unknown;
};

const contentStudioProjectKey = (projectId: string) => ["content-studio-project", projectId] as const;
const contentStudioProjectsKey = ["content-studio-projects"] as const;
const RECENT_PROJECTS_STORAGE_KEY = "content_studio_recent_projects";

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
  const [recentProjects, setRecentProjects] = useState<ContentStudioProjectSummary[]>(() => loadRecentProjects());

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
    setRecentProjects((current) => rememberRecentProject(current, projectSummaryFromProject(project)));
    void queryClient.invalidateQueries({ queryKey: contentStudioProjectsKey });
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
  const projectsQuery = useQuery({
    queryKey: contentStudioProjectsKey,
    queryFn: () => api.contentStudioProjects(),
    refetchOnWindowFocus: false,
    retry: false,
  });
  const visibleProject = projectQuery.data?.project_id === projectId ? projectQuery.data : undefined;
  const activeProjectId = visibleProject?.project_id ?? "";
  const canApproveFinal = projectCanApproveFinal(visibleProject);

  useEffect(() => {
    if (!visibleProject) return;
    setRecentProjects((current) => rememberRecentProject(current, projectSummaryFromProject(visibleProject)));
  }, [visibleProject]);

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
  const projectSummaries = useMemo(
    () => mergeProjectSummaries(projectsQuery.data ?? [], recentProjects, visibleProject),
    [projectsQuery.data, recentProjects, visibleProject],
  );
  const claimEvidenceIds = useMemo(() => parseEvidenceIdText(claimEvidenceIdsText), [claimEvidenceIdsText]);
  const claimNeedsEvidence = claimStatus === "supported" || claimStatus === "partially_supported";
  const canUpdateClaim = Boolean(activeProjectId && claimId.trim() && (!claimNeedsEvidence || claimEvidenceIds.length > 0));
  const statusSummary = useMemo(() => projectStatusSummary(visibleProject), [visibleProject]);
  const historyItems = useMemo(() => collectProjectHistory(visibleProject), [visibleProject]);
  const stageLocked = useMemo(() => projectStageLocks(visibleProject), [visibleProject]);
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

      <div className="content-studio-workspace">
        <main className="content-studio-main">
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

          <StageSection number={1} title="创建项目" status={visibleProject?.status ?? "DRAFT"}>
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

            <section className="content-studio-overview" aria-label="项目概览">
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
          </StageSection>

          <StageSection number={2} title="Research / Fact Check" status={visibleProject?.fact_check_report ? "已完成" : "待运行"} locked={stageLocked.research}>
            <div className="toolbar">
              <button
                type="button"
                className="secondary-action"
                disabled={stageLocked.research || anyActionPending}
                title={PHASE_ACTIONS[0].description}
                onClick={() => runProject.mutate({ id: requireProjectId(activeProjectId), until: PHASE_ACTIONS[0].until })}
              >
                {PHASE_ACTIONS[0].label}
              </button>
            </div>
            <div className="resource-list content-studio-grid">
              <ResearchPanel value={visibleProject?.research_bundle} />
              <EvidencePanel value={visibleProject?.evidence_graph} />
              <StudioPanel title="Fact Check" value={visibleProject?.fact_check_report} />
            </div>
            <div className="form-grid content-studio-inline-form">
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
                <input id="claim-evidence-ids" value={claimEvidenceIdsText} placeholder="EV001, EV002" onChange={(event) => setClaimEvidenceIdsText(event.target.value)} />
              </label>
            </div>
            <div className="content-studio-evidence-list" role="list" aria-label="事实证据列表">
              {evidenceItems.length ? (
                evidenceItems.map((evidence) => (
                  <label key={evidence.evidence_id} className="content-studio-asset-row">
                    <input type="checkbox" checked={claimEvidenceIds.includes(evidence.evidence_id)} onChange={() => toggleEvidence(evidence.evidence_id)} />
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
          </StageSection>

          <StageSection number={3} title="Plan / Script" status={visibleProject?.script ? "脚本可审" : "待运行"} locked={stageLocked.script}>
            <div className="toolbar">
              <button
                type="button"
                className="secondary-action"
                disabled={stageLocked.script || anyActionPending}
                title={PHASE_ACTIONS[1].description}
                onClick={() => runProject.mutate({ id: requireProjectId(activeProjectId), until: PHASE_ACTIONS[1].until })}
              >
                {PHASE_ACTIONS[1].label}
              </button>
            </div>
            <div className="resource-list content-studio-grid">
              <StudioPanel title="Plan" value={visibleProject?.content_plan} />
              <ScriptPanel value={visibleProject?.script} />
            </div>
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
          </StageSection>

          <StageSection number={4} title="Storyboard / Assets" status={visibleProject?.asset_manifest ? "素材可审" : "待运行"} locked={stageLocked.assets}>
            <div className="toolbar">
              <button
                type="button"
                className="secondary-action"
                disabled={stageLocked.assets || anyActionPending}
                title={PHASE_ACTIONS[2].description}
                onClick={() => runProject.mutate({ id: requireProjectId(activeProjectId), until: PHASE_ACTIONS[2].until })}
              >
                {PHASE_ACTIONS[2].label}
              </button>
            </div>
            <div className="resource-list content-studio-grid">
              <StoryboardPanel value={visibleProject?.storyboard} />
              <AssetsPanel value={visibleProject?.asset_manifest} assets={assets} />
            </div>
            <div className="form-grid content-studio-inline-form">
              <label htmlFor="storyboard-revision">
                分镜修改
                <input id="storyboard-revision" value={storyboardRevision} onChange={(event) => setStoryboardRevision(event.target.value)} />
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
            </div>
            <div className="toolbar">
              <button type="button" className="secondary-action" disabled={!activeProjectId || anyActionPending} onClick={() => reviseStoryboard.mutate({ id: requireProjectId(activeProjectId), instruction: storyboardRevision })}>
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
              <button type="button" className="secondary-action" disabled={!activeProjectId || anyActionPending} onClick={() => retryContentStage.mutate({ id: requireProjectId(activeProjectId), stage: retryStage })}>
                {retryContentStage.isPending ? "阶段重试中..." : "重试所选阶段"}
              </button>
            </div>
            <section aria-label="版权批准">
              <h3>按素材批准版权</h3>
              <p className="content-studio-helper">只勾选已经核验来源和授权状态的素材。这里不会自动整包通过。</p>
              <div className="content-studio-asset-list" role="list" aria-label="素材版权列表">
                {assets.length ? (
                  assets.map((asset) => (
                    <label key={asset.asset_id} className="content-studio-asset-row">
                      <input type="checkbox" checked={selectedAssetIds.includes(asset.asset_id)} onChange={() => toggleAsset(asset.asset_id)} />
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
                disabled={!activeProjectId || Boolean(visibleProject?.rights_approved) || selectedAssetIds.length === 0 || !rightsNote.trim() || anyActionPending}
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
          </StageSection>

          <StageSection number={5} title="Voice / Timeline / Preview" status={projectHasPreview(visibleProject) ? "已有预览" : "待运行"} locked={stageLocked.preview}>
            <div className="toolbar">
              <button type="button" className="secondary-action" disabled={stageLocked.preview || anyActionPending} title={PHASE_ACTIONS[3].description} onClick={() => runProject.mutate({ id: requireProjectId(activeProjectId), until: PHASE_ACTIONS[3].until })}>
                {PHASE_ACTIONS[3].label}
              </button>
              <button type="button" className="secondary-action" disabled={stageLocked.preview || anyActionPending} onClick={() => renderPreview.mutate(requireProjectId(activeProjectId))}>
                {renderPreview.isPending ? "预览渲染中..." : "渲染预览"}
              </button>
            </div>
            <div className="resource-list content-studio-grid">
              <StudioPanel title="Voice" value={visibleProject?.voice_track} />
              <TimelinePanel value={visibleProject?.timeline} />
            </div>
          </StageSection>

          <StageSection number={6} title="Video QC / Final" status={visibleProject?.qc_report ? "待终审" : "待质检"} locked={stageLocked.qc}>
            <div className="toolbar">
              <button type="button" className="secondary-action" disabled={stageLocked.qc || anyActionPending} title={PHASE_ACTIONS[4].description} onClick={() => runProject.mutate({ id: requireProjectId(activeProjectId), until: PHASE_ACTIONS[4].until })}>
                {PHASE_ACTIONS[4].label}
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
              <span className="content-studio-helper">终片批准需要脚本批准、版权批准、已有预览和无 BLOCKER 的 QC 报告。</span>
            </div>
            <div className="resource-list content-studio-grid">
              <QcPanel value={visibleProject?.qc_report} />
              <TimelinePanel value={visibleProject?.timeline} finalOnly />
            </div>
            <section aria-label="项目历史">
              <h3>项目历史与产物记录</h3>
              <p className="content-studio-helper">按发生顺序保留调研、事实链、脚本、素材、配音、渲染、QC、失败和重试记录。</p>
              <div className="content-studio-history" role="list">
                {historyItems.length ? (
                  historyItems.map((item) => (
                    <article key={item.id} className="content-studio-history-item" role="listitem">
                      <div>
                        <span className="content-studio-history-index">{item.sequence}</span>
                        <strong>{item.title}</strong>
                        <small>
                          {item.stage} · {item.status}
                        </small>
                      </div>
                      <p>{item.summary}</p>
                      {item.artifactRefs.length ? <p className="content-studio-history-refs">关联产物：{item.artifactRefs.join("、")}</p> : null}
                      {item.payload ? (
                        <details>
                          <summary>查看历史详情</summary>
                          <pre className="json-preview">{JSON.stringify(item.payload, null, 2)}</pre>
                        </details>
                      ) : null}
                    </article>
                  ))
                ) : (
                  <p>暂无历史记录。</p>
                )}
              </div>
            </section>
          </StageSection>
        </main>

        <ProjectHistoryRail
          projects={projectSummaries}
          activeProjectId={activeProjectId || projectId}
          loading={projectsQuery.isLoading}
          onOpen={(nextProjectId) => openProjectId(nextProjectId)}
        />
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

function StageSection({
  children,
  locked = false,
  number,
  status,
  title,
}: {
  children: ReactNode;
  locked?: boolean;
  number: number;
  status: string;
  title: string;
}) {
  return (
    <section
      className={locked ? "section-band content-studio-stage content-studio-stage-locked" : "section-band content-studio-stage"}
      aria-label={`${number} ${title}`}
    >
      <div className="content-studio-stage-header">
        <div>
          <p className="eyebrow">Stage {number}</p>
          <h3>{title}</h3>
        </div>
        <span className={locked ? "status-pill status-pill-muted" : "status-pill status-pill-success"}>
          {locked ? "未解锁" : status}
        </span>
      </div>
      {children}
    </section>
  );
}

function ProjectHistoryRail({
  activeProjectId,
  loading,
  onOpen,
  projects,
}: {
  activeProjectId: string;
  loading: boolean;
  onOpen: (projectId: string) => void;
  projects: ContentStudioProjectSummary[];
}) {
  return (
    <aside className="content-studio-history-rail" aria-label="Content Studio 项目历史">
      <div className="content-studio-history-rail-header">
        <p className="eyebrow">History</p>
        <h3>历史项目</h3>
      </div>
      {loading ? <p className="content-studio-helper">正在加载项目历史...</p> : null}
      <div className="content-studio-project-list" role="list">
        {projects.length ? (
          projects.map((project) => (
            <button
              key={project.project_id}
              type="button"
              className={
                project.project_id === activeProjectId
                  ? "content-studio-project-item content-studio-project-item-active"
                  : "content-studio-project-item"
              }
              onClick={() => onOpen(project.project_id)}
            >
              <strong>{project.title}</strong>
              <span>{project.status}</span>
              <small>{project.project_id}</small>
            </button>
          ))
        ) : (
          <p className="content-studio-helper">暂无历史项目。</p>
        )}
      </div>
    </aside>
  );
}

function ResearchPanel({ value }: { value: unknown }) {
  const coverage = arrayValue(recordValue(value)?.source_coverage);
  const candidates = arrayValue(recordValue(value)?.source_candidates);
  const evidence = arrayValue(recordValue(value)?.evidence);
  const questions = arrayValue(recordValue(value)?.questions);
  return (
    <article>
      <p className="eyebrow">{value ? "ready" : "pending"}</p>
      <h3>Research</h3>
      <p>{value ? `${evidence.length} 条证据 · ${coverage.length} 类来源覆盖 · ${candidates.length} 个候选源` : "等待调研"}</p>
      {questions.length ? (
        <ul className="content-studio-output-list">
          {questions.map((item, index) => {
            const row = recordValue(item);
            return <li key={`${stringValue(row?.question_id) ?? "question"}-${index}`}>{stringValue(row?.text) ?? stringValue(row?.question) ?? valueSummary(row)}</li>;
          })}
        </ul>
      ) : null}
      {coverage.length ? (
        <div className="content-studio-source-coverage" aria-label="Research 来源覆盖">
          {coverage.map((item, index) => {
            const row = recordValue(item);
            const sourceType = stringValue(row?.source_type) ?? `source-${index + 1}`;
            const status = stringValue(row?.status) ?? "unknown";
            const required = Boolean(row?.required);
            return (
              <span key={`${sourceType}-${index}`} className={status === "covered" ? "status-pill status-pill-success" : "status-pill status-pill-muted"}>
                {sourceType} · {required ? "必需" : "补充"} · {status}
              </span>
            );
          })}
        </div>
      ) : null}
      {candidates.length ? (
        <details>
          <summary>查看候选源池</summary>
          <ul className="content-studio-source-list">
            {candidates.slice(0, 16).map((item, index) => {
              const row = recordValue(item);
              return (
                <li key={`${stringValue(row?.source_url) ?? "source"}-${index}`}>
                  <strong>{stringValue(row?.source_type) ?? "source"}</strong>
                  <span>{stringValue(row?.source_url) ?? "unknown"}</span>
                </li>
              );
            })}
          </ul>
        </details>
      ) : null}
      {value ? (
        <details>
          <summary>查看结构化详情</summary>
          <pre className="json-preview">{JSON.stringify(value, null, 2)}</pre>
        </details>
      ) : null}
    </article>
  );
}

function EvidencePanel({ value }: { value: unknown }) {
  const row = recordValue(value);
  const evidence = arrayValue(row?.evidence ?? row?.evidences ?? row?.sources);
  const claims = arrayValue(row?.claims);
  return (
    <article>
      <p className="eyebrow">{value ? "ready" : "pending"}</p>
      <h3>Evidence</h3>
      <p>{value ? `${claims.length} 条 Claim · ${evidence.length} 条 Evidence` : "等待证据图"}</p>
      {claims.length ? (
        <ul className="content-studio-output-list">
          {claims.slice(0, 6).map((item, index) => {
            const claim = recordValue(item);
            return (
              <li key={`${stringValue(claim?.claim_id) ?? "claim"}-${index}`}>
                <strong>{stringValue(claim?.claim_id) ?? `CL${index + 1}`}</strong>
                <span>{stringValue(claim?.text) ?? valueSummary(claim)}</span>
              </li>
            );
          })}
        </ul>
      ) : null}
      {value ? (
        <details>
          <summary>查看结构化详情</summary>
          <pre className="json-preview">{JSON.stringify(value, null, 2)}</pre>
        </details>
      ) : null}
    </article>
  );
}

function ScriptPanel({ value }: { value: unknown }) {
  const row = recordValue(value);
  const hooks = stringArray(row?.hooks);
  const segments = arrayValue(row?.segments);
  const subtitles = stringArray(row?.subtitle_lines);
  return (
    <article>
      <p className="eyebrow">{value ? "ready" : "pending"}</p>
      <h3>Script</h3>
      <p>{value ? `${hooks.length} 个 Hook · ${segments.length} 段脚本 · ${subtitles.length} 条字幕` : "等待脚本"}</p>
      {hooks.length ? (
        <section className="content-studio-output-block">
          <strong>Hooks</strong>
          <ul className="content-studio-output-list">
            {hooks.map((hook, index) => (
              <li key={`${hook}-${index}`}>{hook}</li>
            ))}
          </ul>
        </section>
      ) : null}
      {segments.length ? (
        <section className="content-studio-output-block">
          <strong>脚本段落</strong>
          <ul className="content-studio-output-list">
            {segments.map((item, index) => {
              const segment = recordValue(item);
              return <li key={`${stringValue(segment?.segment_id) ?? "segment"}-${index}`}>{stringValue(segment?.text) ?? valueSummary(segment)}</li>;
            })}
          </ul>
        </section>
      ) : null}
      {value ? (
        <details>
          <summary>查看结构化详情</summary>
          <pre className="json-preview">{JSON.stringify(value, null, 2)}</pre>
        </details>
      ) : null}
    </article>
  );
}

function StoryboardPanel({ value }: { value: unknown }) {
  const row = recordValue(value);
  const shots = arrayValue(row?.shots);
  return (
    <article>
      <p className="eyebrow">{value ? "ready" : "pending"}</p>
      <h3>Storyboard</h3>
      <p>{value ? `${shots.length} 个镜头` : "等待分镜"}</p>
      {shots.length ? (
        <ul className="content-studio-output-list">
          {shots.map((item, index) => {
            const shot = recordValue(item);
            return (
              <li key={`${stringValue(shot?.shot_id) ?? "shot"}-${index}`}>
                <strong>{stringValue(shot?.shot_id) ?? `Shot ${index + 1}`}</strong>
                <span>{stringValue(shot?.shot_type) ?? "shot"} · {numberValue(shot?.duration_ms) ?? 0}ms · {stringValue(shot?.overlay) ?? "无叠加"}</span>
              </li>
            );
          })}
        </ul>
      ) : null}
      {value ? (
        <details>
          <summary>查看结构化详情</summary>
          <pre className="json-preview">{JSON.stringify(value, null, 2)}</pre>
        </details>
      ) : null}
    </article>
  );
}

function AssetsPanel({ assets, value }: { assets: AssetSummary[]; value: unknown }) {
  return (
    <article>
      <p className="eyebrow">{value ? "ready" : "pending"}</p>
      <h3>Assets</h3>
      <p>{assets.length ? `${assets.length} 个素材待管理` : "等待素材"}</p>
      {assets.length ? (
        <ul className="content-studio-output-list">
          {assets.map((asset) => (
            <li key={asset.asset_id}>
              <strong>{asset.title}</strong>
              <span>{asset.asset_id} · {asset.kind} · {asset.rights_status}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {value ? (
        <details>
          <summary>查看结构化详情</summary>
          <pre className="json-preview">{JSON.stringify(value, null, 2)}</pre>
        </details>
      ) : null}
    </article>
  );
}

function TimelinePanel({ finalOnly = false, value }: { finalOnly?: boolean; value: unknown }) {
  const row = recordValue(value);
  const preview = stringValue(row?.preview_artifact_id);
  const final = stringValue(row?.final_artifact_id);
  return (
    <article>
      <p className="eyebrow">{value ? "ready" : "pending"}</p>
      <h3>{finalOnly ? "Final Render" : "Timeline"}</h3>
      <p>{value ? `${numberValue(row?.width) ?? 0}×${numberValue(row?.height) ?? 0} · ${numberValue(row?.duration_ms) ?? 0}ms` : "等待时间线"}</p>
      {!finalOnly && preview ? <p className="content-studio-media-path">{preview}</p> : null}
      {final ? <p className="content-studio-media-path">{final}</p> : null}
      {value ? (
        <details>
          <summary>查看结构化详情</summary>
          <pre className="json-preview">{JSON.stringify(value, null, 2)}</pre>
        </details>
      ) : null}
    </article>
  );
}

function QcPanel({ value }: { value: unknown }) {
  const row = recordValue(value);
  const blockers = stringArray(row?.blockers);
  const majors = stringArray(row?.majors);
  const minors = stringArray(row?.minors);
  const checkedItems = stringArray(row?.checked_items);
  const summary = stringValue(row?.summary);
  return (
    <article>
      <p className="eyebrow">{value ? "ready" : "pending"}</p>
      <h3>QC</h3>
      <p>{value ? `${blockers.length} BLOCKER · ${majors.length} MAJOR · ${minors.length} MINOR` : "等待 Video QC"}</p>
      {summary ? <p>{summary}</p> : null}
      {checkedItems.length ? (
        <ul className="content-studio-output-list">
          {checkedItems.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
      {value ? (
        <details>
          <summary>查看结构化详情</summary>
          <pre className="json-preview">{JSON.stringify(value, null, 2)}</pre>
        </details>
      ) : null}
    </article>
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

function projectStageLocks(project: ContentStudioProject | undefined): {
  research: boolean;
  script: boolean;
  assets: boolean;
  preview: boolean;
  qc: boolean;
} {
  return {
    research: !project,
    script: !project?.fact_check_report,
    assets: !project?.script_approved,
    preview: !project?.rights_approved,
    qc: !project || !projectHasPreview(project),
  };
}

function loadRecentProjects(): ContentStudioProjectSummary[] {
  try {
    const raw = window.localStorage.getItem(RECENT_PROJECTS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.flatMap((item): ContentStudioProjectSummary[] => {
      if (!isRecord(item)) return [];
      const projectId = stringValue(item.project_id);
      const title = stringValue(item.title);
      const status = stringValue(item.status);
      if (!projectId || !title || !status) return [];
      return [
        {
          project_id: projectId,
          title,
          topic: stringValue(item.topic) ?? "",
          status,
          revision: numberValue(item.revision) ?? 0,
          execution_mode: stringValue(item.execution_mode) === "production" ? "production" : "demo",
          updated_at: stringValue(item.updated_at),
        },
      ];
    });
  } catch {
    return [];
  }
}

function rememberRecentProject(
  current: ContentStudioProjectSummary[],
  incoming: ContentStudioProjectSummary,
): ContentStudioProjectSummary[] {
  const next = mergeProjectSummaries([incoming], current).slice(0, 20);
  try {
    window.localStorage.setItem(RECENT_PROJECTS_STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Browsers can reject local storage in private contexts; server history remains authoritative.
  }
  return next;
}

function mergeProjectSummaries(
  primary: ContentStudioProjectSummary[],
  fallback: ContentStudioProjectSummary[],
  current?: ContentStudioProject,
): ContentStudioProjectSummary[] {
  const merged = new Map<string, ContentStudioProjectSummary>();
  for (const item of [...fallback, ...primary]) {
    merged.set(item.project_id, item);
  }
  if (current) merged.set(current.project_id, projectSummaryFromProject(current));
  return Array.from(merged.values()).sort((left, right) => {
    const leftTime = Date.parse(left.updated_at ?? "");
    const rightTime = Date.parse(right.updated_at ?? "");
    if (Number.isFinite(leftTime) || Number.isFinite(rightTime)) {
      return (Number.isFinite(rightTime) ? rightTime : 0) - (Number.isFinite(leftTime) ? leftTime : 0);
    }
    return right.revision - left.revision;
  });
}

function projectSummaryFromProject(project: ContentStudioProject): ContentStudioProjectSummary {
  return {
    project_id: project.project_id,
    title: project.title,
    topic: project.topic,
    status: project.status,
    revision: project.revision,
    execution_mode: project.execution_mode,
    updated_at: new Date().toISOString(),
  };
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
      project.status === "QC_REVIEW" &&
      projectHasPreview(project) &&
      qcReportHasNoBlockers(project.qc_report),
  );
}

function projectHasPreview(project: ContentStudioProject | undefined): boolean {
  if (!project) return false;
  const record = project as Record<string, unknown>;
  const timeline = recordValue(project.timeline);
  return Boolean(
    record.preview_render ??
      record.preview ??
      record.preview_video ??
      record.preview_artifact ??
      record.preview_rendered ??
      timeline?.preview_artifact_id,
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

function collectProjectHistory(project: ContentStudioProject | undefined): HistoryItem[] {
  if (!project) return [];
  const items = [
    ...collectProjectEvents(project),
    ...collectProviderAttemptHistory(project),
  ];
  const keyed = new Map<string, HistoryItem>();
  for (const item of items) {
    keyed.set(item.id, item);
  }
  if (!keyed.size) {
    for (const item of collectCurrentArtifactHistory(project)) {
      keyed.set(item.id, item);
    }
  }
  return Array.from(keyed.values()).sort((left, right) => left.sequence - right.sequence);
}

function collectProjectEvents(project: ContentStudioProject): HistoryItem[] {
  const record = project as Record<string, unknown>;
  const rawEvents = record.project_events ?? record.events ?? record.history;
  if (!Array.isArray(rawEvents)) return [];
  return rawEvents.flatMap((raw, index): HistoryItem[] => {
    if (!isRecord(raw)) return [];
    const id = stringValue(raw.event_id) ?? stringValue(raw.id) ?? `event-${index + 1}`;
    return [
      {
        id,
        sequence: numberValue(raw.sequence) ?? index + 1,
        title: stringValue(raw.title) ?? stringValue(raw.kind) ?? "项目事件",
        stage: stringValue(raw.stage) ?? "project",
        status: stringValue(raw.status) ?? "",
        summary: stringValue(raw.summary) ?? valueSummary(raw.payload),
        artifactRefs: stringArray(raw.artifact_refs),
        payload: raw.payload ?? raw,
      },
    ];
  });
}

function collectProviderAttemptHistory(project: ContentStudioProject): HistoryItem[] {
  const record = project as Record<string, unknown>;
  const rawAttempts = record.provider_attempts;
  if (!Array.isArray(rawAttempts)) return [];
  const offset = collectProjectEvents(project).length + 1;
  return rawAttempts.flatMap((raw, index): HistoryItem[] => {
    if (!isRecord(raw)) return [];
    const stage = stringValue(raw.stage) ?? "provider";
    const status = stringValue(raw.status) ?? "";
    const idempotencyKey = stringValue(raw.idempotency_key) ?? `${stage}-${index + 1}`;
    const resultHash = stringValue(raw.result_hash) ?? "";
    const eventId = `provider-${idempotencyKey}-${status}-${resultHash}`;
    return [
      {
        id: eventId,
        sequence: offset + index,
        title: `${stage} Provider 调用`,
        stage,
        status,
        summary: providerAttemptSummary(raw, stage, status),
        artifactRefs: stringArray(raw.provider_task_id ? [raw.provider_task_id] : []),
        payload: raw,
      },
    ];
  });
}

function collectCurrentArtifactHistory(project: ContentStudioProject): HistoryItem[] {
  const stages: Array<[string, string, unknown]> = [
    ["Research 调研", "research", project.research_bundle],
    ["Evidence 事实链", "evidence", project.evidence_graph],
    ["Fact Check 核验", "fact_check", project.fact_check_report],
    ["Content Plan 计划", "plan", project.content_plan],
    ["Script 脚本", "script", project.script],
    ["Storyboard 分镜", "storyboard", project.storyboard],
    ["Assets 素材", "assets", project.asset_manifest],
    ["Voice 配音", "voice", project.voice_track],
    ["Timeline 时间线", "timeline", project.timeline],
    ["QC 质检", "qc", project.qc_report],
  ];
  return stages.flatMap(([title, stage, value], index): HistoryItem[] => {
    if (!value) return [];
    return [
      {
        id: `current-${stage}`,
        sequence: index + 1,
        title,
        stage,
        status: "current",
        summary: valueSummary(value),
        artifactRefs: artifactRefsFromValue(value),
        payload: value,
      },
    ];
  });
}

function providerAttemptSummary(raw: Record<string, unknown>, stage: string, status: string): string {
  const errorCode = stringValue(raw.error_code);
  const providerTaskId = stringValue(raw.provider_task_id);
  const resultHash = stringValue(raw.result_hash);
  if (status === "completed") return `${stage} 调用完成：${providerTaskId ?? resultHash ?? "已完成"}`;
  if (errorCode) return `${stage} 调用失败：${errorCode}`;
  return `${stage} 调用状态：${status || "unknown"}`;
}

function artifactRefsFromValue(value: unknown): string[] {
  if (!isRecord(value)) return [];
  const refs = [
    stringValue(value.audio_artifact_id),
    stringValue(value.preview_artifact_id),
    stringValue(value.final_artifact_id),
    stringValue(value.asset_id),
    stringValue(value.claim_id),
    stringValue(value.evidence_id),
  ].filter((item): item is string => Boolean(item));
  const nested = Object.values(value).flatMap((item) => {
    if (!Array.isArray(item)) return [];
    return item.flatMap((entry) => {
      if (!isRecord(entry)) return [];
      return [
        stringValue(entry.asset_id),
        stringValue(entry.claim_id),
        stringValue(entry.evidence_id),
        stringValue(entry.shot_id),
        stringValue(entry.segment_id),
      ].filter((inner): inner is string => Boolean(inner));
    });
  });
  return Array.from(new Set([...refs, ...nested])).slice(0, 12);
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

function recordValue(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean);
}
