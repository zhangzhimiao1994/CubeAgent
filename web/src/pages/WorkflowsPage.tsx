import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FormEvent, useMemo, useState } from "react";

import { useNavSection } from "../app/navSections";
import { api, formatApiError, type WorkflowResource } from "../api/client";
import { textContains } from "../components/TableTools";

const WORKFLOW_PRESETS: Array<
  Omit<WorkflowResource, "allow_main_agent_override" | "allow_temporary_agents" | "temporary_agent_policy"> & {
    description: string;
    suggested_roles: string[];
  }
> = [
  {
    id: "custom-workflow",
    name: "自定义协作预设",
    enabled: true,
    mode: "auto",
    task_type: "",
    suggested_roles: [],
    agent_ids: [],
    objective: "",
    steps: [],
    deliverables: [],
    role_selection_policy: "根据任务目标选择真正需要的角色，不按模板固定派单对象。",
    decision_policy: "如果角色意见冲突，主 Agent 根据用户目标、证据质量、风险和可交付性做最终裁决；无法判断时询问用户。",
    description: "从空白配置开始，适合你自己定义新的任务场景、参与角色和协作策略。",
  },
  {
    id: "short-video-dispatch",
    name: "短视频生产派单",
    enabled: true,
    mode: "dispatch",
    task_type: "短视频、脚本、内容生产",
    suggested_roles: ["director", "copywriter", "editor", "critic"],
    agent_ids: [],
    objective: "把短视频任务拆给导演、文案、剪辑师和审查员，输出可执行脚本和成片建议。",
    steps: ["导演确定选题角度和结构", "文案生成脚本与标题", "剪辑师拆镜头和节奏", "审查员检查风险和可交付性"],
    deliverables: ["短视频脚本", "镜头/剪辑建议", "标题与封面建议", "审查意见"],
    role_selection_policy: "优先选择导演、文案、剪辑师；涉及数据或商业判断时追加市场分析师；最终必须由审查员收口。",
    decision_policy: "如果创意方向冲突，优先选择更符合目标受众、平台限制和可拍摄性的方案。",
    description: "适合抖音、视频号、B站脚本和内容生产。",
  },
  {
    id: "software-dispatch",
    name: "软件工程派单",
    enabled: true,
    mode: "dispatch",
    task_type: "代码、部署、故障修复",
    suggested_roles: ["engineer", "qa-tester", "ops-engineer", "security-reviewer"],
    agent_ids: [],
    objective: "把工程任务拆给工程师、测试、运维和安全审查角色，减少生产事故。",
    steps: ["工程师定位实现方案", "测试工程师设计验证", "运维工程师检查部署影响", "安全审查员检查权限和密钥风险"],
    deliverables: ["实现方案", "验证清单", "部署注意事项", "安全审查意见"],
    role_selection_policy: "代码任务不要默认分给内容角色；涉及部署必须包含运维，涉及外部输入/密钥/权限必须包含安全审查。",
    decision_policy: "如果速度和安全冲突，生产环境优先安全和可回滚；无法确认时询问用户。",
    description: "适合代码修复、部署脚本、生产故障和架构调整。",
  },
  {
    id: "finance-analysis-dispatch",
    name: "财经分析派单",
    enabled: true,
    mode: "dispatch",
    task_type: "经济、金融、商业分析",
    suggested_roles: ["economic-analyst", "finance-analyst", "researcher", "critic"],
    agent_ids: [],
    objective: "把财经类问题拆给经济分析、财务分析、研究和审查角色，输出事实与推断分离的结论。",
    steps: ["研究员整理事实和数据来源", "经济分析师分析宏观和行业", "财务分析师分析成本/现金流/利润", "审查员标注风险和不确定性"],
    deliverables: ["事实清单", "分析结论", "风险提示", "可执行建议"],
    role_selection_policy: "财经任务必须包含研究员和审查员；涉及公司经营加入财务分析师；涉及宏观趋势加入经济分析师。",
    decision_policy: "如果结论冲突，优先采用证据来源更清楚、假设更少、风险披露更完整的结论。",
    description: "适合宏观经济、行业、公司、预算和投资相关分析。",
  },
  {
    id: "creative-design-discuss",
    name: "创意设计讨论",
    enabled: true,
    mode: "discuss",
    task_type: "艺术设计、品牌、创意方向",
    suggested_roles: ["director", "copywriter", "market-analyst", "critic"],
    agent_ids: [],
    objective: "围绕创意方向进行多角色讨论，避免把艺术设计任务交给不相关的工程角色。",
    steps: ["提出创意方向", "市场分析师判断受众和定位", "文案生成表达方案", "审查员检查一致性和风险", "主 Agent 裁决"],
    deliverables: ["创意方向", "视觉/文案建议", "受众判断", "最终方案"],
    role_selection_policy: "艺术设计默认不选择工程师；除非任务明确要求实现落地或网页代码，才加入工程师。",
    decision_policy: "如果美学与转化冲突，按用户目标决定：品牌表达优先一致性，营销素材优先转化。",
    description: "适合品牌、视觉方向、内容创意和营销表达。",
  },
  {
    id: "hybrid-production",
    name: "混合生产流程",
    enabled: true,
    mode: "hybrid",
    task_type: "复杂生产任务",
    suggested_roles: ["product-manager", "researcher", "engineer", "qa-tester", "critic"],
    agent_ids: [],
    objective: "先讨论方案，再派单执行，最后审查收口。",
    steps: ["讨论并确定方案", "按专业角色分配执行", "生成产物", "审查和修订"],
    deliverables: ["方案", "执行产物", "审查报告"],
    role_selection_policy: "根据任务目标选择角色，不按模式固定角色；需要内容就选内容角色，需要工程就选工程角色，需要财经就选分析角色。",
    decision_policy: "讨论阶段求异，执行阶段求稳，收口阶段以质量和可交付性为准。",
    description: "适合需要先定方向、再分工执行、最后验收的复杂任务。",
  },
];

function toggle(list: string[], value: string) {
  return list.includes(value) ? list.filter((item) => item !== value) : [...list, value];
}

const DEFAULT_WORKFLOW_STEPS = [
  "主 Agent 判断任务意图和协作强度",
  "按预设角色池派发必要子 Agent",
  "汇总子 Agent 输出并形成最终回复",
];

const DEFAULT_WORKFLOW_DELIVERABLES = ["最终回复", "必要时附子 Agent 摘要"];

function workflowStrategySummary(
  workflow: Pick<WorkflowResource, "objective" | "role_selection_policy" | "decision_policy">,
) {
  return (
    workflow.objective?.trim() ||
    workflow.role_selection_policy?.trim() ||
    workflow.decision_policy?.trim() ||
    "由主 Agent 根据当前任务自动判断协作方式。"
  );
}

function workflowStatus(workflow: WorkflowResource) {
  return workflow.enabled ? "已启用" : "已停用";
}

function workflowMode(workflow: WorkflowResource) {
  return workflow.mode ?? "auto";
}

function workflowRoles(workflow: WorkflowResource) {
  return (workflow.agent_ids ?? []).join(", ") || "未固定";
}

function workflowSearchText(workflow: WorkflowResource) {
  return [
    workflow.id,
    workflow.name,
    workflowStatus(workflow),
    workflowMode(workflow),
    workflow.task_type ?? "",
    workflowRoles(workflow),
    workflow.objective ?? "",
    workflow.role_selection_policy ?? "",
    workflow.decision_policy ?? "",
    ...(workflow.steps ?? []),
    ...(workflow.deliverables ?? []),
  ].join(" ");
}

function matchesWorkflowSearch(workflow: WorkflowResource, query: string) {
  return textContains(workflowSearchText(workflow), query);
}

export function WorkflowsPage() {
  const queryClient = useQueryClient();
  const { navTargetProps } = useNavSection();
  const workflows = useQuery({ queryKey: ["workflows"], queryFn: () => api.workflows() });
  const agents = useQuery({ queryKey: ["agents"], queryFn: () => api.agents() });
  const [presetId, setPresetId] = useState<string>(WORKFLOW_PRESETS[0].id);
  const preset = WORKFLOW_PRESETS.find((item) => item.id === presetId) ?? WORKFLOW_PRESETS[0];
  const [workflowId, setWorkflowId] = useState<string>(preset.id);
  const [name, setName] = useState<string>(preset.name);
  const [enabled, setEnabled] = useState(true);
  const [mode, setMode] = useState<NonNullable<WorkflowResource["mode"]>>(preset.mode ?? "dispatch");
  const [taskType, setTaskType] = useState<string>(preset.task_type ?? "");
  const [agentIds, setAgentIds] = useState<string[]>([]);
  const [strategySummary, setStrategySummary] = useState<string>(workflowStrategySummary(preset));
  const [message, setMessage] = useState<string | null>(null);
  const [workflowSearchTerm, setWorkflowSearchTerm] = useState("");
  const savedWorkflows = workflows.data ?? [];
  const savedAgents = agents.data ?? [];

  const saveWorkflow = useMutation({
    mutationFn: () => {
      const existingWorkflow = savedWorkflows.find((workflow) => workflow.id === workflowId.trim());
      const presetWorkflow = WORKFLOW_PRESETS.find((workflow) => workflow.id === presetId) ?? WORKFLOW_PRESETS[0];
      return api.createWorkflow({
        id: workflowId.trim(),
        name: name.trim(),
        enabled,
        mode,
        allow_main_agent_override: existingWorkflow?.allow_main_agent_override ?? false,
        allow_temporary_agents: existingWorkflow?.allow_temporary_agents ?? false,
        temporary_agent_policy: existingWorkflow?.temporary_agent_policy ?? null,
        task_type: taskType.trim(),
        role_selection_policy: strategySummary.trim(),
        agent_ids: agentIds,
        objective: strategySummary.trim(),
        steps: existingWorkflow?.steps ?? presetWorkflow.steps ?? DEFAULT_WORKFLOW_STEPS,
        deliverables: existingWorkflow?.deliverables ?? presetWorkflow.deliverables ?? DEFAULT_WORKFLOW_DELIVERABLES,
        decision_policy:
          existingWorkflow?.decision_policy ??
          presetWorkflow.decision_policy ??
          "主 Agent 按任务目标、用户反馈、证据质量和可交付性做最终裁决；无法可靠判断时再询问用户。",
      });
    },
    onSuccess: async () => {
      setMessage("协作预设已保存。它不会立即执行，只作为自动调度的参考。");
      await queryClient.invalidateQueries({ queryKey: ["workflows"] });
    },
  });

  const deleteWorkflow = useMutation({
    mutationFn: (id: string) => api.deleteWorkflow(id),
    onSuccess: async () => {
      setMessage("协作预设已删除。");
      await queryClient.invalidateQueries({ queryKey: ["workflows"] });
    },
  });

  function changePreset(nextId: string) {
    const next = WORKFLOW_PRESETS.find((item) => item.id === nextId) ?? WORKFLOW_PRESETS[0];
    setPresetId(next.id);
    setWorkflowId(next.id);
    setName(next.name);
    setEnabled(next.enabled);
    setMode(next.mode ?? "dispatch");
    setTaskType(next.task_type ?? "");
    setAgentIds([]);
    setStrategySummary(workflowStrategySummary(next));
    setMessage(null);
  }

  function applySuggestedRoles() {
    const available = new Set((agents.data ?? []).map((agent) => agent.id));
    setAgentIds(preset.suggested_roles.filter((id) => available.has(id)));
  }

  function editWorkflow(workflow: WorkflowResource) {
    setPresetId("custom-workflow");
    setWorkflowId(workflow.id);
    setName(workflow.name);
    setEnabled(workflow.enabled);
    setMode(workflow.mode ?? "auto");
    setTaskType(workflow.task_type ?? "");
    setAgentIds(workflow.agent_ids ?? []);
    setStrategySummary(workflowStrategySummary(workflow));
    setMessage(`已载入 ${workflow.name}，修改后点击保存。`);
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage(null);
    saveWorkflow.mutate();
  }

  function confirmDelete(workflow: { id: string; name: string }) {
    if (!window.confirm(`确定删除协作预设「${workflow.name}」吗？历史对话不会删除，但后续不能再选择它。`)) {
      return;
    }
    setMessage(null);
    deleteWorkflow.mutate(workflow.id);
  }

  const visibleWorkflows = useMemo(
    () =>
      savedWorkflows
        .filter((workflow) => matchesWorkflowSearch(workflow, workflowSearchTerm))
        .sort((left, right) => `${left.name} ${left.id}`.localeCompare(`${right.name} ${right.id}`)),
    [savedWorkflows, workflowSearchTerm],
  );

  if (workflows.isLoading || agents.isLoading) return <p>正在加载协作预设...</p>;
  if (workflows.isError) return <p role="alert">{formatApiError(workflows.error, "协作预设加载失败")}</p>;
  if (agents.isError) return <p role="alert">{formatApiError(agents.error, "Agent 列表加载失败")}</p>;

  return (
    <section>
      <p className="eyebrow">Workflow configuration</p>
      <h2>协作预设</h2>
      <p>
        这里保留少量高级协作预设，不再作为每轮对话必选配置。聊天页默认由主 Agent 自动判断模式和角色。
      </p>

      <div className="two-column">
        <form onSubmit={submit} aria-label="保存协作预设">
          <h3>新增或更新协作预设</h3>
          <label htmlFor="workflow-preset">
            预设模板
            <select id="workflow-preset" value={presetId} onChange={(event) => changePreset(event.target.value)}>
              {WORKFLOW_PRESETS.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <p className="field-help">
            {preset.description} 预设只负责快速填充；日常对话不需要手动选择它。
          </p>

          <div className="form-grid">
            <label htmlFor="workflow-id">
              预设 ID
              <input id="workflow-id" value={workflowId} onChange={(event) => setWorkflowId(event.target.value)} placeholder="例如 short-video-dispatch" required />
            </label>
            <label htmlFor="workflow-name">
              显示名称
              <input id="workflow-name" value={name} onChange={(event) => setName(event.target.value)} required />
            </label>
            <label htmlFor="workflow-task-type">
              任务类型 / 适用场景
              <input id="workflow-task-type" value={taskType} onChange={(event) => setTaskType(event.target.value)} placeholder="例如短视频、代码修复、财经分析" required />
            </label>
            <label htmlFor="workflow-mode">
              默认运行模式
              <select id="workflow-mode" value={mode} onChange={(event) => setMode(event.target.value as NonNullable<WorkflowResource["mode"]>)}>
                <option value="auto">自动识别</option>
                <option value="direct">直接执行</option>
                <option value="dispatch">派单式</option>
                <option value="discuss">讨论式</option>
                <option value="hybrid">混合式</option>
              </select>
            </label>
          </div>
          <p className="field-help">
            预设只描述某类任务的默认协作倾向；复杂步骤、交付物和裁决规则由系统默认生成，避免配置页过重。
          </p>

          <label htmlFor="workflow-strategy-summary" {...navTargetProps("execution")}>
            一句话策略
            <textarea
              id="workflow-strategy-summary"
              value={strategySummary}
              onChange={(event) => setStrategySummary(event.target.value)}
              placeholder="例如：内容生产任务先确定方向，再派发文案和审查角色收口。"
              required
            />
          </label>

          <fieldset {...navTargetProps("roles")}>
            <legend>默认参与角色</legend>
            <p className="field-help">
              这里只保存默认角色池。实际运行时主 Agent 仍会按任务目标决定是否使用、追加或跳过。
            </p>
            <button type="button" onClick={applySuggestedRoles}>
              使用模板建议角色
            </button>
            {savedAgents.length === 0 ? (
              <p className="field-help">还没有 Agent。可以先保存协作预设，稍后创建角色后回来选择。</p>
            ) : (
              savedAgents.map((agent) => (
                <label key={agent.id} className="inline-check">
                  <input
                    type="checkbox"
                    checked={agentIds.includes(agent.id)}
                    onChange={() => setAgentIds((current) => toggle(current, agent.id))}
                  />
                  {agent.name}（{agent.id}）
                </label>
              ))
            )}
          </fieldset>

          <label className="inline-check">
            <input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} />
            启用该预设
          </label>

          <button type="submit" disabled={saveWorkflow.isPending}>
            {saveWorkflow.isPending ? "正在保存..." : "保存协作预设"}
          </button>
          {message ? <p role="status">{message}</p> : null}
          {saveWorkflow.isError ? <p role="alert">{formatApiError(saveWorkflow.error, "协作预设保存失败")}</p> : null}
        </form>

        <article {...navTargetProps("review")}>
          <h3>使用边界</h3>
          <ol>
            <li>聊天页不再暴露本轮协作预设选择，默认自动运行。</li>
            <li>预设只影响默认倾向，不强制主 Agent 固定派单。</li>
            <li>如果自动判断不可靠，主 Agent 应给出原因并再询问用户。</li>
          </ol>
        </article>
      </div>

      <section aria-label="已保存协作预设" {...navTargetProps("list")}>
        <h3>已保存协作预设</h3>
        {savedWorkflows.length === 0 ? (
          <article>
            <h4>还没有协作预设</h4>
            <p>从上方选择模板并补全少量字段，保存后即可作为自动调度的参考。</p>
          </article>
        ) : (
          <>
            <div className="list-toolbar">
              <label>
                快速搜索协作预设
                <input
                  type="search"
                  aria-label="快速搜索协作预设"
                  value={workflowSearchTerm}
                  onChange={(event) => setWorkflowSearchTerm(event.currentTarget.value)}
                  placeholder="名称、ID、适用场景、角色或策略"
                />
              </label>
              <button type="button" className="secondary-action" onClick={() => setWorkflowSearchTerm("")}>
                清空搜索
              </button>
              <small>显示 {visibleWorkflows.length} / {savedWorkflows.length}</small>
            </div>
            {visibleWorkflows.length === 0 ? (
              <article>
                <h4>当前搜索没有匹配预设</h4>
                <p>清空搜索后可查看全部协作预设。</p>
              </article>
            ) : (
              <div className="workflow-preset-list" role="list" aria-label="已保存协作预设列表">
                {visibleWorkflows.map((workflow) => (
                  <article key={workflow.id} className="workflow-preset-card" role="listitem">
                    <div>
                      <span className={`status-pill ${workflow.enabled ? "status-pill-success" : "status-pill-muted"}`}>
                        {workflowStatus(workflow)}
                      </span>
                      <h4>{workflow.name}</h4>
                      <p>{workflowStrategySummary(workflow)}</p>
                    </div>
                    <dl>
                      <div>
                        <dt>ID</dt>
                        <dd>{workflow.id}</dd>
                      </div>
                      <div>
                        <dt>场景</dt>
                        <dd>{workflow.task_type || "未设置"}</dd>
                      </div>
                      <div>
                        <dt>模式</dt>
                        <dd>{workflowMode(workflow)}</dd>
                      </div>
                      <div>
                        <dt>角色</dt>
                        <dd>{workflowRoles(workflow)}</dd>
                      </div>
                    </dl>
                    <div className="card-actions">
                      <button type="button" onClick={() => editWorkflow(workflow)}>
                        编辑预设
                      </button>
                      <button
                        type="button"
                        className="danger-action"
                        onClick={() => confirmDelete({ id: workflow.id, name: workflow.name })}
                        disabled={deleteWorkflow.isPending}
                      >
                        删除预设
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </>
        )}
        {deleteWorkflow.isError ? <p role="alert">{formatApiError(deleteWorkflow.error, "协作预设删除失败")}</p> : null}
      </section>
    </section>
  );
}
