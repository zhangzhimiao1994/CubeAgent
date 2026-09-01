import { useNavSection } from "../app/navSections";
import { AgentsPage } from "./AgentsPage";
import { WorkflowsPage } from "./WorkflowsPage";

export function CollaborationPage() {
  const { navTargetProps } = useNavSection();

  return (
    <section>
      <p className="eyebrow">Agent collaboration</p>
      <h2>协作配置</h2>
      <p className="compact-page-intro">
        低频的子助手分工和流程模板合并到这里。平时对话由主 Agent 自动判断，只有需要调整角色池或固定流程时再进入本页。
      </p>

      <div className="two-column compact-collaboration-overview">
        <article className="status-card">
          <span>角色</span>
          <p>管理子 Agent 的名称、职责、模型和可用技能。</p>
        </article>
        <article className="status-card">
          <span>工作流</span>
          <p>管理任务类型、默认角色、执行步骤和裁决规则。</p>
        </article>
      </div>

      <section {...navTargetProps("roles", "collaboration-section")} aria-label="角色配置">
        <AgentsPage />
      </section>
      <section {...navTargetProps("workflows", "collaboration-section")} aria-label="流程模板">
        <WorkflowsPage />
      </section>
    </section>
  );
}
