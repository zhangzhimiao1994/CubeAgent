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
        低频的子助手分工和协作预设合并到这里。平时对话由主 Agent 自动判断，只有需要维护默认角色池或预设模板时再进入本页。
      </p>

      <div className="two-column compact-collaboration-overview">
        <article className="status-card">
          <span>角色</span>
          <p>管理子 Agent 的名称、职责、模型和可用技能。</p>
        </article>
        <article className="status-card">
          <span>协作预设</span>
          <p>管理任务场景、默认模式、默认角色和一句话策略。</p>
        </article>
      </div>

      <section {...navTargetProps("roles", "collaboration-section")} aria-label="角色配置">
        <AgentsPage />
      </section>
      <section {...navTargetProps("workflows", "collaboration-section")} aria-label="协作预设">
        <WorkflowsPage />
      </section>
    </section>
  );
}
