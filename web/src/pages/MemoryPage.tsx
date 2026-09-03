import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useNavSection } from "../app/navSections";
import { api, formatApiError, type MemoryCenterActionName, type MemoryCenterItem } from "../api/client";
import { HermesInsightDetail } from "./HermesPage";

type MemoryCenterActionRequest = {
  id: string;
  action: MemoryCenterActionName;
};

export function MemoryPage() {
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();
  const { navTargetProps } = useNavSection(["status", "source", "category", "section"]);
  const memoryCenter = useQuery({ queryKey: ["memory-center"], queryFn: () => api.memoryCenter() });
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);

  const memoryCenterAction = useMutation({
    mutationFn: ({ id, action }: MemoryCenterActionRequest) => api.memoryCenterAction(id, action),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["memory-center"] });
      await queryClient.invalidateQueries({ queryKey: ["memory"] });
      await queryClient.invalidateQueries({ queryKey: ["hermes"] });
      await queryClient.invalidateQueries({ queryKey: ["cognitive", "experiences"] });
    },
  });

  const centerItems = memoryCenter.data ?? [];
  const sourceFilter = searchParams.get("source");
  const statusFilter = searchParams.get("status");
  const activeMemorySection = statusFilter === "pending" || statusFilter === "confirmed" ? statusFilter : sourceFilter ?? "memory";
  const visibleItems = useMemo(
    () =>
      centerItems.filter((item) => {
        if (sourceFilter === "cognitive") {
          if (!item.source.startsWith("cognitive_")) return false;
        } else if (sourceFilter && item.source !== sourceFilter) {
          return false;
        }
        const normalizedStatus = statusFilter === "pending" ? "candidate" : statusFilter;
        if (normalizedStatus && item.status !== normalizedStatus) return false;
        return true;
      }),
    [centerItems, sourceFilter, statusFilter],
  );
  const selectedItem = centerItems.find((item) => item.id === selectedItemId) ?? null;
  const insightId = searchParams.get("insight");

  function runAction(item: MemoryCenterItem, action: MemoryCenterActionName) {
    if (action === "delete" && !window.confirm("确认删除这条记忆资产？删除后不会再进入运行时注入或学习建议。")) return;
    memoryCenterAction.mutate({ id: item.id, action });
    if (action === "delete" || action === "reject") setSelectedItemId(null);
  }

  if (insightId) {
    return (
      <section>
        <p className="eyebrow">Memory control</p>
        <h2>记忆 / 经验管理</h2>
        <HermesInsightDetail insightId={insightId} returnTo="/memory?source=hermes" unifiedActions />
      </section>
    );
  }

  return (
    <section>
      <p className="eyebrow">Memory control</p>
      <h2>记忆 / 经验管理</h2>
      <p>
        这里统一管理普通记忆、Hermes 学习和 Cognitive 经验。主界面只展示可判断的摘要，完整内容放到详情里确认，
        避免台账内容过长影响操作。
      </p>

      {memoryCenterAction.isError ? (
        <p role="alert">{formatApiError(memoryCenterAction.error, "记忆操作失败")}</p>
      ) : null}
      {memoryCenter.isError ? (
        <p role="alert">{formatApiError(memoryCenter.error, "统一记忆资产加载失败")}</p>
      ) : (
        <section aria-label="统一记忆资产" {...navTargetProps(activeMemorySection)}>
          <h3>统一记忆资产</h3>
          <p>学习记录、普通记忆和经验候选都会在这里以摘要列表显示；点击条目查看证据、详情和操作。</p>
          {memoryCenter.isLoading ? (
            <p>正在加载统一记忆资产...</p>
          ) : visibleItems.length === 0 ? (
            <article>
              <h4>还没有可展示的记忆资产</h4>
              <p>当前筛选下没有普通记忆、Hermes 学习或 Cognitive 候选。</p>
            </article>
          ) : (
            <div className="memory-center-list" role="list" aria-label="记忆与经验摘要列表">
              {visibleItems.map((item) => (
                <MemoryCenterRow
                  key={item.id}
                  item={item}
                  isActing={memoryCenterAction.isPending}
                  onOpen={() => setSelectedItemId(item.id)}
                  onAction={(action) => runAction(item, action)}
                />
              ))}
            </div>
          )}
        </section>
      )}

      {selectedItem ? (
        <MemoryCenterDetailDialog
          item={selectedItem}
          isActing={memoryCenterAction.isPending}
          onClose={() => setSelectedItemId(null)}
          onAction={(action) => runAction(selectedItem, action)}
        />
      ) : null}

    </section>
  );
}

function MemoryCenterRow({
  item,
  isActing,
  onOpen,
  onAction,
}: {
  item: MemoryCenterItem;
  isActing: boolean;
  onOpen: () => void;
  onAction: (action: MemoryCenterActionName) => void;
}) {
  return (
    <article
      className="memory-center-row"
      role="listitem"
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen();
        }
      }}
      tabIndex={0}
    >
      <div className="memory-center-row-main">
        <span className="eyebrow">{memoryCenterSourceLabel(item.source)}</span>
        <h3>{item.source === "memory" ? memoryCenterMemoryTitle(item) : item.summary}</h3>
        {item.source === "memory" && item.detail !== item.summary ? <p>{item.summary}</p> : null}
        <div className="inline-status-list">
          <span>{memoryScopeLabel(item.memory_scope)}</span>
          <span>{memoryCenterStatusLabel(item.status)}</span>
          {item.active_for_runtime ? <span>可注入运行时</span> : <span>不直接注入</span>}
          {item.confidence !== null ? <span>置信度 {item.confidence.toFixed(2)}</span> : null}
          {item.evidence_count > 0 ? <span>证据 {item.evidence_count} 条</span> : null}
          {item.contradiction_count > 0 ? <span>冲突 {item.contradiction_count} 条</span> : null}
        </div>
      </div>
      <div className="memory-center-actions" aria-label={`${item.summary} 操作`}>
        {memoryCenterActions(item).map((action) => (
          <button
            key={action}
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onAction(action);
            }}
            disabled={isActing}
            className={memoryCenterActionClassName(action)}
          >
            {memoryCenterActionLabel(action)}
          </button>
        ))}
      </div>
      <button
        className="sr-only"
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          onOpen();
        }}
      >
        打开记忆详情：{item.summary}
      </button>
    </article>
  );
}

function MemoryCenterDetailDialog({
  item,
  isActing,
  onClose,
  onAction,
}: {
  item: MemoryCenterItem;
  isActing: boolean;
  onClose: () => void;
  onAction: (action: MemoryCenterActionName) => void;
}) {
  return (
    <div className="modal-backdrop memory-detail-backdrop" role="presentation" onClick={onClose}>
      <article
        className="modal-card memory-detail-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="记忆详情"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="memory-detail-header">
          <div>
            <span className="eyebrow">{memoryCenterSourceLabel(item.source)}</span>
            <h3>{item.source === "memory" ? memoryCenterMemoryTitle(item) : item.summary}</h3>
          </div>
          <button type="button" onClick={onClose} aria-label="关闭记忆详情">
            关闭
          </button>
        </div>
        <dl className="memory-detail-meta">
          <div>
            <dt>状态</dt>
            <dd>{memoryCenterStatusLabel(item.status)}</dd>
          </div>
          <div>
            <dt>作用域</dt>
            <dd>{memoryScopeLabel(item.memory_scope)}</dd>
          </div>
          <div>
            <dt>运行时</dt>
            <dd>{item.active_for_runtime ? "可注入运行时" : "不直接注入"}</dd>
          </div>
          {item.confidence !== null ? (
            <div>
              <dt>置信度</dt>
              <dd>{item.confidence.toFixed(2)}</dd>
            </div>
          ) : null}
          <div>
            <dt>使用</dt>
            <dd>
              {item.use_count} 次，成功 {item.success_count}，失败 {item.failure_count}
            </dd>
          </div>
          <div>
            <dt>证据/冲突</dt>
            <dd>
              证据 {item.evidence_count} 条，冲突 {item.contradiction_count} 条
            </dd>
          </div>
          {item.user_id ? (
            <div>
              <dt>用户</dt>
              <dd>{item.user_id}</dd>
            </div>
          ) : null}
          {item.created_at || item.updated_at ? (
            <div>
              <dt>时间</dt>
              <dd>
                {item.created_at ? `创建 ${item.created_at}` : null}
                {item.created_at && item.updated_at ? " / " : null}
                {item.updated_at ? `更新 ${item.updated_at}` : null}
              </dd>
            </div>
          ) : null}
        </dl>
        <section className="memory-detail-content" aria-label="完整内容">
          <h4>完整内容</h4>
          <p>{item.detail || item.summary}</p>
        </section>
        <div className="memory-center-actions">
          {memoryCenterActions(item).map((action) => (
            <button
              key={action}
              type="button"
              onClick={() => onAction(action)}
              disabled={isActing}
              className={memoryCenterActionClassName(action)}
            >
              {memoryCenterActionLabel(action)}
            </button>
          ))}
        </div>
      </article>
    </div>
  );
}

function memoryCenterMemoryTitle(item: MemoryCenterItem) {
  return item.id.startsWith("memory:") ? item.id.slice("memory:".length) : item.summary;
}

function memoryCenterActions(item: MemoryCenterItem): MemoryCenterActionName[] {
  if (item.source === "memory") {
    return item.status === "locked" ? ["unlock", "delete"] : ["lock", "delete"];
  }
  if (item.source === "hermes") {
    return item.status === "candidate" ? ["confirm", "reject", "delete"] : ["delete"];
  }
  if (item.source === "cognitive_experience") {
    if (item.status === "candidate") return ["confirm", "reject", "delete"];
    return ["delete"];
  }
  if (item.source === "cognitive_strategy") {
    if (item.status === "candidate") return ["confirm", "reject"];
    return [];
  }
  return [];
}

function memoryCenterActionLabel(action: MemoryCenterActionName) {
  const labels: Record<MemoryCenterActionName, string> = {
    confirm: "确认",
    reject: "拒绝",
    delete: "删除",
    lock: "锁定",
    unlock: "解除锁定",
  };
  return labels[action];
}

function memoryCenterActionClassName(action: MemoryCenterActionName) {
  if (action === "delete") return "danger-action";
  if (action === "reject" || action === "lock" || action === "unlock") return "secondary-action";
  return undefined;
}

function memoryCenterSourceLabel(source: MemoryCenterItem["source"]) {
  const labels: Record<MemoryCenterItem["source"], string> = {
    memory: "普通记忆",
    hermes: "Hermes 学习",
    cognitive_experience: "经验",
    cognitive_strategy: "策略",
    cognitive_reflection: "反思",
    cognitive_outcome: "结果校验",
    cognitive_belief: "信念",
    cognitive_relationship: "关系",
    cognitive_world: "世界状态",
    cognitive_skill: "技能",
  };
  return labels[source];
}

function memoryScopeLabel(scope: string) {
  if (scope === "root") return "根记忆";
  if (scope === "user") return "用户记忆";
  if (scope === "tenant") return "租户记忆";
  return scope;
}

function memoryCenterStatusLabel(status: string) {
  const labels: Record<string, string> = {
    active: "已激活",
    locked: "已锁定",
    candidate: "待确认",
    confirmed: "已确认",
    rejected: "已拒绝",
    deprecated: "已降级",
    superseded: "已替代",
    success: "成功",
    failure: "失败",
  };
  return labels[status] ?? status;
}
