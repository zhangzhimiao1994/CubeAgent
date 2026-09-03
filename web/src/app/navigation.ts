export type ModuleSubItem = {
  to: string;
  label: string;
  permission: string;
};

export type ModuleItem = {
  to: string;
  label: string;
  description: string;
  permission: string;
  children?: ModuleSubItem[];
};

export type ModuleGroup = {
  id: string;
  to: string;
  label: string;
  eyebrow: string;
  description: string;
  tone: "cyan" | "green" | "amber" | "violet" | "blue" | "slate";
  modules: ModuleItem[];
};

export const MODULE_GROUPS: ModuleGroup[] = [
  {
    id: "workspace",
    to: "/workspace",
    label: "对话",
    eyebrow: "Conversation",
    description: "发起对话、接续会话、查看运行过程和交付内容。",
    tone: "cyan",
    modules: [
      {
        to: "/",
        label: "对话",
        description: "连续对话、历史会话、运行过程和附件入口集中在这里。",
        permission: "run:read",
      },
    ],
  },
  {
    id: "orchestration",
    to: "/orchestration",
    label: "编排",
    eyebrow: "Agent Control",
    description: "主 Agent、协作配置和计划任务属于 Agent 编排控制层。",
    tone: "green",
    modules: [
      {
        to: "/main-agent",
        label: "主 Agent",
        description: "单独配置主 Agent 模型、控场风格、决策边界和 Hermes 介入策略。",
        permission: "config:read",
        children: [
          { to: "/main-agent?section=model", label: "专属模型/API", permission: "config:read" },
          { to: "/main-agent?section=scheduler", label: "调度策略", permission: "config:read" },
          { to: "/main-agent?section=concurrency", label: "并发槽", permission: "config:read" },
          { to: "/main-agent?section=hermes", label: "Hermes 介入", permission: "hermes:read" },
        ],
      },
      {
        to: "/collaboration",
        label: "协作配置",
        description: "合并管理子助手分工和协作预设，减少低频配置入口。",
        permission: "agent:read",
        children: [
          { to: "/collaboration?section=roles", label: "角色", permission: "agent:read" },
          { to: "/collaboration?section=workflows", label: "协作预设", permission: "agent:read" },
          { to: "/collaboration?section=execution", label: "默认策略", permission: "agent:read" },
          { to: "/collaboration?section=review", label: "使用边界", permission: "agent:read" },
        ],
      },
      {
        to: "/schedules",
        label: "计划任务",
        description: "按指定时间提交任务，可用于报表填写、提醒和需要 OpenClaw 审批的本机操作。",
        permission: "run:create",
      },
    ],
  },
  {
    id: "resources",
    to: "/resources",
    label: "资源",
    eyebrow: "Models & Memory",
    description: "模型 API、Key、中转站协议、附件和记忆资源统一在资源层管理。",
    tone: "amber",
    modules: [
      {
        to: "/models",
        label: "模型与 API",
        description: "配置普通模型、多媒体模型、能力标签、并发容量和供应商预设。",
        permission: "config:read",
        children: [
          { to: "/models?category=text", label: "普通模型", permission: "config:read" },
          { to: "/models?category=multimedia", label: "多媒体模型", permission: "config:read" },
          { to: "/models?section=capabilities", label: "模型能力", permission: "config:read" },
          { to: "/models?section=capacity", label: "并发与容量", permission: "config:read" },
          { to: "/models?section=presets", label: "预设供应商", permission: "config:read" },
        ],
      },

      {
        to: "/memory",
        label: "记忆 / 经验",
        description: "统一管理普通记忆、Hermes 学习、经验候选和运行时可注入的长期上下文。",
        permission: "memory:read",
        children: [
          { to: "/memory?source=memory", label: "普通记忆", permission: "memory:read" },
          { to: "/memory?source=hermes", label: "学习台账", permission: "hermes:read" },
          { to: "/memory?status=pending", label: "待确认学习", permission: "hermes:read" },
          { to: "/memory?status=confirmed", label: "已确认学习", permission: "hermes:read" },
        ],
      },
      {
        to: "/attachments",
        label: "附件",
        description: "查看和删除从对话页上传的图片、文档、压缩包和上下文文件。",
        permission: "run:read",
      },
    ],
  },
  {
    id: "extensions",
    to: "/extensions",
    label: "工具",
    eyebrow: "Tools",
    description: "Skill、MCP 和后续插件入口集中在工具层，便于做权限边界和扩展。",
    tone: "violet",
    modules: [
      {
        to: "/skills",
        label: "Skill",
        description: "上传、安装、审核、搜索、批量管理和启用 Agent 可调用的技能包。",
        permission: "skill:read",
        children: [
          { to: "/skills?view=installed", label: "已安装 Skill", permission: "skill:read" },
          { to: "/skills?view=upload", label: "上传/安装", permission: "skill:read" },
          { to: "/skills?view=permissions", label: "待审批权限", permission: "skill:read" },
          { to: "/skills?view=bulk", label: "批量管理", permission: "skill:read" },
        ],
      },
      {
        to: "/mcp",
        label: "MCP",
        description: "管理外部工具连接、权限和可调用能力。",
        permission: "mcp:read",
      },
    ],
  },
  {
    id: "channels",
    to: "/channels-hub",
    label: "通道",
    eyebrow: "Channels",
    description: "飞书、Webhook 和后续企业 IM 接入统一放在通道层。",
    tone: "blue",
    modules: [
      {
        to: "/channels",
        label: "通道连接",
        description: "配置聊天软件接入参数、回调地址、附件获取、回复格式和连接状态。",
        permission: "config:read",
        children: [
          { to: "/channels?provider=feishu&mode=websocket", label: "飞书长连接", permission: "config:read" },
          { to: "/channels?provider=feishu&mode=webhook", label: "Webhook 备用", permission: "config:read" },
          { to: "/channels?section=reply", label: "回复格式", permission: "config:read" },
          { to: "/channels?section=resources", label: "资源识别", permission: "config:read" },
          { to: "/channels?section=test", label: "测试与日志", permission: "audit:read" },
        ],
      },
    ],
  },
  {
    id: "system",
    to: "/system",
    label: "系统",
    eyebrow: "System",
    description: "全局设置、用户权限和日志排查收进系统运维入口。",
    tone: "slate",
    modules: [
      {
        to: "/config",
        label: "系统设置",
        description: "配置默认模式、日志等级、工具审批、运行期调度和临时 Agent 策略。",
        permission: "config:read",
      },
      {
        to: "/openclaw",
        label: "OpenClaw 控制",
        description: "配置跨平台电脑/服务器接管、权限模式、远程适配器和审批执行控制台。",
        permission: "config:read",
        children: [
          { to: "/openclaw?section=targets", label: "目标设备", permission: "config:read" },
          { to: "/openclaw?section=policy", label: "权限策略", permission: "config:read" },
          { to: "/openclaw?section=sessions", label: "操作会话", permission: "config:read" },
          { to: "/openclaw?section=schedules", label: "计划任务联动", permission: "run:create" },
        ],
      },
      {
        to: "/users",
        label: "用户管理",
        description: "管理控制台用户、权限、登录状态和初始管理员保护。",
        permission: "user:read",
      },
      {
        to: "/logs",
        label: "日志中心",
        description: "查看登录、对话审计、调度、模型、通道、Agent 和系统日志。",
        permission: "audit:read",
        children: [
          { to: "/logs/audit?details=auth.login", label: "登录日志", permission: "audit:read" },
          { to: "/logs/audit?details=run.submit", label: "对话审计", permission: "audit:read" },
          { to: "/logs/mode", label: "调度日志", permission: "audit:read" },
          { to: "/logs/model", label: "模型调用日志", permission: "audit:read" },
          { to: "/logs/channel", label: "通道日志", permission: "audit:read" },
        ],
      },
    ],
  },
];
