# 语音陪伴机器人架构

CubeAgent 是云端大脑。树莓派只做前端，不跑模型。

```
麦克风 → 树莓派 VAD / 回合 / 打断
        → WebSocket /api/robot/v1
        → RunService.submit (TaskMode.DIRECT)
        → 既有记忆注入（Hermes）+ 控制台模型池
        → 文本流式回推 text_delta / final
        → 喇叭播放（后续可接云端 TTS）
```

## 分工

| 端 | 职责 | 不做什么 |
|---|---|---|
| 树莓派 `device/` | 麦/喇叭、VAD、回合、barge-in、WS 桥 | 不装本地 LLM，不另起对话引擎 |
| CubeAgent | 设备注册与鉴权、提交主 Agent、记忆、模型池、流式回推 | 不在本仓做完整云端 TTS / 主动调度 / 具身感知 |

## 云端复用

- 入口：`RunServiceInboundSubmitter` → `run_service.submit`，与飞书等通道同一边界。
- `Channel.ROBOT`，`conversation_id` 按设备稳定为 `ch-robot-<uuid5>`。
- 机器人回合用 `TaskMode.DIRECT` + `skip_evolution_proposal`，降低陪伴延迟。
- 记忆：robot + DIRECT 在建 run 前走既有 `_safe_hermes_advice`，把 `injected_memories` 写入 routing；DIRECT runtime 用 `hermes_memory_context_text` 注入。不新增第二套记忆。
- 模型：默认走控制台模型池，不写并行 DeepSeek stub 客户端。

## 开箱路径

1. 部署 CubeAgent（见仓库根 README）。
2. `POST /api/robot/v1/devices/register` 拿到 `device_token`。
3. 树莓派执行 `device/image/firstboot.sh`。
4. 说话 → 云端主 Agent 回复。

细节见 `docs/cloud-robot-api.md` 与 `device/README.md`。
