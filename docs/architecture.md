# 语音陪伴机器人架构

CubeAgent 是云端大脑。树莓派只做前端，不跑模型。

```
麦克风 → 树莓派 VAD / 回合 / 打断
        → WebSocket /api/robot/v1（audio_chunk / audio.end）
        → 云端 STT
        → RunService.submit (TaskMode.DIRECT)
        → 既有记忆注入（Hermes）+ 控制台模型池
        → 文本流式回推 text_delta / final
        → 云端 TTS → audio_delta / audio.final
        → 喇叭播放；barge-in 停播并取消 run
```

## 分工

| 端 | 职责 | 不做什么 |
|---|---|---|
| 树莓派 `device/` | 麦/喇叭、VAD、回合、barge-in、WS 桥、播放云端音频 | 不装本地 LLM / ASR / TTS，不另起对话引擎 |
| CubeAgent | 设备注册与鉴权、STT/TTS、提交主 Agent、记忆、模型池、流式回推 | 不在本仓做主动调度 / 具身感知 / harness |

## 云端复用

- 入口：`RunServiceInboundSubmitter` → `run_service.submit`，与飞书等通道同一边界。
- `Channel.ROBOT`，`conversation_id` 按设备稳定为 `ch-robot-<uuid5>`。
- 机器人回合用 `TaskMode.DIRECT` + `skip_evolution_proposal`，降低陪伴延迟。
- 记忆：robot + DIRECT 在建 run 前走既有 `_safe_hermes_advice`，把 `injected_memories` 写入 routing；DIRECT runtime 用 `hermes_memory_context_text` 注入。不新增第二套记忆。
- 模型：默认走控制台模型池，不写并行 DeepSeek stub 客户端。
- 语音：`src/agent_hub/robot/voice/` 可插拔 STT/TTS。生产默认路径是 OpenAI 兼容 Whisper + OpenAI TTS 或 DashScope CosyVoice。密钥只走 `AGENT_HUB_*` 环境变量 / 部署配置，不写进代码。`multimodal/` 仍只负责图/视频，不复用为陪伴语音。

## 开箱路径

1. 部署 CubeAgent（见仓库根 README）。
2. 在服务器配置 STT/TTS 凭证（见下方）。未配置时文本调试路径仍可用，但派上说话不会转写/播报。
3. `POST /api/robot/v1/devices/register` 拿到 `device_token`。
4. 树莓派执行 `device/image/firstboot.sh`。
5. 说话 → 云端 STT → 主 Agent → 云端 TTS → 喇叭。

### STT / TTS 环境变量

前缀均为 `AGENT_HUB_`：

| 变量 | 说明 |
|---|---|
| `ROBOT_STT_PROVIDER` | `none`（默认）/ `fake` / `openai_compatible` |
| `ROBOT_STT_API_KEY` | Whisper 兼容接口密钥 |
| `ROBOT_STT_BASE_URL` | 默认 `https://api.openai.com/v1` |
| `ROBOT_STT_MODEL` | 默认 `whisper-1` |
| `ROBOT_TTS_PROVIDER` | `none`（默认）/ `fake` / `openai_compatible` / `dashscope` |
| `ROBOT_TTS_API_KEY` | TTS 密钥 |
| `ROBOT_TTS_BASE_URL` | OpenAI 兼容或 DashScope compatible-mode URL |
| `ROBOT_TTS_MODEL` | 默认 `tts-1`；DashScope 常用 `cosyvoice-v2` |
| `ROBOT_TTS_VOICE` | 默认 `alloy`；DashScope 常用 `longxiaochun` |
| `ROBOT_TTS_FORMAT` | 默认 `mp3` |

中国区陪伴可用 DashScope compatible-mode 的 Whisper 兼容 ASR + CosyVoice，或任意 OpenAI 兼容网关。不要把密钥写进树莓派镜像。

细节见 `docs/cloud-robot-api.md` 与 `device/README.md`。
