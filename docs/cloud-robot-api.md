# `/api/robot/v1` 接口

树莓派前端与 CubeAgent 云端大脑的协议。鉴权用设备 Token，不走控制台 JWT。

部署与整链路调试（注册、文本/音频 WS 示例、故障表）见 [voice-companion-deploy-debug.md](voice-companion-deploy-debug.md)。

音频帧使用 **JSON + base64**（与 Starlette `receive_json` / 派上 `websockets` 文本帧一致），不使用二进制 WebSocket 帧。

## 注册设备

```http
POST /api/robot/v1/devices/register
Content-Type: application/json

{"device_id":"pi-01"}
```

```json
{"device_id":"pi-01","device_token":"pi-01.<hmac>"}
```

同一 `device_id` + 同一 JWT 签名密钥会得到稳定 Token。请把 Token 只写在派上 `/etc/robot-runtime.env`，不要提交 Git。

## WebSocket

```
GET /api/robot/v1/ws
```

鉴权（二选一）：

- Header：`X-Device-Token: <token>`
- Query：`?device_token=<token>`

失败时先 accept 再以 `4401` 关闭。

### 设备 → 云端

| type | 含义 |
|---|---|
| `hello` | 握手（可选；服务端连上后会先推 `hello.ok`） |
| `ping` | 心跳，回 `pong` |
| `audio_chunk` | 一轮用户语音分片。`audio` 为 base64，`format` 为 `pcm16` / `wav` / `webm` / `mp3` |
| `audio.end` | 本轮采集结束。云端拼接缓冲、跑 STT，再走既有 DIRECT submit。可带 `text` 作调试回退 |
| `final_transcript` / `utterance.end` | 调试/文本回合。读 `text` 或 `payload.transcript`，不经 STT |
| `barge_in` | 打断当前回合：取消进行中的 run **和** 未播完的 TTS |

`audio_chunk` / `audio.end` 也接受把字段放在 `payload` 里（派上 `Envelope` 默认如此）。单轮音频上限约 2MB。兼容旧的 `utterance.audio`（有 `audio` 字段时当成分片）。

### 云端 → 设备

| type | 含义 |
|---|---|
| `hello.ok` | 已认证，含 `device_id` |
| `state` | `listening` / `thinking` / `speaking` |
| `transcript` | STT 结果（便于派上调试显示） |
| `text_delta` | 新增的助手文本后缀 |
| `final` | 本轮完整文本 |
| `audio_delta` | TTS 音频分片，`audio` 为 base64，含 `format` / `mime_type` |
| `audio.final` | 本轮 TTS 结束（文本 `final` 之后；不必等它才显示字幕） |
| `cancelled` | barge-in 或取消 |
| `error` | 友好错误，随后回到 `listening` |
| `pong` | 心跳应答 |

未配置 STT 且 `audio.end` 没有调试文本时，返回 `stt unavailable`。文本路径在未配置语音时仍可用。

## 听 / 说路径

1. 派采集 PCM → `audio_chunk`… → `audio.end`
2. 云端 STT（OpenAI 兼容 Whisper 等）得到转写
3. `RunServiceInboundSubmitter` 提交 `mode=direct`
4. 轮询 run events，推 `text_delta` / `final`
5. 句边界（`。！？!?；;`）或 `final` 剩余文本触发云端 TTS，推 `audio_delta` / `audio.final`
6. `barge_in` 调用 `run_service.cancel` 并取消进行中的 TTS 任务

v1 TTS 适配器一次返回整段音频，再按块下发；接口已是异步迭代，后续可换成真正的流式合成。

## 与主 Agent 的关系

每条有效转写都会：

1. 经 `RunServiceInboundSubmitter` 提交
2. `mode=direct`，`channel_context.source_channel=robot`
3. `conversation_id` 对同一设备保持 `ch-robot-…`
4. 轮询 `run_service.events`，把 `artifact.created` 的回答流式推出
5. `barge_in` 调用 `run_service.cancel` 并停止 TTS
