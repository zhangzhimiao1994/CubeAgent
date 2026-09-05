# `/api/robot/v1` 接口

树莓派前端与 CubeAgent 云端大脑的协议。鉴权用设备 Token，不走控制台 JWT。

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
| `final_transcript` / `utterance.end` | 一轮用户文本。读 `text` 或 `payload.transcript` |
| `barge_in` | 打断当前回合，取消进行中的 run |

### 云端 → 设备

| type | 含义 |
|---|---|
| `hello.ok` | 已认证，含 `device_id` |
| `state` | `listening` / `thinking` / `speaking` |
| `text_delta` | 新增的助手文本后缀 |
| `final` | 本轮完整文本 |
| `cancelled` | barge-in 或取消 |
| `error` | 友好错误，随后回到 `listening` |
| `pong` | 心跳应答 |

音频块（`assistant.audio`）本迭代不实现，占位留给后续云端 TTS。

## 与主 Agent 的关系

每条有效转写都会：

1. 经 `RunServiceInboundSubmitter` 提交
2. `mode=direct`，`channel_context.source_channel=robot`
3. `conversation_id` 对同一设备保持 `ch-robot-…`
4. 轮询 `run_service.events`，把 `artifact.created` 的回答流式推出
5. `barge_in` 调用 `run_service.cancel`
