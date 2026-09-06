# 语音陪伴：部署 CubeAgent 与整链路调试

面向要把树莓派接到 CubeAgent 云端大脑的人。树莓派只做麦/喇叭/VAD/打断/WebSocket；听、想、说都在服务器。

配套文档（本文不重复细节）：

| 文档 | 看什么 |
|---|---|
| [安装说明](installation.md) / 根目录 [README.zh-CN.md](../README.zh-CN.md) | 通用安装、镜像、HTTPS、Docker 离线包 |
| [运维说明](operations.md) | `scripts/agent-hub` |
| [架构](architecture.md) | 云端 vs 派上分工 |
| [机器人协议](cloud-robot-api.md) | `/api/robot/v1` 字段 |
| [树莓派前端](../device/README.md) | firstboot、声卡、`/etc/robot-runtime.env` |

下面命令里的 `YOUR_HOST`、`YOUR_DEVICE_TOKEN`、密钥一律换成你的值，不要提交真实密钥。

---

## 仍未做完（调试预期）

按现状排查，不要假设这些已经可用：

| 缺口 | 现状 |
|---|---|
| 真实 ALSA 采集/播放 | 派上默认是 `NullAudioCapture` / `NullAudioPlayback`。协议会发 `audio_chunk`、收 `audio_delta` 入队列，但**不会真正出声**，除非你自己接 ALSA 并按 `format`/`mime_type` 解码（TTS 默认常为 mp3）。 |
| AEC / 半双工 | `PassthroughEchoCanceller` 原样返回麦克风帧，没有回声消除，也没有半双工闸门。喇叭响时容易误触发 barge-in。 |
| 设备 ↔ 用户绑定 | `POST /api/robot/v1/devices/register` **不需要控制台 JWT**。Token 是 `device_id` + `AGENT_HUB_JWT_SIGNING_KEY` 的 HMAC。没有「这台派属于哪个登录用户」的控制台绑定。 |
| Realtime Memory Gate | 机器人 DIRECT 会走既有 `_safe_hermes_advice` 记忆注入（Hermes advice，超时约 0.8s，失败则空注入）。**没有**单独的实时记忆门、人物/情景记忆升级或离线整理。 |

---

# 甲、在服务器上部署 Agent

原生路径与仓库安装器一致：`install.sh`、`docs/installation.md`、`scripts/agent-hub`。

## 1. 全新原生安装

干净 Linux（建议 1GB+ 内存、2GB+ 磁盘），克隆后：

```bash
git clone https://github.com/zhangzhimiao1994/CubeAgent.git
cd CubeAgent
sudo bash install.sh --mode auto --yes
```

`--mode auto`：有 systemd + apt/dnf 时走原生；否则才落到 Docker。

国内网络可加镜像（`auto` 先官方、失败再国内源；`china` 直接国内；`official` 不改源）：

```bash
sudo env AGENT_HUB_MIRROR_MODE=auto bash install.sh --mode auto --yes
```

HTTPS：`AGENT_HUB_PUBLIC_URL` 以 `https://` 开头且未给证书文件时，Caddy 按解析好的域名自动签证书。已有证书时：

```bash
sudo env AGENT_HUB_PUBLIC_URL=https://agent.example.com \
  AGENT_HUB_TLS_CERT_FILE=/root/certs/fullchain.pem \
  AGENT_HUB_TLS_KEY_FILE=/root/certs/privkey.pem \
  bash install.sh --mode auto --yes
```

API 默认只听 `127.0.0.1:8000`，对外入口是 Caddy。安装器**不会**改云厂商安全组，80/443 要自己放行。

装完后活动指针是 `/opt/agent-hub/current`。密钥与数据在 `/etc/agent-hub/secrets.env`、`/var/lib/agent-hub`，重跑安装器**不会覆盖**已有 secrets。

## 2. 第一次 `/setup`

安装脚本会打印管理地址和一次性初始化码。浏览器打开：

```text
https://YOUR_HOST/setup
```

用初始化码创建第一个超级管理员。码来自 `AGENT_HUB_SETUP_CODE`（原生在 `/etc/agent-hub/secrets.env`）。成功后 `/setup` 关闭。

## 3. 控制台：普通模型 + 主 Agent

机器人回合走 **同一套** RunService + 控制台模型池，没有并行 companion LLM。

1. 登录 Web 控制台。
2. **模型配置**：至少加一个**普通模型**（`text` / 可对话），例如 DeepSeek 或任意 OpenAI 兼容中转。密钥写在控制台模型/密钥配置里，不要写进派。
3. **主 Agent**：选主模型、控制模式、决策策略、Hermes 策略。机器人 submit 固定 `TaskMode.DIRECT`，但仍用这里配置的主模型与记忆策略。
4. 在 Web 里用**直连**跟同一模型聊一句。通了再往下接机器人（见乙、L1）。

## 4. 服务器上的 STT / TTS 环境变量

Settings 前缀是 `AGENT_HUB_`（见 `src/agent_hub/settings.py`）。字段与环境变量对照：

| Settings 字段 | 环境变量 | 默认 | 取值 |
|---|---|---|---|
| `robot_stt_provider` | `AGENT_HUB_ROBOT_STT_PROVIDER` | `none` | `none` / `fake` / `openai_compatible` |
| `robot_stt_api_key` | `AGENT_HUB_ROBOT_STT_API_KEY` | 空 | Whisper 兼容接口密钥 |
| `robot_stt_base_url` | `AGENT_HUB_ROBOT_STT_BASE_URL` | `https://api.openai.com/v1` | 兼容 `/v1` 或站点根 |
| `robot_stt_model` | `AGENT_HUB_ROBOT_STT_MODEL` | `whisper-1` | 如 `whisper-1` |
| `robot_tts_provider` | `AGENT_HUB_ROBOT_TTS_PROVIDER` | `none` | `none` / `fake` / `openai_compatible` / `dashscope` |
| `robot_tts_api_key` | `AGENT_HUB_ROBOT_TTS_API_KEY` | 空 | TTS 密钥 |
| `robot_tts_base_url` | `AGENT_HUB_ROBOT_TTS_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容或 DashScope compatible-mode |
| `robot_tts_model` | `AGENT_HUB_ROBOT_TTS_MODEL` | `tts-1` | DashScope 未改模型时工厂会改成 `cosyvoice-v2` |
| `robot_tts_voice` | `AGENT_HUB_ROBOT_TTS_VOICE` | `alloy` | DashScope 未改音色时工厂会改成 `longxiaochun` |
| `robot_tts_format` | `AGENT_HUB_ROBOT_TTS_FORMAT` | `mp3` | `mp3` / `wav` / `pcm` |

`none`：不建 STT/TTS 适配器。文本 `final_transcript` 仍可用；`audio.end` 且没有调试 `text` 时，WS 回 `error`，`message` 为 `stt unavailable`。

`fake`：测试/通路冒烟。STT 固定转写 `hello`（见 `FakeSpeechToText`）；TTS 下发一块 `fake-audio` 字节（`audio/mpeg`）。**不要当生产语音。**

`openai_compatible`：STT `POST {base}/audio/transcriptions`（pcm16 先包 WAV）；TTS `POST {base}/audio/speech`。缺密钥时工厂**跳过该侧**，并打 warning（见乙、L4）。

`dashscope`：只用于 TTS。`base_url` 为空或仍是 `openai.com` 时，工厂改用 `https://dashscope.aliyuncs.com/compatible-mode/v1`，实际 POST 到该站点 origin 的 `/api/v1/services/audio/tts/SpeechSynthesizer`。中国区常见组合：STT 用 DashScope/中转的 Whisper 兼容接口（`openai_compatible`）+ TTS `dashscope`。

`multimodal/` 只做图/视频，不要把陪伴语音配进那里。

### 写到哪里

**原生安装（推荐陪伴路径）**

- 文件：`/etc/agent-hub/secrets.env`
- systemd：`deploy/native/systemd/agent-hub-api.service` 的 `EnvironmentFile=/etc/agent-hub/secrets.env`
- 改完必须重启进程，Settings 才会重新读入：

```bash
sudo systemctl restart agent-hub.target
# 或只重启 API：
sudo systemctl restart agent-hub-api
```

追加示例（占位符，不要用示例里的假密钥上线）：

```bash
sudo tee -a /etc/agent-hub/secrets.env >/dev/null <<'EOF'
AGENT_HUB_ROBOT_STT_PROVIDER=openai_compatible
AGENT_HUB_ROBOT_STT_API_KEY=replace-me
AGENT_HUB_ROBOT_STT_BASE_URL=https://api.openai.com/v1
AGENT_HUB_ROBOT_STT_MODEL=whisper-1
AGENT_HUB_ROBOT_TTS_PROVIDER=openai_compatible
AGENT_HUB_ROBOT_TTS_API_KEY=replace-me
AGENT_HUB_ROBOT_TTS_BASE_URL=https://api.openai.com/v1
AGENT_HUB_ROBOT_TTS_MODEL=tts-1
AGENT_HUB_ROBOT_TTS_VOICE=alloy
AGENT_HUB_ROBOT_TTS_FORMAT=mp3
EOF
sudo systemctl restart agent-hub.target
```

DashScope TTS 示例：

```text
AGENT_HUB_ROBOT_TTS_PROVIDER=dashscope
AGENT_HUB_ROBOT_TTS_API_KEY=replace-me
AGENT_HUB_ROBOT_TTS_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENT_HUB_ROBOT_TTS_MODEL=cosyvoice-v2
AGENT_HUB_ROBOT_TTS_VOICE=longxiaochun
```

通路冒烟（无真实语音供应商）：

```text
AGENT_HUB_ROBOT_STT_PROVIDER=fake
AGENT_HUB_ROBOT_TTS_PROVIDER=fake
```

**Docker Compose**

- 文件：`deploy/compose/.env`（从 `.env.example` 复制）。服务 `env_file: .env`。
- 完整 Compose / 离线镜像流程见根目录 README.zh-CN「Docker 离线初始部署包」，本文不重写。

## 5. 升级到本分支 / 本切片

`scripts/agent-hub upgrade --version <标记>` 会先做备份并写版本号；**它不会把 git 工作区打进** `/opt/agent-hub/current`。

要把本仓语音切片装到已有原生机器：

```bash
cd /path/to/CubeAgent
git fetch origin
git checkout cursor/voice-companion-robot-2bd6   # 或已合并后的目标分支
sudo bash install.sh --mode auto --yes
```

安装器走 repair/upgrade：保留 `/etc/agent-hub/secrets.env` 与 `/var/lib/agent-hub`，把当前源码树打到 `/opt/agent-hub/releases/<时间戳>`，再把 `/opt/agent-hub/current` 指过去并重启 `agent-hub.target`。

只改 secrets、没换代码时，重启即可，不必重跑安装器。

**JWT 签名密钥必须保持稳定。** 设备 Token 用 `AGENT_HUB_JWT_SIGNING_KEY` 做 HMAC。重装时若让安装器重新生成该键、或手改该行，旧 Token 全部失效（WS 先 accept 再以 `4401` 关闭）。需要新 Token 就重新 `register`，并改派上 `/etc/robot-runtime.env`。

## 6. 健康检查

```bash
scripts/agent-hub status
scripts/agent-hub doctor
scripts/agent-hub logs
```

原生 `status` 看 `agent-hub.target`；`logs` 是 `journalctl -u 'agent-hub-*'`。Docker 则走 `/opt/agent-hub/compose` 的 compose。

HTTP（本机或经 Caddy）：

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/health/live
curl -sS http://127.0.0.1:8000/health/ready
curl -sS https://YOUR_HOST/health
```

`/health` 与 `/health/live` 是存活；`/health/ready` 查数据库/Redis 等，失败为 503。

默认 `AGENT_HUB_LOG_LEVEL=WARNING`。排语音问题时可临时改成 `INFO` 或 `DEBUG`，看完改回。

---

# 乙、整链路调试（从云到派）

一层一层隔离。上一层不通，不要跳到派上调声卡。

## L0 后端存活

```bash
scripts/agent-hub status
scripts/agent-hub doctor
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/health/ready
```

`doctor` 失败按它打印的建议处理（磁盘、80/443、Docker、systemd、Caddy、`/opt/agent-hub/current`）。见 [troubleshooting.md](troubleshooting.md)。

## L1 模型池（不经机器人）

Web 控制台用**直连**、同一主模型发一句。成功说明 RunService + 模型池可用。这里失败时，机器人文本回合也不会出 `text_delta`。

## L2 设备注册

```bash
curl -sS -X POST "https://YOUR_HOST/api/robot/v1/devices/register" \
  -H "content-type: application/json" \
  -d '{"device_id":"pi-01"}'
```

期望：`{"device_id":"pi-01","device_token":"pi-01.<64 位 hex>"}`。

同一 `device_id` + 同一 JWT 签名密钥会得到同一 Token。只写在派上 `/etc/robot-runtime.env`，不要进 Git。

## L3 纯文本 WS（先不要 STT/TTS）

`AGENT_HUB_ROBOT_STT_PROVIDER` / `TTS` 保持 `none` 即可。证明：鉴权、会话、DIRECT submit、事件轮询。

需要本机有 `websockets`（`pip install websockets`）或 Node 的 `wscat`。

```bash
# 查询参数鉴权（与派上一致）
python3 - <<'PY'
import asyncio, json, os
import websockets

url = os.environ["ROBOT_WS"]  # 例 wss://YOUR_HOST/api/robot/v1/ws?device_token=...

async def main() -> None:
    async with websockets.connect(url) as ws:
        print("<-", await ws.recv())
        await ws.send(json.dumps({
            "type": "final_transcript",
            "text": "你好，请用一句话介绍你自己",
            "turn_id": "debug-text-1",
        }, ensure_ascii=False))
        for _ in range(20):
            raw = await ws.recv()
            print("<-", raw)
            data = json.loads(raw)
            if data.get("type") in {"final", "error", "cancelled"}:
                break

asyncio.run(main())
PY
```

```bash
export ROBOT_WS='wss://YOUR_HOST/api/robot/v1/ws?device_token=YOUR_DEVICE_TOKEN'
python3 that-script.py
```

或：

```bash
npx --yes wscat -c 'wss://YOUR_HOST/api/robot/v1/ws?device_token=YOUR_DEVICE_TOKEN'
# 连上后粘贴一行：
{"type":"final_transcript","text":"你好","turn_id":"debug-text-1"}
```

期望顺序（中间可能夹 `state`）：

1. `hello.ok`（服务端连上就推，不必先发 `hello`）
2. `state` = `thinking`
3. `text_delta`（可多条）
4. `final`
5. `state` = `listening`

没有 `text_delta`：回到 L1（主模型/密钥/容量），或 `scripts/agent-hub logs` 里找 `robot_run_submit_failed` / `robot_run_events_failed`。

Token 错：先 accept，再关连接，码 **4401**。

## L4 STT/TTS 配置

按甲、第 4 节写入 secrets 并重启 `agent-hub.target`。

工厂在 `src/agent_hub/robot/voice/factory.py` 的 logger 名是 `agent_hub.robot.voice.factory`。**没有**「voice gateway 已建立」这类成功日志。能核对的只有：

| 现象 | 代码真实行为 |
|---|---|
| provider 为 `openai_compatible`/`dashscope` 但密钥为空 | warning：`robot STT provider … skipped: missing API key` 或 TTS 同句 |
| 未知 provider | warning：`unknown robot STT provider …` / `unknown robot TTS provider …` |
| 两侧都是 `none` 或都被跳过 | `build_robot_voice` 返回 `None`，无日志 |
| STT HTTP/解析失败 | exception：`robot_stt_failed device_id=…` |
| TTS 失败 | exception：`robot_tts_failed device_id=…` |

默认日志级别是 WARNING，上述 warning/exception 会出现在 `journalctl -u agent-hub-api`。

建议顺序：先 `fake`/`fake` 跑通 L5，再换成真实密钥。

## L5 音频 WS

成帧：JSON + base64，不是二进制 WS 帧。`format`：`pcm16` / `wav` / `webm` / `mp3`。单轮缓冲约 2MB。派上默认 16 kHz 单声道 pcm16。

`fake` STT 无论音频内容都转写 `hello`，然后走与 L3 相同的 DIRECT。有 TTS 时，句边界（`。！？!?；;`）或 `final` 剩余文本后会出现 `audio_delta`（`audio` 为 base64）和 `audio.final`。

```bash
python3 - <<'PY'
import asyncio, base64, json, os
import websockets

url = os.environ["ROBOT_WS"]
pcm = b"\x00\x00" * 1600  # 16 kHz 单声道 s16le，约 100ms 静音
b64 = base64.b64encode(pcm).decode("ascii")

async def main() -> None:
    async with websockets.connect(url) as ws:
        print("<-", await ws.recv())
        await ws.send(json.dumps({
            "type": "audio_chunk",
            "turn_id": "debug-audio-1",
            "audio": b64,
            "format": "pcm16",
            "sample_rate_hz": 16000,
        }))
        await ws.send(json.dumps({
            "type": "audio.end",
            "turn_id": "debug-audio-1",
            "format": "pcm16",
            "sample_rate_hz": 16000,
        }))
        for _ in range(30):
            raw = await ws.recv()
            data = json.loads(raw)
            kind = data.get("type")
            if kind == "audio_delta":
                print("<- audio_delta", data.get("format"), data.get("mime_type"), "bytes", len(data.get("audio") or ""))
            else:
                print("<-", raw if kind != "text_delta" else data)
            if kind in {"audio.final", "error", "cancelled"}:
                if kind == "error" or kind == "cancelled":
                    break
                # audio.final 之后可以结束
                break

asyncio.run(main())
PY
```

期望（`fake` 或真实 STT）：

1. `transcript`（STT 成功时；fake 为 `hello`）
2. `state thinking` → `text_delta` / `final`（与 L3 相同）
3. 若配置了 TTS：`audio_delta`… → `audio.final`

未配 STT 且 `audio.end` 无 `text`：`error` / `stt unavailable`。  
未配 STT 但 `audio.end` 带 `"text":"调试文本"`：当文本回合提交（调试回退）。

## L6 派上 firstboot

步骤只在 [device/README.md](../device/README.md)：烧录、`POST register`、`device/image/firstboot.sh`、`/etc/robot-runtime.env`。

```bash
sudo bash device/image/firstboot.sh \
  --backend-ws "wss://YOUR_HOST/api/robot/v1/ws" \
  --device-id "pi-01" \
  --device-token "YOUR_DEVICE_TOKEN" \
  --repo-root "$(pwd)"
```

```bash
systemctl status robot-runtime --no-pager
journalctl -u robot-runtime -f
```

| 变量（`/etc/robot-runtime.env`） | 含义 |
|---|---|
| `ROBOT_DEVICE_ID` | 与注册时一致 |
| `ROBOT_DEVICE_TOKEN` | 注册返回的 token |
| `ROBOT_CLOUD_WS_URL` | `wss://YOUR_HOST/api/robot/v1/ws` |
| `ROBOT_ENABLE_BARGE_IN` | 默认 true |

改完：`sudo systemctl restart robot-runtime`。派上**不要**放 LLM/STT/TTS 密钥。

## L7 声卡

```bash
arecord -l
aplay -l
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 /tmp/test.wav
aplay /tmp/test.wav
```

`arecord`/`aplay` 失败是硬件/ALSA，不是 CubeAgent。

当前运行时采集/播放默认是 **Null**：即便 L5 在电脑上已收到 `audio_delta`，派上也可能「没声音」。要开箱出声，必须换真实 `audio/capture.py`、`audio/playback.py`；mp3 还要解码。

## L8 barge-in

说话中（已有 `text_delta` 或 `audio_delta`）再发：

```json
{"type":"barge_in","turn_id":"debug-text-1"}
```

期望：`cancelled`，然后 `state` = `listening`。服务端会 `run_service.cancel`，并 cancel 未完成的 TTS 任务。派上在检测到打断时会 `playback.clear()` 再发 `barge_in`。

没有真实半双工时，喇叭漏进麦会误打断——属已知缺口。

---

## 故障对照表

| 症状 | 先看哪一层 | 处理 |
|---|---|---|
| `/health` 或 `/health/ready` 失败 | L0 | `doctor`、PostgreSQL/Redis、Caddy、`journalctl -u agent-hub-api` |
| Web 直连也没回复 | L1 | 模型配置、主 Agent 主模型、供应商额度 |
| 注册 4xx / 连不上 | L2 / L0 | URL、防火墙、Caddy、HTTPS 证书 |
| WS 立刻断，码 4401 | L3 鉴权 | `ROBOT_DEVICE_TOKEN`；**重装后改过 `AGENT_HUB_JWT_SIGNING_KEY` 则旧 token 全废**，重新 register |
| 有 `hello.ok` 无 `text_delta` | L1 / L3 | 主模型；日志 `robot_run_submit_failed`；控制台该租户是否已设主 Agent |
| `error`：`stt unavailable` | L4 / L5 | 未配 STT 且 `audio.end` 无调试文本；或 provider 因缺密钥被 skip。先 L3 文本，再配 STT |
| `error`：`empty transcript` | L5 | STT 返回空且没有回退文本 |
| 有 `transcript`/`final` 无 `audio_delta` | L4 | TTS 为 `none` 或被 skip；回答尚无句边界且仍在流式（等到 `final`）；日志 `robot_tts_failed` |
| 电脑 L5 有 `audio_delta`，派上没声 | L7 | Null 播放；`arecord`/`aplay`；mp3 未解码 |
| 派上 journal 持续重连 | L6 | `wss://`、DNS、证书、4401/token |
| 延迟很高 | 协议本身 | STT 在 submit 前；TTS 等句号或 `final`；run events 约 50ms 轮询；Hermes advice 最多约 0.8s；v1 TTS 一次拿整段 |
| 重装/改 secrets 后 Token 无效 | JWT | 不要换 `AGENT_HUB_JWT_SIGNING_KEY`；否则重新 register 并更新派上 env |

---

## 建议验收顺序

1. L0 → L1 → L2 → L3（文本陪伴已成立）。  
2. `fake` STT/TTS → L5（证明音频协议与 submit）。  
3. 真实密钥 → L5（真转写、真合成）。  
4. L6 派上网 → L7 声卡（出声取决于 ALSA，不是本切片默认能力）。  
5. L8 打断。

只靠本文 + [device/README.md](../device/README.md) 应能完成：装 CubeAgent、配模型、配语音环境变量、分层判定卡在云还是派。
