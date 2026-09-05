# 树莓派端使用说明（前端）

这份文档只讲 **树莓派怎么用**。AI 在 CubeAgent 云端跑，派上不跑模型。

相关目录：

| 路径 | 作用 |
|---|---|
| `device/robot_runtime/` | 派上前端程序（麦/喇叭/回合/打断/连后端） |
| `device/image/firstboot.sh` | 一键安装并注册 systemd 服务 |
| `docs/architecture.md` | 整体架构（后端 vs 前端） |
| `docs/cloud-robot-api.md` | 后端 `/api/robot/v1` 接口 |

---

## 开箱路径

1. **部署 CubeAgent**（服务器上 `install.sh` 或现有生产指针 `/opt/agent-hub/current`）
2. **配置云端 STT/TTS 凭证**（`AGENT_HUB_ROBOT_STT_*` / `AGENT_HUB_ROBOT_TTS_*`，见 `docs/architecture.md`）。未配置时仍可用文本调试回合。
3. **注册设备**，拿到 `device_token`
4. **树莓派 firstboot**，指向 `wss://YOUR_HOST/api/robot/v1/ws`
5. **对着麦克风说话**（接好真实声卡后）；云端 STT → 主 Agent + 记忆 → TTS 播报

---

## 1. 你需要准备什么

**硬件**

- 树莓派（建议 4/5，64 位）
- 麦克风（USB 或 I2S）
- 喇叭 / 带扬声器的声卡
- 能上网（Wi‑Fi 或网线）
- 一张 microSD 卡（建议 ≥16GB）

**软件 / 账号**

- 电脑上安装 [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- 已部署并跑起来的 **CubeAgent**（`https://github.com/zhangzhimiao1994/CubeAgent`）
- 后端公网或局域网地址，且派能访问，例如：`https://agent.example.com`
- 后端已开启 WebSocket：`wss://agent.example.com/api/robot/v1/ws`

派负责采集/播放与回合；CubeAgent 复用 RunService、记忆注入、控制台模型池，以及云端 STT/TTS。不要在派上装 DeepSeek、本地大模型或端上 ASR/TTS。

---

## 2. 烧录系统

1. 打开 Raspberry Pi Imager  
2. 选择系统：**Raspberry Pi OS Lite（64-bit）**  
3. 选择存储卡  
4. 点高级设置（齿轮）：
   - 启用 SSH
   - 设置用户名/密码（或密钥）
   - 配置 Wi‑Fi（SSID / 密码 / 国家）
   - 主机名可设为 `cube-robot`
5. 写入并把卡插进树莓派，通电开机

---

## 3. 拿到后端设备 Token

在能访问 CubeAgent 的电脑上执行（把地址换成你的）：

```bash
curl -sS -X POST "https://YOUR_AGENT_HOST/api/robot/v1/devices/register" \
  -H "content-type: application/json" \
  -d '{"device_id":"pi-01"}'
```

返回示例：

```json
{"device_id":"pi-01","device_token":"……"}
```

请保存 `device_token`，后面 firstboot 要用。不要提交到 Git。

---

## 4. 把代码弄到树莓派

SSH 登录派：

```bash
ssh 你的用户名@cube-robot.local
```

克隆本仓库（或只同步 `device/` 目录）：

```bash
sudo apt-get update
sudo apt-get install -y git
git clone https://github.com/zhangzhimiao1994/CubeAgent.git
cd CubeAgent
```

若仓库是私有的，用带权限的 HTTPS/SSH 方式克隆。

---

## 5. 一键安装前端（推荐）

在仓库根目录执行：

```bash
sudo bash device/image/firstboot.sh \
  --backend-ws "wss://YOUR_AGENT_HOST/api/robot/v1/ws" \
  --device-id "pi-01" \
  --device-token "上一步的token" \
  --repo-root "$(pwd)"
```

脚本会：

1. 把 `device/robot_runtime` 同步到 `/opt/robot_runtime`
2. 创建 Python venv 并安装 `robot-runtime`
3. 写入 `/etc/robot-runtime.env`
4. 安装并启动 systemd 服务 `robot-runtime`

查看状态：

```bash
systemctl status robot-runtime --no-pager
journalctl -u robot-runtime -f
```

环境变量文件：`/etc/robot-runtime.env`

| 变量 | 含义 |
|---|---|
| `ROBOT_DEVICE_ID` | 设备 ID，需与注册时一致 |
| `ROBOT_DEVICE_TOKEN` | 注册返回的 token |
| `ROBOT_CLOUD_WS_URL` | CubeAgent WS 地址 |
| `ROBOT_ENABLE_BARGE_IN` | 是否允许打断（默认 true） |

修改后重启：

```bash
sudo systemctl restart robot-runtime
```

---

## 6. 声音设备（麦 / 喇叭）

列出声卡：

```bash
arecord -l
aplay -l
```

试录 / 试播（按你的卡号改 `hw:1,0`）：

```bash
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 /tmp/test.wav
aplay /tmp/test.wav
```

当前 `robot_runtime` 里的采集/播放默认是占位实现（Null）。协议已会把 PCM 以 `audio_chunk` 发给云端，并把 `audio_delta` 送进播放队列。要「开箱出声」，还需要在派上接好 ALSA 设备并把 `audio/capture.py`、`audio/playback.py` 换成真实驱动（后续迭代）。若云端 TTS 下发的是 mp3，真实喇叭驱动需要按 `format` / `mime_type` 解码。

在驱动接好之前，你可以用 CubeAgent HTTP/WS 先验证注册、握手、文本回合，以及（配置了 STT/TTS 后）音频转写与回推。

---

## 7. 联调检查清单

1. 后端健康：浏览器或 curl 打开 `https://YOUR_AGENT_HOST/health`
2. 设备已注册，token 正确
3. 派能解析并访问后端域名（防火墙放行 443/WSS）
4. `journalctl -u robot-runtime` 无持续报错
5. 后端日志能看到设备 `hello.ok` 之后的 run 提交

常见问题：

| 现象 | 处理 |
|---|---|
| WS 401 / 立刻断开 | token 错；检查 `ROBOT_DEVICE_TOKEN` 与 JWT 签名密钥是否和发牌时一致 |
| 连不上 | 检查 `wss://` 地址、证书、防火墙、DNS |
| 服务起不来 | `journalctl -u robot-runtime -e`；确认 venv 与 `robot-runtime` 入口 |
| 没声音 | 先 `arecord`/`aplay` 确认硬件；再查 ALSA 默认设备 |

---

## 8. 日常使用

开机后 `robot-runtime` 应自动启动（systemd enable）。断线会自动重连。

正常流程：

1. 派采集人声 → `audio_chunk` / `audio.end`  
2. 云端 STT 转写后提交主 Agent  
3. 云端用记忆 + 模型池生成回复  
4. 派接收 `text_delta` / `final` 以及 `audio_delta` / `audio.final` 并播放  
5. 说话时可打断（barge-in：停播，并取消进行中的 run 与 TTS）

---

## 9. 升级派上程序

```bash
cd ~/CubeAgent
git pull
sudo bash device/image/firstboot.sh \
  --backend-ws "wss://YOUR_AGENT_HOST/api/robot/v1/ws" \
  --device-id "pi-01" \
  --device-token "你的token" \
  --repo-root "$(pwd)"
```

**升级 AI / 记忆 / 模型**：改 CubeAgent 后端并重启即可，不用重烧系统。

---

## 10. 卸载

```bash
sudo systemctl disable --now robot-runtime
sudo rm -f /etc/systemd/system/robot-runtime.service /etc/robot-runtime.env
sudo rm -rf /opt/robot_runtime
sudo systemctl daemon-reload
```

---

## 11. 和后端的分工（别搞反）

- **树莓派**：前端。只负责听、说、打断、联网。  
- **CubeAgent**：大脑。STT/TTS、RunService、记忆、模型池、流式文本与音频。  
- **不要**在派上装 DeepSeek/本地大模型/端上 ASR 来「陪聊」。

后端 LLM 与 STT/TTS 的 API Key 配在**服务器环境变量 / 控制台模型池**里，不要写进派的 SD 卡镜像，也不要提交 Git。开箱前请在 CubeAgent 上设置 `AGENT_HUB_ROBOT_STT_*` 与 `AGENT_HUB_ROBOT_TTS_*`（见 `docs/architecture.md`）。
