# Raspberry Pi frontend

**完整中文使用说明请看：[树莓派使用说明](../README.md)**（烧录、注册设备、firstboot、联调、升级）。

简要：

1. Imager 刷 Raspberry Pi OS Lite（64-bit）并开 SSH/Wi‑Fi  
2. 后端 `POST /api/robot/v1/devices/register` 拿到 `device_token`  
3. 派上克隆本仓后执行 `device/image/firstboot.sh`  
4. `systemctl status robot-runtime` 确认在跑  

AI 只在后端；本目录是前端 runtime + 烧录脚本。
