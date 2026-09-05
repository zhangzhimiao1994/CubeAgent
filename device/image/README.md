# Flashable Raspberry Pi appliance

Goal: burn an SD card, boot the Pi, and have the **frontend** runtime come up talking to the agent backend. The image does **not** include AI weights or local model servers.

## What gets burned

1. Raspberry Pi OS Lite (64-bit recommended)
2. First-boot script installs `device/robot_runtime`
3. systemd starts `robot-runtime`
4. Env points at your agent backend WebSocket

## Quick burn (manual, v1)

1. Flash Raspberry Pi OS Lite with [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. In Imager advanced settings: enable SSH, set Wi‑Fi, set hostname e.g. `cube-robot`.
3. Copy this folder onto the boot partition as `cube-firstboot/` (or clone the repo after first login).
3. Clone https://github.com/zhangzhimiao1994/CubeAgent or copy `device/` onto the Pi.
4. On first SSH login:

```bash
sudo bash /boot/firmware/cube-firstboot/firstboot.sh \
  --backend-ws wss://YOUR_AGENT_HOST/api/robot/v1/ws \
  --device-id pi-$(hostname) \
  --device-token YOUR_TOKEN
```

`firstboot.sh` will:

- install OS audio deps
- install `robot-runtime` under `/opt/robot_runtime`
- write `/etc/robot-runtime.env`
- enable `robot-runtime.service`

## Register device token

On the agent backend:

```bash
curl -X POST https://YOUR_AGENT_HOST/api/robot/v1/devices/register \
  -H 'content-type: application/json' \
  -d '{"device_id":"pi-prototype-01"}'
```

Put the returned `device_token` into firstboot / env.

## Later: one-click .img

`build-image.md` describes packing a custom image with pi-gen so firstboot is already applied. Not required for bring-up.
