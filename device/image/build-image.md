# Building a flashable .img (optional)

For a true one-shot burn image later:

1. Use [pi-gen](https://github.com/RPi-Distro/pi-gen) or Raspberry Pi Imager + cloud-init.
2. Bake in `device/robot_runtime` under `/opt/robot_runtime`.
3. Bake `firstboot.sh` outcomes (systemd unit + `/etc/robot-runtime.env` template).
4. Leave `ROBOT_CLOUD_WS_URL` / `ROBOT_DEVICE_TOKEN` for first boot customization (boot partition `cube.env`).

Do **not** bake model weights or local inference stacks into the image.

v1 bring-up uses `firstboot.sh` on stock Raspberry Pi OS Lite instead of a custom .img.
