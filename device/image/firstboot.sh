#!/usr/bin/env bash
# Install CubeAgent voice-companion frontend on a Raspberry Pi.
# Does NOT install AI models. Backend URL is CubeAgent /api/robot/v1/ws.
set -euo pipefail

BACKEND_WS=""
DEVICE_ID="pi-$(hostname)"
DEVICE_TOKEN=""
REPO_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend-ws) BACKEND_WS="$2"; shift 2 ;;
    --device-id) DEVICE_ID="$2"; shift 2 ;;
    --device-token) DEVICE_TOKEN="$2"; shift 2 ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${BACKEND_WS}" ]]; then
  echo "--backend-ws wss://host/api/robot/v1/ws is required" >&2
  exit 2
fi

if [[ -z "${REPO_ROOT}" ]]; then
  if [[ -d /opt/cubeagent/device/robot_runtime ]]; then
    REPO_ROOT=/opt/cubeagent
  elif [[ -d "$(cd "$(dirname "$0")/../.." && pwd)/device/robot_runtime" ]]; then
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  else
    echo "Cannot find device/robot_runtime; pass --repo-root" >&2
    exit 2
  fi
fi

RUNTIME_SRC="${REPO_ROOT}/device/robot_runtime"
install -d /opt/robot_runtime
rsync -a --delete "${RUNTIME_SRC}/" /opt/robot_runtime/

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip alsa-utils rsync

python3 -m venv /opt/robot_runtime/.venv
/opt/robot_runtime/.venv/bin/pip install -U pip
/opt/robot_runtime/.venv/bin/pip install -e /opt/robot_runtime

cat >/etc/robot-runtime.env <<EOF
ROBOT_DEVICE_ID=${DEVICE_ID}
ROBOT_DEVICE_TOKEN=${DEVICE_TOKEN}
ROBOT_CLOUD_WS_URL=${BACKEND_WS}
ROBOT_ENABLE_BARGE_IN=true
EOF
chmod 600 /etc/robot-runtime.env

cat >/etc/systemd/system/robot-runtime.service <<'EOF'
[Unit]
Description=CubeAgent voice companion frontend (Raspberry Pi)
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/robot-runtime.env
WorkingDirectory=/opt/robot_runtime
ExecStart=/opt/robot_runtime/.venv/bin/robot-runtime
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now robot-runtime.service
systemctl --no-pager status robot-runtime.service || true
echo "Frontend installed. AI stays on backend: ${BACKEND_WS}"
