#!/bin/bash
# Startup script for the GPU VM: installs an idle-shutdown watchdog.
#
# The watchdog stops (not deletes) the VM after a period of low GPU utilization,
# so a forgotten VM does not burn money. Training keeps the GPU busy, so it will
# not trigger mid-run; it fires only once the GPU has been idle long enough.
#
# Tunables (via instance metadata, with defaults):
#   idle-shutdown-threshold-percent  GPU util % below which counts as idle (default 15)
#   idle-shutdown-minutes            consecutive idle minutes before stop (default 30)

set -euo pipefail

THRESHOLD="$(curl -s -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/idle-shutdown-threshold-percent' \
  2>/dev/null || echo 15)"
IDLE_MINUTES="$(curl -s -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/idle-shutdown-minutes' \
  2>/dev/null || echo 30)"

# The watchdog script itself.
cat > /usr/local/bin/idle-shutdown.sh <<EOF
#!/bin/bash
set -uo pipefail
THRESHOLD=${THRESHOLD}
IDLE_MINUTES=${IDLE_MINUTES}
idle_count=0

while true; do
  # Average GPU utilization across all GPUs (integer percent).
  util="\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk '{s+=\$1; n++} END { if (n>0) printf "%d", s/n; else print 100 }')"

  if [ "\${util:-100}" -lt "\$THRESHOLD" ]; then
    idle_count=\$((idle_count + 1))
  else
    idle_count=0
  fi

  logger -t idle-shutdown "gpu_util=\${util}% idle_count=\${idle_count}/\${IDLE_MINUTES}"

  if [ "\$idle_count" -ge "\$IDLE_MINUTES" ]; then
    logger -t idle-shutdown "GPU idle for \${IDLE_MINUTES} min; shutting down."
    # Stop the instance (preserves the disk). Falls back to OS poweroff.
    ZONE="\$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print \$NF}')"
    NAME="\$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name)"
    gcloud compute instances stop "\$NAME" --zone "\$ZONE" --quiet || sudo poweroff
    exit 0
  fi

  sleep 60
done
EOF
chmod +x /usr/local/bin/idle-shutdown.sh

# Run it as a systemd service so it survives reboots and restarts on failure.
cat > /etc/systemd/system/idle-shutdown.service <<'EOF'
[Unit]
Description=Idle GPU auto-shutdown watchdog
After=network-online.target

[Service]
ExecStart=/usr/local/bin/idle-shutdown.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now idle-shutdown.service
logger -t idle-shutdown "watchdog installed (threshold=${THRESHOLD}% idle=${IDLE_MINUTES}min)"
