# GCP GPU VM for RL Post-Training

The A100 VM used for hands-on Phase 1+ training.

## Instance

- Name: `cliagent-rlpt-a100`
- Project: `emmanuel-genai-sa`
- Zone: `us-central1-f` (A100 stockout in `us-central1-a`; -f had capacity)
- Machine: `a2-highgpu-1g` (1x NVIDIA A100 40GB, 12 vCPU, 85GB RAM)
- Image: Ubuntu 22.04 LTS (plain; install CUDA/drivers yourself)
- Boot disk: 200GB pd-balanced
- Cost: ~$3-4/hr on-demand while RUNNING. $0 for GPU while STOPPED (small disk cost only).

## Cost safety (two layers)

1. **Idle auto-shutdown watchdog** (`idle-shutdown-startup.sh`): a systemd service
   checks GPU utilization every minute and STOPS the VM after N consecutive
   minutes below 15% util (default 45). Training keeps the GPU busy, so it will
   not fire mid-run. Tunable via instance metadata:
   `idle-shutdown-minutes`, `idle-shutdown-threshold-percent`.
   FAIL-SAFE: if `nvidia-smi` is missing or returns no valid reading (e.g. driver
   still installing), it treats the VM as BUSY and will NOT shut down. (An earlier
   version had a bug where the driver-install period read as 0% idle and stopped
   the VM mid-setup; fixed.)
2. **Max-run backstop**: `--max-run-duration=8h --instance-termination-action=STOP`
   stops the VM after 8h even if the watchdog fails.

Both STOP (not delete), so the disk and your work are preserved.

## Everyday commands

```bash
PROJECT=emmanuel-genai-sa
ZONE=us-central1-f
NAME=cliagent-rlpt-a100

# Start (when you want to work)
gcloud compute instances start $NAME --zone=$ZONE --project=$PROJECT

# SSH in
gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT

# Stop (ALWAYS when done; the watchdog also does this after idle)
gcloud compute instances stop $NAME --zone=$ZONE --project=$PROJECT

# Check status
gcloud compute instances describe $NAME --zone=$ZONE --project=$PROJECT \
  --format='value(status)'

# Watch the idle watchdog log (on the VM)
sudo journalctl -t idle-shutdown -f
```

## First-time setup on the VM (plain Ubuntu -> ready to train)

```bash
# 1. NVIDIA driver + CUDA (Ubuntu 22.04)
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot        # then SSH back in
nvidia-smi         # confirm the A100 is visible

# 2. uv (Python package manager) + repo
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/awaemmanuel/rl-post-training.git
cd rl-post-training
uv sync

# 3. Run Phase 1
uv run python phase1-dpo/scripts/train_sft.py  --config phase1-dpo/configs/sft.yaml
uv run python phase1-dpo/scripts/train_dpo.py  --config phase1-dpo/configs/dpo.yaml
uv run python phase1-dpo/scripts/eval_winrate.py \
    --baseline outputs/sft-qwen2.5-1.5b --policy outputs/dpo-qwen2.5-1.5b \
    --prompts phase1-dpo/eval_prompts.jsonl --out outputs/eval
```

## Notes

- The VM starts STOPPED-cheap: stop it whenever you step away. The watchdog is a
  safety net, not a substitute for stopping it yourself.
- To change the idle timeout without recreating the VM:
  `gcloud compute instances add-metadata $NAME --zone=$ZONE --project=$PROJECT \
     --metadata=idle-shutdown-minutes=20` then restart the watchdog on the VM:
  `sudo systemctl restart idle-shutdown`.
