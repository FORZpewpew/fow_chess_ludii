#!/usr/bin/env python3
"""Minimal PPO inference server for Java PPOAgent.
Reads JSON requests from stdin, writes action JSON to stdout.
"""
import os, sys, json, torch

sys.path.insert(0, '/Users/forzpewpew/Downloads/ludii')
from ppo.policy import FoWPolicy

SELFPLAY_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "ppo_selfplay_policy.pt")
OLD_CKPT      = os.path.join(os.path.dirname(__file__), "checkpoints", "ppo_policy.pt")
NUM_OBS       = 128  # 64 cells × 2 channels (owner + piece_type)
NUM_ACTIONS   = 4096

model = FoWPolicy(NUM_OBS, NUM_ACTIONS)
ckpt_path = SELFPLAY_CKPT if os.path.exists(SELFPLAY_CKPT) else OLD_CKPT
sys.stderr.write(f"[PPO-Server] Loading checkpoint: {ckpt_path}\n")
sys.stderr.flush()
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()

# Signal ready to the Java parent process
print("READY", flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req   = json.loads(line)
        obs   = torch.tensor(req['obs'],   dtype=torch.float32).unsqueeze(0)
        legal = torch.tensor(req['legal'], dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = model.select_action(obs, legal)
        print(json.dumps({"action": int(action)}), flush=True)
    except Exception as e:
        # Return action=0 on any error so the Java side can fall back gracefully
        sys.stderr.write(f"[ppo_server] ERROR: {e}\n")
        sys.stderr.flush()
        print(json.dumps({"action": 0}), flush=True)
