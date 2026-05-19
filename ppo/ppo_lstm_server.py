"""
ppo_lstm_server.py — Stateful LSTM inference server for FoW Chess.

Protocol (stdin → stdout, newline-delimited JSON):
  {"type": "new_game"}
      → {"status": "ok"}          Reset LSTM hidden state to zeros.
  {"type": "move", "obs": [...128 floats...], "legal": [...4096 bools...]}
      → {"action": N}             Sample action; advance hidden state.

CRITICAL: ALL informational output goes to stderr. stdout must contain ONLY
the protocol JSON lines and the initial "READY\\n" sentinel — any contamination
crashes the Java subprocess handshake in PPOAgent.java.
"""

import sys
import json
import argparse

import torch
import numpy as np
from policy_lstm import FoWPolicyLSTM, OBS_DIM


def main():
    parser = argparse.ArgumentParser(description="PPO-LSTM inference server")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ppo_lstm_policy.pt",
        help="Path to the policy checkpoint (.pt file).",
    )
    args = parser.parse_args()

    sys.stderr.write("PPO-LSTM server starting...\n"); sys.stderr.flush()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    sys.stderr.write(f"Using device: {device}\n"); sys.stderr.flush()

    checkpoint_path = args.checkpoint

    # Auto-detect checkpoint architecture so inference matches the saved model,
    # including legacy 192-dim observations.
    detected_hidden_dim = None
    detected_obs_dim    = None
    ckpt = None
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Normalise nested state-dict wrappers
        if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
            sd = ckpt["policy_state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
        encoder_weight = sd.get("encoder.0.weight")
        if encoder_weight is not None:
            detected_obs_dim = int(encoder_weight.shape[1])
        lstm_keys = [k for k in sd if "lstm.weight_ih_l0" in k]
        if lstm_keys:
            detected_hidden_dim = sd[lstm_keys[0]].shape[0] // 4
        if detected_obs_dim is not None or detected_hidden_dim is not None:
            sys.stderr.write(
                f"Auto-detected obs_dim={detected_obs_dim if detected_obs_dim is not None else OBS_DIM}, "
                f"hidden_dim={detected_hidden_dim if detected_hidden_dim is not None else 'default'} "
                f"from {checkpoint_path}\n"
            )
            sys.stderr.flush()
    except FileNotFoundError:
        sys.stderr.write(f"No checkpoint found at {checkpoint_path} — using random weights\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"Checkpoint pre-load error: {e} — using default hidden_dim\n")
        sys.stderr.flush()

    policy = FoWPolicyLSTM(
        obs_dim=detected_obs_dim if detected_obs_dim is not None else OBS_DIM,
        hidden_dim=detected_hidden_dim if detected_hidden_dim is not None else FoWPolicyLSTM().hidden_dim,
    ).to(device)

    if ckpt is not None:
        try:
            if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
                sd = ckpt["policy_state_dict"]
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            else:
                sd = ckpt
            # strict=False: v3 checkpoints lack critic_encoder/critic_head keys
            # which are unused during inference.
            missing, _ = policy.load_state_dict(sd, strict=False)
            if missing:
                sys.stderr.write(
                    f"Checkpoint loaded (strict=False): missing keys {missing} "
                    f"— inference layers intact\n"
                )
            else:
                sys.stderr.write(f"Checkpoint loaded: {checkpoint_path}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"Checkpoint load error: {e} — using random weights\n")
            sys.stderr.flush()

    policy.eval()

    h, c = policy.init_hidden(batch_size=1, device=device)

    sys.stdout.write("READY\n"); sys.stdout.flush()

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        req      = json.loads(raw_line)
        msg_type = req.get("type", "move")

        if msg_type == "new_game":
            h, c = policy.init_hidden(batch_size=1, device=device)
            sys.stdout.write(json.dumps({"status": "ok"}) + "\n")
            sys.stdout.flush()

        else:  # "move"
            obs_t   = torch.tensor(req["obs"],   dtype=torch.float32, device=device).unsqueeze(0)
            legal_t = torch.tensor(req["legal"], dtype=torch.bool,    device=device).unsqueeze(0)

            if obs_t.shape[-1] != policy.obs_dim:
                raise RuntimeError(
                    f"Observation dimension mismatch: request has {obs_t.shape[-1]} values, "
                    f"but checkpoint/model expects {policy.obs_dim}"
                )
            if legal_t.shape[-1] != policy.act_dim:
                raise RuntimeError(
                    f"Legal-mask dimension mismatch: request has {legal_t.shape[-1]} values, "
                    f"but model expects {policy.act_dim}"
                )

            with torch.no_grad():
                dist, _, h, c = policy(obs_t, legal_t, h, c)
                action = dist.sample().item()

            sys.stdout.write(json.dumps({"action": action}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
