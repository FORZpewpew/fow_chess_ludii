"""
lstm_belief_server.py — TCP belief-oracle server for FoW Chess IS-MCTS.

Listens on a TCP port (default 9998) and maintains per-connection LSTM
hidden state (h, c) to answer belief queries from LSTMGuidedISMCTSAgent.

Protocol (newline-delimited JSON over TCP):

  Client → Server:
    {"cmd": "reset"}
        → {"status": "ok"}
        Reset the per-connection LSTM hidden state to zeros.

    {"cmd": "obs", "obs": [128 floats]}
        → {"status": "ok"}
        Feed the observation through the LSTM encoder and advance (h, c).
        The hidden state is NOT reset between calls; the server accumulates
        the full game history, matching how the PPO-LSTM policy was trained.

    {"cmd": "belief"}
        → {"probs": [[p0, p1, ..., p6], ...] }   (64 inner lists, 7 floats each)
        Run the belief probe on the current (h, c) hidden state.
        Returns softmax probabilities for each of 64 squares over 7 piece types:
          index 0 = empty
          index 1 = pawn
          index 2 = rook
          index 3 = bishop
          index 4 = knight
          index 5 = queen
          index 6 = king

Architecture (loaded from checkpoints):
  - FoWPolicyLSTM encoder + LSTM from checkpoints/ppo_lstm_v4_policy.pt
  - BeliefProbeHead from checkpoints/belief_probe_v4.pt
  Only the encoder, LSTM, and probe are used; actor/critic heads are not needed.

Usage:
  cd fow_chess_ludii
  python ppo/lstm_belief_server.py [--port 9998] [--host localhost]
      [--lstm-ckpt checkpoints/ppo_lstm_v4_policy.pt]
      [--probe-ckpt checkpoints/belief_probe_v4.pt]
"""

import sys
import os
import json
import argparse
import socket
import threading

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from policy_lstm import FoWPolicyLSTM
from belief_probe import BeliefProbeHead


def load_backbone(ckpt_path: str, device) -> FoWPolicyLSTM:
    """Load and freeze the FoWPolicyLSTM backbone (encoder + LSTM only)."""
    policy = FoWPolicyLSTM()

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Normalise nested state-dict wrappers (same logic as ppo_lstm_server.py)
            if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
                sd = ckpt["policy_state_dict"]
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            else:
                sd = ckpt

            # Auto-detect hidden_dim and obs_dim from checkpoint
            encoder_weight = sd.get("encoder.0.weight")
            lstm_keys = [k for k in sd if "lstm.weight_ih_l0" in k]
            detected_obs_dim = None
            detected_hidden_dim = None
            if encoder_weight is not None:
                detected_obs_dim = int(encoder_weight.shape[1])
            if lstm_keys:
                detected_hidden_dim = sd[lstm_keys[0]].shape[0] // 4

            if detected_obs_dim or detected_hidden_dim:
                policy = FoWPolicyLSTM(
                    obs_dim=detected_obs_dim or 128,
                    hidden_dim=detected_hidden_dim or 512,
                )
                print(
                    f"[lstm_belief_server] Auto-detected obs_dim={detected_obs_dim or 128}, "
                    f"hidden_dim={detected_hidden_dim or 512}",
                    file=sys.stderr, flush=True,
                )

            missing, unexpected = policy.load_state_dict(sd, strict=False)
            if missing:
                print(
                    f"[lstm_belief_server] Checkpoint loaded (strict=False): "
                    f"missing keys = {missing}",
                    file=sys.stderr, flush=True,
                )
            else:
                print(
                    f"[lstm_belief_server] LSTM backbone loaded from {ckpt_path}",
                    file=sys.stderr, flush=True,
                )
        except Exception as e:
            print(
                f"[lstm_belief_server] WARNING: Failed to load backbone from {ckpt_path}: {e}. "
                f"Using random weights.",
                file=sys.stderr, flush=True,
            )
    else:
        print(
            f"[lstm_belief_server] WARNING: Backbone checkpoint not found at {ckpt_path}. "
            f"Using random weights.",
            file=sys.stderr, flush=True,
        )

    # Freeze all parameters — inference only
    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()
    return policy.to(device)


def load_probe(ckpt_path: str, device) -> BeliefProbeHead:
    """Load the trained BeliefProbeHead."""
    probe = BeliefProbeHead()

    if os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
            probe.load_state_dict(sd)
            print(
                f"[lstm_belief_server] Belief probe loaded from {ckpt_path}",
                file=sys.stderr, flush=True,
            )
        except Exception as e:
            print(
                f"[lstm_belief_server] WARNING: Failed to load probe from {ckpt_path}: {e}. "
                f"Using random weights.",
                file=sys.stderr, flush=True,
            )
    else:
        print(
            f"[lstm_belief_server] WARNING: Probe checkpoint not found at {ckpt_path}. "
            f"Using random weights.",
            file=sys.stderr, flush=True,
        )

    for p in probe.parameters():
        p.requires_grad_(False)
    probe.eval()
    return probe.to(device)


class ConnectionHandler(threading.Thread):
    """
    Handles one TCP connection.

    Each connection gets its own LSTM hidden state (h, c) so multiple games
    can run concurrently without interfering with each other.
    """

    def __init__(self, conn: socket.socket, addr, backbone: FoWPolicyLSTM,
                 probe: BeliefProbeHead, device):
        super().__init__(daemon=True)
        self.conn     = conn
        self.addr     = addr
        self.backbone = backbone
        self.probe    = probe
        self.device   = device

        # Per-connection LSTM hidden state — shape (1, 1, hidden_dim)
        self.h, self.c = backbone.init_hidden(batch_size=1, device=device)

    def run(self):
        print(
            f"[lstm_belief_server] Connection from {self.addr}",
            file=sys.stderr, flush=True,
        )
        try:
            buf = b""
            while True:
                chunk = self.conn.recv(4096)
                if not chunk:
                    break  # client closed connection
                buf += chunk

                # Process all complete newline-terminated messages in the buffer
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        response = self._handle_message(line.decode("utf-8"))
                        self.conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
                    except Exception as e:
                        err_resp = {"status": "error", "message": str(e)}
                        print(
                            f"[lstm_belief_server] Error handling message: {e}",
                            file=sys.stderr, flush=True,
                        )
                        try:
                            self.conn.sendall((json.dumps(err_resp) + "\n").encode("utf-8"))
                        except Exception:
                            break
        except Exception as e:
            print(
                f"[lstm_belief_server] Connection {self.addr} error: {e}",
                file=sys.stderr, flush=True,
            )
        finally:
            try:
                self.conn.close()
            except Exception:
                pass
            print(
                f"[lstm_belief_server] Connection {self.addr} closed.",
                file=sys.stderr, flush=True,
            )

    def _handle_message(self, raw: str) -> dict:
        req = json.loads(raw)
        cmd = req.get("cmd", "")

        if cmd == "reset":
            # Reset LSTM hidden state to zeros for a new game
            self.h, self.c = self.backbone.init_hidden(batch_size=1, device=self.device)
            return {"status": "ok"}

        elif cmd == "obs":
            # Advance the LSTM hidden state with the given observation
            obs_list = req["obs"]
            obs_t = torch.tensor(obs_list, dtype=torch.float32, device=self.device).unsqueeze(0)
            # obs_t: (1, obs_dim)
            with torch.no_grad():
                # encoder: (1, obs_dim) → (1, hidden_dim)
                enc = self.backbone.encoder(obs_t)
                # lstm: input (1, 1, hidden_dim), hidden (1, 1, hidden_dim)
                out, (self.h, self.c) = self.backbone.lstm(enc.unsqueeze(1), (self.h, self.c))
                # out: (1, 1, hidden_dim), squeeze to (1, hidden_dim)
                # We keep h/c updated but don't need the output here
            return {"status": "ok"}

        elif cmd == "belief":
            # Run belief probe on current hidden state, return per-square probabilities
            with torch.no_grad():
                # h shape: (1, 1, hidden_dim); squeeze batch dim → (1, hidden_dim)
                h_t = self.h.squeeze(0)   # (1, hidden_dim)
                logits = self.probe(h_t)   # (1, 64, 7)
                # Softmax over piece-type dimension (dim=-1)
                probs = F.softmax(logits, dim=-1)  # (1, 64, 7)
                probs_np = probs.squeeze(0).cpu().tolist()  # list of 64 × list of 7
            return {"probs": probs_np}

        else:
            raise ValueError(f"Unknown command: '{cmd}'. Expected: reset, obs, belief")


def main():
    parser = argparse.ArgumentParser(description="LSTM Belief Oracle Server for FoW Chess")
    parser.add_argument("--port",       type=int,   default=9998,
                        help="TCP port to listen on (default: 9998)")
    parser.add_argument("--host",       type=str,   default="localhost",
                        help="Host/interface to bind (default: localhost)")
    parser.add_argument("--lstm-ckpt",  type=str,
                        default="checkpoints/ppo_lstm_v4_policy.pt",
                        help="Path to PPO-LSTM policy checkpoint (.pt)")
    parser.add_argument("--probe-ckpt", type=str,
                        default="checkpoints/belief_probe_v4.pt",
                        help="Path to belief probe checkpoint (.pt)")
    args = parser.parse_args()

    print("[lstm_belief_server] Starting...", file=sys.stderr, flush=True)

    # Device selection: prefer MPS (Apple Silicon) → CUDA → CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[lstm_belief_server] Using device: {device}", file=sys.stderr, flush=True)

    backbone = load_backbone(args.lstm_ckpt, device)
    probe    = load_probe(args.probe_ckpt, device)

    # Warmup: run one dummy forward pass to trigger MPS JIT compilation before
    # any client connects.  Without this, the first obs/belief call from Java
    # can take 30–60 s on Apple Silicon, causing the Java socket to time out.
    print("[lstm_belief_server] Running MPS warmup inference...", file=sys.stderr, flush=True)
    try:
        with torch.no_grad():
            # Infer obs_dim from first encoder layer weight shape (out_features × in_features)
            enc0_weight = next(backbone.encoder.parameters())  # shape: (hidden, obs_dim)
            obs_dim_warmup = enc0_weight.shape[1]
            dummy_obs = torch.zeros(1, obs_dim_warmup, device=device)
            enc = backbone.encoder(dummy_obs)
            h0 = torch.zeros(1, 1, backbone.lstm.hidden_size, device=device)
            c0 = torch.zeros(1, 1, backbone.lstm.hidden_size, device=device)
            out, (h1, c1) = backbone.lstm(enc.unsqueeze(1), (h0, c0))
            logits = probe(h1.squeeze(0))
            _ = torch.nn.functional.softmax(logits, dim=-1)
        print("[lstm_belief_server] MPS warmup complete.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[lstm_belief_server] Warmup failed (non-fatal): {e}", file=sys.stderr, flush=True)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(16)

    print(
        f"[lstm_belief_server] Listening on {args.host}:{args.port}",
        file=sys.stderr, flush=True,
    )

    try:
        while True:
            conn, addr = server_sock.accept()
            handler = ConnectionHandler(conn, addr, backbone, probe, device)
            handler.start()
    except KeyboardInterrupt:
        print("[lstm_belief_server] Shutting down.", file=sys.stderr, flush=True)
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
