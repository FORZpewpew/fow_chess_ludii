"""
FoWChessEnv — Fog-of-War Chess environment bridging Python PPO ↔ Ludii Java via JPype1.

State representation (per player perspective):
  - 2 channels × 64 squares = 128-dim flat float32 vector:
      channel 0 (indices 0–63):   ownership (egocentric, 4 discrete values):
                   0.000 — square hidden under fog (unknown state)
                   0.333 — confirmed empty visible square
                   0.667 — square occupied by the observing player's own piece
                   1.000 — square visibly occupied by the opponent
      channel 1 (indices 64–127): piece_type (0–6, Ludii component index) normalized by /6;
                   0 if hidden/absent

  Note: the hidden channel (formerly channel 2) has been removed because owner == 0.0
  is a lossless indicator of hidden squares (4-value encoding already encodes fog).

Action space:
  - 64*64 = 4096 discrete actions (from_site * 64 + to_site)
  - Illegal actions are masked during sampling

Rewards:
  - +1.0 at game end if this player wins
  - -1.0 at game end if this player loses
  - 0.0 on draw or move-limit
  - 0.0 for all intermediate steps (sparse reward)
"""
import numpy as np
import jpype
import jpype.imports

JAVA_HOME  = "/Users/forzpewpew/.asdf/installs/java/openjdk-26"
LUDII_JAR  = "/Users/forzpewpew/Downloads/ludii/Ludii-1.3.14.jar"
AGENTS_JAR = "/Users/forzpewpew/Downloads/ludii/agents/jars/agents.jar"
GAME_PATH  = "/Users/forzpewpew/Downloads/ludii/FoW_Chess.lud"
MAX_PLIES  = 400

OBS_DIM    = 128   # 64 cells × 2 channels (owner + piece_type)
ACTION_DIM = 4096  # 64 × 64


def start_jvm():
    if not jpype.isJVMStarted():
        import os
        # Use JVM library directly from JAVA_HOME to avoid /usr/libexec/java_home lookup
        jvm_path = os.path.join(JAVA_HOME, "lib", "server", "libjvm.dylib")
        if not os.path.exists(jvm_path):
            for candidate in [
                os.path.join(JAVA_HOME, "lib", "libjvm.dylib"),
                os.path.join(JAVA_HOME, "lib", "server", "libjvm.so"),
                os.path.join(JAVA_HOME, "lib", "libjvm.so"),
            ]:
                if os.path.exists(candidate):
                    jvm_path = candidate
                    break
        jpype.startJVM(
            jvm_path,
            f"-Djava.class.path={LUDII_JAR}:{AGENTS_JAR}",
            f"-Djava.home={JAVA_HOME}",
            "-Xmx2g",
            convertStrings=False,
        )


class FoWChessEnv:
    """
    Single-agent view of FoW Chess.

    `player` is 1 or 2 — which side this env controls.
    `opponent_fn` is an optional callable(obs: np.ndarray, legal_mask: np.ndarray) -> int
        that returns the action index for the opponent. When None the opponent
        picks uniformly at random from legal moves.
    """
    def __init__(self, player: int = 1, opponent_fn=None):
        start_jvm()
        from other import GameLoader
        from other.context import Context
        from other.trial import Trial
        from java.io import File

        self.GameLoader  = GameLoader
        self.Context     = Context
        self.Trial       = Trial
        self.File        = File

        self.player      = player
        self.opponent_fn = opponent_fn
        self.game        = GameLoader.loadGameFromFile(File(GAME_PATH))
        assert self.game is not None, f"Failed to load game from {GAME_PATH}"

        self.rng     = np.random.default_rng()
        self.context = None
        self.trial   = None
        self.ply     = 0

    def reset(self):
        """Start a new game, return initial observation for `self.player`."""
        self.trial   = self.Trial(self.game)
        self.context = self.Context(self.game, self.trial)
        self.game.start(self.context)
        self.ply = 0
        self._advance_opponent()
        return self._observe(), self._legal_mask()

    def step(self, action_idx: int):
        """
        Apply action_idx (from*64 + to) for self.player.
        Returns (obs, reward, done, info).
        """
        from_site = action_idx // 64
        to_site   = action_idx  % 64

        moves = list(self.game.moves(self.context).moves())
        chosen = None
        for m in moves:
            if m.fromNonDecision() == from_site and m.toNonDecision() == to_site:
                chosen = m
                break
        if chosen is None:
            # Illegal action — fall back to a random legal move (shouldn't happen if mask used)
            chosen = moves[self.rng.integers(len(moves))]

        self.game.apply(self.context, chosen)
        self.ply += 1

        if self.trial.over() or self.ply >= MAX_PLIES:
            return self._observe(), self._outcome_reward(), True, {}

        self._advance_opponent()

        if self.trial.over() or self.ply >= MAX_PLIES:
            return self._observe(), self._outcome_reward(), True, {}

        return self._observe(), 0.0, False, {}

    def _observe_for(self, player: int) -> np.ndarray:
        """Return 128-dim float32 observation from *any* player's perspective.

        Layout:
          obs[i]      (i = 0..63)  — ownership, egocentric 4-value encoding:
                         0.000  hidden square (fog)
                         1/3    confirmed empty visible square
                         2/3    square occupied by this player's own piece
                         1.000  square visibly occupied by the opponent
          obs[64+i]   (i = 0..63) — piece type / 6.0  (0 if hidden or absent)
        """
        owner_ch = np.zeros(64, dtype=np.float32)
        piece_ch = np.zeros(64, dtype=np.float32)
        if self.trial.over():
            return np.concatenate([owner_ch, piece_ch])
        state = self.context.state()
        cs    = state.containerStates()[0]
        from game.types.board import SiteType
        for site in range(64):
            owner      = int(cs.who(site, SiteType.Cell))
            piece_type = int(cs.what(site, SiteType.Cell))
            hidden     = bool(cs.isHidden(player, site, 0, SiteType.Cell))
            if hidden:
                pass  # both channels stay 0.0 (fog sentinel)
            else:
                if owner == 0:
                    owner_ch[site] = 1.0 / 3.0
                elif owner == player:
                    owner_ch[site] = 2.0 / 3.0
                else:
                    owner_ch[site] = 1.0
                piece_ch[site] = float(piece_type) / 6.0
        return np.concatenate([owner_ch, piece_ch])

    def _legal_mask_for(self, player: int) -> np.ndarray:
        """Return bool mask of shape (4096,) for *any* player's legal moves."""
        mask  = np.zeros(ACTION_DIM, dtype=bool)
        if self.trial.over():
            return mask
        if int(self.context.state().mover()) != player:
            return mask
        for m in self.game.moves(self.context).moves():
            f = int(m.fromNonDecision())
            t = int(m.toNonDecision())
            if 0 <= f < 64 and 0 <= t < 64:
                mask[f * 64 + t] = True
        return mask

    def _advance_opponent(self):
        """Play moves for the non-`self.player` side until it's our turn.

        Uses self.opponent_fn(obs, legal_mask) -> action_idx when provided,
        otherwise falls back to uniform-random.
        """
        opp_player = 3 - self.player
        while not self.trial.over() and self.ply < MAX_PLIES:
            if int(self.context.state().mover()) == self.player:
                break
            moves = list(self.game.moves(self.context).moves())
            if not moves:
                break

            if self.opponent_fn is not None:
                opp_obs  = self._observe_for(opp_player)
                opp_mask = self._legal_mask_for(opp_player)
                action_idx = self.opponent_fn(opp_obs, opp_mask)
                from_site  = action_idx // 64
                to_site    = action_idx  % 64
                chosen = None
                for m in moves:
                    if m.fromNonDecision() == from_site and m.toNonDecision() == to_site:
                        chosen = m
                        break
                if chosen is None:
                    chosen = moves[self.rng.integers(len(moves))]
            else:
                chosen = moves[self.rng.integers(len(moves))]

            self.game.apply(self.context, chosen)
            self.ply += 1

    def _observe(self) -> np.ndarray:
        return self._observe_for(self.player)

    def _legal_mask(self) -> np.ndarray:
        return self._legal_mask_for(self.player)

    def _outcome_reward(self) -> float:
        ranking = self.trial.ranking()
        if ranking is None or len(ranking) <= 2:
            return 0.0
        r1, r2 = float(ranking[1]), float(ranking[2])
        if self.player == 1:
            if r1 < r2: return 1.0
            if r1 > r2: return -1.0
        else:
            if r2 < r1: return 1.0
            if r2 > r1: return -1.0
        return 0.0
