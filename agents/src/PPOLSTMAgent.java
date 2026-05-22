package agents;

import game.Game;
import game.types.board.SiteType;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.move.Move;
import other.state.State;

import java.io.*;
import java.nio.charset.StandardCharsets;

/**
 * PPOLSTMAgent — wraps the trained PPO-LSTM policy checkpoint via a Python subprocess.
 *
 * <p>On {@link #initAI} it spawns {@code ppo/venv/bin/python ppo/ppo_lstm_server.py --checkpoint <file>}
 * (relative to the JVM working directory, i.e. the ludii project root),
 * waits for the "READY" handshake, then sends a {@code {"type":"new_game"}} message to reset
 * the LSTM hidden state. On each {@link #selectAction} call it:
 * <ol>
 *   <li>Builds the 128-float observation from the Ludii {@link Context}.</li>
 *   <li>Builds the 4096-bool legal-action mask.</li>
 *   <li>Sends a JSON line {@code {"type":"move","obs":[...],"legal":[...]}} to the subprocess stdin.</li>
 *   <li>Reads the JSON line reply and extracts the action index.</li>
 *   <li>Maps the action index back to the matching legal {@link Move}.</li>
 * </ol>
 *
 * <p>Observation format matches {@code ppo/fow_env.py#_observe()} exactly.
 * For each site 0..63:
 * <ul>
 *   <li>obs[site]     = owner (egocentric 4-value: 0=hidden, 1/3=empty, 2/3=self, 1=opponent)</li>
 *   <li>obs[site+64]  = piece / 6.0f (0 if hidden or absent)</li>
 * </ul>
 *
 * <p>{@code isHidden} is called with the 1-based {@code playerId}, matching
 * {@code fow_env.py}: {@code cs.isHidden(self.player, site, 0, SiteType.Cell)}
 * where {@code self.player ∈ {1, 2}}.
 */
public class PPOLSTMAgent extends AI {

    private static final String PYTHON_BIN       = "ppo/venv/bin/python";
    private static final String SERVER_SCRIPT    = "ppo/ppo_lstm_server.py";
    private static final int    NUM_OBS          = 128;
    private static final int    NUM_ACTIONS      = 4096;
    private static final long   READY_TIMEOUT_MS = 60_000L;

    private final String checkpointFile;

    private Process        serverProcess;
    private PrintWriter    toServer;
    private BufferedReader fromServer;
    private int            playerId;

    public PPOLSTMAgent() {
        this.checkpointFile = "checkpoints/ppo_lstm_v4_policy.pt";
        friendlyName = "PPO-LSTM Agent";
    }

    /**
     * @param ckpt path to the checkpoint .pt file (relative to JVM working dir)
     */
    public PPOLSTMAgent(final String ckpt) {
        this.checkpointFile = (ckpt != null && !ckpt.isEmpty())
                ? ckpt
                : "checkpoints/ppo_lstm_v4_policy.pt";
        friendlyName = "PPO-LSTM Agent";
    }

    @Override
    public void initAI(final Game game, final int playerID) {
        this.playerId = playerID;

        // Tear down any previously running server (e.g. from a previous game)
        if (serverProcess != null && serverProcess.isAlive()) {
            serverProcess.destroy();
            serverProcess = null;
        }

        try {
            final ProcessBuilder pb = new ProcessBuilder(
                    PYTHON_BIN, SERVER_SCRIPT, "--checkpoint", checkpointFile);
            pb.directory(new File(System.getProperty("user.dir")));
            serverProcess = pb.start();

            toServer = new PrintWriter(
                    new BufferedWriter(new OutputStreamWriter(
                            serverProcess.getOutputStream(), StandardCharsets.UTF_8)));
            fromServer = new BufferedReader(
                    new InputStreamReader(
                            serverProcess.getInputStream(), StandardCharsets.UTF_8));

            final InputStream errStream = serverProcess.getErrorStream();
            final Thread errFwd = new Thread(() -> {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(errStream, StandardCharsets.UTF_8))) {
                    String ln;
                    while ((ln = br.readLine()) != null) {
                        System.err.println("[ppo_lstm_server] " + ln);
                    }
                } catch (final IOException ignored) {}
            }, "ppo-lstm-server-stderr");
            errFwd.setDaemon(true);
            errFwd.start();

            final long deadline = System.currentTimeMillis() + READY_TIMEOUT_MS;
            String handshake = null;
            while (System.currentTimeMillis() < deadline) {
                if (!serverProcess.isAlive()) {
                    throw new RuntimeException(
                            "PPO-LSTM server process died before printing READY (exit code "
                            + serverProcess.exitValue() + ")");
                }
                if (fromServer.ready()) {
                    handshake = fromServer.readLine();
                    break;
                }
                Thread.sleep(100);
            }
            if (!"READY".equals(handshake)) {
                throw new RuntimeException(
                        "PPO-LSTM server did not send READY within " + (READY_TIMEOUT_MS / 1000)
                        + " s; got: " + handshake);
            }
            System.out.println("[PPOLSTMAgent] Python inference server ready (player " + playerID + ").");

            // Reset LSTM hidden state for the new game
            toServer.println("{\"type\":\"new_game\"}");
            toServer.flush();
            final String newGameResponse = fromServer.readLine();
            if (newGameResponse == null || !newGameResponse.contains("\"ok\"")) {
                System.err.println("[PPOLSTMAgent] WARNING: unexpected new_game response: "
                        + newGameResponse);
            }

        } catch (final Exception e) {
            throw new RuntimeException("PPOLSTMAgent.initAI failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Move selectAction(
            final Game game,
            final Context context,
            final double maxSeconds,
            final int maxIterations,
            final int maxDepth)
    {
        final FastArrayList<Move> moves = game.moves(context).moves();
        if (moves.isEmpty()) return null;

        try {
            final float[]   obs   = buildObservation(context);
            final boolean[] legal = buildLegalMask(moves);

            toServer.println(buildJson(obs, legal));
            toServer.flush();

            final String response = fromServer.readLine();
            if (response == null) {
                System.err.println("[PPOLSTMAgent] Server stdout closed — using fallback.");
                return moves.get(0);
            }

            return resolveMove(parseAction(response), moves);

        } catch (final Exception e) {
            System.err.println("[PPOLSTMAgent] selectAction error: " + e.getMessage());
            return moves.get(0);
        }
    }

    @Override
    public void closeAI() {
        if (toServer != null) {
            try { toServer.close(); } catch (final Exception ignored) {}
        }
        if (serverProcess != null && serverProcess.isAlive()) {
            serverProcess.destroy();
        }
    }

    /**
     * Builds the 128-float observation matching {@code fow_env.py#_observe()}.
     *
     * <p>obs[site] (0..63): ownership — 0=hidden, 1/3=empty, 2/3=self, 1=opponent<br>
     * obs[site+64] (0..63): piece_type / 6.0 (0 if hidden or absent)
     */
    private float[] buildObservation(final Context context) {
        final float[] obs   = new float[NUM_OBS];
        final State   state = context.state();

        for (int site = 0; site < 64; site++) {
            final int     owner  = state.containerStates()[0].who(site,  SiteType.Cell);
            final int     piece  = state.containerStates()[0].what(site, SiteType.Cell);
            final boolean hidden = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);

            if (!hidden) {
                if (owner == 0) {
                    obs[site] = 1.0f / 3.0f;
                } else if (owner == playerId) {
                    obs[site] = 2.0f / 3.0f;
                } else {
                    obs[site] = 1.0f;
                }
                obs[site + 64] = piece / 6.0f;
            }
            // hidden: both channels remain 0.0f
        }
        return obs;
    }

    private static boolean[] buildLegalMask(final FastArrayList<Move> moves) {
        final boolean[] legal = new boolean[NUM_ACTIONS];
        for (int i = 0; i < moves.size(); i++) {
            final Move m    = moves.get(i);
            final int  from = m.fromNonDecision();
            final int  to   = m.toNonDecision();
            if (from >= 0 && from < 64 && to >= 0 && to < 64) {
                legal[from * 64 + to] = true;
            }
        }
        return legal;
    }

    /**
     * Maps an action index back to a legal {@link Move}.
     * Falls back to the first legal move when no match is found.
     */
    private static Move resolveMove(final int actionIdx, final FastArrayList<Move> moves) {
        final int from = actionIdx / 64;
        final int to   = actionIdx % 64;
        for (int i = 0; i < moves.size(); i++) {
            final Move m = moves.get(i);
            if (m.fromNonDecision() == from && m.toNonDecision() == to) {
                return m;
            }
        }
        System.err.println("[PPOLSTMAgent] action=" + actionIdx
                + " (from=" + from + ",to=" + to + ") not in legal moves — fallback.");
        return moves.get(0);
    }

    /**
     * Builds the move request JSON with a {@code "type":"move"} wrapper required
     * by the LSTM server protocol (distinguishes move requests from new_game resets).
     */
    private static String buildJson(final float[] obs, final boolean[] legal) {
        final StringBuilder sb = new StringBuilder(NUM_OBS * 12 + NUM_ACTIONS * 6 + 32);
        sb.append("{\"type\":\"move\",\"obs\":[");
        for (int i = 0; i < obs.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(obs[i]);
        }
        sb.append("],\"legal\":[");
        for (int i = 0; i < legal.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(legal[i] ? "true" : "false");
        }
        sb.append("]}");
        return sb.toString();
    }

    /** Parses {@code {"action": N}} — hand-rolled to avoid a JSON library dependency. */
    private static int parseAction(final String json) {
        final int keyIdx = json.indexOf("\"action\"");
        if (keyIdx < 0) {
            throw new IllegalArgumentException("No 'action' key in: " + json);
        }
        int pos = json.indexOf(':', keyIdx) + 1;
        while (pos < json.length() && json.charAt(pos) == ' ') pos++;
        int end = pos;
        while (end < json.length()
               && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-')) {
            end++;
        }
        return Integer.parseInt(json.substring(pos, end));
    }
}
