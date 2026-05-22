package agents;

import game.Game;
import game.types.board.SiteType;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.move.Move;
import other.state.State;
import search.mcts.MCTS;

import java.io.*;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * LSTMGuidedISMCTSAgent — IS-MCTS guided by a PPO-LSTM belief probe.
 *
 * <p>Instead of the particle-filter prior or uniform randomisation used by
 * {@link ParticleISMCTSAgent} and {@link ISMCTSAgent}, this agent queries a
 * running {@code lstm_belief_server.py} TCP server that maintains per-game
 * LSTM hidden state and returns per-square piece-type probability distributions
 * from a trained {@code BeliefProbeHead}.
 *
 * <h3>Algorithm per turn</h3>
 * <ol>
 *   <li>Build the 192-dim fog-of-war observation vector (3-channel format used
 *       during PPO-LSTM v4 training) and send it to the belief server to advance the
 *       LSTM hidden state.</li>
 *   <li>For each of {@value #NUM_DETS} determinizations:
 *     <ol>
 *       <li>Query the server for the current belief: {@code probs[64][7]}.</li>
 *       <li>Sample a board configuration from the per-square distributions
 *           (see {@link #sampleFromProbe}).</li>
 *       <li>Apply the sampled configuration to a copy of the context.</li>
 *       <li>Run UCT on the fully-observable copy and cast a vote.</li>
 *     </ol>
 *   </li>
 *   <li>Return the move with the most votes (plurality).</li>
 * </ol>
 *
 * <h3>Determinization sampling algorithm</h3>
 * <pre>
 * Given: probs[64][7] — softmax probability over piece types for each square.
 * Piece-type indices: 0=empty, 1=pawn, 2=rook, 3=bishop, 4=knight, 5=queen, 6=king.
 *
 * 1. Identify hidden squares: hidden from this player AND potentially occupied
 *    by opponent (i.e. owner==opponent OR owner==0 but hidden).
 * 2. For each hidden square, sample a piece type from probs[square].
 * 3. Enforce uniqueness constraints:
 *    - King (type 6): at most one. If sampled on multiple squares, keep the
 *      square with highest P(king) and set the rest to empty (type 0).
 *    - Queen (type 5): at most one (chess rule).
 *    - Rook (type 2): at most 2 etc. — enforced by dropping extras.
 * 4. Build particle: for each hidden square where sampled type != 0, record
 *    (site → pieceType) in the determinization map.
 * 5. Apply via the same setSite() calls used in ParticleISMCTSAgent.
 * </pre>
 *
 * <h3>Belief-accuracy logging (opt-in)</h3>
 * <p>Set {@code -Dfow.belief.log=true} to enable CSV logging.
 * The log path is controlled by {@code -Dfow.belief.logfile} (default:
 * {@code evaluation/results_lstm_guided/lstm_guided_belief_accuracy.csv}).
 *
 * <h3>Server connection</h3>
 * <p>The agent connects to {@code localhost:9998} at {@link #initAI} time
 * with up to 5 retry attempts (1 s apart). Start {@code lstm_belief_server.py}
 * before launching this agent. The PPO inference server uses port 9999
 * (subprocess stdin/stdout), so port 9998 is free for the belief oracle.
 */
public class LSTMGuidedISMCTSAgent extends AI {

    // -----------------------------------------------------------------------
    // Tuning constants
    // -----------------------------------------------------------------------

    /** Number of IS-MCTS determinizations per move. */
    private static final int NUM_DETS = 25;

    /** UCT simulations per determinization. */
    private static final int SIMS_PER_DET = 200;

    /** Wall-clock seconds per determinization (used when no total time given). */
    private static final double THINK_TIME_PER_DET = 0.04;

    /** Belief oracle server host. */
    private static final String SERVER_HOST = "localhost";

    /** Belief oracle server TCP port (separate from PPO inference server). */
    private static final int SERVER_PORT = 9998;

    /** Connection retry attempts and sleep between each. */
    private static final int    CONNECT_RETRIES    = 5;
    private static final long   CONNECT_RETRY_MS   = 1_000L;

    // Piece-type constants matching BeliefProbeHead output indices.
    private static final int PT_EMPTY  = 0;
    private static final int PT_PAWN   = 1;
    private static final int PT_ROOK   = 2;
    private static final int PT_BISHOP = 3;
    private static final int PT_KNIGHT = 4;
    private static final int PT_QUEEN  = 5;
    private static final int PT_KING   = 6;

    /** Max count for each piece type per side (standard chess). */
    private static final int[] MAX_PIECE_COUNT = {
        /* empty */ Integer.MAX_VALUE,
        /* pawn   */ 8,
        /* rook   */ 2,
        /* bishop */ 2,
        /* knight */ 2,
        /* queen  */ 1,
        /* king   */ 1,
    };

    /** Number of piece types the probe outputs (0..6). */
    private static final int NUM_PIECE_TYPES = 7;

    // -----------------------------------------------------------------------
    // Belief-logging configuration
    // -----------------------------------------------------------------------

    private static final boolean BELIEF_LOG_ENABLED;
    private static final String  BELIEF_LOG_FILE;

    static {
        BELIEF_LOG_ENABLED = "true".equalsIgnoreCase(
                System.getProperty("fow.belief.log", "false"));
        BELIEF_LOG_FILE = System.getProperty(
                "fow.belief.logfile",
                "evaluation/results_lstm_guided/lstm_guided_belief_accuracy.csv");
    }

    // -----------------------------------------------------------------------
    // Agent state
    // -----------------------------------------------------------------------

    private int  playerId  = -1;
    private MCTS uct       = null;
    private int  moveCount = 0;
    private long gameId    = 0L;

    // TCP socket to lstm_belief_server.py
    private Socket         serverSocket = null;
    private PrintWriter    toServer     = null;
    private BufferedReader fromServer   = null;

    /** Whether the server is available; if not we fall back to random particles. */
    private boolean serverConnected = false;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    public LSTMGuidedISMCTSAgent() {
        this.friendlyName = "LSTM-Guided-IS-MCTS";
    }

    // -----------------------------------------------------------------------
    // AI lifecycle
    // -----------------------------------------------------------------------

    @Override
    public void initAI(final Game game, final int playerID) {
        this.playerId  = playerID;
        this.moveCount = 0;
        this.gameId    = System.currentTimeMillis();

        // Tear down old MCTS instance
        if (uct != null) {
            try { uct.closeAI(); } catch (final Exception ignored) {}
            uct = null;
        }
        uct = MCTS.createUCT();
        uct.initAI(game, playerID);

        // Connect to the belief oracle server (with retry)
        connectToServer();

        // Send reset to synchronise hidden state for new game
        if (serverConnected) {
            sendReset();
        }

        if (BELIEF_LOG_ENABLED) {
            ensureBeliefLogHeader();
        }
    }

    @Override
    public void closeAI() {
        if (uct != null) {
            try { uct.closeAI(); } catch (final Exception ignored) {}
            uct = null;
        }
        // Send a final reset before closing the connection so the server-side
        // hidden state is cleared (good hygiene if the socket stays open).
        if (serverConnected) {
            try {
                sendReset();
            } catch (final Exception ignored) {}
        }
        disconnectFromServer();
    }

    // -----------------------------------------------------------------------
    // Move selection — main entry point
    // -----------------------------------------------------------------------

    @Override
    public Move selectAction(
            final Game    game,
            final Context context,
            final double  maxSeconds,
            final int     maxIterations,
            final int     maxDepth)
    {
        final FastArrayList<Move> legalMovesRaw = game.moves(context).moves();
        if (legalMovesRaw.isEmpty()) return null;
        if (legalMovesRaw.size() == 1) {
            moveCount++;
            return legalMovesRaw.get(0);
        }

        final List<Move> legalMoves = new ArrayList<>(legalMovesRaw.size());
        for (int i = 0; i < legalMovesRaw.size(); i++) {
            legalMoves.add(legalMovesRaw.get(i));
        }

        // ---------------------------------------------------------------
        // Step 1: Advance LSTM hidden state with the current observation
        // ---------------------------------------------------------------
        if (serverConnected) {
            final float[] obs = buildObservation(context);
            sendObs(obs);
        }

        // ---------------------------------------------------------------
        // Belief logging: capture ground truth BEFORE determinization
        // ---------------------------------------------------------------
        final Map<Integer, Integer> groundTruth;
        final double[] jaccardPerDet;
        if (BELIEF_LOG_ENABLED) {
            groundTruth   = extractTrueHiddenState(context, game);
            jaccardPerDet = new double[NUM_DETS];
            Arrays.fill(jaccardPerDet, Double.NaN);
        } else {
            groundTruth   = null;
            jaccardPerDet = null;
        }

        // ---------------------------------------------------------------
        // Step 2–4: IS-MCTS over NUM_DETS probe-sampled determinizations
        // ---------------------------------------------------------------
        final int[] votes = new int[legalMoves.size()];

        final double totalTime  = (maxSeconds > 0) ? maxSeconds : THINK_TIME_PER_DET * NUM_DETS;
        final double timePerDet = totalTime / NUM_DETS;
        final int    itersPerDet = (maxIterations > 0)
                ? Math.max(1, maxIterations / NUM_DETS)
                : SIMS_PER_DET;

        // Query belief ONCE per move — LSTM state is the same for all determinizations.
        // This avoids 25 redundant round-trip socket calls per move.
        final double[][] probsForThisMove = serverConnected ? queryBelief() : null;

        for (int d = 0; d < NUM_DETS; d++) {

            // Sample a determinization from the pre-queried belief distribution.
            final Map<Integer, Integer> particle;
            if (probsForThisMove != null) {
                particle = sampleFromProbe(context, game, probsForThisMove);
            } else {
                // No server or query failed — uniform random fallback
                particle = randomParticle(context, game);
            }

            final Context det = applyParticle(context, game, particle);

            Move bestForDet;
            try {
                bestForDet = uct.selectAction(game, det, timePerDet, itersPerDet, maxDepth);
            } catch (final Exception e) {
                continue;
            }

            // Jaccard logging
            if (BELIEF_LOG_ENABLED && groundTruth != null && jaccardPerDet != null) {
                jaccardPerDet[d] = jaccard(groundTruth, particle);
            }

            if (bestForDet == null) continue;

            final Move matched = findMatchingMove(bestForDet, legalMoves);
            if (matched == null) continue;

            for (int i = 0; i < legalMoves.size(); i++) {
                if (legalMoves.get(i) == matched) {
                    votes[i]++;
                    break;
                }
            }
        }

        // Plurality vote
        int bestIdx   = 0;
        int bestVotes = -1;
        for (int i = 0; i < legalMoves.size(); i++) {
            if (votes[i] > bestVotes) {
                bestVotes = votes[i];
                bestIdx   = i;
            }
        }
        if (bestVotes <= 0) {
            bestIdx = ThreadLocalRandom.current().nextInt(legalMoves.size());
        }

        if (BELIEF_LOG_ENABLED && groundTruth != null && jaccardPerDet != null) {
            writeBeliefRow(groundTruth.size(), jaccardPerDet);
        }

        moveCount++;
        return legalMoves.get(bestIdx);
    }

    // -----------------------------------------------------------------------
    // Server communication
    // -----------------------------------------------------------------------

    private void connectToServer() {
        disconnectFromServer();   // clean up stale socket if any
        for (int attempt = 1; attempt <= CONNECT_RETRIES; attempt++) {
            try {
                serverSocket = new Socket(SERVER_HOST, SERVER_PORT);
                serverSocket.setSoTimeout(90_000);  // 90 s read timeout (first MPS call is slow)
                toServer = new PrintWriter(
                        new BufferedWriter(new OutputStreamWriter(
                                serverSocket.getOutputStream(), StandardCharsets.UTF_8)));
                fromServer = new BufferedReader(
                        new InputStreamReader(
                                serverSocket.getInputStream(), StandardCharsets.UTF_8));
                serverConnected = true;
                System.out.println("[LSTMGuidedISMCTSAgent] Connected to belief server at "
                        + SERVER_HOST + ":" + SERVER_PORT + " (player " + playerId + ")");
                return;
            } catch (final Exception e) {
                System.err.println("[LSTMGuidedISMCTSAgent] Connect attempt " + attempt
                        + "/" + CONNECT_RETRIES + " failed: " + e.getMessage());
                disconnectFromServer();
                if (attempt < CONNECT_RETRIES) {
                    try { Thread.sleep(CONNECT_RETRY_MS); } catch (final InterruptedException ignored) {}
                }
            }
        }
        serverConnected = false;
        System.err.println("[LSTMGuidedISMCTSAgent] WARNING: Could not connect to belief server — "
                + "falling back to uniform random determinization.");
    }

    private void disconnectFromServer() {
        serverConnected = false;
        if (toServer    != null) { try { toServer.close();    } catch (Exception ignored) {} toServer    = null; }
        if (fromServer  != null) { try { fromServer.close();  } catch (Exception ignored) {} fromServer  = null; }
        if (serverSocket!= null) { try { serverSocket.close();} catch (Exception ignored) {} serverSocket = null; }
    }

    /** Send {"cmd":"reset"} and read the "ok" response (best-effort). */
    private void sendReset() {
        if (!serverConnected) return;
        try {
            toServer.println("{\"cmd\":\"reset\"}");
            toServer.flush();
            fromServer.readLine();  // consume {"status":"ok"}
        } catch (final Exception e) {
            System.err.println("[LSTMGuidedISMCTSAgent] reset error: " + e.getMessage());
            serverConnected = false;
        }
    }

    /**
     * Send {"cmd":"obs","obs":[...]} to advance the LSTM hidden state.
     * No response body is awaited beyond the status line.
     */
    private void sendObs(final float[] obs) {
        if (!serverConnected) return;
        try {
            final StringBuilder sb = new StringBuilder(obs.length * 10 + 20);
            sb.append("{\"cmd\":\"obs\",\"obs\":[");
            for (int i = 0; i < obs.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(obs[i]);
            }
            sb.append("]}");
            toServer.println(sb.toString());
            toServer.flush();
            // Read and discard {"status":"ok"}
            final String resp = fromServer.readLine();
            if (resp == null || !resp.contains("ok")) {
                System.err.println("[LSTMGuidedISMCTSAgent] Unexpected obs response: " + resp);
                serverConnected = false;
            }
        } catch (final Exception e) {
            System.err.println("[LSTMGuidedISMCTSAgent] obs error: " + e.getMessage());
            serverConnected = false;
        }
    }

    /**
     * Send {"cmd":"belief"} and parse the response.
     *
     * @return {@code double[64][7]} — softmax probabilities, or {@code null} on error.
     */
    private double[][] queryBelief() {
        if (!serverConnected) return null;
        try {
            toServer.println("{\"cmd\":\"belief\"}");
            toServer.flush();
            final String resp = fromServer.readLine();
            if (resp == null) {
                serverConnected = false;
                return null;
            }
            return parseProbs(resp);
        } catch (final Exception e) {
            System.err.println("[LSTMGuidedISMCTSAgent] belief query error: " + e.getMessage());
            serverConnected = false;
            return null;
        }
    }

    // -----------------------------------------------------------------------
    // JSON parsing — hand-rolled to avoid a library dependency
    // -----------------------------------------------------------------------

    /**
     * Parse the belief response JSON {@code {"probs":[[p0,p1,...,p6],...]}} into a
     * {@code double[64][7]} array.
     *
     * <p>The approach avoids any JSON library by finding the outer array token
     * and then parsing each inner 7-element array sequentially.
     */
    private static double[][] parseProbs(final String json) {
        final double[][] probs = new double[64][NUM_PIECE_TYPES];

        // Locate the outer array: after "probs":
        final int probsKey = json.indexOf("\"probs\"");
        if (probsKey < 0) {
            throw new IllegalArgumentException("No 'probs' key in: " + json);
        }
        int pos = json.indexOf('[', probsKey + 7);  // opening [ of outer array
        if (pos < 0) throw new IllegalArgumentException("No opening '[' for probs array");
        pos++;  // skip outer '['

        for (int sq = 0; sq < 64; sq++) {
            // Find the inner '[' for this square
            pos = json.indexOf('[', pos);
            if (pos < 0) throw new IllegalArgumentException(
                    "Ran out of inner arrays at square " + sq);
            pos++;  // skip inner '['
            for (int t = 0; t < NUM_PIECE_TYPES; t++) {
                // Skip whitespace and commas
                while (pos < json.length()
                        && (json.charAt(pos) == ' ' || json.charAt(pos) == ',')) {
                    pos++;
                }
                int end = pos;
                while (end < json.length()) {
                    final char ch = json.charAt(end);
                    if (ch == ',' || ch == ']') break;
                    end++;
                }
                probs[sq][t] = Double.parseDouble(json.substring(pos, end).trim());
                pos = end;
            }
            // Skip the closing ']' of this inner array
            pos = json.indexOf(']', pos);
            if (pos >= 0) pos++;
        }
        return probs;
    }

    // -----------------------------------------------------------------------
    // Determinization: probe-guided sampling
    // -----------------------------------------------------------------------

    /**
     * Sample a determinization from the per-square probe distributions.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>Identify squares that are hidden from this player.</li>
     *   <li>For each hidden square, sample a piece type from {@code probs[sq]}.</li>
     *   <li>Enforce piece-type count limits (uniqueness): iterate piece types
     *       from rarest to most common.  For each type, if more squares were
     *       sampled than the maximum count, keep the {@code max} squares with
     *       highest probability and set the others to empty.</li>
     *   <li>Return the resulting {@code Map<site, pieceType>} (omitting empties).</li>
     * </ol>
     *
     * <p>When the probe output doesn't correspond to a valid chess position
     * (e.g. it predicts two kings), the uniqueness enforcement drops the
     * lower-probability extra pieces to empty, producing the nearest valid
     * configuration according to the probe's own confidence.
     */
    private Map<Integer, Integer> sampleFromProbe(
            final Context context,
            final Game    game,
            final double[][] probs)
    {
        final State state    = context.state();
        final int   numSites = game.board().numSites();
        final int   opponent = (playerId == 1) ? 2 : 1;

        // Collect ALL squares that are hidden from this player.
        // FIX (fair Jaccard): previously this filtered to owner==opponent only,
        // which leaked true piece positions into the belief measurement — the
        // probe was only asked about squares already known to contain opponent
        // pieces, so it could only be wrong about piece *type*, not position.
        // Including all hidden squares (occupied AND empty from ground truth)
        // lets the probe predict both position and type without privilege.
        final List<Integer> hiddenSites = new ArrayList<>();
        for (int site = 0; site < numSites; site++) {
            final boolean hidden = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);
            if (hidden) {
                hiddenSites.add(site);
            }
        }

        if (hiddenSites.isEmpty()) {
            return Collections.emptyMap();
        }

        final ThreadLocalRandom rng = ThreadLocalRandom.current();

        // Step 1: sample a piece type for each hidden square
        final int[]    sampledType = new int[hiddenSites.size()];
        final double[] sampledProb = new double[hiddenSites.size()]; // prob of the sampled type

        for (int i = 0; i < hiddenSites.size(); i++) {
            final int    sq       = hiddenSites.get(i);
            final double[] dist   = (sq < probs.length) ? probs[sq] : uniformDist();
            final int    chosen   = categoricalSample(dist, rng);
            sampledType[i] = chosen;
            sampledProb[i] = dist[chosen];
        }

        // Step 2: enforce piece-type count limits
        // For each piece type (skip empty=0), count how many were sampled.
        // If too many, zero out the ones with the lowest probability until
        // the count is within bounds.
        final int[] typeCount = new int[NUM_PIECE_TYPES];
        for (final int t : sampledType) {
            typeCount[t]++;
        }

        for (int pieceType = 1; pieceType < NUM_PIECE_TYPES; pieceType++) {
            final int maxAllowed = MAX_PIECE_COUNT[pieceType];
            if (typeCount[pieceType] <= maxAllowed) continue;

            // Collect indices into hiddenSites that sampled this pieceType
            final List<Integer> candidateIdx = new ArrayList<>();
            for (int i = 0; i < hiddenSites.size(); i++) {
                if (sampledType[i] == pieceType) {
                    candidateIdx.add(i);
                }
            }

            // Sort ascending by probability — lowest-confidence ones will be dropped
            candidateIdx.sort(Comparator.comparingDouble(idx -> sampledProb[idx]));

            // Zero out the excess (lowest probability) ones
            int excess = typeCount[pieceType] - maxAllowed;
            for (int j = 0; j < excess; j++) {
                final int idx = candidateIdx.get(j);
                sampledType[idx] = PT_EMPTY;
            }
            typeCount[pieceType] = maxAllowed;
        }

        // Step 3: build the particle map (site → pieceType), skip empties
        final Map<Integer, Integer> particle = new LinkedHashMap<>();
        for (int i = 0; i < hiddenSites.size(); i++) {
            if (sampledType[i] != PT_EMPTY) {
                particle.put(hiddenSites.get(i), sampledType[i]);
            }
        }
        return particle;
    }

    /**
     * Sample one index from a probability distribution using the inverse CDF method.
     */
    private static int categoricalSample(final double[] probs, final ThreadLocalRandom rng) {
        double cumulative = 0.0;
        final double u = rng.nextDouble();
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (u <= cumulative) return i;
        }
        return probs.length - 1;  // rounding guard
    }

    /** Uniform distribution over all piece types (used if probs array is too short). */
    private static double[] uniformDist() {
        final double[] d = new double[NUM_PIECE_TYPES];
        Arrays.fill(d, 1.0 / NUM_PIECE_TYPES);
        return d;
    }

    // -----------------------------------------------------------------------
    // Fallback: uniform random particle (mirrors ISMCTSAgent.determinize)
    // -----------------------------------------------------------------------

    private Map<Integer, Integer> randomParticle(final Context context, final Game game) {
        final State state    = context.state();
        final int   numSites = game.board().numSites();
        final int   opponent = (playerId == 1) ? 2 : 1;

        final List<Integer> hiddenOccupiedSites  = new ArrayList<>();
        final List<Integer> emptyCandidateSites  = new ArrayList<>();

        for (int site = 0; site < numSites; site++) {
            final int     owner  = state.containerStates()[0].who(site, SiteType.Cell);
            final boolean hidden = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);
            if (hidden) {
                if (owner == opponent) {
                    hiddenOccupiedSites.add(site);
                } else if (owner == 0) {
                    emptyCandidateSites.add(site);
                }
            }
        }

        final Map<Integer, Integer> particle = new LinkedHashMap<>();
        if (hiddenOccupiedSites.isEmpty()) return particle;

        if (emptyCandidateSites.isEmpty()) {
            for (final int site : hiddenOccupiedSites) {
                particle.put(site, state.containerStates()[0].what(site, SiteType.Cell));
            }
            return particle;
        }

        final List<Integer> hiddenPieces = new ArrayList<>();
        for (final int site : hiddenOccupiedSites) {
            hiddenPieces.add(state.containerStates()[0].what(site, SiteType.Cell));
        }

        final List<Integer> allHiddenSites = new ArrayList<>(hiddenOccupiedSites);
        allHiddenSites.addAll(emptyCandidateSites);
        Collections.shuffle(allHiddenSites, ThreadLocalRandom.current());

        for (int i = 0; i < hiddenPieces.size(); i++) {
            particle.put(allHiddenSites.get(i), hiddenPieces.get(i));
        }
        return particle;
    }

    // -----------------------------------------------------------------------
    // Apply a particle → determinized context (same as ParticleISMCTSAgent)
    // -----------------------------------------------------------------------

    private Context applyParticle(
            final Context context,
            final Game    game,
            final Map<Integer, Integer> particle)
    {
        final Context det      = new Context(context);
        final State   state    = det.state();
        final int     numSites = game.board().numSites();
        final int     opponent = (playerId == 1) ? 2 : 1;

        // Clear all currently-hidden opponent pieces
        for (int site = 0; site < numSites; site++) {
            final boolean hidden = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);
            final int owner = state.containerStates()[0].who(site, SiteType.Cell);
            if (hidden && owner == opponent) {
                try {
                    state.containerStates()[0]
                            .setSite(state, site, 0, 0, 0, 0, 0, 0, SiteType.Cell);
                } catch (final Exception ignored) {}
            }
        }

        // Place pieces according to the particle
        for (final Map.Entry<Integer, Integer> e : particle.entrySet()) {
            final int site      = e.getKey();
            final int pieceType = e.getValue();
            if (pieceType <= 0) continue;
            try {
                state.containerStates()[0].setSite(
                        state, site,
                        opponent,    // who
                        pieceType,   // what
                        0, 0, 0, -1, // count, state, rotation, value
                        SiteType.Cell);
            } catch (final Exception ignored) {}
        }

        return det;
    }

    // -----------------------------------------------------------------------
    // Observation builder (mirrors PPOLSTMAgent exactly)
    // -----------------------------------------------------------------------

    /**
     * Builds the 192-float fog-of-war observation matching the 3-channel format
     * used when the PPO-LSTM v4 checkpoint was trained.
     *
     * <p>Channel layout (each channel is 64 floats, one per board square):
     * <ul>
     *   <li>obs[site]       (0–63):   owner — 0=hidden, 1/3=empty, 2/3=self, 1=opponent</li>
     *   <li>obs[site + 64]  (64–127): piece_type / 6.0 — 0 if hidden or absent</li>
     *   <li>obs[site + 128] (128–191): visibility flag — 1.0 if visible to this player, 0.0 if hidden</li>
     * </ul>
     *
     * <p>The visibility channel was present in the training environment but later removed
     * from {@code fow_env.py}. The checkpoint still expects all 192 input features.
     */
    private float[] buildObservation(final Context context) {
        final float[] obs   = new float[192];
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
                obs[site + 64]  = piece / 6.0f;
                obs[site + 128] = 1.0f;   // visible
            }
            // hidden: all three channels remain 0.0f (obs[site+128] stays 0.0)
        }
        return obs;
    }

    // -----------------------------------------------------------------------
    // Ground-truth extraction (for belief logging only)
    // -----------------------------------------------------------------------

    private Map<Integer, Integer> extractTrueHiddenState(final Context context, final Game game) {
        final State state    = context.state();
        final int   numSites = game.board().numSites();
        final int   opponent = (playerId == 1) ? 2 : 1;

        final Map<Integer, Integer> truth = new LinkedHashMap<>();
        for (int site = 0; site < numSites; site++) {
            final boolean hidden = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);
            if (!hidden) continue;
            final int owner = state.containerStates()[0].who(site, SiteType.Cell);
            if (owner != opponent) continue;
            final int ptype = state.containerStates()[0].what(site, SiteType.Cell);
            if (ptype > 0) {
                truth.put(site, ptype);
            }
        }
        return truth;
    }

    // -----------------------------------------------------------------------
    // Jaccard similarity
    // -----------------------------------------------------------------------

    private static double jaccard(
            final Map<Integer, Integer> groundTruth,
            final Map<Integer, Integer> belief)
    {
        if (groundTruth.isEmpty() && belief.isEmpty()) return 1.0;

        int intersection = 0;
        for (final Map.Entry<Integer, Integer> e : groundTruth.entrySet()) {
            final Integer bVal = belief.get(e.getKey());
            if (bVal != null && bVal.equals(e.getValue())) {
                intersection++;
            }
        }
        final int union = groundTruth.size() + belief.size() - intersection;
        if (union == 0) return 1.0;
        return (double) intersection / union;
    }

    // -----------------------------------------------------------------------
    // CSV logging helpers
    // -----------------------------------------------------------------------

    private void ensureBeliefLogHeader() {
        final File f = new File(BELIEF_LOG_FILE);
        if (f.exists() && f.length() > 0) return;
        final File parent = f.getParentFile();
        if (parent != null) parent.mkdirs();
        try (final PrintWriter pw = new PrintWriter(new FileWriter(f, true))) {
            pw.println("game_id,move_num,player,num_hidden_pieces,"
                    + "avg_jaccard,min_jaccard,max_jaccard,num_determinizations");
        } catch (final IOException e) {
            System.err.println("[LSTMGuidedISMCTSAgent] WARNING: Could not write belief log header: "
                    + e.getMessage());
        }
    }

    private void writeBeliefRow(final int numHidden, final double[] jaccardPerDet) {
        double sum  = 0.0;
        double min  = Double.MAX_VALUE;
        double max  = Double.NEGATIVE_INFINITY;
        int    valid = 0;

        for (final double j : jaccardPerDet) {
            if (!Double.isNaN(j)) {
                sum += j;
                if (j < min) min = j;
                if (j > max) max = j;
                valid++;
            }
        }
        if (valid == 0) return;

        final double avg = sum / valid;
        if (min == Double.MAX_VALUE)         min = Double.NaN;
        if (max == Double.NEGATIVE_INFINITY) max = Double.NaN;

        try (final PrintWriter pw = new PrintWriter(
                new FileWriter(new File(BELIEF_LOG_FILE), true))) {
            pw.printf("%d,%d,%d,%d,%.6f,%.6f,%.6f,%d%n",
                    gameId, moveCount, playerId, numHidden,
                    avg, min, max, valid);
        } catch (final IOException e) {
            System.err.println("[LSTMGuidedISMCTSAgent] WARNING: Could not write belief log row: "
                    + e.getMessage());
        }
    }

    // -----------------------------------------------------------------------
    // Move-matching helper
    // -----------------------------------------------------------------------

    private static Move findMatchingMove(final Move target, final List<Move> legalMoves) {
        for (final Move m : legalMoves) {
            if (m.from() == target.from() && m.to() == target.to()) {
                return m;
            }
        }
        return null;
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    @Override
    public String friendlyName() {
        return "LSTM-Guided-IS-MCTS";
    }

    @Override
    public boolean supportsGame(final Game game) {
        return game.players().count() == 2 && game.hiddenInformation();
    }

    @Override
    public String toString() {
        return "LSTMGuidedISMCTSAgent(dets=" + NUM_DETS + ", sims=" + SIMS_PER_DET
                + ", server=" + SERVER_HOST + ":" + SERVER_PORT + ")";
    }
}
