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
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * ParticleISMCTSAgent — IS-MCTS with a particle-filter belief oracle.
 *
 * <p>Instead of uniformly randomising hidden piece positions for each
 * determinization (as plain IS-MCTS does), this agent maintains a set of
 * {@value #K} <em>particles</em> — explicit hypotheses about where the
 * opponent's hidden pieces actually are — and updates that set after each
 * observation.  Determinizations are then drawn from the particle set rather
 * than generated uniformly at random.
 *
 * <h3>Algorithm sketch (per turn)</h3>
 * <ol>
 *   <li><b>Initialise</b> (first call only): generate K particles via uniform
 *       random determinization — same prior as plain IS-MCTS.</li>
 *   <li><b>Filter</b>: discard any particle that contradicts what is now
 *       visible (visible-empty squares must be empty in the particle;
 *       visible-occupied squares must carry the correct piece type).</li>
 *   <li><b>Resample</b>: if fewer than K/2 particles survived, replenish up
 *       to K by drawing new uniform random particles.</li>
 *   <li><b>Run IS-MCTS</b>: for each of {@value #NUM_DETS} determinizations,
 *       pick a random surviving particle, apply it to a context copy, run UCT,
 *       and cast a vote for the recommended move.</li>
 *   <li><b>Return</b> the move with the most votes.</li>
 * </ol>
 *
 * <h3>Belief-accuracy logging (opt-in)</h3>
 * <p>Set the JVM property {@code fow.belief.log=true} to enable per-move
 * Jaccard-similarity logging to a CSV file (path controlled by
 * {@code fow.belief.logfile}, default {@code belief_accuracy_particle.csv}).
 *
 * <p>CSV schema (same as ISMCTSAgent):
 * <pre>
 * game_id, move_num, player, num_hidden_pieces, avg_jaccard, min_jaccard, max_jaccard, num_determinizations
 * </pre>
 */
public class ParticleISMCTSAgent extends AI {

    // -----------------------------------------------------------------------
    // Tuning constants
    // -----------------------------------------------------------------------

    /** Number of particles in the belief set. */
    private static final int K = 100;

    /** UCT iterations per determinization. */
    private static final int SIMS_PER_DET = 200;

    /** Wall-clock budget per determinization (seconds). */
    private static final double THINK_TIME_PER_DET = 0.04;

    // -----------------------------------------------------------------------
    // Starting material for fair Jaccard measurement
    // -----------------------------------------------------------------------

    /**
     * Opponent's starting piece-type distribution in standard chess (16 pieces).
     * Used ONLY in the Jaccard logging code path to avoid leaking true piece
     * types from the ground-truth state into the belief map.
     * Piece-type indices match Ludii: 1=pawn, 2=rook, 3=bishop, 4=knight, 5=queen, 6=king.
     */
    private static final int[] STARTING_MATERIAL = {
        1, 1, 1, 1, 1, 1, 1, 1,  // 8 pawns
        2, 2,                      // 2 rooks
        3, 3,                      // 2 bishops
        4, 4,                      // 2 knights
        5,                         // 1 queen
        6                          // 1 king
    };

    /**
     * Number of determinizations sampled from the particle set per move.
     * Fewer than K because we budget the same total wall-clock as IS-MCTS.
     */
    private static final int NUM_DETS = 25;

    // -----------------------------------------------------------------------
    // Belief-logging configuration
    // -----------------------------------------------------------------------

    private static final boolean BELIEF_LOG_ENABLED;
    private static final String  BELIEF_LOG_FILE;

    static {
        BELIEF_LOG_ENABLED = "true".equalsIgnoreCase(
                System.getProperty("fow.belief.log", "false"));
        BELIEF_LOG_FILE = System.getProperty(
                "fow.belief.logfile", "belief_accuracy_particle.csv");
    }

    // -----------------------------------------------------------------------
    // Agent state
    // -----------------------------------------------------------------------

    private int  playerId  = -1;
    private MCTS uct       = null;
    private int  moveCount = 0;
    private long gameId    = 0L;

    /**
     * The particle set.
     * Each particle is a {@code Map<site, pieceType>} encoding a complete
     * hypothesis about where the opponent's hidden pieces are located.
     * An empty map means "no hidden pieces" which is also a valid hypothesis.
     */
    private final List<Map<Integer, Integer>> particles = new ArrayList<>(K);

    /**
     * Whether {@link #initParticles} has been called for the current game.
     * Reset to {@code false} by {@link #initAI}.
     */
    private boolean particlesInitialized = false;

    /** Previous move count — used to detect a new game when initAI isn't called. */
    private int lastMoveCount = -1;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    public ParticleISMCTSAgent() {
        this.friendlyName = "Particle-IS-MCTS";
    }

    // -----------------------------------------------------------------------
    // AI lifecycle
    // -----------------------------------------------------------------------

    @Override
    public void initAI(final Game game, final int playerID) {
        this.playerId            = playerID;
        this.moveCount           = 0;
        this.lastMoveCount       = -1;
        this.gameId              = System.currentTimeMillis();
        this.particlesInitialized = false;
        this.particles.clear();

        // Close old UCT instance to free its thread pools.
        if (uct != null) {
            try { uct.closeAI(); } catch (final Exception ignored) {}
            uct = null;
        }

        uct = MCTS.createUCT();
        uct.initAI(game, playerID);

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
        // Guard: detect game reset even if initAI wasn't called between games.
        final int currentMoveNum = context.trial().numMoves();
        if (currentMoveNum == 0 || currentMoveNum < lastMoveCount) {
            particles.clear();
            particlesInitialized = false;
        }
        lastMoveCount = currentMoveNum;

        // Gather legal moves for the current (observable) context.
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
        // Particle filter maintenance
        // ---------------------------------------------------------------

        // Step 1 — initialise on first call.
        if (!particlesInitialized) {
            initParticles(context, game);
            particlesInitialized = true;
        }

        // Step 2 — filter: remove particles inconsistent with current observation.
        filterParticles(context, game);

        // Step 3 — resample: replenish if too few survived.
        resampleParticles(context, game);

        // ---------------------------------------------------------------
        // Belief logging: capture ground-truth BEFORE any determinization.
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
        // IS-MCTS: run NUM_DETS determinizations, vote for best move.
        // ---------------------------------------------------------------
        final int[] votes = new int[legalMoves.size()];

        final double totalTime   = (maxSeconds > 0) ? maxSeconds : THINK_TIME_PER_DET * NUM_DETS;
        final double timePerDet  = totalTime / NUM_DETS;
        final int    itersPerDet = (maxIterations > 0)
                ? Math.max(1, maxIterations / NUM_DETS)
                : SIMS_PER_DET;

        for (int d = 0; d < NUM_DETS; d++) {

            // Sample a particle and apply it to produce a determinized context.
            final Map<Integer, Integer> particle;
            final Context det;
            if (BELIEF_LOG_ENABLED) {
                particle = sampleParticle();
                det      = applyParticle(context, game, particle);
            } else {
                particle = null;
                det      = applyParticle(context, game, sampleParticle());
            }

            Move bestForDet;
            try {
                bestForDet = uct.selectAction(game, det, timePerDet, itersPerDet, maxDepth);
            } catch (final Exception e) {
                continue;
            }

            // Record Jaccard (belief quality) for logging.
            // Use a fair-type copy of the particle to avoid leaking true piece
            // types into the metric — only positions come from the particle.
            if (BELIEF_LOG_ENABLED && groundTruth != null && particle != null && jaccardPerDet != null) {
                jaccardPerDet[d] = jaccard(groundTruth, makeFairJaccardParticle(particle));
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

        // Plurality vote.
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

        // Write belief-accuracy row.
        if (BELIEF_LOG_ENABLED && groundTruth != null && jaccardPerDet != null) {
            writeBeliefRow(groundTruth.size(), jaccardPerDet);
        }

        moveCount++;
        return legalMoves.get(bestIdx);
    }

    // -----------------------------------------------------------------------
    // Particle filter helpers
    // -----------------------------------------------------------------------

    /**
     * Initialise the particle set by generating K uniform random
     * determinizations (the same prior as plain IS-MCTS).
     */
    private void initParticles(final Context context, final Game game) {
        particles.clear();
        for (int i = 0; i < K; i++) {
            particles.add(randomParticle(context, game));
        }
    }

    /**
     * Remove particles that contradict the current visible board.
     *
     * <p>A particle is <em>consistent</em> iff:
     * <ul>
     *   <li>Every square that is now <em>visible and empty</em> has no piece
     *       recorded in the particle (i.e. the particle does not place a piece
     *       there).</li>
     *   <li>Every square that is now <em>visible and occupied by the
     *       opponent</em> has the correct piece type in the particle.</li>
     * </ul>
     *
     * <p>Squares that are hidden, or occupied by <em>us</em>, are not
     * constrained by the particle (the particle only records opponent hidden
     * pieces).
     */
    private void filterParticles(final Context context, final Game game) {
        final State state    = context.state();
        final int   numSites = game.board().numSites();
        final int   opponent = (playerId == 1) ? 2 : 1;

        // Collect visible constraints once (avoid recomputing per particle).
        // visibleEmpty[site]    = true  → site must be absent from particle
        // visibleOccupied[site] = ptype → site must map to ptype in particle
        final boolean[] visibleEmpty    = new boolean[numSites];
        final int[]     visibleOccupied = new int[numSites]; // 0 = not constrained
        Arrays.fill(visibleOccupied, 0);

        for (int site = 0; site < numSites; site++) {
            final boolean hidden = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);
            if (hidden) continue; // hidden to us → no observation

            final int owner = state.containerStates()[0].who(site, SiteType.Cell);
            if (owner == 0) {
                // Visible and empty → particle must not have a piece here.
                visibleEmpty[site] = true;
            } else if (owner == opponent) {
                // Visible and occupied by opponent → particle must agree on piece type.
                final int ptype = state.containerStates()[0].what(site, SiteType.Cell);
                visibleOccupied[site] = ptype;
            }
            // owner == playerId → we see our own piece; particle doesn't track that.
        }

        final Iterator<Map<Integer, Integer>> it = particles.iterator();
        while (it.hasNext()) {
            final Map<Integer, Integer> p = it.next();
            if (!isConsistent(p, visibleEmpty, visibleOccupied)) {
                it.remove();
            }
        }
    }

    /**
     * Returns {@code true} iff the particle does not violate any visible
     * constraint.
     */
    private static boolean isConsistent(
            final Map<Integer, Integer> particle,
            final boolean[] visibleEmpty,
            final int[]     visibleOccupied)
    {
        // Check 1: particle must not place a piece on a visible-empty square.
        for (final int site : particle.keySet()) {
            if (site < visibleEmpty.length && visibleEmpty[site]) {
                return false;
            }
        }

        // Check 2: particle must agree on piece type for visible-occupied squares.
        for (int site = 0; site < visibleOccupied.length; site++) {
            final int required = visibleOccupied[site];
            if (required == 0) continue; // not constrained

            final Integer actual = particle.get(site);
            if (actual == null || actual != required) {
                // Particle lacks the piece at this visible-occupied square.
                // This can happen when the particle had the piece elsewhere.
                return false;
            }
        }

        return true;
    }

    /**
     * Replenish the particle set up to K by appending new uniform random
     * particles when fewer than K/2 survived filtering.
     *
     * <p>New particles are generated by the same uniform-random method used
     * in IS-MCTS, then immediately filtered for consistency.  We attempt up
     * to 5×K tries to avoid an infinite loop when the constraint set is very
     * tight.
     */
    private void resampleParticles(final Context context, final Game game) {
        if (particles.size() >= K / 2) return;

        final State state    = context.state();
        final int   numSites = game.board().numSites();
        final int   opponent = (playerId == 1) ? 2 : 1;

        // Rebuild constraint arrays once (same logic as filterParticles).
        final boolean[] visibleEmpty    = new boolean[numSites];
        final int[]     visibleOccupied = new int[numSites];
        Arrays.fill(visibleOccupied, 0);
        for (int site = 0; site < numSites; site++) {
            final boolean hidden = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);
            if (hidden) continue;
            final int owner = state.containerStates()[0].who(site, SiteType.Cell);
            if (owner == 0) {
                visibleEmpty[site] = true;
            } else if (owner == opponent) {
                visibleOccupied[site] = state.containerStates()[0]
                        .what(site, SiteType.Cell);
            }
        }

        final int maxTries = 5 * K;
        int tries = 0;
        while (particles.size() < K && tries < maxTries) {
            tries++;
            final Map<Integer, Integer> candidate = randomParticle(context, game);
            if (isConsistent(candidate, visibleEmpty, visibleOccupied)) {
                particles.add(candidate);
            }
        }

        // Last resort: if still empty, add whatever we can (even inconsistent)
        // so that the agent doesn't crash.
        if (particles.isEmpty()) {
            particles.add(randomParticle(context, game));
        }
    }

    /**
     * Generate a single uniform-random particle by shuffling hidden opponent
     * pieces among hidden squares.
     *
     * <p>This mirrors {@code ISMCTSAgent.determinize()} but returns a
     * {@code Map<site, pieceType>} instead of a modified {@link Context}.
     */
    private Map<Integer, Integer> randomParticle(final Context context, final Game game) {
        final State state    = context.state();
        final int   numSites = game.board().numSites();
        final int   opponent = (playerId == 1) ? 2 : 1;

        final List<Integer> hiddenOccupiedSites = new ArrayList<>();
        final List<Integer> emptyCandidateSites = new ArrayList<>();

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

        if (hiddenOccupiedSites.isEmpty()) {
            // No hidden opponent pieces — return empty particle.
            return particle;
        }

        if (emptyCandidateSites.isEmpty()) {
            // No empty hidden squares — pieces stay in place (cannot shuffle).
            for (final int site : hiddenOccupiedSites) {
                particle.put(site, state.containerStates()[0].what(site, SiteType.Cell));
            }
            return particle;
        }

        // Collect piece types from their observable positions.
        final List<Integer> hiddenPieces = new ArrayList<>();
        for (final int site : hiddenOccupiedSites) {
            hiddenPieces.add(state.containerStates()[0].what(site, SiteType.Cell));
        }

        // Shuffle all hidden sites and assign pieces to the first N positions.
        final List<Integer> allHiddenSites = new ArrayList<>(hiddenOccupiedSites);
        allHiddenSites.addAll(emptyCandidateSites);
        Collections.shuffle(allHiddenSites, ThreadLocalRandom.current());

        for (int i = 0; i < hiddenPieces.size(); i++) {
            particle.put(allHiddenSites.get(i), hiddenPieces.get(i));
        }

        return particle;
    }

    /**
     * Pick a uniformly random surviving particle.
     * Falls back to a singleton empty map if the list is somehow empty.
     */
    private Map<Integer, Integer> sampleParticle() {
        if (particles.isEmpty()) {
            return Collections.emptyMap();
        }
        final int idx = ThreadLocalRandom.current().nextInt(particles.size());
        return particles.get(idx);
    }

    /**
     * Apply a particle to an observable context, producing a fully-observable
     * determinized copy.
     *
     * <p>The approach mirrors {@code ISMCTSAgent.determinize()}:
     * <ol>
     *   <li>Copy the context.</li>
     *   <li>Clear all hidden opponent pieces from their observable positions.</li>
     *   <li>Place pieces according to the particle's site→pieceType mapping.</li>
     * </ol>
     */
    private Context applyParticle(
            final Context context,
            final Game    game,
            final Map<Integer, Integer> particle)
    {
        final Context det     = new Context(context);
        final State   state   = det.state();
        final int     numSites = game.board().numSites();
        final int     opponent = (playerId == 1) ? 2 : 1;

        // First, clear all currently-hidden opponent pieces from the copied state.
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

        // Then, place pieces according to the particle.
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
    // Fair Jaccard particle helper (belief logging only)
    // -----------------------------------------------------------------------

    /**
     * Creates a copy of {@code particle} with the same site keys but piece
     * types replaced by a random sample from {@link #STARTING_MATERIAL}.
     *
     * <p>This eliminates piece-type leakage: the returned map preserves the
     * position hypothesis of the particle but does NOT carry true piece types,
     * so that the Jaccard similarity measures position accuracy only — without
     * privileged knowledge of types not available to the agent at runtime.
     */
    private static Map<Integer, Integer> makeFairJaccardParticle(
            final Map<Integer, Integer> particle)
    {
        final int n = particle.size();
        if (n == 0) return Collections.emptyMap();

        // Sample n types without replacement from starting material.
        final List<Integer> pool = new ArrayList<>(STARTING_MATERIAL.length);
        for (final int t : STARTING_MATERIAL) {
            pool.add(t);
        }
        Collections.shuffle(pool, ThreadLocalRandom.current());

        final Map<Integer, Integer> fair = new LinkedHashMap<>(n * 2);
        int i = 0;
        for (final int site : particle.keySet()) {
            final int fairType = (i < pool.size()) ? pool.get(i) : 1;
            fair.put(site, fairType);
            i++;
        }
        return fair;
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
    // CSV logging helpers (belief accuracy)
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
            System.err.println("[ParticleISMCTSAgent] WARNING: Could not write belief log header: "
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
            System.err.println("[ParticleISMCTSAgent] WARNING: Could not write belief log row: "
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
        return "Particle-IS-MCTS";
    }

    @Override
    public boolean supportsGame(final Game game) {
        return game.players().count() == 2 && game.hiddenInformation();
    }

    @Override
    public String toString() {
        return "ParticleISMCTSAgent(K=" + K + ", dets=" + NUM_DETS + ", sims=" + SIMS_PER_DET + ")";
    }
}
