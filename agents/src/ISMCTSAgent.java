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
 * ISMCTSAgent — Information Set MCTS agent for Fog-of-War Chess.
 *
 * <p>Implements the normalised-determinization / plurality-voting variant of
 * IS-MCTS described in Cowling et al. (2012), simplified as follows:
 *
 * <ol>
 *   <li>For each of {@value #NUM_DETERMINIZATIONS} determinizations:
 *     <ul>
 *       <li>Copy the current (partially observable) context.</li>
 *       <li>Randomly shuffle hidden opponent pieces among hidden squares.</li>
 *       <li>Run UCT for {@value #SIMS_PER_DET} iterations on the fully
 *           observable determinized context.</li>
 *       <li>Record the move UCT recommends as a vote.</li>
 *     </ul>
 *   <li>Return the legal move that accumulated the most votes (plurality).</li>
 * </ol>
 *
 * <p>A single {@link MCTS} instance is created in {@link #initAI} and reused
 * across all determinizations to avoid Ludii-internal thread-pool leaks.
 *
 * <h3>Belief-accuracy logging (opt-in)</h3>
 * <p>When the JVM system property {@code fow.belief.log=true} is set, the agent
 * logs per-move Jaccard-similarity statistics to a CSV file. The file path is
 * controlled by {@code fow.belief.logfile} (default: {@code belief_accuracy.csv}).
 *
 * <p>CSV schema:
 * <pre>
 * game_id, move_num, player, num_hidden_pieces, avg_jaccard, min_jaccard, max_jaccard, num_determinizations
 * </pre>
 *
 * <p>The file is opened in <em>append</em> mode so multiple runs accumulate in
 * the same file.
 */
public class ISMCTSAgent extends AI {

    // -----------------------------------------------------------------------
    // Tuning constants
    // -----------------------------------------------------------------------

    /**
     * UCT iterations per determinization.
     * Kept modest so each move stays well under 1 s wall-clock.
     */
    private static final int    SIMS_PER_DET         = 200;

    /** Wall-clock budget per determinization (seconds). */
    private static final double THINK_TIME_PER_DET   = 0.04;

    // -----------------------------------------------------------------------
    // Belief-logging configuration (read once from system properties)
    // -----------------------------------------------------------------------

    private static final boolean BELIEF_LOG_ENABLED;
    private static final String  BELIEF_LOG_FILE;

    static {
        BELIEF_LOG_ENABLED = "true".equalsIgnoreCase(System.getProperty("fow.belief.log", "false"));
        BELIEF_LOG_FILE    = System.getProperty("fow.belief.logfile", "belief_accuracy.csv");
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /** Number of distinct determinizations per move (instance field, set by constructor). */
    private int numDeterminizations;

    private int  playerId  = -1;
    private MCTS uct       = null;

    /** Monotonically incrementing move counter used as CSV move_num column. */
    private int  moveCount = 0;

    /**
     * Shared game ID assigned once per {@link #initAI} call.
     * Using System.currentTimeMillis gives unique IDs across runs.
     */
    private long gameId    = 0L;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    public ISMCTSAgent() {
        this.numDeterminizations = 10;
        this.friendlyName = "IS-MCTS";
    }

    public ISMCTSAgent(int numDeterminizations) {
        this.numDeterminizations = numDeterminizations;
        this.friendlyName = "IS-MCTS";
    }

    // -----------------------------------------------------------------------
    // AI lifecycle
    // -----------------------------------------------------------------------

    @Override
    public void initAI(final Game game, final int playerID) {
        this.playerId  = playerID;
        this.moveCount = 0;
        this.gameId    = System.currentTimeMillis();

        // Close any previously held MCTS instance so its internal thread-pools
        // are torn down before we create a new one.
        if (uct != null) {
            try { uct.closeAI(); } catch (final Exception ignored) {}
            uct = null;
        }

        uct = MCTS.createUCT();
        uct.initAI(game, playerID);

        // Write CSV header if the log file does not yet exist (or is empty).
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
    // Move selection
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

        // Materialise into a plain List so we can iterate it safely.
        final List<Move> legalMoves = new ArrayList<>(legalMovesRaw.size());
        for (int i = 0; i < legalMovesRaw.size(); i++) {
            legalMoves.add(legalMovesRaw.get(i));
        }

        // votes[i] = # determinizations in which UCT chose legalMoves.get(i)
        final int[] votes = new int[legalMoves.size()];

        // Compute per-determinization time budget.
        // Use caller-supplied maxSeconds if positive, otherwise fall back to
        // THINK_TIME_PER_DET * NUM_DETERMINIZATIONS.
        final double totalTime = (maxSeconds > 0)
                ? maxSeconds
                : THINK_TIME_PER_DET * numDeterminizations;
        final double timePerDet = totalTime / numDeterminizations;

        // Per-determinization iteration budget.
        final int itersPerDet = (maxIterations > 0)
                ? Math.max(1, maxIterations / numDeterminizations)
                : SIMS_PER_DET;

        // ---------------------------------------------------------------
        // Belief logging: capture ground-truth BEFORE any determinization
        // ---------------------------------------------------------------
        // In Ludii's FoW implementation the full board state (including
        // hidden squares) is still present in the context passed to
        // selectAction — hidden pieces are simply flagged as not visible
        // to a given player.  We can therefore read what() and who() for
        // every cell to get the true arrangement and compare it to each
        // random determinization.
        // ---------------------------------------------------------------
        final Map<Integer, Integer> groundTruth;
        final double[] jaccardPerDet;
        if (BELIEF_LOG_ENABLED) {
            groundTruth  = extractTrueHiddenState(context, game);
            jaccardPerDet = new double[numDeterminizations];
        } else {
            groundTruth  = null;
            jaccardPerDet = null;
        }

        for (int d = 0; d < numDeterminizations; d++) {

            // 1. Build a fully observable determinization by randomising
            //    hidden opponent piece positions.
            //    Also capture the placement when logging is on.
            final Map<Integer, Integer> placement;
            final Context det;
            if (BELIEF_LOG_ENABLED) {
                final Object[] result = determinizeWithPlacement(context, game);
                det       = (Context)             result[0];
                @SuppressWarnings("unchecked")
                final Map<Integer, Integer> p = (Map<Integer, Integer>) result[1];
                placement = p;
            } else {
                det       = determinize(context, game);
                placement = null;
            }

            // 2. Run UCT on the determinized context.
            //    We reuse the single pre-created MCTS instance (avoids thread-
            //    pool leaks).  Passing maxDepth=-1 lets MCTS use its default.
            Move bestForDet;
            try {
                bestForDet = uct.selectAction(game, det, timePerDet, itersPerDet, maxDepth);
            } catch (final Exception e) {
                // If UCT fails on this determinization, skip it.
                if (BELIEF_LOG_ENABLED && jaccardPerDet != null) {
                    jaccardPerDet[d] = Double.NaN;
                }
                continue;
            }

            // 2b. Compute Jaccard for this determinization (if logging).
            if (BELIEF_LOG_ENABLED && groundTruth != null && placement != null && jaccardPerDet != null) {
                jaccardPerDet[d] = jaccard(groundTruth, placement);
            }

            if (bestForDet == null) continue;

            // 3. Match UCT's recommendation back to a move in our legal-move
            //    list (determinized context may have a different move-object
            //    identity even for the same from→to pair).
            final Move matched = findMatchingMove(bestForDet, legalMoves);
            if (matched == null) continue;

            // 4. Cast a vote for the matched move.
            for (int i = 0; i < legalMoves.size(); i++) {
                if (legalMoves.get(i) == matched) {
                    votes[i]++;
                    break;
                }
            }
        }

        // 5. Return the move with the most votes (ties broken by first
        //    occurrence, i.e. whichever appears earliest in the legal-move list).
        int bestIdx   = 0;
        int bestVotes = -1;
        for (int i = 0; i < legalMoves.size(); i++) {
            if (votes[i] > bestVotes) {
                bestVotes = votes[i];
                bestIdx   = i;
            }
        }

        // Fallback: if no move received any vote (every determinization failed),
        // choose uniformly at random so we still make a move.
        if (bestVotes <= 0) {
            bestIdx = ThreadLocalRandom.current().nextInt(legalMoves.size());
        }

        // 6. Write belief-accuracy row to CSV.
        if (BELIEF_LOG_ENABLED && groundTruth != null && jaccardPerDet != null) {
            writeBeliefRow(groundTruth.size(), jaccardPerDet);
        }

        moveCount++;
        return legalMoves.get(bestIdx);
    }

    // -----------------------------------------------------------------------
    // Determinization helper
    //
    // Mirrors PIMCAgent.determinize() exactly:
    //   • Collect sites that are hidden from us AND occupied by the opponent
    //     → these are the "hidden pieces" we must redistribute.
    //   • Collect sites that are hidden from us AND empty
    //     → these are candidate destination squares.
    //   • Shuffle all hidden sites together, then place the pieces in the
    //     first |hiddenPieces| positions of the shuffled list.
    // -----------------------------------------------------------------------

    private Context determinize(final Context observableCtx, final Game game) {
        final Context det      = new Context(observableCtx);
        final State   state    = det.state();
        final int     numSites = game.board().numSites();
        final int     opponent = (playerId == 1) ? 2 : 1;

        final List<Integer> hiddenOccupiedSites  = new ArrayList<>();
        final List<Integer> emptyCandidateSites  = new ArrayList<>();

        for (int site = 0; site < numSites; site++) {
            final int     owner        = state.containerStates()[0].who(site, SiteType.Cell);
            final boolean hiddenFromUs = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);

            if (hiddenFromUs) {
                if (owner == opponent) {
                    hiddenOccupiedSites.add(site);
                } else if (owner == 0) {
                    emptyCandidateSites.add(site);
                }
            }
        }

        // Only shuffle if there is at least one hidden opponent piece AND at
        // least one empty hidden square to move it to.
        if (!hiddenOccupiedSites.isEmpty() && !emptyCandidateSites.isEmpty()) {
            // Pool of all hidden sites (occupied + empty)
            final List<Integer> allHiddenSites = new ArrayList<>(hiddenOccupiedSites);
            allHiddenSites.addAll(emptyCandidateSites);
            Collections.shuffle(allHiddenSites, ThreadLocalRandom.current());

            // Remember what piece type and owner was on each hidden occupied site
            final List<Integer> hiddenPieces      = new ArrayList<>();
            final List<Integer> hiddenPieceOwners = new ArrayList<>();
            for (final int site : hiddenOccupiedSites) {
                hiddenPieces.add(state.containerStates()[0].what(site, SiteType.Cell));
                hiddenPieceOwners.add(state.containerStates()[0].who(site, SiteType.Cell));
            }

            // Clear the hidden occupied sites
            for (final int site : hiddenOccupiedSites) {
                try {
                    state.containerStates()[0]
                            .setSite(state, site, 0, 0, 0, 0, 0, 0, SiteType.Cell);
                } catch (final Exception e) {
                    // best-effort; ignore Ludii internal constraint violations
                }
            }

            // Place pieces at the first |hiddenPieces.size()| shuffled sites
            for (int i = 0; i < hiddenPieces.size(); i++) {
                final int targetSite = allHiddenSites.get(i);
                try {
                    state.containerStates()[0].setSite(
                            state, targetSite,
                            hiddenPieceOwners.get(i),   // who
                            hiddenPieces.get(i),         // what (piece type)
                            0, 0, 0, -1,                 // count, state, rotation, value
                            SiteType.Cell);
                } catch (final Exception e) {
                    // best-effort
                }
            }
        }

        return det;
    }

    // -----------------------------------------------------------------------
    // Determinization with placement capture (used only when belief logging)
    //
    // Returns Object[2]:
    //   [0] — the determinized Context
    //   [1] — Map<site, pieceType> representing the determinized placement of
    //          opponent pieces on hidden squares (only the squares that were
    //          randomised; fixed-position pieces are not included because
    //          ground-truth comparison is only meaningful for the hidden set).
    // -----------------------------------------------------------------------

    private Object[] determinizeWithPlacement(final Context observableCtx, final Game game) {
        final Context det      = new Context(observableCtx);
        final State   state    = det.state();
        final int     numSites = game.board().numSites();
        final int     opponent = (playerId == 1) ? 2 : 1;

        final List<Integer> hiddenOccupiedSites = new ArrayList<>();
        final List<Integer> emptyCandidateSites = new ArrayList<>();

        for (int site = 0; site < numSites; site++) {
            final int     owner        = state.containerStates()[0].who(site, SiteType.Cell);
            final boolean hiddenFromUs = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);

            if (hiddenFromUs) {
                if (owner == opponent) {
                    hiddenOccupiedSites.add(site);
                } else if (owner == 0) {
                    emptyCandidateSites.add(site);
                }
            }
        }

        // Placement map: site → pieceType for all hidden opponent positions
        // in this determinization.
        final Map<Integer, Integer> placement = new LinkedHashMap<>();

        if (!hiddenOccupiedSites.isEmpty() && !emptyCandidateSites.isEmpty()) {
            final List<Integer> allHiddenSites = new ArrayList<>(hiddenOccupiedSites);
            allHiddenSites.addAll(emptyCandidateSites);
            Collections.shuffle(allHiddenSites, ThreadLocalRandom.current());

            final List<Integer> hiddenPieces      = new ArrayList<>();
            final List<Integer> hiddenPieceOwners = new ArrayList<>();
            for (final int site : hiddenOccupiedSites) {
                hiddenPieces.add(state.containerStates()[0].what(site, SiteType.Cell));
                hiddenPieceOwners.add(state.containerStates()[0].who(site, SiteType.Cell));
            }

            // Clear original hidden occupied sites
            for (final int site : hiddenOccupiedSites) {
                try {
                    state.containerStates()[0]
                            .setSite(state, site, 0, 0, 0, 0, 0, 0, SiteType.Cell);
                } catch (final Exception e) {
                    // best-effort
                }
            }

            // Place pieces at shuffled destinations and record placement
            for (int i = 0; i < hiddenPieces.size(); i++) {
                final int targetSite  = allHiddenSites.get(i);
                final int pieceType   = hiddenPieces.get(i);
                final int pieceOwner  = hiddenPieceOwners.get(i);
                try {
                    state.containerStates()[0].setSite(
                            state, targetSite,
                            pieceOwner,
                            pieceType,
                            0, 0, 0, -1,
                            SiteType.Cell);
                } catch (final Exception e) {
                    // best-effort
                }
                // Record in placement map: encodes as (site * 1000 + pieceType)
                // to allow multi-piece-per-square comparisons consistently,
                // but here each site has at most one piece, so we store pieceType.
                placement.put(targetSite, pieceType);
            }
        } else if (hiddenOccupiedSites.isEmpty()) {
            // No hidden opponent pieces at all — placement is empty, which is
            // also what ground truth will be → Jaccard = 1.0.
        } else {
            // Has hidden pieces but no empty candidate squares — pieces stay
            // in place (cannot shuffle).  Record their current positions.
            for (final int site : hiddenOccupiedSites) {
                placement.put(site, state.containerStates()[0].what(site, SiteType.Cell));
            }
        }

        return new Object[]{ det, placement };
    }

    // -----------------------------------------------------------------------
    // Ground-truth hidden state extraction
    //
    // Reads the TRUE positions of all opponent pieces that are hidden from
    // this player.  In Ludii's FoW implementation the physical piece data
    // (what / who) is always present in the state — only the visibility flag
    // is restricted.  We therefore read from the full state directly.
    // Returns Map<site, pieceType>.
    // -----------------------------------------------------------------------

    private Map<Integer, Integer> extractTrueHiddenState(final Context context, final Game game) {
        final State state    = context.state();
        final int   numSites = game.board().numSites();
        final int   opponent = (playerId == 1) ? 2 : 1;

        final Map<Integer, Integer> truth = new LinkedHashMap<>();
        for (int site = 0; site < numSites; site++) {
            final boolean hiddenFromUs = state.containerStates()[0]
                    .isHidden(playerId, site, 0, SiteType.Cell);
            if (!hiddenFromUs) continue;

            final int owner = state.containerStates()[0].who(site, SiteType.Cell);
            if (owner != opponent) continue;

            final int pieceType = state.containerStates()[0].what(site, SiteType.Cell);
            if (pieceType > 0) {   // 0 = empty / no piece
                truth.put(site, pieceType);
            }
        }
        return truth;
    }

    // -----------------------------------------------------------------------
    // Jaccard similarity between two site→pieceType maps
    //
    // We treat each (site, pieceType) pair as a set element.
    // Jaccard = |intersection| / |union|.
    // If both maps are empty the situation is "perfectly accurate" → 1.0.
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

        // |union| = |A| + |B| - |A∩B|
        final int union = groundTruth.size() + belief.size() - intersection;
        if (union == 0) return 1.0;
        return (double) intersection / union;
    }

    // -----------------------------------------------------------------------
    // CSV logging helpers
    // -----------------------------------------------------------------------

    /** Creates the belief log file and writes a header row if not already present. */
    private void ensureBeliefLogHeader() {
        final File f = new File(BELIEF_LOG_FILE);
        if (f.exists() && f.length() > 0) return;  // already has content

        // getParentFile() returns null when the path has no directory component
        // (e.g. just "belief_accuracy.csv" in the CWD). Guard against NPE.
        final File parent = f.getParentFile();
        if (parent != null) parent.mkdirs();
        try (final PrintWriter pw = new PrintWriter(new FileWriter(f, true))) {
            pw.println("game_id,move_num,player,num_hidden_pieces,"
                     + "avg_jaccard,min_jaccard,max_jaccard,num_determinizations");
        } catch (final IOException e) {
            System.err.println("[ISMCTSAgent] WARNING: Could not write belief log header: " + e.getMessage());
        }
    }

    /**
     * Appends one row to the belief accuracy CSV.
     *
     * @param numHidden      number of hidden opponent pieces (ground truth size)
     * @param jaccardPerDet  per-determinization Jaccard values (NaN = det failed)
     */
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

        if (valid == 0) return;   // all determinizations failed — skip row

        final double avg = sum / valid;
        if (min == Double.MAX_VALUE)        min = Double.NaN;
        if (max == Double.NEGATIVE_INFINITY) max = Double.NaN;

        try (final PrintWriter pw = new PrintWriter(
                new FileWriter(new File(BELIEF_LOG_FILE), true))) {
            pw.printf("%d,%d,%d,%d,%.6f,%.6f,%.6f,%d%n",
                    gameId, moveCount, playerId, numHidden,
                    avg, min, max, valid);
        } catch (final IOException e) {
            System.err.println("[ISMCTSAgent] WARNING: Could not write belief log row: " + e.getMessage());
        }
    }

    // -----------------------------------------------------------------------
    // Move-matching helper
    // -----------------------------------------------------------------------

    /**
     * Returns the first move in {@code legalMoves} whose {@code from()} and
     * {@code to()} sites match {@code target}, or {@code null} if none match.
     */
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
        return "IS-MCTS";
    }

    @Override
    public boolean supportsGame(final Game game) {
        return game.players().count() == 2 && game.hiddenInformation();
    }

    @Override
    public String toString() {
        return "ISMCTSAgent(det=" + numDeterminizations
                + ", sims=" + SIMS_PER_DET + ")";
    }
}
