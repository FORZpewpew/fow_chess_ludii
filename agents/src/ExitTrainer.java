package agents;

import game.Game;
import metadata.ai.heuristics.Heuristics;
import metadata.ai.heuristics.terms.HeuristicTerm;
import metadata.ai.heuristics.terms.Material;
import metadata.ai.heuristics.terms.MobilityAdvanced;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;
import search.minimax.AlphaBetaSearch;
import search.mcts.MCTS;

import java.io.*;
import java.time.Instant;
import java.util.*;

/**
 * ExitTrainer — Standalone Expert Iteration (ExIT) training harness for FoW Chess.
 *
 * <p>Fixes the root cause of the previous ExIT failure: Ludii's built-in
 * {@code training.expert_iteration.ExpertIteration} class calls
 * {@code GameLoader.loadGameFromName()} internally, which only searches
 * Ludii's internal game library and therefore cannot find the custom
 * {@code FoW_Chess.lud} file.  This class uses
 * {@code GameLoader.loadGameFromFile(new File(path))} instead.
 *
 * <h2>Algorithm (Expert Iteration)</h2>
 * <pre>
 *   weights ← zeros(NUM_FEATURES)
 *   for iter = 1 .. numIterations:
 *     // --- Data generation phase ---
 *     for g = 1 .. gamesPerIter:
 *       play a complete game using WeightedABExpert(weights) vs WeightedABExpert(weights)
 *       sample (feature_vector, outcome) pairs from every SAMPLE_EVERY plies
 *     // --- Supervised learning phase ---
 *     for epoch = 1 .. SGD_EPOCHS:
 *       shuffle samples; run online SGD with L2 regularisation
 *     weights ← updated weights
 * </pre>
 *
 * <h2>Key implementation notes</h2>
 * <ul>
 *   <li><b>plyCount bug</b>: {@code trial.numMoves()} inflates by ~1152× on FoW Chess
 *       because every {@code (set Hidden ...)} ludeme fires as an internal move.
 *       An external {@code int plyCount} counter, incremented once per
 *       {@code game.apply(context, move)}, is used throughout.</li>
 *   <li><b>Feature extraction</b>: delegates to
 *       {@code ABLearnedAgent.fillFeatures()} and {@code ABLearnedAgent.dot()}
 *       static methods — no code duplication.</li>
 *   <li><b>Expert agent</b>: {@code WeightedABExpert} (inner class) extends
 *       {@code AlphaBetaSearch} and injects a {@code Heuristics} object
 *       carrying the current weight vector on each {@code initAI()} call.
 *       Iteration 1 uses vanilla UCT to bootstrap diverse data.</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>
 *   java -Xmx4g -cp "Ludii-1.3.14.jar:agents/jars/agents.jar" agents.ExitTrainer \
 *        --game         &lt;path/to/FoW_Chess.lud&gt; \
 *        --num-iterations  10 \
 *        --games-per-iter  20 \
 *        --time-per-move   1.0 \
 *        --output       training/results/exit_training_v1.csv
 * </pre>
 */
public class ExitTrainer {

    // =========================================================================
    // Constants
    // =========================================================================

    private static final int NUM_FEATURES = 10;   // must match ABLearnedAgent
    private static final int BIAS_IDX     = NUM_FEATURES - 1;

    /** Sample one position per SAMPLE_EVERY plies to avoid highly-correlated data. */
    private static final int SAMPLE_EVERY = 5;

    /** SGD epochs per ExIT iteration. */
    private static final int SGD_EPOCHS   = 30;

    /** Learning rate for online SGD. */
    private static final float LR         = 0.001f;

    /** L2 regularisation coefficient (applied to all weights except bias). */
    private static final float LAMBDA     = 1e-5f;

    /** Hard ply cap per self-play game (prevents infinite FoW games). */
    private static final int MAX_PLIES    = 400;

    // =========================================================================
    // Entry point
    // =========================================================================

    public static void main(final String[] args) throws Exception {

        // ---- Parse CLI arguments ----
        final String gamePath      = getArg(args, "--game",            "ludii/FoW_Chess.lud");
        final int    numIterations = Integer.parseInt(getArg(args, "--num-iterations",  "10"));
        final int    gamesPerIter  = Integer.parseInt(getArg(args, "--games-per-iter",  "20"));
        final double timePerMove   = Double.parseDouble(getArg(args, "--time-per-move", "1.0"));
        final String outputPath    = getArg(args, "--output",
                "training/results/exit_training_v1.csv");

        System.out.println("============================================================");
        System.out.println(" ExitTrainer — Expert Iteration for FoW Chess");
        System.out.printf ("  game:          %s%n", gamePath);
        System.out.printf ("  iterations:    %d%n", numIterations);
        System.out.printf ("  games/iter:    %d%n", gamesPerIter);
        System.out.printf ("  time/move(s):  %.2f%n", timePerMove);
        System.out.printf ("  output:        %s%n", outputPath);
        System.out.println("============================================================");

        // ---- Load game using the file-based API (required for custom .lud files) ----
        // NOTE: GameLoader.loadGameFromName() only searches Ludii's internal library
        //       and would throw/return null for custom FoW_Chess.lud.  The correct API
        //       for any file on disk is GameLoader.loadGameFromFile(File).
        final File gameFile = new File(gamePath);
        if (!gameFile.exists()) {
            System.err.println("ERROR: Game file not found: " + gamePath);
            System.exit(1);
        }
        final Game game = GameLoader.loadGameFromFile(gameFile);
        if (game == null) {
            System.err.println("ERROR: GameLoader.loadGameFromFile returned null for: " + gamePath);
            System.exit(1);
        }
        System.out.println("[ExitTrainer] Loaded game: " + game.name());

        // ---- Ensure output directory exists ----
        final File outFile = new File(outputPath);
        if (outFile.getParentFile() != null) {
            outFile.getParentFile().mkdirs();
        }

        // ---- Initialise weights (all zeros → bias=0, features=0) ----
        // weights has NUM_FEATURES=10 elements (indices 0-9).
        // The bias is at features[9]=1.0f inside ABLearnedAgent's feature vector,
        // so weights[9] is the bias weight — no separate 11th slot needed.
        float[] weights = new float[NUM_FEATURES]; // must match ABLearnedAgent feature dimension

        // ---- Open CSV writer (append-friendly) ----
        final boolean csvExists = outFile.exists() && outFile.length() > 0;
        try (final PrintWriter csv = new PrintWriter(new FileWriter(outFile, true))) {

            // Write header only if the file is new / empty
            if (!csvExists) {
                csv.println("game_id,player1,player2,winner,num_plies,duration_ms,timestamp");
            }

            int globalGameId = 0; // monotonically increasing across all iterations

            // =========================================================
            // ExIT main loop
            // =========================================================
            for (int iter = 1; iter <= numIterations; iter++) {

                System.out.printf("%n[ExitTrainer] ===== Iteration %d/%d =====%n",
                        iter, numIterations);

                // Buffer for this iteration's training data
                final List<float[]> allFeatures = new ArrayList<>();
                final List<Float>   allLabels   = new ArrayList<>();

                // Per-iteration accumulators
                int iterP1Wins = 0, iterP2Wins = 0, iterDraws = 0;

                // ---- Data generation phase ----
                for (int g = 1; g <= gamesPerIter; g++) {
                    globalGameId++;
                    final long gameStart = System.currentTimeMillis();

                    // Iteration 1: use vanilla UCT to generate diverse bootstrap data.
                    // Subsequent iterations: use AlphaBeta weighted by current learned value.
                    final String p1Slug = (iter == 1) ? "uct" : "exit_ab";
                    final String p2Slug = (iter == 1) ? "uct" : "exit_ab";

                    // Create fresh expert agents for this game.
                    // WeightedABExpert is created with a *copy* of weights so that
                    // mid-game weight updates (if any) don't affect the current game.
                    final Object agent1 = createExpertAgent(iter, weights.clone(), game);
                    final Object agent2 = createExpertAgent(iter, weights.clone(), game);

                    // Sets up an agent for the current game
                    initAgent(agent1, game, 1);
                    initAgent(agent2, game, 2);

                    // ---- Play one self-play game ----
                    final Trial   trial   = new Trial(game);
                    final Context context = new Context(game, trial);
                    game.start(context);

                    // Per-game sample buffers (filled retroactively with outcome)
                    final List<float[]> gSamplesP1 = new ArrayList<>();
                    final List<float[]> gSamplesP2 = new ArrayList<>();

                    // External ply counter — MUST NOT use trial.numMoves().
                    // On FoW Chess every (set Hidden …) ludeme fires as an internal move,
                    // inflating trial.numMoves() by ~1152× per actual chess ply.
                    int plyCount = 0;

                    while (!trial.over() && plyCount < MAX_PLIES) {
                        final int mover = context.state().mover();
                        if (mover < 1 || mover > 2) break;

                        // Sample board features at this position before the move
                        if (plyCount % SAMPLE_EVERY == 0) {
                            final float[] fv = safeExtract(context, mover, game, plyCount, MAX_PLIES);
                            if (mover == 1) gSamplesP1.add(fv);
                            else            gSamplesP2.add(fv);
                        }

                        // Expert selects move
                        final Move move = selectMove(mover == 1 ? agent1 : agent2,
                                game, context, timePerMove);
                        if (move == null) break;
                        game.apply(context, move);
                        plyCount++; // increment once per actual chess ply
                    }

                    final boolean hitCap = !trial.over() && plyCount >= MAX_PLIES;

                    // ---- Determine outcome ----
                    float outcome1 = 0.0f;
                    String winnerSlug = "";
                    boolean isDraw = hitCap;

                    if (!hitCap) {
                        final double[] ranking = trial.ranking();
                        if (ranking != null && ranking.length > 2) {
                            if (ranking[1] < ranking[2]) {
                                outcome1  =  1.0f;
                                winnerSlug = p1Slug;
                                iterP1Wins++;
                            } else if (ranking[2] < ranking[1]) {
                                outcome1  = -1.0f;
                                winnerSlug = p2Slug;
                                iterP2Wins++;
                            } else {
                                isDraw = true;
                                iterDraws++;
                            }
                        } else {
                            isDraw = true;
                            iterDraws++;
                        }
                    } else {
                        iterDraws++;
                    }
                    final float outcome2 = -outcome1;

                    // Collect samples with assigned outcomes
                    for (final float[] fv : gSamplesP1) { allFeatures.add(fv); allLabels.add(outcome1); }
                    for (final float[] fv : gSamplesP2) { allFeatures.add(fv); allLabels.add(outcome2); }

                    // Write game record to CSV
                    final long durationMs = System.currentTimeMillis() - gameStart;
                    csv.printf("%d,%s,%s,%s,%d,%d,%s%n",
                            globalGameId, p1Slug, p2Slug,
                            isDraw ? "" : winnerSlug,
                            plyCount, durationMs,
                            Instant.now());
                    csv.flush();

                    System.out.printf("  [iter %d] game %3d/%d: plies=%3d  outcome=[%.0f/%.0f]"
                            + "  samples=%d  %s%n",
                            iter, g, gamesPerIter, plyCount, outcome1, outcome2,
                            gSamplesP1.size() + gSamplesP2.size(),
                            isDraw ? "DRAW" : "win=" + winnerSlug);

                    // Close agents to free internal thread pools
                    closeAgent(agent1);
                    closeAgent(agent2);
                }

                System.out.printf("[ExitTrainer] Iter %d data: %d samples  "
                        + "[P1=%d W  P2=%d W  %d D]%n",
                        iter, allFeatures.size(), iterP1Wins, iterP2Wins, iterDraws);

                // ---- Supervised learning phase (SGD) ----
                weights = runSGD(weights, allFeatures, allLabels);

                // Log final weight vector after this iteration
                System.out.printf("[ExitTrainer] Iter %d weights: ", iter);
                for (int wi = 0; wi < weights.length; wi++) {
                    System.out.printf("w%d=%.4f%s", wi, weights[wi],
                            wi < weights.length - 1 ? "  " : "%n");
                }
            } // end iteration loop
        } // end CSV writer

        System.out.println("\n[ExitTrainer] Training complete. Output: " + outputPath);
        System.exit(0);
    }

    // =========================================================================
    // SGD learner
    // =========================================================================

    /**
     * Runs {@value #SGD_EPOCHS} epochs of online SGD on the supplied
     * (feature_vector, label) dataset and returns updated weights.
     *
     * <p>L2 regularisation ({@value #LAMBDA}) is applied to all weights
     * <em>except</em> the bias term at index {@code NUM_FEATURES - 1}
     * (matching ABLearnedAgent's convention where {@code features[9] = 1.0f}
     * is the bias slot inside the feature vector).
     */
    private static float[] runSGD(final float[] initWeights,
                                   final List<float[]> features,
                                   final List<Float>   labels) {
        final float[] weights = initWeights.clone();
        final int N = features.size();
        if (N == 0) {
            System.out.println("[ExitTrainer] SGD skipped — no training samples.");
            return weights;
        }

        final Random rng = new Random();
        final Integer[] idx = new Integer[N];
        for (int i = 0; i < N; i++) idx[i] = i;

        float finalMse = Float.NaN;
        for (int epoch = 0; epoch < SGD_EPOCHS; epoch++) {
            // Per-epoch shuffle to break correlation
            for (int i = N - 1; i > 0; i--) {
                final int j = rng.nextInt(i + 1);
                final int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

            float totalLoss = 0.0f;
            int   validN    = 0;

            for (int ii = 0; ii < N; ii++) {
                final int     i  = idx[ii];
                final float[] fv = features.get(i);
                final float   y  = labels.get(i);

                // Skip NaN samples
                if (Float.isNaN(y)) continue;
                boolean bad = false;
                for (final float f : fv) { if (Float.isNaN(f)) { bad = true; break; } }
                if (bad) continue;

                final float pred  = ABLearnedAgent.dot(weights, fv);
                final float error = pred - y;
                if (Float.isNaN(error)) continue;

                totalLoss += error * error;
                validN++;

                for (int j = 0; j < weights.length && j < fv.length; j++) {
                    // No L2 on bias term (BIAS_IDX = NUM_FEATURES - 1)
                    final float reg  = (j == BIAS_IDX) ? 0.0f : LAMBDA * weights[j];
                    float grad = error * fv[j] + reg;
                    // Gradient clipping to prevent explosion
                    if (grad >  1.0f) grad =  1.0f;
                    if (grad < -1.0f) grad = -1.0f;
                    weights[j] -= LR * grad;
                }
            }

            if (epoch == 0 || epoch == SGD_EPOCHS - 1) {
                finalMse = validN > 0 ? totalLoss / validN : Float.NaN;
                System.out.printf("[ExitTrainer] SGD epoch %2d  MSE=%.4f  valid=%d%n",
                        epoch, finalMse, validN);
            }
        }
        return weights;
    }

    // =========================================================================
    // Expert agent factory helpers
    // =========================================================================

    /**
     * Creates the expert agent for the given iteration.
     *
     * <ul>
     *   <li>Iteration 1: vanilla UCT (MCTS) to bootstrap diverse training data
     *       without any learned value bias.</li>
     *   <li>Iterations 2+: {@link WeightedABExpert} wrapping current weights,
     *       so the expert improves as training progresses.</li>
     * </ul>
     *
     * Returns either an {@link MCTS} or a {@link WeightedABExpert} instance
     * boxed as {@link Object}; {@link #initAgent}, {@link #selectMove}, and
     * {@link #closeAgent} dispatch on the runtime type.
     */
    private static Object createExpertAgent(final int iteration,
                                             final float[] weights,
                                             final Game game) {
        if (iteration == 1) {
            return MCTS.createUCT();
        }
        return new WeightedABExpert(weights);
    }

    private static void initAgent(final Object agent, final Game game, final int playerIdx) {
        if (agent instanceof MCTS mcts) {
            mcts.initAI(game, playerIdx);
        } else if (agent instanceof WeightedABExpert wab) {
            wab.initAI(game, playerIdx);
        }
    }

    private static Move selectMove(final Object agent,
                                    final Game game,
                                    final Context context,
                                    final double timePerMove) {
        if (agent instanceof MCTS mcts) {
            return mcts.selectAction(game, new Context(context), timePerMove, -1, -1);
        } else if (agent instanceof WeightedABExpert wab) {
            return wab.selectAction(game, new Context(context), timePerMove, -1, -1);
        }
        return null;
    }

    private static void closeAgent(final Object agent) {
        if (agent instanceof MCTS mcts) {
            try { mcts.closeAI(); } catch (final Exception ignored) {}
        } else if (agent instanceof WeightedABExpert wab) {
            try { wab.closeAI(); } catch (final Exception ignored) {}
        }
    }

    // =========================================================================
    // Feature extraction helper
    // =========================================================================

    /**
     * Safely extracts a {@value #NUM_FEATURES}-dimensional feature vector by
     * delegating to {@link ABLearnedAgent#fillFeatures}.  Returns a zero
     * vector on any API exception.
     */
    private static float[] safeExtract(final Context context,
                                        final int player,
                                        final Game game,
                                        final int plyCount,
                                        final int maxPlies) {
        final float[] fv = new float[NUM_FEATURES];
        try {
            ABLearnedAgent.fillFeatures(fv, context, player, game,
                    /*computeMobility=*/false, plyCount, maxPlies);
        } catch (final Exception ignored) {
            // leave as zeros
        }
        return fv;
    }

    // =========================================================================
    // CLI helper
    // =========================================================================

    private static String getArg(final String[] args,
                                  final String flag,
                                  final String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equalsIgnoreCase(flag)) return args[i + 1];
        }
        return defaultValue;
    }

    // =========================================================================
    // Inner class: WeightedABExpert
    // =========================================================================

    /**
     * An AlphaBeta search agent whose heuristic is the learned linear value
     * function with a given weight vector.
     *
     * <p>Unlike {@link ABLearnedAgent}, this class does <em>not</em> run
     * self-play training in {@code initAI()} — it simply injects the supplied
     * weights into a {@link Heuristics} object backed by
     * {@link ABLearnedAgent.LearnedValueTerm}.  This makes it suitable as
     * a stateless "current-best-policy" expert inside the ExIT loop.
     *
     * <p>The class is package-private and lives in {@code agents.*} so that it
     * can reference {@link ABLearnedAgent.LearnedValueTerm} without reflection.
     */
    static final class WeightedABExpert extends AlphaBetaSearch {

        private final float[] weights;

        WeightedABExpert(final float[] weights) {
            super();
            this.friendlyName = "ExIT-WeightedAB";
            this.weights = weights.clone();
        }

        @Override
        public void initAI(final Game game, final int playerIndex) {
            // Call AlphaBetaSearch.initAI() to set up minimax infrastructure,
            // then *override* the heuristic with the current learned weights.
            super.initAI(game, playerIndex);

            final Heuristics h = new Heuristics(new HeuristicTerm[]{
                new Material(null, Float.valueOf(1.0f), null, null),
                new MobilityAdvanced(null, Float.valueOf(0.02f)),
                new ABLearnedAgent.LearnedValueTerm(weights, 0.3f)
            });
            h.init(game);
            this.heuristicValueFunction = h;
        }

        @Override
        public boolean supportsGame(final Game game) {
            return game.players().count() == 2;
        }

        @Override
        public String toString() {
            return "WeightedABExpert(ExIT)";
        }
    }
}
