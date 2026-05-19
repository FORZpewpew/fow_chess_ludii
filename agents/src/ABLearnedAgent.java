package agents;

import game.Game;
import game.types.board.SiteType;
import main.collections.FVector;
import metadata.ai.heuristics.Heuristics;
import metadata.ai.heuristics.terms.HeuristicTerm;
import metadata.ai.heuristics.terms.Material;
import metadata.ai.heuristics.terms.MobilityAdvanced;
import other.context.Context;
import other.move.Move;
import other.state.State;
import other.trial.Trial;
import other.AI;
import search.mcts.MCTS;
import search.minimax.AlphaBetaSearch;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * ABLearnedAgent — Alpha-Beta search with a self-play learned linear value function.
 *
 * <p>Extends the ABHeuristicAgent approach by replacing the FoWHeuristicTerm with
 * a LearnedValueTerm whose weights are fitted via SGD linear regression on
 * 20 headless random-vs-UCT self-play games inside {@link #initAI}.
 *
 * <p>Weights are learned only on the first {@link #initAI} call and reused for
 * all subsequent calls within the same JVM process to avoid repeated training cost.
 */
public class ABLearnedAgent extends AlphaBetaSearch {

    private static final int NUM_FEATURES = 10;

    // Central squares on an 8×8 board (0-indexed, row-major): d4=27, e4=28, d5=35, e5=36
    private static final int[] CENTER_SQUARES = {27, 28, 35, 36};

    // Weights are learned on the first initAI() call and cached for subsequent calls
    private boolean trained       = false;
    private float[] cachedWeights = null;

    public ABLearnedAgent() {
        super();
        this.friendlyName = "ABLearnedAgent (FoW)";
    }

    @Override
    public void initAI(final Game game, final int playerIndex) {
        super.initAI(game, playerIndex);

        if (!trained) {
            System.out.println("[ABLearnedAgent] Starting self-play training (first call)...");
            final long t0 = System.currentTimeMillis();
            cachedWeights = runSelfPlayTraining(game);
            trained = true;
            System.out.printf("[ABLearnedAgent] Training done in %d ms.%n",
                    System.currentTimeMillis() - t0);
        } else {
            System.out.println("[ABLearnedAgent] Reusing cached weights from first training.");
        }

        // Inject heuristics: Material + MobilityAdvanced + LearnedValueTerm
        final Heuristics heuristics = new Heuristics(new HeuristicTerm[]{
            new Material(null, Float.valueOf(1.0f), null, null),
            new MobilityAdvanced(null, Float.valueOf(0.02f)),
            new LearnedValueTerm(cachedWeights, 0.3f)
        });
        // MUST call init() so Material initialises its pieceWeights FVector
        heuristics.init(game);
        this.heuristicValueFunction = heuristics;
    }

    private float[] runSelfPlayTraining(final Game game) {
        final int   NUM_GAMES    = 20;
        final int   MAX_MOVES    = 200;
        final int   SAMPLE_EVERY = 5;
        final int   SGD_EPOCHS   = 30;
        final float LR           = 0.001f;
        final float LAMBDA       = 0.001f;
        final int   BIAS_IDX     = NUM_FEATURES - 1; // bias term: no L2 regularisation

        final List<float[]> allFeatures = new ArrayList<>();
        final List<Float>   allLabels   = new ArrayList<>();
        // Time-based seed produces varied game sequences across calls
        final Random rng = new Random();

        final AI uct = MCTS.createUCT();

        for (int g = 0; g < NUM_GAMES; g++) {
            final int learnedSide = (rng.nextInt(2) == 0) ? 1 : 2;
            final int uctSide     = (learnedSide == 1) ? 2 : 1;

            uct.initAI(game, uctSide);

            final Trial   trial   = new Trial(game);
            final Context context = new Context(game, trial);
            game.start(context);

            final List<float[]> gfLearned = new ArrayList<>();

            int moveCount = 0;
            while (!context.trial().over() && moveCount < MAX_MOVES) {
                final int mover = context.state().mover();
                Move chosenMove = null;

                if (mover == learnedSide) {
                    if (moveCount % SAMPLE_EVERY == 0) {
                        gfLearned.add(extractFeaturesSafe(context, learnedSide, game, moveCount, MAX_MOVES));
                    }
                    // Learned side plays random during training (weights not yet fitted)
                    final var moves = game.moves(context).moves();
                    if (moves.isEmpty()) break;
                    chosenMove = moves.get(rng.nextInt(moves.size()));
                } else {
                    chosenMove = uct.selectAction(game, context, 0.5, -1, -1);
                    if (chosenMove == null) {
                        final var moves = game.moves(context).moves();
                        if (moves.isEmpty()) break;
                        chosenMove = moves.get(rng.nextInt(moves.size()));
                    }
                }

                try {
                    game.apply(context, chosenMove);
                } catch (final Exception e) {
                    break;
                }
                moveCount++;
            }

            // Ludii ranking: rank 1.0 = winner
            final double[] ranking = context.trial().ranking();
            float outcomeP1 = 0.0f;
            if (ranking != null && ranking.length > 2) {
                if      (ranking[1] < ranking[2]) outcomeP1 =  1.0f;
                else if (ranking[1] > ranking[2]) outcomeP1 = -1.0f;
            }
            final float learnedOutcome = (learnedSide == 1) ? outcomeP1 : -outcomeP1;

            for (final float[] fv : gfLearned) {
                allFeatures.add(fv);
                allLabels.add(learnedOutcome);
            }
        }

        System.out.printf("[ABLearnedAgent] Collected %d training samples.%n", allFeatures.size());

        final float[]   weights = new float[NUM_FEATURES];
        final int       N       = allFeatures.size();
        if (N == 0) return weights;

        final Integer[] indices = new Integer[N];
        for (int i = 0; i < N; i++) indices[i] = i;

        for (int epoch = 0; epoch < SGD_EPOCHS; epoch++) {
            // Per-epoch shuffle breaks correlation and prevents oscillation
            for (int i = N - 1; i > 0; i--) {
                final int j = rng.nextInt(i + 1);
                final int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
            }

            float totalLoss = 0.0f;
            int   validN    = 0;
            for (int ii = 0; ii < N; ii++) {
                final int     i  = indices[ii];
                final float[] fv = allFeatures.get(i);
                final float   y  = allLabels.get(i);
                boolean hasNaN = Float.isNaN(y);
                for (final float f : fv) { if (Float.isNaN(f)) { hasNaN = true; break; } }
                if (hasNaN) continue;

                final float pred  = dot(weights, fv);
                final float error = pred - y;
                if (Float.isNaN(error)) continue;

                totalLoss += error * error;
                validN++;
                for (int j = 0; j < NUM_FEATURES; j++) {
                    final float reg = (j == BIAS_IDX) ? 0.0f : LAMBDA * weights[j];
                    // Gradient clipping prevents explosion with large errors
                    float grad = error * fv[j] + reg;
                    if (grad >  1.0f) grad =  1.0f;
                    if (grad < -1.0f) grad = -1.0f;
                    weights[j] -= LR * grad;
                }
            }
            if (epoch == 0 || epoch == SGD_EPOCHS - 1) {
                System.out.printf("[ABLearnedAgent] Epoch %2d  MSE=%.4f  validSamples=%d%n",
                        epoch, validN > 0 ? totalLoss / validN : Float.NaN, validN);
            }
        }

        return weights;
    }

    private float[] extractFeaturesSafe(final Context context,
                                        final int player,
                                        final Game game,
                                        final int plyCount,
                                        final int maxPlies) {
        final float[] features = new float[NUM_FEATURES];
        try {
            fillFeatures(features, context, player, game, false, plyCount, maxPlies);
        } catch (final Exception ignored) {}
        return features;
    }

    static void fillFeatures(final float[] features,
                              final Context context,
                              final int player,
                              final Game game,
                              final boolean computeMobility) {
        fillFeatures(features, context, player, game, computeMobility, -1, 400);
    }

    static void fillFeatures(final float[] features,
                              final Context context,
                              final int player,
                              final Game game,
                              final boolean computeMobility,
                              final int plyCount,
                              final int maxPlies) {
        final State state    = context.state();
        final int   numSites = game.board().numSites();
        final int   opponent = (player == 1) ? 2 : 1;

        int     materialSelf  = 0;
        int     materialOpp   = 0;
        boolean kingSelf      = false;
        boolean kingOpp       = false;
        int     centerControl = 0;
        int     hiddenSites   = 0;

        for (int site = 0; site < numSites; site++) {
            final int     owner  = state.containerStates()[0].who(site, SiteType.Cell);
            final int     what   = state.containerStates()[0].what(site, SiteType.Cell);
            // isHidden uses 1-based player index to match fow_env.py convention
            final boolean hidden = state.containerStates()[0]
                    .isHidden(player, site, 0, SiteType.Cell);

            if (hidden) hiddenSites++;

            if (what > 0) {
                if (owner == player) {
                    final String name = game.equipment().components()[what].name();
                    if (name != null && name.startsWith("King")) {
                        kingSelf = true;
                    } else {
                        materialSelf++;
                    }
                    for (final int cs : CENTER_SQUARES) {
                        if (site == cs) { centerControl++; break; }
                    }
                } else if (owner == opponent && !hidden) {
                    final String name = game.equipment().components()[what].name();
                    if (name != null && name.startsWith("King")) {
                        kingOpp = true;
                    } else {
                        materialOpp++;
                    }
                }
            }
        }

        final float mobility = computeMobility
                ? (float) game.moves(context).moves().size()
                : 0.0f;

        // Normalise all features to approximately [0,1] to prevent gradient explosion.
        // Raw piece counts (0-16) caused 16× gradient scale at LR=0.01.
        final float MAX_PIECES = 16.0f;
        final float MAX_CENTER = 4.0f;
        features[0] = materialSelf / MAX_PIECES;
        features[1] = materialOpp  / MAX_PIECES;
        features[2] = kingSelf  ? 1.0f : 0.0f;
        features[3] = kingOpp   ? 1.0f : 0.0f;
        features[4] = mobility;
        features[5] = numSites > 0 ? (float) hiddenSites / numSites : 0.0f;
        features[6] = (materialSelf - materialOpp) / MAX_PIECES;
        features[7] = centerControl / MAX_CENTER;
        // Game phase as fraction of max plies. At evaluation time plyCount is unknown
        // → use 0.5 as a neutral mid-game value.
        features[8] = (plyCount >= 0) ? Math.min(1.0f, (float) plyCount / maxPlies) : 0.5f;
        features[9] = 1.0f; // bias
    }

    static float dot(final float[] a, final float[] b) {
        float s = 0.0f;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    @Override
    public boolean supportsGame(final Game game) {
        return game.players().count() == 2;
    }

    @Override
    public String toString() {
        return "ABLearnedAgent";
    }

    // =========================================================================
    // Inner class: LearnedValueTerm
    // =========================================================================

    /**
     * HeuristicTerm backed by a dot-product with the self-play learned weight vector.
     * The feature vector is the same 10-dimensional representation used during training
     * (with mobility enabled at evaluation time).
     */
    public static class LearnedValueTerm extends HeuristicTerm {

        private final float[] weights;

        public LearnedValueTerm(final float[] weights, final float termWeight) {
            super(null, termWeight);
            this.weights = weights.clone();
        }

        @Override
        public float computeValue(final Context context,
                                  final int player,
                                  final float absWeightThreshold) {
            try {
                final float[] fv = new float[NUM_FEATURES];
                fillFeatures(fv, context, player, context.game(), true);
                return dot(weights, fv);
            } catch (final Exception e) {
                return 0.0f;
            }
        }

        @Override
        public LearnedValueTerm copy() {
            return new LearnedValueTerm(weights, weight());
        }

        @Override
        public float maxAbsWeight() {
            float max = Math.abs(weight());
            for (final float w : weights) max = Math.max(max, Math.abs(w));
            return max;
        }

        @Override
        public String toString() {
            return "LearnedValueTerm(dims=" + weights.length + ", termWeight=" + weight() + ")";
        }

        @Override
        public String toStringThresholded(final float threshold) {
            return toString();
        }

        @Override
        public boolean isApplicable(final Game game) {
            return game.players().count() == 2;
        }

        @Override
        public String description() {
            return "Self-play learned linear value function (" + weights.length + " features)";
        }

        @Override
        public String toEnglishString(final Context context, final int player) {
            return toString();
        }

        @Override
        public FVector computeStateFeatureVector(final Context context, final int player) {
            final float[] fv = new float[NUM_FEATURES];
            try {
                fillFeatures(fv, context, player, context.game(), false);
            } catch (final Exception ignored) {}
            return FVector.wrap(fv);
        }

        @Override
        public FVector paramsVector() {
            return FVector.wrap(weights);
        }
    }
}
