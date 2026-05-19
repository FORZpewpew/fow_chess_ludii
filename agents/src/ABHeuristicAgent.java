package agents;

import game.Game;
import metadata.ai.heuristics.Heuristics;
import metadata.ai.heuristics.terms.HeuristicTerm;
import metadata.ai.heuristics.terms.Material;
import metadata.ai.heuristics.terms.MobilityAdvanced;
import other.context.Context;
import other.move.Move;
import search.minimax.AlphaBetaSearch;

public class ABHeuristicAgent extends AlphaBetaSearch {

    public ABHeuristicAgent() {
        super();
        this.friendlyName = "ABHeuristicAgent (FoW)";
    }

    @Override
    public void initAI(final Game game, final int playerIndex) {
        super.initAI(game, playerIndex);
        // Override heuristics with FoW-specific combination
        // MUST call init() so Material initialises its pieceWeights FVector
        final Heuristics heuristics = buildFoWHeuristics();
        heuristics.init(game);
        this.heuristicValueFunction = heuristics;
    }

    /**
     * Builds a Heuristics object combining:
     * - Material balance (standard chess piece values)
     * - MobilityAdvanced (weighted move count)
     * - FoWHeuristicTerm (fog penalty + king safety + center control)
     */
    private Heuristics buildFoWHeuristics() {
        final HeuristicTerm[] terms = new HeuristicTerm[]{
            // Standard material: Q=9, R=5, B=3, N=3, P=1
            new Material(null, Float.valueOf(1.0f), null, null),
            // Mobility (weighted count of legal moves)
            new MobilityAdvanced(null, Float.valueOf(0.02f)),
            // FoW-specific: fog penalty + king safety + center control
            new FoWHeuristicTerm(0.5f, 1.5f, 0.1f)
        };
        return new Heuristics(terms);
    }

    @Override
    public boolean supportsGame(final Game game) {
        // Supports any 2-player game
        return game.players().count() == 2;
    }

    @Override
    public String toString() {
        return "ABHeuristicAgent";
    }
}
