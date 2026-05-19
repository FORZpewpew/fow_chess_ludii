package agents;

import game.Game;
import game.types.board.SiteType;
import main.collections.FVector;
import metadata.ai.heuristics.terms.HeuristicTerm;
import other.context.Context;
import other.state.State;
import other.topology.TopologyElement;

import java.util.List;

public class FoWHeuristicTerm extends HeuristicTerm {

    private final float fogPenaltyWeight;
    private final float kingSafetyWeight;
    private final float centerWeight;

    // Central squares on 8x8 board (0-indexed, row-major): d4=27, d5=35, e4=28, e5=36
    private static final int[] CENTER_SQUARES = {27, 28, 35, 36};

    public FoWHeuristicTerm(final float fogPenaltyWeight, final float kingSafetyWeight, final float centerWeight) {
        super(null, Float.NaN);
        this.fogPenaltyWeight = fogPenaltyWeight;
        this.kingSafetyWeight = kingSafetyWeight;
        this.centerWeight = centerWeight;
    }

    @Override
    public float computeValue(final Context context, final int player, final float absWeightThreshold) {
        float score = 0.f;

        final State state = context.state();
        final Game game = context.game();
        final int numSites = game.board().numSites();

        // Determine opponent player index
        final int opponent = (player == 1) ? 2 : 1;

        // --- FogPenalty: count enemy pieces alive but not visible to `player` ---
        int unobservedEnemies = 0;
        for (int site = 0; site < numSites; site++) {
            final int owner = state.containerStates()[0].who(site, SiteType.Cell);
            if (owner == opponent) {
                // Check if this site is hidden from `player`
                if (state.containerStates()[0].isHidden(player - 1, site, 0, SiteType.Cell)) {
                    unobservedEnemies++;
                }
            }
        }
        score -= fogPenaltyWeight * unobservedEnemies;

        // --- KingSafety: squares adjacent to own king not visible to enemy ---
        int safeKingSquares = 0;
        // Find own king's site
        int kingSite = -1;
        for (int site = 0; site < numSites; site++) {
            final int owner = state.containerStates()[0].who(site, SiteType.Cell);
            if (owner == player) {
                final int what = state.containerStates()[0].what(site, SiteType.Cell);
                // King component index — check by piece name if available
                if (what > 0) {
                    final String pieceName = game.equipment().components()[what].name();
                    if (pieceName != null && pieceName.startsWith("King")) {
                        kingSite = site;
                        break;
                    }
                }
            }
        }
        if (kingSite >= 0) {
            final TopologyElement kingElement = game.board().topology().getGraphElement(SiteType.Cell, kingSite);
            if (kingElement != null) {
                final List<? extends TopologyElement> adjacentSites = kingElement.neighbours();
                for (final TopologyElement adj : adjacentSites) {
                    final int adjSite = adj.index();
                    // Safe if hidden from enemy
                    if (state.containerStates()[0].isHidden(opponent - 1, adjSite, 0, SiteType.Cell)) {
                        safeKingSquares++;
                    }
                }
            }
        }
        score += kingSafetyWeight * safeKingSquares;

        // --- CenterControl: own pieces attacking central squares ---
        int centerControlCount = 0;
        for (final int centerSite : CENTER_SQUARES) {
            if (centerSite >= numSites) continue;
            // Count own pieces that have this square in their visible range
            // Approximation: if a central square is visible to `player` from an own piece,
            // count it as controlled (we use visibility as proxy for attack reach)
            if (!state.containerStates()[0].isHidden(player - 1, centerSite, 0, SiteType.Cell)) {
                centerControlCount++;
            }
        }
        score += centerWeight * centerControlCount;

        return score;
    }

    @Override
    public FoWHeuristicTerm copy() {
        return new FoWHeuristicTerm(fogPenaltyWeight, kingSafetyWeight, centerWeight);
    }

    @Override
    public float maxAbsWeight() {
        return Math.max(Math.max(Math.abs(fogPenaltyWeight), Math.abs(kingSafetyWeight)), Math.abs(centerWeight));
    }

    @Override
    public String toString() {
        return "FoWHeuristicTerm(fogPenalty=" + fogPenaltyWeight
                + ", kingSafety=" + kingSafetyWeight
                + ", center=" + centerWeight + ")";
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
        return "FoW heuristic: fog penalty + king safety + center control";
    }

    @Override
    public String toEnglishString(final Context context, final int player) {
        return toString();
    }

    @Override
    public FVector computeStateFeatureVector(final Context context, final int player) {
        return FVector.wrap(new float[]{computeValue(context, player, 0.f)});
    }

    @Override
    public FVector paramsVector() {
        return FVector.wrap(new float[]{fogPenaltyWeight, kingSafetyWeight, centerWeight});
    }
}
