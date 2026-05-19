package agents;

import game.Game;
import game.types.board.SiteType;
import other.AI;
import other.context.Context;
import other.move.Move;
import other.state.State;
import search.mcts.MCTS;
import main.collections.FastArrayList;

import java.util.*;
import java.util.concurrent.*;

/**
 * PIMCAgent — Perfect-Information Monte Carlo agent for Fog-of-War Chess.
 *
 * <p>On each move: builds {@value #NUM_DETERMINIZATIONS} random determinizations
 * of the hidden opponent pieces, runs parallelised UCT on each, then returns the
 * move with the most votes (plurality).
 *
 * <p>A fixed pool of {@value #NUM_THREADS} MCTS instances is created once in
 * {@link #initAI} and reused across all determinizations to avoid Ludii-internal
 * thread-pool leaks.
 */
public class PIMCAgent extends AI {

    private static final int    NUM_DETERMINIZATIONS = 20;
    private static final double TIME_PER_DET_SECS    = 0.05;
    private static final int    NUM_THREADS          = 4;

    private int              playerIndex;
    private ExecutorService  executor;

    // Pool of reusable MCTS instances — created once in initAI(), never recreated per-move.
    // Each task acquires one instance, uses it, then returns it to prevent thread-pool leaks.
    private final ArrayBlockingQueue<MCTS> uctPool = new ArrayBlockingQueue<>(NUM_THREADS);

    public PIMCAgent() {
        this.friendlyName = "PIMC-UCT (parallel, reused pool)";
    }

    @Override
    public void initAI(final Game game, final int playerIndex) {
        this.playerIndex = playerIndex;

        if (executor != null && !executor.isShutdown()) {
            executor.shutdownNow();
        }
        for (final MCTS old : uctPool) {
            try { old.closeAI(); } catch (final Exception ignored) {}
        }
        uctPool.clear();

        executor = Executors.newFixedThreadPool(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; t++) {
            final MCTS uct = MCTS.createUCT();
            uct.initAI(game, playerIndex);
            uctPool.add(uct);
        }
    }

    @Override
    public void closeAI() {
        if (executor != null) {
            executor.shutdownNow();
        }
        for (final MCTS uct : uctPool) {
            try { uct.closeAI(); } catch (final Exception ignored) {}
        }
        uctPool.clear();
    }

    @Override
    public Move selectAction(
            final Game game,
            final Context context,
            final double maxSeconds,
            final int maxIterations,
            final int maxDepth)
    {
        final FastArrayList<Move> legalMovesRaw = game.moves(context).moves();
        final List<Move> legalMoves = new ArrayList<>(legalMovesRaw.size());
        for (int i = 0; i < legalMovesRaw.size(); i++) {
            legalMoves.add(legalMovesRaw.get(i));
        }

        if (legalMoves.isEmpty()) return null;
        if (legalMoves.size() == 1) return legalMoves.get(0);

        // Build determinized context copies on the calling thread before submitting
        // to worker threads (Context is not thread-safe to copy concurrently).
        final List<Context> detContexts = new ArrayList<>(NUM_DETERMINIZATIONS);
        for (int d = 0; d < NUM_DETERMINIZATIONS; d++) {
            detContexts.add(determinize(context, game));
        }

        final List<Future<Move>> futures = new ArrayList<>(NUM_DETERMINIZATIONS);
        for (int d = 0; d < NUM_DETERMINIZATIONS; d++) {
            final Context detCtx = detContexts.get(d);
            futures.add(executor.submit(() -> {
                MCTS uct = null;
                try {
                    // Wait up to 500 ms for a pool slot — with 20 tasks and 4 slots
                    // the queue naturally throttles to 4 concurrent MCTS searches.
                    uct = uctPool.poll(500, TimeUnit.MILLISECONDS);
                    if (uct == null) return null;
                    // Do NOT call uct.initAI() here — doing so per-determinization spawns
                    // fresh Ludii-internal thread pools without tearing down old ones,
                    // causing a 1000+ thread leak. Instances are initialised once in initAI().
                    return uct.selectAction(game, detCtx, TIME_PER_DET_SECS, -1, -1);
                } finally {
                    if (uct != null) uctPool.offer(uct);
                }
            }));
        }

        final Map<String, int[]> voteCounts = new LinkedHashMap<>();
        final Map<String, Move>  moveByKey  = new LinkedHashMap<>();

        for (final Future<Move> future : futures) {
            try {
                final Move chosen = future.get(2, TimeUnit.SECONDS);
                if (chosen == null) continue;

                Move matched = findMatchingMove(chosen, legalMoves);
                if (matched == null) matched = legalMoves.get(0);

                final String key = moveKey(matched);
                voteCounts.computeIfAbsent(key, k -> new int[1])[0]++;
                moveByKey.putIfAbsent(key, matched);
            } catch (final Exception ignored) {
                // timed-out or interrupted — skip this determinization's vote
            }
        }

        if (voteCounts.isEmpty()) {
            return legalMoves.get(ThreadLocalRandom.current().nextInt(legalMoves.size()));
        }

        final String bestKey = Collections.max(
                voteCounts.entrySet(),
                Comparator.comparingInt(e -> e.getValue()[0])
        ).getKey();
        return moveByKey.get(bestKey);
    }

    private Context determinize(final Context observableCtx, final Game game) {
        final Context det      = new Context(observableCtx);
        final State   state    = det.state();
        final int     numSites = game.board().numSites();
        final int     opponent = (playerIndex == 1) ? 2 : 1;

        final List<Integer> hiddenOccupiedSites = new ArrayList<>();
        final List<Integer> emptyCandidateSites = new ArrayList<>();

        for (int site = 0; site < numSites; site++) {
            final int     owner        = state.containerStates()[0].who(site, SiteType.Cell);
            final boolean isHiddenFromUs = state.containerStates()[0]
                    .isHidden(playerIndex, site, 0, SiteType.Cell);

            if (isHiddenFromUs) {
                if (owner == opponent) {
                    hiddenOccupiedSites.add(site);
                } else if (owner == 0) {
                    emptyCandidateSites.add(site);
                }
            }
        }

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

            for (final int site : hiddenOccupiedSites) {
                try {
                    state.containerStates()[0].setSite(state, site, 0, 0, 0, 0, 0, 0, SiteType.Cell);
                } catch (final Exception ignored) {}
            }

            for (int i = 0; i < hiddenPieces.size(); i++) {
                try {
                    state.containerStates()[0].setSite(state, allHiddenSites.get(i),
                            hiddenPieceOwners.get(i), hiddenPieces.get(i),
                            0, 0, 0, -1, SiteType.Cell);
                } catch (final Exception ignored) {}
            }
        }

        return det;
    }

    private static Move findMatchingMove(final Move target, final List<Move> legalMoves) {
        for (final Move m : legalMoves) {
            if (m.from() == target.from() && m.to() == target.to()) {
                return m;
            }
        }
        return null;
    }

    private static String moveKey(final Move m) {
        return m.from() + "->" + m.to() + ":" + m.mover();
    }

    @Override
    public boolean supportsGame(final Game game) {
        return game.players().count() == 2 && game.hiddenInformation();
    }

    @Override
    public String toString() {
        return "PIMCAgent(pooled, det=" + NUM_DETERMINIZATIONS + ", threads=" + NUM_THREADS + ")";
    }
}
