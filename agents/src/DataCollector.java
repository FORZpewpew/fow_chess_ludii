package agents;

import game.Game;
import game.types.board.SiteType;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.state.State;
import other.trial.Trial;

import java.io.*;
import java.util.*;

/**
 * DataCollector — Standalone self-play data generator for FoW Chess.
 *
 * Plays random self-play games and logs per-position data to a JSONL file
 * (one JSON object per line), suitable for training PPO, PPO+LSTM, and
 * IS-MCTS models in Python.
 *
 * Usage:
 *   java -cp "Ludii-1.3.14.jar:agents/jars/agents.jar" agents.DataCollector \
 *     --game      <path/to/FoW_Chess.lud>   \
 *     --num-games <N>                         \   # default 1000
 *     --max-moves <M>                         \   # default 400
 *     --output    <path/to/output.jsonl>      \
 *     --player    <1|2|both>                  \   # default: both
 *     --sample-every <K>                          # default: 3
 */
public class DataCollector {

    public static void main(final String[] args) throws Exception {
        // ---- Parse CLI arguments ----
        final String gamePath    = getArg(args, "--game",         "ludii/FoW_Chess.lud");
        final int    numGames    = Integer.parseInt(getArg(args, "--num-games",    "1000"));
        final int    maxMoves    = Integer.parseInt(getArg(args, "--max-moves",    "400"));
        final String outputPath  = getArg(args, "--output",       "training/results/selfplay_data.jsonl");
        final String playerArg   = getArg(args, "--player",       "both");
        final int    sampleEvery = Integer.parseInt(getArg(args, "--sample-every", "3"));

        // Determine which players to record
        final boolean recordP1 = playerArg.equals("1") || playerArg.equals("both");
        final boolean recordP2 = playerArg.equals("2") || playerArg.equals("both");

        System.out.printf("[DataCollector] games=%d  max-moves=%d  sample-every=%d  player=%s%n",
                numGames, maxMoves, sampleEvery, playerArg);
        System.out.printf("[DataCollector] game=%s%n", gamePath);
        System.out.printf("[DataCollector] output=%s%n", outputPath);

        // ---- Load game ----
        final File gameFile = new File(gamePath);
        if (!gameFile.exists()) {
            System.err.println("ERROR: Game file not found: " + gamePath);
            System.exit(1);
        }
        final Game game = GameLoader.loadGameFromFile(gameFile);
        if (game == null) {
            System.err.println("ERROR: GameLoader returned null for: " + gamePath);
            System.exit(1);
        }
        System.out.println("[DataCollector] Loaded game: " + game.name());

        // Number of board sites (should be 64 for chess)
        final int numSites = game.board().numSites();
        System.out.println("[DataCollector] Board sites: " + numSites);

        // ---- Ensure output directory exists ----
        final File outFile = new File(outputPath);
        if (outFile.getParentFile() != null) {
            outFile.getParentFile().mkdirs();
        }

        // ---- Main self-play loop ----
        final Random rng = new Random();
        int totalLinesWritten = 0;

        try (final PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(outFile, false)))) {

            for (int gameId = 1; gameId <= numGames; gameId++) {

                // --- Setup trial ---
                final Trial   trial   = new Trial(game);
                final Context context = new Context(game, trial);
                game.start(context);

                // Per-game sample buffer: each entry is a GameSample awaiting outcome
                final List<GameSample> gameSamples = new ArrayList<>();

                // Move history for this game: {from, to, mover}
                final List<int[]> moveHistory = new ArrayList<>();

                // ---- Play game ----
                int plyCount = 0;
                while (!trial.over() && plyCount < maxMoves) {
                    final int mover = context.state().mover();
                    if (mover < 1 || mover > 2) break;

                    // Sample this ply?
                    if (plyCount % sampleEvery == 0) {
                        // Count legal moves before recording (doesn't alter state)
                        final int legalMoveCount = game.moves(context).moves().size();

                        // Snapshot move history at this point
                        final List<int[]> historySnapshot = new ArrayList<>(moveHistory.size());
                        for (final int[] entry : moveHistory) {
                            historySnapshot.add(entry.clone());
                        }

                        // Record for the appropriate player perspective(s)
                        if (recordP1) {
                            gameSamples.add(snapshotPosition(
                                    context, game, gameId, plyCount, 1,
                                    numSites, legalMoveCount, historySnapshot));
                        }
                        if (recordP2) {
                            gameSamples.add(snapshotPosition(
                                    context, game, gameId, plyCount, 2,
                                    numSites, legalMoveCount, historySnapshot));
                        }
                    }

                    // Select and apply a random move
                    final var moves = game.moves(context).moves();
                    if (moves.isEmpty()) break;

                    final Move chosen = moves.get(rng.nextInt(moves.size()));

                    // Record move history entry BEFORE applying
                    final int fromSite = chosen.fromNonDecision();
                    final int toSite   = chosen.toNonDecision();
                    moveHistory.add(new int[]{fromSite, toSite, mover});

                    game.apply(context, chosen);
                    plyCount++;
                }

                // ---- Determine outcome ----
                final boolean hitMoveLimit = !trial.over() && plyCount >= maxMoves;

                double outcome1 = 0.0;
                double outcome2 = 0.0;

                if (!hitMoveLimit) {
                    final double[] ranking = trial.ranking();
                    if (ranking != null && ranking.length > 2) {
                        if (ranking[1] < ranking[2]) {
                            // P1 won (lower rank = winner in Ludii)
                            outcome1 =  1.0;
                            outcome2 = -1.0;
                        } else if (ranking[2] < ranking[1]) {
                            // P2 won
                            outcome1 = -1.0;
                            outcome2 =  1.0;
                        }
                        // else equal ranking → draw → 0.0 for both
                    }
                }
                // if hitMoveLimit, both remain 0.0 (draw)

                if (hitMoveLimit) {
                    System.out.printf("  [WARN] Game %d hit max-moves limit (%d) — recording as draw.%n",
                            gameId, maxMoves);
                }

                // ---- Retroactively fill outcome and write ----
                for (final GameSample sample : gameSamples) {
                    final double outcome = (sample.player == 1) ? outcome1 : outcome2;
                    final String json = buildJson(sample, outcome);
                    writer.println(json);
                    totalLinesWritten++;
                }
                writer.flush(); // flush after each game for crash-recovery

                System.out.printf("  Game %4d/%d: plies=%3d  samples=%d  outcome=[%.1f,%.1f]%n",
                        gameId, numGames, plyCount, gameSamples.size(), outcome1, outcome2);
            }
        }

        System.out.printf("%n[DataCollector] Done. Total lines written: %d  Output: %s%n",
                totalLinesWritten, outputPath);
        System.exit(0);
    }

    // =========================================================================
    // Inner data holder for a position snapshot
    // =========================================================================

    private static class GameSample {
        int gameId;
        int ply;
        int player;       // whose perspective
        // board cells: parallel arrays
        int[]     siteIdx;
        int[]     owner;
        int[]     pieceType;
        String[]  pieceName;
        boolean[] hidden;
        // legal move count at this position
        int legalMoveCount;
        // move history snapshot
        List<int[]> moveHistory;
    }

    // =========================================================================
    // Snapshot current board state from a given player's perspective
    // =========================================================================

    private static GameSample snapshotPosition(
            final Context context,
            final Game game,
            final int gameId,
            final int plyCount,
            final int player,
            final int numSites,
            final int legalMoveCount,
            final List<int[]> historySnapshot) {

        final State state = context.state();
        final GameSample s = new GameSample();
        s.gameId        = gameId;
        s.ply           = plyCount;
        s.player        = player;
        s.legalMoveCount = legalMoveCount;
        s.moveHistory   = historySnapshot;

        s.siteIdx   = new int[numSites];
        s.owner     = new int[numSites];
        s.pieceType = new int[numSites];
        s.pieceName = new String[numSites];
        s.hidden    = new boolean[numSites];

        for (int site = 0; site < numSites; site++) {
            s.siteIdx[site] = site;
            try {
                s.owner[site]     = state.containerStates()[0].who(site, SiteType.Cell);
                s.pieceType[site] = state.containerStates()[0].what(site, SiteType.Cell);
                // isHidden uses 0-based player index
                s.hidden[site]    = state.containerStates()[0]
                        .isHidden(player - 1, site, 0, SiteType.Cell);
            } catch (final Exception e) {
                s.owner[site]     = 0;
                s.pieceType[site] = 0;
                s.hidden[site]    = false;
            }

            // Piece name from component registry
            final int what = s.pieceType[site];
            if (what > 0) {
                try {
                    final var comp = game.equipment().components()[what];
                    s.pieceName[site] = (comp != null && comp.name() != null) ? comp.name() : "";
                } catch (final Exception e) {
                    s.pieceName[site] = "";
                }
            } else {
                s.pieceName[site] = "";
            }
        }
        return s;
    }

    // =========================================================================
    // Build JSON string for a sample (no external library)
    // =========================================================================

    private static String buildJson(final GameSample s, final double outcome) {
        final StringBuilder sb = new StringBuilder(512);
        sb.append('{');

        // Scalar fields
        sb.append("\"game_id\":").append(s.gameId).append(',');
        sb.append("\"ply\":").append(s.ply).append(',');
        sb.append("\"player\":").append(s.player).append(',');
        // outcome: format as e.g. 1.0 / -1.0 / 0.0
        sb.append("\"outcome\":").append(formatDouble(outcome)).append(',');

        // board.cells
        sb.append("\"board\":{\"cells\":[");
        for (int i = 0; i < s.siteIdx.length; i++) {
            if (i > 0) sb.append(',');
            sb.append('{');
            sb.append("\"site\":").append(s.siteIdx[i]).append(',');
            sb.append("\"owner\":").append(s.owner[i]).append(',');
            sb.append("\"piece_type\":").append(s.pieceType[i]).append(',');
            sb.append("\"piece_name\":\"").append(escapeJson(s.pieceName[i])).append("\",");
            sb.append("\"hidden\":").append(s.hidden[i]);
            sb.append('}');
        }
        sb.append("]},");

        // legal_move_count
        sb.append("\"legal_move_count\":").append(s.legalMoveCount).append(',');

        // move_history
        sb.append("\"move_history\":[");
        for (int i = 0; i < s.moveHistory.size(); i++) {
            if (i > 0) sb.append(',');
            final int[] entry = s.moveHistory.get(i);
            sb.append('{');
            sb.append("\"from\":").append(entry[0]).append(',');
            sb.append("\"to\":").append(entry[1]).append(',');
            sb.append("\"mover\":").append(entry[2]);
            sb.append('}');
        }
        sb.append(']');

        sb.append('}');
        return sb.toString();
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    /** Format double as fixed-point with one decimal place (e.g. 1.0, -1.0, 0.0). */
    private static String formatDouble(final double v) {
        if (v == 1.0)  return "1.0";
        if (v == -1.0) return "-1.0";
        return "0.0";
    }

    /** Minimal JSON string escaping. */
    private static String escapeJson(final String s) {
        if (s == null || s.isEmpty()) return "";
        final StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            final char c = s.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n");  break;
                case '\r': sb.append("\\r");  break;
                case '\t': sb.append("\\t");  break;
                default:   sb.append(c);      break;
            }
        }
        return sb.toString();
    }

    /** Extracts a named CLI argument or returns defaultValue. */
    private static String getArg(final String[] args, final String flag, final String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equalsIgnoreCase(flag)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
}
