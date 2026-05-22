package agents;

import game.Game;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;
import search.mcts.MCTS;
import utils.AIFactory;

import java.io.*;
import java.time.Instant;
import java.util.*;

/**
 * Headless evaluation harness for FoW Chess agents.
 *
 * Usage:
 *   java -cp "Ludii-1.3.14.jar:agents.jar" agents.EvalRunner \
 *        --agent1 <slug> --agent2 <slug> \
 *        --num-games <N> \
 *        --output <path/to/results.csv> \
 *        [--time-per-move <seconds>]
 *
 * Agent slugs: random, uct, pimc_uct, ab_heuristic, ab_learned, trained_uct,
 *              ismcts, ismcts_v4, ppo, ppo_lstm, ppo_lstm_pretrained,
 *              ppo_lstm_v4, ppo_lstm_pretrained_v4, grave, grave_mast
 */
public class EvalRunner {

    private static final String TRAINED_POLICY_PATH = "ludii/training/results/trained_uct_policy.bin";

    public static void main(final String[] args) throws Exception {
        final String agent1Slug  = getArg(args, "--agent1",        "random");
        final String agent2Slug  = getArg(args, "--agent2",        "uct");
        final int    numGames    = Integer.parseInt(getArg(args, "--num-games",      "100"));
        final String outputPath  = getArg(args, "--output",         "ludii/evaluation/results/results.csv");
        final double timePerMove = Double.parseDouble(getArg(args, "--time-per-move", "1.0"));
        final String gamePath    = getArg(args, "--game",           "ludii/FoW_Chess.lud");
        // Hard move cap prevents infinite FoW games where kings hide indefinitely.
        // Default 300 ≈ 2.5× normal chess max; games exceeding this are recorded as draws.
        final int    maxMoves    = Integer.parseInt(getArg(args, "--max-moves",      "300"));

        System.out.printf("[EvalRunner] %s (P1) vs %s (P2) — %d games, %.1fs/move, max-moves=%d%n",
                agent1Slug, agent2Slug, numGames, timePerMove, maxMoves);

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
        System.out.println("[EvalRunner] Game name: " + game.name());
        if (!"FoW Chess".equals(game.name())) {
            System.err.println("ERROR: Expected game 'FoW Chess' but got '" + game.name()
                    + "'. Check that " + gamePath + " is a valid FoW Chess .lud file.");
            System.exit(1);
        }

        final List<AI> agents = new ArrayList<>();
        agents.add(null); // index 0 unused (Ludii is 1-based)
        agents.add(createAgent(agent1Slug));
        agents.add(createAgent(agent2Slug));

        final File outFile = new File(outputPath);
        outFile.getParentFile().mkdirs();

        final boolean fileExists = outFile.exists() && outFile.length() > 0;
        try (final PrintWriter csv = new PrintWriter(new FileWriter(outFile, true))) {
            if (!fileExists) {
                csv.println("game_id,agent_p1,agent_p2,winner,num_moves,draw,timestamp_utc");
            }

            int p1Wins = 0, p2Wins = 0, draws = 0;

            for (int g = 1; g <= numGames; g++) {
                final Trial   trial   = new Trial(game);
                final Context context = new Context(game, trial);
                game.start(context);

                agents.get(1).initAI(game, 1);
                agents.get(2).initAI(game, 2);

                // Track actual agent plies via an external counter.
                // trial.numMoves() is unusable here: each (set Hidden ...) in UpdateFog
                // fires ~164× per chess ply, inflating the count to ~1152 per actual move.
                int plyCount = 0;
                while (!trial.over() && plyCount < maxMoves) {
                    final int mover = context.state().mover();
                    if (mover < 1 || mover > 2) break;
                    final Move move = agents.get(mover).selectAction(game, new Context(context),
                            timePerMove, -1, -1);
                    if (move == null) break;
                    game.apply(context, move);
                    plyCount++;
                }

                final boolean hitMoveLimit = !trial.over() && plyCount >= maxMoves;
                if (hitMoveLimit) {
                    System.out.printf("  [WARN] Game %d hit max-moves limit (%d) — recording as draw.%n",
                            g, maxMoves);
                }

                final double[] ranking = trial.ranking();
                String  winner = "";
                boolean isDraw = false;

                if (hitMoveLimit) {
                    isDraw = true;
                    draws++;
                } else if (ranking.length > 2 && ranking[1] < ranking[2]) {
                    winner = agent1Slug;
                    p1Wins++;
                } else if (ranking.length > 2 && ranking[2] < ranking[1]) {
                    winner = agent2Slug;
                    p2Wins++;
                } else {
                    isDraw = true;
                    draws++;
                }

                csv.printf("%d,%s,%s,%s,%d,%b,%s%n",
                        g, agent1Slug, agent2Slug,
                        winner, plyCount, isDraw,
                        Instant.now().toString());
                csv.flush();

                System.out.printf("  Game %3d/%d: %-12s  moves=%d  [P1=%d W P2=%d W %d D]%n",
                        g, numGames, isDraw ? "DRAW" : winner, plyCount,
                        p1Wins, p2Wins, draws);
            }

            System.out.printf("%n[EvalRunner] Done. P1(%s)=%d  P2(%s)=%d  Draws=%d%n",
                    agent1Slug, p1Wins, agent2Slug, p2Wins, draws);
        }

        // Close all agents so their executor services and Ludii-internal MCTS thread
        // pools are shut down. Without this, non-daemon threads keep the JVM alive.
        System.out.println("[EvalRunner] Closing agents...");
        for (int i = 1; i <= 2; i++) {
            try { agents.get(i).closeAI(); } catch (final Exception ignored) {}
        }
        System.out.println("[EvalRunner] Exiting.");
        System.exit(0);
    }

    public static AI createAgent(final String slug) throws Exception {
        return switch (slug.toLowerCase(Locale.ROOT)) {
            case "random"       -> AIFactory.createAI("Random");
            case "uct"          -> AIFactory.createAI("UCT");
            case "pimc_uct"     -> new PIMCAgent();
            case "ab_heuristic" -> new ABHeuristicAgent();
            case "ab_learned"   -> new ABLearnedAgent();
            case "ppo"          -> new PPOAgent();
            case "trained_uct"  -> {
                // Trained policy loading from file path is not supported by the Ludii API;
                // fall back to vanilla UCT.
                final boolean policyExists = new File(TRAINED_POLICY_PATH).exists();
                if (!policyExists) {
                    System.err.println("[EvalRunner WARNING] Policy file not found for trained_uct; falling back to vanilla UCT. Results may not match intended evaluation.");
                }
                System.out.println("[EvalRunner] WARNING: Trained policy "
                        + (policyExists ? "found but" : "not found;")
                        + " file-based loading unsupported. Using vanilla UCT.");
                yield MCTS.createUCT();
            }
            case "ismcts"                -> new ISMCTSAgent(10);
            case "ismcts_v4"             -> new ISMCTSAgent(25);
            case "particle_ismcts"       -> new ParticleISMCTSAgent();
            case "lstm_guided_ismcts"    -> new LSTMGuidedISMCTSAgent();
            case "ppo_lstm"              -> new PPOLSTMAgent();
            case "ppo_lstm_pretrained"   -> new PPOLSTMAgent("ppo/checkpoints/ppo_lstm_pretrained.pt");
            case "ppo_lstm_v4"           -> new PPOLSTMAgent("checkpoints/ppo_lstm_v4_policy.pt");
            case "ppo_lstm_pretrained_v4"-> new PPOLSTMAgent("ppo/checkpoints/ppo_lstm_pretrained_v4.pt");
            case "grave" -> new search.mcts.MCTS(
                    new search.mcts.selection.McGRAVE(),
                    new search.mcts.playout.RandomPlayout(200),
                    new search.mcts.backpropagation.MonteCarloBackprop(),
                    new search.mcts.finalmoveselection.RobustChild());
            case "grave_mast" -> new search.mcts.MCTS(
                    new search.mcts.selection.McGRAVE(),
                    new search.mcts.playout.MAST(),
                    new search.mcts.backpropagation.MonteCarloBackprop(),
                    new search.mcts.finalmoveselection.RobustChild());
            default -> throw new IllegalArgumentException(
                    "Unknown agent slug: '" + slug + "'. Valid: random, uct, pimc_uct, "
                    + "ab_heuristic, ab_learned, trained_uct, ppo, ismcts, ismcts_v4, "
                    + "ppo_lstm, ppo_lstm_pretrained, ppo_lstm_v4, ppo_lstm_pretrained_v4, "
                    + "grave, grave_mast, particle_ismcts, lstm_guided_ismcts");
        };
    }

    private static String getArg(final String[] args, final String flag, final String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equalsIgnoreCase(flag)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
}
