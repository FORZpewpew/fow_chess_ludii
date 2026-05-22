#!/usr/bin/env python3
"""Analyze think-time sensitivity results.
Shows how win rates of search agents change at 2s vs 5s per move.

Metrics
-------
win_rate   : wins / (wins + losses + draws)   — draws in denominator, NOT numerator
score_rate : (wins + 0.5 * draws) / total     — traditional chess scoring

Wilson 95 % CI is computed for win_rate only.
"""
import csv
import glob
import os
from collections import defaultdict
from math import sqrt

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results_time_sensitivity')
BASELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results_grave')


def wilson_ci(wins, n, z=1.96):
    """Wilson score 95 % confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_results(pattern):
    """Load game results from CSV files matching pattern.

    CSV columns: game_id, agent_p1, agent_p2, winner, num_moves, draw, timestamp_utc
    """
    results = []
    for fpath in glob.glob(pattern):
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            for row in csv.DictReader(f):
                # Support both column-name conventions just in case
                p1 = row.get('agent_p1', row.get('p1_agent', row.get('agent1', '')))
                p2 = row.get('agent_p2', row.get('p2_agent', row.get('agent2', '')))
                winner = row.get('winner', row.get('result', ''))
                is_draw = row.get('draw', 'false').strip().lower() == 'true'
                results.append({
                    'p1':     p1,
                    'p2':     p2,
                    'winner': winner,
                    'draw':   is_draw,
                    'source': fname,
                })
    return results


def compute_win_rates(results):
    wins:   defaultdict[str, int] = defaultdict(int)
    losses: defaultdict[str, int] = defaultdict(int)
    draws:  defaultdict[str, int] = defaultdict(int)
    games:  defaultdict[str, int] = defaultdict(int)

    for r in results:
        p1, p2, winner, is_draw = r['p1'], r['p2'], r['winner'], r['draw']
        games[p1] += 1
        games[p2] += 1
        if is_draw or winner not in (p1, p2):
            draws[p1] += 1
            draws[p2] += 1
        elif winner == p1:
            wins[p1]   += 1
            losses[p2] += 1
        else:
            wins[p2]   += 1
            losses[p1] += 1

    agents = sorted(games.keys())

    # ── per-agent breakdown ──────────────────────────────────────────────────
    print(f"\n{'Agent':<35} {'Games':>6} {'Wins':>5} {'Losses':>7} {'Draws':>6}")
    print("-" * 64)
    for a in agents:
        g = games[a]
        print(f"{a:<35} {g:>6} {wins[a]:>5} {losses[a]:>7} {draws[a]:>6}")

    # ── win_rate & score_rate ────────────────────────────────────────────────
    header = (f"\n{'Agent':<35} {'Games':>6} {'WinRate':>8} {'95% CI':>18} "
              f"{'ScoreRate':>10}")
    print(header)
    print("-" * 82)
    for a in agents:
        g = games[a]
        w = wins[a]
        d = draws[a]
        win_rate   = w / g if g > 0 else 0.0
        score_rate = (w + 0.5 * d) / g if g > 0 else 0.0
        lo, hi = wilson_ci(w, g)
        ci_str = f"[{lo*100:5.1f}%, {hi*100:5.1f}%]"
        print(f"{a:<35} {g:>6} {win_rate*100:>7.1f}% {ci_str:>18} {score_rate*100:>9.1f}%")

    return {a: {'games': games[a], 'wins': wins[a], 'draws': draws[a],
                'losses': losses[a]} for a in agents}


if __name__ == '__main__':
    for time_tag in ['2s', '5s']:
        pattern = os.path.join(RESULTS_DIR, f'{time_tag}_*.csv')
        results = load_results(pattern)
        n_draws_total = sum(1 for r in results if r['draw'])
        if results:
            print(f"\n{'='*70}")
            print(f"Time control: {time_tag}/move  |  {len(results)} games  |  "
                  f"{n_draws_total} draws")
            print(f"{'='*70}")
            compute_win_rates(results)
        else:
            print(f"\n=== Time control: {time_tag}/move — no results yet ===")
