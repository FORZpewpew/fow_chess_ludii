#!/usr/bin/env python3
"""
Pairwise statistical significance testing for the FoW Chess tournament.

Reads ludii/evaluation/results/games_log.csv and outputs:
  ludii/evaluation/results/results_summary.csv

Usage:
    python3 ludii/evaluation/scripts/significance.py \
        --input  ludii/evaluation/results/games_log.csv \
        --output ludii/evaluation/results/results_summary.csv \
        --elo    ludii/evaluation/results/elo_ratings.csv
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

from scipy.stats import binomtest


# ---------------------------------------------------------------------------
# Data loading (reuses logic from compute_elo.py style)
# ---------------------------------------------------------------------------

def load_games(csv_path: str) -> List[dict]:
    with open(csv_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_elo(elo_path: str) -> Dict[str, float]:
    """Returns {agent_name: elo_rating}."""
    elo = {}
    if not os.path.exists(elo_path):
        return elo
    with open(elo_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            elo[row['agent']] = float(row['elo'])
    return elo


# ---------------------------------------------------------------------------
# Pairwise result accumulation
# ---------------------------------------------------------------------------

def accumulate_pairwise(games: List[dict]) -> Dict[Tuple[str, str], Dict]:
    """
    Returns {(agent_i, agent_j): {'wins_i': float, 'wins_j': float, 'draws': int, 'games': int}}
    for all unordered pairs {i, j}.
    """
    results = defaultdict(lambda: {'wins_i': 0.0, 'wins_j': 0.0, 'draws': 0, 'games': 0})

    for g in games:
        p1     = g['agent_p1']
        p2     = g['agent_p2']
        winner = g.get('winner', '').strip()
        draw   = g.get('draw', 'false').strip().lower() == 'true'

        # Canonical pair key: alphabetically sorted
        key = (min(p1, p2), max(p1, p2))
        ai, aj = key  # ai < aj alphabetically

        results[key]['games'] += 1

        if draw or winner == '':
            results[key]['draws'] += 1
            results[key]['wins_i'] += 0.5
            results[key]['wins_j'] += 0.5
        elif winner == ai:
            results[key]['wins_i'] += 1.0
        elif winner == aj:
            results[key]['wins_j'] += 1.0

    return dict(results)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96 if confidence == 0.95 else 2.576  # z for 95% / 99%
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    delta  = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - delta), min(1.0, center + delta))


def pairwise_significance(wins_i: float, total: int, alpha: float = 0.05) -> dict:
    """
    Two-sided binomial test for H0: win_rate = 0.5.
    wins_i should be integer (or close to integer after draw handling).
    """
    wins_i_int = int(round(wins_i))
    result = binomtest(wins_i_int, total, p=0.5, alternative='two-sided')
    ci_lo, ci_hi = wilson_ci(wins_i_int, total)
    return {
        'p_value':     result.pvalue,
        'significant': result.pvalue < alpha,
        'ci_95_lo':    round(ci_lo, 4),
        'ci_95_hi':    round(ci_hi, 4),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_summary_csv(path: str, pairwise: dict, elo: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        'agent_i', 'agent_j',
        'games_played', 'wins_i', 'wins_j', 'draws',
        'win_rate_i', 'win_rate_j',
        'elo_i', 'elo_j',
        'p_value', 'significant',
        'ci_95_lo', 'ci_95_hi',
    ]

    rows = []
    for (ai, aj), r in sorted(pairwise.items()):
        total = r['games']
        if total == 0:
            continue
        wr_i = r['wins_i'] / total
        wr_j = r['wins_j'] / total
        sig  = pairwise_significance(r['wins_i'], total)
        rows.append({
            'agent_i':      ai,
            'agent_j':      aj,
            'games_played': total,
            'wins_i':       r['wins_i'],
            'wins_j':       r['wins_j'],
            'draws':        r['draws'],
            'win_rate_i':   round(wr_i, 4),
            'win_rate_j':   round(wr_j, 4),
            'elo_i':        round(elo.get(ai, float('nan')), 1),
            'elo_j':        round(elo.get(aj, float('nan')), 1),
            'p_value':      round(sig['p_value'], 6),
            'significant':  sig['significant'],
            'ci_95_lo':     sig['ci_95_lo'],
            'ci_95_hi':     sig['ci_95_hi'],
        })

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[significance] Results summary saved to {path}")
    print(f"\n{'Agent i':15s}  {'Agent j':15s}  {'WR_i':>6s}  {'WR_j':>6s}  {'p-val':>10s}  Sig?")
    print('-' * 75)
    for row in rows:
        sig_mark = '***' if row['significant'] else '   '
        print(f"  {row['agent_i']:13s}  {row['agent_j']:13s}  "
              f"{row['win_rate_i']:6.3f}  {row['win_rate_j']:6.3f}  "
              f"{row['p_value']:10.4e}  {sig_mark}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Pairwise significance tests for FoW Chess tournament.')
    parser.add_argument('--input',  default='ludii/evaluation/results/games_log.csv')
    parser.add_argument('--output', default='ludii/evaluation/results/results_summary.csv')
    parser.add_argument('--elo',    default='ludii/evaluation/results/elo_ratings.csv')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    games    = load_games(args.input)
    elo      = load_elo(args.elo)
    pairwise = accumulate_pairwise(games)
    save_summary_csv(args.output, pairwise, elo)


if __name__ == '__main__':
    main()
