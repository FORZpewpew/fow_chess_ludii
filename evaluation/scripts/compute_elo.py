#!/usr/bin/env python3
"""
Bayesian Elo rating computation for FoW Chess agent tournament.

Reads games_log.csv and outputs:
  - elo_ratings.csv
  - win_rate_matrix.csv

Usage:
    python3 ludii/evaluation/scripts/compute_elo.py \
        --input  ludii/evaluation/results/games_log.csv \
        --output ludii/evaluation/results/elo_ratings.csv \
        --matrix ludii/evaluation/results/win_rate_matrix.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize


def load_games(csv_path: str) -> List[dict]:
    with open(csv_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def collect_agents(games: List[dict]) -> List[str]:
    """Return sorted list of unique agent names seen as p1 or p2."""
    names = set()
    for g in games:
        names.add(g['agent_p1'])
        names.add(g['agent_p2'])
    return sorted(names)


def build_results_matrix(games: List[dict], agents: List[str]) -> List[List[dict]]:
    """
    Returns matrix[i][j] = {'wins': float, 'games': int}
    where i beats j `wins` times out of `games` games.
    Draws count as 0.5 for each side.
    """
    n   = len(agents)
    idx = {a: i for i, a in enumerate(agents)}
    matrix = [[{'wins': 0.0, 'games': 0} for _ in range(n)] for _ in range(n)]

    for g in games:
        p1     = g['agent_p1']
        p2     = g['agent_p2']
        winner = g.get('winner', '').strip()
        draw   = g.get('draw', 'false').strip().lower() == 'true'

        if p1 not in idx or p2 not in idx:
            continue

        i, j = idx[p1], idx[p2]
        matrix[i][j]['games'] += 1
        matrix[j][i]['games'] += 1

        if draw or winner == '':
            matrix[i][j]['wins'] += 0.5
            matrix[j][i]['wins'] += 0.5
        elif winner == p1:
            matrix[i][j]['wins'] += 1.0
        elif winner == p2:
            matrix[j][i]['wins'] += 1.0

    return matrix


def compute_win_rate_matrix(matrix, agents: List[str]) -> np.ndarray:
    """Compute win rate matrix W where W[i][j] = wins_i / games_i_vs_j."""
    n = len(agents)
    W = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j]['games'] > 0:
                W[i][j] = matrix[i][j]['wins'] / matrix[i][j]['games']
    return W


def elo_log_likelihood(ratings: np.ndarray, matrix) -> float:
    """
    Negative log-likelihood of observed results under the Elo model.
    P(i beats j) = 1 / (1 + 10^((r_j - r_i) / 400))
    Draws are handled as 0.5 wins for each side.
    """
    n   = len(ratings)
    nll = 0.0
    for i in range(n):
        for j in range(n):
            if i == j or matrix[i][j]['games'] == 0:
                continue
            wins  = matrix[i][j]['wins']
            games = matrix[i][j]['games']
            p_ij  = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / 400.0))
            p_ij  = np.clip(p_ij, 1e-9, 1 - 1e-9)
            nll  -= wins * np.log(p_ij)
            nll  -= (games - wins) * np.log(1.0 - p_ij)
    return nll


def compute_elo_ratings(matrix, agents: List[str], init_elo: float = 1500.0) -> np.ndarray:
    """
    Find Elo ratings that minimize the negative log-likelihood.
    Anchors the mean rating to init_elo.
    """
    n  = len(agents)
    x0 = np.full(n, init_elo)

    result = minimize(
        fun=elo_log_likelihood,
        x0=x0,
        args=(matrix,),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12},
    )

    ratings = result.x - np.mean(result.x) + init_elo
    return ratings


def save_elo_csv(path: str, agents: List[str], ratings: np.ndarray,
                 matrix, win_rate_matrix: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    n    = len(agents)
    rows = []
    for i, agent in enumerate(agents):
        total_games = sum(matrix[i][j]['games'] for j in range(n) if j != i)
        total_wins  = sum(matrix[i][j]['wins']  for j in range(n) if j != i)
        total_wr    = (total_wins / total_games) if total_games > 0 else float('nan')
        rows.append({
            'agent':            agent,
            'elo':              round(float(ratings[i]), 1),
            'total_games':      total_games,
            'total_wins':       total_wins,
            'overall_win_rate': round(total_wr, 4),
            'rank':             0,
        })

    rows.sort(key=lambda r: r['elo'], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row['rank'] = rank

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['rank', 'agent', 'elo',
                                               'total_games', 'total_wins',
                                               'overall_win_rate'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[compute_elo] Elo ratings saved to {path}")
    for row in rows:
        print(f"  #{row['rank']:1d}  {row['agent']:15s}  Elo={row['elo']:7.1f}  "
              f"WR={row['overall_win_rate']:.3f}  ({row['total_wins']:.0f}/{row['total_games']})")


def save_win_rate_matrix_csv(path: str, agents: List[str],
                              win_rate_matrix: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + agents)
        for i, agent in enumerate(agents):
            row = [agent]
            for j in range(len(agents)):
                val = win_rate_matrix[i][j]
                row.append('' if np.isnan(val) else f'{val:.4f}')
            writer.writerow(row)
    print(f"[compute_elo] Win-rate matrix saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Compute Elo ratings for FoW Chess tournament.')
    parser.add_argument(
        '--results-dir',
        default=None,
        help='Results directory; sets default --input/--output/--matrix paths. '
             'Default: ludii/evaluation/results_v3/',
    )
    parser.add_argument('--input',  default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--matrix', default=None)
    parser.add_argument(
        '--agents',
        default=None,
        help='Comma-separated agent slugs to include (optional; all agents in CSV are used if omitted).',
    )
    args = parser.parse_args()

    results_dir = args.results_dir or 'ludii/evaluation/results_v3'
    if args.input  is None: args.input  = os.path.join(results_dir, 'games_log.csv')
    if args.output is None: args.output = os.path.join(results_dir, 'elo_ratings.csv')
    if args.matrix is None: args.matrix = os.path.join(results_dir, 'win_rate_matrix.csv')

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[compute_elo] Loading games from {args.input}")
    games = load_games(args.input)
    print(f"[compute_elo] Loaded {len(games)} games")

    agents = collect_agents(games)
    print(f"[compute_elo] Agents: {agents}")

    matrix          = build_results_matrix(games, agents)
    win_rate_matrix = compute_win_rate_matrix(matrix, agents)
    ratings         = compute_elo_ratings(matrix, agents)

    save_elo_csv(args.output, agents, ratings, matrix, win_rate_matrix)
    save_win_rate_matrix_csv(args.matrix, agents, win_rate_matrix)


if __name__ == '__main__':
    main()
