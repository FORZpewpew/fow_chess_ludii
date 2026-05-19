#!/usr/bin/env python3
"""
Visualisation for FoW Chess tournament results.

Produces:
  - ludii/evaluation/results/figures/win_rate_heatmap.png
  - ludii/evaluation/results/figures/elo_bar_chart.png
  - ludii/evaluation/results/figures/game_length_distribution.png

Usage:
    python3 ludii/evaluation/scripts/plot_results.py \
        --games  ludii/evaluation/results/games_log.csv \
        --elo    ludii/evaluation/results/elo_ratings.csv \
        --matrix ludii/evaluation/results/win_rate_matrix.csv \
        --outdir ludii/evaluation/results/figures
"""

import argparse
import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_win_rate_matrix(path: str):
    """Returns (agents, matrix_ndarray)."""
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        agents = header[1:]  # first cell is empty
        rows = []
        for row in reader:
            vals = []
            for v in row[1:]:
                vals.append(float(v) if v.strip() else float('nan'))
            rows.append(vals)
    return agents, np.array(rows)


def load_elo(path: str):
    """Returns (agents, elos) sorted by Elo descending."""
    agents, elos = [], []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            agents.append(row['agent'])
            elos.append(float(row['elo']))
    # Sort by elo descending
    paired = sorted(zip(elos, agents), reverse=True)
    elos, agents = zip(*paired) if paired else ([], [])
    return list(agents), list(elos)


def load_game_lengths(path: str) -> Dict[str, List[int]]:
    """Returns {agent_pair_key: [num_moves, ...]}."""
    lengths: Dict[str, List[int]] = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = f"{row['agent_p1']} vs {row['agent_p2']}"
            lengths.setdefault(key, []).append(int(row['num_moves']))
    return lengths


# ---------------------------------------------------------------------------
# Plot 1: Win-rate heatmap
# ---------------------------------------------------------------------------

def plot_win_rate_heatmap(agents: List[str], matrix: np.ndarray, outdir: str) -> None:
    n = len(agents)
    fig, ax = plt.subplots(figsize=(max(6, n + 1), max(5, n)))

    # Replace NaN diagonal with 0.5 for visual consistency
    display_matrix = matrix.copy()
    for i in range(n):
        display_matrix[i, i] = 0.5

    im = ax.imshow(display_matrix, cmap='RdYlGn', vmin=0.0, vmax=1.0, aspect='auto')

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(agents, fontsize=9)
    ax.set_xlabel('Opponent (column agent is P2)', fontsize=10)
    ax.set_ylabel('Agent (row agent is P1)', fontsize=10)
    ax.set_title('Win-rate Matrix: W[row][col] = win rate of row agent vs col agent',
                 fontsize=11, pad=12)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = display_matrix[i, j]
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', fontsize=10, color='gray')
            elif not np.isnan(val):
                color = 'white' if val < 0.25 or val > 0.75 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Win rate', fraction=0.046, pad=0.04)
    plt.tight_layout()

    path = os.path.join(outdir, 'win_rate_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot_results] Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: Elo bar chart
# ---------------------------------------------------------------------------

def plot_elo_bar_chart(agents: List[str], elos: List[float], outdir: str) -> None:
    n = len(agents)
    fig, ax = plt.subplots(figsize=(max(6, n + 2), 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, n))
    bars = ax.barh(range(n), elos, color=colors, edgecolor='#333', linewidth=0.7)

    ax.set_yticks(range(n))
    ax.set_yticklabels(agents, fontsize=11)
    ax.set_xlabel('Elo Rating', fontsize=11)
    ax.set_title('Agent Elo Ratings — FoW Chess Tournament', fontsize=13, pad=10)
    ax.axvline(x=1500, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='Baseline 1500')

    # Annotate bars with elo values
    for bar, elo in zip(bars, elos):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f'{elo:.0f}', va='center', ha='left', fontsize=10)

    ax.legend(fontsize=9)
    ax.invert_yaxis()  # strongest agent on top
    plt.tight_layout()

    path = os.path.join(outdir, 'elo_bar_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot_results] Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Game length distribution
# ---------------------------------------------------------------------------

def plot_game_length_distribution(game_lengths: Dict[str, List[int]], outdir: str) -> None:
    if not game_lengths:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Aggregate all lengths and per-matchup
    all_lengths = [l for lengths in game_lengths.values() for l in lengths]

    ax.hist(all_lengths, bins=range(1, 102, 2), color='steelblue', alpha=0.75,
            edgecolor='white', linewidth=0.5, label='All games')

    ax.axvline(x=99, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
               label='50-move draw limit (99 half-moves)')
    ax.axvline(x=np.mean(all_lengths), color='orange', linestyle='-', linewidth=1.5,
               label=f'Mean = {np.mean(all_lengths):.1f} half-moves')

    ax.set_xlabel('Game length (half-moves)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Game Length Distribution — FoW Chess', fontsize=13, pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(outdir, 'game_length_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot_results] Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot FoW Chess tournament results.')
    parser.add_argument('--games',  default='ludii/evaluation/results/games_log.csv')
    parser.add_argument('--elo',    default='ludii/evaluation/results/elo_ratings.csv')
    parser.add_argument('--matrix', default='ludii/evaluation/results/win_rate_matrix.csv')
    parser.add_argument('--outdir', default='ludii/evaluation/results/figures')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if os.path.exists(args.matrix):
        agents, matrix = load_win_rate_matrix(args.matrix)
        plot_win_rate_heatmap(agents, matrix, args.outdir)
    else:
        print(f"[plot_results] Skipping heatmap — matrix not found: {args.matrix}")

    if os.path.exists(args.elo):
        agents, elos = load_elo(args.elo)
        plot_elo_bar_chart(agents, elos, args.outdir)
    else:
        print(f"[plot_results] Skipping Elo chart — file not found: {args.elo}")

    if os.path.exists(args.games):
        game_lengths = load_game_lengths(args.games)
        plot_game_length_distribution(game_lengths, args.outdir)
    else:
        print(f"[plot_results] Skipping game length plot — file not found: {args.games}")


if __name__ == '__main__':
    main()
