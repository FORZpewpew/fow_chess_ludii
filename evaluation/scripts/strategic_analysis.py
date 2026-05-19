#!/usr/bin/env python3
"""
Strategic insights extraction for FoW Chess agents.

Analyses:
  1. Game-length statistics per agent
  2. Win-rate by colour (P1/P2 advantage) per agent
  3. Policy entropy proxy (move diversity) per agent
  4. Correlation: draw rate vs agent type

Usage:
    python3 ludii/evaluation/scripts/strategic_analysis.py \
        --input  ludii/evaluation/results/games_log.csv \
        --output ludii/evaluation/results/strategic_analysis.csv \
        --outdir ludii/evaluation/results/figures
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_games(path: str) -> List[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Per-agent statistics
# ---------------------------------------------------------------------------

def compute_agent_stats(games: List[dict]) -> Dict[str, dict]:
    """
    Computes per-agent statistics:
      - games_as_p1, games_as_p2
      - wins_as_p1, wins_as_p2
      - draws_total
      - avg_game_length_when_win, avg_game_length_when_loss, avg_game_length_when_draw
      - overall win rate
    """
    stats = defaultdict(lambda: {
        'games_as_p1': 0, 'games_as_p2': 0,
        'wins_as_p1': 0, 'wins_as_p2': 0,
        'draws': 0, 'total_games': 0,
        'lengths_win': [], 'lengths_loss': [], 'lengths_draw': [],
        'opponents_beaten': set(),
    })

    for g in games:
        p1     = g['agent_p1']
        p2     = g['agent_p2']
        winner = g.get('winner', '').strip()
        draw   = g.get('draw', 'false').strip().lower() == 'true'
        nmoves = int(g.get('num_moves', 0))

        for agent in [p1, p2]:
            stats[agent]['total_games'] += 1

        stats[p1]['games_as_p1'] += 1
        stats[p2]['games_as_p2'] += 1

        if draw or winner == '':
            stats[p1]['draws'] += 1
            stats[p2]['draws'] += 1
            stats[p1]['lengths_draw'].append(nmoves)
            stats[p2]['lengths_draw'].append(nmoves)
        elif winner == p1:
            stats[p1]['wins_as_p1'] += 1
            stats[p1]['lengths_win'].append(nmoves)
            stats[p2]['lengths_loss'].append(nmoves)
            stats[p1]['opponents_beaten'].add(p2)
        elif winner == p2:
            stats[p2]['wins_as_p2'] += 1
            stats[p2]['lengths_win'].append(nmoves)
            stats[p1]['lengths_loss'].append(nmoves)
            stats[p2]['opponents_beaten'].add(p1)

    return dict(stats)


def _avg(lst: list) -> float:
    return float(np.mean(lst)) if lst else float('nan')


def _safe_rate(num, denom):
    return num / denom if denom > 0 else float('nan')


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_colour_advantage(stats: dict, outdir: str) -> None:
    """Bar chart: win rate as P1 vs P2 for each agent."""
    agents = sorted(stats.keys())
    wr_p1, wr_p2 = [], []
    for a in agents:
        s = stats[a]
        wr_p1.append(_safe_rate(s['wins_as_p1'], s['games_as_p1']))
        wr_p2.append(_safe_rate(s['wins_as_p2'], s['games_as_p2']))

    x = np.arange(len(agents))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(agents) * 2), 5))
    b1 = ax.bar(x - w/2, wr_p1, w, label='As P1 (White)', color='gold',   edgecolor='#333')
    b2 = ax.bar(x + w/2, wr_p2, w, label='As P2 (Black)', color='dimgray', edgecolor='#333', alpha=0.85)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='50% baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Win Rate', fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Win Rate by Colour Assignment — FoW Chess', fontsize=12, pad=10)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(outdir, 'colour_advantage.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[strategic_analysis] Saved {path}")


def plot_avg_game_length(stats: dict, outdir: str) -> None:
    """Grouped bar chart: avg game length on win/loss/draw per agent."""
    agents = sorted(stats.keys())
    avgs_win  = [_avg(stats[a]['lengths_win'])  for a in agents]
    avgs_loss = [_avg(stats[a]['lengths_loss']) for a in agents]
    avgs_draw = [_avg(stats[a]['lengths_draw']) for a in agents]

    x = np.arange(len(agents))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(9, len(agents) * 2.5), 5))
    ax.bar(x - w,   avgs_win,  w, label='Win',  color='mediumseagreen', edgecolor='#333')
    ax.bar(x,       avgs_loss, w, label='Loss', color='tomato',         edgecolor='#333')
    ax.bar(x + w,   avgs_draw, w, label='Draw', color='cornflowerblue', edgecolor='#333')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Avg. half-moves', fontsize=11)
    ax.set_title('Average Game Length by Outcome — FoW Chess', fontsize=12, pad=10)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(outdir, 'avg_game_length.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[strategic_analysis] Saved {path}")


def plot_draw_rate(stats: dict, outdir: str) -> None:
    """Draw rate per agent."""
    agents = sorted(stats.keys())
    draw_rates = [_safe_rate(stats[a]['draws'], stats[a]['total_games']) for a in agents]

    fig, ax = plt.subplots(figsize=(max(7, len(agents) * 1.8), 4))
    bars = ax.bar(agents, draw_rates, color='cornflowerblue', edgecolor='#333', alpha=0.85)
    ax.axhline(y=np.nanmean(draw_rates), color='orange', linestyle='--', linewidth=1.5,
               label=f"Mean draw rate = {np.nanmean(draw_rates):.3f}")
    ax.set_ylabel('Draw Rate', fontsize=11)
    ax.set_ylim(0, max(0.05, max(r for r in draw_rates if not math.isnan(r))) * 1.2)
    ax.set_title('Draw Rate per Agent — FoW Chess (50-move rule)', fontsize=12, pad=10)
    for bar, rate in zip(bars, draw_rates):
        if not math.isnan(rate):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, 'draw_rate.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[strategic_analysis] Saved {path}")


# ---------------------------------------------------------------------------
# Save summary CSV
# ---------------------------------------------------------------------------

def save_strategic_csv(path: str, stats: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        'agent', 'total_games',
        'wins_as_p1', 'games_as_p1', 'win_rate_p1',
        'wins_as_p2', 'games_as_p2', 'win_rate_p2',
        'draws', 'draw_rate',
        'avg_length_win', 'avg_length_loss', 'avg_length_draw',
    ]
    rows = []
    for agent in sorted(stats.keys()):
        s = stats[agent]
        rows.append({
            'agent': agent,
            'total_games': s['total_games'],
            'wins_as_p1': s['wins_as_p1'],
            'games_as_p1': s['games_as_p1'],
            'win_rate_p1': round(_safe_rate(s['wins_as_p1'], s['games_as_p1']), 4),
            'wins_as_p2': s['wins_as_p2'],
            'games_as_p2': s['games_as_p2'],
            'win_rate_p2': round(_safe_rate(s['wins_as_p2'], s['games_as_p2']), 4),
            'draws': s['draws'],
            'draw_rate': round(_safe_rate(s['draws'], s['total_games']), 4),
            'avg_length_win':  round(_avg(s['lengths_win']),  1),
            'avg_length_loss': round(_avg(s['lengths_loss']), 1),
            'avg_length_draw': round(_avg(s['lengths_draw']), 1),
        })
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[strategic_analysis] Summary saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Strategic analysis for FoW Chess tournament.')
    parser.add_argument('--input',  default='ludii/evaluation/results/games_log.csv')
    parser.add_argument('--output', default='ludii/evaluation/results/strategic_analysis.csv')
    parser.add_argument('--outdir', default='ludii/evaluation/results/figures')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        return

    os.makedirs(args.outdir, exist_ok=True)

    games = load_games(args.input)
    stats = compute_agent_stats(games)

    save_strategic_csv(args.output, stats)
    plot_colour_advantage(stats, args.outdir)
    plot_avg_game_length(stats, args.outdir)
    plot_draw_rate(stats, args.outdir)

    print("\n[strategic_analysis] Per-agent summary:")
    print(f"{'Agent':15s}  {'Total':>6s}  {'WR_P1':>6s}  {'WR_P2':>6s}  {'DrawR':>6s}")
    print('-' * 55)
    for agent in sorted(stats.keys()):
        s = stats[agent]
        wr1   = _safe_rate(s['wins_as_p1'], s['games_as_p1'])
        wr2   = _safe_rate(s['wins_as_p2'], s['games_as_p2'])
        drawr = _safe_rate(s['draws'], s['total_games'])
        print(f"  {agent:13s}  {s['total_games']:6d}  "
              f"{wr1:6.3f}  {wr2:6.3f}  {drawr:6.3f}")


if __name__ == '__main__':
    main()
