#!/usr/bin/env python3
"""Analyze think-time sensitivity results.
Shows how win rates of search agents change at 2s vs 5s per move.
"""
import csv
import glob
import os
from collections import defaultdict

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results_time_sensitivity')
BASELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results_grave')


def load_results(pattern):
    """Load game results from CSV files matching pattern."""
    results = []
    for fpath in glob.glob(pattern):
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            for row in csv.DictReader(f):
                results.append({
                    'p1':     row.get('p1_agent', row.get('agent1', '')),
                    'p2':     row.get('p2_agent', row.get('agent2', '')),
                    'winner': row.get('winner',   row.get('result', '')),
                    'source': fname,
                })
    return results


def compute_win_rates(results):
    wins:  defaultdict[str, float] = defaultdict(float)
    games: defaultdict[str, int]   = defaultdict(int)
    for r in results:
        p1, p2, winner = r['p1'], r['p2'], r['winner']
        games[p1] += 1
        games[p2] += 1
        if winner == p1:
            wins[p1] += 1
        elif winner == p2:
            wins[p2] += 1
        else:
            wins[p1] += 0.5
            wins[p2] += 0.5
    agents = sorted(games.keys())
    print(f"\n{'Agent':<35} {'Games':>6} {'Wins':>6} {'Win%':>6}")
    print("-" * 58)
    for a in agents:
        g   = games[a]
        w   = wins[a]
        pct = 100 * w / g if g > 0 else 0
        print(f"{a:<35} {g:>6} {w:>6.1f} {pct:>5.1f}%")


if __name__ == '__main__':
    for time_tag in ['2s', '5s']:
        pattern = os.path.join(RESULTS_DIR, f'{time_tag}_*.csv')
        results = load_results(pattern)
        if results:
            print(f"\n=== Time control: {time_tag}/move ({len(results)} games) ===")
            compute_win_rates(results)
        else:
            print(f"\n=== Time control: {time_tag}/move — no results yet ===")
