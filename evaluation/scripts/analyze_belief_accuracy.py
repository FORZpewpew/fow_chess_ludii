#!/usr/bin/env python3
"""
analyze_belief_accuracy.py — Belief-accuracy analysis for IS-MCTS

Reads the belief_accuracy.csv produced by ISMCTSAgent (with
-Dfow.belief.log=true) and produces a summary table suitable for thesis
inclusion.

CSV schema (written by ISMCTSAgent):
  game_id, move_num, player, num_hidden_pieces,
  avg_jaccard, min_jaccard, max_jaccard, num_determinizations

Usage
-----
    python3 evaluation/scripts/analyze_belief_accuracy.py \
        evaluation/results_grave/belief_accuracy.csv

Optional: pass --out <path> to write the summary table to a file instead of
printing to stdout.
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd         = None
    HAS_PANDAS = False


def phase(move_num: int) -> str:
    """Classify a ply number into opening / midgame / endgame."""
    if move_num < 20:
        return "opening"
    elif move_num <= 60:
        return "midgame"
    return "endgame"


def analyse_pandas(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)

    required = {"game_id", "move_num", "player", "num_hidden_pieces",
                "avg_jaccard", "min_jaccard", "max_jaccard", "num_determinizations"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["phase"] = df["move_num"].apply(phase)

    lines = [
        "=" * 72,
        "  IS-MCTS Belief-Accuracy Report",
        "=" * 72,
        f"  Source file : {csv_path}",
        f"  Total rows  : {len(df)}",
        f"  Games       : {df['game_id'].nunique()}",
        f"  Players     : {sorted(df['player'].unique())}",
        "",
        "Overall statistics",
        "-" * 40,
    ]

    for col in ["avg_jaccard", "min_jaccard", "max_jaccard"]:
        m, s, lo, hi = df[col].mean(), df[col].std(), df[col].min(), df[col].max()
        lines.append(f"  {col:<18}  mean={m:.4f}  std={s:.4f}  [{lo:.4f}, {hi:.4f}]")
    lines.append(f"  {'num_hidden_pieces':<18}  mean={df['num_hidden_pieces'].mean():.2f}  "
                 f"max={df['num_hidden_pieces'].max()}")
    lines.append("")

    lines.append("By game phase  (move_num: <20=opening, 20-60=midgame, >60=endgame)")
    lines.append("-" * 72)
    lines.append(f"  {'Phase':<10}  {'N':>6}  {'Mean Avg-J':>11}  {'Std':>7}  "
                 f"{'Mean Min-J':>11}  {'Mean Max-J':>11}")
    lines.append("  " + "-" * 68)
    grp = df.groupby("phase")
    for ph in ["opening", "midgame", "endgame"]:
        if ph not in grp.groups:
            continue
        sub = grp.get_group(ph)
        lines.append(
            f"  {ph:<10}  {len(sub):>6}  "
            f"{sub['avg_jaccard'].mean():>11.4f}  "
            f"{sub['avg_jaccard'].std():>7.4f}  "
            f"{sub['min_jaccard'].mean():>11.4f}  "
            f"{sub['max_jaccard'].mean():>11.4f}"
        )
    lines.append("")

    lines.append("By player")
    lines.append("-" * 50)
    for p, sub in df.groupby("player"):
        lines.append(f"  Player {p}:  N={len(sub)}  mean_avg_jaccard={sub['avg_jaccard'].mean():.4f}  "
                     f"std={sub['avg_jaccard'].std():.4f}")
    lines.append("")

    lines.append("Per-game summary (first 20 games shown)")
    lines.append("-" * 72)
    lines.append(f"  {'game_id':>18}  {'moves':>6}  {'mean_jaccard':>13}  "
                 f"{'min_jaccard':>11}  {'max_jaccard':>11}")
    lines.append("  " + "-" * 65)
    for gid, gsub in list(df.groupby("game_id"))[:20]:
        lines.append(
            f"  {str(gid):>18}  {len(gsub):>6}  "
            f"{gsub['avg_jaccard'].mean():>13.4f}  "
            f"{gsub['min_jaccard'].mean():>11.4f}  "
            f"{gsub['max_jaccard'].mean():>11.4f}"
        )
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


def analyse_pure(csv_path: Path) -> str:
    import csv
    import math

    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "game_id":              row["game_id"],
                "move_num":             int(row["move_num"]),
                "player":               int(row["player"]),
                "num_hidden_pieces":    int(row["num_hidden_pieces"]),
                "avg_jaccard":          float(row["avg_jaccard"]),
                "min_jaccard":          float(row["min_jaccard"]),
                "max_jaccard":          float(row["max_jaccard"]),
                "num_determinizations": int(row["num_determinizations"]),
            })

    if not rows:
        return "No data rows found in CSV."

    def mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def std(vals):
        if len(vals) < 2:
            return float("nan")
        m = mean(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    game_ids = set(r["game_id"] for r in rows)
    players  = sorted(set(r["player"] for r in rows))

    lines = [
        "=" * 72,
        "  IS-MCTS Belief-Accuracy Report",
        "=" * 72,
        f"  Source file : {csv_path}",
        f"  Total rows  : {len(rows)}",
        f"  Games       : {len(game_ids)}",
        f"  Players     : {players}",
        "",
        "Overall statistics",
        "-" * 40,
    ]

    avg_j_all = [r["avg_jaccard"]       for r in rows]
    min_j_all = [r["min_jaccard"]       for r in rows]
    max_j_all = [r["max_jaccard"]       for r in rows]
    hid_all   = [r["num_hidden_pieces"] for r in rows]

    for label, vals in [("avg_jaccard", avg_j_all), ("min_jaccard", min_j_all), ("max_jaccard", max_j_all)]:
        lines.append(f"  {label:<18}  mean={mean(vals):.4f}  std={std(vals):.4f}  "
                     f"[{min(vals):.4f}, {max(vals):.4f}]")
    lines.append(f"  {'num_hidden_pieces':<18}  mean={mean(hid_all):.2f}  max={max(hid_all)}")
    lines.append("")

    lines.append("By game phase  (move_num: <20=opening, 20-60=midgame, >60=endgame)")
    lines.append("-" * 72)
    lines.append(f"  {'Phase':<10}  {'N':>6}  {'Mean Avg-J':>11}  {'Std':>7}  "
                 f"{'Mean Min-J':>11}  {'Mean Max-J':>11}")
    lines.append("  " + "-" * 68)
    for ph in ["opening", "midgame", "endgame"]:
        sub = [r for r in rows if phase(r["move_num"]) == ph]
        if not sub:
            continue
        avgs = [r["avg_jaccard"] for r in sub]
        mins = [r["min_jaccard"] for r in sub]
        maxs = [r["max_jaccard"] for r in sub]
        lines.append(
            f"  {ph:<10}  {len(sub):>6}  "
            f"{mean(avgs):>11.4f}  "
            f"{std(avgs):>7.4f}  "
            f"{mean(mins):>11.4f}  "
            f"{mean(maxs):>11.4f}"
        )
    lines.append("")

    lines.append("By player")
    lines.append("-" * 50)
    for p in players:
        sub  = [r for r in rows if r["player"] == p]
        avgs = [r["avg_jaccard"] for r in sub]
        lines.append(f"  Player {p}:  N={len(sub)}  mean_avg_jaccard={mean(avgs):.4f}  std={std(avgs):.4f}")
    lines.append("")

    lines.append("Per-game summary (first 20 games shown)")
    lines.append("-" * 72)
    lines.append(f"  {'game_id':>18}  {'moves':>6}  {'mean_jaccard':>13}  "
                 f"{'min_jaccard':>11}  {'max_jaccard':>11}")
    lines.append("  " + "-" * 65)
    for gid in list(game_ids)[:20]:
        gsub = [r for r in rows if r["game_id"] == gid]
        avgs = [r["avg_jaccard"] for r in gsub]
        mins = [r["min_jaccard"] for r in gsub]
        maxs = [r["max_jaccard"] for r in gsub]
        lines.append(
            f"  {str(gid):>18}  {len(gsub):>6}  "
            f"{mean(avgs):>13.4f}  "
            f"{mean(mins):>11.4f}  "
            f"{mean(maxs):>11.4f}"
        )
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse IS-MCTS belief-accuracy CSV and print a summary table."
    )
    parser.add_argument(
        "csv",
        metavar="CSV_FILE",
        nargs="?",
        default="evaluation/results_grave/belief_accuracy.csv",
        help="Path to belief_accuracy.csv",
    )
    parser.add_argument(
        "--out",
        metavar="OUTPUT_FILE",
        default=None,
        help="Write summary to this file instead of stdout",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if HAS_PANDAS:
        report = analyse_pandas(csv_path)
    else:
        print("[WARN] pandas not available — using pure-Python analysis.", file=sys.stderr)
        report = analyse_pure(csv_path)

    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"Summary written to: {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
