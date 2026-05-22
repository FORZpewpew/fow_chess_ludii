#!/usr/bin/env python3
"""
analyze_belief_accuracy.py — Belief-accuracy analysis for IS-MCTS and PPO probe

Supports two CSV formats:

1. IS-MCTS belief log (written by ISMCTSAgent with -Dfow.belief.log=true):
   Schema:
     game_id, move_num, player, num_hidden_pieces,
     avg_jaccard, min_jaccard, max_jaccard, num_determinizations

2. PPO belief probe CSV (written by ppo/eval_belief_probe.py):
   Schema (flexible — detected automatically):
     Any CSV that contains a 'jaccard' column (single-value per row).
     Optional columns: game_id, move_num, player, num_hidden_squares.

Usage
-----
    # IS-MCTS only
    python3 evaluation/scripts/analyze_belief_accuracy.py \\
        evaluation/results_grave/ismcts_belief_accuracy.csv

    # IS-MCTS + PPO probe comparison
    python3 evaluation/scripts/analyze_belief_accuracy.py \\
        evaluation/results_grave/ismcts_belief_accuracy.csv \\
        --ppo-probe evaluation/results_grave/ppo_belief_probe.csv

    # Write output to file
    python3 evaluation/scripts/analyze_belief_accuracy.py \\
        evaluation/results_grave/ismcts_belief_accuracy.csv \\
        --out report.txt

Options
-------
    CSV_FILE          Path to IS-MCTS belief_accuracy.csv (or ismcts_belief_accuracy.csv)
    --ppo-probe FILE  Optional: path to PPO probe Jaccard CSV for comparison
    --out FILE        Write summary to this file instead of stdout
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as _pd
    HAS_PANDAS = True
except ImportError:
    _pd = None          # type: ignore[assignment]
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Phase classifier
# ---------------------------------------------------------------------------

def phase(move_num: int) -> str:
    """Classify a ply number into opening / midgame / endgame."""
    if move_num < 20:
        return "opening"
    elif move_num <= 60:
        return "midgame"
    return "endgame"


# ---------------------------------------------------------------------------
# CSV format detection
# ---------------------------------------------------------------------------

def detect_format(csv_path: Path) -> str:
    """
    Return 'ismcts' if the file matches the ISMCTSAgent schema
    (has avg_jaccard / min_jaccard / max_jaccard columns),
    'ppo_probe' if it has a single 'jaccard' column,
    or 'unknown'.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = [c.strip().lower() for c in next(reader)]
        except StopIteration:
            return "unknown"

    if "avg_jaccard" in header and "min_jaccard" in header:
        return "ismcts"
    if "jaccard" in header:
        return "ppo_probe"
    return "unknown"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ismcts_rows(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
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
    return rows


def load_ppo_probe_rows(csv_path: Path) -> List[Dict]:
    """
    Load a PPO probe CSV that has at least a 'jaccard' column.
    Handles two column naming conventions:
      • move_num  OR  ply       → normalised to move_num
      • num_hidden_squares  OR  num_hidden_pieces  → normalised to num_hidden_squares
    Optional columns are read when present.
    Returns a list of dicts with normalised keys.
    """
    rows: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            # Normalise column names to lowercase
            norm = {k.strip().lower(): v.strip() for k, v in raw.items()}

            # move_num: accept 'move_num' or 'ply'
            move_num_raw = norm.get("move_num", norm.get("ply", None))
            move_num = int(move_num_raw) if move_num_raw is not None else -1

            # num_hidden: accept several spellings
            hidden_raw = (norm.get("num_hidden_squares")
                          or norm.get("num_hidden_pieces")
                          or norm.get("num_hidden")
                          or None)
            num_hidden = int(hidden_raw) if hidden_raw is not None else -1

            rows.append({
                "game_id":            norm.get("game_id", "?"),
                "move_num":           move_num,
                "player":             int(norm["player"]) if "player" in norm else -1,
                "num_hidden_squares": num_hidden,
                "jaccard":            float(norm["jaccard"]),
                # preserve original phase label if present
                "phase_label":        norm.get("phase", ""),
            })
    return rows


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    m = _mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


# ---------------------------------------------------------------------------
# IS-MCTS report
# ---------------------------------------------------------------------------

def _ismcts_report_pure(rows: List[Dict], csv_path: Path) -> str:
    if not rows:
        return "No data rows found in IS-MCTS CSV."

    game_ids = set(r["game_id"] for r in rows)
    players  = sorted(set(r["player"] for r in rows))

    avg_j_all = [r["avg_jaccard"]       for r in rows]
    min_j_all = [r["min_jaccard"]       for r in rows]
    max_j_all = [r["max_jaccard"]       for r in rows]
    hid_all   = [r["num_hidden_pieces"] for r in rows]

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

    for label, vals in [("avg_jaccard", avg_j_all),
                         ("min_jaccard", min_j_all),
                         ("max_jaccard", max_j_all)]:
        lines.append(
            f"  {label:<18}  mean={_mean(vals):.4f}  std={_std(vals):.4f}  "
            f"[{min(vals):.4f}, {max(vals):.4f}]"
        )
    lines.append(
        f"  {'num_hidden_pieces':<18}  mean={_mean(hid_all):.2f}  max={max(hid_all)}"
    )
    lines.append("")

    lines.append("By game phase  (move_num: <20=opening, 20-60=midgame, >60=endgame)")
    lines.append("-" * 72)
    lines.append(
        f"  {'Phase':<10}  {'N':>6}  {'Mean Avg-J':>11}  {'Std':>7}  "
        f"{'Mean Min-J':>11}  {'Mean Max-J':>11}"
    )
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
            f"{_mean(avgs):>11.4f}  "
            f"{_std(avgs):>7.4f}  "
            f"{_mean(mins):>11.4f}  "
            f"{_mean(maxs):>11.4f}"
        )
    lines.append("")

    lines.append("By player")
    lines.append("-" * 50)
    for p in players:
        sub  = [r for r in rows if r["player"] == p]
        avgs = [r["avg_jaccard"] for r in sub]
        lines.append(
            f"  Player {p}:  N={len(sub)}  "
            f"mean_avg_jaccard={_mean(avgs):.4f}  std={_std(avgs):.4f}"
        )
    lines.append("")

    lines.append("Per-game summary (first 20 games shown)")
    lines.append("-" * 72)
    lines.append(
        f"  {'game_id':>18}  {'moves':>6}  {'mean_jaccard':>13}  "
        f"{'min_jaccard':>11}  {'max_jaccard':>11}"
    )
    lines.append("  " + "-" * 65)
    for gid in list(game_ids)[:20]:
        gsub = [r for r in rows if r["game_id"] == gid]
        avgs = [r["avg_jaccard"] for r in gsub]
        mins = [r["min_jaccard"] for r in gsub]
        maxs = [r["max_jaccard"] for r in gsub]
        lines.append(
            f"  {str(gid):>18}  {len(gsub):>6}  "
            f"{_mean(avgs):>13.4f}  "
            f"{_mean(mins):>11.4f}  "
            f"{_mean(maxs):>11.4f}"
        )
    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def _ismcts_report_pandas(csv_path: Path) -> str:
    assert _pd is not None, "pandas required"
    df = _pd.read_csv(csv_path)

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
        m, s, lo, hi = (df[col].mean(), df[col].std(),
                        df[col].min(), df[col].max())
        lines.append(
            f"  {col:<18}  mean={m:.4f}  std={s:.4f}  [{lo:.4f}, {hi:.4f}]"
        )
    lines.append(
        f"  {'num_hidden_pieces':<18}  "
        f"mean={df['num_hidden_pieces'].mean():.2f}  "
        f"max={df['num_hidden_pieces'].max()}"
    )
    lines.append("")

    lines.append("By game phase  (move_num: <20=opening, 20-60=midgame, >60=endgame)")
    lines.append("-" * 72)
    lines.append(
        f"  {'Phase':<10}  {'N':>6}  {'Mean Avg-J':>11}  {'Std':>7}  "
        f"{'Mean Min-J':>11}  {'Mean Max-J':>11}"
    )
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
        lines.append(
            f"  Player {p}:  N={len(sub)}  "
            f"mean_avg_jaccard={sub['avg_jaccard'].mean():.4f}  "
            f"std={sub['avg_jaccard'].std():.4f}"
        )
    lines.append("")

    lines.append("Per-game summary (first 20 games shown)")
    lines.append("-" * 72)
    lines.append(
        f"  {'game_id':>18}  {'moves':>6}  {'mean_jaccard':>13}  "
        f"{'min_jaccard':>11}  {'max_jaccard':>11}"
    )
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


# ---------------------------------------------------------------------------
# PPO probe report
# ---------------------------------------------------------------------------

def _ppo_report_pure(rows: List[Dict], csv_path: Path) -> str:
    if not rows:
        return "No data rows found in PPO probe CSV."

    jaccards = [r["jaccard"] for r in rows]
    game_ids = set(r["game_id"] for r in rows)

    lines = [
        "=" * 72,
        "  PPO Belief-Probe Report",
        "=" * 72,
        f"  Source file : {csv_path}",
        f"  Total rows  : {len(rows)}",
        f"  Games       : {len(game_ids)}",
        "",
        "Overall Jaccard statistics",
        "-" * 40,
        f"  mean  = {_mean(jaccards):.4f}",
        f"  std   = {_std(jaccards):.4f}",
        f"  min   = {min(jaccards):.4f}",
        f"  max   = {max(jaccards):.4f}",
        "",
    ]

    # Phase breakdown (only if move_num is available)
    has_move_num = any(r["move_num"] >= 0 for r in rows)
    if has_move_num:
        lines.append("By game phase")
        lines.append("-" * 50)
        for ph in ["opening", "midgame", "endgame"]:
            sub = [r for r in rows if r["move_num"] >= 0 and phase(r["move_num"]) == ph]
            if not sub:
                continue
            j_vals = [r["jaccard"] for r in sub]
            lines.append(
                f"  {ph:<10}  N={len(sub):>5}  "
                f"mean={_mean(j_vals):.4f}  std={_std(j_vals):.4f}"
            )
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def _ppo_report_pandas(csv_path: Path) -> str:
    assert _pd is not None, "pandas required"
    df = _pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "jaccard" not in df.columns:
        raise ValueError("PPO probe CSV has no 'jaccard' column.")

    # Normalise move_num: accept 'move_num' or 'ply'
    if "move_num" not in df.columns and "ply" in df.columns:
        df = df.rename(columns={"ply": "move_num"})

    lines = [
        "=" * 72,
        "  PPO Belief-Probe Report",
        "=" * 72,
        f"  Source file : {csv_path}",
        f"  Total rows  : {len(df)}",
    ]

    if "game_id" in df.columns:
        lines.append(f"  Games       : {df['game_id'].nunique()}")

    lines += [
        "",
        "Overall Jaccard statistics",
        "-" * 40,
        f"  mean  = {df['jaccard'].mean():.4f}",
        f"  std   = {df['jaccard'].std():.4f}",
        f"  min   = {df['jaccard'].min():.4f}",
        f"  max   = {df['jaccard'].max():.4f}",
        "",
    ]

    if "move_num" in df.columns:
        df["phase_computed"] = df["move_num"].apply(phase)
        lines.append("By game phase  (move_num: <20=opening, 20-60=midgame, >60=endgame)")
        lines.append("-" * 72)
        lines.append(
            f"  {'Phase':<10}  {'N':>6}  {'Mean Jaccard':>13}  {'Std':>7}  "
            f"{'Min':>7}  {'Max':>7}"
        )
        lines.append("  " + "-" * 60)
        for ph in ["opening", "midgame", "endgame"]:
            sub = df[df["phase_computed"] == ph]
            if sub.empty:
                continue
            lines.append(
                f"  {ph:<10}  {len(sub):>6}  "
                f"{sub['jaccard'].mean():>13.4f}  "
                f"{sub['jaccard'].std():>7.4f}  "
                f"{sub['jaccard'].min():>7.4f}  "
                f"{sub['jaccard'].max():>7.4f}"
            )
        lines.append("")

    # If CSV already has a 'phase' column (e.g. 'early'/'mid'/'late'), show that too
    if "phase" in df.columns:
        lines.append("By original phase label  (from CSV)")
        lines.append("-" * 50)
        for ph, sub in df.groupby("phase"):
            lines.append(
                f"  {str(ph):<12}  N={len(sub):>5}  "
                f"mean={sub['jaccard'].mean():.4f}  std={sub['jaccard'].std():.4f}"
            )
        lines.append("")

    if "player" in df.columns:
        lines.append("By player")
        lines.append("-" * 50)
        for p, sub in df.groupby("player"):
            lines.append(
                f"  Player {p}:  N={len(sub):>5}  "
                f"mean={sub['jaccard'].mean():.4f}  std={sub['jaccard'].std():.4f}"
            )
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison section
# ---------------------------------------------------------------------------

def _comparison_section(ismcts_rows: List[Dict], ppo_rows: List[Dict]) -> str:
    ismcts_avg = [r["avg_jaccard"] for r in ismcts_rows]
    ppo_j      = [r["jaccard"]     for r in ppo_rows]

    lines = [
        "",
        "=" * 72,
        "  IS-MCTS vs PPO Probe — Belief Accuracy Comparison",
        "=" * 72,
        f"  {'Agent':<20}  {'N':>6}  {'Mean Jaccard':>13}  {'Std':>7}  "
        f"{'Min':>7}  {'Max':>7}",
        "  " + "-" * 65,
    ]

    for label, vals in [("IS-MCTS (avg_jaccard)", ismcts_avg),
                         ("PPO probe (jaccard)",   ppo_j)]:
        if not vals:
            continue
        lines.append(
            f"  {label:<20}  {len(vals):>6}  "
            f"{_mean(vals):>13.4f}  "
            f"{_std(vals):>7.4f}  "
            f"{min(vals):>7.4f}  "
            f"{max(vals):>7.4f}"
        )

    lines += [
        "",
        "  Note: IS-MCTS avg_jaccard is the mean Jaccard over all",
        "  determinizations for a single move decision.  The PPO probe",
        "  jaccard is computed per hidden-piece prediction step.  Both",
        "  measure how well the agent's internal belief matches the true",
        "  hidden board state, but are not directly comparable.",
        "",
        "=" * 72,
    ]
    return "\n".join(lines)


def _comparison_section_pandas(ismcts_path: Path, ppo_path: Path) -> str:
    assert _pd is not None, "pandas required"
    df_i = _pd.read_csv(ismcts_path)
    df_p = _pd.read_csv(ppo_path)
    df_p.columns = [c.strip().lower() for c in df_p.columns]

    ismcts_avg = df_i["avg_jaccard"].dropna().tolist()
    ppo_j      = df_p["jaccard"].dropna().tolist()

    lines = [
        "",
        "=" * 72,
        "  IS-MCTS vs PPO Probe — Belief Accuracy Comparison",
        "=" * 72,
        f"  {'Agent':<25}  {'N':>6}  {'Mean Jaccard':>13}  {'Std':>7}  "
        f"{'Min':>7}  {'Max':>7}",
        "  " + "-" * 67,
    ]

    for label, vals in [("IS-MCTS (avg_jaccard)", ismcts_avg),
                         ("PPO probe (jaccard)",   ppo_j)]:
        if not vals:
            continue
        import statistics
        lines.append(
            f"  {label:<25}  {len(vals):>6}  "
            f"{sum(vals)/len(vals):>13.4f}  "
            f"{statistics.stdev(vals) if len(vals) > 1 else float('nan'):>7.4f}  "
            f"{min(vals):>7.4f}  "
            f"{max(vals):>7.4f}"
        )

    lines += [
        "",
        "  Note: IS-MCTS avg_jaccard is the mean Jaccard over all",
        "  determinizations for a single move decision.  The PPO probe",
        "  jaccard is computed per hidden-piece prediction step.  Both",
        "  measure how well the agent's internal belief matches the true",
        "  hidden board state, but are not directly comparable.",
        "",
        "=" * 72,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def analyse(ismcts_path: Path,
            ppo_probe_path: Optional[Path] = None) -> str:
    fmt = detect_format(ismcts_path)
    if fmt == "unknown":
        # Try IS-MCTS format anyway (old header variants)
        fmt = "ismcts"

    parts = []

    # --- IS-MCTS section ---
    if fmt == "ismcts":
        if HAS_PANDAS:
            try:
                parts.append(_ismcts_report_pandas(ismcts_path))
            except Exception:
                rows = load_ismcts_rows(ismcts_path)
                parts.append(_ismcts_report_pure(rows, ismcts_path))
        else:
            rows = load_ismcts_rows(ismcts_path)
            parts.append(_ismcts_report_pure(rows, ismcts_path))

    elif fmt == "ppo_probe":
        # Called with a PPO probe file as the primary argument
        if HAS_PANDAS:
            parts.append(_ppo_report_pandas(ismcts_path))
        else:
            rows = load_ppo_probe_rows(ismcts_path)
            parts.append(_ppo_report_pure(rows, ismcts_path))

    # --- PPO probe section (optional second file) ---
    if ppo_probe_path is not None:
        if HAS_PANDAS:
            try:
                parts.append(_ppo_report_pandas(ppo_probe_path))
            except Exception:
                ppo_rows = load_ppo_probe_rows(ppo_probe_path)
                parts.append(_ppo_report_pure(ppo_rows, ppo_probe_path))
        else:
            ppo_rows = load_ppo_probe_rows(ppo_probe_path)
            parts.append(_ppo_report_pure(ppo_rows, ppo_probe_path))

        # Comparison table
        if fmt == "ismcts":
            if HAS_PANDAS:
                parts.append(_comparison_section_pandas(ismcts_path, ppo_probe_path))
            else:
                ismcts_rows = load_ismcts_rows(ismcts_path)
                ppo_rows    = load_ppo_probe_rows(ppo_probe_path)
                parts.append(_comparison_section(ismcts_rows, ppo_rows))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse IS-MCTS belief-accuracy CSV and (optionally) compare "
            "with a PPO belief-probe CSV."
        )
    )
    parser.add_argument(
        "csv",
        metavar="CSV_FILE",
        nargs="?",
        default="evaluation/results_grave/ismcts_belief_accuracy.csv",
        help="Path to IS-MCTS belief_accuracy CSV "
             "(or ismcts_belief_accuracy.csv). "
             "Also accepts a PPO probe CSV directly.",
    )
    parser.add_argument(
        "--ppo-probe",
        metavar="PPO_PROBE_CSV",
        default=None,
        help="Optional PPO probe Jaccard CSV for comparison table.",
    )
    parser.add_argument(
        "--out",
        metavar="OUTPUT_FILE",
        default=None,
        help="Write summary to this file instead of stdout.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    ppo_path: Optional[Path] = None
    if args.ppo_probe:
        ppo_path = Path(args.ppo_probe)
        if not ppo_path.exists():
            print(f"ERROR: PPO probe file not found: {ppo_path}", file=sys.stderr)
            sys.exit(1)

    if not HAS_PANDAS:
        print("[WARN] pandas not available — using pure-Python analysis.",
              file=sys.stderr)

    report = analyse(csv_path, ppo_path)

    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"Summary written to: {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
