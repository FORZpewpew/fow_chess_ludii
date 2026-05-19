#!/usr/bin/env python3
"""
run_all_analyses.py
===================
Master script that performs all four thesis analysis tasks:

  1. Think-time sensitivity analysis  → results_time_sensitivity/sensitivity_summary.csv
  2. Belief accuracy (Jaccard) analysis → results_grave/belief_analysis_summary.csv
  3. Training curve plot               → figures/training_curves.png
  4. Jaccard similarity plot           → figures/belief_jaccard.png
"""

import csv
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_DEPS = True
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with:  pip install pandas matplotlib numpy")
    sys.exit(1)

SCRIPT_DIR    = Path(__file__).resolve().parent
EVAL_DIR      = SCRIPT_DIR.parent
RESULTS_TS    = EVAL_DIR / "results_time_sensitivity"
RESULTS_GRAVE = EVAL_DIR / "results_grave"
FIGURES_DIR   = EVAL_DIR / "figures"
LOGS_DIR      = EVAL_DIR.parent / "ppo" / "logs"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TASK 1 — Think-time sensitivity analysis
# ---------------------------------------------------------------------------

def load_ts_results(time_tag: str):
    """Load all game results for a given time tag (e.g. '2s' or '5s')."""
    pattern = str(RESULTS_TS / f"{time_tag}_*.csv")
    results = []
    for fpath in glob.glob(pattern):
        fname = os.path.basename(fpath)
        try:
            with open(fpath) as f:
                for row in csv.DictReader(f):
                    # Support both column naming conventions
                    p1     = row.get("agent_p1") or row.get("p1_agent") or row.get("agent1", "")
                    p2     = row.get("agent_p2") or row.get("p2_agent") or row.get("agent2", "")
                    winner = row.get("winner") or row.get("result", "")
                    draw   = row.get("draw", "false").strip().lower() == "true"
                    results.append({
                        "p1":     p1.strip(),
                        "p2":     p2.strip(),
                        "winner": winner.strip(),
                        "draw":   draw,
                        "source": fname,
                    })
        except Exception as ex:
            print(f"  [WARN] Could not read {fpath}: {ex}")
    return results


def compute_ts_stats(results):
    """Return per-agent (games, wins) dicts."""
    wins  = defaultdict(float)
    games = defaultdict(int)
    for r in results:
        p1, p2, winner, draw = r["p1"], r["p2"], r["winner"], r["draw"]
        if not p1 or not p2:
            continue
        games[p1] += 1
        games[p2] += 1
        if draw:
            wins[p1] += 0.5
            wins[p2] += 0.5
        elif winner == p1:
            wins[p1] += 1.0
        elif winner == p2:
            wins[p2] += 1.0
        else:
            # winner string matches neither — count as draw
            wins[p1] += 0.5
            wins[p2] += 0.5
    return games, wins


def run_time_sensitivity():
    print("\n" + "=" * 60)
    print("TASK 1: Think-time Sensitivity Analysis")
    print("=" * 60)

    rows = []
    for time_tag in ["2s", "5s"]:
        results = load_ts_results(time_tag)
        if not results:
            print(f"  [WARN] No results found for time_tag={time_tag}")
            continue

        games, wins = compute_ts_stats(results)
        agents = sorted(games.keys())

        print(f"\n  Time control: {time_tag}/move  ({len(results)} game records)")
        print(f"  {'Agent':<35} {'Games':>6} {'Wins':>7} {'Win%':>7}")
        print("  " + "-" * 58)

        for agent in agents:
            g   = games[agent]
            w   = wins[agent]
            pct = 100.0 * w / g if g > 0 else 0.0
            print(f"  {agent:<35} {g:>6} {w:>7.1f} {pct:>6.1f}%")
            rows.append({
                "time_control": time_tag,
                "agent":        agent,
                "games":        g,
                "wins":         w,
                "win_pct":      round(pct, 2),
            })

    out_path = RESULTS_TS / "sensitivity_summary.csv"
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time_control", "agent", "games", "wins", "win_pct"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  ✓ Saved: {out_path}")
    else:
        print("  [WARN] No data to save.")

    return rows


# ---------------------------------------------------------------------------
# TASK 2 — Belief accuracy (Jaccard) analysis
# ---------------------------------------------------------------------------

def phase_label(ply: int) -> str:
    if ply < 20:
        return "opening"
    elif ply <= 60:
        return "midgame"
    return "endgame"


def run_belief_analysis():
    print("\n" + "=" * 60)
    print("TASK 2: Belief-State Jaccard Analysis")
    print("=" * 60)

    csv_path = RESULTS_GRAVE / "ppo_belief_accuracy.csv"
    if not csv_path.exists():
        print(f"  [ERROR] File not found: {csv_path}")
        return None

    # ppo_belief_accuracy.csv columns: game_id, player, ply, num_hidden_squares, jaccard, phase
    # Rename to match analysis conventions.
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from {csv_path.name}")
    print(f"  Columns: {list(df.columns)}")

    df = df.rename(columns={
        "ply":                "move_num",
        "num_hidden_squares": "num_hidden_pieces",
        "jaccard":            "avg_jaccard",
    })
    # CSV has a single jaccard value per row; mirror to min/max for compatibility
    df["min_jaccard"] = df["avg_jaccard"]
    df["max_jaccard"] = df["avg_jaccard"]
    df["phase_derived"] = df["move_num"].apply(phase_label)

    overall_mean = df["avg_jaccard"].mean()
    overall_std  = df["avg_jaccard"].std()

    print(f"\n  Total rows  : {len(df)}")
    print(f"  Games       : {df['game_id'].nunique()}")
    print(f"  Players     : {sorted(df['player'].unique())}")
    print(f"\n  Overall Jaccard:  mean={overall_mean:.4f}  std={overall_std:.4f}  "
          f"[{df['avg_jaccard'].min():.4f}, {df['avg_jaccard'].max():.4f}]")
    print(f"  num_hidden_pieces: mean={df['num_hidden_pieces'].mean():.2f}  "
          f"max={df['num_hidden_pieces'].max()}")

    phase_col = "phase" if "phase" in df.columns else "phase_derived"
    print(f"\n  By game phase (column '{phase_col}'):")
    print(f"  {'Phase':<10} {'N':>6}  {'Mean Jaccard':>13}  {'Std':>8}")
    print("  " + "-" * 45)

    phase_summary_rows = []
    for ph, sub in df.groupby(phase_col):
        mean_j = sub["avg_jaccard"].mean()
        std_j  = sub["avg_jaccard"].std()
        print(f"  {str(ph):<10} {len(sub):>6}  {mean_j:>13.4f}  {std_j:>8.4f}")
        phase_summary_rows.append({
            "phase":        ph,
            "n":            len(sub),
            "mean_jaccard": round(mean_j, 6),
            "std_jaccard":  round(std_j, 6),
            "min_jaccard":  round(sub["avg_jaccard"].min(), 6),
            "max_jaccard":  round(sub["avg_jaccard"].max(), 6),
        })

    print(f"\n  By player:")
    for p, sub in df.groupby("player"):
        print(f"    Player {p}: N={len(sub)}  mean={sub['avg_jaccard'].mean():.4f}  "
              f"std={sub['avg_jaccard'].std():.4f}")

    out_path = RESULTS_GRAVE / "belief_analysis_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "n", "mean_jaccard", "std_jaccard",
                                               "min_jaccard", "max_jaccard"])
        writer.writeheader()
        writer.writerows(phase_summary_rows)
    print(f"\n  ✓ Saved: {out_path}")

    return df


# ---------------------------------------------------------------------------
# TASK 3 — Training curve plots
# ---------------------------------------------------------------------------

def parse_ppo_log(log_path: Path):
    """Parse a PPO-LSTM training log; return list of dicts."""
    records = []
    pat = re.compile(
        r"\[PPO-LSTM\] Update\s+(\d+)/\d+\s+"
        r"win_rate=(\S+)\s+mean_ep_len=(\S+)\s+pool_size=(\d+)\s+"
        r"p_loss=(\S+)\s+v_loss=(\S+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                records.append({
                    "update":      int(m.group(1)),
                    "win_rate":    float(m.group(2)),
                    "mean_ep_len": float(m.group(3)),
                    "pool_size":   int(m.group(4)),
                    "p_loss":      float(m.group(5)),
                    "v_loss":      float(m.group(6)),
                })
    return records


def smooth(values, window=10):
    """Simple rolling mean."""
    out = []
    for i, v in enumerate(values):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i+1]) / (i - lo + 1))
    return out


def run_training_curves():
    print("\n" + "=" * 60)
    print("TASK 3: Training Curve Plots")
    print("=" * 60)

    ppo_log = LOGS_DIR / "train_ppo_lstm_v4.log"
    pre_log = LOGS_DIR / "pretrain_v4.log"

    if not ppo_log.exists():
        print(f"  [ERROR] Not found: {ppo_log}")
        return

    ppo_records = parse_ppo_log(ppo_log)
    print(f"  PPO-LSTM v4 updates parsed: {len(ppo_records)}")

    pre_records = []
    if pre_log.exists():
        pre_records = parse_ppo_log(pre_log)
        print(f"  Pretrained v4 PPO updates parsed: {len(pre_records)}")

    df_ppo = pd.DataFrame(ppo_records)
    df_pre = pd.DataFrame(pre_records) if pre_records else None

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle("PPO-LSTM v4 Training Curves (Fog-of-War Kriegspiel)", fontsize=14)

    # Subplot 1: Win Rate
    ax1 = axes[0]
    wr_raw    = df_ppo["win_rate"].tolist()
    wr_smooth = smooth(wr_raw, window=15)
    ax1.plot(df_ppo["update"], wr_raw,    alpha=0.3, color="#4C72B0", label="Win rate (raw)")
    ax1.plot(df_ppo["update"], wr_smooth, color="#4C72B0", linewidth=2, label="Win rate (smooth, w=15)")

    if df_pre is not None and len(df_pre) > 0:
        pr_raw    = df_pre["win_rate"].tolist()
        pr_smooth = smooth(pr_raw, window=15)
        ax1.plot(df_pre["update"], pr_raw,    alpha=0.3, color="#DD8452", label="Pretrained v4 (raw)")
        ax1.plot(df_pre["update"], pr_smooth, color="#DD8452", linewidth=2, label="Pretrained v4 (smooth)")

    pool_changes = df_ppo[df_ppo["pool_size"].diff() > 0]["update"].tolist()
    for u in pool_changes:
        ax1.axvline(x=u, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    ax1.axhline(y=0.5, color="black", linestyle=":", linewidth=1, alpha=0.6, label="50% baseline")
    ax1.set_ylabel("Win Rate", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_title("Self-play Win Rate over Training Updates")
    ax1.grid(True, alpha=0.3)

    for _, row in df_ppo[df_ppo["update"].isin([25, 50, 75, 100, 125, 150, 175, 200])].iterrows():
        ax1.annotate(
            f"p={int(row['pool_size'])}",
            xy=(row["update"], 0.02),
            fontsize=6, color="gray", ha="center"
        )

    # Subplot 2: Value and policy loss
    ax2      = axes[1]
    vl_raw   = df_ppo["v_loss"].tolist()
    ax2.plot(df_ppo["update"], vl_raw,              alpha=0.3, color="#55A868", label="v_loss (raw)")
    ax2.plot(df_ppo["update"], smooth(vl_raw, 15),  color="#55A868", linewidth=2, label="v_loss (smooth, w=15)")

    pl_raw   = df_ppo["p_loss"].tolist()
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df_ppo["update"], pl_raw,             alpha=0.2, color="#C44E52", label="p_loss (raw)")
    ax2_twin.plot(df_ppo["update"], smooth(pl_raw, 15), color="#C44E52", linewidth=2, label="p_loss (smooth)")
    ax2_twin.set_ylabel("Policy Loss", color="#C44E52", fontsize=10)
    ax2_twin.tick_params(axis="y", labelcolor="#C44E52")

    ax2.set_xlabel("Training Update", fontsize=11)
    ax2.set_ylabel("Value Loss", color="#55A868", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#55A868")
    ax2.set_title("Value Loss and Policy Loss over Training")
    ax2.grid(True, alpha=0.3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    for u in pool_changes:
        axes[1].axvline(x=u, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    out_path = FIGURES_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")

    if len(df_ppo) > 0:
        last_50 = df_ppo.tail(50)
        print(f"\n  PPO-LSTM v4 final 50 updates avg win_rate: {last_50['win_rate'].mean():.3f}")
        print(f"  PPO-LSTM v4 max win_rate: {df_ppo['win_rate'].max():.3f} "
              f"(update {df_ppo.loc[df_ppo['win_rate'].idxmax(), 'update']})")

    return df_ppo, df_pre


# ---------------------------------------------------------------------------
# TASK 4 — Jaccard similarity plot by game phase
# ---------------------------------------------------------------------------

def run_jaccard_plot(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("TASK 4: Jaccard Similarity Plot")
    print("=" * 60)

    if df is None:
        print("  [ERROR] No belief data available (Task 2 may have failed).")
        return

    phase_col = "phase" if "phase" in df.columns else "phase_derived"

    raw_phases = df[phase_col].unique()
    order_map  = {
        "early": 0, "mid": 1, "late": 2,
        "opening": 0, "midgame": 1, "endgame": 2,
    }
    phases_ordered = sorted(raw_phases, key=lambda p: order_map.get(str(p).lower(), 99))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("PPO-LSTM Belief-State Accuracy (Jaccard Similarity)\n"
                 "Fog-of-War Kriegspiel", fontsize=13)

    phase_data   = [df.loc[df[phase_col] == ph, "avg_jaccard"].values for ph in phases_ordered]
    phase_means  = [d.mean() for d in phase_data]
    phase_stds   = [d.std()  for d in phase_data]
    phase_labels = [str(p).capitalize() for p in phases_ordered]
    colors       = ["#4C72B0", "#55A868", "#C44E52"][:len(phases_ordered)]

    ax = axes[0]
    bp = ax.boxplot(phase_data, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(phase_labels, fontsize=11)
    ax.set_ylabel("Jaccard Similarity", fontsize=11)
    ax.set_xlabel("Game Phase", fontsize=11)
    ax.set_title("Distribution by Game Phase")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, axis="y", alpha=0.3)

    for i, (m, _) in enumerate(zip(phase_means, phase_stds), start=1):
        ax.text(i, m + 0.02, f"μ={m:.3f}", ha="center", fontsize=9, color="black",
                fontweight="bold")

    ax2 = axes[1]
    scatter_sample = df.sample(min(2000, len(df)), random_state=42) if len(df) > 2000 else df

    color_map = {
        "early": "#4C72B0", "mid": "#55A868", "late": "#C44E52",
        "opening": "#4C72B0", "midgame": "#55A868", "endgame": "#C44E52",
    }
    point_colors = [color_map.get(str(ph).lower(), "#999999")
                    for ph in scatter_sample[phase_col]]
    ax2.scatter(scatter_sample["move_num"], scatter_sample["avg_jaccard"],
                c=point_colors, alpha=0.25, s=8, linewidths=0)

    # Binned mean trend line
    df_sorted  = df.sort_values("move_num")
    bin_size   = 4
    max_ply    = int(df_sorted["move_num"].max())
    bin_means, bin_centers = [], []
    for lo in range(0, max_ply + bin_size, bin_size):
        sub = df_sorted[(df_sorted["move_num"] >= lo) & (df_sorted["move_num"] < lo + bin_size)]
        if len(sub) >= 3:
            bin_means.append(sub["avg_jaccard"].mean())
            bin_centers.append(lo + bin_size / 2)
    ax2.plot(bin_centers, bin_means, color="black", linewidth=2.5, label="Binned mean (w=4 plies)")

    # Phase boundary lines
    existing_phases = [str(p).lower() for p in df[phase_col].unique()]
    drawn = set()
    for boundary, a_phase, b_phase in [(20, "early", "mid"), (20, "opening", "midgame"),
                                        (60, "mid", "late"),  (60, "midgame", "endgame")]:
        if boundary not in drawn and (a_phase in existing_phases or b_phase in existing_phases):
            ax2.axvline(x=boundary, color="gray", linestyle="--", alpha=0.6)
            drawn.add(boundary)

    legend_patches = [
        mpatches.Patch(color=color_map.get(str(ph).lower(), "#999"), label=str(ph).capitalize())
        for ph in phases_ordered
    ]
    legend_patches.append(plt.Line2D([0], [0], color="black", linewidth=2.5, label="Binned mean"))
    ax2.legend(handles=legend_patches, fontsize=9)

    ax2.set_xlabel("Ply (half-move number)", fontsize=11)
    ax2.set_ylabel("Jaccard Similarity", fontsize=11)
    ax2.set_title("Belief Accuracy vs. Game Progression")
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "belief_jaccard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")

    print(f"\n  Phase breakdown:")
    for ph, data in zip(phases_ordered, phase_data):
        print(f"    {str(ph):8s}  N={len(data):5d}  mean={data.mean():.4f}  "
              f"std={data.std():.4f}  min={data.min():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Fog-of-War Chess AI — Thesis Analysis Runner")
    print("=" * 60)

    run_time_sensitivity()
    belief_df = run_belief_analysis()
    run_training_curves()
    run_jaccard_plot(belief_df)

    print("\n" + "=" * 60)
    print("  All tasks complete.")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
