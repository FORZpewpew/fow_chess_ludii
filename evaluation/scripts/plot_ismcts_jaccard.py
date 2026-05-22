#!/usr/bin/env python3
"""
Generate IS-MCTS Belief Jaccard figure by game phase.

Reads: fow_chess_ludii/evaluation/results_grave/ismcts_belief_accuracy.csv
Saves: fow_chess_ludii/evaluation/figures/ismcts_belief_jaccard.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")           # fow_chess_ludii/
CSV_PATH = os.path.join(ROOT, "evaluation", "results_grave", "ismcts_belief_accuracy.csv")
OUT_PATH  = os.path.join(ROOT, "evaluation", "figures",  "ismcts_belief_jaccard.png")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Assign game phase based on move number (half-moves / ply)
def phase(move_num):
    if move_num < 20:
        return "Opening\n(move < 20)"
    elif move_num <= 60:
        return "Midgame\n(move 20–60)"
    else:
        return "Endgame\n(move > 60)"

df["phase"] = df["move_num"].apply(phase)

# Phase ordering
PHASES = ["Opening\n(move < 20)", "Midgame\n(move 20–60)"]
# (Endgame has N=0 in this dataset)

phase_data = {p: df.loc[df["phase"] == p, "avg_jaccard"].values for p in PHASES}

means  = [phase_data[p].mean() if len(phase_data[p]) else np.nan for p in PHASES]
stds   = [phase_data[p].std()  if len(phase_data[p]) else np.nan for p in PHASES]
ns     = [len(phase_data[p]) for p in PHASES]

# ── summary statistics (for reference) ────────────────────────────────────────
overall_mean = df["avg_jaccard"].mean()
overall_std  = df["avg_jaccard"].std()
print(f"Overall  : N={len(df):4d}  mean={overall_mean:.4f}  std={overall_std:.4f}")
for p, m, s, n in zip(PHASES, means, stds, ns):
    label = p.replace("\n", " ")
    print(f"{label:30s}: N={n:4d}  mean={m:.4f}  std={s:.4f}")

# ── plot ───────────────────────────────────────────────────────────────────────
PHASE_LABELS  = ["Opening\n(move<20)", "Midgame\n(move 20–60)"]
PHASE_COLOURS = ["#4C72B0", "#DD8452"]   # blue, orange — seaborn default palette

fig, ax = plt.subplots(figsize=(6.5, 4.8))

x_pos = np.arange(len(PHASES))
bar_w = 0.45

bars = ax.bar(
    x_pos, means, bar_w,
    yerr=stds, capsize=5,
    color=PHASE_COLOURS,
    edgecolor="black", linewidth=0.8,
    error_kw=dict(elinewidth=1.2, ecolor="black", capthick=1.2),
    zorder=3,
)

# Overlay individual data-points (jittered) for each phase
rng = np.random.default_rng(42)
for i, p in enumerate(PHASES):
    vals = phase_data[p]
    if len(vals) == 0:
        continue
    jitter = rng.uniform(-0.18, 0.18, size=len(vals))
    ax.scatter(
        x_pos[i] + jitter, vals,
        s=3, alpha=0.25, color=PHASE_COLOURS[i], zorder=2,
    )

# Random-baseline band  (J ≈ 0.04–0.08)
ax.axhspan(0.04, 0.08, color="grey", alpha=0.15, zorder=1, label="Random-baseline band (J≈0.04–0.08)")

# Overall mean line
ax.axhline(overall_mean, color="black", linestyle="--", linewidth=1.2,
           label=f"Overall mean  J={overall_mean:.3f}")

# Annotate bars with mean ± std and N
for i, (m, s, n) in enumerate(zip(means, stds, ns)):
    if np.isnan(m):
        ax.text(x_pos[i], 0.01, "N/A", ha="center", va="bottom", fontsize=9, color="grey")
    else:
        ax.text(x_pos[i], m + s + 0.003,
                f"J={m:.3f}±{s:.3f}\nN={n:,}",
                ha="center", va="bottom", fontsize=8.5, linespacing=1.4)

ax.set_xticks(x_pos)
ax.set_xticklabels(PHASE_LABELS, fontsize=11)
ax.set_xlabel("Game Phase", fontsize=12)
ax.set_ylabel("Mean Jaccard Similarity (avg_jaccard)", fontsize=11)
ax.set_title(
    "IS-MCTS Determinization Quality by Game Phase\n"
    r"(Jaccard similarity between sampled belief and true hidden state)",
    fontsize=11, pad=10,
)
ax.set_ylim(0, max(means) + max(stds) + 0.035)
ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"\nFigure saved → {OUT_PATH}")
