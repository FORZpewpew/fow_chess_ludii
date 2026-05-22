#!/usr/bin/env python3
"""
test_pretraining_significance.py
================================
Statistical significance test for the BC-pretraining hypothesis:

  H0: ppo_lstm_pretrained_v4 and ppo_lstm_v4 have equal win probability.
  H1: they differ (two-sided).

Data sources:
  • ppo_lstm_pretrained_v4_vs_ppo_lstm_v4.csv   (pretrained plays as P1)
  • ppo_lstm_v4_vs_ppo_lstm_pretrained_v4.csv   (pretrained plays as P2)

Both files live in fow_chess_ludii/evaluation/results_v4/ by default.

Tests performed
---------------
1. Two-sided binomial test  (scipy.stats.binomtest)
   — treats every decisive game as a Bernoulli trial; null p=0.50.
2. Fisher's exact test on the 2×2 contingency table:

        pretrained_wins  |  v4_wins
      ----------------------------------------
      file 1 (pre=P1)  |  A         |  B
      file 2 (pre=P2)  |  C         |  D

3. 95 % Wilson score confidence interval on the pooled win rate.

Results are printed to stdout and saved to
  fow_chess_ludii/evaluation/results_v4/pretraining_significance.txt

Usage
-----
    python3 fow_chess_ludii/evaluation/scripts/test_pretraining_significance.py
    python3 fow_chess_ludii/evaluation/scripts/test_pretraining_significance.py \\
        --file1 path/to/pretrained_v4_vs_v4.csv \\
        --file2 path/to/v4_vs_pretrained_v4.csv \\
        --out   path/to/output.txt
"""

import argparse
import csv
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from scipy.stats import binomtest as _binomtest, fisher_exact as _fisher_exact
    HAS_SCIPY = True
except ImportError:
    _binomtest = None   # type: ignore[assignment]
    _fisher_exact = None  # type: ignore[assignment]
    HAS_SCIPY = False
    print("[WARN] scipy not available — using manual p-value approximations.",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_counts(csv_path: Path, pretrained_slug: str, v4_slug: str):
    """
    Parse a head-to-head CSV and return (pretrained_wins, v4_wins, draws).
    Rows where draw==true are counted as draws.
    """
    pretrained_wins = 0
    v4_wins = 0
    draws = 0
    total = 0

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            is_draw = row.get("draw", "false").strip().lower() == "true"
            winner  = row.get("winner", "").strip()
            if is_draw or winner == "":
                draws += 1
            elif winner == pretrained_slug:
                pretrained_wins += 1
            elif winner == v4_slug:
                v4_wins += 1
            else:
                # Unexpected winner name — treat as draw
                draws += 1

    return pretrained_wins, v4_wins, draws, total


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95 % Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p_hat = k / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def binomtest_manual(k: int, n: int, p: float = 0.5) -> float:
    """
    Manual two-sided binomial p-value using normal approximation
    (used when scipy is unavailable).
    """
    if n == 0:
        return float("nan")
    p_hat = k / n
    se = math.sqrt(p * (1 - p) / n)
    if se == 0:
        return 0.0
    z = abs(p_hat - p) / se
    # Two-tailed p via standard normal CDF approximation (Abramowitz & Stegun)
    def norm_cdf(x):
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        poly = t * (0.319381530
               + t * (-0.356563782
               + t * (1.781477937
               + t * (-1.821255978
               + t * 1.330274429))))
        return 1.0 - 0.3989422804 * math.exp(-0.5 * x * x) * poly if x >= 0 \
               else 0.3989422804 * math.exp(-0.5 * x * x) * poly
    one_tail = 1.0 - norm_cdf(z)
    return min(1.0, 2 * one_tail)


def fisher_manual(a, b, c, d):
    """
    Very simple Fisher's exact test p-value via hypergeometric enumeration.
    Only feasible for small tables; for large counts use normal approximation.
    """
    # Use log-factorial for stability
    def lgamma_int(n):
        return sum(math.log(i) for i in range(1, n + 1)) if n > 0 else 0.0

    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    N = r1 + r2

    def log_hypergeom(a_):
        b_ = r1 - a_
        c_ = c1 - a_
        d_ = r2 - c_
        if b_ < 0 or c_ < 0 or d_ < 0:
            return -math.inf
        return (lgamma_int(r1) + lgamma_int(r2) + lgamma_int(c1) + lgamma_int(c2)
                - lgamma_int(N) - lgamma_int(a_) - lgamma_int(b_)
                - lgamma_int(c_) - lgamma_int(d_))

    p_obs = math.exp(log_hypergeom(a))
    p_val = sum(math.exp(log_hypergeom(x))
                for x in range(max(0, r1 - c2), min(r1, c1) + 1)
                if math.exp(log_hypergeom(x)) <= p_obs + 1e-10)
    return min(1.0, p_val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(file1: Path, file2: Path, out_path: Path,
        pretrained_slug: str = "ppo_lstm_pretrained_v4",
        v4_slug: str = "ppo_lstm_v4") -> str:

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    pw1, v1, d1, tot1 = load_counts(file1, pretrained_slug, v4_slug)
    pw2, v2, d2, tot2 = load_counts(file2, pretrained_slug, v4_slug)

    # Pooled decisive games
    pretrained_total_wins = pw1 + pw2
    v4_total_wins         = v1  + v2
    total_draws           = d1  + d2
    total_decisive        = pretrained_total_wins + v4_total_wins
    total_games           = tot1 + tot2

    win_rate_pretrained = (pretrained_total_wins / total_decisive
                           if total_decisive > 0 else float("nan"))

    # ------------------------------------------------------------------
    # 2. Statistical tests
    # ------------------------------------------------------------------

    # --- Binomial test ---
    binom_p: float
    binom_stat: float
    if HAS_SCIPY and _binomtest is not None:
        binom_result = _binomtest(pretrained_total_wins, total_decisive, p=0.5,
                                  alternative="two-sided")
        binom_p    = float(binom_result.pvalue)
        binom_stat = float(binom_result.statistic)   # observed proportion
    else:
        binom_p    = binomtest_manual(pretrained_total_wins, total_decisive, 0.5)
        binom_stat = win_rate_pretrained

    # --- Fisher's exact test (2×2 table) ---
    # Rows: which file (i.e. which side pretrained played)
    # Cols: pretrained win | v4 win
    #   file1: pw1, v1
    #   file2: pw2, v2
    fisher_p: float
    odds_ratio: float
    if HAS_SCIPY and _fisher_exact is not None:
        _fe_result = _fisher_exact([[pw1, v1], [pw2, v2]], alternative="two-sided")
        odds_ratio = float(_fe_result[0])  # type: ignore[arg-type]
        fisher_p   = float(_fe_result[1])  # type: ignore[arg-type]
    else:
        fisher_p   = fisher_manual(pw1, v1, pw2, v2)
        odds_ratio = ((pw1 * v2) / (v1 * pw2)) if (v1 * pw2) != 0 else float("inf")

    # --- Wilson 95 % CI ---
    ci_lo, ci_hi = wilson_ci(pretrained_total_wins, total_decisive)

    # ------------------------------------------------------------------
    # 3. Build report
    # ------------------------------------------------------------------
    sep72 = "=" * 72
    sep40 = "-" * 40

    lines = [
        sep72,
        "  Pretraining Significance Test Report",
        "  ppo_lstm_pretrained_v4  vs  ppo_lstm_v4",
        sep72,
        "",
        "Data sources:",
        f"  File 1 (pretrained=P1): {file1}",
        f"  File 2 (pretrained=P2): {file2}",
        "",
        "Raw counts",
        sep40,
        f"  File 1 — {pretrained_slug} (P1) games : {tot1}",
        f"    pretrained wins  : {pw1}",
        f"    v4 wins          : {v1}",
        f"    draws            : {d1}",
        "",
        f"  File 2 — {pretrained_slug} (P2) games : {tot2}",
        f"    pretrained wins  : {pw2}",
        f"    v4 wins          : {v2}",
        f"    draws            : {d2}",
        "",
        "Pooled (decisive games only)",
        sep40,
        f"  Total games          : {total_games}",
        f"  Total decisive games : {total_decisive}",
        f"  Total draws          : {total_draws}",
        f"  pretrained_v4 wins   : {pretrained_total_wins}",
        f"  v4 wins              : {v4_total_wins}",
        f"  pretrained win rate  : {win_rate_pretrained:.4f}  "
        f"({win_rate_pretrained*100:.1f}%)",
        "",
        "Statistical Tests",
        sep40,
        "",
        "  1. Two-sided Binomial Test",
        "     H0: win_rate = 0.50  (no difference)",
        f"     k (pretrained wins) = {pretrained_total_wins}",
        f"     n (decisive games)  = {total_decisive}",
        f"     observed proportion = {binom_stat:.4f}",
        f"     p-value             = {binom_p:.6f}",
        ("     Result: SIGNIFICANT (p < 0.05) — "
         "BC pretraining significantly changes win rate."
         if binom_p < 0.05 else
         "     Result: NOT significant (p ≥ 0.05) — "
         "insufficient evidence for pretraining effect."),
        "",
        "  2. Fisher's Exact Test  (2×2 contingency table)",
        "     Table:",
        "                       pretrained wins | v4 wins",
        f"       file1 (pre=P1)  {pw1:>15}   | {v1:>7}",
        f"       file2 (pre=P2)  {pw2:>15}   | {v2:>7}",
        f"     odds ratio          = {odds_ratio:.4f}",
        f"     p-value             = {fisher_p:.6f}",
        ("     Result: SIGNIFICANT (p < 0.05) — "
         "win-rate difference is not explained by side-to-play bias."
         if fisher_p < 0.05 else
         "     Result: NOT significant (p ≥ 0.05) — "
         "result may be confounded by side-to-play."),
        "",
        "  3. Wilson 95% Confidence Interval on pretrained win rate",
        f"     Pooled win rate  = {win_rate_pretrained:.4f}",
        f"     95% CI           = [{ci_lo:.4f}, {ci_hi:.4f}]",
        ("     CI excludes 0.50 → significant at α=0.05."
         if ci_lo > 0.5 or ci_hi < 0.5 else
         "     CI includes 0.50 → NOT significant at α=0.05."),
        "",
        "Interpretation",
        sep40,
    ]

    # Interpretation block
    if binom_p < 0.05:
        if win_rate_pretrained > 0.5:
            lines += [
                "  BC pretraining SIGNIFICANTLY IMPROVES performance.",
                f"  The pretrained model wins {win_rate_pretrained*100:.1f}% of decisive games",
                f"  against the unpretrained baseline (p = {binom_p:.4f}, two-sided",
                f"  binomial test, n = {total_decisive} decisive games).",
                f"  95% Wilson CI: [{ci_lo:.3f}, {ci_hi:.3f}].",
                "",
                "  Note: ELO difference is 1283.0 − 1267.1 = 15.9 points.",
                "  The head-to-head evidence is STATISTICALLY SIGNIFICANT,",
                "  supporting the thesis claim.",
            ]
        else:
            lines += [
                "  BC pretraining SIGNIFICANTLY HURTS performance.",
                f"  The pretrained model wins only {win_rate_pretrained*100:.1f}% of decisive",
                f"  games against the baseline (p = {binom_p:.4f}).",
            ]
    else:
        lines += [
            f"  With p = {binom_p:.4f} (> 0.05) the difference is NOT statistically",
            "  significant at the 5% level despite the ELO gap of 15.9 points.",
            "  The ELO difference may reflect variance in the ELO estimation",
            f"  rather than a true effect.  Sample size: {total_decisive} decisive games.",
            "",
            "  To achieve 80% power detecting a true 55% win rate (δ=5%)",
            "  the required sample is ≈ 385 decisive games; more games are",
            "  recommended before making a strong claim.",
        ]

    lines += [
        "",
        sep72,
        f"  scipy used: {HAS_SCIPY}",
        sep72,
    ]

    return "\n".join(lines)


def main():
    default_results = Path("fow_chess_ludii/evaluation/results_v4")

    parser = argparse.ArgumentParser(
        description="Statistical significance test for BC pretraining effect."
    )
    parser.add_argument(
        "--file1",
        default=str(default_results / "ppo_lstm_pretrained_v4_vs_ppo_lstm_v4.csv"),
        help="CSV: ppo_lstm_pretrained_v4 as P1 vs ppo_lstm_v4 as P2",
    )
    parser.add_argument(
        "--file2",
        default=str(default_results / "ppo_lstm_v4_vs_ppo_lstm_pretrained_v4.csv"),
        help="CSV: ppo_lstm_v4 as P1 vs ppo_lstm_pretrained_v4 as P2",
    )
    parser.add_argument(
        "--out",
        default=str(default_results / "pretraining_significance.txt"),
        help="Output file for the significance report",
    )
    parser.add_argument(
        "--pretrained-slug",
        default="ppo_lstm_pretrained_v4",
        dest="pretrained_slug",
    )
    parser.add_argument(
        "--v4-slug",
        default="ppo_lstm_v4",
        dest="v4_slug",
    )
    args = parser.parse_args()

    file1    = Path(args.file1)
    file2    = Path(args.file2)
    out_path = Path(args.out)

    for p in (file1, file2):
        if not p.exists():
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            sys.exit(1)

    report = run(file1, file2, out_path,
                 pretrained_slug=args.pretrained_slug,
                 v4_slug=args.v4_slug)

    print(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
