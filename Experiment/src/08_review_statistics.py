"""
Compute the statistical results reported in the paper's revised manuscript.

This script reproduces the numbers added during the minor-revision:
  1. Per-format feasibility numerators/denominators and 95% Wilson intervals
     (main phase, tau = 0) for both models.
  2. Paired McNemar tests (with continuity correction) between prompt formats,
     per model, on the same 120 instances.
  3. Mean input token counts per format (from API-reported token_usage) for
     both models in the main phase.

Inputs (existing in the repository, no new queries):
  - data/parsed/metrics.csv            (per-instance parsed results)
  - data/llm_responses/{deepseek,gpt}/main_t0/*.json   (raw responses + usage)

Outputs (written under results/comparison/tables/):
  - table_wilson_ci.csv
  - table_mcnemar.csv
  - table_prompt_tokens.csv

Usage:
  python src/08_review_statistics.py [--input data/parsed/metrics.csv]
                                     [--responses data/llm_responses]
                                     [--output results]
"""

import os
import sys
import glob
import json
import argparse
import math
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, setup_logging

MAIN_EXPERIMENT = "main_t0"
FORMATS = ["F1", "F2", "F3", "F4", "F5"]
MODELS = ["deepseek", "gpt"]


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return (centre - half, centre + half)


def mcnemar(a, b):
    """Paired McNemar test with continuity correction on two binary series.

    Returns (b, c, chi2, p): b = a=0,b=1 count; c = a=1,b=0 count.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    bb = int(((a == 0) & (b == 1)).sum())
    c = int(((a == 1) & (b == 0)).sum())
    if bb + c == 0:
        return (bb, c, 0.0, 1.0)
    chi2 = (abs(bb - c) - 1.0) ** 2 / (bb + c)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = 1.0 - chi2_dist.cdf(chi2, 1)
    except ImportError:
        p = float("nan")
    return (bb, c, chi2, p)


def compute_feasibility_and_cis(metrics_csv):
    """Per-format feasibility numerator/denominator and Wilson 95% CI."""
    df = pd.read_csv(metrics_csv)
    main = df[df["experiment"] == MAIN_EXPERIMENT]
    rows = []
    for model in MODELS:
        sub = main[main["model"] == model]
        for fmt in FORMATS:
            s = sub[sub["format"] == fmt]
            k = int(s["feasible"].sum())
            n = int(len(s))
            lo, hi = wilson_ci(k, n)
            rows.append({
                "model": model, "format": fmt,
                "feasible": k, "total": n,
                "rate_pct": round(100.0 * k / n, 1) if n else 0.0,
                "ci_low_pct": round(100.0 * lo, 1),
                "ci_high_pct": round(100.0 * hi, 1),
            })
        k = int(sub["feasible"].sum())
        n = int(len(sub))
        lo, hi = wilson_ci(k, n)
        rows.append({
            "model": model, "format": "ALL",
            "feasible": k, "total": n,
            "rate_pct": round(100.0 * k / n, 1) if n else 0.0,
            "ci_low_pct": round(100.0 * lo, 1),
            "ci_high_pct": round(100.0 * hi, 1),
        })
    return pd.DataFrame(rows)


def compute_mcnemar(metrics_csv):
    """Paired McNemar tests between all format pairs, per model."""
    df = pd.read_csv(metrics_csv)
    main = df[df["experiment"] == MAIN_EXPERIMENT]
    rows = []
    for model in MODELS:
        pivot = main[main["model"] == model].pivot_table(
            index="problem_id", columns="format", values="feasible"
        )
        for i, a in enumerate(FORMATS):
            for b in FORMATS[i + 1:]:
                if a not in pivot or b not in pivot:
                    continue
                bb, c, chi2, p = mcnemar(pivot[a].values, pivot[b].values)
                rows.append({
                    "model": model, "format_a": a, "format_b": b,
                    "discord_ab": bb, "discord_ba": c,
                    "chi2": round(chi2, 3), "p_value": round(p, 4),
                })
    return pd.DataFrame(rows)


def compute_prompt_tokens(responses_dir):
    """Mean prompt token counts per format from API-reported usage."""
    rows = []
    for model in MODELS:
        files = glob.glob(os.path.join(responses_dir, model, MAIN_EXPERIMENT, "*.json"))
        by_fmt = {f: [] for f in FORMATS}
        for fp in files:
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            fmt = d.get("format")
            if fmt not in by_fmt:
                continue
            pt = (d.get("token_usage") or {}).get("prompt_tokens", 0)
            by_fmt[fmt].append(pt)
        for fmt in FORMATS:
            vals = by_fmt[fmt]
            rows.append({
                "model": model, "format": fmt,
                # truncate (int) to match the values reported in the paper
                "mean_prompt_tokens": int(np.mean(vals)) if vals else 0,
                "median_prompt_tokens": int(np.median(vals)) if vals else 0,
                "n": len(vals),
            })
        all_vals = [v for f in FORMATS for v in by_fmt[f]]
        rows.append({
            "model": model, "format": "ALL",
            "mean_prompt_tokens": int(np.mean(all_vals)) if all_vals else 0,
            "median_prompt_tokens": int(np.median(all_vals)) if all_vals else 0,
            "n": len(all_vals),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Reproduce revised-manuscript statistics")
    parser.add_argument("--input", type=str, default="data/parsed/metrics.csv")
    parser.add_argument("--responses", type=str, default=None)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    logger = setup_logging("review_statistics")
    config = load_environment()

    responses_dir = args.responses or config["LLM_RESPONSES_DIR"]
    out_dir = os.path.join(args.output, "comparison", "tables")
    os.makedirs(out_dir, exist_ok=True)

    feas = compute_feasibility_and_cis(args.input)
    feas_path = os.path.join(out_dir, "table_wilson_ci.csv")
    feas.to_csv(feas_path, index=False)
    logger.info("Saved %s", feas_path)

    mc = compute_mcnemar(args.input)
    mc_path = os.path.join(out_dir, "table_mcnemar.csv")
    mc.to_csv(mc_path, index=False)
    logger.info("Saved %s", mc_path)

    tok = compute_prompt_tokens(responses_dir)
    tok_path = os.path.join(out_dir, "table_prompt_tokens.csv")
    tok.to_csv(tok_path, index=False)
    logger.info("Saved %s", tok_path)

    # ---- console summary (should match the paper) ----
    print("\n=== Feasibility + 95% Wilson CI (main phase, tau=0) ===")
    for _, r in feas.iterrows():
        print("%-8s %-4s %d/%d = %5.1f%%  CI [%4.1f, %4.1f]" % (
            r["model"], r["format"], r["feasible"], r["total"],
            r["rate_pct"], r["ci_low_pct"], r["ci_high_pct"]))

    print("\n=== McNemar (paired, continuity-corrected) ===")
    for _, r in mc.iterrows():
        print("%-8s %s vs %s: discord(%d,%d) chi2=%.2f p=%.4f" % (
            r["model"], r["format_a"], r["format_b"],
            r["discord_ab"], r["discord_ba"], r["chi2"], r["p_value"]))

    print("\n=== Mean prompt tokens per format ===")
    for _, r in tok.iterrows():
        print("%-8s %-4s mean=%d median=%d (n=%d)" % (
            r["model"], r["format"], r["mean_prompt_tokens"],
            r["median_prompt_tokens"], r["n"]))


if __name__ == "__main__":
    main()
