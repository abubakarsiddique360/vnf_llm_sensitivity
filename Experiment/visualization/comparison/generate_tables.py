"""Generate 5 comparison CSV tables — FOR THE PAPER."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Generate comparison tables')
    parser.add_argument('--input', type=str, default='data/parsed/metrics.csv')
    parser.add_argument('--output', type=str, default='results/comparison/tables')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.input)

    # Table 1: Aggregated metrics both models
    t1 = df.groupby(['model', 'format']).agg(
        feasibility_rate=('feasible', lambda x: f"{x.mean()*100:.1f}%"),
        mean_gap=('optimality_gap', lambda x: f"{x.mean():.2f}"),
        median_gap=('optimality_gap', lambda x: f"{x.median():.2f}"),
        count=('feasible', 'count')
    ).reset_index()
    t1.to_csv(os.path.join(args.output, 'table1_aggregated_metrics.csv'), index=False)
    print("  Table 1 saved")

    # Table 2: Pairwise comparisons
    pairs = []
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        formats = mdf['format'].unique()
        for f1 in formats:
            for f2 in formats:
                if f1 < f2:
                    d1 = mdf[mdf['format'] == f1]['optimality_gap'].dropna()
                    d2 = mdf[mdf['format'] == f2]['optimality_gap'].dropna()
                    if len(d1) > 0 and len(d2) > 0:
                        from scipy import stats
                        t, p = stats.ttest_ind(d1, d2)
                        pairs.append({'model': model, 'format_a': f1, 'format_b': f2,
                                      'mean_diff': d1.mean()-d2.mean(), 'p_value': p})
    pd.DataFrame(pairs).to_csv(os.path.join(args.output, 'table2_pairwise_comparisons.csv'), index=False)
    print("  Table 2 saved")

    # Table 3: Regression summary
    t3 = df.groupby(['model', 'format']).agg(
        mean=('optimality_gap', 'mean'), std=('optimality_gap', 'std'),
        min=('optimality_gap', 'min'), max=('optimality_gap', 'max')
    ).reset_index()
    t3.to_csv(os.path.join(args.output, 'table3_regression_results.csv'), index=False)
    print("  Table 3 saved")

    # Table 4: Summary
    t4 = df.groupby(['model', 'experiment']).agg(
        total_queries=('feasible', 'count'),
        feasible=('feasible', 'sum'),
        mean_gap=('optimality_gap', 'mean')
    ).reset_index()
    t4.to_csv(os.path.join(args.output, 'table4_baseline_human_comparison.csv'), index=False)
    print("  Table 4 saved")

    # Table 5: Experiment summary
    t5 = df.groupby('experiment').agg(
        total_queries=('feasible', 'count'),
        feasibility_rate=('feasible', lambda x: f"{x.mean()*100:.1f}%")
    ).reset_index()
    t5.to_csv(os.path.join(args.output, 'table5_experiment_summary.csv'), index=False)
    print("  Table 5 saved")

    print(f"All comparison tables saved to {args.output}")


if __name__ == '__main__':
    main()