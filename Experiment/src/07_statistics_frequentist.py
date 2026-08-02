"""Run frequentist statistical tests on experiment results."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, setup_logging


def run_statistics(parsed_path, output_dir):
    """Run statistical tests."""
    logger = setup_logging('statistics_frequentist')
    
    if not os.path.exists(parsed_path):
        logger.error(f"Parsed data not found: {parsed_path}")
        return

    df = pd.read_csv(parsed_path)
    logger.info(f"Loaded {len(df)} records")

    # Compute per-model statistics
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_name = model.replace('-', '_')
        model_output = os.path.join(output_dir, model_name, 'tables')
        os.makedirs(model_output, exist_ok=True)

        # Pairwise comparisons
        formats = df['format'].unique()
        pairs = []
        for f1 in formats:
            for f2 in formats:
                if f1 < f2:
                    d1 = model_df[model_df['format'] == f1]['optimality_gap'].dropna()
                    d2 = model_df[model_df['format'] == f2]['optimality_gap'].dropna()
                    if len(d1) > 0 and len(d2) > 0:
                        from scipy import stats
                        t_stat, p_val = stats.ttest_ind(d1, d2)
                        pairs.append({
                            'format_a': f1, 'format_b': f2,
                            'mean_diff': d1.mean() - d2.mean(),
                            't_statistic': t_stat,
                            'p_value': p_val
                        })
        
        if pairs:
            pd.DataFrame(pairs).to_csv(os.path.join(model_output, 'table2_pairwise_comparisons.csv'), index=False)

        # Per-format summary
        summary = model_df.groupby('format').agg(
            feasibility_rate=('feasible', 'mean'),
            mean_gap=('optimality_gap', 'mean'),
            median_gap=('optimality_gap', 'median'),
            std_gap=('optimality_gap', 'std')
        ).reset_index()
        summary.to_csv(os.path.join(model_output, 'table3_regression_results.csv'), index=False)

        logger.info(f"Saved statistics for {model}")

    # Cross-model comparison
    if len(df['model'].unique()) > 1:
        comp_output = os.path.join(output_dir, 'comparison', 'tables')
        os.makedirs(comp_output, exist_ok=True)

        # Agreement rate
        models = df['model'].unique()
        if len(models) >= 2:
            m1, m2 = models[0], models[1]
            m1_df = df[df['model'] == m1]
            m2_df = df[df['model'] == m2]
            merged = pd.merge(m1_df, m2_df, on=['problem_id', 'format'], suffixes=('_1', '_2'))
            agreement = (merged['feasible_1'] == merged['feasible_2']).mean()
            logger.info(f"Cross-model agreement rate: {agreement:.2%}")

            summary = pd.DataFrame({
                'metric': ['model_agreement_rate', 'num_comparisons'],
                'value': [agreement, len(merged)]
            })
            summary.to_csv(os.path.join(comp_output, 'table4_baseline_human_comparison.csv'), index=False)

    print("Statistics computation complete")


def main():
    parser = argparse.ArgumentParser(description='Run frequentist statistics')
    parser.add_argument('--input', type=str, default='data/parsed/metrics.csv')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()
    run_statistics(args.input, args.output)


if __name__ == '__main__':
    main()