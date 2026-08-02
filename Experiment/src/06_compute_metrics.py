"""Compute aggregated metrics from parsed results."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, setup_logging


def compute_metrics(csv_path, output_dir):
    """Compute aggregated metrics from parsed CSV."""
    logger = setup_logging('compute_metrics')
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records")

    # Per-format per-model metrics
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_name = model.replace('-', '_')
        model_output = os.path.join(output_dir, model_name, 'tables')
        os.makedirs(model_output, exist_ok=True)

        agg = model_df.groupby('format').agg(
            feasibility_rate=('feasible', 'mean'),
            mean_gap=('optimality_gap', 'mean'),
            median_gap=('optimality_gap', 'median'),
            std_gap=('optimality_gap', 'std'),
            count=('feasible', 'count')
        ).reset_index()
        agg['feasibility_rate'] *= 100
        agg.to_csv(os.path.join(model_output, 'table1_aggregated_metrics.csv'), index=False)
        logger.info(f"Saved {model} aggregated metrics")

    # Cross-model comparison
    comparison_output = os.path.join(output_dir, 'comparison', 'tables')
    os.makedirs(comparison_output, exist_ok=True)

    # Both models combined
    combined = df.groupby(['model', 'format']).agg(
        feasibility_rate=('feasible', 'mean'),
        mean_gap=('optimality_gap', 'mean'),
        median_gap=('optimality_gap', 'median'),
        std_gap=('optimality_gap', 'std'),
        count=('feasible', 'count')
    ).reset_index()
    combined['feasibility_rate'] *= 100
    combined.to_csv(os.path.join(comparison_output, 'table1_aggregated_metrics.csv'), index=False)

    # Format sensitivity score per problem
    sensitivity = df.groupby(['problem_id', 'model']).agg(
        sensitivity_score=('optimality_gap', 'std')
    ).reset_index()
    avg_sensitivity = sensitivity.groupby('model')['sensitivity_score'].mean()
    logger.info(f"Format sensitivity: {avg_sensitivity.to_dict()}")

    # Print summary
    print(f"\nMetrics Summary:")
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        feas = mdf['feasible'].mean() * 100
        gap = mdf['optimality_gap'].mean()
        print(f"  {model}: Feasibility={feas:.1f}%, Mean Gap={gap:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Compute metrics')
    parser.add_argument('--input', type=str, default='data/parsed/metrics.csv')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    compute_metrics(args.input, args.output)


if __name__ == '__main__':
    main()