"""Generate CSV tables for DeepSeek-V4-Flash only, per experiment."""
import os, sys, argparse, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_tables_for_exp(df, output_dir, label):
    """Generate 4 tables for one experiment's data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Table 1: Aggregated metrics by format
    t1 = df.groupby('format').agg(
        feasibility_rate=('feasible', lambda x: f"{x.mean()*100:.1f}%"),
        mean_gap=('optimality_gap', 'mean'),
        median_gap=('optimality_gap', 'median'),
        count=('feasible', 'count')
    ).reset_index()
    t1.to_csv(os.path.join(output_dir, 'table1_aggregated_metrics.csv'), index=False)
    
    # Table 2: Pairwise comparisons
    formats = df['format'].unique()
    pairs = []
    for f1 in formats:
        for f2 in formats:
            if f1 < f2:
                d1 = df[df['format'] == f1]['optimality_gap'].dropna()
                d2 = df[df['format'] == f2]['optimality_gap'].dropna()
                if len(d1) > 0 and len(d2) > 0:
                    from scipy import stats
                    t, p = stats.ttest_ind(d1, d2)
                    pairs.append({'format_a': f1, 'format_b': f2, 'mean_diff': round(d1.mean()-d2.mean(), 2), 'p_value': round(p, 4)})
    pd.DataFrame(pairs).to_csv(os.path.join(output_dir, 'table2_pairwise_comparisons.csv'), index=False)
    
    # Table 3: Regression results (dummy-coded)
    feas = df[df['feasible'] == 1].copy()
    if len(feas) > 0:
        feas['log_gap'] = np.log(feas['optimality_gap'].clip(lower=0.1))
        from scipy import stats as sp_stats
        results = []
        for fmt in formats:
            subset = feas[feas['format'] == fmt]['optimality_gap']
            if len(subset) > 0:
                results.append({'format': fmt, 'mean_gap': subset.mean(), 'std_gap': subset.std(), 'n': len(subset)})
        pd.DataFrame(results).to_csv(os.path.join(output_dir, 'table3_regression_results.csv'), index=False)
    
    # Table 4: Baseline comparison
    t4 = df.groupby('format').agg(feasibility_rate=('feasible', 'mean')).reset_index()
    t4['method'] = 'deepseek-v4-flash'
    t4.to_csv(os.path.join(output_dir, 'table4_baseline_comparison.csv'), index=False)
    
    print(f"  Tables saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate DeepSeek tables')
    parser.add_argument('--input', type=str, default='data/parsed/metrics.csv')
    parser.add_argument('--output', type=str, default='results/deepseek/tables')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    ds_df = df[df['model'] == 'deepseek']
    if len(ds_df) == 0:
        print("No DeepSeek data found")
        return
    
    experiments = {
        'main_t0': 'T=0 (Main)',
        'main_t0.5': 'T=0.5 (Temperature Sensitivity)',
        'large_graphs': 'Large Graphs (100-node)',
        'vm_placement': 'VM Placement',
        'noise': 'Noise',
    }
    
    for exp_name, label in experiments.items():
        exp_df = ds_df[ds_df['experiment'] == exp_name]
        if len(exp_df) > 0:
            exp_dir = os.path.join(args.output, exp_name)
            print(f"Generating tables for: {label} ({len(exp_df)} responses)")
            generate_tables_for_exp(exp_df, exp_dir, label)


if __name__ == '__main__':
    main()
