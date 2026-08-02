"""Generate 7 PNG figures for DeepSeek-V4-Pro only."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import load_environment, setup_logging


def generate_figures_for_exp(df, output_dir, exp_name, model_label):
    """Generate 7 figures for one experiment's data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    COLOR = '#DD8452'
    label = f'DeepSeek-V4-Flash ({model_label})'
    
    # Figure 1: Feasibility by format
    plt.figure(figsize=(8, 5))
    feas = df.groupby('format')['feasible'].mean() * 100
    feas.plot(kind='bar', color=COLOR, edgecolor='black')
    plt.title(f'{label}: Feasibility Rate by Format')
    plt.ylabel('Feasibility Rate (%)')
    plt.xlabel('Format')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_feasibility_by_format.png'), dpi=300)
    plt.close()
    
    # Figure 2: Optimality gap by format
    plt.figure(figsize=(8, 5))
    feas_df = df[df['feasible'] == 1]
    if len(feas_df) > 0:
        feas_df.boxplot(column='optimality_gap', by='format')
        plt.title(f'{label}: Optimality Gap by Format')
        plt.ylabel('Optimality Gap (%)')
        plt.xlabel('Format')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig2_optimality_gap_by_format.png'), dpi=300)
        plt.close()
    
    # Figure 3: Sensitivity by family
    plt.figure(figsize=(8, 5))
    sens = df.groupby(['problem_id', 'family'])['optimality_gap'].std().reset_index()
    sens.groupby('family')['optimality_gap'].mean().plot(kind='bar', color=COLOR, edgecolor='black')
    plt.title(f'{label}: Format Sensitivity by Graph Family')
    plt.ylabel('Sensitivity Score')
    plt.xlabel('Graph Family')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_sensitivity_by_family.png'), dpi=300)
    plt.close()
    
    # Figure 4: Confusion matrix
    infeas = df[df['feasible'] == 0]
    if len(infeas) > 0:
        plt.figure(figsize=(8, 6))
        ct = pd.crosstab(infeas['format'], infeas['violation_type'])
        import seaborn as sns
        sns.heatmap(ct, annot=True, fmt='d', cmap='Oranges')
        plt.title(f'{label}: Confusion Matrix - Violations by Format')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig4_confusion_matrix.png'), dpi=300)
        plt.close()
    
    # Figure 5: Rank agreement
    plt.figure(figsize=(8, 5))
    ranks = df.groupby(['problem_id', 'format'])['optimality_gap'].mean().reset_index()
    ranks['rank'] = ranks.groupby('problem_id')['optimality_gap'].rank()
    avg_rank = ranks.groupby('format')['rank'].mean()
    avg_rank.plot(kind='bar', color=COLOR, edgecolor='black')
    plt.title(f'{label}: Average Format Rank (1=best)')
    plt.ylabel('Average Rank')
    plt.xlabel('Format')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_rank_agreement.png'), dpi=300)
    plt.close()
    
    # Figure 7: Performance by size
    plt.figure(figsize=(8, 5))
    size_feas = df.groupby('size_category')['feasible'].mean() * 100
    size_feas.plot(kind='bar', color=COLOR, edgecolor='black')
    plt.title(f'{label}: Feasibility by Graph Size')
    plt.ylabel('Feasibility Rate (%)')
    plt.xlabel('Size Category')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_performance_by_size.png'), dpi=300)
    plt.close()
    
    print(f"  Figures saved to {output_dir}")


def generate_figures(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    ds_df = df[df['model'] == 'deepseek']
    
    if len(ds_df) == 0:
        print("No DeepSeek data found")
        return
    
    # Map experiment names to display labels
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
            exp_dir = os.path.join(output_dir, exp_name)
            print(f"Generating figures for: {label} ({len(exp_df)} responses)")
            generate_figures_for_exp(exp_df, exp_dir, exp_name, label)


def main():
    parser = argparse.ArgumentParser(description='Generate DeepSeek figures')
    parser.add_argument('--input', type=str, default='data/parsed/metrics.csv')
    parser.add_argument('--output', type=str, default='results/deepseek/figures')
    args = parser.parse_args()
    generate_figures(args.input, args.output)


if __name__ == '__main__':
    main()