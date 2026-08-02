"""Generate 13 comparison PNG figures combining BOTH models — FOR THE PAPER."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_figures(csv_path, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print("No data found")
        return

    GPT_COLOR = '#4C72B0'   # blue
    DS_COLOR = '#DD8452'    # orange
    
    # Seaborn assigns palette colors alphabetically by hue.
    # 'deepseek' < 'gpt', so deepseek gets palette[0], gpt gets palette[1].
    # For GPT=blue, palette must be [DS_COLOR, GPT_COLOR]

    # Figure 1: Feasibility by format - grouped bars
    plt.figure(figsize=(10, 6))
    feas = df.groupby(['model', 'format'])['feasible'].mean() * 100
    feas_df = feas.reset_index()
    sns.barplot(data=feas_df, x='format', y='feasible', hue='model',
                palette=[DS_COLOR, GPT_COLOR], edgecolor='black')
    plt.title('Feasibility Rate by Format: GPT vs DeepSeek', fontsize=14)
    plt.ylabel('Feasibility Rate (%)')
    plt.xlabel('Format')
    plt.ylim(0, 100)
    plt.legend(title='Model')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_feasibility_by_format.png'), dpi=300)
    plt.close()
    print("  Figure 1 saved")

    # Figure 2: Optimality gap box plots
    plt.figure(figsize=(12, 6))
    fig2_df = df[df['feasible'] == 1]
    sns.boxplot(data=fig2_df, x='format', y='optimality_gap', hue='model',
                palette=[DS_COLOR, GPT_COLOR])
    plt.title('Optimality Gap by Format: GPT vs DeepSeek', fontsize=14)
    plt.ylabel('Optimality Gap (%)')
    plt.xlabel('Format')
    plt.legend(title='Model')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_optimality_gap_by_format.png'), dpi=300)
    plt.close()
    print("  Figure 2 saved")

    # Figure 3: Sensitivity by family
    plt.figure(figsize=(10, 6))
    sens = df.groupby(['problem_id', 'model', 'family'])['optimality_gap'].std().reset_index()
    sens = sens.groupby(['model', 'family'])['optimality_gap'].mean().reset_index()
    sns.barplot(data=sens, x='family', y='optimality_gap', hue='model',
                palette=[DS_COLOR, GPT_COLOR], edgecolor='black')
    plt.title('Format Sensitivity by Graph Family', fontsize=14)
    plt.ylabel('Sensitivity Score')
    plt.xlabel('Graph Family')
    plt.legend(title='Model')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_sensitivity_by_family.png'), dpi=300)
    plt.close()
    print("  Figure 3 saved")

    # Figure 4: Confusion matrix heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, model in enumerate(['gpt', 'deepseek']):
        infeas = df[(df['model'] == model) & (df['feasible'] == 0)]
        if len(infeas) > 0:
            ct = pd.crosstab(infeas['format'], infeas['violation_type'])
            sns.heatmap(ct, annot=True, fmt='d', cmap='Reds' if idx == 0 else 'Oranges', ax=axes[idx])
            axes[idx].set_title(f'{model}')
    plt.suptitle('Violation Types by Format', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_confusion_matrix.png'), dpi=300)
    plt.close()
    print("  Figure 4 saved")

    # Figure 5: Rank agreement
    plt.figure(figsize=(10, 6))
    for model, color in [('gpt', GPT_COLOR), ('deepseek', DS_COLOR)]:
        mdf = df[df['model'] == model]
        ranks = mdf.groupby(['problem_id', 'format'])['optimality_gap'].mean().reset_index()
        ranks['rank'] = ranks.groupby('problem_id')['optimality_gap'].rank()
        avg_rank = ranks.groupby('format')['rank'].mean()
        plt.plot(avg_rank.index, avg_rank.values, marker='o', label=model, color=color, linewidth=2)
    plt.title('Average Format Rank (1=best)', fontsize=14)
    plt.ylabel('Average Rank')
    plt.xlabel('Format')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_rank_agreement.png'), dpi=300)
    plt.close()
    print("  Figure 5 saved")

    # Figure 6: Cross-model agreement scatter
    plt.figure(figsize=(10, 8))
    gpt = df[df['model'] == 'gpt'].groupby(['problem_id', 'format'])['optimality_gap'].mean()
    ds = df[df['model'] == 'deepseek'].groupby(['problem_id', 'format'])['optimality_gap'].mean()
    merged = pd.DataFrame({'gpt': gpt, 'deepseek': ds}).dropna()
    plt.scatter(merged['gpt'], merged['deepseek'], alpha=0.5, c='#8172B2', edgecolors='black')
    lims = [min(merged.min().min(), 0), max(merged.max().max(), 100)]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.title('Cross-Model Agreement: GPT vs DeepSeek', fontsize=14)
    plt.xlabel('GPT Optimality Gap (%)')
    plt.ylabel('DeepSeek Optimality Gap (%)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_cross_model_agreement.png'), dpi=300)
    plt.close()
    print("  Figure 6 saved")

    # Figure 7: Performance by size
    plt.figure(figsize=(10, 6))
    size_feas = df.groupby(['model', 'size_category'])['feasible'].mean() * 100
    size_feas_df = size_feas.reset_index()
    sns.barplot(data=size_feas_df, x='size_category', y='feasible', hue='model',
                palette=[DS_COLOR, GPT_COLOR], edgecolor='black')
    plt.title('Feasibility by Graph Size', fontsize=14)
    plt.ylabel('Feasibility Rate (%)')
    plt.xlabel('Size Category')
    plt.ylim(0, 100)
    plt.legend(title='Model')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_performance_by_size.png'), dpi=300)
    plt.close()
    print("  Figure 7 saved")

    # Figure 13: Radar chart
    plt.figure(figsize=(8, 8))
    from math import pi
    categories = ['F1', 'F2', 'F3', 'F4', 'F5']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    for model, color in [('gpt', GPT_COLOR), ('deepseek', DS_COLOR)]:
        mdf = df[df['model'] == model]
        values = [mdf[mdf['format'] == f]['feasible'].mean() * 100 for f in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title('Format Comparison Radar', fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig13_format_radar.png'), dpi=300)
    plt.close()
    print("  Figure 13 saved")

    print(f"All comparison figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate comparison figures')
    parser.add_argument('--input', type=str, default='data/parsed/metrics.csv')
    parser.add_argument('--output', type=str, default='results/comparison/figures')
    args = parser.parse_args()
    generate_figures(args.input, args.output)


if __name__ == '__main__':
    main()