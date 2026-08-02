"""Generate overall summary figures for each model across ALL experiments."""
import os, sys, argparse, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/parsed/metrics.csv')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    experiments = ['main_t0', 'main_t0.5', 'large_graphs', 'vm_placement', 'noise']
    exp_labels = ['main_t0', 'main_t0.5', '100-node', 'VM placement', 'Noise']
    formats = ['F1', 'F2', 'F3', 'F4', 'F5']
    fmt_colors = ['#E15759', '#F28E2B', '#EDC948', '#59A14F', '#4E79A7']
    
    for model_name, model_label, color in [
        ('deepseek', 'DeepSeek-V4-Flash', '#DD8452'),
        ('gpt', 'GPT-5-mini', '#4C72B0'),
    ]:
        mdf = df[df['model'] == model_name]
        if len(mdf) == 0:
            print(f"No data for {model_name}"); continue
        
        base_out = os.path.join(
            r'E:\vnf_llm_sensitivity\Experiment\results', model_name, 'figures', 'overall output figures'
        )
        os.makedirs(base_out, exist_ok=True)
        print(f"\n=== {model_label} -> {base_out} ===")
        
        # Figure 1: Grouped bar chart - feasibility by format across experiments
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(experiments))
        w = 0.15
        for i, fmt in enumerate(formats):
            vals = []
            for exp in experiments:
                sub = mdf[(mdf['experiment'] == exp) & (mdf['format'] == fmt)]
                vals.append(sub['feasible'].mean() * 100 if len(sub) > 0 else 0)
            ax.bar(x + i * w - 2 * w, vals, w, label=fmt, color=fmt_colors[i], edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels)
        ax.set_ylabel('Feasibility Rate (%)')
        ax.set_title(f'{model_label}: Feasibility by Format Across All Experiments')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(base_out, 'overall_fig1_feasibility_all_experiments.png'), dpi=300)
        plt.close()
        print("  Figure 1 saved - Feasibility")
        
        # Figure 2: Format sensitivity across experiments
        fig, ax = plt.subplots(figsize=(12, 6))
        sens = []
        for exp in experiments:
            sub = mdf[mdf['experiment'] == exp]
            s = sub.groupby('problem_id')['optimality_gap'].std().mean()
            sens.append(s if not pd.isna(s) else 0)
        ax.bar(x, sens, color=color, edgecolor='black', width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels)
        ax.set_ylabel('Mean Format Sensitivity Score')
        ax.set_title(f'{model_label}: Format Sensitivity Across All Experiments')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_out, 'overall_fig2_sensitivity_all_experiments.png'), dpi=300)
        plt.close()
        print("  Figure 2 saved - Sensitivity")
        
        # Figure 3: Performance trend line chart
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, fmt in enumerate(formats):
            vals = []
            for exp in experiments:
                sub = mdf[(mdf['experiment'] == exp) & (mdf['format'] == fmt)]
                vals.append(sub['optimality_gap'].mean() if len(sub) > 0 else 0)
            ax.plot(exp_labels, vals, marker='o', label=fmt, color=fmt_colors[i], linewidth=2, markersize=8)
        ax.set_ylabel('Mean Optimality Gap (%)')
        ax.set_title(f'{model_label}: Format Performance Trend Across Experiments')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_out, 'overall_fig3_performance_trend.png'), dpi=300)
        plt.close()
        print("  Figure 3 saved - Performance trend")
        
        print(f"  DONE: {len(os.listdir(base_out))} files in overall output figures")

if __name__ == '__main__':
    main()