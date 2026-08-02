"""Add confusion matrix and rank agreement figures to overall output figures folders."""
import os, sys, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'E:\vnf_llm_sensitivity\Experiment\data\parsed\metrics.csv')
experiments = ['main_t0', 'main_t0.5', 'large_graphs', 'vm_placement', 'noise']
exp_labels = ['main_t0', 'main_t0.5', '100-node', 'VM', 'Noise']
formats = ['F1','F2','F3','F4','F5']
fmt_colors = ['#E15759','#F28E2B','#EDC948','#59A14F','#4E79A7']

for model_name, model_label, cmap in [('deepseek','DeepSeek-V4-Flash','Oranges'),('gpt','GPT-5-mini','Reds')]:
    mdf = df[df['model'] == model_name]
    if len(mdf) == 0:
        print(f"No data for {model_name}")
        continue
    base_out = os.path.join(r'E:\vnf_llm_sensitivity\Experiment\results', model_name, 'figures', 'overall output figures')
    os.makedirs(base_out, exist_ok=True)
    print(f"\n{model_label} -> {base_out}")
    
    # Figure 4: Confusion Matrix
    plt.figure(figsize=(10, 7))
    infeas = mdf[mdf['feasible'] == 0]
    if len(infeas) > 0:
        ct = pd.crosstab(infeas['experiment'], infeas['violation_type'])
        for e in experiments:
            if e not in ct.index:
                ct.loc[e] = 0
        ct = ct.loc[[e for e in experiments if e in ct.index]]
        sns.heatmap(ct, annot=True, fmt='d', cmap=cmap)
        plt.title(f'{model_label}: Violation Types by Experiment')
        plt.xlabel('Violation Type')
        plt.ylabel('Experiment')
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, 'overall_fig4_confusion_matrix.png'), dpi=300)
    plt.close()
    print("  Figure 4 saved - Confusion matrix")
    
    # Figure 5: Rank Agreement
    plt.figure(figsize=(10, 6))
    for i, fmt in enumerate(formats):
        vals = []
        for exp in experiments:
            sub = mdf[(mdf['experiment'] == exp) & (mdf['format'] == fmt)]
            vals.append(sub['optimality_gap'].mean() if len(sub) > 0 else 0)
        plt.plot(exp_labels, vals, marker='o', label=fmt, color=fmt_colors[i], linewidth=2, markersize=8)
    plt.ylabel('Mean Optimality Gap (%)')
    plt.title(f'{model_label}: Format Rank Agreement Across Experiments')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, 'overall_fig5_rank_agreement.png'), dpi=300)
    plt.close()
    print("  Figure 5 saved - Rank agreement")
    
    files = [f for f in os.listdir(base_out) if f.endswith('.png')]
    print(f"  Total: {len(files)} files in folder")

print("\nDone!")