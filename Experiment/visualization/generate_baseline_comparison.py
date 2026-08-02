"""
Generate baseline comparison figure: greedy vs LLMs feasibility bar chart.
IEEE-style: compact size, large readable fonts.
"""
import os, sys, csv, pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
})

df = pd.read_csv(r'E:\vnf_llm_sensitivity\Experiment\data\parsed\metrics.csv')
GPT = '#4C72B0'; DS = '#DD8452'; GR = '#2CA02C'
out = r'E:\vnf_llm_sensitivity\Experiment\results\comparison\figures'
os.makedirs(out, exist_ok=True)

# Build greedy baseline
greedy = {}
with open(r'E:\vnf_llm_sensitivity\Experiment\data\baselines\greedy_results.csv') as f:
    for r in csv.DictReader(f):
        pid = int(r['problem_id'])
        gap_val = r.get('optimality_gap', '')
        greedy[pid] = {'feasible': int(r['feasible']),
                       'gap': float(gap_val) if gap_val and gap_val != '' else None}

greedy_feas = sum(1 for pid in range(1, 121) if pid in greedy and greedy[pid]['feasible'] == 1) / 120 * 100
gpt_feas = df[(df['model'] == 'gpt') & (df['experiment'] == 'main_t0')]['feasible'].mean() * 100
ds_feas = df[(df['model'] == 'deepseek') & (df['experiment'] == 'main_t0')]['feasible'].mean() * 100

print("Baseline Fig 1: Feasibility Comparison...")
fig, ax = plt.subplots(figsize=(6.5, 3.8))
x = [0, 1, 2, 3]
vals = [greedy_feas, gpt_feas, ds_feas, 0.0]
labels = ['Greedy\nHeuristic', 'GPT-5-mini', 'DeepSeek\nV4-Flash', 'Random']
colors = [GR, GPT, DS, '#999999']
bars = ax.bar(x, vals, color=colors, edgecolor='black', width=0.55, linewidth=1.0)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Feasibility Rate (%)', fontsize=12)
ax.set_ylim(0, 105)
ax.set_title('Feasibility: LLMs vs Baselines (120 Problems)', fontsize=13, fontweight='bold', pad=8)
ax.grid(axis='y', alpha=0.25)
fig.tight_layout(pad=0.8)
fig.savefig(os.path.join(out, 'baseline_fig1_feasibility_comparison.png'), dpi=300); plt.close()
print('  Saved.')

print(f'Baseline figure regenerated in {out}')
