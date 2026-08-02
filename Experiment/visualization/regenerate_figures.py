"""
Regenerate 4 comparison figures with compact dimensions and large, clear text.
IEEE-style: small figure size, large readable fonts.
"""
import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# ── IEEE style: small figure size, large readable fonts ──
# Target: single-column ~3.5" wide, double-column ~7" wide.
# Font sizes set so text is readable when figure is placed at 100% in LaTeX.
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

df = pd.read_csv(r'E:\vnf_llm_sensitivity\Experiment\data\parsed\metrics.csv')
GPT_COLOR = '#4C72B0'
DS_COLOR = '#DD8452'
experiments = ['main_t0', 'main_t0.5', 'large_graphs', 'vm_placement', 'noise']
exp_labels = ['main_t0', 'main_t0.5', '100-node', 'VM', 'Noise']
formats = ['F1', 'F2', 'F3', 'F4', 'F5']
fmt_colors = ['#E15759', '#F28E2B', '#EDC948', '#59A14F', '#4E79A7']
out = r'E:\vnf_llm_sensitivity\Experiment\results\comparison\figures'
os.makedirs(out, exist_ok=True)

ANNOT_FS = 8    # annotation font size
TITLE_FS = 10   # title font size (small — IEEE uses captions)
LABEL_FS = 9    # axis label font size
TICK_FS = 8     # tick label font size
LEGEND_FS = 8   # legend font size

# ═══════════════════════════════════════════════
# FIGURE 2: Format Sensitivity Comparison (single-col)
# ═══════════════════════════════════════════════
print("Regenerating Figure 2: Format Sensitivity...")
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for model, label, color in [('gpt', 'GPT-5-mini', GPT_COLOR), ('deepseek', 'DeepSeek-V4-Flash', DS_COLOR)]:
    vals = []
    for e in experiments:
        sub = df[(df['model'] == model) & (df['experiment'] == e)]
        s = sub.groupby('problem_id')['optimality_gap'].std().mean()
        vals.append(s if not pd.isna(s) else 0)
    ax.plot(exp_labels, vals, marker='o', label=label, color=color, linewidth=1.5, markersize=5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=ANNOT_FS, fontweight='bold', color=color)

ax.set_xlabel('Experiment', fontsize=LABEL_FS)
ax.set_ylabel('Mean Format Sensitivity Score', fontsize=LABEL_FS)
ax.set_title('Format Sensitivity Across Experiments', fontsize=TITLE_FS, fontweight='bold', pad=6)
ax.legend(fontsize=LEGEND_FS, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(labelsize=TICK_FS)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(out, 'comparison_fig2_sensitivity_comparison.png'), dpi=300)
plt.close()
print('  Figure 2 saved.')

# ═══════════════════════════════════════════════
# FIGURE 3: Heatmap Comparison (double-col)
# ═══════════════════════════════════════════════
print("Regenerating Figure 3: Heatmap...")
feas = df.groupby(['model', 'experiment', 'format'])['feasible'].mean().unstack('format')
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
for idx, (model, cmap) in enumerate([('gpt', 'Blues'), ('deepseek', 'Oranges')]):
    sub = feas.loc[model].reindex(experiments).fillna(0)
    sns.heatmap(sub, annot=True, fmt='.1%', cmap=cmap, ax=axes[idx],
                cbar_kws={'label': 'Feasibility', 'shrink': 0.8},
                annot_kws={'fontsize': 7, 'fontweight': 'bold'},
                linewidths=0.5)
    axes[idx].set_title(f'{model.upper()}', fontsize=9, fontweight='bold')
    axes[idx].set_ylabel('Experiment', fontsize=LABEL_FS)
    axes[idx].set_xlabel('Format', fontsize=LABEL_FS)
    axes[idx].tick_params(labelsize=TICK_FS)
    # Colorbar label size
    cbar = axes[idx].collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Feasibility', fontsize=LABEL_FS)

fig.suptitle('Feasibility Heatmap by Format and Experiment', fontsize=TITLE_FS, fontweight='bold', y=1.02)
fig.tight_layout(pad=1.0)
fig.savefig(os.path.join(out, 'comparison_fig3_heatmap_comparison.png'), dpi=300)
plt.close()
print('  Figure 3 saved.')

# ═══════════════════════════════════════════════
# FIGURE: Baseline Feasibility Comparison (single-col)
# ═══════════════════════════════════════════════
print("Regenerating Baseline Figure 1: Feasibility Comparison...")
import csv

greedy = {}
with open(r'E:\vnf_llm_sensitivity\Experiment\data\baselines\greedy_results.csv') as f:
    for r in csv.DictReader(f):
        pid = int(r['problem_id'])
        greedy[pid] = {'feasible': int(r['feasible']), 'gap': float(r['optimality_gap']) if r['optimality_gap'] else None}

greedy_feas = sum(1 for pid in range(1, 121) if pid in greedy and greedy[pid]['feasible'] == 1) / 120 * 100
gpt_feas = df[(df['model'] == 'gpt') & (df['experiment'] == 'main_t0')]['feasible'].mean() * 100
ds_feas = df[(df['model'] == 'deepseek') & (df['experiment'] == 'main_t0')]['feasible'].mean() * 100

G_REEDY_COLOR = '#2CA02C'

fig, ax = plt.subplots(figsize=(3.5, 2.5))
x = [0, 1, 2]
vals = [greedy_feas, gpt_feas, ds_feas]
labels = ['Greedy\nHeuristic', 'GPT-5-mini', 'DeepSeek\nV4-Flash']
colors = [G_REEDY_COLOR, GPT_COLOR, DS_COLOR]
bars = ax.bar(x, vals, color=colors, edgecolor='black', width=0.55, linewidth=0.8)

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', fontsize=ANNOT_FS, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=TICK_FS)
ax.set_xlabel('Method', fontsize=LABEL_FS)
ax.set_ylabel('Feasibility Rate (%)', fontsize=LABEL_FS)
ax.set_ylim(0, 110)
ax.set_title('Feasibility: LLMs vs Baselines (120 Problems)', fontsize=TITLE_FS, fontweight='bold', pad=6)
ax.grid(axis='y', alpha=0.25)
ax.tick_params(labelsize=TICK_FS)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(out, 'baseline_fig1_feasibility_comparison.png'), dpi=300)
plt.close()
print('  Baseline Figure 1 saved.')

# ═══════════════════════════════════════════════
# FIGURE 4: Radar Comparison (single-col)
# ═══════════════════════════════════════════════
print("Regenerating Figure 4: Radar...")
metrics = ['main_t0\nfeas.', 'main_t0.5\nfeas.', 'large graphs\nfeas.', 'VM placement\nfeas.', 'Noise\nfeas.', 'Format\nsensitivity']
gpt_vals = []
ds_vals = []
for e in experiments:
    gpt_vals.append(df[(df['model'] == 'gpt') & (df['experiment'] == e)]['feasible'].mean() * 100)
    ds_vals.append(df[(df['model'] == 'deepseek') & (df['experiment'] == e)]['feasible'].mean() * 100)
sens_gpt = df[df['model'] == 'gpt'].groupby('problem_id')['optimality_gap'].std().mean()
sens_ds = df[df['model'] == 'deepseek'].groupby('problem_id')['optimality_gap'].std().mean()
max_sens = max(sens_gpt, sens_ds) if max(sens_gpt, sens_ds) > 0 else 1
gpt_vals.append(max(0, 100 - (sens_gpt / max_sens) * 100))
ds_vals.append(max(0, 100 - (sens_ds / max_sens) * 100))

N = len(metrics)
angles = [n / float(N) * 2 * pi for n in range(N)]
gpt_vals_closed = gpt_vals + gpt_vals[:1]
ds_vals_closed = ds_vals + ds_vals[:1]
angles_closed = angles + angles[:1]

fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))
ax.fill(angles_closed, gpt_vals_closed, alpha=0.15, color=GPT_COLOR)
ax.plot(angles_closed, gpt_vals_closed, 'o-', linewidth=1.5, markersize=4, label='GPT-5-mini', color=GPT_COLOR)
ax.fill(angles_closed, ds_vals_closed, alpha=0.15, color=DS_COLOR)
ax.plot(angles_closed, ds_vals_closed, 'o-', linewidth=1.5, markersize=4, label='DeepSeek-V4-Flash', color=DS_COLOR)
ax.set_xticks(angles_closed[:-1])
ax.set_xticklabels(metrics, fontsize=7, fontweight='bold')
ax.set_title('Performance Radar: DeepSeek vs GPT', fontsize=TITLE_FS, fontweight='bold', pad=14)
ax.legend(loc='upper right', fontsize=LEGEND_FS, bbox_to_anchor=(1.25, 1.08))
ax.tick_params(labelsize=7)
fig.tight_layout(pad=1.2)
fig.savefig(os.path.join(out, 'comparison_fig4_radar_comparison.png'), dpi=300)
plt.close()
print('  Figure 4 saved.')
print('  Figure 4 saved.')

print(f'\nAll 4 figures regenerated successfully in:\n  {out}')
