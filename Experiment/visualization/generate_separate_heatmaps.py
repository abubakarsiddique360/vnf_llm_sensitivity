"""Generate separate GPT and DeepSeek heatmap figures from the same data."""
import os, sys, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── IEEE style: small figure, large readable fonts ──
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

LABEL_FS = 9
TICK_FS = 8

df = pd.read_csv(r'E:\vnf_llm_sensitivity\Experiment\data\parsed\metrics.csv')
out = r'E:\vnf_llm_sensitivity\Experiment\results\comparison\figures'
os.makedirs(out, exist_ok=True)

experiments = ['main_t0', 'main_t0.5', 'large_graphs', 'vm_placement', 'noise']
exp_labels = ['Main T=0', 'Main T=0.5', '100-Node Graphs', 'VM Placement', 'Noise Injection']

# Pivot: feasibility rate per model × experiment × format
feas = df.groupby(['model', 'experiment', 'format'])['feasible'].mean().unstack('format')

# ─── Separate GPT Heatmap ───
print("Generating GPT heatmap...")
fig, ax = plt.subplots(figsize=(3.8, 3.2))
sub_gpt = feas.loc['gpt'].reindex(experiments).fillna(0)
sub_gpt.index = exp_labels
sns.heatmap(sub_gpt, annot=True, fmt='.1%', cmap='Blues', ax=ax,
            cbar_kws={'label': 'Feasibility', 'shrink': 0.8},
            annot_kws={'fontsize': 8, 'fontweight': 'bold'},
            linewidths=0.5)
ax.set_title('GPT-5-mini: Feasibility Heatmap by Format & Experiment',
             fontsize=10, fontweight='bold', pad=8)
ax.set_ylabel('Experiment', fontsize=LABEL_FS)
ax.set_xlabel('Serialization Format', fontsize=LABEL_FS)
ax.tick_params(labelsize=TICK_FS)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=7)
cbar.set_label('Feasibility', fontsize=LABEL_FS)
fig.tight_layout(pad=0.8)
fig.savefig(os.path.join(out, 'comparison_fig3a_heatmap_gpt.png'), dpi=300)
plt.close()
print('  GPT heatmap saved.')

# ─── Separate DeepSeek Heatmap ───
print("Generating DeepSeek heatmap...")
fig, ax = plt.subplots(figsize=(3.8, 3.2))
sub_ds = feas.loc['deepseek'].reindex(experiments).fillna(0)
sub_ds.index = exp_labels
sns.heatmap(sub_ds, annot=True, fmt='.1%', cmap='Oranges', ax=ax,
            cbar_kws={'label': 'Feasibility', 'shrink': 0.8},
            annot_kws={'fontsize': 8, 'fontweight': 'bold'},
            linewidths=0.5)
ax.set_title('DeepSeek-V4-Flash: Feasibility Heatmap by Format & Experiment',
             fontsize=10, fontweight='bold', pad=8)
ax.set_ylabel('Experiment', fontsize=LABEL_FS)
ax.set_xlabel('Serialization Format', fontsize=LABEL_FS)
ax.tick_params(labelsize=TICK_FS)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=7)
cbar.set_label('Feasibility', fontsize=LABEL_FS)
fig.tight_layout(pad=0.8)
fig.savefig(os.path.join(out, 'comparison_fig3b_heatmap_deepseek.png'), dpi=300)
plt.close()
print('  DeepSeek heatmap saved.')

print(f'\nSeparate heatmaps saved to:\n  {out}')
