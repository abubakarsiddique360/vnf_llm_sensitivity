"""
Generate comparison figures: feasibility overview, format sensitivity, heatmaps, radar.
IEEE-style: compact figure size, large readable fonts, proper axis labels.
"""
import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
})

df = pd.read_csv(r'E:\vnf_llm_sensitivity\Experiment\data\parsed\metrics.csv')
GPT = '#4C72B0'; DS = '#DD8452'
experiments = ['main_t0','main_t0.5','large_graphs','vm_placement','noise']
exp_labels = ['Deterministic\n(τ = 0)','Stochastic\n(τ = 0.5)','100-node','VM','Noise']
out = r'E:\vnf_llm_sensitivity\Experiment\results\comparison\figures'
os.makedirs(out, exist_ok=True)

# ---- Fig 1: Feasibility Overview ----
print("Fig 1: Feasibility Overview...")
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(experiments)); w = 0.3
gf = [df[(df['model']=='gpt')&(df['experiment']==e)]['feasible'].mean()*100 for e in experiments]
dfv = [df[(df['model']=='deepseek')&(df['experiment']==e)]['feasible'].mean()*100 for e in experiments]
ax.bar(x-w/2, gf, w, label='GPT-5-mini', color=GPT, edgecolor='black', linewidth=0.5)
ax.bar(x+w/2, dfv, w, label='DeepSeek-V4-Flash', color=DS, edgecolor='black', linewidth=0.5)
for i in x:
    ax.text(i-w/2, gf[i]+1, f'{gf[i]:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color=GPT)
    ax.text(i+w/2, dfv[i]+1, f'{dfv[i]:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color=DS)
ax.set_xticks(x); ax.set_xticklabels(exp_labels)
ax.set_xlabel('Experiment Phase', fontsize=12)
ax.set_ylabel('Feasibility Rate (%)', fontsize=12)
ax.set_ylim(0, 100)
ax.set_title('Feasibility Across All Experiments', fontsize=13, fontweight='bold', pad=8)
ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(pad=0.8)
fig.savefig(os.path.join(out, 'comparison_fig1_feasibility_overview.png'), dpi=300); plt.close()
print('  Saved.')

# ---- Fig 2: Format Sensitivity ----
print("Fig 2: Format Sensitivity...")
fig, ax = plt.subplots(figsize=(7, 3.8))
max_v = 0
for model, label, color in [('gpt','GPT-5-mini',GPT),('deepseek','DeepSeek-V4-Flash',DS)]:
    vals = []
    for e in experiments:
        sub = df[(df['model']==model)&(df['experiment']==e)]
        s = sub.groupby('problem_id')['optimality_gap'].std().mean()
        v = s if not pd.isna(s) else 0
        vals.append(v)
        max_v = max(max_v, v)
    ax.plot(exp_labels, vals, marker='o', label=label, color=color, linewidth=2.5, markersize=10)
    for i, v in enumerate(vals):
        ax.text(i, v+0.3, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold', color=color)
ax.set_xlabel('Experiment Phase', fontsize=12)
ax.set_ylabel('Mean Format Sensitivity Score', fontsize=12)
ax.set_title('Format Sensitivity Across Experiments', fontsize=13, fontweight='bold', pad=8)
ax.set_ylim(bottom=0, top=max_v + 1.2)
ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(pad=0.8)
fig.savefig(os.path.join(out, 'comparison_fig2_sensitivity_comparison.png'), dpi=300); plt.close()
print('  Saved.')

# ---- Fig 3a: GPT Heatmap ----
print("Fig 3a: GPT Heatmap...")
feas = df.groupby(['model','experiment','format'])['feasible'].mean().unstack('format')
fig, ax = plt.subplots(figsize=(5, 3.2))
sub = feas.loc['gpt'].reindex(experiments).fillna(0)
sub.index = ['Deterministic\n(τ = 0)','Stochastic\n(τ = 0.5)','100-node','VM','Noise']
sns.heatmap(sub, annot=True, fmt='.1%', cmap='Blues', ax=ax,
            cbar_kws={'label':'Feasibility','shrink':0.8},
            annot_kws={'fontsize':10,'fontweight':'bold'}, linewidths=0.5)
ax.set_title('GPT-5-mini', fontsize=12, fontweight='bold')
ax.set_xlabel('Format', fontsize=11); ax.set_ylabel('Experiment', fontsize=11)
ax.tick_params(labelsize=10)
cbar = ax.collections[0].colorbar; cbar.ax.tick_params(labelsize=10)
cbar.set_label('Feasibility', fontsize=11)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(out, 'comparison_fig3a_heatmap_gpt.png'), dpi=300); plt.close()
print('  Saved.')

# ---- Fig 3b: DeepSeek Heatmap ----
print("Fig 3b: DeepSeek Heatmap...")
fig, ax = plt.subplots(figsize=(5, 3.2))
sub = feas.loc['deepseek'].reindex(experiments).fillna(0)
sub.index = ['Deterministic\n(τ = 0)','Stochastic\n(τ = 0.5)','100-node','VM','Noise']
sns.heatmap(sub, annot=True, fmt='.1%', cmap='Oranges', ax=ax,
            cbar_kws={'label':'Feasibility','shrink':0.8},
            annot_kws={'fontsize':10,'fontweight':'bold'}, linewidths=0.5)
ax.set_title('DeepSeek-V4-Flash', fontsize=12, fontweight='bold')
ax.set_xlabel('Format', fontsize=11); ax.set_ylabel('Experiment', fontsize=11)
ax.tick_params(labelsize=10)
cbar = ax.collections[0].colorbar; cbar.ax.tick_params(labelsize=10)
cbar.set_label('Feasibility', fontsize=11)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(out, 'comparison_fig3b_heatmap_deepseek.png'), dpi=300); plt.close()
print('  Saved.')

# ---- Fig 4: Radar ----
print("Fig 4: Radar...")
metrics = ['Deterministic (τ = 0)\nfeas.','Stochastic (τ = 0.5)\nfeas.','large graphs\nfeas.','VM placement\nfeas.','Noise\nfeas.','Format\nsensitivity']
gv = [df[(df['model']=='gpt')&(df['experiment']==e)]['feasible'].mean()*100 for e in experiments]
dv = [df[(df['model']=='deepseek')&(df['experiment']==e)]['feasible'].mean()*100 for e in experiments]
sg = df[df['model']=='gpt'].groupby('problem_id')['optimality_gap'].std().mean()
sd_ = df[df['model']=='deepseek'].groupby('problem_id')['optimality_gap'].std().mean()
mx = max(sg,sd_,1)
gv.append(max(0,100-(sg/mx)*100)); dv.append(max(0,100-(sd_/mx)*100))
N=len(metrics); ang=[n/float(N)*2*pi for n in range(N)]
gc=gv+gv[:1]; dc=dv+dv[:1]; ac=ang+ang[:1]

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
ax.fill(ac,gc,alpha=0.15,color=GPT); ax.plot(ac,gc,'o-',linewidth=2.5,markersize=10,label='GPT-5-mini',color=GPT)
ax.fill(ac,dc,alpha=0.15,color=DS); ax.plot(ac,dc,'o-',linewidth=2.5,markersize=10,label='DeepSeek-V4-Flash',color=DS)
ax.set_xticks(ac[:-1]); ax.set_xticklabels(metrics,fontsize=12,fontweight='bold')
ax.set_title('Performance Radar: DeepSeek vs GPT',fontsize=13,fontweight='bold',pad=25)
ax.legend(loc='upper center',fontsize=11,bbox_to_anchor=(0.5, -0.06),ncol=2)
fig.tight_layout(pad=2.0)
fig.savefig(os.path.join(out,'comparison_fig4_radar_comparison.png'),dpi=300); plt.close()
print('  Saved.')

print(f'\nAll 4 comparison figures regenerated in {out}')
