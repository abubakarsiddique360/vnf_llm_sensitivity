"""
Updated experiment overview diagram.
Black arrows, black text, clean professional colors, compact layout.
"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = r'E:\vnf_llm_sensitivity\Experiment\results\comparison\figures'
os.makedirs(OUT, exist_ok=True)

def box(ax, x, y, w, h, text, fc='white', ec='black', fs=10, fw='normal', ta='center'):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.06",
                                facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=3))
    ax.text(x, y, text, ha=ta, va='center', fontsize=fs, fontweight=fw, color='black', zorder=4)

def arr(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5), zorder=2)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis('off'); ax.set_facecolor('white')

ax.text(5, 6.25, 'LLM Format Sensitivity in VNF Placement — Experimental Framework',
        ha='center', fontsize=12, fontweight='bold', color='black')

# Level 1: Problems + Formats
box(ax, 1.3, 5.5, 2.2, 0.55, 'VNF Placement\nInstances\n(3 families, 4 sizes)', 
    '#E3F2FD', 'black', 7.5)

box(ax, 5.0, 5.5, 3.5, 0.55, '5 Prompt Formats\n(Same info, different representation)', 
    '#FFF3E0', 'black', 8, 'bold')

arr(ax, 2.4, 5.5, 3.25, 5.5)

# Level 1b: 5 format labels
fmts = [('F1: Edge List', '#FFF3E0'), ('F2: Adj. Matrix', '#FFF3E0'),
        ('F3: Natural Language', '#FFF3E0'), ('F4: JSON', '#FFF3E0'), ('F5: Ranked Neigh.', '#FFF3E0')]
for i, (n, c) in enumerate(fmts):
    box(ax, 2.0+i*1.3, 4.6, 1.1, 0.3, n, c, 'black', 6)

arr(ax, 5.0, 5.2, 5.0, 4.9)

# Level 2: ILP
box(ax, 1.3, 4.0, 2.2, 0.4, 'ILP Optimal Solutions\n(Ground Truth)', '#E8F5E9', 'black', 7)
arr(ax, 1.3, 4.95, 1.3, 4.35)

arr(ax, 5.0, 4.3, 5.0, 3.7)

# Level 3: Two Models
box(ax, 5.0, 3.5, 4.5, 0.4, 'Query Both LLMs (each gets all 5 formats × all problems)', 
    '#F3E5F5', 'black', 7.5, 'bold')

box(ax, 3.2, 2.8, 2.0, 0.5, 'DeepSeek-v4-flash\n(1,750 queries)', '#E3F2FD', 'black', 7)
box(ax, 6.8, 2.8, 2.0, 0.5, 'GPT-5-mini\n(1,750 queries)', '#F3E5F5', 'black', 7)

arr(ax, 5.0, 3.3, 4.2, 3.05)
arr(ax, 5.0, 3.3, 5.8, 3.05)
arr(ax, 4.2, 3.05, 4.2, 2.8)
arr(ax, 5.8, 3.05, 5.8, 2.8)
ax.plot([4.2, 5.8], [3.05, 3.05], color='black', linewidth=1.2, zorder=2)

# Level 4: Five Experiments
arr(ax, 5.0, 2.55, 5.0, 2.25)

box(ax, 5.0, 2.05, 8.8, 0.3, 'Five Experiment Phases (grouped by problem count)', 
    '#F3E5F5', 'black', 7.5, 'bold')

exps = [
    ("Deterministic\n(τ = 0)", '120 prob.\n5 fmts\n600 q', '#C8E6C9'),
    ("Stochastic\n(τ = 0.5)", '120 prob.\n5 fmts\n600 q', '#E1BEE7'),
    ('Large graphs', '30 prob.\n100-node\n150 q', '#FFE0B2'),
    ('Verification', '20 prob.\n5 fmts\n100 q', '#F8BBD0'),
    ('Noise', '20 × 3\nnoise levels\n300 q', '#FFCDD2'),
]
for i, (n, d, c) in enumerate(exps):
    xp = 0.9 + i * 1.8
    box(ax, xp, 1.0, 1.5, 0.95, f'{n}\n\n{d}', c, 'black', 6.5, 'bold')
    arr(ax, xp, 1.7, xp, 1.88)

# Level 5: Baselines + Analysis
box(ax, 1.3, 0.6, 1.8, 0.4, 'Baselines\nGreedy + Random', '#E8F5E9', 'black', 6.5)
arr(ax, 1.3, 4.95, 1.3, 0.8)

box(ax, 6.5, 0.3, 5.5, 0.3, 'Analysis: Parse & Validate  |  Metrics  |  Figures & Tables',
    '#F3E5F5', 'black', 7.5, 'bold')

arr(ax, 5.0, 0.55, 5.5, 0.45)

plt.tight_layout()
fpath = os.path.join(OUT, 'experiment_overview_diagram.png')
plt.savefig(fpath, dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Diagram saved: {fpath}")