"""Generate PNG figures for GPT-5-mini per experiment section (like DeepSeek)."""
import os, sys, argparse, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/parsed/metrics.csv')
    parser.add_argument('--output', default='results/gpt/figures')
    args = parser.parse_args()
    
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = pd.read_csv(args.input)
    gpt = df[df['model'] == 'gpt']
    if len(gpt) == 0: print("No GPT data"); return
    
    COLOR = '#4C72B0'
    experiments = [
        ('large_graphs', '100-node Graphs'),
        ('vm_placement', 'VM Placement'),
        ('noise', 'Noise'),
        ('main_t0', 'Main T=0'),
        ('main_t0.5', 'T=0.5 (Temperature Sensitivity)'),
    ]
    
    total_files = 0
    for exp_name, exp_label in experiments:
        edf = gpt[gpt['experiment'] == exp_name]
        if len(edf) == 0:
            print(f"No data for {exp_name}")
            continue
        out = os.path.join(args.output, exp_name)
        os.makedirs(out, exist_ok=True)
        
        print(f"Generating figures for: {exp_label} ({len(edf)} responses)")
        
        # Figure 1: Feasibility by format
        plt.figure(figsize=(8, 5))
        edf.groupby('format')['feasible'].mean().mul(100).plot(kind='bar', color=COLOR, edgecolor='black')
        plt.title(f'GPT-5-mini: Feasibility Rate by Format - {exp_label}')
        plt.ylabel('Feasibility Rate (%)'); plt.xlabel('Format'); plt.ylim(0, 100); plt.grid(axis='y', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, 'fig1_feasibility_by_format.png'), dpi=300); plt.close()
        
        # Figure 2: Optimality gap by format
        plt.figure(figsize=(8, 5))
        fdf = edf[edf['feasible'] == 1]
        if len(fdf) > 0:
            fdf.boxplot(column='optimality_gap', by='format')
            plt.title(f'GPT-5-mini: Optimality Gap by Format - {exp_label}')
            plt.ylabel('Optimality Gap (%)'); plt.xlabel('Format'); plt.grid(axis='y', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, 'fig2_optimality_gap_by_format.png'), dpi=300); plt.close()
        
        # Figure 3: Sensitivity by family
        plt.figure(figsize=(8, 5))
        sens = edf.groupby(['problem_id', 'family'])['optimality_gap'].std().reset_index()
        sens.groupby('family')['optimality_gap'].mean().plot(kind='bar', color=COLOR, edgecolor='black')
        plt.title(f'GPT-5-mini: Format Sensitivity by Family - {exp_label}')
        plt.ylabel('Sensitivity Score'); plt.xlabel('Graph Family'); plt.grid(axis='y', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, 'fig3_sensitivity_by_family.png'), dpi=300); plt.close()
        
        # Figure 4: Confusion matrix
        plt.figure(figsize=(8, 6))
        infeas = edf[edf['feasible'] == 0]
        if len(infeas) > 0:
            ct = pd.crosstab(infeas['format'], infeas['violation_type'])
            sns.heatmap(ct, annot=True, fmt='d', cmap='Reds')
            plt.title(f'GPT-5-mini: Violation Types by Format - {exp_label}')
        plt.tight_layout(); plt.savefig(os.path.join(out, 'fig4_confusion_matrix.png'), dpi=300); plt.close()
        
        # Figure 5: Rank agreement
        plt.figure(figsize=(8, 5))
        ranks = edf.groupby(['problem_id', 'format'])['optimality_gap'].mean().reset_index()
        ranks['rank'] = ranks.groupby('problem_id')['optimality_gap'].rank()
        ranks.groupby('format')['rank'].mean().plot(kind='bar', color=COLOR, edgecolor='black')
        plt.title(f'GPT-5-mini: Average Format Rank (1=best) - {exp_label}')
        plt.ylabel('Average Rank'); plt.xlabel('Format'); plt.grid(axis='y', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, 'fig5_rank_agreement.png'), dpi=300); plt.close()
        
        total_files += 1
        print(f"  Figures saved to {out}")
    
    print(f"\nAll GPT figures saved ({total_files} experiment folders)")

if __name__ == '__main__':
    main()