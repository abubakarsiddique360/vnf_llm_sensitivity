"""Generate CSV tables for GPT-5-mini per experiment section (like DeepSeek)."""
import os, sys, argparse, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/parsed/metrics.csv')
    parser.add_argument('--output', default='results/gpt/tables')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    gpt = df[df['model'] == 'gpt']
    if len(gpt) == 0: print("No GPT data"); return
    
    experiments = [
        ('large_graphs', 'Large Graphs (100-node)'),
        ('vm_placement', 'VM Placement'),
        ('noise', 'Noise'),
        ('main_t0', 'Main T=0'),
        ('main_t0.5', 'T=0.5 (Temperature Sensitivity)'),
    ]
    
    for exp_name, exp_label in experiments:
        edf = gpt[gpt['experiment'] == exp_name]
        if len(edf) == 0:
            print(f"No data for {exp_name}")
            continue
        out = os.path.join(args.output, exp_name)
        os.makedirs(out, exist_ok=True)
        print(f"Generating tables for: {exp_label} ({len(edf)} responses)")
        
        # Table 1: Format sensitivity
        t1 = edf.groupby('format').agg(
            feasibility_rate=('feasible', lambda x: f"{x.mean()*100:.1f}%"),
            mean_gap=('optimality_gap', lambda x: f"{x.mean():.2f}"),
            median_gap=('optimality_gap', lambda x: f"{x.median():.2f}"),
            std_gap=('optimality_gap', lambda x: f"{x.std():.2f}"),
            count=('feasible', 'count')
        ).reset_index()
        t1.to_csv(os.path.join(out, 'table1_format_sensitivity.csv'), index=False)
        
        # Table 2: Graph family breakdown
        if 'family' in edf.columns:
            t2 = edf.groupby(['format', 'family']).agg(
                feasible_rate=('feasible', lambda x: f"{x.mean()*100:.1f}%"),
                count=('feasible', 'count')
            ).reset_index()
            t2.to_csv(os.path.join(out, 'table2_by_family.csv'), index=False)
        
        # Table 3: Size category breakdown
        if 'size_category' in edf.columns:
            t3 = edf.groupby(['format', 'size_category']).agg(
                feasible_rate=('feasible', lambda x: f"{x.mean()*100:.1f}%"),
                count=('feasible', 'count')
            ).reset_index()
            t3.to_csv(os.path.join(out, 'table3_by_size.csv'), index=False)
        
        print(f"  Tables saved to {out}")
    
    print("All GPT tables saved")

if __name__ == '__main__':
    main()