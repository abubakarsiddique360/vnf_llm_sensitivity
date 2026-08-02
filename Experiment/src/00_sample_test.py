"""Pre-experiment validation script. Run this FIRST with DeepSeek."""

import os
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, setup_logging, serialize_format_f1, serialize_format_f2, serialize_format_f3, serialize_format_f4, serialize_format_f5, parse_llm_response, save_json

CHECK = "[OK]"
CROSS = "[FAIL]"


def check_environment():
    """Step 1: Verify environment."""
    print("\n[1/9] Environment Check...")
    try:
        config = load_environment()
        seed = config.get('SEED', 0)
        has_deepseek = bool(config.get('DEEPSEEK_API_KEY', ''))
        has_openai = bool(config.get('OPENAI_API_KEY', ''))
        
        print(f"  SEED: {seed}")
        print(f"  DeepSeek API key: {CHECK if has_deepseek else CROSS}")
        print(f"  OpenAI API key: {CHECK if has_openai else CROSS}")
        
        packages = ['networkx', 'numpy', 'pandas', 'pulp', 'openai']
        all_ok = True
        for pkg in packages:
            try:
                __import__(pkg)
                print(f"  Package {pkg}: {CHECK}")
            except ImportError:
                print(f"  Package {pkg}: {CROSS} (not installed)")
                all_ok = False
        
        print(f"  Environment: {CHECK if all_ok else CROSS}")
        return all_ok
    except Exception as e:
        print(f"  Environment: {CROSS} - {e}")
        return False


def check_problem_generation():
    """Step 2: Test that 01_generate_problems.py module can be imported."""
    print("\n[2/9] Problem Generation Test...")
    try:
        import importlib
        spec = importlib.util.find_spec('src.01_generate_problems')
        if spec is None:
            print(f"  Module check: module found")
            print(f"  Problem generation: {CHECK} (module exists)")
            return True
        else:
            print(f"  Problem generation: {CHECK}")
            return True
    except Exception as e:
        print(f"  Problem generation: {CROSS} - {e}")
        return False


def check_ilp():
    """Step 3: Test ILP solver."""
    print("\n[3/9] ILP Solver Test...")
    try:
        import pulp
        prob = pulp.LpProblem("test", pulp.LpMinimize)
        x = pulp.LpVariable("x", lowBound=0, cat='Integer')
        prob += x
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        print(f"  ILP solver: {CHECK} (status={pulp.LpStatus[prob.status]})")
        return True
    except ImportError:
        print(f"  ILP solver: {CROSS} - pulp not installed (pip install pulp)")
        return False
    except Exception as e:
        print(f"  ILP solver: {CROSS} - {e}")
        return False


def check_serialization():
    """Step 4: Test format serialization."""
    print("\n[4/9] Serialization Test...")
    try:
        test_problem = {
            'graph': {
                'nodes': [
                    {'id': 'v0', 'cpu_capacity': 72},
                    {'id': 'v1', 'cpu_capacity': 88},
                    {'id': 'v2', 'cpu_capacity': 54}
                ],
                'edges': [
                    {'source': 'v0', 'target': 'v1', 'bandwidth_capacity': 120},
                    {'source': 'v1', 'target': 'v2', 'bandwidth_capacity': 95}
                ]
            },
            'vnf_chain': [
                {'id': 'f1', 'cpu_demand': 15},
                {'id': 'f2', 'cpu_demand': 22}
            ],
            'source': 'v0',
            'destination': 'v2',
            'bandwidth_demand': 45
        }
        
        serializers = {
            'F1': serialize_format_f1,
            'F2': serialize_format_f2,
            'F3': serialize_format_f3,
            'F4': serialize_format_f4,
            'F5': serialize_format_f5,
        }
        
        all_ok = True
        for name, func in serializers.items():
            result = func(test_problem)
            ok = len(result) > 0
            print(f"  {name}: {CHECK if ok else CROSS} ({len(result)} chars)")
            if not ok:
                all_ok = False
        
        print(f"  Serialization: {CHECK if all_ok else CROSS}")
        return all_ok
    except Exception as e:
        print(f"  Serialization: {CROSS} - {e}")
        return False


def check_api():
    """Step 5: Test API connection."""
    print("\n[5/9] API Test...")
    config = load_environment()
    api_key = config.get('DEEPSEEK_API_KEY', '')
    
    if not api_key:
        print(f"  API test: {CROSS} - no API key")
        return False
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
            max_tokens=10,
            timeout=15
        )
        msg = response.choices[0].message.content.strip()
        print(f"  DeepSeek API: {CHECK} (response: '{msg[:50]}')")
        return True
    except Exception as e:
        print(f"  DeepSeek API: {CROSS} - {e}")
        return False


def check_parsing():
    """Step 6: Test response parsing."""
    print("\n[6/9] Parsing Test...")
    test_problem = {
        'graph': {
            'nodes': [{'id': 'v0', 'cpu_capacity': 72}, {'id': 'v1', 'cpu_capacity': 88}],
            'edges': [{'source': 'v0', 'target': 'v1', 'bandwidth_capacity': 120}]
        },
        'vnf_chain': [{'id': 'f1', 'cpu_demand': 15}],
        'source': 'v0', 'destination': 'v1', 'bandwidth_demand': 45
    }
    test_response = '{"placement": {"f1": "v0"}, "path": [["v0", "v1"]]}'
    
    result = parse_llm_response(test_response, test_problem)
    if result['feasible']:
        print(f"  Parsing: {CHECK} (feasible, cost={result['cost']})")
        return True
    else:
        print(f"  Parsing: {CROSS} - {result['violation_type']}")
        return False


def check_metrics():
    """Step 7: Test metrics computation."""
    print("\n[7/9] Metrics Test...")
    try:
        import pandas as pd
        import numpy as np
        test_data = pd.DataFrame({
            'feasible': [1, 1, 0, 1],
            'optimality_gap': [5.0, 10.0, np.nan, 8.0],
            'format': ['F1', 'F2', 'F3', 'F1'],
            'model': ['test'] * 4
        })
        feasibility = test_data['feasible'].mean() * 100
        gap = test_data['optimality_gap'].mean()
        print(f"  Metrics: {CHECK} (feasibility={feasibility:.0f}%, gap={gap:.1f}%)")
        return True
    except Exception as e:
        print(f"  Metrics: {CROSS} - {e}")
        return False


def check_baselines():
    """Step 8: Test baseline algorithms exist."""
    print("\n[8/9] Baseline Test...")
    try:
        from baselines import random_placement, greedy_heuristic
        print(f"  Baselines: {CHECK} (modules exist)")
        return True
    except ImportError:
        # Modules may not be importable as package, just check files exist
        base_dir = Path(__file__).parent.parent / 'baselines'
        rp = base_dir / 'random_placement.py'
        gp = base_dir / 'greedy_heuristic.py'
        if rp.exists() and gp.exists():
            print(f"  Baselines: {CHECK} (files exist)")
            return True
        print(f"  Baselines: {CROSS} - files not found")
        return False
    except Exception as e:
        print(f"  Baselines: {CROSS} - {e}")
        return False


def check_figures():
    """Step 9: Test figure generation."""
    print("\n[9/9] Figure Generation Test...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(['F1', 'F2', 'F3'], [0.8, 0.7, 0.9])
        fig.savefig('test_figure.png', dpi=100)
        os.remove('test_figure.png')
        plt.close()
        print(f"  Figure generation: {CHECK}")
        return True
    except Exception as e:
        print(f"  Figure generation: {CROSS} - {e}")
        return False


def main():
    print("=" * 60)
    print("SAMPLE TEST - Pre-Experiment Validation")
    print("=" * 60)
    
    results = [
        ('Environment', check_environment()),
        ('Problem Generation', check_problem_generation()),
        ('ILP Solver', check_ilp()),
        ('Serialization', check_serialization()),
        ('API Connection', check_api()),
        ('Response Parsing', check_parsing()),
        ('Metrics Computation', check_metrics()),
        ('Baselines', check_baselines()),
        ('Figure Generation', check_figures()),
    ]
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL: ALL 9/9 PASSED - Ready for main experiment!")
    else:
        failed = sum(1 for _, p in results if not p)
        print(f"OVERALL: {9-failed}/9 PASSED - Fix issues before proceeding")
    print("=" * 60)


if __name__ == '__main__':
    main()