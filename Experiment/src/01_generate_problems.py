"""Generate network topology problems. Guarantees connected graphs."""

import os
import sys
import argparse
import random
import networkx as nx
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, setup_logging, save_json


def _ensure_connected(G):
    """If G is not connected, add edges to make it connected."""
    if nx.is_connected(G):
        return G
    G = nx.Graph(G)
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        c1 = list(components[i])
        c2 = list(components[i + 1])
        G.add_edge(c1[0], c2[0])
    return G


def generate_standard_topologies(seed, num_problems=120):
    problems = []
    rng = random.Random(seed)

    families, sizes = [], []

    for _ in range(25): families.append('waxman'); sizes.append(rng.randint(15, 20))
    for _ in range(13): families.append('erdos_renyi'); sizes.append(rng.randint(15, 20))
    for _ in range(12): families.append('barabasi_albert'); sizes.append(rng.randint(15, 20))

    for _ in range(50): families.append('waxman'); sizes.append(rng.randint(21, 35))
    for _ in range(25): families.append('erdos_renyi'); sizes.append(rng.randint(21, 35))
    for _ in range(25): families.append('barabasi_albert'); sizes.append(rng.randint(21, 35))

    for _ in range(25): families.append('waxman'); sizes.append(rng.randint(36, 50))
    for _ in range(13): families.append('erdos_renyi'); sizes.append(rng.randint(36, 50))
    for _ in range(12): families.append('barabasi_albert'); sizes.append(rng.randint(36, 50))

    combined = list(zip(families, sizes))
    rng.shuffle(combined)
    families, sizes = zip(*combined)

    for problem_id in range(1, num_problems + 1):
        family = families[problem_id - 1]
        n = sizes[problem_id - 1]
        problem_seed = seed + problem_id * 1000
        local_rng = random.Random(problem_seed)

        # Generate graph (guaranteed connected)
        if family == 'waxman':
            G = nx.waxman_graph(n, alpha=0.4, beta=0.1, seed=problem_seed)
        elif family == 'erdos_renyi':
            G = nx.gnp_random_graph(n, p=0.4, seed=problem_seed)  # Higher p for connectivity
        elif family == 'barabasi_albert':
            G = nx.barabasi_albert_graph(n, m=3, seed=problem_seed)
        else:
            continue

        G = _ensure_connected(G)

        nodes = [{'id': f'v{i}', 'cpu_capacity': local_rng.randint(50, 100)} for i in range(n)]
        edges = [{'source': f'v{u}', 'target': f'v{v}', 'bandwidth_capacity': local_rng.randint(100, 200)}
                 for u, v in G.edges()]

        k = local_rng.choice([4, 5, 6, 7, 8])
        vnf_chain = [{'id': f'f{i}', 'cpu_demand': local_rng.randint(10, 30)} for i in range(1, k + 1)]

        all_nodes = list(range(n))
        source_idx = local_rng.choice(all_nodes)
        dest_idx = local_rng.choice([n for n in all_nodes if n != source_idx])
        size_cat = 'small' if n <= 20 else ('medium' if n <= 35 else 'large')

        problems.append({
            'problem_id': problem_id, 'seed': problem_seed, 'family': family,
            'size_category': size_cat, 'num_nodes': n, 'num_edges': len(edges),
            'graph': {'nodes': nodes, 'edges': edges},
            'vnf_chain': vnf_chain, 'source': f'v{source_idx}',
            'destination': f'v{dest_idx}', 'bandwidth_demand': local_rng.randint(20, 80)
        })

    return problems


def generate_large_topologies(seed, num_problems=30):
    problems = []
    rng = random.Random(seed + 10000)
    families = (['waxman'] * 17 + ['erdos_renyi'] * 17 + ['barabasi_albert'] * 16)
    rng.shuffle(families)

    for problem_id in range(1, num_problems + 1):
        family = families[problem_id - 1]
        n = 100
        problem_seed = seed + 10000 + problem_id * 1000
        local_rng = random.Random(problem_seed)

        if family == 'waxman':
            G = nx.waxman_graph(n, alpha=0.4, beta=0.1, seed=problem_seed)
        elif family == 'erdos_renyi':
            G = nx.gnp_random_graph(n, p=0.4, seed=problem_seed)
        elif family == 'barabasi_albert':
            G = nx.barabasi_albert_graph(n, m=3, seed=problem_seed)
        else:
            continue

        G = _ensure_connected(G)

        nodes = [{'id': f'v{i}', 'cpu_capacity': local_rng.randint(100, 200)} for i in range(n)]
        edges = [{'source': f'v{u}', 'target': f'v{v}', 'bandwidth_capacity': local_rng.randint(200, 400)}
                 for u, v in G.edges()]

        k = local_rng.choice([4, 5, 6, 7, 8])
        vnf_chain = [{'id': f'f{i}', 'cpu_demand': local_rng.randint(10, 30)} for i in range(1, k + 1)]

        all_nodes = list(range(n))
        source_idx = local_rng.choice(all_nodes)
        dest_idx = local_rng.choice([n for n in all_nodes if n != source_idx])

        problems.append({
            'problem_id': 1000 + problem_id, 'seed': problem_seed, 'family': family,
            'size_category': 'extra_large', 'num_nodes': n, 'num_edges': len(edges),
            'graph': {'nodes': nodes, 'edges': edges},
            'vnf_chain': vnf_chain, 'source': f'v{source_idx}',
            'destination': f'v{dest_idx}', 'bandwidth_demand': local_rng.randint(20, 80)
        })

    return problems


def main():
    parser = argparse.ArgumentParser(description='Generate network topology problems')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='data/problems')
    parser.add_argument('--num_standard', type=int, default=120)
    parser.add_argument('--num_large', type=int, default=30)
    args = parser.parse_args()

    logger = setup_logging('generate_problems')
    os.makedirs(args.output, exist_ok=True)

    standard = generate_standard_topologies(args.seed, args.num_standard)
    large = generate_large_topologies(args.seed, args.num_large)
    all_problems = standard + large

    for problem in all_problems:
        save_json(problem, os.path.join(args.output, f"problem_{problem['problem_id']}.json"))

    small = sum(1 for p in all_problems if p['size_category'] == 'small')
    medium = sum(1 for p in all_problems if p['size_category'] == 'medium')
    large_c = sum(1 for p in all_problems if p['size_category'] == 'large')
    xl = sum(1 for p in all_problems if p['size_category'] == 'extra_large')

    logger.info(f"Total: {len(all_problems)} topologies - Small:{small} Medium:{medium} Large:{large_c} XL:{xl}")
    print(f"Generated {len(all_problems)} topologies: Small:{small} Medium:{medium} Large:{large_c} XL:{xl}")


if __name__ == '__main__':
    main()