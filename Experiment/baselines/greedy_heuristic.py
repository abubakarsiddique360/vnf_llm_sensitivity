"""Greedy heuristic baseline algorithm."""

import os
import sys
import csv
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, load_problems, load_ilp_solutions, compute_cost


def greedy_placement(problem):
    """Greedy placement: for each VNF, choose closest node with highest residual CPU."""
    nodes = [n['id'] for n in problem['graph']['nodes']]
    vnfs = [v['id'] for v in problem['vnf_chain']]
    source = problem['source']
    destination = problem['destination']
    node_cap = {n['id']: n['cpu_capacity'] for n in problem['graph']['nodes']}

    # Build adjacency
    adj = {n: [] for n in nodes}
    for e in problem['graph']['edges']:
        adj[e['source']].append(e['target'])
        adj[e['target']].append(e['source'])

    # BFS for shortest path length
    def shortest_path_len(start, end):
        from collections import deque
        if start == end:
            return 0, [start]
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            for nb in adj[node]:
                if nb == end:
                    return len(path), path + [nb]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return float('inf'), []

    placement = {}
    current = source
    used_cpu = {n: 0 for n in nodes}
    feasible = True

    for vnf in vnfs:
        cpu = next((v['cpu_demand'] for v in problem['vnf_chain'] if v['id'] == vnf), 0)

        # Score each node: prefer close + high residual CPU
        best_node = None
        best_score = float('-inf')

        for node in nodes:
            if node in placement.values():
                continue
            dist, _ = shortest_path_len(current, node)
            if dist == float('inf'):
                continue
            residual = node_cap[node] - used_cpu[node]
            if residual < cpu:
                continue
            score = residual - dist * 10  # trade-off
            if score > best_score:
                best_score = score
                best_node = node

        if best_node is None:
            feasible = False
            break

        placement[vnf] = best_node
        used_cpu[best_node] += cpu
        current = best_node

    if not feasible:
        return None, None, None

    # Build path
    path = []
    current = source
    for vnf in vnfs:
        node = placement[vnf]
        _, subpath = shortest_path_len(current, node)
        if not subpath:
            return None, None, None
        for i in range(len(subpath) - 1):
            path.append([subpath[i], subpath[i + 1]])
        current = node
    _, final_path = shortest_path_len(current, destination)
    if not final_path:
        return None, None, None
    for i in range(len(final_path) - 1):
        path.append([final_path[i], final_path[i + 1]])

    # Check bandwidth
    bw_demand = problem['bandwidth_demand']
    edges = {}
    for e in problem['graph']['edges']:
        edges[(e['source'], e['target'])] = e['bandwidth_capacity']
        edges[(e['target'], e['source'])] = e['bandwidth_capacity']
    trav = {}
    for p in path:
        key = (p[0], p[1])
        trav[key] = trav.get(key, 0) + 1
    for (a, b), count in trav.items():
        if count * bw_demand > edges.get((a, b), 0):
            return None, None, None

    cost = compute_cost(placement, path, problem)
    return placement, path, cost


def main():
    parser = argparse.ArgumentParser(description='Greedy heuristic baseline')
    parser.add_argument('--problems', type=str, default=None)
    parser.add_argument('--ilp', type=str, default=None)
    parser.add_argument('--output', type=str, default='data/baselines')
    args = parser.parse_args()

    config = load_environment()
    problems = load_problems(args.problems or config['PROBLEMS_DIR'])
    ilp_solutions = load_ilp_solutions(args.ilp or config['ILP_SOLUTIONS_DIR'])

    # Only evaluate the 120 main problems (IDs 1-120) used in the LLM comparison
    problems = [p for p in problems if 1 <= int(p['problem_id']) <= 120]

    os.makedirs(args.output, exist_ok=True)
    results = []

    for problem in problems:
        pid = problem['problem_id']
        placement, path, cost = greedy_placement(problem)
        ilp = ilp_solutions.get(pid, {})
        optimal = ilp.get('optimal_cost', None)
        gap = ((cost - optimal) / optimal * 100) if optimal and cost else None

        results.append({
            'problem_id': pid,
            'feasible': 1 if placement else 0,
            'cost': cost if cost else None,
            'optimality_gap': gap
        })

        if pid % 50 == 0:
            print(f"  Greedy baseline: {pid}/{len(problems)}")

    with open(os.path.join(args.output, 'greedy_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['problem_id', 'feasible', 'cost', 'optimality_gap'])
        writer.writeheader()
        writer.writerows(results)

    feasible = sum(1 for r in results if r['feasible'])
    print(f"Greedy baseline complete: {feasible}/{len(results)} feasible")


if __name__ == '__main__':
    main()