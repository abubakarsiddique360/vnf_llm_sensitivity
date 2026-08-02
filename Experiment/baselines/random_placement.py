"""Random placement baseline algorithm."""

import os
import sys
import csv
import argparse
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, load_problems, load_ilp_solutions, compute_cost, setup_logging


def random_placement(problem, max_attempts=100):
    """Try random placements and return the best feasible one."""
    nodes = [n['id'] for n in problem['graph']['nodes']]
    vnfs = [v['id'] for v in problem['vnf_chain']]
    source = problem['source']
    destination = problem['destination']
    bw_demand = problem['bandwidth_demand']

    best_cost = float('inf')
    best_placement = None
    best_path = None

    for _ in range(max_attempts):
        shuffled = list(nodes)
        random.shuffle(shuffled)
        placement = {}
        for i, vnf in enumerate(vnfs):
            if i < len(shuffled):
                placement[vnf] = shuffled[i]

        # Check distinctness
        if len(set(placement.values())) != len(vnfs):
            continue

        # Check CPU capacity
        node_cpu = {n: 0 for n in nodes}
        for vnf, node in placement.items():
            cpu = next((v['cpu_demand'] for v in problem['vnf_chain'] if v['id'] == vnf), 0)
            node_cpu[node] += cpu
        node_cap = {n['id']: n['cpu_capacity'] for n in problem['graph']['nodes']}
        if any(node_cpu[n] > node_cap.get(n, 0) for n in nodes):
            continue

        # Build path visiting VNF nodes in order
        path = []
        current = source
        ordered_nodes = [placement[v] for v in vnfs] + [destination]

        # Build edge lookup
        edges = {}
        for e in problem['graph']['edges']:
            edges[(e['source'], e['target'])] = e['bandwidth_capacity']
            edges[(e['target'], e['source'])] = e['bandwidth_capacity']

        success = True
        for next_node in ordered_nodes:
            if current == next_node:
                continue
            # Find shortest path (simple BFS)
            from collections import deque
            queue = deque([(current, [current])])
            visited = {current}
            found = False
            while queue:
                node, path_so_far = queue.popleft()
                if node == next_node:
                    for i in range(len(path_so_far) - 1):
                        path.append([path_so_far[i], path_so_far[i + 1]])
                    current = next_node
                    found = True
                    break
                for e in problem['graph']['edges']:
                    for neighbor in [e['source'], e['target']]:
                        if neighbor != node:
                            continue
                        other = e['target'] if e['source'] == node else e['source']
                        if other not in visited:
                            visited.add(other)
                            queue.append((other, path_so_far + [other]))
                if not found:
                    success = False
                    break

        if not success:
            continue

        # Check bandwidth
        trav_count = {}
        for p in path:
            key = (p[0], p[1])
            trav_count[key] = trav_count.get(key, 0) + 1
        bw_ok = True
        for (a, b), count in trav_count.items():
            cap = edges.get((a, b), 0)
            if count * bw_demand > cap:
                bw_ok = False
                break
        if not bw_ok:
            continue

        cost = compute_cost(placement, path, problem)
        if cost < best_cost:
            best_cost = cost
            best_placement = placement
            best_path = path

    return best_placement, best_path, best_cost


def main():
    parser = argparse.ArgumentParser(description='Random placement baseline')
    parser.add_argument('--problems', type=str, default=None)
    parser.add_argument('--ilp', type=str, default=None)
    parser.add_argument('--output', type=str, default='data/baselines')
    parser.add_argument('--trials', type=int, default=100)
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
        placement, path, cost = random_placement(problem, args.trials)
        ilp = ilp_solutions.get(pid, {})
        optimal = ilp.get('optimal_cost', None)
        gap = ((cost - optimal) / optimal * 100) if optimal and cost < float('inf') else None

        results.append({
            'problem_id': pid,
            'feasible': 1 if placement else 0,
            'cost': cost if cost < float('inf') else None,
            'optimality_gap': gap
        })

        if pid % 50 == 0:
            print(f"  Random baseline: {pid}/{len(problems)}")

    # Save results
    with open(os.path.join(args.output, 'random_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['problem_id', 'feasible', 'cost', 'optimality_gap'])
        writer.writeheader()
        writer.writerows(results)

    feasible = sum(1 for r in results if r['feasible'])
    print(f"Random baseline complete: {feasible}/{len(results)} feasible")


if __name__ == '__main__':
    main()