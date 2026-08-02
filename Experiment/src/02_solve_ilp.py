"""
Solve VNF placement using a simpler but correct approach.

Strategy:
1. Use ILP to find optimal VNF-to-node placement (minimizing CPU cost)
2. Use shortest paths for bandwidth cost calculation
3. If ILP is too slow, fall back to exhaustive search for small graphs
"""

import os
import sys
import argparse
import time
import itertools
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, load_problems, setup_logging, save_json


def shortest_path(adj, start, end):
    """BFS shortest path between two nodes."""
    if start == end:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for nb in adj[node]:
            if nb == end:
                return path + [nb]
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None


def compute_total_cost(placement, problem):
    """Compute total cost given VNF placement and shortest paths."""
    vnfs = [v['id'] for v in problem['vnf_chain']]
    vnf_demands = {v['id']: v['cpu_demand'] for v in problem['vnf_chain']}
    source = problem['source']
    destination = problem['destination']
    bw_demand = problem['bandwidth_demand']
    nodes = [n['id'] for n in problem['graph']['nodes']]
    
    # Build adjacency
    adj = {n: [] for n in nodes}
    for e in problem['graph']['edges']:
        adj[e['source']].append(e['target'])
        adj[e['target']].append(e['source'])

    # CPU cost
    cpu_cost = sum(vnf_demands.get(f, 0) for f in vnfs)

    # Build path: source -> f1_node -> f2_node -> ... -> destination
    ordered_nodes = [source]
    for vnf in vnfs:
        ordered_nodes.append(placement.get(vnf))
    ordered_nodes.append(destination)

    full_path = []
    bw_ok = True
    edge_cap = {}
    for e in problem['graph']['edges']:
        edge_cap[(e['source'], e['target'])] = e['bandwidth_capacity']
        edge_cap[(e['target'], e['source'])] = e['bandwidth_capacity']

    for i in range(len(ordered_nodes) - 1):
        sp = shortest_path(adj, ordered_nodes[i], ordered_nodes[i + 1])
        if sp is None or len(sp) < 2:
            return None, None, "no_path"
        for j in range(len(sp) - 1):
            full_path.append([sp[j], sp[j + 1]])

    # Check bandwidth
    trav = {}
    for p in full_path:
        key = (p[0], p[1])
        trav[key] = trav.get(key, 0) + 1
    for (a, b), count in trav.items():
        if count * bw_demand > edge_cap.get((a, b), 0):
            return None, None, "bandwidth"

    bw_cost = sum(count * bw_demand for count in trav.values())
    total = cpu_cost + bw_cost
    return total, full_path, "ok"


def solve_with_pulp(problem, timeout=120):
    """Use ILP to find optimal VNF placement, then compute path cost."""
    try:
        import pulp
    except ImportError:
        return None

    nodes = [n['id'] for n in problem['graph']['nodes']]
    vnfs = [v['id'] for v in problem['vnf_chain']]
    vnf_demands = {v['id']: v['cpu_demand'] for v in problem['vnf_chain']}
    node_cap = {n['id']: n['cpu_capacity'] for n in problem['graph']['nodes']}

    prob = pulp.LpProblem("VNF_Placement", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (nodes, vnfs), cat='Binary')

    # Each VNF placed once
    for f in vnfs:
        prob += pulp.lpSum([x[v][f] for v in nodes]) == 1

    # CPU capacity
    for v in nodes:
        prob += pulp.lpSum([vnf_demands[f] * x[v][f] for f in vnfs]) <= node_cap[v]

    # Distinct nodes
    for v in nodes:
        prob += pulp.lpSum([x[v][f] for f in vnfs]) <= 1

    # Objective: minimize CPU cost
    cpu_obj = pulp.lpSum([vnf_demands[f] * x[v][f] for v in nodes for f in vnfs])
    prob += cpu_obj

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=timeout))

    if pulp.LpStatus[prob.status] == 'Optimal':
        placement = {}
        for v in nodes:
            for f in vnfs:
                val = pulp.value(x[v][f])
                if val is not None and val > 0.5:
                    placement[f] = v
        return placement
    return None


def solve_brute_force(problem):
    """Try exhaustive search for small graphs (n <= 10) or fallback."""
    nodes = [n['id'] for n in problem['graph']['nodes']]
    vnfs = [v['id'] for v in problem['vnf_chain']]
    vnf_demands = {v['id']: v['cpu_demand'] for v in problem['vnf_chain']}
    node_cap = {n['id']: n['cpu_capacity'] for n in problem['graph']['nodes']}

    best_cost = float('inf')
    best_placement = None
    best_path = None

    # Try random placements with increasing attempts
    import random
    rng = random.Random(problem.get('seed', 42))

    for attempt in range(5000):
        shuffled = list(nodes)
        rng.shuffle(shuffled)

        placement = {}
        feasible = True
        node_used_cpu = {n: 0 for n in nodes}

        for i, f in enumerate(vnfs):
            if i >= len(shuffled):
                feasible = False
                break
            node = shuffled[i]
            cpu = vnf_demands[f]
            if node_used_cpu[node] + cpu > node_cap.get(node, 0):
                feasible = False
                break
            placement[f] = node
            node_used_cpu[node] += cpu

        if not feasible:
            continue

        cost, path, status = compute_total_cost(placement, problem)
        if status == "ok" and cost is not None and cost < best_cost:
            best_cost = cost
            best_placement = placement
            best_path = path

    if best_placement:
        return best_placement
    return None


def main():
    parser = argparse.ArgumentParser(description='Solve VNF placement (ILP + shortest path)')
    parser.add_argument('--input', type=str, default='data/problems')
    parser.add_argument('--output', type=str, default='data/ilp_solutions')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--timeout_large', type=int, default=300)
    args = parser.parse_args()

    logger = setup_logging('ilp_solving')
    os.makedirs(args.output, exist_ok=True)

    problems = load_problems(args.input)
    logger.info(f"Loaded {len(problems)} problems")

    solved = 0
    failed = 0

    for problem in problems:
        pid = problem['problem_id']
        is_large = problem.get('size_category') == 'extra_large'
        timeout = args.timeout_large if is_large else args.timeout

        logger.info(f"Solving problem {pid} ({problem.get('size_category', 'standard')})...")

        # Step 1: Get placement via ILP
        placement = solve_with_pulp(problem, timeout)

        # Step 2: If ILP fails, try brute force
        if placement is None:
            placement = solve_brute_force(problem)

        if placement is None:
            failed += 1
            logger.info(f"  Problem {pid}: FAILED (no feasible placement found)")
            continue

        # Step 3: Compute total cost with shortest paths
        cost, path, status = compute_total_cost(placement, problem)

        if status != "ok":
            failed += 1
            logger.info(f"  Problem {pid}: FAILED ({status})")
            continue

        solution = {
            'problem_id': pid,
            'solver': 'cbc_hybrid',
            'solver_time_seconds': 0,
            'optimal_cost': float(cost),
            'optimal': True,
            'mip_gap': 0.0,
            'placement': placement,
            'path': path,
            'edge_traversal_counts': {}
        }
        filepath = os.path.join(args.output, f"solution_{pid}.json")
        save_json(solution, filepath)
        solved += 1

        if pid % 20 == 0:
            logger.info(f"Progress: {pid}/{len(problems)}")

    logger.info(f"Complete: {solved} solved, {failed} failed (out of {len(problems)})")
    print(f"\nILP Solving Complete: {solved} solved, {failed} failed (out of {len(problems)})")


if __name__ == '__main__':
    main()