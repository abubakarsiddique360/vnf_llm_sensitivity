"""
Shared utility functions for the LLM Graph Representation Sensitivity experiment.

This module contains ALL shared functions used across the experiment pipeline,
visualization, and baseline scripts.
"""

import os
import sys
import json
import re
import logging
import random
import numpy as np
from pathlib import Path
from dotenv import load_dotenv


def load_environment():
    """Load environment variables from .env file and return config dict."""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        print("ERROR: .env file not found. Copy .env.example to .env and add your API keys.")
        print("  Example: copy .env.example .env")
        print("  Then edit .env to add your OpenAI and DeepSeek API keys.")

    config = {
        'SEED': int(os.getenv('SEED', '42')),
        'LLM_TEMPERATURE_PRIMARY': float(os.getenv('LLM_TEMPERATURE_PRIMARY', '0.0')),
        'LLM_TEMPERATURE_SENSITIVITY': float(os.getenv('LLM_TEMPERATURE_SENSITIVITY', '0.5')),
        'LLM_MAX_TOKENS': int(os.getenv('LLM_MAX_TOKENS', '1000')),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
        'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY', ''),
        'PROBLEMS_DIR': os.getenv('PROBLEMS_DIR', 'data/problems'),
        'ILP_SOLUTIONS_DIR': os.getenv('ILP_SOLUTIONS_DIR', 'data/ilp_solutions'),
        'PROMPTS_DIR': os.getenv('PROMPTS_DIR', 'prompts'),
        'LLM_RESPONSES_DIR': os.getenv('LLM_RESPONSES_DIR', 'data/llm_responses'),
        'PARSED_DIR': os.getenv('PARSED_DIR', 'data/parsed'),
        'BASELINES_DIR': os.getenv('BASELINES_DIR', 'data/baselines'),
        'LOGS_DIR': os.getenv('LOGS_DIR', 'logs'),
    }

    experiment_root = Path(__file__).parent.parent
    for key in ['PROBLEMS_DIR', 'ILP_SOLUTIONS_DIR', 'PROMPTS_DIR',
                'LLM_RESPONSES_DIR', 'PARSED_DIR', 'BASELINES_DIR', 'LOGS_DIR']:
        config[key] = str(experiment_root / config[key])

    return config


def setup_logging(experiment_name, log_dir=None):
    """Create a logger for the given experiment."""
    config = load_environment()
    if log_dir is None:
        log_dir = config['LOGS_DIR']
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(log_dir, f'{experiment_name}.log'))
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def ensure_directories(paths_dict):
    """Create all directories in the given dict if they don't exist."""
    for name, path in paths_dict.items():
        os.makedirs(path, exist_ok=True)


def load_problems(problems_dir, num_problems=None):
    """Load all problem JSON files from the problems directory."""
    problems = []
    problems_path = Path(problems_dir)
    if not problems_path.exists():
        return problems

    for f in sorted(problems_path.glob('problem_*.json')):
        try:
            with open(f, 'r') as fh:
                problem = json.load(fh)
                problems.append(problem)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {f}: {e}")

    problems.sort(key=lambda p: p.get('problem_id', 0))

    if num_problems is not None:
        return problems[:num_problems]
    return problems


def load_ilp_solutions(ilp_dir):
    """Load all ILP solution JSON files and return dict mapping problem_id to solution."""
    solutions = {}
    ilp_path = Path(ilp_dir)
    if not ilp_path.exists():
        return solutions

    for f in sorted(ilp_path.glob('solution_*.json')):
        try:
            with open(f, 'r') as fh:
                sol = json.load(fh)
                solutions[sol['problem_id']] = sol
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {f}: {e}")

    return solutions


def save_json(data, filepath):
    """Save data as formatted JSON to filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load and return JSON file content. Returns None on failure."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return None


def serialize_format_f1(problem):
    """Serialize problem as Edge List format (F1)."""
    nodes = problem['graph']['nodes']
    edges = problem['graph']['edges']
    node_str = ', '.join([f"{n['id']}(CPU={n['cpu_capacity']})" for n in nodes])
    edge_str = ', '.join([f"({e['source']},{e['target']},bw={e['bandwidth_capacity']})" for e in edges])
    return f"Nodes: {node_str}\nEdges (undirected): {edge_str}"


def serialize_format_f2(problem):
    """Serialize problem as Adjacency Matrix format (F2)."""
    nodes = problem['graph']['nodes']
    edges = problem['graph']['edges']
    n = len(nodes)
    node_ids = [n['id'] for n in nodes]
    
    cap_str = 'Node capacities: ' + ', '.join([f"{n['id']}:{n['cpu_capacity']}" for n in nodes])
    
    # For large graphs (50+ nodes), use sparse format to keep input manageable
    if n >= 50:
        # Sparse adjacency: list non-zero entries only
        edge_lines = []
        for e in edges:
            edge_lines.append(f"{e['source']} -> {e['target']}: bw={e['bandwidth_capacity']}")
        sparse = '; '.join(edge_lines)
        return f"Adjacency matrix (sparse, {len(edges)} edges):\n{sparse}\n\n{cap_str}"

    matrix = [[0] * n for _ in range(n)]
    for e in edges:
        i = node_ids.index(e['source'])
        j = node_ids.index(e['target'])
        matrix[i][j] = e['bandwidth_capacity']
        matrix[j][i] = e['bandwidth_capacity']

    header = '     ' + ' '.join(f'{nid:>6}' for nid in node_ids)
    rows = []
    for i, nid in enumerate(node_ids):
        row_vals = ' '.join(f'{matrix[i][j]:>6}' for j in range(n))
        rows.append(f'{nid:>4} {row_vals}')

    return f"Node order: {', '.join(node_ids)}\nAdjacency matrix (row=source, col=target, value=bandwidth):\n{header}\n" + '\n'.join(rows) + f'\n\n{cap_str}'


def serialize_format_f3(problem):
    """Serialize problem as Natural Language format (F3)."""
    nodes = problem['graph']['nodes']
    edges = problem['graph']['edges']

    sentences = [f"We have a network with {len(nodes)} nodes."]
    for n in nodes:
        sentences.append(f"Node {n['id']} has a CPU capacity of {n['cpu_capacity']} units.")
    for e in edges:
        sentences.append(f"The link between {e['source']} and {e['target']} can carry up to {e['bandwidth_capacity']} units of bandwidth.")

    return ' '.join(sentences)


def serialize_format_f4(problem):
    """Serialize problem as JSON format (F4)."""
    graph = {
        'nodes': [{'id': n['id'], 'cpu_capacity': n['cpu_capacity']} for n in problem['graph']['nodes']],
        'edges': [{'source': e['source'], 'target': e['target'], 'bandwidth_capacity': e['bandwidth_capacity']} for e in problem['graph']['edges']]
    }
    return json.dumps(graph, indent=2)


def serialize_format_f5(problem):
    """Serialize problem as Ranked Neighbor List format (F5)."""
    nodes = problem['graph']['nodes']
    edges = problem['graph']['edges']

    neighbors = {n['id']: [] for n in nodes}
    for e in edges:
        neighbors[e['source']].append((e['target'], e['bandwidth_capacity']))
        neighbors[e['target']].append((e['source'], e['bandwidth_capacity']))

    for nid in neighbors:
        neighbors[nid].sort(key=lambda x: -x[1])

    lines = []
    for n in nodes:
        if neighbors[n['id']]:
            neighbor_str = ', '.join([f"({nb},{bw})" for nb, bw in neighbors[n['id']]])
            lines.append(f"{n['id']} -> {neighbor_str}")
        else:
            lines.append(f"{n['id']} -> (none)")

    cap_str = 'Node capacities: ' + ', '.join([f"{n['id']}:{n['cpu_capacity']}" for n in nodes])
    lines.append(cap_str)
    return '\n'.join(lines)


def build_vnf_chain_string(problem):
    """Build formatted VNF chain string from problem."""
    chain = problem.get('vnf_chain', [])
    return ', '.join([f"{vnf['id']}(CPU={vnf['cpu_demand']})" for vnf in chain])


def build_user_prompt(problem, format_name, template='A'):
    """Build a complete user prompt by combining format serialization with problem data.

    template: 'A' or 'B' - which prompt template to use
    """
    serializers = {
        'F1': serialize_format_f1,
        'F2': serialize_format_f2,
        'F3': serialize_format_f3,
        'F4': serialize_format_f4,
        'F5': serialize_format_f5,
    }

    if format_name not in serializers:
        raise ValueError(f"Unknown format: {format_name}. Use F1-F5.")

    topology_str = serializers[format_name](problem)
    chain_str = build_vnf_chain_string(problem)

    # Try to load from template file
    config = load_environment()
    template_dir = os.path.join(config['PROMPTS_DIR'], f'template_{template.lower()}')
    template_file = os.path.join(template_dir, f'{format_name}_edge_list.txt')

    if format_name == 'F2':
        template_file = os.path.join(template_dir, 'F2_adjacency_matrix.txt')
    elif format_name == 'F3':
        template_file = os.path.join(template_dir, 'F3_natural_language.txt')
    elif format_name == 'F4':
        template_file = os.path.join(template_dir, 'F4_json.txt')
    elif format_name == 'F5':
        template_file = os.path.join(template_dir, 'F5_ranked_neighbor.txt')

    if os.path.exists(template_file):
        with open(template_file, 'r') as f:
            prompt_template = f.read()
        prompt = prompt_template.replace('{edge_list_serialization}', topology_str)
        prompt = prompt.replace('{adjacency_matrix_serialization}', topology_str)
        prompt = prompt.replace('{natural_language_serialization}', topology_str)
        prompt = prompt.replace('{json_serialization}', topology_str)
        prompt = prompt.replace('{ranked_neighbor_list_serialization}', topology_str)
        prompt = prompt.replace('{vnf_chain_string}', chain_str)
        prompt = prompt.replace('{source_node}', problem.get('source', 'N/A'))
        prompt = prompt.replace('{destination_node}', problem.get('destination', 'N/A'))
        prompt = prompt.replace('{bandwidth}', str(problem.get('bandwidth_demand', 'N/A')))
        return prompt

    # Fallback if template file not found
    prompt = f"""{topology_str}

VNF chain and demand:
Chain order: {chain_str}
Source node: {problem.get('source', 'N/A')}
Destination node: {problem.get('destination', 'N/A')}
Bandwidth demand: {problem.get('bandwidth_demand', 'N/A')}

Important: Each VNF must be placed on a distinct node. The path must go source -> node(f1) -> node(f2) -> ... -> destination. Respect all node CPU and link bandwidth capacities.

Output your placement and path as a single JSON object with keys "placement" (mapping VNF id to node id) and "path" (list of node pairs, e.g., [["v3","v5"], ["v5","v7"]])."""

    return prompt


def parse_llm_response(response_text, problem):
    """Parse LLM response and validate against problem constraints.

    Returns dict with: feasible (bool), cost (float), violation_type (str),
                       placement (dict), path (list)
    """
    result = {
        'feasible': False,
        'cost': None,
        'violation_type': 'none',
        'placement': None,
        'path': None,
        'parse_error': False,
        'cpu_cost': 0,
        'bw_cost': 0,
    }

    if not response_text or not isinstance(response_text, str):
        result['violation_type'] = 'parse_error'
        result['parse_error'] = True
        return result

    # Step 1: Extract JSON - look for the LAST valid JSON with placement and path
    data = None
    # Strategy 1: Try finding JSON with both 'placement' and 'path' keys
    for m in re.finditer(r'\{[^{}]*placement[^{}]*path[^{}]*\}', response_text, re.DOTALL):
        try:
            candidate = json.loads(m.group())
            if 'placement' in candidate and 'path' in candidate:
                data = candidate
        except:
            continue
    
    # Strategy 2: Try nested braces (e.g., {"a":{"b":"c"}})
    if data is None:
        depth = 0
        start = -1
        for i, ch in enumerate(response_text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        candidate = json.loads(response_text[start:i+1])
                        if 'placement' in candidate and 'path' in candidate:
                            data = candidate
                            break
                    except:
                        pass
    
    # Strategy 3: Simple brace match
    if data is None:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except:
                pass
    
    if data is None:
        result['violation_type'] = 'parse_error'
        result['parse_error'] = True
        return result

    placement = data.get('placement', {})
    path = data.get('path', [])

    # Normalize path format: convert various formats to list of (node1, node2) tuples
    if isinstance(path, list) and len(path) > 0:
        first = path[0]
        if not isinstance(first, (list, tuple)):
            # Flat format: [node1, node2, node3] → [(node1, node2), (node2, node3)]
            path = [(path[i], path[i+1]) for i in range(len(path)-1)]
        elif isinstance(first, dict):
            # Dict format: [{"source":"v1","target":"v4"}, ...] → [("v1","v4"), ...]
            new_path = []
            for p in path:
                if isinstance(p, dict):
                    vals = list(p.values())
                    new_path.append((vals[0], vals[1]) if len(vals) >= 2 else tuple(vals))
                else:
                    new_path.append(tuple(p) if len(p) == 2 else (p[0], p[1]))
            path = new_path
        else:
            # List format: [["v1","v4"], ...] → [("v1","v4"), ...]
            path = [tuple(p) if len(p) == 2 else (p[0], p[1]) for p in path]
        data['path'] = path

    if not placement or not path:
        result['violation_type'] = 'parse_error'
        result['parse_error'] = True
        return result

    node_ids = [n['id'] for n in problem['graph']['nodes']]
    vnf_chain = problem.get('vnf_chain', [])
    vnf_ids = [v['id'] for v in vnf_chain]
    bandwidth_demand = problem.get('bandwidth_demand', 0)

    # Validate placement keys
    for vnf_id in placement.keys():
        if vnf_id not in vnf_ids:
            result['violation_type'] = 'node_existence'
            return result

    # Validate placement values
    for node_id in placement.values():
        if node_id not in node_ids:
            result['violation_type'] = 'node_existence'
            return result

    # Check distinctness
    if len(set(placement.values())) != len(placement.values()):
        result['violation_type'] = 'distinctness'
        return result

    # Validate path edges
    edges_lookup = {}
    for e in problem['graph']['edges']:
        edges_lookup[(e['source'], e['target'])] = e['bandwidth_capacity']
        edges_lookup[(e['target'], e['source'])] = e['bandwidth_capacity']

    for pair in path:
        # Handle dict pair — extract values regardless of key names
        if isinstance(pair, dict):
            vals = list(pair.values())
            if len(vals) >= 2:
                pair = (vals[0], vals[1])
            else:
                result['violation_type'] = 'path_adjacency'
                return result
        if len(pair) != 2:
            result['violation_type'] = 'path_adjacency'
            return result
        key = (pair[0], pair[1])
        if key not in edges_lookup:
            result['violation_type'] = 'path_adjacency'
            return result

    # Check node CPU capacity
    node_cpu_used = {}
    for vnf_id, node_id in placement.items():
        cpu_demand = None
        for vnf in vnf_chain:
            if vnf['id'] == vnf_id:
                cpu_demand = vnf['cpu_demand']
                break
        if cpu_demand is None:
            result['violation_type'] = 'node_existence'
            return result
        if node_id not in node_cpu_used:
            node_cpu_used[node_id] = 0
        node_cpu_used[node_id] += cpu_demand

    for node_id, total_cpu in node_cpu_used.items():
        node_capacity = None
        for n in problem['graph']['nodes']:
            if n['id'] == node_id:
                node_capacity = n['cpu_capacity']
                break
        if node_capacity is not None and total_cpu > node_capacity:
            result['violation_type'] = 'node_capacity'
            return result

    # Check link bandwidth
    edge_traversal_count = {}
    for pair in path:
        key = (pair[0], pair[1])
        edge_traversal_count[key] = edge_traversal_count.get(key, 0) + 1

    for (src, dst), count in edge_traversal_count.items():
        cap = edges_lookup.get((src, dst), 0)
        if count * bandwidth_demand > cap:
            result['violation_type'] = 'link_capacity'
            return result

    # Path order check
    source = problem.get('source')
    destination = problem.get('destination')

    if not path or path[0][0] != source:
        result['violation_type'] = 'path_adjacency'
        return result
    if path[-1][1] != destination:
        result['violation_type'] = 'path_adjacency'
        return result

    # Verify VNF order in path
    path_nodes = [pair[0] for pair in path] + [path[-1][1]]
    ordered_placements = []
    for node in path_nodes:
        for vnf_id, placed_node in placement.items():
            if node == placed_node and vnf_id not in ordered_placements:
                ordered_placements.append(vnf_id)
                break

    expected_order = [v['id'] for v in vnf_chain]
    if ordered_placements != expected_order:
        result['violation_type'] = 'path_adjacency'
        return result

    # All checks passed - compute cost
    cpu_cost = sum(node_cpu_used.values())
    bw_cost = sum(count * bandwidth_demand for count in edge_traversal_count.values())
    total_cost = cpu_cost + bw_cost

    result['feasible'] = True
    result['cost'] = total_cost
    result['placement'] = placement
    result['path'] = path
    result['violation_type'] = 'none'
    result['cpu_cost'] = cpu_cost
    result['bw_cost'] = bw_cost

    return result


def compute_cost(placement, path, problem):
    """Compute total cost for a given placement and path."""
    vnf_chain = problem.get('vnf_chain', [])
    bandwidth_demand = problem.get('bandwidth_demand', 0)

    cpu_cost = 0
    for vnf in vnf_chain:
        if vnf['id'] in placement:
            cpu_cost += vnf['cpu_demand']

    edge_count = {}
    for pair in path:
        key = (pair[0], pair[1])
        edge_count[key] = edge_count.get(key, 0) + 1

    bw_cost = sum(count * bandwidth_demand for count in edge_count.values())
    return cpu_cost + bw_cost


def compute_optimality_gap(llm_cost, optimal_cost):
    """Compute optimality gap as percentage."""
    if llm_cost is None or optimal_cost is None or optimal_cost == 0:
        return float('nan')
    return ((llm_cost - optimal_cost) / optimal_cost) * 100


def inject_noise(serialized_text, format_name, noise_level):
    """Inject noise into serialized format text.

    noise_level: 1 (minor), 2 (moderate), 3 (major)
    """
    if noise_level == 0:
        return serialized_text

    lines = serialized_text.split('\n')
    noisy_lines = list(lines)

    # Find all node IDs
    node_ids = list(set(re.findall(r'\b(v\d+)\b', serialized_text)))

    if noise_level == 1:
        # Minor: swap 2 node IDs
        if len(node_ids) >= 2:
            a, b = node_ids[0], node_ids[min(1, len(node_ids)-1)]
            noisy_lines = [line.replace(a, '___TMP___').replace(b, a).replace('___TMP___', b) for line in noisy_lines]

    elif noise_level == 2:
        # Moderate: swap 3 pairs
        for i in range(min(3, len(node_ids) // 2)):
            if i * 2 + 1 < len(node_ids):
                a, b = node_ids[i * 2], node_ids[i * 2 + 1]
                noisy_lines = [line.replace(a, '___TMP___').replace(b, a).replace('___TMP___', b) for line in noisy_lines]

    elif noise_level == 3:
        # Major: omit one edge line, duplicate another
        edge_lines = [i for i, line in enumerate(noisy_lines) if re.search(r'v\d+.*v\d+', line)]
        if len(edge_lines) >= 2:
            noisy_lines.pop(edge_lines[0])
            dup_idx = edge_lines[1] - 1 if edge_lines[1] > edge_lines[0] else edge_lines[1]
            if dup_idx < len(noisy_lines):
                noisy_lines.insert(dup_idx + 1, noisy_lines[dup_idx])

    return '\n'.join(noisy_lines)


def get_model_name():
    """Return which model(s) are available based on .env keys."""
    config = load_environment()
    models = []
    if config.get('DEEPSEEK_API_KEY'):
        models.append('deepseek_v4pro')
    if config.get('OPENAI_API_KEY'):
        models.append('gpt5mini')
    return models