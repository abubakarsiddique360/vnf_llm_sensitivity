"""Parse ALL LLM responses, check feasibility, compute costs, and save metrics CSV."""

import os
import sys
import csv
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, load_problems, load_ilp_solutions, setup_logging, parse_llm_response, save_json, load_json, compute_optimality_gap


def parse_all_responses(llm_responses_dir, problems, ilp_solutions):
    """Parse all LLM response files and return list of result dicts."""
    results = []
    responses_path = Path(llm_responses_dir)

    if not responses_path.exists():
        print(f"ERROR: Responses directory not found: {responses_path}")
        return results

    # Walk through all model directories
    for model_dir in sorted(responses_path.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # Walk through experiment directories
        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            experiment = exp_dir.name

            # Process all JSON files
            for resp_file in sorted(exp_dir.glob('*.json')):
                resp_data = load_json(str(resp_file))
                if not resp_data:
                    continue

                problem_id = resp_data.get('problem_id')
                format_name = resp_data.get('format', '')
                run = resp_data.get('run', 1)
                temperature = resp_data.get('temperature', 0.0)
                prompt_version = resp_data.get('prompt_template', 'A')
                noise_level = resp_data.get('noise_level', 0)
                inference_time = resp_data.get('latency_seconds', 0)
                raw_response = resp_data.get('raw_response', '')

                # Find the corresponding problem
                problem = None
                for p in problems:
                    if p.get('problem_id') == problem_id:
                        problem = p
                        break

                if not problem:
                    continue

                # Parse the response
                try:
                    parse_result = parse_llm_response(raw_response, problem)
                except Exception as e:
                    if 'unhashable' in str(e):
                        parse_result = {'feasible': False, 'cost': None, 'violation_type': 'path_adjacency', 'parse_error': True, 'cpu_cost': 0, 'bw_cost': 0}
                    else:
                        raise

                # Get ILP solution for optimal cost
                ilp_sol = ilp_solutions.get(problem_id, {})
                optimal_cost = ilp_sol.get('optimal_cost', None)

                # Compute optimality gap
                gap = compute_optimality_gap(parse_result.get('cost'), optimal_cost)

                # Determine size category
                size_cat = problem.get('size_category', 'unknown')
                family = problem.get('family', 'unknown')

                # Get path safely (could be None)
                path_val = parse_result.get('path')
                num_edges = len(path_val) if path_val else 0

                # Build result row
                row = {
                    'problem_id': problem_id,
                    'model': model_name,
                    'experiment': experiment,
                    'temperature': temperature,
                    'prompt_version': prompt_version,
                    'format': format_name,
                    'run': run,
                    'feasible': 1 if parse_result.get('feasible') else 0,
                    'cost': parse_result.get('cost', float('nan')),
                    'optimality_gap': gap,
                    'parse_error': 1 if parse_result.get('parse_error') else 0,
                    'violation_type': parse_result.get('violation_type', 'none'),
                    'cpu_cost': parse_result.get('cpu_cost', 0),
                    'bandwidth_cost': parse_result.get('bw_cost', 0),
                    'num_path_edges': num_edges,
                    'family': family,
                    'size_category': size_cat,
                    'noise_level': noise_level,
                    'inference_time_seconds': inference_time
                }
                results.append(row)

    return results


def save_metrics_csv(results, output_path):
    """Save parsed results to CSV."""
    if not results:
        print("ERROR: No results to save!")
        return

    fieldnames = [
        'problem_id', 'model', 'experiment', 'temperature', 'prompt_version',
        'format', 'run', 'feasible', 'cost', 'optimality_gap', 'parse_error',
        'violation_type', 'cpu_cost', 'bandwidth_cost', 'num_path_edges',
        'family', 'size_category', 'noise_level', 'inference_time_seconds'
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Parse LLM responses')
    parser.add_argument('--input', type=str, default='data/llm_responses')
    parser.add_argument('--problems', type=str, default=None)
    parser.add_argument('--ilp', type=str, default=None)
    parser.add_argument('--output', type=str, default='data/parsed')
    args = parser.parse_args()

    config = load_environment()
    problems_dir = args.problems or config['PROBLEMS_DIR']
    ilp_dir = args.ilp or config['ILP_SOLUTIONS_DIR']
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging('parsing')

    problems = load_problems(problems_dir)
    ilp_solutions = load_ilp_solutions(ilp_dir)

    logger.info(f"Loaded {len(problems)} problems, {len(ilp_solutions)} ILP solutions")

    results = parse_all_responses(args.input, problems, ilp_solutions)
    logger.info(f"Parsed {len(results)} responses")

    csv_path = os.path.join(output_dir, 'metrics.csv')
    save_metrics_csv(results, csv_path)

    # Print summary
    feasible = sum(1 for r in results if r['feasible'])
    print(f"\nParsing Summary: {len(results)} responses")
    print(f"  Feasible: {feasible}")
    print(f"  Infeasible: {len(results) - feasible}")
    print(f"  Parse errors: {sum(1 for r in results if r['parse_error'])}")


if __name__ == '__main__':
    main()