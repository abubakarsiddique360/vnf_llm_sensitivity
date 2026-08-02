"""
Query gpt-5-mini via OpenAI API. Run AFTER DeepSeek experiments are complete.

Supports experiments: main_t0, main_t0.5, prompt_b, large_graphs, vm_placement, noise
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, load_problems, setup_logging, save_json, build_user_prompt, inject_noise


def query_gpt(prompt, temperature=0.0, max_tokens=16384, timeout=180):
    """Send a query to GPT-5-mini API with exponential backoff retry."""
    try:
        import openai
    except ImportError:
        print("ERROR: openai package required.")
        return None, 0, None

    config = load_environment()
    api_key = config.get('OPENAI_API_KEY', '')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return None, 0, None

    client = openai.OpenAI(api_key=api_key)

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a VNF placement solver.\n\nIMPORTANT: Output the JSON solution FIRST, then explain your reasoning.\n\nCORRECT format (JSON FIRST, then reasoning):\n{\"placement\": {\"f1\": \"v4\", \"f2\": \"v5\", \"f3\": \"v8\"}, \"path\": [[\"source\", \"v4\"], [\"v4\", \"v5\"], [\"v5\", \"v8\"], [\"v8\", \"destination\"]]}\nThe path starts at source v1... (your reasoning here)\n\nWRONG format (do NOT output reasoning before JSON):\nWe need to place VNFs... (reasoning first)\n{\"placement\": {\"f1\": \"v4\"}, \"path\": [[\"source\", \"v4\"]]}\n\nRULES:\n- First line of output MUST be the JSON object.\n- Then briefly explain your reasoning.\n- No ```json markers, no markdown.\n- 'placement' maps VNF id -> node id.\n- 'path' is list of [node1,node2] pairs.\n- Each VNF on a distinct node. Path in chain order source -> f1 -> f2 -> ... -> destination.\n- Respect CPU and bandwidth capacities."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens,
                timeout=timeout
            )
            elapsed = time.time() - start
            content = response.choices[0].message.content
            usage = response.usage
            token_info = {
                'prompt_tokens': usage.prompt_tokens if usage else 0,
                'completion_tokens': usage.completion_tokens if usage else 0,
                'total_tokens': usage.total_tokens if usage else 0
            }
            return content, elapsed, token_info

        except Exception as e:
            delay = base_delay * (2 ** attempt)
            print(f"  GPT Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    return None, 0, None


def run_experiment(experiment, temperature, template, problems, concurrent):
    """Run a complete experiment phase with GPT."""
    config = load_environment()
    logger = setup_logging(f'gpt_{experiment}')

    model_dir = os.path.join(config['LLM_RESPONSES_DIR'], 'gpt', experiment)
    os.makedirs(model_dir, exist_ok=True)

    # Determine which problems to use based on experiment type
    if experiment == 'main_t0':
        selected_problems = problems[:120]
        repeats = 1
        noise_levels = [0]
    elif experiment == 'main_t0.5':
        selected_problems = problems[:120]
        repeats = 1
        noise_levels = [0]
    elif experiment == 'prompt_b':
        selected_problems = problems[:120]
        repeats = 1
        noise_levels = [0]
    elif experiment == 'large_graphs':
        selected_problems = [p for p in problems if p.get('size_category') == 'extra_large'][:30]
        repeats = 1
        noise_levels = [0]
    elif experiment == 'vm_placement':
        selected_problems = problems[:20]
        repeats = 1
        noise_levels = [0]
    elif experiment == 'noise':
        selected_problems = problems[:20]
        repeats = 1
        noise_levels = [1, 2, 3]
    else:
        selected_problems = problems[:200]
        repeats = 3
        noise_levels = [0]

    if not selected_problems:
        logger.error("No problems to process!")
        return

    formats = ['F1', 'F2', 'F3', 'F4', 'F5']
    total_queries = len(selected_problems) * len(formats) * repeats * len(noise_levels)

    logger.info(f"Starting GPT {experiment}: {len(selected_problems)} problems, {repeats} repeats, {total_queries} total queries")
    logger.info(f"Template: {template}")

    completed = 0
    failed = 0

    tasks = []
    for problem in selected_problems:
        for fmt in formats:
            prompt = build_user_prompt(problem, fmt, template if template else 'A')
            for run in range(1, repeats + 1):
                for nl in noise_levels:
                    if nl > 0:
                        noisy_prompt = inject_noise(prompt, fmt, nl)
                    else:
                        noisy_prompt = prompt
                    tasks.append((problem, fmt, run, noisy_prompt, nl))

    def process_task(task):
        problem, fmt, run, prompt, nl = task
        pid = problem['problem_id']
        if nl > 0:
            filename = f"{pid}_{fmt}_noise{nl}_run{run}.json"
        else:
            filename = f"{pid}_{fmt}_run{run}.json"
        filepath = os.path.join(model_dir, filename)

        if os.path.exists(filepath):
            return True, pid, fmt, run, "skipped"

        response_text, latency, token_info = query_gpt(prompt, temperature)
        if response_text is None:
            return False, pid, fmt, run, "failed"

        response_data = {
            'problem_id': pid,
            'model': 'gpt-5-mini',
            'experiment': experiment,
            'format': fmt,
            'run': run,
            'temperature': temperature,
            'prompt_template': template if template else 'A',
            'noise_level': nl,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'prompt': {'system': '', 'user': prompt},
            'raw_response': response_text,
            'token_usage': token_info,
            'latency_seconds': round(latency, 2)
        }
        save_json(response_data, filepath)
        return True, pid, fmt, run, "success"

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for future in as_completed(futures):
            completed += 1
            success, pid, fmt, run, status = future.result()
            if not success:
                failed += 1
            if completed % 100 == 0 or completed == total_queries:
                pct = completed / total_queries * 100
                logger.info(f"Progress: {completed}/{total_queries} ({pct:.1f}%)")
                print(f"  GPT {experiment}: {completed}/{total_queries} ({pct:.1f}%)")

    logger.info(f"Completed GPT {experiment}: {completed} queries, {failed} failures")
    print(f"GPT {experiment}: DONE ({completed} queries, {failed} failures)")


def main():
    parser = argparse.ArgumentParser(description='Query GPT-5-mini API')
    parser.add_argument('--experiment', type=str, default='main_t0',
                        choices=['main_t0', 'main_t0.5', 'prompt_b', 'large_graphs', 'vm_placement', 'noise'],
                        help='Experiment type')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM temperature (0.0 for primary, 0.5 for sensitivity)')
    parser.add_argument('--template', type=str, default='A', choices=['A', 'B'],
                        help='Prompt template version')
    parser.add_argument('--concurrent', type=int, default=10,
                        help='Number of concurrent API queries')
    args = parser.parse_args()

    config = load_environment()
    problems = load_problems(config['PROBLEMS_DIR'])
    if not problems:
        print("ERROR: No problems found. Run 01_generate_problems.py first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"GPT Experiment: {args.experiment}")
    print(f"Template: {args.template}")
    print(f"Problems loaded: {len(problems)}")
    print(f"Make sure VPN is ON if in China!")
    print(f"{'='*60}\n")

    run_experiment(
        experiment=args.experiment,
        temperature=args.temperature,
        template=args.template,
        problems=problems,
        concurrent=args.concurrent
    )


if __name__ == '__main__':
    main()