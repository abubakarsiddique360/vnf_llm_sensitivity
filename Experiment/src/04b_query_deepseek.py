"""
Query DeepSeek-V4-Flash API for all experiment phases.
Uses requests library directly for reliable response capture.
"""
import os, sys, json, time, argparse, re
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, load_problems, setup_logging, save_json, build_user_prompt, inject_noise

BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = (
    "You are a VNF placement solver.\n\n"
    "===== RESPONSE STRUCTURE (FOLLOW THIS EXACTLY) =====\n\n"
    "REASONING:\n"
    "Write your step-by-step reasoning here. Think carefully about:\n"
    "- CPU capacities of each node\n"
    "- Bandwidth requirements on each link\n"
    "- Finding a valid path from source to destination\n"
    "- Placing each VNF on a distinct node in chain order\n\n"
    "FINAL ANSWER:\n"
    "Output ONLY the JSON. No extra text, no code fences, no markdown.\n\n"
    "WRONG final answer (do NOT use):\n"
    '```json\n{"placement": {"f1": "v4"}, "path": ["source", "v4"]}\n```\n\n'
    "CORRECT final answer (use exactly this format):\n"
    '{"placement": {"f1": "v4", "f2": "v5", "f3": "v8"}, "path": [["source", "v4"], ["v4", "v5"], ["v5", "v8"], ["v8", "destination"]]}\n\n'
    "===== END OF STRUCTURE =====\n\n"
    "Summary:\n"
    "1. First, write your REASONING section explaining how you solve the problem.\n"
    "2. Then, write FINAL ANSWER with ONLY the JSON.\n"
    "3. JSON must have 'placement' (VNF id -> node id) and 'path' (list of [node1,node2] pairs).\n"
    "4. Each VNF on a distinct node. Path in chain order source -> f1 -> f2 -> ... -> destination.\n"
    "5. Respect CPU and bandwidth capacities."
)

def query_deepseek(prompt, temperature=0.0, max_tokens=16384, timeout=180):
    """Query DeepSeek API using raw requests for reliability."""
    config = load_environment()
    API_KEY = config.get('DEEPSEEK_API_KEY', '')
    if not API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not found in .env file")
        return None, 0, None
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            start = time.time()
            resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=timeout)
            elapsed = time.time() - start
            
            if resp.status_code != 200:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
                continue
            
            data = resp.json()
            choice = data['choices'][0]
            msg = choice['message']
            content = msg.get('content', '') or ''
            reasoning = msg.get('reasoning_content', '') or ''
            
            # Extract JSON from FINAL ANSWER section
            final_json = None
            if 'FINAL ANSWER:' in content:
                parts = content.split('FINAL ANSWER:')
                if len(parts) >= 2:
                    final_part = parts[-1].strip()
                    # Try regex match first
                    for m in re.finditer(r'\{[^{}]*placement[^{}]*path[^{}]*\}', final_part, re.DOTALL):
                        try:
                            if json.loads(m.group()):
                                final_json = m.group()
                                break
                        except: pass
                    # Try nested brace matching
                    if final_json is None:
                        depth=0; start_i=-1
                        for i,ch in enumerate(final_part):
                            if ch=='{':
                                if depth==0: start_i=i
                                depth+=1
                            elif ch=='}':
                                depth-=1
                                if depth==0 and start_i>=0:
                                    try:
                                        cand = json.loads(final_part[start_i:i+1])
                                        if 'placement' in cand and 'path' in cand:
                                            final_json = final_part[start_i:i+1]
                                            break
                                    except: pass
            
            if final_json:
                content = final_json
            elif '{' in content and 'placement' in content:
                pass
            elif '{' in reasoning and 'placement' in reasoning:
                content = reasoning
            elif not content.strip() and reasoning.strip():
                content = reasoning
            
            token_info = {
                'prompt_tokens': data['usage'].get('prompt_tokens', 0),
                'completion_tokens': data['usage'].get('completion_tokens', 0),
                'total_tokens': data['usage'].get('total_tokens', 0)
            }
            return content, elapsed, token_info
            
        except Exception as e:
            delay = 2 ** attempt
            print(f"  Attempt {attempt+1} failed: {e}. Retry in {delay}s...")
            time.sleep(delay)
    
    return None, 0, None


def run_experiment(experiment, temperature, template, problems, concurrent):
    """Run experiment with DeepSeek."""
    config = load_environment()
    logger = setup_logging(f'deepseek_{experiment}')
    model_dir = os.path.join(config['LLM_RESPONSES_DIR'], 'deepseek', experiment)
    os.makedirs(model_dir, exist_ok=True)

    if experiment == 'main_t0':
        selected = problems[:120]; repeats = 1; noise_levels = [0]
    elif experiment == 'main_t0.5':
        selected = problems[:120]; repeats = 1; noise_levels = [0]
    elif experiment == 'prompt_b':
        selected = problems[:120]; repeats = 1; noise_levels = [0]
    elif experiment == 'large_graphs':
        selected = [p for p in problems if p.get('size_category') == 'extra_large'][:30]
        repeats = 1; noise_levels = [0]
    elif experiment == 'vm_placement':
        selected = problems[:20]; repeats = 1; noise_levels = [0]
    elif experiment == 'noise':
        selected = problems[:20]; repeats = 1; noise_levels = [1,2,3]
    else:
        selected = problems[:200]; repeats = 3; noise_levels = [0]

    if not selected:
        logger.error("No problems!"); return

    formats = ['F1','F2','F3','F4','F5']
    total = len(selected) * len(formats) * repeats * len(noise_levels)
    logger.info(f"Starting DeepSeek {experiment}: {total} queries, {concurrent} concurrent")
    
    completed = 0; failed = 0

    tasks = []
    for problem in selected:
        for fmt in formats:
            prompt = build_user_prompt(problem, fmt, template or 'A')
            for run in range(1, repeats + 1):
                for nl in noise_levels:
                    p = inject_noise(prompt, fmt, nl) if nl > 0 else prompt
                    tasks.append((problem, fmt, run, p, nl))

    def process(task):
        problem, fmt, run, prompt, nl = task
        pid = problem['problem_id']
        fname = f"{pid}_{fmt}_noise{nl}_run{run}.json" if nl > 0 else f"{pid}_{fmt}_run{run}.json"
        fpath = os.path.join(model_dir, fname)
        if os.path.exists(fpath):
            return True, pid, fmt, run
        
        content, latency, tokens = query_deepseek(prompt, temperature)
        if content is None:
            return False, pid, fmt, run
        
        save_json({
            'problem_id': pid, 'model': 'deepseek-v4-flash', 'experiment': experiment,
            'format': fmt, 'run': run, 'temperature': temperature,
            'prompt_template': template or 'A', 'noise_level': nl,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'prompt': {'system': '', 'user': prompt},
            'raw_response': content, 'token_usage': tokens,
            'latency_seconds': round(latency, 2)
        }, fpath)
        return True, pid, fmt, run

    with ThreadPoolExecutor(max_workers=concurrent) as ex:
        fut = [ex.submit(process, t) for t in tasks]
        for f in as_completed(fut):
            completed += 1
            ok, pid, fmt, run = f.result()
            if not ok: failed += 1
            if completed % 100 == 0 or completed == total:
                pct = completed / total * 100
                print(f"  DeepSeek {experiment}: {completed}/{total} ({pct:.1f}%)")

    logger.info(f"Done: {completed} queries, {failed} failures")
    print(f"DeepSeek {experiment}: DONE ({completed} queries, {failed} failures)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', default='main_t0', choices=['main_t0','main_t0.5','prompt_b','large_graphs','vm_placement','noise'])
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--template', default='A', choices=['A','B'])
    p.add_argument('--concurrent', type=int, default=10)
    args = p.parse_args()
    
    config = load_environment()
    problems = load_problems(config['PROBLEMS_DIR'])
    if not problems:
        print("ERROR: No problems found. Run 01_generate_problems.py first.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"DeepSeek Experiment: {args.experiment} (model: deepseek-v4-flash)")
    print(f"Temperature: {args.temperature}, Template: {args.template}")
    print(f"{'='*60}\n")
    
    run_experiment(args.experiment, args.temperature, args.template, problems, args.concurrent)

if __name__ == '__main__':
    main()