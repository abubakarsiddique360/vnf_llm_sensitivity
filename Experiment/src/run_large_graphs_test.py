"""Run 5-query test for large_graphs with fixed F2 sparse format."""
import sys, os, json, time, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fresh import - clear any cached modules
for mod in list(sys.modules.keys()):
    if 'src' in mod or 'utils' in mod:
        del sys.modules[mod]

from src.utils import load_problems, build_user_prompt, save_json

API_KEY = "sk-your-deepseek-api-key-here"
BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = "You are a VNF placement solver.\n\nINCORRECT (do NOT output like this):\n```json\n{\"placement\": {\"f1\": \"v4\"}, \"path\": [\"source\", \"v4\"]}\n```\n\nCORRECT (output ONLY this format, no markdown, no code fences):\n{\"placement\": {\"f1\": \"v4\", \"f2\": \"v5\", \"f3\": \"v8\"}, \"path\": [[\"source\", \"v4\"], [\"v4\", \"v5\"], [\"v5\", \"v8\"], [\"v8\", \"destination\"]]}\n\nRules:\n- Output ONLY the raw JSON object. No ```json markers, no explanation before or after.\n- 'placement' maps each VNF id to a node id.\n- 'path' is a list of [node1, node2] pairs.\n- Path visits VNFs in chain order from source to destination.\n- Each VNF on a distinct node.\n- Respect CPU and bandwidth capacities."

probs = load_problems(r'E:\vnf_llm_sensitivity\Experiment\data\problems')
lp = [p for p in probs if p.get('size_category') == 'extra_large'][0]
out = r'E:\vnf_llm_sensitivity\Experiment\data\llm_responses\deepseek\large_graphs'
os.makedirs(out, exist_ok=True)

print(f"Problem {lp['problem_id']}: {lp['num_nodes']} nodes, {len(lp['graph']['edges'])} edges")
print()

for fmt in ['F1','F2','F3','F4','F5']:
    prompt = build_user_prompt(lp, fmt, 'A')
    print(f'{fmt}: prompt={len(prompt)} chars', end='')
    
    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-v4-flash",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 16384
            },
            timeout=180)
        elapsed = round(time.time() - start, 2)
        
        if resp.status_code != 200:
            print(f'  ERROR {resp.status_code}')
            continue
            
        data = resp.json()
        msg = data['choices'][0]['message']
        content = msg.get('content', '') or ''
        reasoning = msg.get('reasoning_content', '') or ''
        
        final = content
        if '{' not in final or 'placement' not in final:
            if '{' in reasoning and 'placement' in reasoning:
                final = reasoning
        if not final.strip():
            final = reasoning
            
        has_json = '{' in final and 'placement' in final
        
        save_json({
            'problem_id': lp['problem_id'], 'model': 'deepseek-v4-flash',
            'experiment': 'large_graphs', 'format': fmt, 'run': 1,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'prompt': {'system': SYSTEM_PROMPT, 'user': prompt},
            'raw_response': final, 'reasoning': reasoning,
            'token_usage': {'completion_tokens': data['usage'].get('completion_tokens', 0)},
            'latency_seconds': elapsed
        }, os.path.join(out, f'{lp["problem_id"]}_{fmt}_run1.json'))
        
        print(f'  resp={len(final)}c, json={has_json}, {elapsed}s')
    except Exception as e:
        print(f'  ERROR: {e}')

print("\nDone! All 5 large_graphs queries complete.")
