# VNF LLM Sensitivity

This repository contains the full experimental pipeline behind a study of whether the
*input representation format* of a problem matters to a Large Language Model (LLM).
We take **Virtual Network Function (VNF) placement** — an NP-hard problem in Network
Function Virtualization (NFV) — and encode the *same* problem instance in five
structurally different ways:

| Format | Description              |
| ------ | ------------------------ |
| F1     | Edge list                |
| F2     | Adjacency matrix         |
| F3     | Natural language (prose) |
| F4     | JSON                     |
| F5     | Ranked neighbor lists    |

We query two LLM families (**GPT-5-mini** and **DeepSeek-V4-Flash**) across
150 problem instances (3,500 queries in total), and compare their outputs against
an ILP-based placement-then-route baseline (PuLP/CBC) and against greedy/random
placement baselines.

> Key finding: format choice alone shifts feasibility by **17.5 percentage points**
> for one model while the other remains nearly stable. In network automation,
> representation deserves the same care as the algorithm.

## Repository layout

```
VNF_LLM_SENSITIVITY/
├── LICENSE                     # MIT license
├── README.md
├── CITATION.cff                # citation metadata (GitHub "Cite this repository")
├── .gitignore
└── Experiment/
    ├── requirements.txt        # Python dependencies
    ├── .env.example            # template for API keys/config (copy to .env)
    ├── .gitignore
    ├── src/                    # main pipeline scripts (01–08, noise_injection, utils)
    ├── baselines/              # greedy heuristic and random placement
    ├── prompts/                # F1–F5 prompt templates (template_a / template_b)
    ├── data/
    │   ├── problems/           # 150 generated VNF placement instances
    │   ├── ilp_solutions/      # ILP ground-truth solutions
    │   ├── baselines/          # greedy/random baseline result CSVs
    │   ├── parsed/             # parsed LLM outputs and metrics
    │   └── llm_responses/      # raw LLM API responses (gitignored, ~3500 files)
    ├── results/                # per-model and comparison results + figures
    ├── logs/                   # run logs (gitignored)
    └── visualization/          # figure-generation scripts
```

## Getting started

Requires **Python 3.10+**.

```bash
cd Experiment
python -m venv .venv
.\.venv\Scripts\activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Configure API keys

Copy the example configuration and fill in your keys:

```bash
cp .env.example .env            # then edit .env with your OPENAI_API_KEY and DEEPSEEK_API_KEY
```

`.env` is gitignored — never commit real keys. `.env.example` is the committed template.

## Running the pipeline

The scripts in `src/` run in order:

| Step | Script                                                               | Purpose                                                            |
| ---- | -------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 1    | `01_generate_problems.py`                                          | Generate VNF placement instances (150: 120 main + 30 large graphs) |
| 2    | `02_solve_ilp.py`                                                  | Solve ground truth with ILP (PuLP/CBC)                             |
| 3    | `03_serialize_formats.py`                                          | Build the five prompt formats F1–F5                               |
| 4    | `04a_query_gpt5mini.py`, `04b_query_deepseek.py`                 | Query the LLMs (needs API keys)                                    |
| 5    | `05_parse_responses.py`                                            | Parse raw LLM responses into tables                                |
| 6    | `06_compute_metrics.py`                                            | Feasibility rate, optimality gap, format sensitivity               |
| 7    | `07_statistics_frequentist.py`, `07b_statistics_bayesian.py`     | Statistical analysis                                               |
| 8    | `08_review_statistics.py`                                        | Reproduce revised-manuscript stats (Wilson CIs, paired McNemar tests, prompt-token counts) |
| –   | `noise_injection.py`                                               | Noise-robustness experiments                                       |
| –   | `baselines/greedy_heuristic.py`, `baselines/random_placement.py` | Traditional baselines                                              |
| –   | `visualization/`                                                   | Generate paper figures                                             |

## Metrics

- **Feasibility rate** — percentage of valid (capacity-respecting) deployments.
- **Cost gap** — relative deviation of a feasible answer's cost from the placement-then-route baseline (a comparator, not a joint optimum; a negative value means an edge-reusing integrated route is cheaper than the decomposed baseline).
- **Format sensitivity** — mean per-problem standard deviation of the cost gap across the five formats.

## Citation

If you use this repository, please cite it (see `CITATION.cff`):

**Authors:** Abu Bakar Siddique (ORCID: 0009-0001-2138-0802), Cang Lis, Odgarig Bayarkhuu, Yeasin Arafat

```bibtex
@misc{siddique2026vnfllm,
  title  = {VNF LLM Sensitivity: Input Format Sensitivity in LLM-Based VNF Placement},
  author = {Siddique, Abu Bakar and Lis, Cang and Bayarkhuu, Odgarig and Arafat, Yeasin},
  year   = {2026},
  note   = {GitHub repository},
  howpublished = {\url{https://github.com/abubakarsiddique360/VNF_LLM_SENSITIVITY}}
}
```

## License

This project is released under the [MIT License](LICENSE).
