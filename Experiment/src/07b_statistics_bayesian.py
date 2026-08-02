"""Bayesian statistics (optional - requires PyMC)."""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, setup_logging


def main():
    parser = argparse.ArgumentParser(description='Bayesian statistics (optional)')
    parser.add_argument('--input', type=str, default='data/parsed/metrics.csv')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    logger = setup_logging('statistics_bayesian')

    try:
        import pymc as pm
        import arviz as az
        HAS_PYMC = True
    except ImportError:
        HAS_PYMC = False
        logger.warning("PyMC not installed. Skipping Bayesian analysis.")
        print("Bayesian analysis requires PyMC. Install with: pip install pymc arviz")
        return

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} records")

    for model in df['model'].unique():
        model_df = df[(df['model'] == model) & df['feasible'] == 1]
        model_name = model.replace('-', '_')
        model_output = os.path.join(args.output, model_name, 'tables')
        os.makedirs(model_output, exist_ok=True)

        # Simple Bayesian model for optimality gap
        formats = model_df['format'].unique()
        format_map = {f: i for i, f in enumerate(formats)}
        data = model_df[model_df['format'].isin(formats)]

        if len(data) > 10:
            try:
                with pm.Model() as model:
                    mu = pm.Normal('mu', mu=0, sigma=10, shape=len(formats))
                    sigma = pm.HalfCauchy('sigma', beta=5)
                    obs = pm.Normal('obs', mu=mu[data['format'].map(format_map)], sigma=sigma,
                                    observed=data['optimality_gap'].values)
                    trace = pm.sample(1000, tune=1000, progressbar=False)
                    summary = az.summary(trace)
                    summary.to_csv(os.path.join(model_output, 'bayesian_results.csv'))
                    logger.info(f"Bayesian analysis complete for {model}")
            except Exception as e:
                logger.error(f"Bayesian analysis failed for {model}: {e}")

    print("Bayesian analysis complete")


if __name__ == '__main__':
    main()