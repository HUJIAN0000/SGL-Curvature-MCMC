"""Conditional predictive checks of the new matching-sensitivity runs.

Read-only operation. The existing summary explicitly stores the iterations
at which latent-distance arrays were saved; use those exact theta states.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from run_sensitivity import SupplementModel
from supplement_diagnostics import predict_joint, loglike, sha256
from analytic_source_trend_checks import normalized_source_weights, tail_summary

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'diagnostics'


def check_case(case, replications=32):
    folder = ROOT / 'sensitivity' / case
    summary_path = folder / 'nonlinear_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'Completed summary not available: {summary_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    chain_path = folder / 'nonlinear_chains.npy'
    mu_path = folder / 'nonlinear_mu.npy'
    chains = np.load(chain_path, mmap_mode='r')
    mus = np.load(mu_path, mmap_mode='r')
    indices = np.asarray(summary['mu_saved_indices'], dtype=int)
    assert np.all(indices >= summary['burn']) and np.all(indices < summary['steps'])
    assert np.all(np.diff(indices) > 0)
    assert chains.shape == (summary['chains'], summary['steps'], 5)
    assert mus.shape == (summary['chains'], len(indices), 2 * summary['N'])
    theta = np.asarray(chains[:, indices, :]).reshape(-1, 5)
    mu = np.asarray(mus).reshape(-1, 2 * summary['N'])
    model = SupplementModel(folder)
    assert model.n == summary['N']
    prediction, variance = predict_joint(model, theta, mu)
    residual = (model.obs - prediction) / np.sqrt(variance)
    rng = np.random.default_rng(20261021 + sum(map(ord, case)))
    args = (model.zl, model.zs, model.te, model.ta, model.delta,
            model.obs, model.err, np.asarray(model.bounds), False, False)
    max_error = 0.
    for k in rng.choice(len(theta), min(30, len(theta)), replace=False):
        exact = loglike(theta[k], mu[k], *args)
        vectorized = -.5 * np.sum(residual[k]**2 + np.log(2 * np.pi * variance[k]))
        assert np.isfinite(exact)
        max_error = max(max_error, abs(exact - vectorized))
    assert max_error < 1e-8
    w, _ = normalized_source_weights(model)
    wp, _ = normalized_source_weights(model, partial=True)
    raw = tail_summary(residual @ w, summary['chains'])
    partial = tail_summary(residual @ wp, summary['chains'])
    total_observed = np.sum(residual**2, axis=1)
    # A sum of n independent squared standard normals. Random replication
    # matches the primary diagnostic convention; exact survival probability
    # is additionally supplied for verification.
    total_replicated = rng.chisquare(model.n, size=(replications, len(theta)))
    total_indicators = total_replicated >= total_observed[None, :]
    p_total = float(total_indicators.mean())
    from scipy.special import gammaincc
    total_analytic = gammaincc(model.n / 2, total_observed / 2)
    design = json.loads((folder / 'design.json').read_text(encoding='utf-8'))
    output = {
        'case': case, 'N_lenses': model.n,
        'sigma_mu_max': design['sigma_mu_max'],
        'relative_distance_tolerance': design['relative_distance_tolerance'],
        'joint_states': len(theta), 'saved_theta_indices_from_summary': True,
        'saved_indices_first_last': [int(indices[0]), int(indices[-1])],
        'source_redshift_trend': raw,
        'partial_source_redshift_trend': partial,
        'squared_standardized_residual_sum': {
            'replicate_posterior_predictive_probability': p_total,
            'analytic_posterior_predictive_probability': float(total_analytic.mean()),
            'replicate_chainwise_probabilities': total_indicators.mean(axis=0).reshape(summary['chains'], -1).mean(axis=1).tolist(),
            'analytic_chainwise_probabilities': total_analytic.reshape(summary['chains'], -1).mean(axis=1).tolist(),
            'observed_discrepancy_q16_median_q84': np.percentile(total_observed, [15.865, 50, 84.135]).tolist(),
            'replications_per_state': replications,
            'conditional_replication_distribution': f'Chi-square with n={model.n}; exactly equivalent to generating n independent conditional Gaussian standardized residuals.',
        },
        'max_absolute_vectorized_loglike_discrepancy': float(max_error),
        'hashes': {path.name: sha256(path) for path in [summary_path, chain_path, mu_path]},
        'interpretation': 'In-sample conditional predictive checks, not held-out or frequentist validation. Source-trend tail uses the exact N(0,1) conditional replicate distribution. Partial trend residualizes fixed zs on intercept, zl and additive survey groups, without causal interpretation. Original posterior states and chains are unchanged.',
    }
    (OUT / f'sensitivity_predictive_{case}.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='quality025,quality035,tol03,tol07')
    parser.add_argument('--replications', type=int, default=32)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    results = [check_case(case, args.replications) for case in args.cases.split(',')]
    rows = []
    for result in results:
        rows.append({
            'case': result['case'], 'N_lenses': result['N_lenses'],
            'sigma_mu_max': result['sigma_mu_max'],
            'relative_distance_tolerance': result['relative_distance_tolerance'],
            'joint_states': result['joint_states'],
            'source_redshift_trend_analytic_probability': result['source_redshift_trend']['analytic_posterior_predictive_probability'],
            'signed_source_trend_median': result['source_redshift_trend']['signed_observed_score_q16_median_q84'][1],
            'partial_source_redshift_trend_analytic_probability': result['partial_source_redshift_trend']['analytic_posterior_predictive_probability'],
            'squared_residual_sum_replicate_probability': result['squared_standardized_residual_sum']['replicate_posterior_predictive_probability'],
            'squared_residual_sum_analytic_probability': result['squared_standardized_residual_sum']['analytic_posterior_predictive_probability'],
        })
    pd.DataFrame(rows).to_csv(OUT / 'sensitivity_predictive_comparison.csv', index=False)
    (OUT / 'sensitivity_predictive_checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__': main()
