"""Analytic Gaussian-tail confirmation and a descriptive partial-zs check.

The posterior sample is unchanged. A normalized fixed linear combination of
the replicated conditional standardized residuals is exactly N(0,1). Thus
the replicate tail probability can be integrated analytically at each joint
posterior state; no replication Monte Carlo noise is needed for these scores.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from supplement_diagnostics import read_joint, predict_joint, OUT, PhysicalModel


def tail_summary(score, nchains):
    probabilities = 2 * ndtr(-np.abs(score))
    per_chain = probabilities.reshape(nchains, -1).mean(axis=1)
    return {
        'signed_observed_score_q16_median_q84': np.percentile(score, [15.865, 50, 84.135]).tolist(),
        'analytic_posterior_predictive_probability': float(probabilities.mean()),
        'chainwise_analytic_probabilities': per_chain.tolist(),
        'conditional_replication_distribution': 'Exactly standard normal, given theta and mu; normalized deterministic weights have squared norm one.',
    }


def normalized_source_weights(model, partial=False):
    zs = model.zs
    if partial:
        surveys = model.df['Survey name'].astype(str)
        dummies = pd.get_dummies(surveys, drop_first=True).to_numpy(dtype=float)
        design = np.column_stack([np.ones(model.n), model.zl, dummies])
        coefficients, _, rank, _ = np.linalg.lstsq(design, zs, rcond=None)
        x = zs - design @ coefficients
        assert rank == design.shape[1]
        assert np.max(np.abs(design.T @ x)) < 1e-10
    else:
        design = np.ones((model.n, 1))
        x = zs - zs.mean()
    weight = x / np.linalg.norm(x)
    assert abs(np.sum(weight**2) - 1) < 1e-12
    return weight, design


def main():
    model = PhysicalModel('base')
    summary, _, theta, mu, iterations, _ = read_joint('base')
    prediction, variance = predict_joint(model, theta, mu)
    residual = (model.obs - prediction) / np.sqrt(variance)
    w, _ = normalized_source_weights(model)
    raw = tail_summary(residual @ w, summary['chains'])
    wp, design = normalized_source_weights(model, partial=True)
    partial = tail_summary(residual @ wp, summary['chains'])

    n = model.n
    x = model.zs - model.zs.mean()
    numerator = (residual @ x)[:, None] - n / (n - 1) * residual * x + residual.sum(axis=1)[:, None] * x / (n - 1)
    denominator = np.sqrt(np.sum(x**2) - x**2 - x**2 / (n - 1))
    omit_score = numerator / denominator
    omitted_p = (2 * ndtr(-np.abs(omit_score))).mean(axis=0)
    influences = model.df[['Lens name', 'Survey name', 'zl', 'zs']].copy()
    influences['analytic_source_trend_probability_omit_one_no_refit'] = omitted_p
    influences['source_weight_partial_on_survey_and_zl'] = wp
    influences.to_csv(OUT / 'analytic_source_redshift_influence.csv', index=False)

    broad_summary, _, broad_theta, broad_mu, _, _ = read_joint('gzwide')
    bp, bv = predict_joint(PhysicalModel('gzwide'), broad_theta, broad_mu)
    br = (model.obs - bp) / np.sqrt(bv)
    broad = {'raw_source_redshift': tail_summary(br @ w, broad_summary['chains']),
             'partial_source_redshift': tail_summary(br @ wp, broad_summary['chains'])}
    result = {
        'primary_raw_source_redshift': raw,
        'primary_partial_source_redshift': partial,
        'wider_gamma_prior_diagnostic': broad,
        'primary_analytic_single_point_omission_no_refit_probability_min_max': [float(omitted_p.min()), float(omitted_p.max())],
        'partial_weight_construction': 'Residualize fixed source-redshift vector on intercept, observed lens redshift, and survey indicator columns (one reference group), then normalize Euclidean norm to one.',
        'partial_design_rank': int(np.linalg.matrix_rank(design)),
        'partial_design_columns': int(design.shape[1]),
        'max_abs_partial_weight_inner_product_with_design': float(np.max(np.abs(design.T @ wp))),
        'joint_states': len(theta),
        'sample_iteration_first_last': [int(iterations[0]), int(iterations[-1])],
        'interpretation': 'A descriptive residual projection conditional on the fitted model. Removing linear lens-redshift and additive survey-group projections is not causal adjustment, independent validation, or a frequentist significance test. No parameters were refitted. Posterior Monte Carlo dependence remains, although conditional replication noise has been eliminated analytically.',
    }
    (OUT / 'analytic_source_trend_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
