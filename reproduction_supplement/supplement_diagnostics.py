"""Joint posterior checks for the physically constrained SGL/SN model.

Read-only postprocessing of pre-existing nonlinear chains.  Every stored
latent-distance state is paired with the theta state at the exact same
iteration.  The conditional predictive distribution is the SAME independent
Gaussian velocity likelihood used by the joint sampler; the full SN
covariance has already entered the joint latent-distance posterior.

Run with the project-approved sn5070 Python interpreter, CPU/float64.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

from pathlib import Path
import sys
import json
import hashlib
import argparse

import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproduction'))
from physical_prior_revision import PhysicalModel
from inference import C, structural
from latent_validation import loglike

OUT = Path(__file__).resolve().parent / 'diagnostics'
OUT.mkdir(parents=True, exist_ok=True)
INPUT = ROOT / 'reproduction_physical'
PERCENTILES = [2.275, 15.865, 50, 84.135, 97.725]
SEED = 20261021


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_joint(case):
    paths = [INPUT / f'nonlinear_{case}_{suffix}'
             for suffix in ('summary.json', 'chains.npy', 'mu.npy')]
    summary = json.loads(paths[0].read_text(encoding='utf-8'))
    chains = np.load(paths[1], mmap_mode='r')
    latent = np.load(paths[2], mmap_mode='r')
    burn = summary['burn']
    iterations = np.arange(((burn + 99) // 100) * 100, summary['steps'], 100)
    assert chains.shape == (summary['chains'], summary['steps'], 5)
    assert latent.shape == (summary['chains'], len(iterations), 2 * summary['N'])
    theta = np.asarray(chains[:, iterations, :]).reshape(-1, 5)
    mu = np.asarray(latent).reshape(-1, 2 * summary['N'])
    return summary, chains, theta, mu, iterations, paths


def predict_joint(model, theta, mu):
    n = model.n
    q, g0, gz, beta, scatter = theta.T
    g = g0[:, None] + gz[:, None] * model.zl
    xl = 70 / C * 10 ** ((mu[:, :n] - 25) / 5) / (1 + model.zl)
    xs = 70 / C * 10 ** ((mu[:, n:] - 25) / 5) / (1 + model.zs)
    al, ass = 1 + q[:, None] * xl**2, 1 + q[:, None] * xs**2
    assert np.all(al > 0) and np.all(ass > 0)
    ratio = np.sqrt(al) - xl / xs * np.sqrt(ass)
    f = structural(g, model.delta, beta[:, None])
    xi = g + model.delta - 2
    assert np.all((g > 1.05) & (g < 2.95))
    assert np.all((xi > 1) & (xi < 3) & (xi > 2 * beta[:, None]))
    assert np.all(ratio > 0) and np.all(f > 0)
    factor = C**2 / (2 * np.sqrt(np.pi)) * model.te * np.pi / 648000
    sigma = np.sqrt(factor / ratio * f * (model.ta / model.te)**(2 - g))
    variance = model.err**2 + (sigma * scatter[:, None])**2
    assert np.all(np.isfinite(sigma)) and np.all(variance > 0)
    return sigma, variance


def gamma_bands():
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 180})
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    model = PhysicalModel('base')
    zgrid = np.unique(np.r_[np.linspace(0, 1, 151), .6, model.zl.min(), model.zl.max()])
    results = {}
    for case, color, label in [('base', '#1764a0', r'$\gamma_z\in[-0.5,0.5]$'),
                                ('gzwide', '#bd692c', r'$\gamma_z\in[-1,1]$ (diagnostic)')]:
        summary, chains, _, _, _, _ = read_joint(case)
        flat = np.asarray(chains[:, summary['burn']:, :]).reshape(-1, 5)
        quant = np.array([np.percentile(flat[:, 1] + flat[:, 2] * z, PERCENTILES)
                          for z in zgrid])
        table = pd.DataFrame(quant, columns=['q02275', 'q15865', 'q50', 'q84135', 'q97725'])
        table.insert(0, 'lens_redshift', zgrid)
        table.to_csv(OUT / f'gamma_evolution_{case}.csv', index=False)
        at06 = np.percentile(flat[:, 1] + .6 * flat[:, 2], PERCENTILES)
        results[case] = {
            'gamma_at_z0p6_percentiles': dict(zip(map(str, PERCENTILES), at06.tolist())),
            'P_gamma_z_less_than_zero_conditional_on_model_and_prior': float(np.mean(flat[:, 2] < 0)),
            'gamma0_gamma_z_correlation': float(np.corrcoef(flat[:, 1], flat[:, 2])[0, 1]),
            'sampled_lens_redshift_range': [float(model.zl.min()), float(model.zl.max())],
            'bands': 'Pointwise joint-posterior credible bands, not simultaneous bands; extrapolation outside observed lens redshifts.',
        }
        ax.fill_between(zgrid, quant[:, 0], quant[:, 4], color=color, alpha=.07)
        ax.fill_between(zgrid, quant[:, 1], quant[:, 3], color=color, alpha=.22)
        ax.plot(zgrid, quant[:, 2], color=color, lw=1.8, label=label)
    ax.axvspan(0, model.zl.min(), color='gray', alpha=.09)
    ax.axvspan(model.zl.max(), 1, color='gray', alpha=.09)
    ax.text(.99, .02, 'Gray: outside observed lens redshifts', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=8)
    ax.set(xlabel=r'Lens redshift $z_l$', ylabel=r'Total density slope $\gamma(z_l)$', xlim=(0, 1))
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(OUT / 'gamma_evolution.png')
    plt.close(fig)
    return results


def statistical_design(model):
    df = model.df
    predictors = {
        'lens_redshift': model.zl,
        'source_redshift': model.zs,
        'log_effective_angular_radius': np.log(df.thetaeff.to_numpy()),
        'log_einstein_angular_radius': np.log(model.te),
        'log_aperture_over_effective_radius': np.log(model.ta / df.thetaeff.to_numpy()),
        'log_einstein_over_effective_radius': np.log(model.te / df.thetaeff.to_numpy()),
    }
    names = list(predictors)
    weights = np.array([predictors[key] - np.mean(predictors[key]) for key in names])
    weights /= np.sqrt(np.sum(weights**2, axis=1))[:, None]
    surveys = df['Survey name'].astype(str).to_numpy()
    survey_names = sorted(set(surveys))
    masks = np.array([surveys == name for name in survey_names])
    group_weights = masks / masks.sum(axis=1)[:, None]
    return names, weights, survey_names, masks, group_weights


def discrepancies(residuals, trend_weights, group_masks, group_weights):
    """Last dimension is lens; preceding dimensions label posterior/replicates."""
    trend = np.abs(residuals @ trend_weights.T)
    group_means = residuals @ group_weights.T
    overall = residuals.mean(axis=-1, keepdims=True)
    survey = np.sum((group_means - overall)**2 * group_masks.sum(axis=1), axis=-1)
    return np.concatenate([
        np.sum(residuals**2, axis=-1)[..., None],
        np.max(np.abs(residuals), axis=-1)[..., None],
        np.abs(residuals.mean(axis=-1) * np.sqrt(residuals.shape[-1]))[..., None],
        trend,
        np.max(trend, axis=-1)[..., None],
        survey[..., None],
        np.abs(group_means) * np.sqrt(group_masks.sum(axis=1)),
    ], axis=-1)


def posterior_predictive(replicates):
    summary, chains, theta, mu, iterations, paths = read_joint('base')
    model = PhysicalModel('base')
    prediction, variance = predict_joint(model, theta, mu)
    scale = np.sqrt(variance)
    r_obs = (model.obs - prediction) / scale
    args = (model.zl, model.zs, model.te, model.ta, model.delta, model.obs,
            model.err, np.array(model.bounds), False, False)
    rng = np.random.default_rng(SEED)
    selected = rng.choice(len(theta), 100, replace=False)
    discrepancies_ll = []
    for k in selected:
        ll_numba = loglike(theta[k], mu[k], *args)
        ll_numpy = -.5 * np.sum(r_obs[k]**2 + np.log(2 * np.pi * variance[k]))
        assert np.isfinite(ll_numba)
        discrepancies_ll.append(abs(ll_numba - ll_numpy))
    assert max(discrepancies_ll) < 1e-8

    trend_names, weights, survey_names, masks, group_weights = statistical_design(model)
    names = ['standardized_residual_sum_squares', 'maximum_absolute_standardized_residual',
             'absolute_standardized_residual_mean']
    names += ['absolute_residual_trend_' + name for name in trend_names]
    names += ['maximum_absolute_trend_over_six_fixed_predictors']
    names += ['survey_mean_contrast_sum_squares']
    names += ['absolute_group_mean_' + name for name in survey_names]
    observed_stat = discrepancies(r_obs, weights, masks, group_weights)
    # independent noise given each sampled joint state.  Repeated states remain
    # posterior-correlated; the number below is not an effective sample size.
    zrep = rng.normal(size=(replicates, len(theta), model.n))
    replicated_stat = discrepancies(zrep, weights, masks, group_weights)
    indicators = replicated_stat >= observed_stat[None, :, :]
    ppp = indicators.mean(axis=(0, 1))
    by_chain = indicators.mean(axis=0).reshape(summary['chains'], len(iterations), -1).mean(axis=1)
    rows = []
    for j, name in enumerate(names):
        rows.append({
            'discrepancy': name,
            'posterior_predictive_probability_rep_ge_observed': float(ppp[j]),
            'observed_discrepancy_q16_median_q84': np.percentile(observed_stat[:, j], [15.865, 50, 84.135]).tolist(),
            'replicated_discrepancy_q16_median_q84': np.percentile(replicated_stat[:, :, j], [15.865, 50, 84.135]).tolist(),
            'chainwise_posterior_predictive_probabilities': by_chain[:, j].tolist(),
        })
    pd.DataFrame(rows).to_csv(OUT / 'posterior_predictive_checks.csv', index=False)

    qs = np.percentile(prediction, [15.865, 50, 84.135], axis=0)
    qres = np.percentile(r_obs, [15.865, 50, 84.135], axis=0)
    yrep = prediction[None, :, :] + scale[None, :, :] * zrep
    qpred = np.percentile(yrep.reshape(-1, model.n), [2.275, 15.865, 50, 84.135, 97.725], axis=0)
    per_lens = model.df[['Lens name', 'Survey name', 'zl', 'zs', 'thetaE', 'thetaeff', 'thetaap', 'sigma_ap', 'dsigma_ap', 'delta']].copy()
    for j, name in enumerate(['predicted_mean_sigma_q16', 'predicted_mean_sigma_median', 'predicted_mean_sigma_q84']):
        per_lens[name] = qs[j]
    for j, name in enumerate(['predictive_sigma_q02275', 'predictive_sigma_q15865', 'predictive_sigma_q50', 'predictive_sigma_q84135', 'predictive_sigma_q97725']):
        per_lens[name] = qpred[j]
    for j, name in enumerate(['standardized_residual_q16', 'standardized_residual_median', 'standardized_residual_q84']):
        per_lens[name] = qres[j]
    per_lens['conditional_noise_scale_median_km_s'] = np.median(scale, axis=0)
    per_lens['posterior_mean_conditional_two_sided_tail_probability'] = (2 * ndtr(-np.abs(r_obs))).mean(axis=0)
    per_lens.to_csv(OUT / 'per_lens_predictions_and_residuals.csv', index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    colors = ['#1764a0', '#bd692c', '#49835b', '#9b5597']
    for k, name in enumerate(survey_names):
        mask = masks[k]
        axes[0, 0].errorbar(qpred[2, mask], model.obs[mask],
                            xerr=[qpred[2, mask] - qpred[1, mask], qpred[3, mask] - qpred[2, mask]],
                            yerr=model.err[mask], fmt='o', ms=3.2, alpha=.75, capsize=0,
                            color=colors[k % len(colors)], label=f'{name} ({mask.sum()})')
        for ax, x in [(axes[0, 1], model.zl), (axes[0, 2], model.zs),
                       (axes[1, 0], np.log10(model.df.thetaeff)),
                       (axes[1, 1], model.ta / model.df.thetaeff),
                       (axes[1, 2], model.te / model.df.thetaeff)]:
            ax.errorbar(x[mask], qres[1, mask], yerr=[qres[1, mask] - qres[0, mask], qres[2, mask] - qres[1, mask]],
                        fmt='o', ms=3.2, alpha=.8, color=colors[k % len(colors)])
    low = min(model.obs.min() - model.err.max(), qpred[1].min()) - 10
    high = max(model.obs.max() + model.err.max(), qpred[3].max()) + 10
    axes[0, 0].plot([low, high], [low, high], 'k:', lw=1)
    axes[0, 0].set(xlabel=r'Posterior predictive $\sigma$ (km s$^{-1}$)',
                   ylabel=r'Observed $\sigma_{\rm ap}$ (km s$^{-1}$)', xlim=(low, high), ylim=(low, high))
    axes[0, 0].legend(fontsize=7, loc='upper left')
    axes[0, 1].set(xlabel='Lens redshift')
    axes[0, 2].set(xlabel='Source redshift')
    axes[1, 0].set(xlabel=r'$\log_{10}(\theta_{\rm eff}/{\rm arcsec})$')
    axes[1, 1].set(xlabel=r'$\theta_{\rm ap}/\theta_{\rm eff}$')
    axes[1, 2].set(xlabel=r'$\theta_E/\theta_{\rm eff}$')
    for ax in [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
        ax.axhline(0, color='black', ls=':', lw=1)
        ax.set(ylabel='Conditional standardized residual')
    for ax in axes.ravel(): ax.grid(alpha=.13)
    fig.tight_layout()
    fig.savefig(OUT / 'posterior_predictive_residuals.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, j, label in [(axes[0], 0, r'$\sum_i r_i^2$'), (axes[1], 1, r'$\max_i |r_i|$')]:
        combined = np.r_[observed_stat[:, j], replicated_stat[:, :, j].ravel()]
        bins = np.linspace(np.percentile(combined, .05), np.percentile(combined, 99.95), 45)
        ax.hist(replicated_stat[:, :, j].ravel(), bins=bins, density=True, alpha=.5, color='#1764a0', label='Replicated data')
        ax.hist(observed_stat[:, j], bins=bins, density=True, histtype='step', lw=1.6, color='#bd692c', label='Observed data')
        ax.set(xlabel=label, ylabel='Posterior density')
        ax.text(.98, .93, f'Predictive probability = {ppp[j]:.3f}', ha='right', transform=ax.transAxes, fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'posterior_predictive_discrepancies.png')
    plt.close(fig)

    worst = per_lens.sort_values('posterior_mean_conditional_two_sided_tail_probability').head(5)
    source_detail = source_redshift_checks(model, r_obs, zrep, theta, mu, weights, trend_names)
    return {
        'N_lenses': model.n,
        'posterior_joint_states': len(theta),
        'joint_states_per_chain': len(iterations),
        'saved_iteration_first_last_stride': [int(iterations[0]), int(iterations[-1]), 100],
        'theta_mu_alignment': 'Exact same chain and iteration; theta is stored after the accepted elliptical latent-distance update.',
        'conditional_replications_per_state': replicates,
        'total_replicated_datasets': len(theta) * replicates,
        'seed': SEED,
        'max_absolute_vectorized_loglike_discrepancy': float(max(discrepancies_ll)),
        'input_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        'conditional_model': 'sigma_obs_i ~ Normal(sigma_model_i(theta, mu), dsigma_ap_i^2 + (scatter*sigma_model_i)^2); mu drawn from the fitted joint posterior with full SN covariance and the assumed mismatch envelope.',
        'prediction_intervals': 'Central pointwise posterior intervals; velocity predictive intervals include conditional measurement plus intrinsic scatter.',
        'checks': rows,
        'surveys': dict(zip(survey_names, masks.sum(axis=1).astype(int).tolist())),
        'five_smallest_local_tail_probabilities': worst[['Lens name', 'posterior_mean_conditional_two_sided_tail_probability', 'standardized_residual_median']].to_dict(orient='records'),
        'source_redshift_trend_detail': source_detail,
        'interpretation': [
            'Posterior predictive probabilities reuse the fitted data and are not calibrated frequentist p-values or independent model validation.',
            'The examined residual trends are descriptive checks, with correlated predictors and no multiplicity-adjusted discovery claim.',
            'Small or large values indicate a mismatch for the chosen discrepancy, rather than a unique physical cause.',
            'Latent distances have been updated using the same lens velocities; this is an in-sample conditional check, not a held-out prediction.',
            'Angular radii and their ratios are observed fixed predictors; no physical-size inference is substituted for these quantities.',
            'Monte Carlo counts are not effective posterior sample sizes; chainwise probabilities are supplied to assess stability.',
        ],
    }


def source_redshift_checks(model, residual, replicated, theta, mu, trend_weights, trend_names):
    """Descriptive influence checks WITHOUT refitting or leave-one-out claims."""
    zs = model.zs
    x = zs - zs.mean()
    w = x / np.linalg.norm(x)
    obs_score = residual @ w
    rep_score = replicated @ w
    n = model.n
    # Exact recentered, renormalized trend score after omitting each point.
    numerator_obs = (residual @ x)[:, None] - n / (n - 1) * residual * x + residual.sum(axis=1)[:, None] * x / (n - 1)
    numerator_rep = (replicated @ x)[..., None] - n / (n - 1) * replicated * x + replicated.sum(axis=-1)[..., None] * x / (n - 1)
    denominator = np.sqrt(np.sum(x**2) - x**2 - x**2 / (n - 1))
    obs_omit = numerator_obs / denominator
    rep_omit = numerator_rep / denominator
    p_omit = (np.abs(rep_omit) >= np.abs(obs_omit)[None, :, :]).mean(axis=(0, 1))
    # Validate the algebra against a direct calculation for several omissions.
    for omit in [0, model.n // 2, model.n - 1]:
        keep = np.arange(n) != omit
        direct_w = zs[keep] - zs[keep].mean()
        direct_w /= np.linalg.norm(direct_w)
        assert np.allclose(obs_omit[:, omit], residual[:, keep] @ direct_w, atol=1e-12)
    contributions = residual * w
    detail = model.df[['Lens name', 'Survey name', 'zl', 'zs', 'sigma_ap', 'dsigma_ap']].copy()
    detail['source_z_centered_normalized_weight'] = w
    detail['signed_trend_contribution_median'] = np.median(contributions, axis=0)
    detail['standardized_residual_median'] = np.median(residual, axis=0)
    detail['omit_this_lens_predictive_probability_no_refit'] = p_omit
    detail['omitted_signed_trend_median_no_refit'] = np.median(obs_omit, axis=0)
    detail.to_csv(OUT / 'source_redshift_trend_influence_no_refit.csv', index=False)
    correlations = np.corrcoef(trend_weights)
    pd.DataFrame(correlations, columns=trend_names, index=trend_names).to_csv(OUT / 'residual_predictor_correlations.csv')

    quantile_group = pd.qcut(zs, q=4, labels=False)
    bin_table = []
    for j in range(4):
        mask = quantile_group == j
        oq = np.percentile(residual[:, mask].mean(axis=1), PERCENTILES)
        rq = np.percentile(replicated[:, :, mask].mean(axis=-1), PERCENTILES)
        bin_table.append({'bin': j + 1, 'N': int(mask.sum()),
                          'z_min': float(zs[mask].min()), 'z_max': float(zs[mask].max()),
                          'z_mean': float(zs[mask].mean()),
                          **{f'observed_mean_r_q{p}': float(v) for p, v in zip(PERCENTILES, oq)},
                          **{f'replicated_mean_r_q{p}': float(v) for p, v in zip(PERCENTILES, rq)}})
    bins = pd.DataFrame(bin_table)
    bins.to_csv(OUT / 'source_redshift_binned_residuals.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].scatter(zs, np.median(residual, axis=0), s=17, alpha=.65, color='#777777')
    for j, row in bins.iterrows():
        left, right = row.z_min, row.z_max
        axes[0].fill_between([left, right], row['replicated_mean_r_q2.275'], row['replicated_mean_r_q97.725'], color='#1764a0', alpha=.16, label='Replicated bin mean: 95.45%' if j == 0 else None)
        axes[0].errorbar(row.z_mean, row['observed_mean_r_q50'],
                        yerr=[[row['observed_mean_r_q50'] - row['observed_mean_r_q15.865']],
                              [row['observed_mean_r_q84.135'] - row['observed_mean_r_q50']]],
                        fmt='s', color='#bd692c', capsize=3, ms=6,
                        label='Observed bin mean: 68.27%' if j == 0 else None)
    axes[0].axhline(0, color='black', lw=1, ls=':')
    axes[0].set(xlabel='Source redshift', ylabel='Conditional standardized residual')
    axes[0].legend(fontsize=7, loc='lower left')
    histbins = np.linspace(-5.5, 4.5, 50)
    axes[1].hist(rep_score.ravel(), bins=histbins, density=True, color='#1764a0', alpha=.5, label='Replicated data')
    axes[1].hist(obs_score, bins=histbins, density=True, color='#bd692c', histtype='step', lw=1.7, label='Observed data')
    axes[1].set(xlabel='Signed standardized source-redshift trend', ylabel='Posterior density')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'source_redshift_predictive_check.png')
    plt.close(fig)

    # A bounded residual diagnostic reduces sensitivity to a single outlier.
    robust_obs = np.clip(residual, -2, 2) @ w
    robust_rep = np.clip(replicated, -2, 2) @ w
    robust_ppp = float(np.mean(np.abs(robust_rep) >= np.abs(robust_obs)[None, :]))
    # Same check at the wider gamma prior: this is a diagnostic comparison,
    # not a data-selected replacement for the primary physical prior.
    _, _, theta_wide, mu_wide, _, _ = read_joint('gzwide')
    prediction_wide, variance_wide = predict_joint(PhysicalModel('gzwide'), theta_wide, mu_wide)
    rwide = (model.obs - prediction_wide) / np.sqrt(variance_wide)
    wide_obs = rwide @ w
    wide_rep = np.random.default_rng(SEED + 1).normal(size=replicated.shape) @ w
    wide_ppp = float(np.mean(np.abs(wide_rep) >= np.abs(wide_obs)[None, :]))
    shift_source = mu[:, model.n:] - model.ms
    return {
        'signed_score_definition': 'sum_i (z_s,i - mean(z_s))*r_i / sqrt(sum_i (z_s,i - mean(z_s))^2); r_i=(sigma_obs-sigma_model)/sqrt(dsigma_ap^2+(scatter*sigma_model)^2).',
        'direction': 'negative' if np.median(obs_score) < 0 else 'positive',
        'signed_observed_score_q16_median_q84': np.percentile(obs_score, [15.865, 50, 84.135]).tolist(),
        'each_single_point_omission_predictive_probability_min_max_no_refit': [float(p_omit.min()), float(p_omit.max())],
        'point_whose_omission_most_reduces_discrepancy': detail.iloc[int(np.argmax(p_omit))].to_dict(),
        'clipped_at_abs_r2_source_trend_predictive_probability': robust_ppp,
        'gzwide_source_trend_predictive_probability': wide_ppp,
        'gzwide_signed_score_q16_median_q84': np.percentile(wide_obs, [15.865, 50, 84.135]).tolist(),
        'latent_source_mu_shift_from_SN_mean_median_range_mag': [float(np.median(shift_source, axis=0).min()), float(np.median(shift_source, axis=0).max())],
        'caveat': 'Single-point omission recomputes the discrepancy on the existing full-data posterior only; it is NOT leave-one-out prediction, influence reweighting, or a refit. Clipping is a descriptive post-hoc robustness check. No causal mechanism is identified.',
    }


def manuscript_text(result):
    checks = {row['discrepancy']: row for row in result['posterior_predictive']['checks']}
    ppp = lambda key: checks[key]['posterior_predictive_probability_rep_ge_observed']
    normal = result['gamma_evolution']['base']['gamma_at_z0p6_percentiles']
    wide = result['gamma_evolution']['gzwide']['gamma_at_z0p6_percentiles']
    interval = lambda q: f"{q['50']:.3f} (+{q['84.135']-q['50']:.3f}, -{q['50']-q['15.865']:.3f})"
    trends = [(key.replace('absolute_residual_trend_', ''), ppp(key)) for key in checks if key.startswith('absolute_residual_trend_')]
    text = (
        "We propagated the joint posterior of the density-slope intercept and redshift coefficient to gamma(z), rather than varying the coefficient at a fixed intercept. "
        f"At z_l=0.6, the central 68.27% interval is gamma={interval(normal)} under the adopted gamma_z in [-0.5,0.5] prior, "
        f"and gamma={interval(wide)} in the wider-prior diagnostic with the same CMB geometric constraint. "
        "These are pointwise, model-conditional intervals; portions of the plotted band outside the observed lens-redshift range are extrapolations.\n\n"
        f"We generated replicated velocity-dispersion datasets at {result['posterior_predictive']['posterior_joint_states']} precisely paired joint parameter and latent-distance posterior states, "
        "using the same measurement-plus-intrinsic-scatter likelihood as in the fit. "
        f"The posterior predictive probabilities P(T_rep >= T_obs | data) are {ppp('standardized_residual_sum_squares'):.3f} for the sum of squared standardized residuals, "
        f"{ppp('maximum_absolute_standardized_residual'):.3f} for their maximum absolute value, and {ppp('survey_mean_contrast_sum_squares'):.3f} for the survey-mean contrast. "
        "Absolute residual-trend discrepancies give " + ', '.join(f'{name}: {p:.3f}' for name, p in trends) + '. '
        "The residual trend with source redshift is negative and is unusually large relative to replicated datasets, even though the overall residual sum of squares is compatible with the model. "
        f"The maximum absolute trend over the six fixed predictors has posterior predictive probability {ppp('maximum_absolute_trend_over_six_fixed_predictors'):.4f}. "
        "Recomputing the source-redshift discrepancy after omitting each individual lens, while retaining the fitted full-data posterior, gives probabilities ranging from "
        f"{result['posterior_predictive']['source_redshift_trend_detail']['each_single_point_omission_predictive_probability_min_max_no_refit'][0]:.4f} to "
        f"{result['posterior_predictive']['source_redshift_trend_detail']['each_single_point_omission_predictive_probability_min_max_no_refit'][1]:.4f}; this is a descriptive stability check without refitting, not leave-one-out validation. "
        "The individual discrepancies and per-lens predictions are supplied as supplementary material. "
        "These in-sample checks reuse the observations, including their information on latent distances; they are not calibrated frequentist tests, independent validation, or a demonstration that lens-model systematics are controlled. "
        "In particular, any discrepant trend motivates further modeling and cannot by itself be attributed to external convergence or selection effects.\n"
    )
    (OUT / 'manuscript_paragraphs_en.txt').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicates', type=int, default=32)
    args = parser.parse_args()
    assert args.replicates >= 1
    result = {'method_version': '1.0', 'gamma_evolution': gamma_bands(),
              'posterior_predictive': posterior_predictive(args.replicates)}
    manuscript_text(result)
    (OUT / 'diagnostics_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'gamma_evolution': result['gamma_evolution'],
                      'posterior_predictive_checks': result['posterior_predictive']['checks'],
                      'output': str(OUT)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
