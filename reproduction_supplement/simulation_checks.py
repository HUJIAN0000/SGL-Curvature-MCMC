"""Fixed-design, known-truth checks for the nonlinear main likelihood.

This is a small inference sanity check, not a frequentist coverage study and
not an end-to-end test of the survey selection or the galaxy dynamical model.
All products are written below this script's simulations/ directory.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'reproduction'))
from physical_prior_revision import PhysicalModel, LOWER
from latent_validation import loglike
from inference import C, structural

OUT = HERE / 'simulations'
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ['Omega_k', 'gamma0', 'gamma_z', 'beta', 'scatter']
TRUTHS = {
    'constant': np.array([0., 2.05, 0., 0., .11]),
    'evolving': np.array([0., 2.05, -.2, 0., .11]),
}


def fiducial_mu(z):
    """Flat FLRW luminosity distance at the *target* redshift; H0=70."""
    z = np.asarray(z, dtype=np.float64)
    chi = np.array([quad(lambda t: 1 / np.sqrt(.3 * (1 + t)**3 + .7),
                         0, float(zi), epsabs=1e-12, epsrel=1e-12)[0]
                    for zi in z], dtype=np.float64)
    dl = (C / 70.) * (1 + z) * chi
    return 25 + 5 * np.log10(dl), chi


def prepare_data(case, realization):
    model = PhysicalModel('base')
    truth = TRUTHS[case].copy()
    seed = 20261100 + 100 * list(TRUTHS).index(case) + realization
    rng = np.random.default_rng(seed)
    mu_truth, chi = fiducial_mu(np.r_[model.zl, model.zs])
    cov = np.block([[model.cl, model.cls], [model.cls.T, model.cs]])
    # The same negligible numerical jitter used by the original main sampler.
    chol = np.linalg.cholesky(cov + np.eye(2 * model.n) * 1e-13)
    noise_sn = rng.normal(size=2 * model.n)
    mu_observed = mu_truth + chol @ noise_sn

    g = truth[1] + truth[2] * model.zl
    ratio = 1 - chi[:model.n] / chi[model.n:]
    sigma2 = (C**2 / (2 * np.sqrt(np.pi)) * model.te * np.pi / 648000
              / ratio * structural(g, model.delta, truth[3])
              * (model.ta / model.te)**(2 - g))
    sigma_truth = np.sqrt(sigma2)
    noise_velocity = rng.normal(size=model.n)
    velocity_sd = np.sqrt(model.err**2 + sigma2 * truth[4]**2)
    observed_velocity = sigma_truth + velocity_sd * noise_velocity

    # Independently check that the model's distance-sum rule equals the FLRW
    # angular-distance ratio for the noiseless truth used above.
    model.ml, model.ms = mu_truth[:model.n], mu_truth[model.n:]
    model.xl = 70 / C * 10**((model.ml - 25) / 5) / (1 + model.zl)
    model.xs = 70 / C * 10**((model.ms - 25) / 5) / (1 + model.zs)
    predicted, valid = model.predict(truth)
    np.testing.assert_allclose(predicted[0], sigma_truth, rtol=2e-13, atol=1e-10)
    assert valid[0] and np.all(np.isfinite(sigma_truth))

    model.ml, model.ms = mu_observed[:model.n], mu_observed[model.n:]
    model.xl = 70 / C * 10**((model.ml - 25) / 5) / (1 + model.zl)
    model.xs = 70 / C * 10**((model.ms - 25) / 5) / (1 + model.zs)
    model.obs = observed_velocity
    value = loglike(truth, mu_truth, model.zl, model.zs, model.te, model.ta,
                    model.delta, model.obs, model.err, np.array(model.bounds),
                    False, False)
    assert np.isfinite(value)
    name = f'{case}_{realization + 1:02d}'
    np.savez_compressed(OUT / f'{name}_data.npz',
                        truth=truth, mu_truth=mu_truth, mu_observed=mu_observed,
                        covariance=cov, covariance_cholesky=chol,
                        sn_standard_noise=noise_sn,
                        sigma_truth=sigma_truth, observed_velocity=model.obs,
                        velocity_standard_noise=noise_velocity,
                        velocity_sd=velocity_sd, zl=model.zl, zs=model.zs,
                        thetaE=model.te, thetaap=model.ta, delta=model.delta,
                        dsigma=model.err, bounds=np.array(model.bounds))
    metadata = {
        'name': name, 'data_seed': seed, 'N_lenses': model.n,
        'labels': LABELS, 'truth': truth.tolist(),
        'fiducial_FLRW': {'Omega_m': .3, 'Omega_k': 0., 'H0_km_s_Mpc': 70.},
        'truth_redshifts': 'actual target lens/source redshifts, not noisy SN redshifts',
        'fixed_design': 'actual 88-lens redshifts, slopes delta, apertures, Einstein radii and velocity errors',
        'SN_factor': 'full selected 176x176 covariance plus the main analysis independent mismatch envelope',
        'SN_mean': 'coherent fiducial FLRW truth plus one full-covariance Gaussian noise realization',
        'velocity_factor': 'independent Gaussian with variance dsigma^2 + (scatter*sigma_truth)^2',
        'CMB_geometric_lower': LOWER,
        'gamma_z_bounds': [-.5, .5],
        'finite_loglike_at_noiseless_distance_truth': float(value),
        'minimum_covariance_eigenvalue': float(np.linalg.eigvalsh(cov).min()),
        'maximum_velocity_truth_crosscheck_relative_error': float(np.max(abs(predicted[0]/sigma_truth - 1))),
        'limitations': [
            'Fixed-design conditional likelihood check; matching and survey selection are held fixed.',
            'Uses the assumed main Gaussian SN factor, including its symmetric mismatch envelope.',
            'Uses exactly the assumed P2 velocity model; cannot validate omitted galaxy physics.',
            'A few noise realizations do not estimate frequentist coverage or population selection bias.',
            'CMB and gamma priors can shift posterior medians even for correctly generated data.',
        ],
    }
    (OUT / f'{name}_data.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return model, metadata


def summarize():
    rows, results = [], []
    for path in sorted(OUT.glob('*_summary.json')):
        if path.name == 'aggregate_summary.json':
            continue
        name = path.name.removesuffix('_summary.json')
        data_file = OUT / f'{name}_data.json'
        if not data_file.exists():
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        meta = json.loads(data_file.read_text(encoding='utf-8'))
        chains = np.load(OUT / f'{name}_chains.npy', mmap_mode='r')
        flat = np.asarray(chains[:, record['burn']:]).reshape(-1, 5)
        quant = np.percentile(flat, [2.275, 15.865, 50, 84.135, 97.725], axis=0)
        diag = record['diagnostics']
        item = {'name': name, 'truth': meta['truth'], 'quantiles_95_68': quant.tolist(),
                'max_split_rhat': float(max(diag['split_rhat'])),
                'max_rank_folded_or_split_rhat': float(diag.get('max_rhat', max(diag['split_rhat']))),
                'min_conservative_ess': float(min(diag['conservative_ess'])),
                'mass_gamma_z_below_minus_0p45': float(np.mean(flat[:, 2] < -.45)),
                'mass_gamma_z_above_plus_0p45': float(np.mean(flat[:, 2] > .45)),
                'mass_Omega_k_below_lower_plus_0p02': float(np.mean(flat[:, 0] < LOWER + .02)),
                'mass_Omega_k_above_1p9': float(np.mean(flat[:, 0] > 1.9)),
                'posterior_p_gamma_z_below_zero': float(np.mean(flat[:, 2] < 0))}
        results.append(item)
        for p, label in enumerate(LABELS):
            rows.append({'realization': name, 'parameter': label,
                         'truth': meta['truth'][p], 'q2p275': quant[0,p],
                         'q15p865': quant[1,p], 'median': quant[2,p],
                         'q84p135': quant[3,p], 'q97p725': quant[4,p],
                         'split_rhat': diag['split_rhat'][p],
                         'conservative_ess': diag['conservative_ess'][p]})
    if not results:
        return
    pd.DataFrame(rows).to_csv(OUT / 'recovery_intervals.csv', index=False)
    (OUT / 'aggregate_summary.json').write_text(json.dumps({
        'purpose': 'small fixed-design known-truth nonlinear-likelihood sanity check',
        'interpretation': 'intervals for separate noise realizations; no coverage calibration is claimed',
        'results': results,
    }, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8))
    y = np.arange(len(results))
    for ax, p in zip(axes, [0, 2]):
        for j, result in enumerate(results):
            q = np.array(result['quantiles_95_68'])[:, p]
            color = '#2166ac' if result['name'].startswith('constant') else '#b35806'
            ax.plot(q[[0,4]], [j,j], color=color, lw=1.2)
            ax.plot(q[[1,3]], [j,j], color=color, lw=4)
            ax.plot(q[2], j, 'o', color=color, ms=5)
            ax.plot(result['truth'][p], j, marker='|', color='black', ms=15, mew=1.6)
        ax.set_yticks(y, [r['name'].replace('_', ' ') for r in results])
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel(r'$\Omega_k$' if p == 0 else r'$\gamma_z$')
        ax.axvline(LOWER if p == 0 else -.5, color='.5', ls=':', lw=1)
    axes[0].set_title('Curvature')
    axes[1].set_title('Slope evolution')
    fig.suptitle('Fixed-design simulation check: 68.27% / 95.45% intervals; black marks = truth', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(OUT / 'simulation_recovery.png', dpi=180)
    fig.savefig(OUT / 'simulation_recovery.pdf')
    plt.close(fig)
    text = '''Known-truth fixed-design simulation checks

The design is held at the actual 88-lens redshifts, light slopes, apertures,
Einstein radii and velocity errors. Flat LambdaCDM (Omega_m=0.3, H0=70) gives
coherent noiseless luminosity distances at the target lens/source redshifts.
The supernova Gaussian means are then perturbed with the full selected
covariance, including the same independent mismatch envelope assumed by the
main analysis. Velocity data are independently generated from the assumed P2
model with the input scatter and measurement variance. Inputs gamma_z=0 and
gamma_z=-0.2 share Omega_k=0, gamma0=2.05, beta=0, scatter=0.11.

Each realization is fitted with the same nonlinear latent-distance likelihood,
hard CMB geometric lower limit and gamma_z in [-0.5,0.5]. Adaptation stops at
burn-in; four independently seeded scalar chains are compared. Intervals and
chain diagnostics are in recovery_intervals.csv and aggregate_summary.json.

This is a conditional implementation sanity check. It does not simulate survey
selection, re-run matching or include missing galaxy physics. A few noise
realizations cannot estimate frequentist coverage. Prior truncation and lens
parameter degeneracies can shift posterior medians under correctly generated
data; recovery must be assessed from joint uncertainty and each stated prior.

Reproduction, using the approved sn5070 CPU environment:
  python simulation_checks.py --generate-only
  python simulation_checks.py --steps 60000 --chains 4
  python simulation_checks.py --summarize-only
  python simulation_checks.py --source-trend-only
The source-trend step reuses saved joint states and evaluates the exact
conditional standard-Normal tail, without new MCMC or replication noise.
'''
    (OUT / 'README.txt').write_text(text, encoding='utf-8')
    lines = [
        '# Fixed-design known-truth checks', '',
        'Six planned noise realizations use two input slope histories; each is fitted independently with four scalar chains. '
        'These simulations test the implementation under its assumed likelihood, conditional on the observed lens design. '
        'They do not validate the galaxy model or calibrate interval coverage.', '',
        '| Input history / realization | Omega_k median [95.45% interval] | gamma_z median [95.45% interval] | P(gamma_z < -0.45) | max Rhat | min ESS |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for result in results:
        q = np.asarray(result['quantiles_95_68'])
        lines.append(f"| {result['name']} | {q[2,0]:.3f} [{q[0,0]:.3f}, {q[4,0]:.3f}] | "
                     f"{q[2,2]:.3f} [{q[0,2]:.3f}, {q[4,2]:.3f}] | "
                     f"{result['mass_gamma_z_below_minus_0p45']:.3f} | "
                     f"{result['max_rank_folded_or_split_rhat']:.5f} | {result['min_conservative_ess']:.0f} |")
    lines.extend(['',
        'In these six illustrative realizations the curvature posterior medians are '
        'positive even though the input curvature is zero; the plot shows that the '
        'central 68.27% intervals can exclude zero while the wider intervals include it. '
        'Consequently these calculations must not be described as accurate or unbiased '
        'curvature recovery. The finite sample cannot identify a calibrated coverage rate.', '',
        'The input curvature is zero in every realization. The prior has support only '
        '-0.0940537 < Omega_k < 2, and the likelihood has substantial covariance among '
        'curvature, the slope intercept, slope evolution and anisotropy. Posterior medians '
        'therefore need not equal the input values in individual noise realizations. '
        'Noise, degeneracy and finite asymmetric prior support must be distinguished from '
        'sampling failure; the displayed chain diagnostics address the latter. '
        'A small simulation set does not establish that an estimator is unbiased.', '',
        'The mock covariance equals the main adopted distance covariance, including its '
        'mismatch envelope. This explicitly tests that assumed Gaussian approximation; '
        'it does not test whether the approximation is adequate for every actual BEAMS '
        'distance entry or whether missing lens structure explains the real-data trend.',
        '',
    ])
    (OUT / 'interpretation_en.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


def mock_source_trend_checks():
    """Exact conditional Normal tail for the signed, normalized zs score.

    This reuses each mock's fitted joint posterior. It is not a new MCMC run,
    a calibrated p-value, an out-of-sample check or a large mock ensemble.
    """
    from types import SimpleNamespace
    from supplement_diagnostics import predict_joint
    rows = []
    for data_path in sorted(OUT.glob('*_data.npz')):
        name = data_path.name.removesuffix('_data.npz')
        summary_path = OUT / f'{name}_summary.json'
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        arrays = np.load(data_path)
        model = SimpleNamespace(n=len(arrays['zl']), zl=arrays['zl'], zs=arrays['zs'],
                                te=arrays['thetaE'], ta=arrays['thetaap'],
                                delta=arrays['delta'], obs=arrays['observed_velocity'],
                                err=arrays['dsigma'])
        # Check that the saved sampler actually used these data and this SN factor.
        recorded_hash = hashlib.sha256()
        for array in [arrays['mu_observed'], arrays['covariance'], model.zl, model.zs,
                      model.te, model.ta, model.delta, model.obs, model.err, arrays['bounds']]:
            recorded_hash.update(np.asarray(array).tobytes())
        assert recorded_hash.hexdigest() == summary['input_array_sha256']
        chain = np.load(OUT / f'{name}_chains.npy', mmap_mode='r')
        latent = np.load(OUT / f'{name}_mu.npy', mmap_mode='r')
        indices = np.asarray(summary['mu_saved_indices'], dtype=int)
        assert latent.shape == (summary['chains'], len(indices), 2 * model.n)
        theta = np.asarray(chain[:, indices, :]).reshape(-1, 5)
        mu = np.asarray(latent).reshape(-1, 2 * model.n)
        sigma, variance = predict_joint(model, theta, mu)
        residuals = (model.obs - sigma) / np.sqrt(variance)
        w = model.zs - np.mean(model.zs)
        w /= np.linalg.norm(w)
        observed_score = residuals @ w
        conditional_tails = 2 * ndtr(-np.abs(observed_score))
        by_chain = conditional_tails.reshape(summary['chains'], len(indices)).mean(axis=1)
        truth_noise_score = arrays['velocity_standard_noise'] @ w
        row = {
            'realization': name,
            'truth_gamma_z': float(arrays['truth'][2]),
            'joint_posterior_states': len(theta),
            'analytic_source_trend_PPC': float(np.mean(conditional_tails)),
            'observed_signed_score_q02275_q15865_median_q84135_q97725': np.percentile(observed_score, [2.275,15.865,50,84.135,97.725]).tolist(),
            'chainwise_analytic_source_trend_PPC': by_chain.tolist(),
            'generating_velocity_noise_signed_score_at_truth': float(truth_noise_score),
            'generating_velocity_noise_two_sided_normal_tail_at_truth': float(2 * ndtr(-abs(truth_noise_score))),
            'max_Rhat': summary['diagnostics']['max_rhat'],
            'min_ESS': summary['diagnostics']['min_ess'],
            'fit_input_array_hash_matches_generated_data': True,
        }
        rows.append(row)
    result = {
        'discrepancy': 'abs(sum_i w_i * (sigma_obs_i - sigma_model_i) / sqrt(dsigma_i^2 + (f_int*sigma_model_i)^2))',
        'weight': 'source redshift centered across the fixed 88 lenses, then normalized to unit L2 norm',
        'conditional_replicated_distribution': 'signed score is standard Normal at each fitted joint state',
        'analytic_PPC': 'posterior mean of 2*Phi(-abs(observed signed score)); no replication Monte Carlo error',
        'theta_mu_alignment': 'same chain and exact mu_saved_indices from the corresponding mock summary',
        'limitations': 'Six illustrative mocks are not an empirical null calibration; the data are reused for fitting and PPC, and matching/selection and omitted lens physics are not simulated.',
        'results': rows,
    }
    (OUT / 'mock_source_trend_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT / 'mock_source_trend_checks.csv', index=False)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='constant,evolving')
    parser.add_argument('--realizations', type=int, default=3)
    parser.add_argument('--steps', type=int, default=60000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--only', default='', help='Comma-separated names, e.g. constant_01,evolving_02')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--source-trend-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.source_trend_only:
        mock_source_trend_checks()
        return
    if args.summarize_only:
        summarize()
        return
    wanted = set(args.only.split(',')) if args.only else None
    if not args.generate_only:
        sys.path.insert(0, str(HERE))
        from supplement_sampler import sample_joint
    for case in args.cases.split(','):
        for realization in range(args.realizations):
            name = f'{case}_{realization + 1:02d}'
            if wanted and name not in wanted:
                continue
            if (OUT / f'{name}_summary.json').exists() and not args.overwrite:
                print('Skipping existing complete simulation', name, flush=True)
                continue
            model, meta = prepare_data(case, realization)
            print('Prepared', name, 'data seed', meta['data_seed'], flush=True)
            if not args.generate_only:
                start = time.perf_counter()
                sample_joint(model, OUT / name, steps=args.steps, nchains=args.chains,
                             seed=20261300 + 100 * list(TRUTHS).index(case) + realization,
                             initial=np.array(meta['truth']) + np.array([.08, .01, -.02, -.03, 0.]),
                             overwrite=args.overwrite)
                print('Completed', name, 'seconds', time.perf_counter() - start, flush=True)
    if not args.generate_only:
        summarize()


if __name__ == '__main__':
    main()
