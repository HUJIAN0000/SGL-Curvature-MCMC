# -*- coding: utf-8 -*-
"""
systematics_sensitivity.py
--------------------------
Lens-modelling systematics sensitivity test for the DES-Dovekie exact-pairing
analysis (fills the TODO in Section 3.3 of the manuscript).

It re-uses the EXACT velocity-dispersion model and full-covariance likelihood
of run_dual_validation.py (Module A) and re-runs the MCMC under several
external-convergence (kappa_ext) scenarios, then reports the shift in the
medians of Omega_k and gamma_z relative to the baseline.

Two complementary tests are run:
  (T1) MEAN-BIAS  : a constant kappa0 applied to every system. External mass
                    sheets along the line of sight bias the galaxy-only
                    Einstein radius to theta_E_eff = theta_E_obs * (1 - kappa0).
                    This probes a systematic SHIFT of the medians.
  (T2) SCATTER    : a per-system kappa scatter sigma_kappa, folded into the
                    error budget as an extra fractional variance on sigma_th
                    (d sigma_th / sigma_th = 0.5 * d kappa, since sigma_th^2 ~ theta_E).
                    This probes the BROADENING of the posterior.

>>> CONVENTION NOTE <<<
The (1 - kappa) scaling of the galaxy Einstein radius is the standard first-order
mass-sheet treatment, but the exact sign/placement depends on how theta_E enters
your derivation (Chen et al. 2019). Verify it against your lensing equation before
quoting numbers; the scenarios below are deliberately conservative (|kappa| ~ few %).

Run in the same directory as matched_sgls_milp_dd.csv and the cov_*.txt files.
Requires: numpy, scipy, pandas, emcee.
"""

import os
os.environ.update({k: "1" for k in
                   ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]})

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gamma as gamma_func
import emcee

C_LIGHT = 299792.458

# ----- sampler settings (reduce for a quick scan, restore for the final run) --
NWALKERS = 64
N_STEPS  = 8000
N_BURN   = 3000
THIN     = 15
SEED     = 42

# ----- scenarios -------------------------------------------------------------
MEAN_BIAS_KAPPAS = [0.00, 0.02, 0.03, 0.05]   # T1: constant kappa0
SCATTER_SIGMAS   = [0.03, 0.05]               # T2: per-system kappa scatter

# ============================ data ==========================================
df = pd.read_csv("matched_sgls_milp_dd.csv")
zl   = df["zl"].values
zs   = df["zs"].values
theta_E  = df["thetaE"].values
theta_ap = df["thetaap"].values
sigma_obs = df["sigma_ap"].values
sigma_err = df["dsigma_ap"].values
delta_lum = df["delta"].values
mu_l_obs = df["sn_lens_MU"].values
mu_s_obs = df["sn_source_MU"].values

C_l  = np.loadtxt("cov_sgls_milp_dd_lens.txt")
C_s  = np.loadtxt("cov_sgls_milp_dd_source.txt")
C_ls = np.loadtxt("cov_sgls_milp_dd_cross.txt")

# ============================ model =========================================
def calc_sigma_th(mu_l_arr, mu_s_arr, Omk, g_val, delta_val, beta_ani, H0_fit,
                  theta_E_use):
    """Identical to calc_sigma_th_des, but theta_E is passed in so that an
    external-convergence-corrected Einstein radius can be used."""
    dl = (H0_fit / C_LIGHT) * 10**((mu_l_arr - 25.0) / 5.0) / (1.0 + zl)
    ds = (H0_fit / C_LIGHT) * 10**((mu_s_arr - 25.0) / 5.0) / (1.0 + zs)
    al, as_ = 1.0 + Omk * dl**2, 1.0 + Omk * ds**2
    if np.any(al <= 0) or np.any(as_ <= 0):
        return np.full_like(dl, np.nan)
    dls = ds * np.sqrt(al) - dl * np.sqrt(as_)
    if np.any(dls <= 1e-5):
        return np.full_like(dl, np.nan)

    xi = g_val + delta_val - 2.0
    t1 = gamma_func((xi - 1.0) / 2.0) / gamma_func(xi / 2.0)
    t2 = beta_ani * gamma_func((xi + 1.0) / 2.0) / gamma_func((xi + 2.0) / 2.0)
    t3 = (gamma_func(g_val / 2.0) * gamma_func(delta_val / 2.0)) / \
         (gamma_func((g_val - 1.0) / 2.0) * gamma_func((delta_val - 1.0) / 2.0))
    denom = (xi - 2.0 * beta_ani) * (3.0 - xi)
    if np.any(denom == 0) or np.any(~np.isfinite(t1)) or np.any(~np.isfinite(t2)) \
            or np.any(~np.isfinite(t3)):
        return np.full_like(dl, np.nan)

    F = ((3.0 - delta_val) / denom) * (t1 - t2) * t3
    theta_E_rad = theta_E_use * (np.pi / 648000.0)
    sigma_sq = (C_LIGHT**2 / (2.0 * np.sqrt(np.pi))) * (ds / dls) * theta_E_rad \
        * F * (theta_ap / theta_E_use)**(2.0 - g_val)
    return np.sqrt(np.maximum(sigma_sq, 0))

def log_prior(theta):
    Omk, g0, gz, beta, dint, H0_fit = theta
    if not (-0.1 < Omk < 2.0 and 1.0 < g0 < 3.0 and -0.5 < gz < 0.5 and
            -1.0 < beta < 1.0 and 0.0 < dint < 0.3 and 60 < H0_fit < 85):
        return -np.inf
    d_cmb = (H0_fit / C_LIGHT) * (1.0 + 1090.0) * 12.8
    if 1.0 + Omk * (d_cmb**2) <= 0.001:
        return -np.inf
    return -0.5 * ((H0_fit - 73.04) / 1.42)**2

def make_log_prob(kappa0=0.0, sigma_kappa=0.0):
    """Return a log-probability closure for a given systematics scenario."""
    theta_E_use = theta_E * (1.0 - kappa0)          # T1: mean-bias

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        Omk, g0, gz, beta, dint, H0_fit = theta
        gamma_z = g0 + gz * zl
        if np.any(gamma_z <= 1.05) or np.any(gamma_z >= 2.95):
            return -np.inf

        s0 = calc_sigma_th(mu_l_obs, mu_s_obs, Omk, gamma_z, delta_lum, beta,
                           H0_fit, theta_E_use)
        if np.any(np.isnan(s0)):
            return -np.inf

        eps = 1e-4
        s_dl = calc_sigma_th(mu_l_obs + eps, mu_s_obs, Omk, gamma_z, delta_lum,
                             beta, H0_fit, theta_E_use)
        s_ds = calc_sigma_th(mu_l_obs, mu_s_obs + eps, Omk, gamma_z, delta_lum,
                             beta, H0_fit, theta_E_use)
        J_l = (s_dl - s0) / eps
        J_s = (s_ds - s0) / eps

        C_prop = (J_l[:, None] * C_l * J_l[None, :]
                  + J_s[:, None] * C_s * J_s[None, :]
                  + J_l[:, None] * C_ls * J_s[None, :]
                  + J_s[:, None] * C_ls.T * J_l[None, :])

        # diagonal: measurement + intrinsic scatter (+ optional kappa scatter, T2)
        diag = sigma_err**2 + (s0 * dint)**2
        if sigma_kappa > 0.0:
            diag = diag + (0.5 * sigma_kappa * s0)**2
        C_total = np.diag(diag) + C_prop

        dsig = sigma_obs - s0
        try:
            cl = cho_factor(C_total)
            logL = -0.5 * np.dot(dsig, cho_solve(cl, dsig)) \
                - np.sum(np.log(np.diag(cl[0])))
            return lp + logL
        except np.linalg.LinAlgError:
            return -np.inf

    return log_prob

# ============================ runner ========================================
def run(log_prob, tag):
    ndim = 6
    init = np.array([0.15, 1.9, -0.35, -0.15, 0.10, 73.0])
    rng = np.random.default_rng(SEED)
    pos = init + 1e-4 * rng.standard_normal((NWALKERS, ndim))
    sampler = emcee.EnsembleSampler(NWALKERS, ndim, log_prob)
    sampler.run_mcmc(pos, N_STEPS, progress=True)
    chain = sampler.get_chain(discard=N_BURN, thin=THIN, flat=True)
    med = np.median(chain, axis=0)
    lo = med - np.percentile(chain, 16, axis=0)
    hi = np.percentile(chain, 84, axis=0) - med
    print(f"[{tag}]  Omk = {med[0]:.3f} (+{hi[0]:.3f}/-{lo[0]:.3f})   "
          f"gz = {med[2]:.3f} (+{hi[2]:.3f}/-{lo[2]:.3f})")
    return med, lo, hi

def main():
    rows = []
    # baseline + mean-bias scenarios (T1)
    for k0 in MEAN_BIAS_KAPPAS:
        med, lo, hi = run(make_log_prob(kappa0=k0), f"kappa0={k0:.2f}")
        rows.append((f"mean-bias kappa0={k0:.2f}", med, lo, hi))
    # scatter scenarios (T2)
    for sk in SCATTER_SIGMAS:
        med, lo, hi = run(make_log_prob(sigma_kappa=sk), f"sigma_kappa={sk:.2f}")
        rows.append((f"scatter sigma_kappa={sk:.2f}", med, lo, hi))

    base = rows[0][1]               # medians of the kappa0=0 baseline
    sig_Omk = 0.5 * (rows[0][2][0] + rows[0][3][0])   # baseline 1-sigma on Omk
    sig_gz  = 0.5 * (rows[0][2][2] + rows[0][3][2])

    print("\n| scenario | Omega_k | gamma_z | dOmega_k/sigma | dgamma_z/sigma |")
    print("|---|---|---|---|---|")
    for name, med, lo, hi in rows:
        d_omk = (med[0] - base[0]) / sig_Omk
        d_gz = (med[2] - base[2]) / sig_gz
        print(f"| {name} | {med[0]:.3f} +{hi[0]:.3f}/-{lo[0]:.3f} | "
              f"{med[2]:.3f} +{hi[2]:.3f}/-{lo[2]:.3f} | {d_omk:+.2f} | {d_gz:+.2f} |")

if __name__ == "__main__":
    main()
