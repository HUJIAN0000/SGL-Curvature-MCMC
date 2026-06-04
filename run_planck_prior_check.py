# -*- coding: utf-8 -*-
"""
Planck-prior cross-check (Section 3.1).
Same patched model as run_dual_validation_patched.py (residual-mismatch folded
into C_l/C_s), but with the H0 prior replaced by the Planck value
H0 = 67.4 +/- 0.5 km/s/Mpc. Prints the Omega_k constraint for both methods.

Does NOT draw or overwrite figure01.pdf, and does not need cosmo_tools.
"""

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gamma as gamma_func
from scipy.integrate import quad
import emcee
import time
import os
from multiprocessing import Pool, cpu_count

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

C_LIGHT = 299792.458

# ----- Planck prior -----
H0_MEAN, H0_SIG = 67.4, 0.5

# ===================== [Module A] DES-Dovekie =======================
df_des = pd.read_csv('matched_sgls_milp_dd.csv')
zl_des = df_des['zl'].values
zs_des = df_des['zs'].values
theta_E_des = df_des['thetaE'].values
theta_ap_des = df_des['thetaap'].values
sigma_obs_des = df_des['sigma_ap'].values
sigma_err_des = df_des['dsigma_ap'].values
delta_lum_des = df_des['delta'].values
mu_l_obs_des = df_des['sn_lens_MU'].values
mu_s_obs_des = df_des['sn_source_MU'].values

C_l  = np.loadtxt('cov_sgls_milp_dd_lens.txt')
C_s  = np.loadtxt('cov_sgls_milp_dd_source.txt')
C_ls = np.loadtxt('cov_sgls_milp_dd_cross.txt')

# ----- residual-mismatch patch (Section 2.2) -----
_OM0_FID = 0.30
def _Efid(z): return np.sqrt(_OM0_FID * (1.0 + z)**3 + (1.0 - _OM0_FID))
def _Ifid(z):
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.array([quad(lambda x: 1.0 / _Efid(x), 0.0, zi)[0] for zi in z])
def _dmu_mismatch(z_sn, z_target):
    z_sn = np.asarray(z_sn, dtype=float); z_target = np.asarray(z_target, dtype=float)
    return np.abs(5.0 * np.log10(((1.0 + z_sn) * _Ifid(z_sn)) / ((1.0 + z_target) * _Ifid(z_target))))
C_l = C_l + np.diag(_dmu_mismatch(df_des['sn_lens_zHD'].values,   zl_des)**2)
C_s = C_s + np.diag(_dmu_mismatch(df_des['sn_source_zHD'].values, zs_des)**2)

def calc_sigma_th_des(mu_l_arr, mu_s_arr, Omk, g_val, delta_val, beta_ani, H0_fit):
    dl = (H0_fit/C_LIGHT) * 10**((mu_l_arr - 25.0)/5.0) / (1.0 + zl_des)
    ds = (H0_fit/C_LIGHT) * 10**((mu_s_arr - 25.0)/5.0) / (1.0 + zs_des)
    al, as_ = 1.0 + Omk * dl**2, 1.0 + Omk * ds**2
    if np.any(al <= 0) or np.any(as_ <= 0): return np.full_like(dl, np.nan)
    dls = ds * np.sqrt(al) - dl * np.sqrt(as_)
    if np.any(dls <= 1e-5): return np.full_like(dl, np.nan)
    xi = g_val + delta_val - 2.0
    t1 = gamma_func((xi - 1.0)/2.0) / gamma_func(xi/2.0)
    t2 = beta_ani * gamma_func((xi + 1.0)/2.0) / gamma_func((xi + 2.0)/2.0)
    t3 = (gamma_func(g_val/2.0) * gamma_func(delta_val/2.0)) / \
         (gamma_func((g_val - 1.0)/2.0) * gamma_func((delta_val - 1.0)/2.0))
    denom = (xi - 2.0 * beta_ani) * (3.0 - xi)
    if np.any(denom == 0) or np.any(~np.isfinite(t1)) or np.any(~np.isfinite(t2)) or np.any(~np.isfinite(t3)):
        return np.full_like(dl, np.nan)
    F = ((3.0 - delta_val) / denom) * (t1 - t2) * t3
    theta_E_rad = theta_E_des * (np.pi / 648000.0)
    sigma_sq = (C_LIGHT**2 / (2.0 * np.sqrt(np.pi))) * (ds / dls) * theta_E_rad * F * (theta_ap_des / theta_E_des)**(2.0 - g_val)
    return np.sqrt(np.maximum(sigma_sq, 0))

def log_prior_des(theta):
    Omk, g0, gz, beta, dint, H0_fit = theta
    if not (-0.1 < Omk < 2.0 and 1.0 < g0 < 3.0 and -0.5 < gz < 0.5 and
            -1.0 < beta < 1.0 and 0.0 < dint < 0.3 and 60 < H0_fit < 85):
        return -np.inf
    d_cmb = (H0_fit / C_LIGHT) * (1.0 + 1090.0) * 12.8
    if 1.0 + Omk * (d_cmb**2) <= 0.001: return -np.inf
    return -0.5 * ((H0_fit - H0_MEAN) / H0_SIG)**2     # Planck prior

def log_prob_des(theta):
    lp = log_prior_des(theta)
    if not np.isfinite(lp): return -np.inf
    Omk, g0, gz, beta, dint, H0_fit = theta
    gamma_z = g0 + gz * zl_des
    if np.any(gamma_z <= 1.05) or np.any(gamma_z >= 2.95): return -np.inf
    s0 = calc_sigma_th_des(mu_l_obs_des, mu_s_obs_des, Omk, gamma_z, delta_lum_des, beta, H0_fit)
    if np.any(np.isnan(s0)): return -np.inf
    eps = 1e-4
    s_dl = calc_sigma_th_des(mu_l_obs_des + eps, mu_s_obs_des, Omk, gamma_z, delta_lum_des, beta, H0_fit)
    s_ds = calc_sigma_th_des(mu_l_obs_des, mu_s_obs_des + eps, Omk, gamma_z, delta_lum_des, beta, H0_fit)
    J_l = (s_dl - s0) / eps; J_s = (s_ds - s0) / eps
    C_prop = (J_l[:, None]*C_l*J_l[None, :] + J_s[:, None]*C_s*J_s[None, :]
              + J_l[:, None]*C_ls*J_s[None, :] + J_s[:, None]*C_ls.T*J_l[None, :])
    C_total = np.diag(sigma_err_des**2 + (s0 * dint)**2) + C_prop
    dsig = sigma_obs_des - s0
    try:
        cl = cho_factor(C_total)
        return lp - 0.5*np.dot(dsig, cho_solve(cl, dsig)) - np.sum(np.log(np.diag(cl[0])))
    except np.linalg.LinAlgError:
        return -np.inf

# ===================== [Module B] Union3 GP =========================
cov_raw = np.loadtxt('mag_covmat.txt')
n_bins = int(cov_raw[0])
cov_matrix = cov_raw[1:].reshape((n_bins, n_bins)) + np.eye(n_bins) * 1e-6
lc_data = np.loadtxt('lcparam_full.txt', usecols=(1, 4))
z_nodes, mu_nodes = lc_data[:, 0], lc_data[:, 1]
sgl_df = pd.read_csv('130sgls2.CSV').dropna(subset=['zl', 'zs', 'thetaE', 'thetaap', 'sigma_ap', 'dsigma_ap', 'delta'])
sgl_df = sgl_df[(sgl_df['zs'] <= z_nodes.max()) & (sgl_df['thetaap'] > 0)].reset_index(drop=True)
zl_u, zs_u = sgl_df['zl'].values, sgl_df['zs'].values
theta_E_u, theta_ap_u = sgl_df['thetaE'].values, sgl_df['thetaap'].values
sigma_obs_u, sigma_err_u = sgl_df['sigma_ap'].values, sgl_df['dsigma_ap'].values
delta_lum_u = sgl_df['delta'].values

def kernel(x1, x2, var=1.5, scale=1.2):
    return var * np.exp(-0.5 * (x1[:, None] - x2[None, :])**2 / scale**2)
def mean_f(z): return 5.0 * np.log10(z) + 43.0
c_and_l_gp = cho_factor(kernel(z_nodes, z_nodes) + cov_matrix)
alpha = cho_solve(c_and_l_gp, mu_nodes - mean_f(z_nodes))
z_targets = np.concatenate([zl_u, zs_u])
mu_pred = np.dot(kernel(z_nodes, z_targets).T, alpha) + mean_f(z_targets)
mu_l_gp, mu_s_gp = mu_pred[:len(sgl_df)], mu_pred[len(sgl_df):]

def calc_sigma_th_union(dl, ds, dls, tE, tap, g_val, delta_val, beta_ani):
    xi = g_val + delta_val - 2.0
    t1 = gamma_func((xi - 1.0)/2.0) / gamma_func(xi/2.0)
    t2 = beta_ani * gamma_func((xi + 1.0)/2.0) / gamma_func((xi + 2.0)/2.0)
    t3 = (gamma_func(g_val/2.0) * gamma_func(delta_val/2.0)) / \
         (gamma_func((g_val - 1.0)/2.0) * gamma_func((delta_val - 1.0)/2.0))
    denom = (xi - 2.0 * beta_ani) * (3.0 - xi)
    if np.any(denom == 0) or np.any(~np.isfinite(t1)) or np.any(~np.isfinite(t2)) or np.any(~np.isfinite(t3)):
        return np.full_like(dl, np.nan)
    F = ((3.0 - delta_val) / denom) * (t1 - t2) * t3
    theta_E_rad = tE * (np.pi / 648000.0)
    sigma_sq = (C_LIGHT**2 / (2.0 * np.sqrt(np.pi))) * (ds / dls) * theta_E_rad * F * (tap / tE)**(2.0 - g_val)
    return np.sqrt(np.maximum(sigma_sq, 0))

def log_prior_union(theta):
    Omk, g0, gz, beta, dint, H0_fit = theta
    if not (-0.1 < Omk < 2.0 and 1.0 < g0 < 3.0 and -0.5 < gz < 0.5 and
            -1.0 < beta < 1.0 and 0.0 < dint < 0.3 and 60 < H0_fit < 85):
        return -np.inf
    return -0.5 * ((H0_fit - H0_MEAN) / H0_SIG)**2      # Planck prior

def log_prob_union(theta):
    lp = log_prior_union(theta)
    if not np.isfinite(lp): return -np.inf
    Omk, g0, gz, beta, dint, H0_fit = theta
    gamma_z = g0 + gz * zl_u
    if np.any(gamma_z <= 1.05) or np.any(gamma_z >= 2.95): return -np.inf
    dl = (H0_fit/C_LIGHT) * 10**((mu_l_gp - 25.0)/5.0) / (1.0 + zl_u)
    ds = (H0_fit/C_LIGHT) * 10**((mu_s_gp - 25.0)/5.0) / (1.0 + zs_u)
    al, as_ = 1.0 + Omk * dl**2, 1.0 + Omk * ds**2
    if np.any(al <= 0) or np.any(as_ <= 0): return -np.inf
    dls = ds * np.sqrt(al) - dl * np.sqrt(as_)
    if np.any(dls <= 1e-5): return -np.inf
    sigma_th = calc_sigma_th_union(dl, ds, dls, theta_E_u, theta_ap_u, gamma_z, delta_lum_u, beta)
    if np.any(np.isnan(sigma_th)): return -np.inf
    var_tot = sigma_err_u**2 + (sigma_th * dint)**2
    return lp - 0.5 * np.sum(((sigma_obs_u - sigma_th)**2 / var_tot) + np.log(2.0 * np.pi * var_tot))

# ============================ run ===================================
def run(log_prob, init, tag):
    ndim, nwalkers, N = 6, 64, 15000
    pos = init + 1e-4 * np.random.randn(nwalkers, ndim)
    with Pool(processes=cpu_count()) as pool:
        s = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        t0 = time.time()
        s.run_mcmc(pos, N, progress=True)
        print(f"  {tag} done in {(time.time()-t0)/60:.2f} min")
    chain = s.get_chain(discard=5000, thin=15, flat=True)
    med = np.median(chain, axis=0)
    lo = med - np.percentile(chain, 16, axis=0)
    hi = np.percentile(chain, 84, axis=0) - med
    return med, lo, hi

if __name__ == '__main__':
    print(f"Planck prior H0 = {H0_MEAN} +/- {H0_SIG}\n")
    md, ld, hd = run(log_prob_des,   np.array([0.15, 1.9, -0.35, -0.15, 0.10, 67.4]), "DES-Dovekie")
    mu, lu, hu = run(log_prob_union, np.array([0.12, 2.1, -0.42, -0.45, 0.09, 67.4]), "Union3 GP")
    print("\n=== Omega_k under the Planck H0 prior ===")
    print(f"  DES-Dovekie : Omega_k = {md[0]:.3f} +{hd[0]:.3f}/-{ld[0]:.3f}")
    print(f"  Union3 GP   : Omega_k = {mu[0]:.3f} +{hu[0]:.3f}/-{lu[0]:.3f}")
    names = ["Omega_k", "gamma_0", "gamma_z", "beta_ani", "delta_int", "H_0"]
    print("\n(full parameter set for reference)")
    for i, nm in enumerate(names):
        print(f"  {nm:<9} DES {md[i]:+.3f} +{hd[i]:.3f}/-{ld[i]:.3f}   |   U3 {mu[i]:+.3f} +{hu[i]:.3f}/-{lu[i]:.3f}")
