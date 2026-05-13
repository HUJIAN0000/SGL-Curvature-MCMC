# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 02:40:55 2026

@author: Hu Jian
Email: dg1626002@smail.nju.edu.cn
"""

"""

Precision Cosmology Pipeline for Strong Gravitational Lensing (SGL)
-------------------------------------------------------------------
This script performs a dual-method Markov Chain Monte Carlo (MCMC) cross-validation
to constrain cosmic curvature (Omega_k) and lens density profile evolution (gamma_z).

Method A (Main): Systematics-controlled Exact Pairing using the DES-Dovekie SN Ia 
                 sample, with full non-diagonal covariance propagation via Jacobian 
                 broadcasting.
Method B (Baseline): Gaussian Process (GP) regression interpolation using the 
                     Union3 SN Ia compilation.

Underlying Physics: Cosmic Distance Duality Relation (CDDR) & P2 Velocity Dispersion Model.
H0: SNe Ia
"""

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gamma as gamma_func
import emcee
import time
import os
import multiprocessing
from tqdm import tqdm
import cosmo_tools

# ====================================================================
# [CRITICAL]: Prevent thread thrashing when using multiprocessing with Numpy
# ====================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

C_LIGHT = 299792.458 

# ====================================================================
# 1. Load Data for Module A (DES Exact Pairing) and Module B (Union3 GP)
# ====================================================================
print("📥 [Module A] Loading DES-Dovekie paired data...")
df = pd.read_csv('matched_sgls_milp_dd.csv')
zl_list = df['zl'].values
zs_list = df['zs'].values
theta_E = df['thetaE'].values
theta_ap = df['thetaap'].values    
sigma_obs = df['sigma_ap'].values
sigma_err = df['dsigma_ap'].values 
delta_lum = df['delta'].values 
mu_l_obs = df['sn_lens_MU'].values
mu_s_obs = df['sn_source_MU'].values

C_l  = np.loadtxt('cov_sgls_milp_dd_lens.txt')
C_s  = np.loadtxt('cov_sgls_milp_dd_source.txt')
C_ls = np.loadtxt('cov_sgls_milp_dd_cross.txt')

print("📥 [Module B] Loading Union3 Cobaya GP background data...")
cov_raw = np.loadtxt('mag_covmat.txt')
n_bins = int(cov_raw[0])
cov_matrix = cov_raw[1:].reshape((n_bins, n_bins)) + np.eye(n_bins) * 1e-6

lc_data = np.loadtxt('lcparam_full.txt', usecols=(1, 4))
z_nodes = lc_data[:, 0]
mu_nodes = lc_data[:, 1]

sgl_df = pd.read_csv('130sgls2.CSV')
sgl_df = sgl_df.dropna(subset=['zl', 'zs', 'thetaE', 'thetaap', 'sigma_ap', 'dsigma_ap', 'delta'])
sgl_df = sgl_df[(sgl_df['zs'] <= z_nodes.max()) & (sgl_df['thetaap'] > 0)].reset_index(drop=True)

zl_list_union, zs_list_union = sgl_df['zl'].values, sgl_df['zs'].values
theta_E_union, theta_ap_union = sgl_df['thetaE'].values, sgl_df['thetaap'].values    
sigma_obs_union, sigma_err_union = sgl_df['sigma_ap'].values, sgl_df['dsigma_ap'].values 
delta_lum_union = sgl_df['delta'].values 

# Gaussian Process background reconstruction (GP)
def kernel(x1, x2, var=1.5, scale=1.2): return var * np.exp(-0.5 * (x1[:, None] - x2[None, :])**2 / scale**2)
def mean_f(z): return 5.0 * np.log10(z) + 43.0

c_and_l_gp = cho_factor(kernel(z_nodes, z_nodes) + cov_matrix)
alpha = cho_solve(c_and_l_gp, mu_nodes - mean_f(z_nodes))

z_targets = np.concatenate([zl_list_union, zs_list_union])
K_Xstar = kernel(z_nodes, z_targets).T
mu_pred = np.dot(K_Xstar, alpha) + mean_f(z_targets)
mu_l_gp, mu_s_gp = mu_pred[:len(sgl_df)], mu_pred[len(sgl_df):]

# ====================================================================
# 2. Physics & Likelihood for Module A (DES Exact Pairing)
# ====================================================================
def calc_sigma_th_des(mu_l_arr, mu_s_arr, Omk, g_val, delta_val, beta_ani, H0_fit):
    dl = (H0_fit/C_LIGHT) * 10**((mu_l_arr - 25.0)/5.0) / (1.0 + zl_list)
    ds = (H0_fit/C_LIGHT) * 10**((mu_s_arr - 25.0)/5.0) / (1.0 + zs_list)
    al, as_ = 1.0 + Omk * dl**2, 1.0 + Omk * ds**2
    if np.any(al <= 0) or np.any(as_ <= 0): return np.full_like(dl, np.nan)
    dls = ds * np.sqrt(al) - dl * np.sqrt(as_)
    if np.any(dls <= 1e-5): return np.full_like(dl, np.nan)
    xi = g_val + delta_val - 2.0
    t1 = gamma_func((xi - 1.0)/2.0) / gamma_func(xi/2.0)
    t2 = beta_ani * gamma_func((xi + 1.0)/2.0) / gamma_func((xi + 2.0)/2.0)
    t3 = (gamma_func(g_val/2.0) * gamma_func(delta_val/2.0)) / (gamma_func((g_val - 1.0)/2.0) * gamma_func((delta_val - 1.0)/2.0))
    denom = (xi - 2.0 * beta_ani) * (3.0 - xi)
    if np.any(denom == 0) or np.any(~np.isfinite(t1)) or np.any(~np.isfinite(t2)) or np.any(~np.isfinite(t3)): return np.full_like(dl, np.nan)
    F = ((3.0 - delta_val) / denom) * (t1 - t2) * t3
    theta_E_rad = theta_E * (np.pi / 648000.0)
    sigma_sq = (C_LIGHT**2 / (2.0 * np.sqrt(np.pi))) * (ds / dls) * theta_E_rad * F * (theta_ap / theta_E)**(2.0 - g_val)
    return np.sqrt(np.maximum(sigma_sq, 0))

def log_prob_des(theta):
    Omk, g0, gz, beta, dint, H0_fit = theta
    # Basic bounds
    if not (-0.5 < Omk < 2.0 and 1.0 < g0 < 3.0 and -0.5 < gz < 0.5 and -1.0 < beta < 1.0 and 0.0 < dint < 0.3 and 60 < H0_fit < 85): return -np.inf
    # Dynamic CMB cutoff for DES model
    d_cmb = (H0_fit / C_LIGHT) * (1.0 + 1090.0) * 12.8
    if 1.0 + Omk * (d_cmb**2) <= 0.001: return -np.inf
    
    lp = -0.5 * ((H0_fit - 73.04) / 1.42)**2
    gamma_z = g0 + gz * zl_list
    if np.any(gamma_z <= 1.05) or np.any(gamma_z >= 2.95): return -np.inf
    
    sigma_th_0 = calc_sigma_th_des(mu_l_obs, mu_s_obs, Omk, gamma_z, delta_lum, beta, H0_fit)
    if np.any(np.isnan(sigma_th_0)): return -np.inf
    
    eps = 1e-4
    sigma_th_dl = calc_sigma_th_des(mu_l_obs + eps, mu_s_obs, Omk, gamma_z, delta_lum, beta, H0_fit)
    sigma_th_ds = calc_sigma_th_des(mu_l_obs, mu_s_obs + eps, Omk, gamma_z, delta_lum, beta, H0_fit)
    J_l = (sigma_th_dl - sigma_th_0) / eps
    J_s = (sigma_th_ds - sigma_th_0) / eps
    
    term_l  = J_l[:, None] * C_l * J_l[None, :]
    term_s  = J_s[:, None] * C_s * J_s[None, :]
    term_ls = J_l[:, None] * C_ls * J_s[None, :]
    term_sl = J_s[:, None] * C_ls.T * J_l[None, :]
    
    C_prop_SNe = term_l + term_s + term_ls + term_sl
    C_total = np.diag(sigma_err**2 + (sigma_th_0 * dint)**2) + C_prop_SNe
    delta_sig = sigma_obs - sigma_th_0
    
    try:
        c_and_l = cho_factor(C_total)
        logL = -0.5 * np.dot(delta_sig, cho_solve(c_and_l, delta_sig)) - np.sum(np.log(np.diag(c_and_l[0])))
        return lp + logL
    except np.linalg.LinAlgError: return -np.inf

# ====================================================================
# 3. Physics & Likelihood for Module B (Union3 GP Baseline)
# ====================================================================
def calc_sigma_th_union(dl, ds, dls, theta_E_arc, theta_ap_arc, gamma_val, delta_val, beta_ani):
    xi = gamma_val + delta_val - 2.0
    t1 = gamma_func((xi - 1.0)/2.0) / gamma_func(xi/2.0)
    t2 = beta_ani * gamma_func((xi + 1.0)/2.0) / gamma_func((xi + 2.0)/2.0)
    t3 = (gamma_func(gamma_val/2.0) * gamma_func(delta_val/2.0)) / (gamma_func((gamma_val - 1.0)/2.0) * gamma_func((delta_val - 1.0)/2.0))
    denom = (xi - 2.0 * beta_ani) * (3.0 - xi)
    if np.any(denom == 0) or np.any(~np.isfinite(t1)) or np.any(~np.isfinite(t2)) or np.any(~np.isfinite(t3)): return np.full_like(dl, np.nan)
    F = ((3.0 - delta_val) / denom) * (t1 - t2) * t3
    theta_E_rad = theta_E_arc * (np.pi / 648000.0)
    sigma_sq = (C_LIGHT**2 / (2.0 * np.sqrt(np.pi))) * (ds / dls) * theta_E_rad * F * (theta_ap_arc / theta_E_arc)**(2.0 - gamma_val)
    return np.sqrt(np.maximum(sigma_sq, 0))

def log_prob_union(theta):
    Omk, g0, gz, beta, dint, H0_fit = theta
    # Hardcoded CMB boundary for Union model (-0.09)
    if not (-0.09 < Omk < 2.0 and 1.0 < g0 < 3.0 and -0.5 < gz < 0.5 and -1.0 < beta < 1.0 and 0.0 < dint < 0.3 and 60 < H0_fit < 85): return -np.inf
    lp = -0.5 * ((H0_fit - 73.04) / 1.42)**2
    gamma_z = g0 + gz * zl_list_union
    if np.any(gamma_z <= 1.05) or np.any(gamma_z >= 2.95): return -np.inf
        
    dl = (H0_fit/C_LIGHT) * 10**((mu_l_gp - 25.0)/5.0) / (1.0 + zl_list_union)
    ds = (H0_fit/C_LIGHT) * 10**((mu_s_gp - 25.0)/5.0) / (1.0 + zs_list_union)
    al, as_ = 1.0 + Omk * dl**2, 1.0 + Omk * ds**2
    if np.any(al <= 0) or np.any(as_ <= 0): return -np.inf
    
    dls = ds * np.sqrt(al) - dl * np.sqrt(as_)
    if np.any(dls <= 1e-5): return -np.inf
    
    sigma_th = calc_sigma_th_union(dl, ds, dls, theta_E_union, theta_ap_union, gamma_z, delta_lum_union, beta)
    if np.any(np.isnan(sigma_th)): return -np.inf
    
    var_tot = sigma_err_union**2 + (sigma_th * dint)**2
    logL = -0.5 * np.sum(((sigma_obs_union - sigma_th)**2 / var_tot) + np.log(2.0 * np.pi * var_tot))
    return lp + logL

# ====================================================================
# 4. Parallel Sampling Engine
# ====================================================================
def run_mcmc(lp_func, initial, steps, desc, pool):
    ndim, nwalkers = len(initial), 64
    pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_func, pool=pool)
    print(f"\n🔥 Starting [{desc}]: ({steps} steps)")
    for _ in tqdm(sampler.sample(pos, iterations=steps), total=steps): pass
    return sampler.get_chain(discard=4000, thin=15, flat=True)

# ====================================================================
# 5. Main Execution
# ====================================================================
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    num_cores = multiprocessing.cpu_count()
    print(f"🚀 Preparing to start! Detected {num_cores} CPU logical cores.")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        start_time = time.time()
        samples_des = run_mcmc(log_prob_des, [0.0, 2.1, -0.3, -0.4, 0.09, 73.0], 15000, "Model A: DES Exact Pairing", pool)
        t_des = (time.time() - start_time)/60
        print(f"✅ Model A sampling complete, took: {t_des:.2f} minutes.")

        mid_time = time.time()
        samples_union = run_mcmc(log_prob_union, [0.0, 2.1, -0.3, -0.4, 0.09, 73.0], 15000, "Model B: Union3 GP Baseline", pool)
        t_union = (time.time() - mid_time)/60
        print(f"✅ Model B sampling complete, took: {t_union:.2f} minutes.")

    labels = [r"\Omega_k", r"\gamma_0", r"\gamma_z", r"\beta_{ani}", r"\delta_{int}", r"H_0"]
    print_labels = ["Omega_k", "gamma_0", "gamma_z", "beta_ani", "delta_int", "H_0"]
    
    print("\n" + "★"*85)
    print(f"{'Parameter':<12} | {'DES-Dovekie Exact Pairing':<30} | {'Union3 Gaussian Process':<30}")
    print("-" * 85)
    for i in range(len(labels)):
        q_des = np.percentile(samples_des[:, i], [15.865, 50.0, 84.135])
        q_uni = np.percentile(samples_union[:, i], [15.865, 50.0, 84.135])
        print(f"{print_labels[i]:<12} | {q_des[1]:.4f} +{q_des[2]-q_des[1]:.4f}/-{q_des[1]-q_des[0]:.4f} | {q_uni[1]:.4f} +{q_uni[2]-q_uni[1]:.4f}/-{q_uni[1]-q_uni[0]:.4f}")
    print("★"*85)

    cosmo_tools.plot_getdist_comparison(
        [samples_des, samples_union], labels, 
        legend_labels=["DES-Dovekie (Exact Pairing)", "Union3 (Gaussian Process)"],
        colors=["#e31a1c", "#1f78b4"], filename="figure01.pdf"
    )
    print("✅ Double cross-validation comparison plot completed! Saved as: figure01.pdf")