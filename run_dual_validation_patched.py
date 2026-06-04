# -*- coding: utf-8 -*-
"""
Dual Cross-Validation: DES-Dovekie Exact Pairing vs. Union3 GP Baseline
This script unifies both methodologies to generate an overlay contour plot.

PATCHED: folds the residual redshift-mismatch term Dmu_mis (Section 2.2) into
the diagonal of the SN lens/source covariance sub-blocks before propagation.
"""

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gamma as gamma_func
from scipy.integrate import quad
import emcee
import time
import os
import cosmo_tools
from multiprocessing import Pool, cpu_count

# 强制限制底层单线程，防止多进程并行时发生线程灾难 (Thread Thrashing)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

C_LIGHT = 299792.458

# ====================================================================
# [模块 A] DES-Dovekie 严苛配对全协方差传递 (Exact Pairing)
# ====================================================================
print("📥 [Module A] 加载 DES-Dovekie 配对数据...")
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

# --------------------------------------------------------------------
# [补丁] 残余红移失配项 (Section 2.2)
# Dmu_mis = |mu(z_SN) - mu(z_target)| 在 fiducial flat LCDM 下的等效距离模数偏移。
# 它是 SN 距离的额外不确定度，独立作用于透镜面/源面，因此加到 C_l / C_s 的对角元，
# 不加到 C_ls（两平面失配相互独立）。随后经同样的 Jacobian 传播进似然。
# --------------------------------------------------------------------
_OM0_FID = 0.30  # 仅用于把小的红移偏移映射成距离，H0 在比值中约掉，结果对该取值不敏感

def _Efid(z):
    return np.sqrt(_OM0_FID * (1.0 + z)**3 + (1.0 - _OM0_FID))

def _Ifid(z):
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.array([quad(lambda x: 1.0 / _Efid(x), 0.0, zi)[0] for zi in z])

def _dmu_mismatch(z_sn, z_target):
    z_sn = np.asarray(z_sn, dtype=float)
    z_target = np.asarray(z_target, dtype=float)
    dL_sn = (1.0 + z_sn) * _Ifid(z_sn)
    dL_t  = (1.0 + z_target) * _Ifid(z_target)
    return np.abs(5.0 * np.log10(dL_sn / dL_t))

dmu_l = _dmu_mismatch(df_des['sn_lens_zHD'].values,   zl_des)
dmu_s = _dmu_mismatch(df_des['sn_source_zHD'].values, zs_des)
C_l = C_l + np.diag(dmu_l**2)
C_s = C_s + np.diag(dmu_s**2)
print(f"   ↳ 残余失配项已并入: median Dmu_l={np.median(dmu_l):.3f}, "
      f"median Dmu_s={np.median(dmu_s):.3f} mag")
# --------------------------------------------------------------------

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
    if not (-0.1 < Omk < 2.0 and 1.0 < g0 < 3.0 and -0.5 < gz < 0.5 and \
            -1.0 < beta < 1.0 and 0.0 < dint < 0.3 and 60 < H0_fit < 85):
        return -np.inf
    d_cmb = (H0_fit / C_LIGHT) * (1.0 + 1090.0) * 12.8
    if 1.0 + Omk * (d_cmb**2) <= 0.001: return -np.inf
    return -0.5 * ((H0_fit - 73.04) / 1.42)**2

def log_prob_des(theta):
    lp = log_prior_des(theta)
    if not np.isfinite(lp): return -np.inf

    Omk, g0, gz, beta, dint, H0_fit = theta
    gamma_z = g0 + gz * zl_des
    if np.any(gamma_z <= 1.05) or np.any(gamma_z >= 2.95): return -np.inf

    sigma_th_0 = calc_sigma_th_des(mu_l_obs_des, mu_s_obs_des, Omk, gamma_z, delta_lum_des, beta, H0_fit)
    if np.any(np.isnan(sigma_th_0)): return -np.inf

    eps = 1e-4
    sigma_th_dl = calc_sigma_th_des(mu_l_obs_des + eps, mu_s_obs_des, Omk, gamma_z, delta_lum_des, beta, H0_fit)
    sigma_th_ds = calc_sigma_th_des(mu_l_obs_des, mu_s_obs_des + eps, Omk, gamma_z, delta_lum_des, beta, H0_fit)

    J_l = (sigma_th_dl - sigma_th_0) / eps
    J_s = (sigma_th_ds - sigma_th_0) / eps

    term_l  = J_l[:, None] * C_l * J_l[None, :]
    term_s  = J_s[:, None] * C_s * J_s[None, :]
    term_ls = J_l[:, None] * C_ls * J_s[None, :]
    term_sl = J_s[:, None] * C_ls.T * J_l[None, :]

    C_prop_SNe = term_l + term_s + term_ls + term_sl
    C_total = np.diag(sigma_err_des**2 + (sigma_th_0 * dint)**2) + C_prop_SNe
    delta_sig = sigma_obs_des - sigma_th_0

    try:
        c_and_l = cho_factor(C_total)
        logL = -0.5 * np.dot(delta_sig, cho_solve(c_and_l, delta_sig)) - np.sum(np.log(np.diag(c_and_l[0])))
        return lp + logL
    except np.linalg.LinAlgError:
        return -np.inf

# ====================================================================
# [模块 B] Union3-Cobaya 高斯过程基准模型 (GP Baseline)
# ====================================================================
print("📥 [Module B] 加载 Union3 Cobaya GP 背景数据...")
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
K_Xstar = kernel(z_nodes, z_targets).T
mu_pred = np.dot(K_Xstar, alpha) + mean_f(z_targets)
mu_l_gp, mu_s_gp = mu_pred[:len(sgl_df)], mu_pred[len(sgl_df):]

def calc_sigma_th_union(dl, ds, dls, theta_E_arc, theta_ap_arc, gamma_val, delta_val, beta_ani):
    xi = gamma_val + delta_val - 2.0
    t1 = gamma_func((xi - 1.0)/2.0) / gamma_func(xi/2.0)
    t2 = beta_ani * gamma_func((xi + 1.0)/2.0) / gamma_func((xi + 2.0)/2.0)
    t3 = (gamma_func(gamma_val/2.0) * gamma_func(delta_val/2.0)) / \
         (gamma_func((gamma_val - 1.0)/2.0) * gamma_func((delta_val - 1.0)/2.0))
    denom = (xi - 2.0 * beta_ani) * (3.0 - xi)
    if np.any(denom == 0) or np.any(~np.isfinite(t1)) or np.any(~np.isfinite(t2)) or np.any(~np.isfinite(t3)):
        return np.full_like(dl, np.nan)

    F = ((3.0 - delta_val) / denom) * (t1 - t2) * t3
    theta_E_rad = theta_E_arc * (np.pi / 648000.0)
    sigma_sq = (C_LIGHT**2 / (2.0 * np.sqrt(np.pi))) * (ds / dls) * theta_E_rad * F * (theta_ap_arc / theta_E_arc)**(2.0 - gamma_val)
    return np.sqrt(np.maximum(sigma_sq, 0))

def log_prior_union(theta):
    Omk, g0, gz, beta, dint, H0_fit = theta
    if not (-0.1 < Omk < 2.0 and 1.0 < g0 < 3.0 and -0.5 < gz < 0.5 and \
            -1.0 < beta < 1.0 and 0.0 < dint < 0.3 and 60 < H0_fit < 85):
        return -np.inf
    return -0.5 * ((H0_fit - 73.04) / 1.42)**2

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
    logL = -0.5 * np.sum(((sigma_obs_u - sigma_th)**2 / var_tot) + np.log(2.0 * np.pi * var_tot))
    return lp + logL

# ====================================================================
# [主控中心] 运行并行采样并生成重叠对比图
# ====================================================================
if __name__ == '__main__':
    labels = [r"\Omega_k", r"\gamma_0", r"\gamma_z", r"\beta_{ani}", r"\delta_{int}", r"H_0"]
    print_labels = ["Omega_k", "gamma_0", "gamma_z", "beta_ani", "delta_int", "H_0"]

    ndim, nwalkers = 6, 64
    N_steps = 15000
    num_cores = cpu_count()

    # 优秀的起始点可加速预热收敛
    initial_des = np.array([0.15, 1.9, -0.35, -0.15, 0.10, 73.0])
    initial_uni = np.array([0.12, 2.1, -0.42, -0.45, 0.09, 73.0])

    pos_des = initial_des + 1e-4 * np.random.randn(nwalkers, ndim)
    pos_uni = initial_uni + 1e-4 * np.random.randn(nwalkers, ndim)

    print(f"\n🚀 准备开启火力！检测到 {num_cores} 个 CPU 逻辑核心。")

    # ------------------ 运行 DES MCMC ------------------
    print("\n" + "="*50)
    print(f"🔥 开始运行 [模型 A]: DES-Dovekie 严苛配对 ({N_steps} 步)")
    with Pool(processes=num_cores) as pool:
        sampler_des = emcee.EnsembleSampler(nwalkers, ndim, log_prob_des, pool=pool)
        t0 = time.time()
        sampler_des.run_mcmc(pos_des, N_steps, progress=True)
        print(f"✅ 模型 A 采样完成，耗时: {(time.time()-t0)/60:.2f} 分钟。")

    samples_des = sampler_des.get_chain(discard=5000, thin=15, flat=True)
    stats_des = cosmo_tools.calculate_stats(samples_des, labels)

    # ------------------ 运行 Union MCMC ------------------
    print("\n" + "="*50)
    print(f"🔥 开始运行 [模型 B]: Union3 高斯过程平滑 ({N_steps} 步)")
    with Pool(processes=num_cores) as pool:
        sampler_uni = emcee.EnsembleSampler(nwalkers, ndim, log_prob_union, pool=pool)
        t0 = time.time()
        sampler_uni.run_mcmc(pos_uni, N_steps, progress=True)
        print(f"✅ 模型 B 采样完成，耗时: {(time.time()-t0)/60:.2f} 分钟。")

    samples_uni = sampler_uni.get_chain(discard=5000, thin=15, flat=True)
    stats_uni = cosmo_tools.calculate_stats(samples_uni, labels)

    # ------------------ 打印对比结果表 ------------------
    print("\n" + "★"*85)
    print(f"{'参数':<12} | {'DES-Dovekie Exact Pairing':<30} | {'Union3 Gaussian Process':<30}")
    print("-" * 85)
    for i in range(ndim):
        des_fmt = f"{stats_des[i]['median']:.4f} +{stats_des[i]['upper']:.4f}/-{stats_des[i]['lower']:.4f}"
        uni_fmt = f"{stats_uni[i]['median']:.4f} +{stats_uni[i]['upper']:.4f}/-{stats_uni[i]['lower']:.4f}"
        print(f"{print_labels[i]:<12} | {des_fmt:<30} | {uni_fmt:<30}")
    print("★"*85)

    # ------------------ 绘制终极 Overlay Contour Plot ------------------
    output_pdf = "figure01.pdf"

    # 调用 cosmo_tools 的高级多链对比画图接口
    cosmo_tools.plot_getdist_comparison(
        samples_list=[samples_des, samples_uni],
        labels=labels,
        legend_labels=["DES-Dovekie (Exact Pairing)", "Union3 (Gaussian Process)"],
        colors=["#d62728", "#1f77b4"],  # 红色给DES(系统要求高)，蓝色给Union(平滑性好)
        filename=output_pdf
    )

    print(f"\n🎉 双重交叉验证对比图绘制完成！已保存为: {output_pdf}")
    print("你可以直接将其作为 APJ 论文的核心图表 ")
