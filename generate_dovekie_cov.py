"""
Created on Wed Apr 29 01:26:04 2026

@author: Hu Jian
Email: dg1626002@smail.nju.edu.cn
"""

import numpy as np
import scipy.linalg as la
import time

start_time = time.time()

# =========================================================
# 1. Load compressed inverse matrix data
# =========================================================
print("📥 1/3 Loading DES-Dovekie compressed inverse matrix...")
npz_file = np.load('covtot_inv_000.npz')
nsn = int(npz_file['nsn'][0]) 
cov_1d = npz_file['cov']
print(f"   - Number of SNe (nsn): {nsn}")

# =========================================================
# 2. Reconstruct 1820x1820 2D symmetric inverse matrix
# =========================================================
print("🔄 2/3 Reconstructing 2D symmetric matrix...")
C_inv_full = np.zeros((nsn, nsn))
row_indices, col_indices = np.triu_indices(nsn)

# Fill upper triangle and make symmetric
C_inv_full[row_indices, col_indices] = cov_1d
C_inv_full = C_inv_full + C_inv_full.T - np.diag(np.diag(C_inv_full))

# =========================================================
# 3. Fast Matrix Inversion using Cholesky Decomposition
# =========================================================
print("⚡ 3/3 Performing ultra-fast Cholesky inversion on CPU...")
t_cpu_start = time.time()

try:
    # 协方差的逆矩阵是对称正定的，Cholesky 分解速度极快且调用 AVX-512/FMA
    c_and_lower = la.cho_factor(C_inv_full)
    # 求解 C_inv * X = I，得到原协方差矩阵
    cov_matrix = la.cho_solve(c_and_lower, np.eye(nsn))
    print(f"   - CPU computation time: {time.time() - t_cpu_start:.4f} seconds")
except la.LinAlgError:
    print("⚠️ Cholesky decomposition failed. Matrix may not be positive definite. Falling back to standard inversion...")
    cov_matrix = la.inv(C_inv_full)

# Save result
output_file = 'DES_Dovekie_CovTotal.txt'
print(f"💾 Saving to {output_file}...")
np.savetxt(output_file, cov_matrix, fmt='%.8e')

print(f"✅ Finished! Total program time: {time.time() - start_time:.2f} seconds.")