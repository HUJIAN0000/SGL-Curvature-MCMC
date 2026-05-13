# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 01:29:36 2026

@author: Hu Jian
Email: dg1626002@smail.nju.edu.cn
"""

import numpy as np
import cupy as cp
import time

start_time = time.time()

# =========================================================
# 1. Load compressed data from hard drive on CPU
# =========================================================
print("📥 1/4 Loading compressed data...")
npz_file = np.load('covtot_inv_000.npz')
nsn = int(npz_file['nsn'][0]) 
cov_1d = npz_file['cov']
print(f"   - Number of SNe (nsn): {nsn}")

# =========================================================
# 2. Transfer to GPU and reconstruct 2D symmetric matrix
# =========================================================
print("🚀 2/4 Sending data to RTX 5090 and reconstructing matrix in GPU VRAM...")
t_gpu_start = time.time()

# Transfer 1D data to GPU VRAM
cov_1d_gpu = cp.asarray(cov_1d)

# Allocate and build 1820x1820 matrix directly on GPU
C_inv_full_gpu = cp.zeros((nsn, nsn))
row_indices, col_indices = cp.triu_indices(nsn)

# Fill upper triangle and make symmetric (All done in GPU)
C_inv_full_gpu[row_indices, col_indices] = cov_1d_gpu
C_inv_full_gpu = C_inv_full_gpu + C_inv_full_gpu.T - cp.diag(cp.diag(C_inv_full_gpu))

# =========================================================
# 3. Matrix Inversion on GPU
# =========================================================
print("⚡ 3/4 Calling GPU kernels for matrix inversion...")
# High precision inversion on GPU
cov_matrix_gpu = cp.linalg.inv(C_inv_full_gpu)

# Wait for GPU to finish calculations
cp.cuda.Stream.null.synchronize()
t_gpu = time.time() - t_gpu_start
print(f"   - GPU computation and transfer total time: {t_gpu:.4f} seconds")

# =========================================================
# 4. Save Result
# =========================================================
print(f"💾 4/4 Writing {nsn*nsn/1e6:.2f} million data points to DES_Dovekie_CovTotal.txt ...")
print("   (Note: Writing large TXT files is limited by disk I/O, please wait patiently)")

# Convert back to Numpy and save
cov_np = cp.asnumpy(cov_matrix_gpu)
np.savetxt('DES_Dovekie_CovTotal.txt', cov_np, fmt='%.8e')

print(f"✅ Generation complete! Total program time: {time.time() - start_time:.2f} seconds.")