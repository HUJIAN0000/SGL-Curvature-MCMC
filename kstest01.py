# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 02:45:10 2026

@author: Hu Jian
Email: dg1626002@smail.nju.edu.cn
"""

import pandas as pd
from scipy.stats import ks_2samp

# Kolmogorov-Smirnov Test
# Objective: Verify whether the 92 matched samples have selection bias relative to the original 130 samples.

try:
    # 1. Load Parent Sample (130) and Matched Sample (92)
    df_parent = pd.read_csv('130sgls2.CSV')
    df_matched = pd.read_csv('matched_sgls_milp_dd.csv')

    print("=== Kolmogorov-Smirnov (K-S) Test Results ===")
    print("Null Hypothesis: The 92 matched samples are drawn from the same distribution as the 130 parent samples.\n")

    # List of key physical parameters to test
    test_params = {
        'zl': 'Lens Redshift',
        'zs': 'Source Redshift',
        'thetaE': 'Einstein Radius',
        'sigma_ap': 'Velocity Dispersion'
    }

    for col, name in test_params.items():
        data1 = df_parent[col].dropna()
        data2 = df_matched[col].dropna()
        
        # Execute K-S Test
        stat, p_val = ks_2samp(data1, data2)
        
        print(f"[{name} ({col})]")
        print(f"  - K-S Statistic (D) = {stat:.4f}")
        print(f"  - p-value = {p_val:.4f}")
        
        # If p-value > 0.05, the null hypothesis cannot be rejected (distributions are consistent)
        if p_val > 0.05:
            print("  -> Conclusion: NO significant statistical difference (consistent).")
        else:
            print("  -> Conclusion: Significant statistical difference detected.")
        print()

except FileNotFoundError:
    print("Error: [Errno 2] No such file or directory. Please ensure that '130sgls2.CSV' and 'matched_result_sgls.csv' exist in the current directory.")