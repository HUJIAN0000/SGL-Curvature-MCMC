# -*- coding: utf-8 -*-
"""
residual_mismatch.py
--------------------
Quantify the residual redshift-mismatch error of the MILP pairing sample
(fills the TODO in Section 2.2 of the manuscript).

For each of the 92 SGL-SN pairs it reports, separately for the lens-plane and
source-plane matches:

  1. the redshift offset |z_sn - z_target|;
  2. the fractional comoving-distance mismatch |dDc/Dc| (the quantity the MILP
     matcher minimises, capped at 0.05);
  3. the equivalent distance-modulus offset  Dmu_mis = |mu(z_sn) - mu(z_target)|
     under a fiducial flat LCDM cosmology;
  4. a comparison of Dmu_mis against the SN's own distance-modulus error MUERR,
     and the effect of adding Dmu_mis in quadrature to MUERR.

No astropy required: the comoving-distance integral is done with scipy.
The fiducial cosmology only enters as a smooth interpolator over a small
redshift interval, so the result is insensitive to its exact choice.
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad

# ---- fiducial cosmology (only used to map small redshift offsets) ----
OM0 = 0.30                      # matter density; flat LCDM assumed
# H0 cancels in every ratio below, so its value is irrelevant here.

def _E(z):
    return np.sqrt(OM0 * (1.0 + z)**3 + (1.0 - OM0))

def _I(z):
    r"""Dimensionless comoving-distance integral I(z) = \int_0^z dz'/E(z')."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.array([quad(lambda x: 1.0 / _E(x), 0.0, zi)[0] for zi in z])
    return out

def frac_dc(z_sn, z_target):
    """|Dc(z_sn) - Dc(z_target)| / Dc(z_target)  (c/H0 cancels)."""
    Isn, It = _I(z_sn), _I(z_target)
    return np.abs(Isn - It) / It

def dmu_mismatch(z_sn, z_target):
    """|mu(z_sn) - mu(z_target)| in mag for flat LCDM (c/H0 cancels in the ratio)."""
    z_sn = np.asarray(z_sn, float); z_target = np.asarray(z_target, float)
    dL_sn = (1.0 + z_sn) * _I(z_sn)
    dL_t  = (1.0 + z_target) * _I(z_target)
    return np.abs(5.0 * np.log10(dL_sn / dL_t))

def stats(x):
    return np.median(x), np.percentile(x, 95), np.max(x)

def line(label, x, unit=""):
    m, p95, mx = stats(x)
    print(f"  {label:<26s} median={m:8.4f}  95th={p95:8.4f}  max={mx:8.4f}  {unit}")

def main():
    df = pd.read_csv("matched_sgls_milp_dd.csv")
    n = len(df)
    print(f"Loaded {n} pairs.\n")

    zl   = df["zl"].to_numpy(float)
    zs   = df["zs"].to_numpy(float)
    zsn_l = df["sn_lens_zHD"].to_numpy(float)
    zsn_s = df["sn_source_zHD"].to_numpy(float)
    muerr_l = df["sn_lens_MUERR"].to_numpy(float)
    muerr_s = df["sn_source_MUERR"].to_numpy(float)

    # 1) redshift offsets
    dz_l = np.abs(zsn_l - zl)
    dz_s = np.abs(zsn_s - zs)
    print("Redshift offset |z_sn - z_target|:")
    line("lens plane",   dz_l)
    line("source plane", dz_s)
    line("combined",     np.concatenate([dz_l, dz_s]))
    print()

    # 2) fractional comoving-distance mismatch
    fl = frac_dc(zsn_l, zl)
    fs = frac_dc(zsn_s, zs)
    f_all = np.concatenate([fl, fs])
    print("Fractional comoving-distance mismatch |dDc/Dc|:")
    line("lens plane",   fl)
    line("source plane", fs)
    line("combined",     f_all)
    n_over = np.sum(f_all > 0.05)
    print(f"  pairs above 0.05 tolerance: {n_over} / {2*n}")
    print()

    # 3) equivalent distance-modulus offset
    dmu_l = dmu_mismatch(zsn_l, zl)
    dmu_s = dmu_mismatch(zsn_s, zs)
    dmu_all = np.concatenate([dmu_l, dmu_s])
    print("Equivalent distance-modulus offset Dmu_mis = |mu(z_sn)-mu(z_target)| [mag]:")
    line("lens plane",   dmu_l, "mag")
    line("source plane", dmu_s, "mag")
    line("combined",     dmu_all, "mag")
    # cross-check: the cap 0.05 in |dDc/Dc| corresponds to (5/ln10)*0.05 mag
    print(f"  (cross-check: tolerance 0.05 in |dDc/Dc| ~ {5/np.log(10)*0.05:.4f} mag)")
    print()

    # 4) comparison with the SN uncertainty actually used in the likelihood,
    #    i.e. the diagonal of the propagated SN covariance sub-blocks C_l, C_s.
    sig_l = np.sqrt(np.abs(np.diag(np.loadtxt("cov_sgls_milp_dd_lens.txt"))))
    sig_s = np.sqrt(np.abs(np.diag(np.loadtxt("cov_sgls_milp_dd_source.txt"))))
    sig_sn = np.concatenate([sig_l, sig_s])
    print("SN distance-modulus sigma from the covariance diagonal sqrt(diag(C)) [mag]:")
    good = sig_sn < 5.0     # drop unconstrained-SN outlier(s)
    line("sqrt(diag C) (good)", sig_sn[good], "mag")
    print(f"  dropped {np.sum(~good)} outlier(s) with sigma > 5 mag (unconstrained SNe)")
    ratio = dmu_all[good] / sig_sn[good]
    print(f"\n  Dmu_mis / sigma_SN : median={np.median(ratio):.3f}  95th={np.percentile(ratio,95):.3f}")
    infl = np.sqrt(sig_sn[good]**2 + dmu_all[good]**2) / sig_sn[good] - 1.0
    print(f"  fractional inflation of sigma_SN if Dmu_mis added in quadrature: "
          f"median={np.median(infl)*100:.2f}%  95th={np.percentile(infl,95)*100:.2f}%")

if __name__ == "__main__":
    main()
