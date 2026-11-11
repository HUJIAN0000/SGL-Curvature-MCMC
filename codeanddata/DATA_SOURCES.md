# Input provenance and roles

| File | Role and origin |
|---|---|
| `130sgls2.CSV` | 130-lens parent compilation associated with Chen et al. (2019), MNRAS, 488, 3745 (`10.1093/mnras/stz1902`), drawing from SLACS, S4TM, BELLS, and BELLS GALLERY. Contains survey identifiers, lens/source redshifts, individual stellar luminosity-density slopes ($\delta_i$), angular scales, and velocity dispersions/errors. |
| `DES-Dovekie_HD.csv` | 1820-row DES-Dovekie Hubble-diagram catalog from `https://github.com/des-science/DES-SN5YR` (Popovic et al. 2026, MNRAS, 548, stag632). Used for the primary paired-distance and exact-redshift analyses. |
| `covtot_inv_000.npz` | Packed upper triangle of the full $1820 \times 1820$ DES-Dovekie statistical-plus-systematic **inverse** covariance matrix (`cov`, `nsn`). `python reproduce.py prepare` reconstructs the full precision matrix and inverts it via Cholesky solve before any row selection. |
| `lcparam_full.txt`, `mag_covmat.txt` | Compressed 22-node **Union3** distance-modulus spline vector and $22 \times 22$ covariance matrix (Rubin et al. 2025, ApJ, 986, 231) as distributed in `https://github.com/CobayaSampler/sn_data/tree/master/Union3`. Used strictly for the separate Union3 GP comparison runs. |
| `matched_sgls_milp_dd.csv`, `cov_sgls_milp_dd_{lens,source,cross}.txt` | Initial 92-lens single-stage pairing table and covariance sub-blocks retained solely for historical comparison in Table 1 and Figure 5; all revised analyses select directly from the reconstructed $1820 \times 1820$ covariance. |

File checksums are recorded in `provenance/repository_manifest.json`. Derived two-stage v1.4.3 matching outputs and quality-filtered index maps are stored under `reproduction/` and `reproduction_supplement/sensitivity/`.

Original survey collaborations and compilation authors retain full credit and data-usage terms for these observational products; see `LICENSE_SCOPE.md`.
