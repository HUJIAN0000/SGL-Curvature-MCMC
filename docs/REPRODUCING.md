> **This upload omits MCMC and latent-distance chains.** Existing figure PDFs and previews are in `figures/`; inputs, summaries and code remain included. Chain-dependent commands below require separately supplied arrays. Historical provenance reports describe earlier checks; the upload verification record is `provenance/package_checks.json`.

# Reproducing and verifying results

All commands in `README.md` resolve paths relative to `reproduce.py`, so they can be executed from the repository root or any working directory without network access once Python dependencies (`requirements.txt`) are installed.

## Verification workflow

* `python reproduce.py prepare` reconstructs `reproduction/des_covariance.npy` ($1820 \times 1820$, `float64`) from the packed precision matrix `codeanddata/covtot_inv_000.npz` via Cholesky solve.
* `python reproduce.py verify` checks every tracked file against `provenance/repository_manifest.json`, validates the 88-lens / 176-SN covariance extraction and mismatch envelope, and evaluates reference joint likelihoods. (If you intentionally modify tracked files in a fork, regenerate `provenance/repository_manifest.json` accordingly.)
* `python reproduce.py restore /path/to/saved_chains_apj_v6_20261025.zip` verifies sizes and SHA-256 digests of all 50 chain and latent-modulus `.npy` files before restoring them, after which `python reproduce.py verify --chains` recomputes all 28 parameter quantile summaries and verifies aligned joint-state likelihoods.

## Chain conventions and latent-state alignment

* The first third of every chain is discarded as burn-in.
* In the 100,000-step nonlinear runs (`burn = 33333`), parameter chains have shape `(4, 100000, 5)` and thinned latent distance-modulus arrays have shape `(4, 666, 176)`, corresponding to exact iteration indices `33400, 33500, ..., 99900` (`((burn + 99) // 100) * 100` with stride `100`, recorded explicitly in new summaries as `mu_saved_indices`, with the documented index formula used for historical summaries). Do not pair `mu` with `theta[:, 33333::100]`.
* Linearized `emcee` comparison chains have shape `(2, 12000, 40, 5)` (`burn = 4000`).

## Running new fits and compiling the manuscript

* Sampling subcommands (`sample`, `sensitivity`, `simulate`, `exact-redshift`) and `diagnostics` create isolated timestamped workspaces under `outputs/` so that frozen reference summaries are never overwritten.
* Note that `reproduction/inference.py` defines the base linearized `Model` class without the external CMB bound; always run primary and sensitivity fits via `reproduce.py sample` (which invokes `physical_prior_revision.PhysicalModel`).

## Numerical portability

`reproduction_supplement/validate_exact_redshift.py` verifies the exact SHA-256 digest of each frozen distance factor used in the saved reference chains while comparing freshly recomputed GP mean vectors and predictive covariances within floating-point BLAS tolerances (`2e-10` mag and `2e-12` $\mathrm{mag}^2$). Across operating systems or BLAS implementations, fresh MCMC runs should be compared via converged posterior quantiles and $\widehat R$ / ESS diagnostics rather than byte-for-byte chain identity.

Convergence evidence is described separately for nonlinear chains and linearized ensembles in [DIAGNOSTICS.md](DIAGNOSTICS.md).
