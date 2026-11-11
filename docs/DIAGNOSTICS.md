> **This upload omits MCMC and latent-distance chains.** Existing figure PDFs and previews are in `figures/`; inputs, summaries and code remain included. Chain-dependent commands below require separately supplied arrays. Historical provenance reports describe earlier checks; the upload verification record is `provenance/package_checks.json`.

# Sampling diagnostics across the 28 saved analyses

The reproducibility registry (`provenance/result_registry.csv`) comprises **28 saved analyses** across two sampler classes:

1. **22 hierarchical nonlinear latent-modulus MCMC fits** (7 physical-prior variants in `reproduction_physical/`, 5 rematching/bound refits in `reproduction_supplement/sensitivity/`, 6 fixed-design simulations in `reproduction_supplement/simulations/`, and 4 same-DES exact-target-redshift reconstructions in `reproduction_supplement/exact_redshift/`):
   * Every nonlinear observational run executes 4 independent chains of 100,000 steps (`burn = 33333`); each synthetic-data run executes 4 independent chains of 60,000 steps (`burn = 20000`).
   * Across all 22 nonlinear runs, rank-normalized and folded split-$\widehat R$ values are below `1.002` and conservative autocorrelation-based effective sample sizes exceed `4500` (primary baseline: maximum $\widehat R = 1.000483$, minimum $\mathrm{ESS} = 6843$).
2. **6 linearized analytical-Jacobian `emcee` comparison fits** (`linear_base`, `linear_diagonal`, `linear_nomismatch`, `linear_transfer`, `linear_union_same`, `linear_union_all` in `reproduction_physical/`):
   * Each linearized comparison executes two independent 40-walker ensembles of 12,000 steps (`burn = 4000`), recording acceptance fractions and integrated autocorrelation-time summaries appropriate to affine-invariant ensemble walkers rather than four-chain split-$\widehat R$.

Running `python reproduce.py verify --chains` verifies all 50 restored `.npy` digests, recomputes the `[15.865, 50.0, 84.135]` percentiles for all 28 saved runs, and checks parameter/latent-state iteration alignment.
