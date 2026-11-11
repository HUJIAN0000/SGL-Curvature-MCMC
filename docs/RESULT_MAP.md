> **This upload omits MCMC and latent-distance chains.** Existing figure PDFs and previews are in `figures/`; inputs, summaries and code remain included. Chain-dependent commands below require separately supplied arrays. Historical provenance reports describe earlier checks; the upload verification record is `provenance/package_checks.json`.

# Result-to-code map

`provenance/result_registry.csv` is the machine-readable inventory of all **28** saved MCMC analyses reported in the *ApJ* manuscript. Each row specifies its JSON summary, parameter chain path, optional aligned latent-modulus chain path, sample size, burn-in, modeling conventions, and posterior intervals. Full `.npy` chain arrays are supplied in the companion chain archive (`saved_chains_apj_v6_20261025.zip`) indexed by `provenance/chain_files.json`.

| Manuscript role | Core modules & inputs | Saved summaries / outputs | CLI reproduction command |
|---|---|---|---|
| Primary 88-lens nonlinear result (Table 2, Fig. 1–2) | `reproduction/physical_prior_revision.py`, `latent_validation.py` | `reproduction_physical/nonlinear_base_*` | `python reproduce.py sample --cases base` |
| Soft CMB weight, dynamical & prior checks (6 nonlinear runs, Table 2–3) | `PhysicalModel` variant switches (`soft07`, `lens_noise`, `beta_prior`, `delta_plus`, `delta_minus`, `gzwide`) | `reproduction_physical/nonlinear_{soft07,lens_noise,beta_prior,delta_plus,delta_minus,gzwide}_*` | `python reproduce.py sample --cases ...` |
| Linearized Jacobian & Union3 GP comparisons (6 runs, Table 2–3, Fig. 1–2) | `PhysicalModel`, `inference.py`, analytic Jacobians, Union3 nodes | `reproduction_physical/linear_{base,diagonal,nomismatch,transfer,union_same,union_all}_*` | `python reproduce.py sample --kind linear --cases ... --steps 12000` |
| SN-cut, matching-tolerance & upper-bound refits (5 nonlinear runs, Table 4) | `reproduction_supplement/run_sensitivity.py`, `supplement_sampler.py`, `CosmoMatcherv1.4.3.py` | `reproduction_supplement/sensitivity/{quality025,quality035,tol03,tol07,upper5}/nonlinear_*` | `python reproduce.py sensitivity` |
| Fixed-design injection--recovery simulations (6 nonlinear runs, App. B, Fig. 6) | `reproduction_supplement/simulation_checks.py`, `supplement_sampler.py` | `reproduction_supplement/simulations/{constant,evolving}_{01,02,03}_*` | `python reproduce.py simulate` |
| Same-DES exact-target-redshift alternatives (4 nonlinear runs, Table 5, Fig. 4) | `reproduction_supplement/exact_redshift_checks.py`, `supplement_sampler.py` | `reproduction_supplement/exact_redshift/{signed_envelope,signed_no_envelope,gp_log,gp_fid}/joint_*` | `python reproduce.py exact-redshift` |
| Posterior-predictive velocity residuals & $\gamma(z_l)$ bands (Sec. 4.1 & 4.4, Fig. 3) | `supplement_diagnostics.py`, `analytic_source_trend_checks.py`, `sensitivity_predictive_checks.py` | `reproduction_supplement/diagnostics/` | `python reproduce.py diagnostics` |
| Publication figures & consolidated interval table | `render_results.py`, `addcode/cosmo_tools.py` | Frozen: `figures/fig_*.pdf`; Regenerated: `outputs/figures/` | `python reproduce.py figures` |

Note: `reproduction_supplement/sensitivity/reference/` stores the standalone v1.4.3 assignment verification against the primary 88-lens map rather than a separate MCMC run.
