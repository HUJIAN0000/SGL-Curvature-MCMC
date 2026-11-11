"""Same-lens, same-DES-data exact-target-redshift distance checks.

Two signed-transfer fits alter only the finite-redshift assignment model.
Two GP fits use the same 176 observed DES rows and full covariance, but add
a smooth common Hubble-curve prior. They are sensitivity analyses, not
independent data or a model-free replacement for paired distances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr, chdtrc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "exact_redshift"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "reproduction"))
from inference import C, distance
from physical_prior_revision import PhysicalModel, LOWER
from supplement_sampler import sample_joint
from supplement_diagnostics import predict_joint

CASES = ("signed_envelope", "signed_no_envelope", "gp_log", "gp_fid")
GP_VARIANCE = 1.5  # mag^2; same fixed reference kernel as prior Union3 comparison
GP_LENGTH = 1.2    # redshift units


def frame_term(zhel, zhd):
    """mu_Dovekie - mu_standard(zHD), from (1+zHD)(1+zHEL) D_A."""
    return 5 * np.log10((1 + zhel) / (1 + zhd))


def luminosity_distance(z):
    z = np.asarray(z, dtype=np.float64)
    return (C / 70.0) * (1 + z) * distance(z)


def mu_fid(z):
    return 25 + 5 * np.log10(luminosity_distance(z))


def kernel(x, y):
    return GP_VARIANCE * np.exp(-0.5 * ((x[:, None] - y[None, :]) / GP_LENGTH) ** 2)


def universal_gp(ztrain, y, noise, ztarget, mean_kind):
    """Condition a zero-mean GP with an uninformative common intercept.

    The extra rank-one predictive covariance term propagates estimation of
    the catalogue modulus zero-point. Full correlated DES noise is retained.
    """
    mean = (lambda z: 5 * np.log10(z) + 43) if mean_kind == "log" else mu_fid
    residual = y - mean(ztrain)
    train = kernel(ztrain, ztrain) + noise
    factor = cho_factor(train, lower=True, check_finite=True)
    ones = np.ones(len(ztrain))
    solve_ones = cho_solve(factor, ones)
    denominator = float(ones @ solve_ones)
    intercept = float(ones @ cho_solve(factor, residual) / denominator)
    cross = kernel(ztarget, ztrain)
    solved_residual = cho_solve(factor, residual - intercept)
    predicted = mean(ztarget) + intercept + cross @ solved_residual
    u = np.ones(len(ztarget)) - cross @ solve_ones
    covariance = kernel(ztarget, ztarget) - cross @ cho_solve(factor, cross.T)
    covariance += np.outer(u, u) / denominator
    covariance = (covariance + covariance.T) / 2
    eigmin = float(np.linalg.eigvalsh(covariance)[0])
    if eigmin < -1e-8:
        raise ValueError(f"GP predictive covariance not PSD: {eigmin}")
    covariance += np.eye(len(ztarget)) * 1e-10
    np.linalg.cholesky(covariance)
    return predicted, covariance, {"intercept_mag": intercept, "minimum_covariance_eigenvalue_before_jitter": eigmin,
                                   "conditional_covariance_diagonal_median": float(np.median(np.diag(covariance))),
                                   "mean_kind": mean_kind, "kernel_variance_mag2": GP_VARIANCE, "kernel_length_redshift": GP_LENGTH,
                                   "intercept_treatment": "flat prior, universal-kriging rank-one covariance propagated"}


class ExactRedshiftModel(PhysicalModel):
    def __init__(self, case):
        assert case in CASES
        super().__init__("base")
        self.case = case
        self.n = len(self.df)
        assert self.n == 88
        df = self.df
        lens_indices = df.sn_lens_idx.to_numpy(int)
        source_indices = df.sn_source_idx.to_numpy(int)
        indices = np.r_[lens_indices, source_indices]
        assert len(np.unique(indices)) == 176
        parent = np.load(ROOT / "reproduction" / "des_covariance.npy", mmap_mode="r")
        full_noise = np.asarray(parent[np.ix_(indices, indices)], dtype=np.float64)
        full_noise = (full_noise + full_noise.T) / 2
        np.linalg.cholesky(full_noise)
        ztrain = np.r_[df.sn_lens_zHD.to_numpy(float), df.sn_source_zHD.to_numpy(float)]
        zhel = np.r_[df.sn_lens_zHEL.to_numpy(float), df.sn_source_zHEL.to_numpy(float)]
        ztarget = np.r_[df.zl.to_numpy(float), df.zs.to_numpy(float)]
        observed = np.r_[df.sn_lens_MU.to_numpy(float), df.sn_source_MU.to_numpy(float)]
        corrected = observed - frame_term(zhel, ztrain)
        transfer = mu_fid(ztarget) - mu_fid(ztrain)
        if case.startswith("signed"):
            predicted = corrected + transfer
            covariance = full_noise.copy()
            if case == "signed_envelope":
                # Same mismatch rms envelope as primary, with its signed
                # fiducial mean transfer stated separately.
                covariance += np.diag(transfer**2)
            details = {"method": "fiducial signed luminosity-distance transfer",
                       "reference_Omega_m": .3, "mismatch_envelope_included": case == "signed_envelope"}
        else:
            predicted, covariance, details = universal_gp(ztrain, corrected, full_noise, ztarget, case.split("_")[1])
            details["method"] = "full-covariance DES GP on the same 176 matched SN rows"
            details["mismatch_envelope_included"] = False
        np.linalg.cholesky(covariance)
        n = self.n
        self.ml = np.ascontiguousarray(predicted[:n])
        self.ms = np.ascontiguousarray(predicted[n:])
        self.cl = np.ascontiguousarray(covariance[:n, :n])
        self.cs = np.ascontiguousarray(covariance[n:, n:])
        self.cls = np.ascontiguousarray(covariance[:n, n:])
        self.xl = 70 / C * 10 ** ((self.ml - 25) / 5) / (1 + self.zl)
        self.xs = 70 / C * 10 ** ((self.ms - 25) / 5) / (1 + self.zs)
        self.details = {**details,
                        "same_88_lens_order": df["Lens name"].tolist(),
                        "selected_DES_original_indices": indices.tolist(),
                        "training_n": len(ztrain), "target_n": len(ztarget),
                        "frame_term_min_max_mag": [float(frame_term(zhel, ztrain).min()), float(frame_term(zhel, ztrain).max())],
                        "fiducial_signed_shift_abs_median_mag": float(np.median(np.abs(transfer))),
                        "frozen_parent_covariance_selected_sha256": hashlib.sha256(full_noise.tobytes()).hexdigest(),
                        "input_mean_and_covariance_sha256": hashlib.sha256(predicted.tobytes() + covariance.tobytes()).hexdigest(),
                        "same_CMB_lower": LOWER, "same_gamma_z_prior": [-.5, .5], "same_H0": 70.0}
        np.savez_compressed(OUT / f"{case}_distance_factor.npz", mean=predicted, covariance=covariance,
                            observed=observed, frame_corrected=corrected, train_zHD=ztrain, train_zHEL=zhel,
                            target_z=ztarget, selected_indices=indices, fiducial_signed_shift=transfer)
        (OUT / f"{case}_design.json").write_text(json.dumps(self.details, indent=2), encoding="utf-8")


def diagnostics(case, model):
    folder = OUT / case
    summary = json.loads((folder / "joint_summary.json").read_text())
    chains = np.load(folder / "joint_chains.npy", mmap_mode="r")
    mu = np.load(folder / "joint_mu.npy", mmap_mode="r")
    iterations = np.asarray(summary["mu_saved_indices"], dtype=int)
    assert chains.shape == (4, 100000, 5)
    assert mu.shape == (4, len(iterations), 176)
    theta = np.asarray(chains[:, iterations]).reshape(-1, 5)
    latent = np.asarray(mu).reshape(-1, 176)
    prediction, variance = predict_joint(model, theta, latent)
    residual = (model.obs - prediction) / np.sqrt(variance)
    zs = model.zs
    weight = (zs - zs.mean()) / np.linalg.norm(zs - zs.mean())
    design = np.column_stack([np.ones(model.n), model.zl,
                              pd.get_dummies(model.df["Survey name"], drop_first=True).to_numpy(float)])
    partial_x = zs - design @ np.linalg.lstsq(design, zs, rcond=None)[0]
    partial_weight = partial_x / np.linalg.norm(partial_x)
    raw_score = residual @ weight
    partial_score = residual @ partial_weight
    overall = np.sum(residual**2, axis=1)
    result = {"case": case, "N_lenses": model.n, "joint_states": len(theta),
              "source_redshift_PPC_analytic": float(np.mean(2 * ndtr(-np.abs(raw_score)))),
              "source_redshift_signed_score_median": float(np.median(raw_score)),
              "partial_source_redshift_PPC_analytic": float(np.mean(2 * ndtr(-np.abs(partial_score)))),
              "overall_squared_residual_PPC_analytic": float(np.mean(chdtrc(model.n, overall))),
              "Omega_quantiles": np.asarray(summary["quantiles"])[:, 0].tolist(),
              "gamma_z_quantiles": np.asarray(summary["quantiles"])[:, 2].tolist(),
              "max_Rhat": summary["diagnostics"]["max_rhat"],
              "min_ESS": summary["diagnostics"]["min_ess"],
              "interpretation": "In-sample posterior-predictive probability under this specified distance assignment; not a frequentist p-value or an independent validation."}
    (folder / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()
    rows = []
    for case in args.cases.split(","):
        model = ExactRedshiftModel(case)
        folder = OUT / case
        folder.mkdir(exist_ok=True)
        prefix = folder / "joint"
        if not (folder / "joint_summary.json").exists():
            sample_joint(model, prefix, steps=args.steps, nchains=args.chains,
                         seed=2026102201 + CASES.index(case) * 100003)
        else:
            print("PRESERVING EXISTING", case, flush=True)
        if args.steps == 100000 and args.chains == 4:
            row = diagnostics(case, model)
            rows.append(row)
            print("DIAGNOSTIC", case, json.dumps({k: row[k] for k in
                  ("Omega_quantiles", "gamma_z_quantiles", "source_redshift_PPC_analytic", "partial_source_redshift_PPC_analytic", "max_Rhat", "min_ESS")}), flush=True)
    if len(rows) == len(CASES):
        (OUT / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
