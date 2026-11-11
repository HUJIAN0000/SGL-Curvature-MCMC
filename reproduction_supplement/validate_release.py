"""Independent structural and numerical checks for the v4 supplement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = ["quality025", "quality035", "tol03", "tol07", "upper5"]
MOCKS = [f"{kind}_{i:02d}" for kind in ("constant", "evolving") for i in range(1, 4)]
Q = (15.865, 50.0, 84.135)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, folder: Path, expected_steps: int) -> dict:
    summary = json.loads((folder / f"{name}_summary.json").read_text())
    theta_path = folder / f"{name}_chains.npy"
    mu_path = folder / f"{name}_mu.npy"
    theta = np.load(theta_path, mmap_mode="r")
    mu = np.load(mu_path, mmap_mode="r")
    assert theta.shape == (4, expected_steps, 5)
    assert mu.shape == (4, len(summary["mu_saved_indices"]), 2 * summary["N"])
    assert np.isfinite(theta).all() and np.isfinite(mu).all()
    assert summary["convergence_screen"]
    assert summary["diagnostics"]["max_rhat"] < 1.01
    assert summary["diagnostics"]["min_ess"] > 1000
    assert summary["steps"] == expected_steps
    indices = np.asarray(summary["mu_saved_indices"])
    assert len(indices) > 0 and np.all(np.diff(indices) == 100)
    assert indices[0] >= summary["burn"] and indices[-1] < expected_steps
    bounds = np.asarray(summary["bounds"])
    assert np.all(theta.min(axis=(0, 1)) > bounds[:, 0])
    assert np.all(theta.max(axis=(0, 1)) < bounds[:, 1])
    quantiles = np.percentile(theta[:, summary["burn"]:, :], Q, axis=(0, 1))
    assert np.allclose(quantiles, summary["quantiles"], atol=2e-6, rtol=0)
    return {
        "name": name,
        "N": summary["N"],
        "max_rhat": summary["diagnostics"]["max_rhat"],
        "min_ess": summary["diagnostics"]["min_ess"],
        "theta_shape": list(theta.shape),
        "mu_shape": list(mu.shape),
        "Omega_quantiles": quantiles[:, 0].tolist(),
        "gamma_z_quantiles": quantiles[:, 2].tolist(),
        "sha256": {theta_path.name: digest(theta_path), mu_path.name: digest(mu_path)},
    }


def main() -> None:
    result = {"sensitivity": [], "mock": [], "checks": []}
    for case in CASES:
        folder = ROOT / "sensitivity" / case
        row = check("nonlinear", folder, 100000)
        design = json.loads((folder / "design.json").read_text())
        assert design["matched_lenses"] == row["N"]
        assert design["software_version"] == "1.4.3"
        assert design["Omega_lower"] == -0.09405370006986057
        assert design["gamma_z_bounds"] == [-0.5, 0.5]
        result["sensitivity"].append({**row, "case": case, "design": design})
    for name in MOCKS:
        folder = ROOT / "simulations"
        row = check(name, folder, 60000)
        data = json.loads((folder / f"{name}_data.json").read_text())
        assert data["truth"][0] == 0.0
        assert data["truth"][2] == (0.0 if name.startswith("constant") else -0.2)
        assert data["N_lenses"] == row["N"] == 88
        result["mock"].append({**row, "truth": data["truth"], "data_sha256": digest(folder / f"{name}_data.npz")})
    base = ROOT.parent / "reproduction_physical"
    diagnostics = json.loads((ROOT / "diagnostics" / "diagnostics_summary.json").read_text())
    for rel, expected in diagnostics["posterior_predictive"]["input_sha256"].items():
        path = ROOT.parent / Path(rel.replace("\\", "/"))
        assert digest(path) == expected, path
    # The reference assignment must reproduce the original frozen 88 triplets.
    design = json.loads((ROOT / "sensitivity" / "reference" / "design.json").read_text())
    assert design["matched_lenses"] == design["identical_triplets_with_reference"] == 88
    assert (base / "nonlinear_base_chains.npy").is_file()
    result["checks"] = ["all stored quantiles", "joint theta/mu shapes and iteration alignment", "hard bounds", "diagnostic thresholds", "mock truth/design", "old primary input digests", "v1.4.3 reference assignment"]
    (ROOT / "validation_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"sensitivity": len(result["sensitivity"]), "mock": len(result["mock"]), "max_rhat": max(x["max_rhat"] for x in result["sensitivity"] + result["mock"]), "min_ess": min(x["min_ess"] for x in result["sensitivity"] + result["mock"])}, indent=2))


if __name__ == "__main__":
    main()
