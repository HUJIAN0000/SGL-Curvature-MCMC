"""Scientific figure for same-lens DES distance-assignment comparisons."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from exact_redshift_checks import ExactRedshiftModel, OUT
from supplement_diagnostics import read_joint, predict_joint


def sample(case):
    if case == "baseline":
        summary, chains, theta, latent, _, _ = read_joint("base")
        model = ExactRedshiftModel("signed_envelope")
        # Baseline predictions use the same lens observables and target z;
        # their latent distances are from the baseline factor, not this model.
        return summary, chains, theta, latent, model
    folder = OUT / case
    summary = json.loads((folder / "joint_summary.json").read_text())
    chains = np.load(folder / "joint_chains.npy", mmap_mode="r")
    latent = np.load(folder / "joint_mu.npy", mmap_mode="r")
    idx = np.asarray(summary["mu_saved_indices"], int)
    theta = np.asarray(chains[:, idx, :]).reshape(-1, 5)
    latent = np.asarray(latent).reshape(-1, 176)
    return summary, chains, theta, latent, ExactRedshiftModel(case)


def main():
    cases = ["baseline", "signed_envelope", "gp_log", "gp_fid"]
    labels = ["paired baseline", "signed transfer", "DES GP: log mean", "DES GP: fiducial mean"]
    colors = ["#333333", "#bc5637", "#2677a8", "#557747"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.65))
    for j, (case, label, color) in enumerate(zip(cases, labels, colors)):
        summary, chains, theta, latent, model = sample(case)
        posterior = np.asarray(chains[:, summary["burn"]:, 0]).ravel()
        values = np.sort(posterior)
        axes[0].plot(values[::80], np.arange(1, len(values) + 1)[::80] / len(values),
                     color=color, lw=1.8, label=label)
        prediction, variance = predict_joint(model, theta, latent)
        residual = (model.obs - prediction) / np.sqrt(variance)
        order = np.argsort(model.zs)
        groups = np.array_split(order, 4)
        z = np.array([model.zs[g].mean() for g in groups])
        means = np.column_stack([residual[:, g].mean(axis=1) for g in groups])
        q = np.percentile(means, [15.865, 50, 84.135], axis=0)
        axes[1].errorbar(z + (j-1.5)*.008, q[1], yerr=[q[1]-q[0],q[2]-q[1]],
                         fmt="o-", color=color, lw=1.25, ms=4, capsize=2.5,
                         label=label)
    axes[0].axvline(0, color="0.55", ls="--", lw=.8)
    axes[0].axvline(-.09405370006986057, color="0.55", ls=":", lw=.8)
    axes[0].set(xlim=(-.10, .85), ylim=(0, 1), xlabel=r"Curvature $\Omega_k$", ylabel="Marginal posterior CDF")
    axes[1].axhline(0, color="0.55", ls="--", lw=.8)
    axes[1].set(xlim=(.35, 1.1), ylim=(-.7, .7), xlabel=r"Source redshift $z_s$ (quartile mean)",
                ylabel="Mean standardized velocity residual")
    axes[0].legend(loc="lower right", fontsize=7.2, frameon=False)
    axes[1].legend(loc="upper right", fontsize=7.2, frameon=False)
    for ax in axes: ax.grid(alpha=.14)
    fig.tight_layout()
    fig.savefig(OUT / "exact_redshift_comparison.pdf")
    fig.savefig(OUT / "exact_redshift_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
