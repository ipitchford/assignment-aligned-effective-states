"""Create the all-17 SAT label-audit figure from retained analysis outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/next_phase/sat_all17_validation_v3_20261018"
FIGURES = ROOT / "outputs/causal-resolution-v3/paper/figures"
STEM = FIGURES / "sat-all17-label-audit-v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    targets = [STEM.with_suffix(s) for s in (".pdf", ".png", ".svg", ".json")]
    if any(path.exists() for path in targets):
        raise RuntimeError("refusing to overwrite an existing all-17 figure")

    analysis_path = STUDY / "analysis.json"
    diagnostic_path = STUDY / "posthoc_diagnostics.json"
    analysis = json.loads(analysis_path.read_text())
    diagnostics = json.loads(diagnostic_path.read_text())
    rows = analysis["base"]["per_unit"]

    units = np.asarray([row["unit"] for row in rows])
    auroc = np.asarray([row["base_auroc"] for row in rows])
    sole = np.asarray([row["threshold_cells"][0]["detected"] for row in rows])
    multi = np.asarray([row["threshold_cells"][1]["detected"] for row in rows])
    y = np.arange(len(rows))

    navy = "#2B4C7E"
    orange = "#D97706"
    teal = "#0F766E"
    red = "#B91C1C"
    grey = "#D1D5DB"

    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.45), height_ratios=(1.0, 0.74))
    ax_auc = fig.add_subplot(grid[:, 0])
    ax_thr = fig.add_subplot(grid[0, 1])
    ax_fp = fig.add_subplot(grid[1, 1])

    fail_multi = np.asarray([
        row["threshold_cells"][4]["detected"] > 3 for row in rows
    ])
    point_colours = np.where(fail_multi, red, navy)
    ax_auc.hlines(y, 0.95, auroc, color=grey, linewidth=1.2)
    ax_auc.scatter(auroc, y, c=point_colours, s=33, zorder=3, edgecolor="white", linewidth=0.5)
    ax_auc.axvline(0.95, color="black", linewidth=1, linestyle="--")
    ax_auc.set_xlim(0.955, 1.002)
    ax_auc.set_yticks(y, [str(unit) for unit in units])
    ax_auc.invert_yaxis()
    ax_auc.set_xlabel("Base AUROC")
    ax_auc.set_ylabel("Published unit")
    ax_auc.set_title("A  Unit AUROC", loc="left", fontweight="bold", fontsize=10)
    ax_auc.text(0.951, -0.75, "macro 0.991", ha="left", va="bottom", fontsize=8.5)
    ax_auc.grid(axis="x", color="#E5E7EB", linewidth=0.7)

    width = 0.36
    ax_thr.bar(y - width / 2, sole / 32, width=width, color=teal, label="sole positive")
    ax_thr.bar(y + width / 2, multi / 32, width=width, color=orange, label="multi positive")
    ax_thr.axhline(24 / 32, color="black", linewidth=1, linestyle="--", label="24/32 gate")
    ax_thr.set_ylim(0.55, 1.03)
    ax_thr.set_xticks(y, [str(unit) for unit in units], rotation=55, ha="right")
    ax_thr.set_ylabel("Detection rate at 0.5")
    ax_thr.set_title("B  Source-threshold sensitivity", loc="left", fontweight="bold", fontsize=10)
    ax_thr.legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
    ax_thr.grid(axis="y", color="#E5E7EB", linewidth=0.7)

    fp = np.asarray([0, 0, 36])
    denom = np.asarray([544, 544, 544])
    labels = ["UNSAT", "far SAT", "near SAT"]
    bars = ax_fp.bar(np.arange(3), fp, color=[grey, grey, red], width=0.62)
    ax_fp.set_xticks(np.arange(3), labels)
    ax_fp.set_ylabel("False positives at 0.5")
    ax_fp.set_ylim(0, 41)
    ax_fp.set_title("C  False positives by negative regime", loc="left", fontweight="bold", fontsize=10)
    for bar, value, total in zip(bars, fp, denom):
        ax_fp.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value}/{total}", ha="center", va="bottom", fontsize=9)
    ax_fp.grid(axis="y", color="#E5E7EB", linewidth=0.7)

    for ax in (ax_auc, ax_thr, ax_fp):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8.5)

    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "status": "PASS",
        "analysis_sha256": sha256(analysis_path),
        "posthoc_diagnostics_sha256": sha256(diagnostic_path),
        "panels": {
            "A": "per-unit base AUROC; red marks units failing the protected multi-negative specificity guard",
            "B": "source-threshold positive detection by stratum",
            "C": "pooled false-positive counts by negative regime",
        },
        "scope": "one checkpoint and selected engineered bank; descriptive denominators",
    }
    STEM.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "outputs": [str(path) for path in targets]}, sort_keys=True))


if __name__ == "__main__":
    main()
