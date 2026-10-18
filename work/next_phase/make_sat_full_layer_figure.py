"""Render the protected full-layer basis and readout result from retained JSON."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/next_phase/sat_full_layer_basis_v7_20261018"
STEM = ROOT / "outputs/causal-resolution-v3/paper/figures/sat-full-layer-basis-v3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    targets = [STEM.with_suffix(suffix) for suffix in (".pdf", ".png", ".svg", ".json")]
    if any(path.exists() for path in targets):
        raise RuntimeError("refusing to overwrite full-layer figure")
    analysis_path = STUDY / "analysis.json"
    result = json.loads(analysis_path.read_text())
    comparison = result["basis_comparisons"]

    assignment_rows = [
        ("all 32", comparison["all_rows"]["matching_32"]),
        ("uncovered 16", comparison["all_rows"]["matching_uncovered16"]),
        ("exclusive 16", comparison["exclusive_uncovered16"]),
    ]
    control_keys = [
        ("clause\npresence", "clause_presence_matching_mean"),
        ("SAT", "sat_label_best_axis_auc"),
        ("one\nsolution", "sole_solution_label_best_axis_auc"),
        ("multiple\nsolutions", "multi_solution_label_best_axis_auc"),
    ]
    controls = result["control_label_comparisons"]
    readout = result["readout_published"]
    units = [row["unit"] for row in readout]
    accuracy_drop = [100 * (result["task"]["baseline_accuracy"] - row["zero_ablation"]["accuracy"]) for row in readout]
    resample_change = [100 * row["resample_summary"]["argmax_change_rate_mean"] for row in readout]

    navy, blue, teal, orange, red, grey = "#253B6E", "#4776B4", "#0F766E", "#D97706", "#B91C1C", "#9CA3AF"
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.55), constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    y = np.arange(len(assignment_rows))
    for index, (_, row) in enumerate(assignment_rows):
        ax_a.hlines(index, row["rotation_min"], row["rotation_max"], color=grey, linewidth=4, alpha=0.55)
        ax_a.scatter(row["rotation_mean"], index, color=blue, s=30, marker="|", linewidths=2.2, zorder=3)
        ax_a.scatter(row["native"], index, color=navy, s=38, zorder=4)
        ax_a.text(row["native"] + 0.0007, index, f'{row["native"]:.3f}', va="center", fontsize=7.5)
    ax_a.set_yticks(y, [row[0] for row in assignment_rows])
    ax_a.invert_yaxis(); ax_a.set_xlim(0.938, 0.991)
    ax_a.set_xlabel("One-to-one matching mean AUROC")
    ax_a.set_title("A  Assignment axes beat Haar", loc="left", fontweight="bold", fontsize=10)
    x = np.arange(len(control_keys))
    for index, (_, key) in enumerate(control_keys):
        row = controls[key]
        ax_b.vlines(index, row["rotation_min"], row["rotation_max"], color=grey, linewidth=5, alpha=0.55)
        ax_b.scatter(index, row["rotation_mean"], color=blue, s=35, marker="_", linewidths=2.2, zorder=3)
        ax_b.scatter(index, row["native"], color=red, s=34, zorder=4)
    ax_b.set_xticks(x, [label for label, _ in control_keys])
    ax_b.set_ylim(0.50, 0.95); ax_b.set_ylabel("Matching mean or best-axis AUROC")
    ax_b.set_title("B  Controls favour dense rotations", loc="left", fontweight="bold", fontsize=10)

    x = np.arange(len(units))
    ax_c.bar(x, accuracy_drop, color=teal, alpha=0.78, width=0.72, label="zero-ablation accuracy drop")
    ax_c.set_xticks(x, [str(unit) for unit in units], rotation=65, ha="right")
    ax_c.set_ylabel("Accuracy drop (percentage points)", color=teal)
    ax_c.tick_params(axis="y", labelcolor=teal)
    twin = ax_c.twinx()
    twin.plot(x, resample_change, color=orange, marker="o", markersize=2.8, linewidth=1.1,
              label="within-stratum resample argmax changes")
    twin.set_ylabel("Mean argmax changes (%)", color=orange)
    twin.tick_params(axis="y", labelcolor=orange)
    ax_c.set_title("C  Readout effects depend on intervention", loc="left", fontweight="bold", fontsize=10)
    ax_c.text(0.02, 0.98, "joint zeroing 17 units:\n99.49% → 61.33% accuracy",
              transform=ax_c.transAxes, va="top", fontsize=7.4,
              bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "pad": 3})

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="x" if ax is ax_a else "y", color="#E5E7EB", linewidth=0.6)
        ax.tick_params(labelsize=8)
    twin.spines["top"].set_visible(False); twin.tick_params(labelsize=8)

    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "PASS", "analysis_sha256": sha256(analysis_path),
        "figure_scope": "protected v7 analysis; one checkpoint and engineered bank",
        "primary_conclusion_key": result["primary_inference"]["conclusion_key"],
        "primary_p_values": result["primary_inference"]["p_values"],
    }
    STEM.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "outputs": [str(path) for path in targets]}, sort_keys=True))


if __name__ == "__main__":
    main()
