"""Render the source-report assignment census and exact-readout audit."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/next_phase/sat_full_layer_basis_v7_20261018"
PRIOR = ROOT / "work/next_phase/sat_all17_validation_v3_20261018"
STEM = ROOT / "figures/sat-sparse-report-v49"
sys.path.insert(0, str(ROOT / "work/next_phase"))
from sat_full_layer_posthoc_v1 import (  # noqa: E402
    SOURCE_REPORT_UNITS,
    SOURCE_RULES,
    UNITS,
    assignment_name,
    auc_matrix,
    semantic_counts,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    targets = [STEM.with_suffix(suffix) for suffix in (".pdf", ".png", ".svg", ".json")]
    if any(path.exists() for path in targets):
        raise RuntimeError("refusing to overwrite sparse-report figure")
    analysis_path = STUDY / "analysis.json"
    posthoc_path = STUDY / "posthoc_diagnostics_v2.json"
    analysis = json.loads(analysis_path.read_text())
    posthoc = json.loads(posthoc_path.read_text())
    with np.load(STUDY / "full_native_outputs.npz", allow_pickle=False) as z:
        post = z["post"].astype(np.float64)
    with np.load(PRIOR / "selected_bank.npz", allow_pickle=False) as z:
        formulas = z["unique_clauses"].copy()
    labels = semantic_counts(formulas) == 0
    positive_auc = auc_matrix(post[:, SOURCE_REPORT_UNITS], labels).T

    protected = analysis["native_basis"]["all_rows"]["matching_32"]
    matched_axes = list(map(int, protected["axes"]))
    row_units = matched_axes + [int(u) for u in SOURCE_REPORT_UNITS if int(u) not in set(matched_axes)]
    row_lookup = {unit: index for index, unit in enumerate(row_units)}
    column_names = [assignment_name(index) for index in range(32)]
    matrix = positive_auc[[int(np.flatnonzero(SOURCE_REPORT_UNITS == unit)[0]) for unit in row_units]]

    navy, blue, teal, orange, grey, pale = "#253B6E", "#4776B4", "#0F766E", "#D97706", "#9CA3AF", "#F3F4F6"
    fig = plt.figure(figsize=(10.6, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.65, 1.0], width_ratios=[1.15, 1.0],
                            wspace=0.22)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    image = ax_a.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0, interpolation="nearest")
    ax_a.set_xticks(np.arange(32), column_names, rotation=70, ha="right", fontsize=6.4)
    ax_a.set_yticks(np.arange(34), [str(unit) for unit in row_units], fontsize=6.8)
    ax_a.set_xlabel("Satisfying assignment label")
    ax_a.set_ylabel("Source-nominated final-MLP unit")
    ax_a.set_title("A  The 34-unit source report spans all 32 assignment rules", loc="left", fontweight="bold", fontsize=10)
    for unit, rules in SOURCE_RULES.items():
        row = row_lookup[int(unit)]
        for rule in rules:
            col = column_names.index(rule)
            ax_a.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96,
                                     fill=False, edgecolor=orange, linewidth=0.7))
    for assignment, axis in zip(protected["assignments"], protected["axes"]):
        ax_a.plot(int(assignment), row_lookup[int(axis)], marker="o", markersize=2.5,
                  markerfacecolor="none", markeredgecolor="white", markeredgewidth=0.75)
    colourbar = fig.colorbar(image, ax=ax_a, shrink=0.75, pad=0.01)
    colourbar.set_label("Positive-direction AUROC", fontsize=8)
    colourbar.ax.tick_params(labelsize=7)
    ax_a.text(1.005, 0.02, "orange = source rule\nwhite circle = protected match",
              transform=ax_a.transAxes, fontsize=6.8, va="bottom")

    sensitivity = posthoc["matching_sensitivity"]
    restrictions = ["all_512", "exclude_17_singleton_units", "exclude_all_34_source_report_units"]
    labels_short = ["all 512", "exclude\nsingleton 17", "exclude\nsource 34"]
    x = np.arange(len(restrictions))
    full = [sensitivity["full_7252"][key]["uncovered_16"]["mean"] for key in restrictions]
    base = [sensitivity["unique_base_2543"][key]["uncovered_16"]["mean"] for key in restrictions]
    ax_b.plot(x, full, color=navy, marker="o", linewidth=1.4, label="full linked bank")
    ax_b.plot(x, base, color=blue, marker="s", linewidth=1.2, label="unique base formulas")
    ax_b.axhline(analysis["basis_comparisons"]["all_rows"]["matching_uncovered16"]["rotation_max"],
                 color=grey, linestyle="--", linewidth=1, label="best Haar reference")
    ax_b.axhline(0.5, color="#D1D5DB", linestyle=":", linewidth=0.9)
    ax_b.set_xticks(x, labels_short)
    ax_b.set_ylim(0.48, 1.005)
    ax_b.set_ylabel("Uncovered-16\nmatching mean AUROC", labelpad=8)
    ax_b.set_title("B  Removing source units removes alignment", loc="left", fontweight="bold", fontsize=9)
    ax_b.legend(frameon=False, fontsize=8, loc="lower left")
    ax_b.grid(axis="y", color="#E5E7EB", linewidth=0.6)

    zero_by_unit = {int(row["unit"]): int(row["argmax_changes"]) for row in analysis["all_unit_zero_ablation"]}
    bar_units = list(map(int, SOURCE_REPORT_UNITS))
    values = [zero_by_unit[unit] for unit in bar_units]
    colours = [teal if unit in set(map(int, UNITS)) else grey for unit in bar_units]
    ax_c.bar(np.arange(34), values, color=colours, width=0.78)
    ax_c.set_ylim(0, 215)  # Keep the joint-effect annotation above every bar.
    ax_c.set_xticks([])
    ax_c.set_xlabel("Source-report units in increasing unit-ID order")
    ax_c.set_ylabel("Argmax changes\nunder unit zeroing", labelpad=7)
    ax_c.set_title("C  Only the 34 report units change argmax", loc="left", fontweight="bold", fontsize=9)
    all34 = posthoc["readout_structure"]["all_34_source_report_joint"]["overall"]
    all34_accuracy = (all34["baseline_correct"] - all34["correct_to_incorrect"] + all34["incorrect_to_correct"]) / all34["n"]
    ax_c.text(0.02, 0.97,
              "joint zeroing\npublished 17: 99.49% → 61.33%\nsource 34: 99.49% → " + f"{100*all34_accuracy:.2f}%",
              transform=ax_c.transAxes, va="top", fontsize=8,
              bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "pad": 3})
    ax_c.legend(handles=[Patch(facecolor=teal, label="published singleton unit"),
                         Patch(facecolor=grey, label="other source-report unit")],
                frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax_c.grid(axis="y", color="#E5E7EB", linewidth=0.6)

    for ax in (ax_a, ax_b, ax_c):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "PASS",
        "analysis_sha256": sha256(analysis_path),
        "posthoc_sha256": sha256(posthoc_path),
        "native_outputs_sha256": sha256(STUDY / "full_native_outputs.npz"),
        "selected_bank_sha256": sha256(PRIOR / "selected_bank.npz"),
        "figure_scope": "protected full-layer result plus explicitly post hoc selection and readout diagnostics",
    }
    STEM.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "outputs": [str(path) for path in targets]}, sort_keys=True))


if __name__ == "__main__":
    main()
