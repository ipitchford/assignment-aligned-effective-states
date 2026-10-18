"""Create the revised all-17 SAT label-audit figure from retained arrays."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/next_phase/sat_all17_validation_v3_20261018"
FIGURES = ROOT / "outputs/causal-resolution-v3/paper/figures"
STEM = FIGURES / "sat-all17-label-audit-v3"
sys.path.insert(0, str(STUDY))

from protocol import SOURCE_THRESHOLD  # noqa: E402
from verify_bank import truth_counts  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    targets = [STEM.with_suffix(suffix) for suffix in (".pdf", ".png", ".svg", ".json")]
    if any(path.exists() for path in targets):
        raise RuntimeError("refusing to overwrite an existing version-3 figure")
    analysis_path = STUDY / "analysis.json"
    diagnostic_path = STUDY / "posthoc_diagnostics_v3.json"
    bank_path = STUDY / "selected_bank.npz"
    native_path = STUDY / "native_outputs.npz"
    analysis = json.loads(analysis_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    with np.load(bank_path, allow_pickle=False) as z:
        bank = {key: z[key].copy() for key in z.files}
    with np.load(native_path, allow_pickle=False) as z:
        native = {key: z[key].copy() for key in z.files}
    counts = np.stack([truth_counts(row) for row in bank["unique_clauses"]])

    rows = analysis["base"]["per_unit"]
    boundary = diagnostic["per_unit_boundary_diagnostics"]
    units = np.asarray([row["unit"] for row in rows])
    protected_auc = np.asarray([row["base_auroc"] for row in rows])
    near_auc = np.asarray([row["near_only_auroc"] for row in boundary])
    source_tp = np.asarray([row["source_threshold"]["positive_detected"] for row in boundary])
    zero_fp_tp = np.asarray([row["max_sensitivity_at_zero_observed_near_fp"]["positive_detected"] for row in boundary])
    one_fp_tp = np.asarray([row["max_sensitivity_at_one_observed_near_fp"]["positive_detected"] for row in boundary])

    strata = {name: [] for name in ("sole +", "multi +", "near -", "far -", "UNSAT -")}
    q1 = {name: [] for name in ("near, 2 nbrs", "near, 1 nbr", "UNSAT", "far")}
    for ui, assignment0 in enumerate(bank["unit_assignment_ids"]):
        assignment = int(assignment0)
        ai = int(np.flatnonzero(bank["unique_assignment_ids"] == assignment)[0])
        ids = bank["base_unique_ids"][ai]
        score = native["pre"][:, ui]
        strata["sole +"].extend(score[ids[0]])
        strata["multi +"].extend(score[ids[1]])
        strata["UNSAT -"].extend(score[ids[2]])
        for si in (2, 3, 4):
            for uid0 in ids[si]:
                uid = int(uid0)
                violated = int(counts[uid, assignment])
                solutions = np.flatnonzero(counts[uid] == 0)
                if si == 2:
                    regime, neighbours = "unsat", 0
                else:
                    distances = np.asarray([(int(solution) ^ assignment).bit_count() for solution in solutions])
                    regime = "near" if distances.min() == 1 else "far"
                    neighbours = int(np.count_nonzero(distances == 1))
                    strata["near -" if regime == "near" else "far -"].append(score[uid])
                if violated == 1:
                    if regime == "near" and neighbours == 2:
                        q1["near, 2 nbrs"].append(score[uid])
                    elif regime == "near" and neighbours == 1:
                        q1["near, 1 nbr"].append(score[uid])
                    elif regime == "unsat":
                        q1["UNSAT"].append(score[uid])
                    elif regime == "far":
                        q1["far"].append(score[uid])

    navy, blue, teal, orange, red, grey = "#253B6E", "#4776B4", "#0F766E", "#D97706", "#B91C1C", "#9CA3AF"
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.6), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    y = np.arange(len(units))

    ax_a.hlines(y, near_auc, protected_auc, color="#D1D5DB", linewidth=1.2)
    ax_a.scatter(protected_auc, y, color=navy, s=24, label="all negatives")
    ax_a.scatter(near_auc, y, color=red, marker="s", s=22, label="near negatives")
    ax_a.set_yticks(y, [str(unit) for unit in units])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.87, 1.005)
    ax_a.set_xlabel("AUROC")
    ax_a.set_ylabel("Published unit")
    ax_a.set_title("A  Ranking weakens at the boundary", loc="left", fontweight="bold", fontsize=10)
    ax_a.legend(frameon=False, fontsize=7.5, loc="lower right")

    x = np.arange(len(units))
    ax_b.plot(x, source_tp / 64, color=orange, marker="o", markersize=3.2, linewidth=1, label="source threshold 0.5")
    ax_b.plot(x, zero_fp_tp / 64, color=teal, marker="s", markersize=3.0, linewidth=1, label="zero observed near FP")
    ax_b.plot(x, one_fp_tp / 64, color=blue, marker="^", markersize=3.0, linewidth=1, label="at most one near FP")
    ax_b.set_xticks(x, [str(unit) for unit in units], rotation=55, ha="right")
    ax_b.set_ylim(0, 1.04)
    ax_b.set_ylabel("Positive sensitivity")
    ax_b.set_title("B  Threshold placement versus overlap", loc="left", fontweight="bold", fontsize=10)
    ax_b.legend(frameon=False, fontsize=7.2, loc="lower left")

    strata_values = [np.asarray(strata[name], dtype=float) for name in strata]
    boxes = ax_c.boxplot(strata_values, labels=list(strata), showfliers=False, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 1})
    for box, colour in zip(boxes["boxes"], [teal, teal, red, grey, grey]):
        box.set_facecolor(colour); box.set_alpha(0.68)
    ax_c.axhline(SOURCE_THRESHOLD, color="black", linestyle="--", linewidth=1, label="source threshold")
    ax_c.set_ylabel("Preactivation")
    ax_c.set_title("C  Base-bank score distributions", loc="left", fontweight="bold", fontsize=10)
    ax_c.tick_params(axis="x", rotation=18)
    ax_c.legend(frameon=False, fontsize=7.5, loc="lower left")

    q1_values = [np.asarray(q1[name], dtype=float) for name in q1]
    boxes = ax_d.boxplot(q1_values, labels=list(q1), showfliers=False, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 1})
    for box, colour in zip(boxes["boxes"], [red, orange, grey, grey]):
        box.set_facecolor(colour); box.set_alpha(0.68)
    ax_d.axhline(SOURCE_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax_d.set_ylabel("Preactivation")
    ax_d.set_title("D  Negative scores at violated-clause count one", loc="left", fontweight="bold", fontsize=10)
    ax_d.tick_params(axis="x", rotation=18)
    for index, name in enumerate(q1, start=1):
        ax_d.text(index, ax_d.get_ylim()[0], f"n={len(q1[name])}", ha="center", va="bottom", fontsize=7)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y" if ax in (ax_c, ax_d) else "x", color="#E5E7EB", linewidth=0.6)
        ax.tick_params(labelsize=8)

    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "status": "PASS",
        "analysis_sha256": sha256(analysis_path),
        "posthoc_diagnostics_v3_sha256": sha256(diagnostic_path),
        "selected_bank_sha256": sha256(bank_path),
        "native_outputs_sha256": sha256(native_path),
        "panel_denominators": {"base_strata": {key: len(value) for key, value in strata.items()},
                               "within_q1": {key: len(value) for key, value in q1.items()}},
        "scope": "post hoc zero-forward visualisation of one checkpoint and engineered dependent cohorts",
    }
    STEM.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "outputs": [str(path) for path in targets]}, sort_keys=True))


if __name__ == "__main__":
    main()
