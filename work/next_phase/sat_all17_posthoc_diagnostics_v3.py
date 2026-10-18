"""Decision-taxonomy and boundary diagnostics for the protected all-17 SAT study.

This is a post hoc, zero-forward analysis.  It reads only the frozen selected
bank, the retained 17-unit native outputs, the protected analysis, and the v2
post hoc diagnostic.  It cannot alter any protected decision.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/next_phase/sat_all17_validation_v3_20261018"
OUT_JSON = STUDY / "posthoc_diagnostics_v3.json"
OUT_MD = STUDY / "POSTHOC_DIAGNOSTICS_V3.md"
sys.path.insert(0, str(STUDY))

from protocol import SOURCE_THRESHOLD  # noqa: E402
from verify_bank import truth_counts  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def assignment_text(value: int) -> str:
    return "".join("T" if bit == "1" else "F" for bit in f"{value:05b}")


def max_sensitivity_at_fp_limit(pos: np.ndarray, neg: np.ndarray, limit: int) -> dict:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    candidates = np.unique(np.concatenate([pos, neg, [np.inf, -np.inf]]))
    feasible = []
    for threshold in candidates:
        fp = int(np.count_nonzero(neg >= threshold))
        if fp <= limit:
            feasible.append((int(np.count_nonzero(pos >= threshold)), fp, float(threshold)))
    best = max(feasible, key=lambda row: (row[0], -row[1], row[2]))
    return {"positive_detected": best[0], "positive_denominator": len(pos),
            "near_false_positives": best[1], "near_negative_denominator": len(neg),
            "threshold": best[2]}


def binom_tail(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def main() -> None:
    if OUT_JSON.exists() or OUT_MD.exists():
        raise RuntimeError("v3 post hoc output already exists")
    with np.load(STUDY / "selected_bank.npz", allow_pickle=False) as z:
        bank = {key: z[key].copy() for key in z.files}
    with np.load(STUDY / "native_outputs.npz", allow_pickle=False) as z:
        native = {key: z[key].copy() for key in z.files}
    analysis = json.loads((STUDY / "analysis.json").read_text())
    v2 = json.loads((STUDY / "posthoc_diagnostics_v2.json").read_text())
    counts = np.stack([truth_counts(row) for row in bank["unique_clauses"]])
    task_target = np.where(np.any(counts == 0, axis=1), analysis["task_overall"]["sat_token"],
                           analysis["task_overall"]["unsat_token"])
    task_prediction = native["logits"].argmax(axis=1)
    task_correct = task_prediction == task_target

    per_unit = []
    auc_matrix = np.empty((17, 32), dtype=float)
    task_aggregate = {"sole_solution_misses": 0, "sole_solution_misses_task_wrong": 0,
                      "multi_solution_misses": 0, "multi_solution_misses_task_wrong": 0,
                      "satisfiable_negative_false_positives": 0,
                      "satisfiable_negative_false_positives_task_wrong": 0}
    for ui, (unit0, assignment0) in enumerate(zip(bank["unit_ids"], bank["unit_assignment_ids"])):
        unit, assignment = int(unit0), int(assignment0)
        ai = int(np.flatnonzero(bank["unique_assignment_ids"] == assignment)[0])
        ids = bank["base_unique_ids"][ai]
        score = native["pre"][:, ui]
        pos_ids = ids[:2].reshape(-1)
        pos = score[pos_ids]
        near_ids, far_ids = [], []
        for si in (3, 4):
            for uid0 in ids[si]:
                uid = int(uid0)
                solutions = np.flatnonzero(counts[uid] == 0)
                distance = min((int(solution) ^ assignment).bit_count() for solution in solutions)
                (near_ids if distance == 1 else far_ids).append(uid)
        near_ids = np.asarray(near_ids, dtype=int)
        far_ids = np.asarray(far_ids, dtype=int)
        near = score[near_ids]
        far = score[far_ids]
        unsat = score[ids[2]]
        for candidate in range(32):
            positive = counts[:, candidate] == 0
            auc_matrix[ui, candidate] = exact_auc(score[positive], score[~positive])
        order = np.argsort(-auc_matrix[ui], kind="mergesort")
        runner = int(next(candidate for candidate in order if int(candidate) != assignment))

        sole_scores = score[ids[0]]
        multi_scores = score[ids[1]]
        sole_miss = sole_scores < SOURCE_THRESHOLD
        multi_miss = multi_scores < SOURCE_THRESHOLD
        sat_negative_ids = ids[3:].reshape(-1)
        sat_negative_fp = score[sat_negative_ids] >= SOURCE_THRESHOLD
        task_counts = {
            "sole_solution_misses": int(sole_miss.sum()),
            "sole_solution_misses_task_wrong": int(np.count_nonzero(sole_miss & ~task_correct[ids[0]])),
            "multi_solution_misses": int(multi_miss.sum()),
            "multi_solution_misses_task_wrong": int(np.count_nonzero(multi_miss & ~task_correct[ids[1]])),
            "satisfiable_negative_false_positives": int(sat_negative_fp.sum()),
            "satisfiable_negative_false_positives_task_wrong": int(np.count_nonzero(sat_negative_fp & ~task_correct[sat_negative_ids])),
        }
        for key, value in task_counts.items():
            task_aggregate[key] += value

        per_unit.append({
            "unit": unit,
            "published_assignment_id": assignment,
            "published_assignment": assignment_text(assignment),
            "protected_base_auroc": float(analysis["base"]["per_unit"][ui]["base_auroc"]),
            "near_only_auroc": exact_auc(pos, near),
            "far_only_auroc": exact_auc(pos, far),
            "unsat_only_auroc": exact_auc(pos, unsat),
            "source_threshold": {
                "positive_detected": int(np.count_nonzero(pos >= SOURCE_THRESHOLD)),
                "positive_denominator": 64,
                "near_false_positives": int(np.count_nonzero(near >= SOURCE_THRESHOLD)),
                "near_negative_denominator": 32,
            },
            "max_sensitivity_at_zero_observed_near_fp": max_sensitivity_at_fp_limit(pos, near, 0),
            "max_sensitivity_at_one_observed_near_fp": max_sensitivity_at_fp_limit(pos, near, 1),
            "candidate_competition": {
                "published_full_bank_auroc": float(auc_matrix[ui, assignment]),
                "runner_up_assignment_id": runner,
                "runner_up_assignment": assignment_text(runner),
                "runner_up_full_bank_auroc": float(auc_matrix[ui, runner]),
                "runner_up_hamming_distance": int((runner ^ assignment).bit_count()),
                "bank_warning": "full 7,252-input bank was engineered around the 16 published labels; candidate populations are asymmetric",
            },
            "task_stratification": task_counts,
        })

    # Aggregate q=1 comparisons directly from the v2 cell table.
    q1 = {}
    for regime, neighbours, name in (("near", 2, "near_two_neighbours"),
                                     ("near", 1, "near_one_neighbour"),
                                     ("unsat", 0, "unsat"), ("far", 0, "far")):
        cells = [row for row in v2["negative_clause_count_cells"]
                 if row["regime"] == regime and row["violated_clauses"] == 1
                 and row["hamming_one_solutions"] == neighbours]
        q1[name] = {
            "rows": int(sum(row["rows"] for row in cells)),
            "false_positives": int(sum(row["false_positives"] for row in cells)),
            "strict_inversions": int(sum(row["strict_inversions"] for row in cells)),
            "score_mean": float(sum(row["rows"] * row["score_mean"] for row in cells) /
                                sum(row["rows"] for row in cells)),
        }

    regression = v2["orbit_grouped_five_fold_regression"]
    r2_q = regression["unit_means_plus_q_categorical"]["cv_r2_global"]
    r2_d = regression["unit_means_plus_distance_categorical"]["cv_r2_global"]
    r2_both = regression["unit_means_plus_q_and_distance_categorical"]["cv_r2_global"]
    partial = {"q_added_after_distance": float(r2_both - r2_d),
               "distance_added_after_q": float(r2_both - r2_q)}

    # Descriptive dispersion, with no chi-square p-value because cells are dependent
    # and unit is confounded with assignment-specific bank composition.
    near_fp = np.asarray([row["source_threshold"]["near_false_positives"] for row in per_unit], dtype=float)
    common_rate = float(near_fp.sum() / (32 * len(near_fp)))
    pearson_dispersion = float(np.sum((near_fp - 32 * common_rate) ** 2 /
                                     (32 * common_rate * (1 - common_rate))))

    gate = v2["gate_uncertainty"]
    positive_joint = {
        p: float(gate["probability_pass_positive_gate_x_at_least_24"][str(p)] ** 34)
        for p in (0.8, 0.9)
    }
    near_multi_rate = 23 / 272
    per_cell_fail = binom_tail(4, 16, near_multi_rate)
    negative_any_fail = 1 - (1 - per_cell_fail) ** 17

    base_near_mean = float(np.mean([row["near_only_auroc"] for row in per_unit]))
    result = {
        "status": "PASS",
        "scope": "post hoc zero-forward analysis of retained all-17 arrays; protected decisions unchanged",
        "inputs": {name: sha256(STUDY / name) for name in
                   ("selected_bank.npz", "native_outputs.npz", "analysis.json", "posthoc_diagnostics_v2.json")},
        "decision_taxonomy": {
            "threshold_free_ranking_gates": {"passed": 3, "failed": 0},
            "shared_threshold_specificity_guard": "failed by units 10 and 81 at 4/32 multi-solution negatives",
            "source_threshold_fidelity_other_failure": "positive cells fail for units 81, 261, 411 and 416",
            "warning": "the protected broad-discrimination and source-threshold fidelity decisions share the same threshold-specificity failure and are not independent",
        },
        "illustrative_conjunctive_gate_operating_characteristics": {
            "assumptions": "independent homogeneous cells; illustrative only and false for the dependent engineered bank",
            "probability_all_34_positive_cells_pass_if_each_true_sensitivity_0_8": positive_joint[0.8],
            "probability_all_34_positive_cells_pass_if_each_true_sensitivity_0_9": positive_joint[0.9],
            "near_multi_rate_used": near_multi_rate,
            "probability_one_16_case_near_half_has_at_least_4_fp": per_cell_fail,
            "probability_any_of_17_near_halves_has_at_least_4_fp": negative_any_fail,
        },
        "per_unit_boundary_diagnostics": per_unit,
        "base_near_only_auroc_mean": base_near_mean,
        "within_q1_negative_comparison": q1,
        "regression_partial_cv_r2_increments": partial,
        "regression_fold_rank_disclosure": {
            name: row["folds"] for name, row in regression.items()
            if name in ("unit_means_plus_q_categorical", "unit_means_plus_q_and_distance_categorical")
        },
        "categorical_unseen_level_policy": "global columns are retained; a level absent in a training fold has coefficient zero under least squares, so its held-out rows receive the remaining unit and covariate terms",
        "near_false_positive_dispersion": {
            "counts_by_unit": near_fp.astype(int).tolist(), "common_rate": common_rate,
            "pearson_statistic": pearson_dispersion, "reference_df": 16,
            "warning": "descriptive only; unit is confounded with assignment-specific formula composition and rows share formula orbits",
        },
        "task_error_geometry": task_aggregate,
        "transformed_mixture": {
            "base_negative_near_share": 0.5,
            "fixed_transformed_negative_near_share": 374 / 2064,
            "switch_transformed_negative_near_share": 184 / 936,
            "base_near_only_auroc_mean": base_near_mean,
            "fixed_transformed_near_auroc_mean": v2["transformed_auroc_by_regime"]["fixed"]["near"]["mean_auroc"],
            "switch_transformed_near_auroc_mean": v2["transformed_auroc_by_regime"]["switch"]["near"]["mean_auroc"],
        },
        "logical_necessity": "In this two-literal grammar, two distinct Hamming-one satisfying neighbours force every target-violated clause to be the unique clause type on the two flipped variables; one distinct violated type is therefore definitional, not an empirical finding.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Version-3 zero-forward diagnostics", "",
        "These analyses are post hoc and cannot alter the protected decisions.", "",
        "## Decision taxonomy", "", result["decision_taxonomy"]["warning"], "",
        "## Boundary ranking and separability", "",
        f"Mean base near-only AUROC: {base_near_mean:.6f}.", "",
        "| Unit | Near AUROC | TP at source | Near FP | TP at zero near FP | TP at <=1 near FP | Runner-up | Hamming |", "|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in per_unit:
        comp = row["candidate_competition"]
        lines.append(f"| {row['unit']} | {row['near_only_auroc']:.6f} | {row['source_threshold']['positive_detected']}/64 | {row['source_threshold']['near_false_positives']}/32 | {row['max_sensitivity_at_zero_observed_near_fp']['positive_detected']}/64 | {row['max_sensitivity_at_one_observed_near_fp']['positive_detected']}/64 | {comp['runner_up_assignment']} | {comp['runner_up_hamming_distance']} |")
    lines += ["", "## Within-q=1 negative comparison", "",
              "| Regime | Rows | FP | Mean score | Strict inversions |", "|:--|--:|--:|--:|--:|"]
    for name, row in q1.items():
        lines.append(f"| {name.replace('_', ' ')} | {row['rows']} | {row['false_positives']} | {row['score_mean']:.6f} | {row['strict_inversions']} |")
    lines += ["", "## Task-error geometry", "", json.dumps(task_aggregate, sort_keys=True), "",
              "## Gate operating characteristics", "", json.dumps(result["illustrative_conjunctive_gate_operating_characteristics"], sort_keys=True), ""]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))


if __name__ == "__main__":
    main()
