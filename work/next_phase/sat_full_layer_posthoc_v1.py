"""Zero-forward diagnostics for the protected full-layer SAT study.

This script does not alter the frozen v7 result. It analyses retained arrays after
the protected outcome and writes a separately labelled, exclusive JSON artifact.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/next_phase/sat_full_layer_basis_v7_20261018"
PRIOR = ROOT / "work/next_phase/sat_all17_validation_v3_20261018"
sys.path.insert(0, str(STUDY))
from protocol import (  # noqa: E402
    ASSIGNMENTS,
    COVERED_ASSIGNMENTS,
    SAT_TOKEN,
    SOURCE_THRESHOLD,
    UNCOVERED_ASSIGNMENTS,
    UNITS,
    UNSAT_TOKEN,
    semantic_counts,
)


OUTPUT = STUDY / "posthoc_diagnostics_v2.json"
SOURCE_REPORT_UNITS = np.array(
    [10, 29, 36, 41, 48, 55, 77, 81, 96, 150, 185, 189, 195, 201,
     224, 231, 245, 261, 291, 304, 317, 326, 334, 374, 380, 411, 416,
     435, 450, 482, 490, 492, 495, 499], dtype=np.int64,
)
SOURCE_RULES = {
    10: ["TFFFF", "TFTFF", "TTFFF"], 29: ["FTTFF"], 36: ["TTFFF"],
    41: ["FFFFF", "FTFFF"], 48: ["FTFFF", "FTTFF"],
    55: ["TTTFF", "TTTFT"], 77: ["FFTTF", "FFTTT", "FTTTT"],
    81: ["FTTTF", "FTTTT"], 96: ["TFTTF"], 150: ["FTFTF"],
    185: ["FFFTT"], 189: ["FTTTT"],
    195: ["TFFTF", "TFTTF", "TTFTF"], 201: ["TTFFT"],
    224: ["FFFFT"], 231: ["FTFTT", "FTTTT"],
    245: ["FTFFF", "FTFFT", "FTTFT"], 261: ["TTFTT"],
    291: ["TFTTF", "TFTTT", "TTTTT"], 304: ["FFTTF"],
    317: ["TFFFT"], 326: ["TTFTF", "TTTTF", "TTTTT"],
    334: ["FTTTF"], 374: ["TTFTT"], 380: ["TFFTT"],
    411: ["TTFFT", "TTTFT"], 416: ["TFTFF", "TTTFF"],
    435: ["TTFTF"], 450: ["FFTFF", "FFTFT", "FTTFT"],
    482: ["TTTTT"], 490: ["FTTFT"],
    492: ["TFTFF", "TFTFT", "TTTFT"], 495: ["FFFTF"],
    499: ["FFTFF"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment_name(assignment: int) -> str:
    return format(int(assignment), "05b").replace("0", "F").replace("1", "T")


def average_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    n, d = scores.shape
    order = np.argsort(scores, axis=0, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    ranks = np.empty((n, d), dtype=np.float64)
    for column in range(d):
        values = sorted_scores[:, column]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        ends = np.r_[starts[1:], n]
        ranks[order[:, column], column] = np.repeat(
            (starts + 1 + ends) / 2.0, ends - starts
        )
    return ranks


def auc_matrix(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ranks = average_ranks(scores)
    labels = np.asarray(labels, dtype=bool)
    npos = labels.sum(axis=0).astype(np.float64)
    nneg = len(labels) - npos
    if np.any(npos == 0) or np.any(nneg == 0):
        raise RuntimeError("assignment class missing")
    rank_sums = labels.astype(np.float64).T @ ranks
    return (rank_sums - npos[:, None] * (npos[:, None] + 1) / 2) / (
        npos[:, None] * nneg[:, None]
    )


def maximum_matching(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    n, m = weights.shape
    if n > m:
        raise ValueError("matching requires rows <= columns")
    cost = -weights
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int32)
    way = np.zeros(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.full(n, -1, dtype=np.int32)
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    if np.any(assignment < 0):
        raise RuntimeError("incomplete matching")
    return assignment, weights[np.arange(n), assignment]


def matching_record(
    two_sided: np.ndarray, assignments: np.ndarray, allowed_axes: np.ndarray
) -> dict:
    rows = np.asarray(assignments, dtype=np.int64)
    allowed = np.asarray(allowed_axes, dtype=np.int64)
    local_axes, scores = maximum_matching(two_sided[rows][:, allowed])
    axes = allowed[local_axes]
    return {
        "assignments": rows.tolist(),
        "assignment_names": [assignment_name(row) for row in rows],
        "axes": axes.tolist(),
        "scores": scores.tolist(),
        "mean": float(scores.mean()),
        "minimum": float(scores.min()),
        "axes_in_source_report": np.isin(axes, SOURCE_REPORT_UNITS).tolist(),
        "axes_in_singleton_set": np.isin(axes, UNITS).tolist(),
    }


def change_summary(mask, pred, changed_pred, correct, changed_correct) -> dict:
    return {
        "n": int(mask.sum()),
        "baseline_correct": int(np.count_nonzero(mask & correct)),
        "argmax_changes": int(np.count_nonzero(mask & (changed_pred != pred))),
        "correct_to_incorrect": int(np.count_nonzero(mask & correct & ~changed_correct)),
        "incorrect_to_correct": int(np.count_nonzero(mask & ~correct & changed_correct)),
    }


def changed_state(logits, post, coefficient, units, task_label):
    selected = np.asarray(units, dtype=np.int64)
    changed_logits = logits - post[:, selected] @ coefficient[selected]
    changed_pred = changed_logits.argmax(axis=1)
    return changed_pred, changed_pred == task_label


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("refusing to overwrite post hoc diagnostic")
    analysis_path = STUDY / "analysis.json"
    native_path = STUDY / "full_native_outputs.npz"
    bank_path = PRIOR / "selected_bank.npz"
    notebook_path = ROOT / "work/sat_case/upstream/interpretation.ipynb"
    analysis = json.loads(analysis_path.read_text())
    with np.load(native_path, allow_pickle=False) as z:
        post = z["post"].astype(np.float64)
        logits = z["logits"].astype(np.float64)
        coefficient = z["W_out"].astype(np.float64).T @ z["W_U"].astype(np.float64)
    with np.load(bank_path, allow_pickle=False) as z:
        formulas = z["unique_clauses"].copy()
        base_ids = np.unique(z["base_unique_ids"])

    counts = semantic_counts(formulas)
    labels = counts == 0
    sat = labels.any(axis=1)
    solution_count = labels.sum(axis=1)
    task_label = np.where(sat, SAT_TOKEN, UNSAT_TOKEN)
    pred = logits.argmax(axis=1)
    correct = pred == task_label
    if int(correct.sum()) != 7215:
        raise RuntimeError("baseline drift")

    post_auc = auc_matrix(post, labels)
    two_sided = np.maximum(post_auc, 1 - post_auc)
    base_auc = auc_matrix(post[base_ids], labels[base_ids])
    base_two_sided = np.maximum(base_auc, 1 - base_auc)
    all_axes = np.arange(512, dtype=np.int64)
    without_singletons = np.setdiff1d(all_axes, UNITS)
    without_report = np.setdiff1d(all_axes, SOURCE_REPORT_UNITS)
    restriction_sets = {
        "all_512": all_axes,
        "exclude_17_singleton_units": without_singletons,
        "exclude_all_34_source_report_units": without_report,
    }
    match_sensitivity = {}
    for bank_name, matrix in (("full_7252", two_sided), ("unique_base_2543", base_two_sided)):
        match_sensitivity[bank_name] = {}
        for restriction_name, allowed in restriction_sets.items():
            match_sensitivity[bank_name][restriction_name] = {
                "all_32": matching_record(matrix, np.arange(32), allowed),
                "uncovered_16": matching_record(matrix, UNCOVERED_ASSIGNMENTS, allowed),
            }

    protected_match = analysis["native_basis"]["all_rows"]["matching_32"]
    source_rule_rows = []
    for assignment, axis, score in zip(
        protected_match["assignments"], protected_match["axes"], protected_match["scores"]
    ):
        name = assignment_name(assignment)
        rules = SOURCE_RULES[int(axis)]
        source_rule_rows.append({
            "assignment": int(assignment),
            "assignment_name": name,
            "matched_axis": int(axis),
            "matching_auc": float(score),
            "source_rules": rules,
            "matched_assignment_in_source_rules": name in rules,
            "matched_axis_in_singleton_set": int(axis) in set(map(int, UNITS)),
        })

    active = np.flatnonzero(np.any(post != 0, axis=0))
    activity = np.mean(post > 0, axis=0)
    source_inventory = {
        "source_selection_statement": "interpretation.ipynb cell 77: units_for_sat = (MLP_max_effect > 0.1).nonzero().flatten()",
        "source_report_units": SOURCE_REPORT_UNITS.tolist(),
        "source_report_count": int(len(SOURCE_REPORT_UNITS)),
        "active_unit_count": int(len(active)),
        "active_units": active.tolist(),
        "source_report_units_all_active": bool(np.all(np.isin(SOURCE_REPORT_UNITS, active))),
        "source_report_activity_rates": {str(int(u)): float(activity[u]) for u in SOURCE_REPORT_UNITS},
        "post_matrix_rank": int(np.linalg.matrix_rank(post)),
        "all_32_matched_axes_inside_source_report": bool(
            np.all(np.isin(protected_match["axes"], SOURCE_REPORT_UNITS))
        ),
        "source_report_units_unused_by_all_32_matching": np.setdiff1d(
            SOURCE_REPORT_UNITS, protected_match["axes"]
        ).tolist(),
        "source_two_leaf_tree_uncovered_assignments": [8, 20, 29],
        "source_two_leaf_tree_uncovered_assignment_names": [assignment_name(x) for x in (8, 20, 29)],
        "source_rule_match_rows": source_rule_rows,
        "all_32_matches_agree_with_source_rule_inventory": bool(
            all(row["matched_assignment_in_source_rules"] for row in source_rule_rows)
        ),
    }

    individual_rows = []
    individual_changed_masks = {}
    for unit0, assignment0 in zip(UNITS, ASSIGNMENTS):
        unit = int(unit0)
        assignment = int(assignment0)
        changed_pred, changed_correct = changed_state(
            logits, post, coefficient, [unit], task_label
        )
        individual_changed_masks[unit] = changed_pred != pred
        observed = post[:, unit] >= SOURCE_THRESHOLD
        expected = labels[:, assignment]
        masks = {
            "sole_positive": expected & (solution_count == 1),
            "multi_positive": expected & (solution_count >= 2),
            "satisfiable_negative": ~expected & sat,
            "unsat": ~sat,
            "threshold_true_positive": expected & observed,
            "threshold_false_negative": expected & ~observed,
            "threshold_false_positive": ~expected & observed,
            "threshold_true_negative": ~expected & ~observed,
        }
        individual_rows.append({
            "unit": unit,
            "assignment": assignment,
            "assignment_name": assignment_name(assignment),
            "direct_sat_minus_unsat_coefficient": float(
                coefficient[unit, SAT_TOKEN] - coefficient[unit, UNSAT_TOKEN]
            ),
            "by_semantic_and_threshold_category": {
                name: change_summary(mask, pred, changed_pred, correct, changed_correct)
                for name, mask in masks.items()
            },
        })

    joint_pred, joint_correct = changed_state(logits, post, coefficient, UNITS, task_label)
    report_joint_pred, report_joint_correct = changed_state(
        logits, post, coefficient, SOURCE_REPORT_UNITS, task_label
    )
    covered_any = labels[:, COVERED_ASSIGNMENTS].any(axis=1)
    uncovered_any = labels[:, UNCOVERED_ASSIGNMENTS].any(axis=1)
    joint_masks = {
        "unsat": ~sat,
        "covered_only": covered_any & ~uncovered_any,
        "uncovered_only": uncovered_any & ~covered_any,
        "both_covered_and_uncovered": covered_any & uncovered_any,
        "has_any_covered_solution": covered_any,
        "has_any_uncovered_solution": uncovered_any,
    }
    sum_individual_changes = int(sum(mask.sum() for mask in individual_changed_masks.values()))
    union_individual = np.logical_or.reduce(list(individual_changed_masks.values()))
    readout_structure = {
        "individual_rows": individual_rows,
        "sum_of_individual_argmax_change_counts": sum_individual_changes,
        "unique_rows_changed_by_any_individual_ablation": int(union_individual.sum()),
        "joint_argmax_changes": int(np.count_nonzero(joint_pred != pred)),
        "joint_minus_sum_individual_change_counts": int(
            np.count_nonzero(joint_pred != pred) - sum_individual_changes
        ),
        "joint_by_solution_coverage": {
            name: change_summary(mask, pred, joint_pred, correct, joint_correct)
            for name, mask in joint_masks.items()
        },
        "all_34_source_report_joint": {
            "overall": change_summary(
                np.ones(len(pred), dtype=bool), pred, report_joint_pred, correct, report_joint_correct
            ),
            "by_solution_coverage": {
                name: change_summary(
                    mask, pred, report_joint_pred, correct, report_joint_correct
                )
                for name, mask in joint_masks.items()
            },
        },
    }

    pair_overlap = {}
    for pair in ((55, 416), (81, 334)):
        first_pred, first_correct = changed_state(logits, post, coefficient, [pair[0]], task_label)
        second_pred, second_correct = changed_state(logits, post, coefficient, [pair[1]], task_label)
        pair_pred, pair_correct = changed_state(logits, post, coefficient, pair, task_label)
        first = first_pred != pred
        second = second_pred != pred
        pair_overlap[f"{pair[0]}_{pair[1]}"] = {
            "first_argmax_changes": int(first.sum()),
            "second_argmax_changes": int(second.sum()),
            "intersection_of_individual_change_rows": int(np.count_nonzero(first & second)),
            "union_of_individual_change_rows": int(np.count_nonzero(first | second)),
            "pair": change_summary(np.ones(len(pred), dtype=bool), pred, pair_pred, correct, pair_correct),
        }
    readout_structure["pair_overlap"] = pair_overlap

    zero_rows = analysis["all_unit_zero_ablation"]
    zero_changes = np.array([row["argmax_changes"] for row in zero_rows], dtype=np.int64)
    zero_reference = {
        "units_with_no_argmax_change": int(np.count_nonzero(zero_changes == 0)),
        "units_with_any_argmax_change": int(np.count_nonzero(zero_changes)),
        "unit_ids_with_any_argmax_change": np.flatnonzero(zero_changes).tolist(),
        "changed_unit_ids_equal_source_report_units": bool(
            np.array_equal(np.flatnonzero(zero_changes), SOURCE_REPORT_UNITS)
        ),
        "change_count_quantiles": {
            str(q): float(np.quantile(zero_changes, q, method="inverted_cdf"))
            for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
        },
        "top20": sorted(
            ({"unit": int(row["unit"]), "argmax_changes": int(row["argmax_changes"]),
              "in_source_report": int(row["unit"]) in set(map(int, SOURCE_REPORT_UNITS)),
              "in_singleton_set": int(row["unit"]) in set(map(int, UNITS))}
             for row in zero_rows),
            key=lambda row: (-row["argmax_changes"], row["unit"]),
        )[:20],
    }

    all_rows = analysis["basis_comparisons"]["all_rows"]
    omitted_descriptives = {
        "native_axis_counts": analysis["native_basis"]["all_rows"]["axis_counts"],
        "axis_count_comparisons": {
            key: all_rows[key] for key in (
                "axis_count_0.9", "axis_count_0.95", "axis_count_0.98", "axis_count_0.99"
            )
        },
        "native_reusable_axis_minimum": analysis["native_basis"]["all_rows"]["reusable_axis_min_32"],
        "reusable_axis_minimum_comparison": all_rows["reusable_axis_min_32"],
        "native_reusable_axis_mean": analysis["native_basis"]["all_rows"]["reusable_axis_mean_32"],
        "reusable_axis_mean_comparison": all_rows["reusable_axis_mean_32"],
        "zero_ablation_512_reference": zero_reference,
        "verbatim_licensed_conclusion": analysis["primary_inference"]["licensed_conclusion"],
        "raw_finite_reference_tail_areas": analysis["primary_inference"]["p_values"],
        "holm_adjusted_tail_areas_for_two_equal_raw_values": {
            "matching_32": 0.03125,
            "matching_uncovered16": 0.03125,
        },
    }

    result = {
        "status": "PASS",
        "scope": "post hoc zero-forward diagnostics on retained protected-study arrays; does not change the frozen primary inference",
        "inputs": {
            "analysis_sha256": sha256(analysis_path),
            "native_outputs_sha256": sha256(native_path),
            "selected_bank_sha256": sha256(bank_path),
            "source_notebook_sha256": sha256(notebook_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "source_nomination_inventory": source_inventory,
        "matching_sensitivity": match_sensitivity,
        "omitted_prespecified_descriptives": omitted_descriptives,
        "readout_structure": readout_structure,
        "limitations": [
            "All results are conditional on one checkpoint and the engineered bank.",
            "Restriction and base-only matchings were designed after the protected result.",
            "Counts across units reuse formula rows and are not independent trials.",
            "Zero ablation is exact at the final affine cut but can be off the natural activation manifold.",
        ],
    }
    with OUTPUT.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
