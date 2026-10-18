"""Extended zero-forward diagnostics for the protected all-17 SAT study.

The analyses are post hoc and cannot alter the protected decisions.  They test
candidate-label competition, clause-count structure, transformed-bank mixture,
and the operating characteristics of the study-defined gates.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/next_phase/sat_all17_validation_v3_20261018"
OUT_JSON = STUDY / "posthoc_diagnostics_v2.json"
OUT_MD = STUDY / "POSTHOC_DIAGNOSTICS_V2.md"
sys.path.insert(0, str(STUDY))

from pilot_common import CLAUSES  # noqa: E402
from protocol import SOURCE_THRESHOLD  # noqa: E402
from verify_bank import truth_counts  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    values = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
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
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def literal_true(literal: int, assignment: int) -> bool:
    value = bool((assignment >> (4 - (literal % 5))) & 1)
    return not value if literal >= 5 else value


def violated_clause_types(formula: np.ndarray, assignment: int) -> tuple[int, ...]:
    return tuple(
        int(clause_id)
        for clause_id in formula
        if not any(literal_true(int(literal), assignment) for literal in CLAUSES[int(clause_id)])
    )


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def binom_tail(k: int, n: int, p: float) -> float:
    return 1.0 - binom_cdf(k - 1, n, p)


def bisect_monotone(fn, target: float, increasing: bool) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = fn(mid)
        if (value < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if x == 0 else bisect_monotone(lambda p: binom_tail(x, n, p), alpha / 2, True)
    upper = 1.0 if x == n else bisect_monotone(lambda p: binom_cdf(x, n, p), alpha / 2, False)
    return [lower, upper]


def categorical(values: np.ndarray, categories: list[int]) -> list[np.ndarray]:
    return [(values == category).astype(float) for category in categories[1:]]


def unit_design(unit_index: np.ndarray, n_units: int = 17) -> list[np.ndarray]:
    return [(unit_index == unit).astype(float) for unit in range(n_units)]


def cross_validated_model(
    y: np.ndarray,
    folds: np.ndarray,
    make_columns,
) -> dict:
    prediction = np.full(len(y), np.nan)
    fold_rows = []
    for fold in sorted(set(map(int, folds))):
        train, test = folds != fold, folds == fold
        x_train = np.column_stack(make_columns(train))
        x_test = np.column_stack(make_columns(test))
        beta, *_ = np.linalg.lstsq(x_train, y[train], rcond=None)
        prediction[test] = x_test @ beta
        fold_rows.append({"fold": fold, "train": int(train.sum()), "test": int(test.sum()), "rank": int(np.linalg.matrix_rank(x_train))})
    residual = y - prediction
    centred = y - y.mean()
    return {
        "rows": int(len(y)),
        "cv_r2_global": float(1 - (residual @ residual) / (centred @ centred)),
        "cv_rmse": float(np.sqrt(np.mean(residual**2))),
        "folds": fold_rows,
    }


def main() -> None:
    if OUT_JSON.exists() or OUT_MD.exists():
        raise RuntimeError("extended post hoc output already exists")
    with np.load(STUDY / "selected_bank.npz", allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in z.files}
    with np.load(STUDY / "native_outputs.npz", allow_pickle=False) as z:
        native = {k: z[k].copy() for k in z.files}
    analysis = json.loads((STUDY / "analysis.json").read_text())
    counts = np.stack([truth_counts(row) for row in bank["unique_clauses"]])

    # Candidate-label competition on every deduplicated native input.
    label_competition = []
    auc_matrix = np.empty((17, 32), dtype=float)
    for ui, (unit0, published0) in enumerate(zip(bank["unit_ids"], bank["unit_assignment_ids"])):
        score = native["pre"][:, ui]
        for assignment in range(32):
            positive = counts[:, assignment] == 0
            auc_matrix[ui, assignment] = exact_auc(score[positive], score[~positive])
        published = int(published0)
        order = np.argsort(-auc_matrix[ui], kind="mergesort")
        rank = int(np.flatnonzero(order == published)[0]) + 1
        runner_up = float(np.max(np.delete(auc_matrix[ui], published)))
        label_competition.append({
            "unit": int(unit0), "published_assignment_id": published,
            "published_auroc": float(auc_matrix[ui, published]),
            "best_assignment_id": int(order[0]), "best_auroc": float(auc_matrix[ui, order[0]]),
            "published_rank_of_32": rank,
            "published_margin_over_best_alternative": float(auc_matrix[ui, published] - runner_up),
        })
    unit_rank_for_assignment = []
    for ui, row in enumerate(label_competition):
        assignment = row["published_assignment_id"]
        order = np.argsort(-auc_matrix[:, assignment], kind="mergesort")
        unit_rank_for_assignment.append({
            "unit": row["unit"], "assignment_id": assignment,
            "rank_among_17_saved_units": int(np.flatnonzero(order == ui)[0]) + 1,
            "best_saved_unit": int(bank["unit_ids"][order[0]]),
        })

    # Base negative events, including clause multiplicity and neighbour structure.
    negative_events = []
    positive_scores = {}
    for ui, (unit0, assignment0) in enumerate(zip(bank["unit_ids"], bank["unit_assignment_ids"])):
        unit, assignment = int(unit0), int(assignment0)
        ai = int(np.flatnonzero(bank["unique_assignment_ids"] == assignment)[0])
        ids = bank["base_unique_ids"][ai]
        score = native["pre"][ids, ui]
        positive_scores[unit] = score[:2].reshape(-1)
        for si in (2, 3, 4):
            for qi, uid0 in enumerate(ids[si]):
                uid = int(uid0)
                solutions = np.flatnonzero(counts[uid] == 0)
                if si == 2:
                    regime, distance, neighbours = "unsat", None, 0
                else:
                    distances = np.asarray([(int(solution) ^ assignment).bit_count() for solution in solutions])
                    distance = int(distances.min())
                    regime = "near" if distance == 1 else "far"
                    neighbours = int(np.count_nonzero(distances == 1))
                violated = violated_clause_types(bank["unique_clauses"][uid], assignment)
                value = float(score[si, qi])
                negative_events.append({
                    "unit_index": ui, "unit": unit, "assignment": assignment,
                    "uid": uid, "orbit": bytes(bank["orbit_ids"][uid]).decode(),
                    "regime": regime, "distance": distance, "hamming_one_solutions": neighbours,
                    "violated_clauses": len(violated), "distinct_violated_clause_types": len(set(violated)),
                    "score": value, "false_positive": bool(value >= SOURCE_THRESHOLD),
                    "strict_inversions": int(np.count_nonzero(value > positive_scores[unit])),
                })

    grouped = defaultdict(lambda: {"rows": 0, "false_positives": 0, "strict_inversions": 0, "scores": []})
    for event in negative_events:
        key = (event["regime"], event["violated_clauses"], event["distinct_violated_clause_types"], event["hamming_one_solutions"])
        cell = grouped[key]
        cell["rows"] += 1
        cell["false_positives"] += int(event["false_positive"])
        cell["strict_inversions"] += event["strict_inversions"]
        cell["scores"].append(event["score"])
    grouped_rows = []
    for key, cell in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        scores = np.asarray(cell.pop("scores"), dtype=float)
        grouped_rows.append({
            "regime": key[0], "violated_clauses": key[1],
            "distinct_violated_clause_types": key[2], "hamming_one_solutions": key[3],
            **cell, "score_mean": float(scores.mean()), "score_median": float(np.median(scores)),
            "score_q10_q90": np.quantile(scores, [0.1, 0.9]).tolist(),
        })
    near_two = [e for e in negative_events if e["regime"] == "near" and e["hamming_one_solutions"] >= 2]
    near_one = [e for e in negative_events if e["regime"] == "near" and e["hamming_one_solutions"] == 1]

    # Orbit-grouped five-fold regression comparisons on satisfiable negatives only.
    sat_events = [e for e in negative_events if e["regime"] != "unsat"]
    y = np.asarray([e["score"] for e in sat_events], dtype=float)
    unit_index = np.asarray([e["unit_index"] for e in sat_events], dtype=int)
    q = np.asarray([e["violated_clauses"] for e in sat_events], dtype=int)
    distinct = np.asarray([e["distinct_violated_clause_types"] for e in sat_events], dtype=int)
    distance = np.asarray([e["distance"] for e in sat_events], dtype=int)
    folds = np.asarray([int(e["orbit"][:8], 16) % 5 for e in sat_events], dtype=int)
    q_categories = sorted(set(map(int, q)))
    distinct_categories = sorted(set(map(int, distinct)))
    distance_categories = sorted(set(map(int, distance)))
    models = {
        "unit_means": lambda mask: unit_design(unit_index[mask]),
        "unit_means_plus_q_categorical": lambda mask: unit_design(unit_index[mask]) + categorical(q[mask], q_categories),
        "unit_means_plus_distance_categorical": lambda mask: unit_design(unit_index[mask]) + categorical(distance[mask], distance_categories),
        "unit_means_plus_distinct_type_categorical": lambda mask: unit_design(unit_index[mask]) + categorical(distinct[mask], distinct_categories),
        "unit_means_plus_q_and_distance_categorical": lambda mask: unit_design(unit_index[mask]) + categorical(q[mask], q_categories) + categorical(distance[mask], distance_categories),
        "unit_specific_linear_q": lambda mask: unit_design(unit_index[mask]) + [
            (unit_index[mask] == unit).astype(float) * q[mask] for unit in range(17)
        ],
    }
    regression_cv = {name: cross_validated_model(y, folds, maker) for name, maker in models.items()}

    # Maximum threshold with zero observed false positives and retained sensitivity.
    zero_fp = []
    for ui, unit0 in enumerate(bank["unit_ids"]):
        unit = int(unit0)
        neg = np.asarray([e["score"] for e in negative_events if e["unit"] == unit])
        pos = positive_scores[unit]
        threshold = float(np.nextafter(neg.max(), np.inf))
        zero_fp.append({
            "unit": unit, "threshold_above_max_negative": threshold,
            "positive_detected": int(np.count_nonzero(pos >= threshold)), "positive_denominator": 64,
            "source_threshold_positive_detected": int(np.count_nonzero(pos >= SOURCE_THRESHOLD)),
        })

    # Transformed results stratified by negative difficulty for each edge kind.
    transformed_cells = defaultdict(lambda: {"positive": [], "unsat": [], "near": [], "far": []})
    transformed_threshold = defaultdict(lambda: defaultdict(lambda: {"detected": 0, "rows": 0}))
    for row in analysis["transformed_records_edge_weighted"]:
        kind = "fixed" if row["fixed_assignment"] else "switch"
        unit = int(row["unit"])
        uid = int(row["unique_id"])
        assignment = int(row["target_assignment_id"])
        if row["stratum"] in ("sole_positive", "multi_positive"):
            category = "positive"
        elif row["stratum"] == "unsat_negative":
            category = "unsat"
        else:
            solutions = np.flatnonzero(counts[uid] == 0)
            d = min((int(solution) ^ assignment).bit_count() for solution in solutions)
            category = "near" if d == 1 else "far"
        transformed_cells[(unit, kind)][category].append(float(row["preactivation"]))
        transformed_threshold[kind][category]["rows"] += 1
        transformed_threshold[kind][category]["detected"] += int(row["detected"])
    transformed_auc = {}
    for kind in ("fixed", "switch"):
        transformed_auc[kind] = {}
        for category in ("unsat", "near", "far"):
            values = []
            for (unit, kind0), cells in transformed_cells.items():
                if kind0 == kind and cells["positive"] and cells[category]:
                    values.append(exact_auc(np.asarray(cells["positive"]), np.asarray(cells[category])))
            transformed_auc[kind][category] = {
                "unit_edge_cells": len(values), "mean_auroc": float(np.mean(values)),
                "median_auroc": float(np.median(values)), "min_auroc": float(np.min(values)),
                "max_auroc": float(np.max(values)),
            }

    gate_uncertainty = {
        "multi_negative_4_of_32_interval_95": clopper_pearson(4, 32),
        "probability_fail_guard_x_at_least_4_if_true_rate_0_10": binom_tail(4, 32, 0.10),
        "probability_fail_guard_x_at_least_4_if_true_rate_0_05": binom_tail(4, 32, 0.05),
        "unit81_pooled_positive_39_of_64_interval_95": clopper_pearson(39, 64),
        "positive_23_of_32_interval_95": clopper_pearson(23, 32),
        "positive_24_of_32_interval_95": clopper_pearson(24, 32),
        "probability_pass_positive_gate_x_at_least_24": {
            str(p): binom_tail(24, 32, p) for p in (0.70, 0.75, 0.80, 0.90)
        },
    }

    result = {
        "status": "PASS",
        "scope": "post hoc zero-forward analysis of stored all-17 formulas, seventeen preactivations and transformed records; protected decisions unchanged",
        "inputs": {
            "selected_bank.npz": sha256(STUDY / "selected_bank.npz"),
            "native_outputs.npz": sha256(STUDY / "native_outputs.npz"),
            "analysis.json": sha256(STUDY / "analysis.json"),
        },
        "candidate_assignment_competition": label_competition,
        "nominated_unit_rank_among_17_saved_units": unit_rank_for_assignment,
        "all_published_assignments_rank_first": all(row["published_rank_of_32"] == 1 for row in label_competition),
        "negative_clause_count_cells": grouped_rows,
        "near_two_or_more_hamming_one_solutions": {
            "false_positives": sum(e["false_positive"] for e in near_two), "rows": len(near_two),
            "all_one_violated_type": all(e["distinct_violated_clause_types"] == 1 for e in near_two),
        },
        "near_exactly_one_hamming_one_solution": {
            "false_positives": sum(e["false_positive"] for e in near_one), "rows": len(near_one),
        },
        "orbit_grouped_five_fold_regression": regression_cv,
        "zero_observed_false_positive_thresholds": zero_fp,
        "transformed_selection": {
            "rule": "for every labelled assignment-transposition edge and each of five base strata, retain the first eight frozen hash-ordered base cases whose transformed token differs and is absent from the historical ledger",
            "frozen": True, "rows_per_edge": 40, "rows_per_edge_stratum": 8,
        },
        "transformed_threshold_by_regime": {
            kind: {category: dict(values) for category, values in cells.items()}
            for kind, cells in transformed_threshold.items()
        },
        "transformed_auroc_by_regime": transformed_auc,
        "gate_uncertainty": gate_uncertainty,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    published_ranked = sum(row["published_rank_of_32"] == 1 for row in label_competition)
    lines = [
        "# Extended post hoc diagnostics for the all-17 SAT study", "",
        "These analyses fit no neural model and make no new checkpoint forward. They cannot alter the protected decisions.", "",
        "## Candidate-label competition", "",
        f"The published assignment ranks first among all 32 assignment labels for {published_ranked}/17 units on the 7,252-input selected bank.", "",
        "| Unit | Published rank / 32 | Published AUROC | Best alternative margin | Rank among 17 saved units for that label |", "|---:|---:|---:|---:|---:|",
    ]
    for row, urank in zip(label_competition, unit_rank_for_assignment):
        lines.append(f"| {row['unit']} | {row['published_rank_of_32']} | {row['published_auroc']:.6f} | {row['published_margin_over_best_alternative']:.6f} | {urank['rank_among_17_saved_units']} |")
    lines += ["", "## Neighbour multiplicity", "",
              f"Near negatives with at least two Hamming-one solutions: {sum(e['false_positive'] for e in near_two)}/{len(near_two)} false positives; every row has one distinct violated clause type.",
              f"Near negatives with exactly one Hamming-one solution: {sum(e['false_positive'] for e in near_one)}/{len(near_one)} false positives.", "",
              "## Orbit-grouped five-fold regression", "", "| Model | CV R² | CV RMSE |", "|:--|--:|--:|"]
    for name, row in regression_cv.items():
        lines.append(f"| {name.replace('_', ' ')} | {row['cv_r2_global']:.6f} | {row['cv_rmse']:.6f} |")
    lines += ["", "## Gate uncertainty", "",
              f"Exact 95% interval for 4/32: {gate_uncertainty['multi_negative_4_of_32_interval_95'][0]:.4f}--{gate_uncertainty['multi_negative_4_of_32_interval_95'][1]:.4f}.",
              f"At a true rate of 0.10, the probability of failing the <=3/32 guard is {gate_uncertainty['probability_fail_guard_x_at_least_4_if_true_rate_0_10']:.3f}; at 0.05 it is {gate_uncertainty['probability_fail_guard_x_at_least_4_if_true_rate_0_05']:.3f}.", "",
              "## Transformed selection", "", result["transformed_selection"]["rule"].capitalize() + ".", ""]
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))


if __name__ == "__main__":
    main()
