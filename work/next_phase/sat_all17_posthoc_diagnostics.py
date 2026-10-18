"""Post hoc diagnostics for the protected all-17 SAT label study.

This script is deliberately outside the frozen packet.  It reads the retained
bank and native outputs, fits no neural model, and cannot change the protected
criteria or their recorded decisions.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/next_phase/sat_all17_validation_v3_20261018"
OUT_JSON = STUDY / "posthoc_diagnostics.json"
OUT_MD = STUDY / "POSTHOC_DIAGNOSTICS.md"
sys.path.insert(0, str(STUDY))

from protocol import (  # noqa: E402
    ASSIGNMENT_NAMES,
    NEGATIVE_STRATA,
    POSITIVE_STRATA,
    SOURCE_THRESHOLD,
    STRATA,
    UNITS,
)
from verify_bank import truth_counts  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_r2(y: np.ndarray, columns: list[np.ndarray]) -> dict:
    x = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    sse = float(residual @ residual)
    centred = y - y.mean()
    sst = float(centred @ centred)
    return {
        "rows": int(len(y)),
        "rank": int(np.linalg.matrix_rank(x)),
        "r2": float(1.0 - sse / sst),
        "coefficients": beta.tolist(),
    }


def unit_fixed_effects(unit_index: np.ndarray) -> list[np.ndarray]:
    return [np.ones(len(unit_index), dtype=np.float64)] + [
        (unit_index == i).astype(np.float64) for i in range(1, len(UNITS))
    ]


def nearest_flip_variables(counts: np.ndarray, assignment: int) -> tuple[int, ...]:
    solutions = np.flatnonzero(counts == 0)
    out = set()
    for solution in solutions:
        diff = int(solution) ^ int(assignment)
        if diff.bit_count() == 1:
            # Assignment bit 4 is x_0 and bit 0 is x_4.
            out.add(4 - (diff.bit_length() - 1))
    return tuple(sorted(out))


def main() -> None:
    if OUT_JSON.exists() or OUT_MD.exists():
        raise RuntimeError("post hoc output already exists")

    with np.load(STUDY / "selected_bank.npz", allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in z.files}
    with np.load(STUDY / "native_outputs.npz", allow_pickle=False) as z:
        native = {k: z[k].copy() for k in z.files}
    analysis = json.loads((STUDY / "analysis.json").read_text())

    counts = np.stack([truth_counts(row) for row in bank["unique_clauses"]])
    task_expected = np.where(np.any(counts == 0, axis=1), 13, 14)
    task_predicted = native["logits"].argmax(axis=1)

    events = []
    near_variable = {
        "all_nearest_membership": [defaultdict(int) for _ in range(5)],
        "exclusive_nearest": [defaultdict(int) for _ in range(5)],
    }
    ranking_loss = defaultdict(lambda: {"strict_inversions": 0, "ties": 0, "pairs": 0})
    per_unit_cross_tabs = {}

    for ui, (unit0, assignment0) in enumerate(
        zip(bank["unit_ids"], bank["unit_assignment_ids"])
    ):
        unit, assignment = int(unit0), int(assignment0)
        ai = int(np.flatnonzero(bank["unique_assignment_ids"] == assignment)[0])
        ids = bank["base_unique_ids"][ai]
        scores = native["pre"][ids, ui]
        unit_tab = defaultdict(int)

        positives = scores[list(POSITIVE_STRATA)].reshape(-1)
        for si, stratum in enumerate(STRATA):
            for qi, uid0 in enumerate(ids[si]):
                uid = int(uid0)
                target_positive = si in POSITIVE_STRATA
                detected = bool(scores[si, qi] >= SOURCE_THRESHOLD)
                label_error = detected != target_positive
                task_error = bool(task_predicted[uid] != task_expected[uid])
                distance = None
                flips: tuple[int, ...] = ()
                if np.any(counts[uid] == 0):
                    distance = min(
                        (int(solution) ^ assignment).bit_count()
                        for solution in np.flatnonzero(counts[uid] == 0)
                    )
                    if distance == 1 and not target_positive:
                        flips = nearest_flip_variables(counts[uid], assignment)
                        for variable in flips:
                            near_variable["all_nearest_membership"][variable]["denominator"] += 1
                            near_variable["all_nearest_membership"][variable]["false_positive"] += int(detected)
                        if len(flips) == 1:
                            variable = flips[0]
                            near_variable["exclusive_nearest"][variable]["denominator"] += 1
                            near_variable["exclusive_nearest"][variable]["false_positive"] += int(detected)
                events.append(
                    {
                        "unit_index": ui,
                        "unit": unit,
                        "assignment": assignment,
                        "stratum": stratum,
                        "unique_id": uid,
                        "score": float(scores[si, qi]),
                        "target_positive": target_positive,
                        "detected": detected,
                        "label_error": label_error,
                        "task_error": task_error,
                        "violated_clauses": int(counts[uid, assignment]),
                        "nearest_solution_hamming": distance,
                        "nearest_flip_variables": flips,
                    }
                )
                unit_tab[(label_error, task_error)] += 1

        for si in NEGATIVE_STRATA:
            neg = scores[si]
            if si == 2:
                category = "unsat_negative"
                categories = np.full(len(neg), category, dtype=object)
            else:
                distances = bank["negative_distance_class"][ai, si]
                categories = np.where(distances == 1, "near_negative", "far_negative")
            for score, category in zip(neg, categories):
                ranking_loss[str(category)]["strict_inversions"] += int(np.count_nonzero(score > positives))
                ranking_loss[str(category)]["ties"] += int(np.count_nonzero(score == positives))
                ranking_loss[str(category)]["pairs"] += len(positives)

        per_unit_cross_tabs[str(unit)] = {
            "both_correct": unit_tab[(False, False)],
            "label_error_task_correct": unit_tab[(True, False)],
            "label_correct_task_error": unit_tab[(False, True)],
            "both_error": unit_tab[(True, True)],
        }

    unit_index = np.asarray([e["unit_index"] for e in events], dtype=np.int16)
    score = np.asarray([e["score"] for e in events], dtype=np.float64)
    violated = np.asarray([e["violated_clauses"] for e in events], dtype=np.float64)
    sat_mask = np.asarray([e["nearest_solution_hamming"] is not None for e in events])
    distance = np.asarray(
        [np.nan if e["nearest_solution_hamming"] is None else e["nearest_solution_hamming"] for e in events],
        dtype=np.float64,
    )
    fixed_all = unit_fixed_effects(unit_index)
    fixed_sat = unit_fixed_effects(unit_index[sat_mask])
    regressions = {
        "all_rows_unit_fixed_effects_plus_violated_clauses": fit_r2(
            score, fixed_all + [violated]
        ),
        "sat_rows_unit_fixed_effects_plus_violated_clauses": fit_r2(
            score[sat_mask], fixed_sat + [violated[sat_mask]]
        ),
        "sat_rows_unit_fixed_effects_plus_nearest_solution_hamming": fit_r2(
            score[sat_mask], fixed_sat + [distance[sat_mask]]
        ),
        "sat_rows_unit_fixed_effects_plus_both": fit_r2(
            score[sat_mask], fixed_sat + [violated[sat_mask], distance[sat_mask]]
        ),
    }

    pooled_tab = defaultdict(int)
    for e in events:
        pooled_tab[(e["label_error"], e["task_error"])] += 1
    cross_tabs = {
        "pooled_unit_formula_events": {
            "both_correct": pooled_tab[(False, False)],
            "label_error_task_correct": pooled_tab[(True, False)],
            "label_correct_task_error": pooled_tab[(False, True)],
            "both_error": pooled_tab[(True, True)],
            "denominator": len(events),
            "dependence_note": "Rows share formulas and units; units 55 and 416 share the TTTFF base formulas.",
        },
        "per_unit": per_unit_cross_tabs,
    }

    variable_rows = {}
    for mode, cells in near_variable.items():
        variable_rows[mode] = {}
        for variable, cell in enumerate(cells):
            denominator = int(cell["denominator"])
            false_positive = int(cell["false_positive"])
            variable_rows[mode][f"x_{variable}"] = {
                "false_positive": false_positive,
                "denominator": denominator,
                "rate": float(false_positive / denominator) if denominator else None,
            }

    transformed = defaultdict(lambda: defaultdict(int))
    transformed_distinct = {}
    for row in analysis["transformed_records_edge_weighted"]:
        uid = int(row["unique_id"])
        assignment = int(row["target_assignment_id"])
        kind = "fixed" if row["fixed_assignment"] else "switch"
        stratum = row["stratum"]
        if stratum in ("sole_positive", "multi_positive"):
            category = "positive"
        elif stratum == "unsat_negative":
            category = "unsat_negative"
        else:
            distance0 = min(
                (int(solution) ^ assignment).bit_count()
                for solution in np.flatnonzero(counts[uid] == 0)
            )
            category = "near_negative" if distance0 == 1 else "far_negative"
        transformed[kind][category] += 1
        key = (int(row["unit"]), uid, assignment, stratum)
        transformed_distinct[key] = (kind, category)
    transformed_distinct_counts = defaultdict(lambda: defaultdict(int))
    for kind, category in transformed_distinct.values():
        transformed_distinct_counts[kind][category] += 1

    strict_total = sum(v["strict_inversions"] for v in ranking_loss.values())
    near_strict = ranking_loss["near_negative"]["strict_inversions"]
    ranking_result = {
        **{k: dict(v) for k, v in ranking_loss.items()},
        "strict_inversions_total": strict_total,
        "near_share_of_strict_inversions": float(near_strict / strict_total),
    }

    task_by_solution_class = {}
    solutions = np.count_nonzero(counts == 0, axis=1)
    for label, mask in {
        "unsat": solutions == 0,
        "sole_solution": solutions == 1,
        "multi_solution": solutions >= 2,
    }.items():
        task_by_solution_class[label] = {
            "errors": int(np.count_nonzero(task_predicted[mask] != task_expected[mask])),
            "denominator": int(np.count_nonzero(mask)),
        }

    result = {
        "status": "PASS",
        "scope": "post hoc descriptive analysis of stored all-17 bank and native outputs; no protected gate changed",
        "input_sha256": {
            "analysis.json": sha256(STUDY / "analysis.json"),
            "selected_bank.npz": sha256(STUDY / "selected_bank.npz"),
            "native_outputs.npz": sha256(STUDY / "native_outputs.npz"),
        },
        "regressions": regressions,
        "near_false_positives_by_flipped_variable": variable_rows,
        "unit_label_error_by_task_error": cross_tabs,
        "ranking_loss_decomposition": ranking_result,
        "transformed_near_far_composition": {
            "edge_weighted": {k: dict(v) for k, v in transformed.items()},
            "distinct_unit_input_events": {k: dict(v) for k, v in transformed_distinct_counts.items()},
            "note": "Transformed rows are orbit-linked dependent consistency checks, not independent samples.",
        },
        "task_errors_by_solution_class": task_by_solution_class,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    pooled = cross_tabs["pooled_unit_formula_events"]
    lines = [
        "# Post hoc diagnostics for the all-17 SAT study",
        "",
        "These diagnostics use only the retained bank and native outputs. They do not alter the protected decisions.",
        "",
        "## Competing graded-score summaries",
        "",
        "| Descriptive OLS model | Rows | R² |",
        "|:--|--:|--:|",
    ]
    for label, row in regressions.items():
        lines.append(f"| {label.replace('_', ' ')} | {row['rows']} | {row['r2']:.6f} |")
    lines += [
        "",
        "The regressions include unit fixed effects. They are descriptive fits on an engineered, dependent bank; they do not identify the represented variable.",
        "",
        "## Ranking-loss locality",
        "",
        f"Near negatives account for {near_strict}/{strict_total} ({100 * near_strict / strict_total:.1f}%) strict positive-negative inversions.",
        "",
        "## Threshold-label error versus task error",
        "",
        "| Event cell | Count |",
        "|:--|--:|",
        f"| Both correct | {pooled['both_correct']} |",
        f"| Label error, task correct | {pooled['label_error_task_correct']} |",
        f"| Label correct, task error | {pooled['label_correct_task_error']} |",
        f"| Both error | {pooled['both_error']} |",
        "",
        "Counts are unit-formula events and therefore inherit shared-formula dependence.",
        "",
        "## Near-negative false positives by flipped variable",
        "",
        "| Variable | FP / nearest-membership rows | FP / exclusive-nearest rows |",
        "|:--|--:|--:|",
    ]
    for variable in range(5):
        a = variable_rows["all_nearest_membership"][f"x_{variable}"]
        b = variable_rows["exclusive_nearest"][f"x_{variable}"]
        lines.append(
            f"| x_{variable} | {a['false_positive']}/{a['denominator']} | {b['false_positive']}/{b['denominator']} |"
        )
    lines += [
        "",
        "Membership rows can appear under more than one variable when a formula has several Hamming-one solutions; exclusive rows cannot.",
        "",
        "## Task errors by solution class",
        "",
        "| Class | Errors / rows |",
        "|:--|--:|",
    ]
    for label, row in task_by_solution_class.items():
        lines.append(f"| {label.replace('_', ' ')} | {row['errors']}/{row['denominator']} |")
    lines += [
        "",
        "## Transformed composition",
        "",
        "The JSON records fixed/switch and near/far/UNSAT composition for both edge-weighted and deduplicated unit-input events. The transformed bank is a dependent consistency check under a permutation-symmetric generator.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))


if __name__ == "__main__":
    main()
