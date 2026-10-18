"""Analyze the frozen all-unit bank without fitting a learner."""
from __future__ import annotations

import json
import platform
from collections import defaultdict
import numpy as np

from protocol import *
from verify_bank import truth_counts


def weighted_auc(pos, neg, wp, wn):
    delta = pos[:, None] - neg[None, :]
    weight = wp[:, None] * wn[None, :]
    denom = weight.sum()
    if denom <= 0:
        return np.nan
    return float(((delta > 0) * weight + 0.5 * (delta == 0) * weight).sum() / denom)


def unit_indices_for_assignment(assignment):
    return np.flatnonzero(ASSIGNMENT_IDS == assignment)


def load_bound_json(name):
    value = json.loads((HERE / name).read_text())
    if value.get("status") != "PASS":
        raise RuntimeError(f"{name} is not a passing receipt")
    return value


def solution_class(counts):
    nsol = int(np.count_nonzero(np.asarray(counts) == 0))
    return 0 if nsol == 0 else 1 if nsol == 1 else 2


def main() -> None:
    frozen = verify_frozen_packet()
    for name in ("analysis.json", "REPORT.md", "bootstrap_macro_auroc.npz"):
        if (HERE / name).exists():
            raise RuntimeError("exclusive analysis outputs already exist")
    environment = {"python": platform.python_version(), "numpy": np.__version__}
    if environment != {"python": ANALYSIS_PYTHON, "numpy": ANALYSIS_NUMPY}:
        raise RuntimeError("pinned analysis environment drift")

    construction = load_bound_json("construction.json")
    verification = load_bound_json("bank_verification.json")
    native_receipt = load_bound_json("native_receipt.json")
    replay_receipt = load_bound_json("replay_receipt.json")
    selected_hash = sha256(HERE / "selected_bank.npz")
    native_hash = sha256(HERE / "native_outputs.npz")
    replay_hash = sha256(HERE / "replay_outputs.npz")
    for label, receipt in (("construction", construction), ("verification", verification),
                           ("native", native_receipt), ("replay", replay_receipt)):
        if any(receipt.get(key) != frozen[key] for key in ("freeze_sha256", "code_sha256", "selfcheck_sha256", "source_sha256")):
            raise RuntimeError(f"{label} receipt frozen-code mismatch")
        if receipt.get("selected_bank_sha256") != selected_hash:
            raise RuntimeError(f"{label} receipt selected-bank mismatch")
    if native_receipt.get("output_sha256") != native_hash:
        raise RuntimeError("native output hash mismatch")
    if replay_receipt.get("native_outputs_sha256") != native_hash:
        raise RuntimeError("replay source hash mismatch")
    if replay_receipt.get("output_sha256") != replay_hash:
        raise RuntimeError("replay output hash mismatch")
    if not replay_receipt.get("same_threshold_bits") or not replay_receipt.get("same_task_argmax"):
        raise RuntimeError("decision-relevant replay mismatch")

    with np.load(HERE / "selected_bank.npz", allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in z.files}
    with np.load(HERE / "native_outputs.npz", allow_pickle=False) as z:
        native = {k: z[k].copy() for k in z.files}
    expected_ids = np.arange(len(bank["unique_tokens"]), dtype=np.int32)
    if not np.array_equal(native["unique_ids"], expected_ids):
        raise RuntimeError("native output rows are not the complete bank in canonical order")
    if native["pre"].shape != (len(expected_ids), len(UNITS)) or native["logits"].shape[0] != len(expected_ids):
        raise RuntimeError("native output shape mismatch")
    if SAT_TOKEN != 2 * 5 + 3 or UNSAT_TOKEN != 2 * 5 + 4:
        raise RuntimeError("task-label constants do not match the published five-variable vocabulary")
    if native_receipt.get("source_bindings") != {
        "helpers_py_sha256": HELPERS_SHA256,
        "interpretation_py_sha256": INTERPRETATION_SHA256,
    }:
        raise RuntimeError("native receipt lacks the frozen vocabulary/assignment source bindings")

    counts = np.stack([truth_counts(row) for row in bank["unique_clauses"]])
    solution_classes = np.asarray([solution_class(row) for row in counts], dtype=np.int8)
    task_expected = np.where(solution_classes > 0, SAT_TOKEN, UNSAT_TOKEN)
    task_predicted = native["logits"].argmax(axis=1)

    per_unit = []
    base_score_cells = {}
    base_orbit_cells = {}
    for ui, (unit0, assignment0) in enumerate(zip(UNITS, ASSIGNMENT_IDS)):
        unit, assignment = int(unit0), int(assignment0)
        ai = int(np.where(bank["unique_assignment_ids"] == assignment)[0][0])
        ids = bank["base_unique_ids"][ai]
        scores = native["pre"][ids, ui]
        orbits = bank["orbit_ids"][ids]
        positive = scores[list(POSITIVE_STRATA)].reshape(-1)
        negative = scores[list(NEGATIVE_STRATA)].reshape(-1)
        auc = exact_auc(positive, negative)
        cells = []
        for si, name in enumerate(STRATA):
            target = si in POSITIVE_STRATA
            detected = scores[si] >= SOURCE_THRESHOLD
            cells.append({"stratum": name, "target_positive": target,
                          "detected": int(detected.sum()), "denominator": int(len(detected)),
                          "rate": float(detected.mean())})
        positive_sensitivity = float(np.mean(positive >= SOURCE_THRESHOLD))
        negative_specificity = float(np.mean(negative < SOURCE_THRESHOLD))
        task_cells = []
        for si, name in enumerate(STRATA):
            uid = ids[si]
            correct = task_predicted[uid] == task_expected[uid]
            task_cells.append({"stratum": name, "correct": int(correct.sum()),
                               "denominator": int(len(correct))})
        near_far = {}
        for si in (3, 4):
            near_far[STRATA[si]] = {}
            for distance_class, label in ((1, "near_hamming_1"), (2, "far_hamming_at_least_2")):
                mask = bank["negative_distance_class"][ai, si] == distance_class
                if int(mask.sum()) != 16:
                    raise RuntimeError("missing or unbalanced near/far analysis cell")
                negative_subset = scores[si][mask]
                detected = negative_subset >= SOURCE_THRESHOLD
                near_far[STRATA[si]][label] = {
                    "false_positive_detected": int(detected.sum()),
                    "denominator": int(len(detected)), "rate": float(detected.mean()),
                    "positive_vs_subset_auroc": exact_auc(positive, negative_subset),
                }
        per_unit.append({"unit": unit, "assignment": ASSIGNMENT_NAMES[ui],
                         "assignment_id": assignment, "base_auroc": auc,
                         "threshold_cells": cells,
                         "threshold_pooled_balanced_accuracy": 0.5 * (positive_sensitivity + negative_specificity),
                         "near_far_specificity": near_far,
                         "task_cells": task_cells})
        base_score_cells[ui] = scores
        base_orbit_cells[ui] = orbits

    by_assignment = defaultdict(list)
    for row in per_unit:
        by_assignment[row["assignment_id"]].append(row["base_auroc"])
    assignment_aurocs = {str(k): float(np.mean(v)) for k, v in by_assignment.items()}
    macro = float(np.mean(list(assignment_aurocs.values())))

    transformed_records = []
    transformed_distinct = defaultdict(dict)
    transformed_edges = defaultdict(lambda: {"fixed": [], "switch": []})
    for row, uid0 in zip(bank["transform_rows"], bank["transform_unique_ids"]):
        ti, source_assignment, target_assignment, stratum, proposal = map(int, row)
        uid = int(uid0)
        target_positive = stratum in POSITIVE_STRATA
        fixed = source_assignment == target_assignment
        for ui0 in unit_indices_for_assignment(target_assignment):
            ui = int(ui0)
            score = float(native["pre"][uid, ui])
            prior = transformed_distinct[ui].get(uid)
            event = (score, target_positive)
            if prior is not None and prior != event:
                raise RuntimeError("duplicate transformed unit/input event has inconsistent semantics")
            transformed_distinct[ui][uid] = event
            transformed_edges[ui]["fixed" if fixed else "switch"].append(event)
            transformed_records.append({
                "unit": int(UNITS[ui]), "transposition": list(TRANSPOSITIONS[ti]),
                "source_assignment_id": source_assignment,
                "target_assignment_id": target_assignment, "stratum": STRATA[stratum],
                "fixed_assignment": fixed, "proposal_id": proposal, "unique_id": uid,
                "preactivation": score, "detected": bool(score >= SOURCE_THRESHOLD),
            })
    for ui, row in enumerate(per_unit):
        distinct = list(transformed_distinct[ui].values())
        pos = np.asarray([score for score, target in distinct if target], dtype=np.float64)
        neg = np.asarray([score for score, target in distinct if not target], dtype=np.float64)
        row["transformed_distinct_unit_input_events"] = len(distinct)
        row["transformed_auroc"] = exact_auc(pos, neg)
        row["base_minus_transformed_auroc"] = row["base_auroc"] - row["transformed_auroc"]
        row["transformed_threshold_edge_weighted"] = {}
        for kind in ("fixed", "switch"):
            values = transformed_edges[ui][kind]
            if kind == "switch" and int(UNITS[ui]) == 41 and values:
                raise RuntimeError("unit 41 unexpectedly has a switching edge")
            if not values:
                if kind != "switch" or int(UNITS[ui]) != 41:
                    raise RuntimeError("unexpected missing transformed threshold cell")
                row["transformed_threshold_edge_weighted"][kind] = {
                    "applicable": False, "positive_detected": 0, "positive_denominator": 0,
                    "negative_detected": 0, "negative_denominator": 0,
                }
                continue
            row["transformed_threshold_edge_weighted"][kind] = {
                "applicable": True,
                "positive_detected": int(sum(score >= SOURCE_THRESHOLD for score, target in values if target)),
                "positive_denominator": int(sum(bool(target) for _, target in values)),
                "negative_detected": int(sum(score >= SOURCE_THRESHOLD for score, target in values if not target)),
                "negative_denominator": int(sum(not bool(target) for _, target in values)),
            }

    orbit_class = {}
    for orbit, cls in zip(bank["orbit_ids"], solution_classes):
        key = bytes(orbit)
        if key in orbit_class and orbit_class[key] != int(cls):
            raise RuntimeError("a variable-permutation orbit crosses solution-count strata")
        orbit_class[key] = int(cls)
    class_orbits = {cls: sorted(o for o, value in orbit_class.items() if value == cls) for cls in range(3)}
    if any(not values for values in class_orbits.values()):
        raise RuntimeError("empty solution-count bootstrap stratum")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap_macro = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        orbit_weight = {}
        for population in class_orbits.values():
            sampled = rng.integers(0, len(population), size=len(population))
            multiplicity = np.bincount(sampled, minlength=len(population))
            orbit_weight.update({orbit: float(multiplicity[i]) for i, orbit in enumerate(population)})
        draw_unit = []
        for ui in range(len(UNITS)):
            scores = base_score_cells[ui]
            weights = np.asarray([[orbit_weight[bytes(orbit)] for orbit in row] for row in base_orbit_cells[ui]])
            draw_unit.append(weighted_auc(
                scores[list(POSITIVE_STRATA)].reshape(-1), scores[list(NEGATIVE_STRATA)].reshape(-1),
                weights[list(POSITIVE_STRATA)].reshape(-1), weights[list(NEGATIVE_STRATA)].reshape(-1),
            ))
        draw_assignment = []
        for assignment in UNIQUE_ASSIGNMENTS:
            values = [draw_unit[i] for i in unit_indices_for_assignment(int(assignment))]
            draw_assignment.append(float(np.mean(values)))
        bootstrap_macro[draw] = float(np.mean(draw_assignment))
    if not np.isfinite(bootstrap_macro).all():
        raise RuntimeError("non-finite bootstrap draw")
    with (HERE / "bootstrap_macro_auroc.npz").open("xb") as f:
        np.savez_compressed(f, macro_auroc=bootstrap_macro,
                            solution_class=np.asarray(sorted(class_orbits), dtype=np.int8),
                            orbit_count=np.asarray([len(class_orbits[i]) for i in sorted(class_orbits)], dtype=np.int32))

    unit_aurocs = np.array([row["base_auroc"] for row in per_unit])
    multi_fp = np.asarray([row["threshold_cells"][4]["rate"] for row in per_unit])
    discrimination_gates = {
        "assignment_macro_auroc_at_least_0_95": bool(macro >= 0.95),
        "every_unit_auroc_at_least_0_80": bool(np.all(unit_aurocs >= 0.80)),
        "at_least_15_of_17_unit_auroc_at_least_0_90": bool(np.count_nonzero(unit_aurocs >= 0.90) >= 15),
        "no_multi_negative_fpr_above_0_10": bool(np.all(multi_fp <= 0.10)),
    }
    positive_cell_pass = [all(row["threshold_cells"][si]["detected"] >= 24 for si in POSITIVE_STRATA) for row in per_unit]
    negative_cell_pass = [all(row["threshold_cells"][si]["detected"] <= 3 for si in NEGATIVE_STRATA) for row in per_unit]
    balanced_pass = [row["threshold_pooled_balanced_accuracy"] >= 0.85 for row in per_unit]
    calibration_gates = {
        "every_unit_each_positive_stratum_at_least_24_of_32": bool(all(positive_cell_pass)),
        "every_unit_each_negative_stratum_at_most_3_of_32_false_positives": bool(all(negative_cell_pass)),
        "at_least_15_of_17_pooled_balanced_accuracy_at_least_0_85": bool(sum(balanced_pass) >= 15),
    }
    discrimination_met = bool(all(discrimination_gates.values()))
    calibration_met = bool(all(calibration_gates.values()))
    result = {
        "status": "PASS", "protected_broad_discrimination_criterion_met": discrimination_met,
        "source_threshold_calibration_criterion_met": calibration_met,
        "discrimination_gates": discrimination_gates, "calibration_gates": calibration_gates,
        "base": {"per_unit": per_unit, "assignment_aurocs": assignment_aurocs,
                 "assignment_macro_auroc": macro, "unit_median_auroc": float(np.median(unit_aurocs)),
                 "unit_min_auroc": float(unit_aurocs.min()), "unit_max_auroc": float(unit_aurocs.max()),
                 "bootstrap_macro_interval_2_5_97_5": np.quantile(bootstrap_macro, [0.025, 0.975]).tolist(),
                 "bootstrap_orbit_counts_by_solution_class": {str(k): len(v) for k, v in class_orbits.items()}},
        "transformed_records_edge_weighted": transformed_records,
        "task_overall": {"correct": int(np.count_nonzero(task_expected == task_predicted)),
                         "denominator": int(len(task_expected)), "sat_token": SAT_TOKEN, "unsat_token": UNSAT_TOKEN},
        "inputs": {"selected_bank_sha256": selected_hash, "native_outputs_sha256": native_hash,
                   "replay_outputs_sha256": replay_hash,
                   "construction_receipt_sha256": sha256(HERE / "construction.json"),
                   "verification_receipt_sha256": sha256(HERE / "bank_verification.json"),
                   "native_receipt_sha256": sha256(HERE / "native_receipt.json"),
                   "replay_receipt_sha256": sha256(HERE / "replay_receipt.json")},
        "environment": environment,
        "scope": "fixed checkpoint and selected formula banks; no fitted learner; solution-count-stratified full-permutation-orbit bootstrap",
        **frozen,
    }
    with (HERE / "analysis.json").open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    lines = ["# All-unit SAT label validation", "",
             f"Protected broad-discrimination criterion met: **{discrimination_met}**", "",
             f"Separate source-threshold calibration criterion met: **{calibration_met}**", "",
             f"Assignment-macro base AUROC: {macro:.6f} (solution-count-stratified permutation-orbit interval {result['base']['bootstrap_macro_interval_2_5_97_5']})", "",
             "| Unit | Assignment | Base AUROC | Transformed AUROC | Threshold balanced accuracy |",
             "|---:|:---:|---:|---:|---:|"]
    for row in per_unit:
        lines.append(f"| {row['unit']} | {row['assignment']} | {row['base_auroc']:.6f} | {row['transformed_auroc']:.6f} | {row['threshold_pooled_balanced_accuracy']:.6f} |")
    lines.extend(["", "All threshold numerators, near/far specificity cells, semantic strata, task cells, gates, transformed records, hashes and environment evidence are retained in `analysis.json`.", "",
                  "Even if every discrimination gate passes, the conclusion is restricted to one checkpoint and this balanced conditional mixture. It does not validate the full source threshold unless the separate calibration criterion passes; it does not establish equivariance, specificity outside the declared near/far-balanced cohort, prevalence, causal role, or training-set exclusion."])
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "broad_discrimination_met": discrimination_met,
                      "threshold_calibration_met": calibration_met, "macro_auroc": macro,
                      "unit_min": float(unit_aurocs.min()), "unit_median": float(np.median(unit_aurocs)),
                      "transformed_edge_records": len(transformed_records)}, sort_keys=True))


if __name__ == "__main__":
    main()
