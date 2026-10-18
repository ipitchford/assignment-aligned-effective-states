"""Pre-specified native-axis, Haar-basis and exact-readout analysis."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import traceback

import numpy as np

from protocol import *


def save_json_exclusive(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def average_ranks(scores):
    scores = np.asarray(scores, dtype=np.float64)
    n, d = scores.shape
    order = np.argsort(scores, axis=0, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    ranks = np.empty((n, d), dtype=np.float64)
    for column in range(d):
        values = sorted_scores[:, column]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        ends = np.r_[starts[1:], n]
        tied = np.repeat((starts + 1 + ends) / 2.0, ends - starts)
        ranks[order[:, column], column] = tied
    return ranks


def auc_matrix(scores, labels):
    ranks = average_ranks(scores)
    labels = np.asarray(labels, dtype=bool)
    npos = labels.sum(axis=0).astype(np.float64)
    nneg = len(labels) - npos
    if np.any(npos == 0) or np.any(nneg == 0):
        raise RuntimeError("assignment class missing")
    rank_sums = labels.astype(np.float64).T @ ranks
    return (rank_sums - npos[:, None] * (npos[:, None] + 1) / 2) / (npos[:, None] * nneg[:, None])


def maximum_matching(weights, return_certificate=False):
    weights = np.asarray(weights, dtype=np.float64)
    n, m = weights.shape
    if n > m: raise ValueError("matching requires rows <= columns")
    cost = -weights
    u = np.zeros(n + 1); v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int32); way = np.zeros(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(m + 1, np.inf); used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta = np.inf; j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(m + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    assignment = np.full(n, -1, dtype=np.int32)
    for j in range(1, m + 1):
        if p[j] > 0: assignment[p[j] - 1] = j - 1
    if np.any(assignment < 0): raise RuntimeError("incomplete matching")
    selected = weights[np.arange(n), assignment]
    reduced_cost = cost - u[1:, None] - v[None, 1:]
    primal = float(cost[np.arange(n), assignment].sum())
    dual = float(u[1:].sum() + v[1:].sum())
    certificate = {
        "minimum_reduced_cost": float(reduced_cost.min()),
        "maximum_matched_reduced_cost_abs": float(np.max(np.abs(reduced_cost[np.arange(n), assignment]))),
        "primal_min_cost": primal,
        "dual_value": dual,
        "absolute_duality_gap": float(abs(primal - dual)),
        "maximum_column_dual": float(v[1:].max()),
    }
    if (certificate["minimum_reduced_cost"] < -1e-10 or
            certificate["maximum_matched_reduced_cost_abs"] > 1e-10 or
            certificate["absolute_duality_gap"] > 1e-9 or
            certificate["maximum_column_dual"] > 1e-10):
        raise RuntimeError("Hungarian primal-dual certificate failed")
    if return_certificate:
        return assignment, selected, certificate
    return assignment, selected


def subset_matching(two_sided, assignments):
    axes, selected, certificate = maximum_matching(two_sided[np.asarray(assignments, dtype=int)], True)
    return {"assignments": np.asarray(assignments, dtype=int).tolist(), "axes": axes.tolist(),
            "scores": selected.tolist(), "mean": float(selected.mean()), "minimum": float(selected.min()),
            "hungarian_certificate": certificate}


def direct_matching(two_sided, assignments):
    axes, selected, certificate = maximum_matching(two_sided, True)
    return {"assignments": np.asarray(assignments, dtype=int).tolist(), "axes": axes.tolist(),
            "scores": selected.tolist(), "mean": float(selected.mean()), "minimum": float(selected.min()),
            "hungarian_certificate": certificate}


def basis_statistics(two_sided):
    all_match = subset_matching(two_sided, np.arange(32))
    covered = subset_matching(two_sided, COVERED_ASSIGNMENTS)
    uncovered = subset_matching(two_sided, UNCOVERED_ASSIGNMENTS)
    best_per_assignment = two_sided.max(axis=1)
    best_per_axis = two_sided.max(axis=0)
    uncovered_set = set(map(int, UNCOVERED_ASSIGNMENTS))
    joint_uncovered = [score for assignment, score in zip(all_match["assignments"], all_match["scores"])
                       if int(assignment) in uncovered_set]
    return {"matching_32": all_match, "matching_covered16": covered, "matching_uncovered16": uncovered,
            "matching_32_uncovered_row_mean": float(np.mean(joint_uncovered)),
            "reusable_axis_mean_32": float(best_per_assignment.mean()),
            "reusable_axis_min_32": float(best_per_assignment.min()),
            "axis_counts": {str(t): int(np.count_nonzero(best_per_axis >= t)) for t in ALIGNMENT_THRESHOLDS}}


def exclusive_uncovered_auc(scores, labels, exclusive_scope):
    rows = []
    for assignment in UNCOVERED_ASSIGNMENTS:
        assignment = int(assignment)
        rows.append(auc_matrix(scores[exclusive_scope], labels[exclusive_scope, assignment, None])[0])
    return np.stack(rows)


def event_summary(mask, values, changed_pred, pred, correct, changed_correct, activation):
    n = int(mask.sum())
    if not n: return {"n": 0}
    return {"n": n, "mean_activation": float(activation[mask].mean()),
            "median_activation": float(np.median(activation[mask])),
            "mean_sat_minus_unsat_contribution": float(values[mask].mean()),
            "median_sat_minus_unsat_contribution": float(np.median(values[mask])),
            "accuracy": float(changed_correct[mask].mean()),
            "argmax_changes": int(np.count_nonzero(changed_pred[mask] != pred[mask])),
            "correct_to_incorrect": int(np.count_nonzero(correct[mask] & ~changed_correct[mask])),
            "incorrect_to_correct": int(np.count_nonzero(~correct[mask] & changed_correct[mask]))}


def derangement(n, rng):
    if n == 1:
        raise RuntimeError("singleton stratum cannot be deranged")
    original = np.arange(n)
    while True:
        proposal = rng.permutation(n)
        if np.all(proposal != original): return proposal


def ablation_summary(logits, post, coefficient, units, task_label, pred, correct):
    units = np.asarray(units, dtype=int)
    changed = logits - post[:, units] @ coefficient[units]
    changed_pred = changed.argmax(axis=1); changed_correct = changed_pred == task_label
    sorted_logits = np.sort(changed, axis=1)
    return {"units": units.tolist(), "accuracy": float(changed_correct.mean()),
            "argmax_changes": int(np.count_nonzero(changed_pred != pred)),
            "correct_to_incorrect": int(np.count_nonzero(correct & ~changed_correct)),
            "incorrect_to_correct": int(np.count_nonzero(~correct & changed_correct)),
            "top_two_logit_gap_below_1e-4": int(np.count_nonzero(sorted_logits[:, -1] - sorted_logits[:, -2] < 1e-4))}


def monte_carlo(native, rotations):
    values = np.asarray(rotations, dtype=np.float64)
    exceed = int(np.count_nonzero(values >= native))
    sd = float(values.std(ddof=1))
    return {"inferential_status": "descriptive", "native": float(native), "rotation_mean": float(values.mean()),
            "rotation_min": float(values.min()), "rotation_max": float(values.max()),
            "rotation_exceedance_count": exceed, "native_minus_rotation_max": float(native - values.max()),
            "native_minus_rotation_mean": float(native - values.mean()),
            "rotation_sd": sd,
            "standardised_native_minus_rotation_mean": float((native - values.mean()) / sd) if sd else None,
            "native_reference_quantile": float(np.mean(values < native)),
            "directional_monte_carlo_p": float((1 + exceed) / (1 + len(values)))}


def primary_decision(primary_p, exclusive_win):
    primary_keys = ("matching_uncovered16", "matching_32")
    ordered = sorted(primary_keys, key=lambda key: primary_p[key])
    passed = {key: False for key in primary_keys}
    if primary_p[ordered[0]] <= 0.025:
        passed[ordered[0]] = True
        if primary_p[ordered[1]] <= 0.05:
            passed[ordered[1]] = True
    both = all(passed.values())
    if both and exclusive_win:
        conclusion_key = "both_primaries_and_exclusive"
    elif both:
        conclusion_key = "both_primaries_only"
    elif any(passed.values()):
        conclusion_key = "one_primary"
    else:
        conclusion_key = "neither_primary"
    fixed_limit = (" This is a within-checkpoint, within-bank single-axis-separability comparison against Haar references whose exchangeability null is generically false for nonnegative sparse ReLU coordinates; it does not separate assignment-specific alignment from generic coordinate privilege (see the clause-presence and SAT/solution-count control gaps) and is not evidence that any coordinate is causally used.")
    conclusion_map = {
        "both_primaries_and_exclusive": "Native coordinates exceed the fixed Haar references on both primary matching statistics; the 32-assignment primary includes 16 assignments selected for published native units. The native-over-Haar matching advantage is also observed in the descriptive exclusive-positive secondary (uncovered assignments, SAT formulas with no covered solution; matched axes may include published units; not multiplicity-controlled)." + fixed_limit,
        "both_primaries_only": "Native coordinates exceed the fixed Haar references on both primary matching statistics; the 32-assignment primary includes 16 assignments selected for published native units. The exclusive-positive secondary does not exceed every reference; do not claim evidence beyond the published selection set." + fixed_limit,
        "one_primary": "Only one prespecified primary passes its Holm extremeness threshold; the 32-assignment primary, if passed, includes 16 assignments selected for published native units. Report mixed evidence and do not claim general native-basis concentration." + fixed_limit,
        "neither_primary": "Neither prespecified primary passes its Holm extremeness threshold; do not claim native-basis concentration over the fixed Haar references." + fixed_limit,
    }
    return passed, conclusion_key, conclusion_map[conclusion_key]


def execute(frozen, versions):
    preflight = verify_live_preflight()
    output_path = HERE / "analysis.json"
    if output_path.exists(): raise RuntimeError("exclusive analysis output already exists")
    native_path = HERE / "full_native_outputs.npz"
    replay_path = HERE / "replay_outputs.npz"
    native_receipt = json.loads((HERE / "native_receipt.json").read_text())
    replay_receipt = json.loads((HERE / "replay_receipt.json").read_text())
    for receipt, stage in ((native_receipt, "initial"), (replay_receipt, "replay")):
        if receipt.get("status") != "PASS" or receipt.get("stage") != stage or receipt.get("freeze_sha256") != frozen["freeze_sha256"]:
            raise RuntimeError(f"{stage} receipt gate failed")
    if sha256(native_path) != native_receipt.get("output_sha256") or sha256(replay_path) != replay_receipt.get("output_sha256"):
        raise RuntimeError("native output hash mismatch")
    if replay_receipt.get("initial_output_sha256") != native_receipt.get("output_sha256"):
        raise RuntimeError("replay-to-initial linkage mismatch")
    with np.load(BANK, allow_pickle=False) as z:
        tokens = z["unique_tokens"].copy(); formulas = z["unique_clauses"].copy()
        base_ids = z["base_unique_ids"].copy(); unique_assignments = z["unique_assignment_ids"].copy()
        bank_units = z["unit_ids"].copy(); bank_assignments = z["unit_assignment_ids"].copy()
    check_bank_tokens(tokens, formulas)
    if not np.array_equal(bank_units, UNITS) or not np.array_equal(bank_assignments, ASSIGNMENTS):
        raise RuntimeError("published mapping mismatch")
    with np.load(native_path, allow_pickle=False) as z:
        native = {key: z[key].copy() for key in z.files}
    expected_shapes = {"pre": (EXPECTED_ROWS, 512), "post": (EXPECTED_ROWS, 512),
                       "resid_mid": (EXPECTED_ROWS, 128), "resid_post": (EXPECTED_ROWS, 128),
                       "logits": (EXPECTED_ROWS, 16), "unit_logit_coefficients": (512, 16),
                       "W_out": (128, 512), "b_out": (128,), "W_U": (128, 16),
                       "unique_ids": (EXPECTED_ROWS,)}
    for key, shape in expected_shapes.items():
        if key not in native or native[key].shape != shape: raise RuntimeError(f"native shape mismatch: {key}")
    if not np.array_equal(native["unique_ids"], np.arange(EXPECTED_ROWS, dtype=np.int32)):
        raise RuntimeError("native row order mismatch")
    if not all(np.isfinite(native[key]).all() for key in expected_shapes if key != "unique_ids"):
        raise RuntimeError("non-finite native array")
    counts = semantic_counts(formulas); labels = counts == 0; sat_rows = labels.any(axis=1)
    solution_counts = labels.sum(axis=1)
    logits = native["logits"].astype(np.float64); post = native["post"].astype(np.float64)
    coefficient = native["W_out"].astype(np.float64).T @ native["W_U"].astype(np.float64)
    coefficient_delta = float(np.max(np.abs(
        coefficient - native["unit_logit_coefficients"].astype(np.float64))))
    if coefficient_delta > 1e-6:
        raise RuntimeError("saved float32 coefficient disagrees with float64 recomputation")
    task_label = np.where(sat_rows, SAT_TOKEN, UNSAT_TOKEN)
    pred = logits.argmax(axis=1); correct = pred == task_label
    if int(correct.sum()) != EXPECTED_BASELINE_CORRECT:
        raise RuntimeError("baseline accuracy does not reproduce prior analysis")
    prior_analysis = json.loads(PRIOR_ANALYSIS.read_text())
    expected_names = ("sole_positive", "multi_positive", "unsat_negative", "sole_negative", "multi_negative")
    for ui, (unit0, assignment0) in enumerate(zip(UNITS, ASSIGNMENTS)):
        unit = int(unit0); assignment = int(assignment0)
        observed = post[:, unit] >= SOURCE_THRESHOLD
        ai = int(np.flatnonzero(unique_assignments == assignment)[0])
        prior_row = prior_analysis["base"]["per_unit"][ui]
        if prior_row.get("unit") != unit or len(prior_row.get("threshold_cells", [])) != 5:
            raise RuntimeError("prior threshold-cell inventory mismatch")
        for si, cell in enumerate(prior_row["threshold_cells"]):
            ids = base_ids[ai, si]; detected = int(np.count_nonzero(observed[ids]))
            if cell.get("stratum") != expected_names[si] or cell.get("denominator") != len(ids) or detected != cell["detected"]:
                raise RuntimeError(f"base threshold count mismatch unit {unit} stratum {si}")
    clause_presence = np.column_stack([np.any(formulas == clause_id, axis=1) for clause_id in range(40)])
    control_labels = np.column_stack([clause_presence, sat_rows, solution_counts == 1, solution_counts >= 2])
    combined_labels = np.column_stack([labels, control_labels])
    positive_counts = labels.sum(axis=0)
    covered_any = labels[:, COVERED_ASSIGNMENTS].any(axis=1)
    exclusive_scope = exclusive_scope_from_labels(labels)
    pre_auc = auc_matrix(native["pre"], labels)
    combined_auc = auc_matrix(native["post"], combined_labels)
    post_auc = combined_auc[:32]
    post_two = np.maximum(post_auc, 1 - post_auc)
    sat_auc = auc_matrix(native["post"][sat_rows], labels[sat_rows])
    sat_two = np.maximum(sat_auc, 1 - sat_auc)
    exclusive_auc = exclusive_uncovered_auc(post, labels, exclusive_scope)
    native_stats = {"all_rows": basis_statistics(post_two), "sat_only": basis_statistics(sat_two),
                    "exclusive_uncovered16": direct_matching(
                        np.maximum(exclusive_auc, 1-exclusive_auc), UNCOVERED_ASSIGNMENTS)}
    exclusive_row = native_stats["exclusive_uncovered16"]
    exclusive_axes = np.asarray(exclusive_row["axes"], dtype=int)
    exclusive_row["positive_direction"] = (exclusive_auc[np.arange(len(UNCOVERED_ASSIGNMENTS)), exclusive_axes] >= 0.5).tolist()
    exclusive_row["published_axis"] = np.isin(exclusive_axes, UNITS).tolist()
    native_controls = {"clause_presence_matching_mean": float(maximum_matching(np.maximum(combined_auc[32:72], 1-combined_auc[32:72]))[1].mean()),
                       "sat_label_best_axis_auc": float(np.maximum(combined_auc[72], 1-combined_auc[72]).max()),
                       "sole_solution_label_best_axis_auc": float(np.maximum(combined_auc[73], 1-combined_auc[73]).max()),
                       "multi_solution_label_best_axis_auc": float(np.maximum(combined_auc[74], 1-combined_auc[74]).max())}
    native_stats["dead_units"] = int(np.count_nonzero(np.all(native["post"] == 0, axis=0)))
    native_stats["post_matrix_rank"] = int(np.linalg.matrix_rank(native["post"].astype(np.float64)))
    for scope, raw_auc in (("all_rows", post_auc), ("sat_only", sat_auc)):
        for key in ("matching_32", "matching_covered16", "matching_uncovered16"):
            row = native_stats[scope][key]
            assignments = np.asarray(row["assignments"], dtype=int); axes = np.asarray(row["axes"], dtype=int)
            row["positive_direction"] = (raw_auc[assignments, axes] >= 0.5).tolist()
            row["published_axis"] = np.isin(axes, UNITS).tolist()
    per_unit = [{"unit": unit, "best_positive_pre_assignment": int(np.argmax(pre_auc[:, unit])),
                 "best_positive_pre_auc": float(np.max(pre_auc[:, unit])),
                 "best_two_sided_post_assignment": int(np.argmax(post_two[:, unit])),
                 "best_two_sided_post_auc": float(np.max(post_two[:, unit]))} for unit in range(512)]
    per_assignment = []
    for assignment in range(32):
        order = np.argsort(-pre_auc[assignment], kind="stable")
        per_assignment.append({"assignment": assignment, "positive_count": int(positive_counts[assignment]),
                               "sole_positive_count": int(np.count_nonzero(labels[:, assignment] & (solution_counts == 1))),
                               "multi_positive_count": int(np.count_nonzero(labels[:, assignment] & (solution_counts >= 2))),
                               "best_unit": int(order[0]), "best_positive_auc": float(pre_auc[assignment, order[0]]),
                               "top5_units": order[:5].tolist(), "top5_aurocs": pre_auc[assignment, order[:5]].tolist()})
    published = []
    for unit, assignment in zip(UNITS, ASSIGNMENTS):
        scores = pre_auc[int(assignment)]; value = scores[int(unit)]
        published.append({"unit": int(unit), "assignment": int(assignment), "positive_auc": float(value),
                          "units_strictly_better": int(np.count_nonzero(scores > value)),
                          "units_tied": int(np.count_nonzero(scores == value)),
                          "rank_min": int(1 + np.count_nonzero(scores > value)),
                          "two_sided_post_auc": float(post_two[int(assignment), int(unit)])})
    rng = np.random.default_rng(ROTATION_SEED); rotations = []; post64 = post
    for index in range(ROTATIONS):
        gaussian = rng.standard_normal((512, 512))
        gaussian_sha256 = hashlib.sha256(gaussian.tobytes(order="C")).hexdigest()
        q, r = np.linalg.qr(gaussian)
        q *= np.where(np.diag(r) >= 0, 1.0, -1.0)
        orthogonal_error = float(np.max(np.abs(q.T @ q - np.eye(512))))
        if orthogonal_error > 1e-10: raise RuntimeError("rotation orthogonality failure")
        rotated = post64 @ q
        full_combined_auc = auc_matrix(rotated, combined_labels); full_auc = full_combined_auc[:32]
        sat_rot_auc = auc_matrix(rotated[sat_rows], labels[sat_rows])
        exclusive_rot_auc = exclusive_uncovered_auc(rotated, labels, exclusive_scope)
        exclusive_record = direct_matching(np.maximum(exclusive_rot_auc, 1-exclusive_rot_auc), UNCOVERED_ASSIGNMENTS)
        exclusive_axes = np.asarray(exclusive_record["axes"], dtype=int)
        exclusive_record["positive_direction"] = (exclusive_rot_auc[np.arange(len(UNCOVERED_ASSIGNMENTS)), exclusive_axes] >= 0.5).tolist()
        exclusive_record["published_axis"] = np.isin(exclusive_axes, UNITS).tolist()
        rotations.append({"index": index, "q_sha256": hashlib.sha256(q.tobytes(order="C")).hexdigest(),
                          "gaussian_sha256": gaussian_sha256,
                          "max_orthogonality_error": orthogonal_error,
                          "all_rows": basis_statistics(np.maximum(full_auc, 1-full_auc)),
                          "sat_only": basis_statistics(np.maximum(sat_rot_auc, 1-sat_rot_auc)),
                          "exclusive_uncovered16": exclusive_record,
                          "controls": {"clause_presence_matching_mean": float(maximum_matching(np.maximum(full_combined_auc[32:72], 1-full_combined_auc[32:72]))[1].mean()),
                                       "sat_label_best_axis_auc": float(np.maximum(full_combined_auc[72], 1-full_combined_auc[72]).max()),
                                       "sole_solution_label_best_axis_auc": float(np.maximum(full_combined_auc[73], 1-full_combined_auc[73]).max()),
                                       "multi_solution_label_best_axis_auc": float(np.maximum(full_combined_auc[74], 1-full_combined_auc[74]).max())}})
        print(json.dumps({"rotation_completed": index + 1, "of": ROTATIONS}), flush=True)
    comparisons = {}
    for scope in ("all_rows", "sat_only"):
        comparisons[scope] = {}
        for key in ("matching_32", "matching_covered16", "matching_uncovered16"):
            comparisons[scope][key] = monte_carlo(native_stats[scope][key]["mean"], [row[scope][key]["mean"] for row in rotations])
        for key in ("reusable_axis_mean_32", "reusable_axis_min_32"):
            comparisons[scope][key] = monte_carlo(native_stats[scope][key], [row[scope][key] for row in rotations])
        for threshold in ALIGNMENT_THRESHOLDS:
            key = str(threshold)
            comparisons[scope][f"axis_count_{key}"] = monte_carlo(native_stats[scope]["axis_counts"][key], [row[scope]["axis_counts"][key] for row in rotations])
        comparisons[scope]["matching_32_uncovered_row_mean"] = monte_carlo(
            native_stats[scope]["matching_32_uncovered_row_mean"],
            [row[scope]["matching_32_uncovered_row_mean"] for row in rotations])
    comparisons["exclusive_uncovered16"] = monte_carlo(
        native_stats["exclusive_uncovered16"]["mean"],
        [row["exclusive_uncovered16"]["mean"] for row in rotations])
    control_comparisons = {key: monte_carlo(native_controls[key], [row["controls"][key] for row in rotations])
                           for key in native_controls}
    primary_keys = ("matching_uncovered16", "matching_32")
    primary_p = {key: comparisons["all_rows"][key]["directional_monte_carlo_p"] for key in primary_keys}
    exclusive_win = bool(comparisons["exclusive_uncovered16"]["rotation_exceedance_count"] == 0)
    passed, conclusion_key, licensed_conclusion = primary_decision(primary_p, exclusive_win)
    primary_inference = {"familywise_alpha": 0.05,
                         "method": "Holm-labelled extremeness ranks over two pre-specified full-bank matching means; Haar exchangeability is generically false",
                         "p_values": primary_p, "passed_holm_extremeness_threshold_against_haar_references": passed}
    for key in primary_keys:
        comparisons["all_rows"][key]["inferential_status"] = "primary_holm_family"
    primary_inference["exclusive_secondary_native_exceeds_all_rotations"] = exclusive_win
    primary_inference.update(conclusion_key=conclusion_key,
                             licensed_conclusion=licensed_conclusion)
    readout = []; child_sequences = np.random.SeedSequence(RESAMPLE_SEED).spawn(len(UNITS))
    for ui, (unit0, assignment0) in enumerate(zip(UNITS, ASSIGNMENTS)):
        unit = int(unit0); assignment = int(assignment0); expected = labels[:, assignment]; observed = post[:, unit] >= SOURCE_THRESHOLD
        masks = {"true_positive": expected & observed, "false_negative": expected & ~observed,
                 "false_positive": ~expected & observed, "true_negative": ~expected & ~observed}
        margin = post[:, unit] * (coefficient[unit, SAT_TOKEN] - coefficient[unit, UNSAT_TOKEN])
        zero_logits = logits - post[:, unit, None] * coefficient[unit]
        zero_pred = zero_logits.argmax(axis=1); zero_correct = zero_pred == task_label
        categories = {name: event_summary(mask, margin, zero_pred, pred, correct, zero_correct, post[:, unit]) for name, mask in masks.items()}
        semantic = strata(counts, assignment); unit_rng = np.random.default_rng(child_sequences[ui]); reps = []
        for replicate in range(RESAMPLES):
            replacement = np.empty(len(post), dtype=np.float64)
            for group in range(5):
                ids = np.flatnonzero(semantic == group)
                replacement[ids] = post[ids[derangement(len(ids), unit_rng)], unit]
            changed_logits = logits + (replacement - post[:, unit])[:, None] * coefficient[unit]
            changed_pred = changed_logits.argmax(axis=1); changed_correct = changed_pred == task_label
            resampled_margin = replacement * (coefficient[unit, SAT_TOKEN] - coefficient[unit, UNSAT_TOKEN])
            reps.append({"replicate": replicate, "accuracy": float(changed_correct.mean()),
                         "argmax_change_rate": float(np.mean(changed_pred != pred)),
                         "categories": {name: {"baseline": event_summary(mask, margin, pred, pred, correct, correct, post[:, unit]),
                                               "resampled": event_summary(mask, resampled_margin, changed_pred, pred, correct, changed_correct, replacement),
                                               "mean_delta_contribution": float((resampled_margin[mask] - margin[mask]).mean()) if np.any(mask) else None}
                                        for name, mask in masks.items()}})
        readout.append({"unit": unit, "assignment": assignment,
                        "seed_spawn_key": list(child_sequences[ui].spawn_key),
                        "direct_margin_coefficient": float(coefficient[unit, SAT_TOKEN] - coefficient[unit, UNSAT_TOKEN]),
                        "zero_ablation": ablation_summary(logits, post, coefficient, [unit], task_label, pred, correct),
                        "categories": categories, "resamples": reps,
                        "resample_summary": {"accuracy_mean": float(np.mean([row["accuracy"] for row in reps])),
                                             "accuracy_min": float(np.min([row["accuracy"] for row in reps])),
                                             "accuracy_max": float(np.max([row["accuracy"] for row in reps])),
                                             "argmax_change_rate_mean": float(np.mean([row["argmax_change_rate"] for row in reps]))}})
    all_unit_ablation = [dict(unit=unit, **ablation_summary(logits, post, coefficient, [unit], task_label, pred, correct)) for unit in range(512)]
    joint_ablation = {"duplicate_label_units_55_416": ablation_summary(logits, post, coefficient, [55, 416], task_label, pred, correct),
                      "all_published_17": ablation_summary(logits, post, coefficient, UNITS, task_label, pred, correct)}
    result = {"status": "PASS", "scope": "one checkpoint and engineered 7,252-formula bank",
              "environment": versions, "bank_sha256": sha256(BANK), "native_outputs_sha256": sha256(native_path),
              "native_receipt_sha256": sha256(HERE / "native_receipt.json"),
              "replay_receipt_sha256": sha256(HERE / "replay_receipt.json"),
              "task": {"baseline_accuracy": float(correct.mean()), "correct": int(correct.sum()), "total": int(len(correct)),
                       "top_two_logit_gap_below_1e-4": int(np.count_nonzero(
                           np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2] < 1e-4))},
              "coefficient_check": {"method": "float64 W_out.T @ W_U",
                                    "max_difference_from_saved_float32": coefficient_delta},
              "assignment_composition": per_assignment, "native_basis": native_stats,
              "native_control_labels": native_controls, "control_label_comparisons": control_comparisons,
              "preflight": preflight,
              "native_per_unit_alignment": per_unit, "published_unit_ranks": published,
              "rotations": rotations, "basis_comparisons": comparisons, "primary_inference": primary_inference,
              "readout_published": readout, "all_unit_zero_ablation": all_unit_ablation,
              "joint_ablation": joint_ablation, **frozen}
    required = ("native_basis", "rotations", "basis_comparisons", "control_label_comparisons",
                "primary_inference", "readout_published", "all_unit_zero_ablation", "joint_ablation")
    if result["status"] != "PASS" or any(key not in result for key in required) or len(result["rotations"]) != ROTATIONS or len(result["all_unit_zero_ablation"]) != 512:
        raise RuntimeError("analysis schema incomplete")
    save_json_exclusive(output_path, result)
    print(json.dumps({"status": "PASS", "analysis_sha256": sha256(output_path),
                      "primary_inference": primary_inference}, sort_keys=True))


def main():
    frozen = verify_frozen_packet()
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    if versions != ANALYSIS_ENV: raise RuntimeError(f"analysis environment drift: {versions}")
    thread_env = {key: os.environ.get(key) for key in
                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    if any(value != "1" for value in thread_env.values()):
        raise RuntimeError(f"analysis BLAS thread drift: {thread_env}")
    versions.update(machine=platform.machine(), system=platform.system(),
                    blas_thread_environment=thread_env)
    validate_analysis_prerequisites(HERE, frozen["freeze_sha256"])
    save_json_exclusive(HERE / "attempt_analysis.json",
                        {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         "freeze_sha256": frozen["freeze_sha256"], "environment": versions})
    try:
        execute(frozen, versions)
    except Exception as error:
        save_json_exclusive(HERE / "failure_analysis.json",
                            {"error_type": type(error).__name__, "error": str(error),
                             "traceback": traceback.format_exc(), "freeze_sha256": frozen["freeze_sha256"]})
        raise


if __name__ == "__main__":
    main()
