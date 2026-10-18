#!/usr/bin/env python3
"""Retrospective zero-forward learned-basis and coding diagnostics.

This follow-up operates only on arrays retained by the protected full-layer study.
It adds activation-mass purity, learned-basis comparators, task-error intervals,
joint-zero checks, and a cross-validated refitted-bit comparator.  It does not
change any protected criterion or decision.
"""
from __future__ import annotations

import hashlib
import json
import sys
import platform
import warnings
import sklearn
import scipy
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import FastICA, MiniBatchDictionaryLearning, NMF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import auc_matrix, basis_statistics, maximum_matching  # noqa: E402
from protocol import BANK, SAT_TOKEN, UNSAT_TOKEN, semantic_counts  # noqa: E402


OUTPUT = HERE / "effective_state_followup_v6.json"
SUPERSEDES = HERE / "effective_state_followup_v5.json"
NATIVE = HERE / "full_native_outputs.npz"
SOURCE_REPORT = np.array([
    10, 29, 36, 41, 48, 55, 77, 81, 96, 150, 185, 189, 195, 201,
    224, 231, 245, 261, 291, 304, 317, 326, 334, 374, 380, 411, 416,
    435, 450, 482, 490, 492, 495, 499,
], dtype=np.int32)
SEEDS = (0, 1, 2, 3, 4)
FOLDS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assignment_name(index: int) -> str:
    return "".join("T" if (index >> (4 - bit)) & 1 else "F" for bit in range(5))


def two_sided_auc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = auc_matrix(scores, labels)
    return np.maximum(values, 1.0 - values)


def matching_record(scores: np.ndarray, labels: np.ndarray) -> dict:
    stats = basis_statistics(two_sided_auc(scores, labels))
    return {
        "matching_32_mean": float(stats["matching_32"]["mean"]),
        "matching_32_minimum": float(stats["matching_32"]["minimum"]),
        "matching_uncovered16_mean": float(stats["matching_uncovered16"]["mean"]),
        "matching_uncovered16_minimum": float(stats["matching_uncovered16"]["minimum"]),
        "reusable_axis_minimum": float(stats["reusable_axis_min_32"]),
    }


def fold_ids(orbit_ids: np.ndarray) -> np.ndarray:
    result = np.empty(len(orbit_ids), dtype=np.int8)
    for index, raw in enumerate(orbit_ids):
        digest = hashlib.sha256(bytes(raw)).digest()
        result[index] = int.from_bytes(digest[:8], "little") % FOLDS
    return result


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [lower, upper]


def activation_mass_purity(source: np.ndarray, labels: np.ndarray) -> dict:
    auc = two_sided_auc(source, labels)
    rows = []
    for column, unit in enumerate(SOURCE_REPORT):
        assignment = int(np.argmax(auc[:, column]))
        values = source[:, column]
        total_mass = float(values.sum())
        on = labels[:, assignment]
        on_mass = float(values[on].sum())
        active = values > 0
        rows.append({
            "unit": int(unit),
            "best_assignment": assignment,
            "best_assignment_name": assignment_name(assignment),
            "best_assignment_auc": float(auc[assignment, column]),
            "total_activation_mass": total_mass,
            "on_label_activation_mass": on_mass,
            "on_label_mass_fraction": float(on_mass / total_mass),
            "off_label_mass_fraction": float(1.0 - on_mass / total_mass),
            "active_rows": int(active.sum()),
            "on_label_active_rows": int(np.count_nonzero(active & on)),
            "on_label_active_row_fraction": float(np.count_nonzero(active & on) / active.sum()),
        })
    purities = np.asarray([row["on_label_mass_fraction"] for row in rows])
    return {
        "definition": "fraction of each unit's summed post-ReLU activation mass occurring on rows satisfying that unit's best-AUROC assignment label",
        "units": rows,
        "minimum": float(purities.min()),
        "median": float(np.median(purities)),
        "maximum": float(purities.max()),
        "units_below_0_80": int(np.count_nonzero(purities < 0.80)),
        "units_below_0_90": int(np.count_nonzero(purities < 0.90)),
    }


def cosine_axis_recovery(components: np.ndarray) -> dict:
    norms = np.linalg.norm(components, axis=1)
    if np.any(norms <= 0):
        raise RuntimeError("zero learned component")
    cosines = np.abs(components / norms[:, None])
    axes, selected, certificate = maximum_matching(cosines, True)
    return {
        "matched_native_axes": axes.tolist(),
        "mean_absolute_cosine": float(selected.mean()),
        "minimum_absolute_cosine": float(selected.min()),
        "maximum_absolute_cosine": float(selected.max()),
        "components_above_0_90": int(np.count_nonzero(selected >= 0.90)),
        "components_above_0_99": int(np.count_nonzero(selected >= 0.99)),
        "hungarian_certificate": certificate,
    }


def learned_run(family: str, seed: int, source: np.ndarray, labels: np.ndarray,
                components: int) -> dict:
    if family == "NMF":
        model = NMF(
            n_components=components, init="nndsvdar", random_state=seed,
            max_iter=2000, tol=1e-6,
        )
    elif family == "FastICA":
        model = FastICA(
            n_components=components, whiten="unit-variance", random_state=seed,
            max_iter=2000, tol=1e-6,
        )
    elif family == "sparse_dictionary":
        model = MiniBatchDictionaryLearning(
            n_components=components, alpha=1.0, max_iter=1000,
            batch_size=512, random_state=seed,
            transform_algorithm="lasso_lars", transform_alpha=1.0,
        )
    else:
        raise ValueError(family)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        scores = model.fit_transform(source)
    analysis_components = np.asarray(model.components_, dtype=np.float64)
    # Compare synthesis directions for every family, in the original feature space.
    if family == "FastICA":
        learned_components = np.asarray(model.mixing_.T, dtype=np.float64)
        reconstruction = model.inverse_transform(scores)
        np.testing.assert_allclose(reconstruction, scores @ learned_components + model.mean_,
                                   atol=1e-12, rtol=1e-12)
    else:
        learned_components = analysis_components
        reconstruction = scores @ learned_components
    relative_error = float(np.linalg.norm(source - reconstruction) / np.linalg.norm(source))
    return {
        "seed": seed,
        "iterations": int(getattr(model, "n_iter_", -1)),
        "convergence_warnings": [str(item.message) for item in caught
                                 if issubclass(item.category, ConvergenceWarning)],
        "axis_comparison": "unit-normalised synthesis directions in the native coordinate space",
        "hyperparameters": {key: str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
                            for key, value in model.get_params().items()},
        "relative_frobenius_reconstruction_error": relative_error,
        "assignment_matching": matching_record(scores, labels),
        "native_axis_recovery": cosine_axis_recovery(learned_components),
    }


def summarise_runs(runs: list[dict]) -> dict:
    keys = (
        "matching_32_mean", "matching_32_minimum",
        "matching_uncovered16_mean", "matching_uncovered16_minimum",
    )
    summary = {}
    for key in keys:
        values = np.asarray([run["assignment_matching"][key] for run in runs])
        summary[key] = {
            "minimum_across_seeds": float(values.min()),
            "mean_across_seeds": float(values.mean()),
            "maximum_across_seeds": float(values.max()),
        }
    cosines = np.asarray([run["native_axis_recovery"]["mean_absolute_cosine"] for run in runs])
    summary["mean_axis_recovery_cosine"] = {
        "minimum_across_seeds": float(cosines.min()),
        "mean_across_seeds": float(cosines.mean()),
        "maximum_across_seeds": float(cosines.max()),
    }
    return summary


def learned_bases(source: np.ndarray, labels: np.ndarray) -> dict:
    result = {
        "native": matching_record(source, labels),
        "methods": {},
        "status": "retrospective learned comparators; hyperparameters fixed for this follow-up but not independently protected",
    }
    for family, components in (("NMF", 32), ("NMF", 34), ("FastICA", 34), ("sparse_dictionary", 34)):
        name = f"{family}_k{components}"
        runs = [learned_run(family, seed, source, labels, components) for seed in SEEDS]
        result["methods"][name] = {
            "seeds": list(SEEDS),
            "runs": runs,
            "summary": summarise_runs(runs),
        }
    return result


def task_error_intervals(logits: np.ndarray, counts: np.ndarray) -> dict:
    truth = np.where(np.any(counts == 0, axis=1), SAT_TOKEN, UNSAT_TOKEN)
    prediction = logits.argmax(axis=1)
    solution_count = np.sum(counts == 0, axis=1)
    sole = solution_count == 1
    sole_assignment = np.argmax(counts == 0, axis=1)
    weak = sole & np.isin(sole_assignment, [8, 20, 29])
    other = sole & ~weak
    rows = {}
    for name, mask in (("three_weak_single_axis_assignments", weak), ("other_sole_solution_assignments", other)):
        errors = int(np.count_nonzero(mask & (prediction != truth)))
        trials = int(mask.sum())
        rows[name] = {
            "errors": errors,
            "rows": trials,
            "error_rate": float(errors / trials),
            "binomial_95_interval_if_independent_rows": clopper_pearson(errors, trials),
        }
    rows["caveat"] = "exploratory localisation; linked formulas are dependent, so binomial intervals are only an independence sensitivity, not valid bank-sampling intervals; assignment composition differs and FTTFF has 4/45 errors"
    return rows


def joint_zero_check(source: np.ndarray, resid_mid: np.ndarray, w_u: np.ndarray,
                     bias_logits: np.ndarray, logits: np.ndarray, counts: np.ndarray) -> dict:
    truth = np.where(np.any(counts == 0, axis=1), SAT_TOKEN, UNSAT_TOKEN)
    unsat = truth == UNSAT_TOKEN
    all_zero = np.all(source == 0, axis=1)
    residual_logits = resid_mid.astype(np.float64) @ w_u.astype(np.float64) + bias_logits
    residual_prediction = residual_logits.argmax(axis=1)
    answer = np.isin(residual_prediction, [SAT_TOKEN, UNSAT_TOKEN])
    native_prediction = logits.argmax(axis=1)
    return {
        "all_zero_source_state": {
            "all_rows": int(all_zero.sum()),
            "all_row_fraction": float(all_zero.mean()),
            "unsat_rows": int(np.count_nonzero(all_zero & unsat)),
            "unsat_fraction": float(np.mean(all_zero[unsat])),
            "sat_rows": int(np.count_nonzero(all_zero & ~unsat)),
        },
        "after_all_34_zeroing": {
            "task_accuracy": float(np.mean(residual_prediction == truth)),
            "predicted_unsat": int(np.count_nonzero(residual_prediction == UNSAT_TOKEN)),
            "predicted_sat": int(np.count_nonzero(residual_prediction == SAT_TOKEN)),
            "non_answer_argmaxes": int(np.count_nonzero(~answer)),
        },
        "native_non_answer_argmaxes": int(np.count_nonzero(~np.isin(native_prediction, [SAT_TOKEN, UNSAT_TOKEN]))),
        "unsat_prevalence": float(unsat.mean()),
    }


def fit_predict(train_x, train_y, test_x, c, scale):
    classifier = LogisticRegression(C=float(c), solver="lbfgs", max_iter=10000,
                                    random_state=0)
    model = make_pipeline(StandardScaler(), classifier) if scale else classifier
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_x, train_y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("logistic readout failed to converge")
    return model.predict(test_x)


def prediction_summary(prediction, truth):
    return {"accuracy": float(np.mean(prediction == truth)),
            "errors": int(np.count_nonzero(prediction != truth)),
            "error_row_ids": np.flatnonzero(prediction != truth).tolist()}


def refitted_bit_readout(source: np.ndarray, truth: np.ndarray, groups: np.ndarray) -> dict:
    """Paired outer folds; scaling and C selection use training rows only."""
    representations = {"threshold_bits": (source >= 0.5).astype(np.float64),
                       "continuous_activations": source}
    truth_binary = (truth == SAT_TOKEN).astype(np.int8)
    grid = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    result = {"method": "five outer orbit-hash folds; inner leave-one-of-four-fold-out selection of C by total validation errors; ties select smaller C",
              "C_grid": list(grid), "primary_preprocessing": "standardise both representations within each training fit",
              "fold_row_counts": [int(np.count_nonzero(groups == fold)) for fold in range(FOLDS)],
              "matched_preprocessing": {}}
    for scale in (True, False):
        setting = "both_standardised" if scale else "both_unscaled"
        condition = {}
        predictions = {}
        for name, features in representations.items():
            fixed = {str(c): np.empty(len(source), dtype=np.int8) for c in grid}
            nested = np.empty(len(source), dtype=np.int8)
            selections = []
            for fold in range(FOLDS):
                train, test = groups != fold, groups == fold
                inner_errors = []
                for c in grid:
                    errors = 0
                    for inner in range(FOLDS):
                        if inner == fold:
                            continue
                        inner_train = train & (groups != inner)
                        inner_test = train & (groups == inner)
                        predicted = fit_predict(features[inner_train], truth_binary[inner_train],
                                                features[inner_test], c, scale)
                        errors += int(np.count_nonzero(predicted != truth_binary[inner_test]))
                    inner_errors.append(errors)
                    fixed[str(c)][test] = fit_predict(features[train], truth_binary[train],
                                                     features[test], c, scale)
                best = int(np.argmin(inner_errors))
                selected = grid[best]
                nested[test] = fixed[str(selected)][test]
                selections.append({"outer_fold": fold, "selected_C": selected,
                                   "inner_total_errors_by_C": inner_errors})
            condition[name] = {"nested": prediction_summary(nested, truth_binary),
                               "selections": selections,
                               "fixed_C_sensitivity": {c: prediction_summary(p, truth_binary)
                                                       for c, p in fixed.items()}}
            predictions[name] = nested
        bits_ok = predictions["threshold_bits"] == truth_binary
        cont_ok = predictions["continuous_activations"] == truth_binary
        condition["paired_nested_comparison"] = {
            "bits_wrong_continuous_correct": int(np.count_nonzero(~bits_ok & cont_ok)),
            "bits_correct_continuous_wrong": int(np.count_nonzero(bits_ok & ~cont_ok)),
            "both_wrong": int(np.count_nonzero(~bits_ok & ~cont_ok)),
            "both_correct": int(np.count_nonzero(bits_ok & cont_ok)),
            "continuous_minus_bits_accuracy": float(cont_ok.mean() - bits_ok.mean())}
        result["matched_preprocessing"][setting] = condition
    result["interpretation"] = "exploratory supervised predictive comparisons with matched preprocessing; no optimum over decoders or causal information-loss bound is claimed"
    result["selection_status"] = "grid and nested procedure chosen retrospectively; outer rows excluded from fitting, scaling and C selection"
    return result


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"exclusive output already exists: {OUTPUT}")
    prior = json.loads(SUPERSEDES.read_text())
    if prior.get("status") != "PASS":
        raise RuntimeError("superseded diagnostic is not PASS")
    with np.load(NATIVE, allow_pickle=False) as archive:
        post = archive["post"].astype(np.float64)
        logits = archive["logits"].astype(np.float64)
        resid_mid = archive["resid_mid"].astype(np.float64)
        w_u = archive["W_U"].astype(np.float64)
        bias_logits = archive["mlp_bias_logits"].astype(np.float64)
    with np.load(BANK, allow_pickle=False) as archive:
        formulas = archive["unique_clauses"].copy()
        orbit_ids = archive["orbit_ids"].copy()
    counts = semantic_counts(formulas)
    labels = counts == 0
    truth = np.where(np.any(labels, axis=1), SAT_TOKEN, UNSAT_TOKEN)
    groups = fold_ids(orbit_ids)
    source = post[:, SOURCE_REPORT]
    result = {
        "status": "PASS",
        "scope": "retrospective zero-forward analysis of retained protected-study arrays; protected decisions unchanged",
        "supersedes": {
            "path": str(SUPERSEDES),
            "sha256": sha256(SUPERSEDES),
            "reason": "v6 corrects ICA reconstruction and synthesis-axis comparisons, records convergence and software, and replaces unequal-scaling headline readouts with matched nested comparisons",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "script_sha256": sha256(Path(__file__)),
        "inputs": {
            "native_outputs_sha256": sha256(NATIVE),
            "selected_bank_sha256": sha256(BANK),
        },
        "activation_mass_purity": activation_mass_purity(source, labels),
        "learned_basis_comparators": learned_bases(source, labels),
        "task_error_localisation_with_intervals": task_error_intervals(logits, counts),
        "joint_zero_state_check": joint_zero_check(source, resid_mid, w_u, bias_logits, logits, counts),
        "cross_validated_refitted_readout": refitted_bit_readout(source, truth, groups),
        "limitations": [
            "one checkpoint and one engineered bank",
            "all diagnostics were designed after protected outcomes were opened",
            "learned-basis hyperparameters were not independently protected or tuned for a common complexity budget",
            "axis recovery is cosine alignment in the retained native 34-dimensional coordinate system",
            "the refitted classifier is a supervised predictive comparator, not a causal abstraction",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(OUTPUT),
        "sha256": sha256(OUTPUT),
        "activation_mass_purity": {
            key: result["activation_mass_purity"][key]
            for key in ("minimum", "median", "maximum", "units_below_0_80", "units_below_0_90")
        },
        "learned_basis_summaries": {
            name: value["summary"]
            for name, value in result["learned_basis_comparators"]["methods"].items()
        },
        "cross_validated_refitted_readout": result["cross_validated_refitted_readout"],
        "joint_zero_state_check": result["joint_zero_state_check"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
