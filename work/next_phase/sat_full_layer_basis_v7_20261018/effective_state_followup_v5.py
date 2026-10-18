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
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.decomposition import FastICA, MiniBatchDictionaryLearning, NMF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import auc_matrix, basis_statistics, maximum_matching  # noqa: E402
from protocol import BANK, SAT_TOKEN, UNSAT_TOKEN, semantic_counts  # noqa: E402


OUTPUT = HERE / "effective_state_followup_v5.json"
SUPERSEDES = HERE / "effective_state_followup_v4.json"
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
    scores = model.fit_transform(source)
    learned_components = np.asarray(model.components_, dtype=np.float64)
    reconstruction = scores @ learned_components
    relative_error = float(np.linalg.norm(source - reconstruction) / np.linalg.norm(source))
    return {
        "seed": seed,
        "converged_iterations": int(getattr(model, "n_iter_", -1)),
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
            "exact_binomial_95_interval": clopper_pearson(errors, trials),
        }
    rows["caveat"] = "exploratory localisation; assignment composition differs and one strong-axis assignment (FTTFF) has 4/45 errors"
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


def refitted_bit_readout(source: np.ndarray, truth: np.ndarray, groups: np.ndarray) -> dict:
    bits = (source >= 0.5).astype(np.float64)
    truth_binary = (truth == SAT_TOKEN).astype(np.int8)
    result = {}
    for name, features, scale in (
        ("threshold_bits", bits, False),
        ("continuous_activations", source, True),
    ):
        prediction = np.empty(len(source), dtype=np.int8)
        for fold in range(FOLDS):
            train = groups != fold
            test = groups == fold
            classifier = LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
                random_state=0,
            )
            model = make_pipeline(StandardScaler(), classifier) if scale else classifier
            model.fit(features[train], truth_binary[train])
            prediction[test] = model.predict(features[test])
        result[name] = {
            "accuracy": float(np.mean(prediction == truth_binary)),
            "errors": int(np.count_nonzero(prediction != truth_binary)),
        }
    result["method"] = "five-fold orbit-grouped L2 logistic regression with C=1; post hoc supervised comparator"
    result["interpretation"] = "difference from fixed-readout substitution can reflect readout recalibration and does not itself identify a causal code"
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
            "reason": "v5 retains v4 by hash and adds reviewer-requested purity, learned-basis, refitted-code, error-interval, and joint-zero diagnostics",
        },
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
