#!/usr/bin/env python3
"""Zero-forward audit of effective dimensionality and geometry-respecting references.

This analysis is retrospective.  It operates only on the protected study's retained
arrays and therefore does not alter the protected primary comparison or its status.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(HERE))

from analyze import auc_matrix, basis_statistics, direct_matching, maximum_matching  # noqa: E402
from protocol import (  # noqa: E402
    ASSIGNMENTS,
    BANK,
    COVERED_ASSIGNMENTS,
    ROTATION_SEED,
    SAT_TOKEN,
    UNSAT_TOKEN,
    SOURCE_THRESHOLD,
    UNCOVERED_ASSIGNMENTS,
    UNITS,
    semantic_counts,
)


OUTPUT = HERE / "effective_live_state_analysis_v2.json"
SUPERSEDES = HERE / "effective_live_state_analysis.json"
SOURCE_REPORT = np.array([
    10, 29, 36, 41, 48, 55, 77, 81, 96, 150, 185, 189, 195, 201,
    224, 231, 245, 261, 291, 304, 317, 326, 334, 374, 380, 411, 416,
    435, 450, 482, 490, 492, 495, 499,
], dtype=np.int32)
N_REFERENCES = 63
RIDGE = 1.0
FOLDS = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assignment_name(index: int) -> str:
    return "".join("T" if (index >> (4 - bit)) & 1 else "F" for bit in range(5))


def two_sided_auc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = auc_matrix(scores, labels)
    return np.maximum(values, 1.0 - values)


def summarise_reference(native_value: float, values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "native": float(native_value),
        "reference_mean": float(array.mean()),
        "reference_sd": float(array.std(ddof=1)),
        "reference_min": float(array.min()),
        "reference_max": float(array.max()),
        "reference_exceedance_count": int(np.count_nonzero(array >= native_value)),
        "native_minus_reference_max": float(native_value - array.max()),
        "directional_rank_tail": float((1 + np.count_nonzero(array >= native_value)) / (1 + len(array))),
        "inferential_status": "retrospective descriptive reference comparison",
    }


def matching_record(scores: np.ndarray, labels: np.ndarray) -> dict:
    auc = two_sided_auc(scores, labels)
    stats = basis_statistics(auc)
    return {
        "matching_32_mean": stats["matching_32"]["mean"],
        "matching_32_minimum": stats["matching_32"]["minimum"],
        "matching_uncovered16_mean": stats["matching_uncovered16"]["mean"],
        "matching_uncovered16_minimum": stats["matching_uncovered16"]["minimum"],
        "axis_counts": stats["axis_counts"],
        "reusable_axis_min_32": stats["reusable_axis_min_32"],
    }


def random_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    gaussian = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def positive_mixing(dimension: int, rng: np.random.Generator, nonzeros: int | None) -> np.ndarray:
    if nonzeros is None:
        matrix = rng.exponential(size=(dimension, dimension))
    else:
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        for column in range(dimension):
            chosen = rng.choice(dimension, size=nonzeros, replace=False)
            matrix[chosen, column] = rng.exponential(size=nonzeros)
    norms = np.linalg.norm(matrix, axis=0)
    return matrix / norms


def compare_references(native: dict, references: list[dict]) -> dict:
    keys = ("matching_32_mean", "matching_uncovered16_mean", "matching_32_minimum", "reusable_axis_min_32")
    return {key: summarise_reference(native[key], [row[key] for row in references]) for key in keys}


def fold_ids(orbit_ids: np.ndarray) -> np.ndarray:
    result = np.empty(len(orbit_ids), dtype=np.int8)
    for index, raw in enumerate(orbit_ids):
        digest = hashlib.sha256(bytes(raw)).digest()
        result[index] = int.from_bytes(digest[:8], "little") % FOLDS
    return result


def cross_validated_ridge_scores(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    scores = np.empty((len(features), labels.shape[1]), dtype=np.float64)
    for fold in range(FOLDS):
        test = groups == fold
        train = ~test
        mean = features[train].mean(axis=0)
        scale = features[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        x_train = (features[train] - mean) / scale
        x_test = (features[test] - mean) / scale
        design_train = np.column_stack([np.ones(train.sum()), x_train])
        design_test = np.column_stack([np.ones(test.sum()), x_test])
        penalty = np.eye(design_train.shape[1]) * RIDGE
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(design_train.T @ design_train + penalty, design_train.T @ labels[train])
        scores[test] = design_test @ weights
    return scores


def per_unit_summary(post: np.ndarray, coefficient: np.ndarray, active: np.ndarray) -> list[dict]:
    total_energy = float(np.square(post).sum())
    rows = []
    for unit in active:
        values = post[:, unit]
        sat_coefficient = float(coefficient[unit, SAT_TOKEN])
        margin_coefficient = float(coefficient[unit, SAT_TOKEN] - coefficient[unit, UNSAT_TOKEN])
        rows.append({
            "unit": int(unit),
            "in_source_report": bool(unit in set(SOURCE_REPORT.tolist())),
            "activation_frequency": float(np.mean(values > 0)),
            "activation_mean": float(values.mean()),
            "activation_maximum": float(values.max()),
            "squared_norm": float(np.square(values).sum()),
            "squared_norm_share_total_post": float(np.square(values).sum() / total_energy),
            "sat_logit_coefficient": sat_coefficient,
            "sat_minus_unsat_coefficient": margin_coefficient,
            "bank_max_sat_direct_effect": float(values.max() * sat_coefficient),
        })
    return rows


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"exclusive output already exists: {OUTPUT}")
    native_path = HERE / "full_native_outputs.npz"
    with np.load(native_path, allow_pickle=False) as archive:
        post = archive["post"].astype(np.float64)
        coefficient = archive["unit_logit_coefficients"].astype(np.float64)
    with np.load(BANK, allow_pickle=False) as archive:
        formulas = archive["unique_clauses"].copy()
        orbit_ids = archive["orbit_ids"].copy()
        transform_unique_ids = archive["transform_unique_ids"].copy()

    counts = semantic_counts(formulas)
    labels = counts == 0
    active = np.flatnonzero(np.any(post > 0, axis=0))
    non_source_active = np.setdiff1d(active, SOURCE_REPORT)
    if len(active) != 70 or len(non_source_active) != 36:
        raise RuntimeError("active-set identity drift")

    singular_values = np.linalg.svd(post[:, active], full_matrices=False, compute_uv=False)
    default_tolerance = float(singular_values[0] * max(post[:, active].shape) * np.finfo(np.float64).eps)
    rank_rows = []
    for relative in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        tolerance = float(singular_values[0] * relative)
        rank_rows.append({"relative_to_largest": relative, "tolerance": tolerance,
                          "rank": int(np.count_nonzero(singular_values > tolerance))})

    source_state = post[:, SOURCE_REPORT]
    active_state = post[:, active]
    native_source = matching_record(source_state, labels)
    native_active = matching_record(active_state, labels)
    rng = np.random.default_rng(ROTATION_SEED + 1000)
    in_span = []
    active_span = []
    positive_dense = []
    positive_pair = []
    for _ in range(N_REFERENCES):
        in_span.append(matching_record(source_state @ random_orthogonal(34, rng), labels))
        active_span.append(matching_record(active_state @ random_orthogonal(70, rng), labels))
        positive_dense.append(matching_record(source_state @ positive_mixing(34, rng, None), labels))
        positive_pair.append(matching_record(source_state @ positive_mixing(34, rng, 2), labels))

    groups = fold_ids(orbit_ids)
    ridge_scores = cross_validated_ridge_scores(source_state, labels.astype(np.float64), groups)
    ridge_auc_matrix = two_sided_auc(ridge_scores, labels)
    ridge_auc = np.diag(ridge_auc_matrix)
    best_native_auc = two_sided_auc(source_state, labels).max(axis=1)
    probe_rows = []
    for assignment in range(32):
        probe_rows.append({
            "assignment": assignment,
            "assignment_name": assignment_name(assignment),
            "positive_count": int(labels[:, assignment].sum()),
            "best_native_auc": float(best_native_auc[assignment]),
            "cross_validated_ridge_auc": float(ridge_auc[assignment]),
            "native_to_probe_gap": float(best_native_auc[assignment] - ridge_auc[assignment]),
        })

    # A direct prevalence control: independently permute each assignment label,
    # preserving its exact positive count, and compare source axes with in-span rotations.
    shuffled = np.empty_like(labels)
    for column in range(32):
        shuffled[:, column] = labels[rng.permutation(len(labels)), column]
    shuffled_native = matching_record(source_state, shuffled)
    shuffled_refs = [matching_record(source_state @ random_orthogonal(34, rng), shuffled)
                     for _ in range(N_REFERENCES)]

    near_labels = counts == 1
    near_native = matching_record(source_state, near_labels)
    near_refs = [matching_record(source_state @ random_orthogonal(34, rng), near_labels)
                 for _ in range(N_REFERENCES)]

    base_scope = np.ones(len(post), dtype=bool)
    base_scope[np.unique(transform_unique_ids)] = False
    base_native = matching_record(source_state[base_scope], labels[base_scope])
    base_refs = [matching_record(source_state[base_scope] @ random_orthogonal(34, rng), labels[base_scope])
                 for _ in range(N_REFERENCES)]

    all_auc = two_sided_auc(post, labels)
    best_axes = all_auc.argmax(axis=1)
    best_rows = [{
        "assignment": index,
        "assignment_name": assignment_name(index),
        "unit": int(best_axes[index]),
        "auc": float(all_auc[index, best_axes[index]]),
        "in_source_report": bool(best_axes[index] in set(SOURCE_REPORT.tolist())),
        "published_singleton_unit": bool(best_axes[index] in set(UNITS.tolist())),
    } for index in range(32)]

    result = {
        "status": "PASS",
        "supersedes": {"path": str(SUPERSEDES), "sha256": sha256(SUPERSEDES),
                       "reason": "v1 indexed column zero instead of the diagonal of the label-by-score probe AUROC matrix; other sections are unaffected"},
        "scope": "retrospective zero-forward analysis of retained protected-study arrays",
        "protected_inference_unchanged": True,
        "inputs": {
            "native_outputs_sha256": sha256(native_path),
            "selected_bank_sha256": sha256(BANK),
            "source_notebook_sha256": "5410cc7cc28e28d303906cd8879bf95c161cda669c01d608014d7644bc742582",
        },
        "effective_dimensionality": {
            "rows": len(post),
            "units": post.shape[1],
            "active_units": active.tolist(),
            "active_unit_count": len(active),
            "source_report_units": SOURCE_REPORT.tolist(),
            "source_report_count": len(SOURCE_REPORT),
            "non_source_active_units": non_source_active.tolist(),
            "non_source_active_count": len(non_source_active),
            "singular_values": singular_values.tolist(),
            "numpy_default_tolerance": default_tolerance,
            "numpy_default_rank": int(np.count_nonzero(singular_values > default_tolerance)),
            "rank_sensitivity": rank_rows,
            "source_report_energy_share": float(np.square(source_state).sum() / np.square(post).sum()),
            "non_source_active_energy_share": float(np.square(post[:, non_source_active]).sum() / np.square(post).sum()),
            "per_active_unit": per_unit_summary(post, coefficient, active),
        },
        "reference_comparisons": {
            "source_native": native_source,
            "active_native": native_active,
            "haar_within_source_34d": compare_references(native_source, in_span),
            "haar_within_active_70d": compare_references(native_active, active_span),
            "positive_dense_mixtures_source_34d": compare_references(native_source, positive_dense),
            "positive_pair_mixtures_source_34d": compare_references(native_source, positive_pair),
            "references_per_family": N_REFERENCES,
            "inferential_status": "retrospective descriptive; reference families are not exchangeability nulls",
        },
        "basis_invariant_probe": {
            "method": "five-fold orbit-grouped ridge least-squares probe on the 34 source-report activations",
            "ridge": RIDGE,
            "folds": FOLDS,
            "per_assignment": probe_rows,
            "mean_best_native_auc": float(best_native_auc.mean()),
            "mean_cross_validated_ridge_auc": float(ridge_auc.mean()),
            "minimum_cross_validated_ridge_auc": float(ridge_auc.min()),
        },
        "prevalence_and_geometry_controls": {
            "prevalence_matched_permuted_labels": {
                "native": shuffled_native,
                "in_span_haar": compare_references(shuffled_native, shuffled_refs),
                "note": "each column preserves the exact assignment-label prevalence but destroys row semantics",
            },
            "one_violation_labels": {
                "native": near_native,
                "in_span_haar": compare_references(near_native, near_refs),
                "positive_counts": near_labels.sum(axis=0).astype(int).tolist(),
            },
        },
        "base_only_sensitivity": {
            "row_count": int(base_scope.sum()),
            "native": base_native,
            "in_span_haar": compare_references(base_native, base_refs),
        },
        "best_native_axis_per_assignment": best_rows,
        "published_singleton_ranks_among_512": [
            {
                "unit": int(unit),
                "assignment": int(assignment),
                "assignment_name": assignment_name(int(assignment)),
                "rank": int(1 + np.count_nonzero(all_auc[int(assignment)] > all_auc[int(assignment), int(unit)])),
                "auc": float(all_auc[int(assignment), int(unit)]),
                "best_unit": int(best_axes[int(assignment)]),
                "best_auc": float(all_auc[int(assignment), best_axes[int(assignment)]]),
            }
            for unit, assignment in zip(UNITS, ASSIGNMENTS)
        ],
        "limitations": [
            "retrospective analysis of one checkpoint and one engineered bank",
            "ridge scores are a cross-validated linear probe, not a causal direction",
            "positive mixtures preserve nonnegativity but not native sparsity or independence",
            "permuted labels preserve prevalence but not logical structure",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "PASS",
        "output": str(OUTPUT),
        "sha256": sha256(OUTPUT),
        "rank": result["effective_dimensionality"]["numpy_default_rank"],
        "source_energy_share": result["effective_dimensionality"]["source_report_energy_share"],
        "in_span": result["reference_comparisons"]["haar_within_source_34d"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
