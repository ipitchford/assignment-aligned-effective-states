#!/usr/bin/env python3
"""Retrospective zero-forward follow-up for the effective final-MLP state.

This script loads retained activations and the frozen checkpoint weights, performs
no model forward passes, and writes one exclusive JSON result.  It addresses
questions raised by the v4.4 technical review without changing any protected
decision or reference ensemble.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import auc_matrix, basis_statistics  # noqa: E402
from protocol import (  # noqa: E402
    ASSIGNMENTS,
    BANK,
    CHECKPOINT,
    ROTATION_SEED,
    RUNTIME,
    SAT_TOKEN,
    SOURCE_HASHES,
    SOURCE_THRESHOLD,
    UNSAT_TOKEN,
    UNITS,
    semantic_counts,
)


OUTPUT = HERE / "effective_state_followup_v4.json"
SUPERSEDES = HERE / "effective_state_followup_v3.json"
N_REFERENCES = 63
FOLDS = 5
SOURCE_REPORT = np.array([
    10, 29, 36, 41, 48, 55, 77, 81, 96, 150, 185, 189, 195, 201,
    224, 231, 245, 261, 291, 304, 317, 326, 334, 374, 380, 411, 416,
    435, 450, 482, 490, 492, 495, 499,
], dtype=np.int32)


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


def random_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    gaussian = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def matching(scores: np.ndarray, labels: np.ndarray) -> dict:
    stats = basis_statistics(two_sided_auc(scores, labels))
    return {
        "matching_32_mean": float(stats["matching_32"]["mean"]),
        "matching_32_minimum": float(stats["matching_32"]["minimum"]),
        "matching_uncovered16_mean": float(stats["matching_uncovered16"]["mean"]),
        "matching_uncovered16_minimum": float(stats["matching_uncovered16"]["minimum"]),
    }


def summarise_reference(native: dict, refs: list[dict]) -> dict:
    result = {}
    for key in native:
        values = np.asarray([row[key] for row in refs], dtype=np.float64)
        result[key] = {
            "native": native[key],
            "reference_mean": float(values.mean()),
            "reference_min": float(values.min()),
            "reference_max": float(values.max()),
            "reference_exceedance_count": int(np.count_nonzero(values >= native[key])),
            "native_minus_reference_max": float(native[key] - values.max()),
        }
    return result


def fold_ids(orbit_ids: np.ndarray) -> np.ndarray:
    result = np.empty(len(orbit_ids), dtype=np.int8)
    for index, raw in enumerate(orbit_ids):
        digest = hashlib.sha256(bytes(raw)).digest()
        result[index] = int.from_bytes(digest[:8], "little") % FOLDS
    return result


def load_model_weights() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = importlib.util.spec_from_file_location("sat_followup_runtime", RUNTIME)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, config, receipt = module.load_model(threads=1)
    if receipt.get("checkpoint_sha256") != SOURCE_HASHES[CHECKPOINT]:
        raise RuntimeError("checkpoint identity drift")
    if config.d_mlp != 512 or config.d_model != 128:
        raise RuntimeError("model shape drift")
    final = model.blocks[-1].mlp
    w_in = final.W_in.detach().cpu().numpy().copy()
    b_in = final.b_in.detach().cpu().numpy().copy()
    w_out = final.W_out.detach().cpu().numpy().copy()
    if w_in.shape != (512, 128) or b_in.shape != (512,) or w_out.shape != (128, 512):
        raise RuntimeError(f"unexpected MLP shapes: {w_in.shape}, {b_in.shape}, {w_out.shape}")
    return w_in, b_in, w_out


def weight_summary(w_in: np.ndarray, b_in: np.ndarray, w_out: np.ndarray) -> dict:
    source = np.zeros(512, dtype=bool)
    source[SOURCE_REPORT] = True
    in_norm = np.linalg.norm(w_in, axis=1)
    out_norm = np.linalg.norm(w_out, axis=0)
    scale = np.maximum.reduce([in_norm, np.abs(b_in), out_norm])
    outside = ~source
    outside_units = np.flatnonzero(outside)
    order = outside_units[np.argsort(-scale[outside_units], kind="stable")]
    return {
        "definition": "per-unit maximum of input-column norm, absolute input bias, and output-column norm",
        "source_minimum": float(scale[source].min()),
        "source_maximum": float(scale[source].max()),
        "outside_maximum": float(scale[outside].max()),
        "outside_median": float(np.median(scale[outside])),
        "outside_above_1e_minus_12": int(np.count_nonzero(scale[outside] > 1e-12)),
        "outside_above_1e_minus_8": int(np.count_nonzero(scale[outside] > 1e-8)),
        "ratio_outside_max_to_source_min": float(scale[outside].max() / scale[source].min()),
        "largest_outside_units": [
            {
                "unit": int(unit),
                "W_in_row_norm": float(in_norm[unit]),
                "abs_b_in": float(abs(b_in[unit])),
                "W_out_column_norm": float(out_norm[unit]),
                "maximum_component_norm": float(scale[unit]),
            }
            for unit in order[:10]
        ],
        "per_component": {
            "W_in_column_norm": {
                "source_minimum": float(in_norm[source].min()),
                "outside_maximum": float(in_norm[outside].max()),
            },
            "abs_b_in": {
                "source_minimum": float(np.abs(b_in[source]).min()),
                "outside_maximum": float(np.abs(b_in[outside]).max()),
            },
            "W_out_column_norm": {
                "source_minimum": float(out_norm[source].min()),
                "outside_maximum": float(out_norm[outside].max()),
            },
        },
    }


def stratum_masks(counts: np.ndarray) -> dict[str, np.ndarray]:
    solution_count = np.sum(counts == 0, axis=1)
    return {
        "all": np.ones(len(counts), dtype=bool),
        "unsat": solution_count == 0,
        "sole_solution": solution_count == 1,
        "multi_solution": solution_count > 1,
    }


def agreement_rows(predicted: np.ndarray, native: np.ndarray, truth: np.ndarray,
                   masks: dict[str, np.ndarray]) -> dict:
    result = {}
    for name, mask in masks.items():
        result[name] = {
            "rows": int(mask.sum()),
            "argmax_agreement_with_native": float(np.mean(predicted[mask] == native[mask])),
            "task_accuracy": float(np.mean(predicted[mask] == truth[mask])),
            "native_task_accuracy": float(np.mean(native[mask] == truth[mask])),
        }
    return result


def threshold_substitution(post: np.ndarray, resid_mid: np.ndarray, coefficient: np.ndarray,
                           bias_logits: np.ndarray, w_u: np.ndarray, logits: np.ndarray,
                           truth: np.ndarray, groups: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    native_pred = logits.argmax(axis=1)
    substituted = np.empty((len(post), logits.shape[1]), dtype=np.float64)
    source_post = post[:, SOURCE_REPORT]
    bits = source_post >= SOURCE_THRESHOLD
    kappas = np.empty((FOLDS, len(SOURCE_REPORT)), dtype=np.float64)
    base_logits = resid_mid.astype(np.float64) @ w_u.astype(np.float64) + bias_logits
    source_coeff = coefficient[SOURCE_REPORT]
    for fold in range(FOLDS):
        train = groups != fold
        test = groups == fold
        for column in range(len(SOURCE_REPORT)):
            selected = source_post[train, column][bits[train, column]]
            kappas[fold, column] = float(np.median(selected)) if len(selected) else 0.0
        reconstructed = bits[test].astype(np.float64) * kappas[fold]
        substituted[test] = base_logits[test] + reconstructed @ source_coeff
    predicted = substituted.argmax(axis=1)
    return {
        "method": "five-fold orbit-grouped, per-unit median supra-threshold activation fitted outside each fold",
        "threshold": SOURCE_THRESHOLD,
        "folds": FOLDS,
        "kappa_minimum": float(kappas.min()),
        "kappa_maximum": float(kappas.max()),
        "overall_argmax_agreement_with_native": float(np.mean(predicted == native_pred)),
        "overall_task_accuracy": float(np.mean(predicted == truth)),
        "native_task_accuracy": float(np.mean(native_pred == truth)),
        "by_stratum": agreement_rows(predicted, native_pred, truth, masks),
    }


def task_error_localisation(logits: np.ndarray, counts: np.ndarray) -> dict:
    truth = np.where(np.any(counts == 0, axis=1), SAT_TOKEN, UNSAT_TOKEN)
    predicted = logits.argmax(axis=1)
    solution_count = np.sum(counts == 0, axis=1)
    sole = solution_count == 1
    errors = sole & (predicted != truth)
    sole_assignment = np.argmax(counts == 0, axis=1)
    per_assignment = []
    for assignment in range(32):
        mask = sole & (sole_assignment == assignment)
        per_assignment.append({
            "assignment": assignment,
            "assignment_name": assignment_name(assignment),
            "sole_rows": int(mask.sum()),
            "errors": int(np.count_nonzero(errors & (sole_assignment == assignment))),
        })
    weak = {8, 20, 29}  # FTFFF, TFTFF, TTTFT
    return {
        "total_sole_errors": int(errors.sum()),
        "errors_on_three_weak_single_axis_assignments": int(sum(row["errors"] for row in per_assignment if row["assignment"] in weak)),
        "three_weak_single_axis_assignments": [assignment_name(i) for i in sorted(weak)],
        "per_assignment": per_assignment,
    }


def all_unit_competition(source_post: np.ndarray, labels: np.ndarray) -> dict:
    auc = two_sided_auc(source_post, labels)  # assignment x unit
    rows = []
    for column, unit in enumerate(SOURCE_REPORT):
        order = np.argsort(-auc[:, column], kind="stable")
        rows.append({
            "unit": int(unit),
            "source_singleton_unit": bool(unit in set(UNITS.tolist())),
            "best_assignment": int(order[0]),
            "best_assignment_name": assignment_name(int(order[0])),
            "best_auc": float(auc[order[0], column]),
            "runner_up_assignment": int(order[1]),
            "runner_up_assignment_name": assignment_name(int(order[1])),
            "runner_up_auc": float(auc[order[1], column]),
            "margin": float(auc[order[0], column] - auc[order[1], column]),
        })
    return {
        "score": "post-ReLU two-sided AUROC on all 7,252 bank rows",
        "units": rows,
        "distinct_best_assignments": len(set(row["best_assignment"] for row in rows)),
        "non_singleton_source_units_best_matched_to_one_assignment": int(sum(
            (not row["source_singleton_unit"]) and row["best_auc"] >= 0.9 for row in rows
        )),
    }


def standardised_references(source_post: np.ndarray, labels: np.ndarray) -> dict:
    mean = source_post.mean(axis=0)
    scale = source_post.std(axis=0)
    if np.any(scale <= 0):
        raise RuntimeError("zero-variance source coordinate")
    standardised = (source_post - mean) / scale
    native = matching(standardised, labels)
    rng = np.random.default_rng(ROTATION_SEED + 3000)
    refs = [matching(standardised @ random_orthogonal(34, rng), labels)
            for _ in range(N_REFERENCES)]
    return {
        "method": "column-centre and scale to unit variance, then apply fixed-seed 34-dimensional Haar rotations",
        "references": N_REFERENCES,
        "comparison": summarise_reference(native, refs),
        "status": "retrospective descriptive sensitivity, not an exchangeability null",
    }


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"exclusive output already exists: {OUTPUT}")
    native_path = HERE / "full_native_outputs.npz"
    with np.load(native_path, allow_pickle=False) as archive:
        post = archive["post"].astype(np.float64)
        resid_mid = archive["resid_mid"].astype(np.float64)
        logits = archive["logits"].astype(np.float64)
        coefficient = archive["unit_logit_coefficients"].astype(np.float64)
        bias_logits = archive["mlp_bias_logits"].astype(np.float64)
        w_u = archive["W_U"].astype(np.float64)
    with np.load(BANK, allow_pickle=False) as archive:
        formulas = archive["unique_clauses"].copy()
        orbit_ids = archive["orbit_ids"].copy()
    counts = semantic_counts(formulas)
    labels = counts == 0
    truth = np.where(np.any(labels, axis=1), SAT_TOKEN, UNSAT_TOKEN)
    groups = fold_ids(orbit_ids)
    masks = stratum_masks(counts)
    w_in, b_in, w_out = load_model_weights()
    result = {
        "status": "PASS",
        "scope": "retrospective zero-forward analysis; protected results unchanged",
        "supersedes": {
            "path": str(SUPERSEDES),
            "sha256": sha256(SUPERSEDES),
            "reason": "v4 adds unit identities and component norms for the largest non-report weight outliers; all v3 calculations are unchanged",
        },
        "inputs": {
            "native_outputs_sha256": sha256(native_path),
            "selected_bank_sha256": sha256(BANK),
            "checkpoint_sha256": SOURCE_HASHES[CHECKPOINT],
        },
        "weight_level_effective_width": weight_summary(w_in, b_in, w_out),
        "threshold_bit_substitution": threshold_substitution(
            post, resid_mid, coefficient, bias_logits, w_u, logits, truth, groups, masks
        ),
        "task_error_localisation": task_error_localisation(logits, counts),
        "all_34_unit_assignment_competition": all_unit_competition(post[:, SOURCE_REPORT], labels),
        "standardised_in_span_references": standardised_references(post[:, SOURCE_REPORT], labels),
        "limitations": [
            "one checkpoint and one engineered bank",
            "threshold substitution is a cross-validated diagnostic, not a trained causal abstraction",
            "standardised rotations are constructed references, not a generative null",
            "weight norms establish parameter decay but do not explain its developmental cause",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(OUTPUT),
        "sha256": sha256(OUTPUT),
        "weight_level_effective_width": result["weight_level_effective_width"],
        "threshold_bit_substitution": result["threshold_bit_substitution"],
        "task_error_localisation": result["task_error_localisation"],
        "standardised_in_span_references": result["standardised_in_span_references"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
