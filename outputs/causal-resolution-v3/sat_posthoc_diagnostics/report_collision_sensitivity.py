#!/usr/bin/env python3
"""Post hoc activity and bank-size sensitivity for SAT report collisions.

The script reads retained semantic and native arrays only. It performs no model
forward and fits no learner. Assessment origins are put in one deterministic,
outcome-blind SHA-256 order before forming nested bank-size prefixes.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE
INPUTS = HERE / "inputs"
BANK_PATH = INPUTS / "pilot_bank.npz"
NATIVE_PATH = INPUTS / "original_native.npz"
ASSESSMENT = np.arange(3072, 4096, dtype=np.int64)
SIZES = (128, 256, 512, 1024)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_ids(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    packed = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()
    return np.unique(packed, return_inverse=True)[1].astype(np.int64)


def edit_rows(clauses: np.ndarray, post_sat: np.ndarray, origins: np.ndarray):
    row_origins, actions, outcomes = [], [], []
    for origin in origins:
        for slot in range(10):
            old = int(clauses[origin, slot])
            new = np.arange(40, dtype=np.int64)
            keep = new != old
            new = new[keep]
            row_origins.append(np.full(39, origin, dtype=np.int64))
            actions.append((slot * 1600 + old * 40 + new).astype(np.int64))
            outcomes.append((~post_sat[origin, slot, keep]).astype(np.uint8))
    return np.concatenate(row_origins), np.concatenate(actions), np.concatenate(outcomes)


def decomposition(published_ids: np.ndarray, truth_ids: np.ndarray,
                  activity: np.ndarray, row_origins: np.ndarray,
                  actions: np.ndarray, outcomes: np.ndarray) -> dict:
    published_keys = published_ids[row_origins] * 16_000 + actions
    _, group_first, published_inverse, published_sizes = np.unique(
        published_keys, return_index=True, return_inverse=True, return_counts=True
    )
    published_positive = np.bincount(
        published_inverse, weights=outcomes, minlength=len(published_sizes)
    ).astype(np.int64)
    published_negative = published_sizes - published_positive
    published_error = np.minimum(published_positive, published_negative)
    published_conflict = published_error > 0
    non_singleton = published_sizes > 1
    group_activity = activity[row_origins[group_first]]

    refined_keys = (
        (published_ids[row_origins] * 4096 + truth_ids[row_origins]) * 16_000 + actions
    )
    _, refined_first, refined_inverse, refined_sizes = np.unique(
        refined_keys, return_index=True, return_inverse=True, return_counts=True
    )
    refined_positive = np.bincount(
        refined_inverse, weights=outcomes, minlength=len(refined_sizes)
    ).astype(np.int64)
    refined_error = np.minimum(refined_positive, refined_sizes - refined_positive)
    refined_parent = published_inverse[refined_first]
    refined_error_by_parent = np.bincount(
        refined_parent, weights=refined_error, minlength=len(published_sizes)
    ).astype(np.int64)
    removed_error = published_error - refined_error_by_parent
    if np.any(removed_error < 0):
        raise AssertionError("truth-table refinement increased a parent error")

    origin_activity_counts = Counter(map(int, activity[np.unique(row_origins)]))
    strata = {}
    for level in sorted(origin_activity_counts):
        mask = group_activity == level
        row_mask = activity[row_origins] == level
        group_rows = int(published_sizes[mask].sum())
        non_singleton_rows = int(published_sizes[mask & non_singleton].sum())
        strata[str(level)] = {
            "origins": int(origin_activity_counts[level]),
            "edit_rows": int(row_mask.sum()),
            "report_action_groups": int(mask.sum()),
            "non_singleton_groups": int((mask & non_singleton).sum()),
            "non_singleton_rows": non_singleton_rows,
            "conflicting_groups": int((mask & published_conflict).sum()),
            "rows_in_conflicting_groups": int(published_sizes[mask & published_conflict].sum()),
            "published_minimum_errors": int(published_error[mask].sum()),
            "logical_errors_remaining_after_truth_refinement": int(refined_error_by_parent[mask].sum()),
            "report_specific_errors_removed_by_truth_refinement": int(removed_error[mask].sum()),
            "report_specific_error_rate_all_rows": float(removed_error[mask].sum() / group_rows) if group_rows else 0.0,
            "conflicting_fraction_non_singleton_groups": float(
                (mask & published_conflict).sum() / (mask & non_singleton).sum()
            ) if (mask & non_singleton).any() else 0.0,
        }

    published_total = int(published_error.sum())
    logical_total = int(refined_error_by_parent.sum())
    removed_total = int(removed_error.sum())
    return {
        "origins": int(len(np.unique(row_origins))),
        "rows": int(len(outcomes)),
        "groups": int(len(published_sizes)),
        "non_singleton_groups": int(non_singleton.sum()),
        "conflicting_groups": int(published_conflict.sum()),
        "published_minimum_errors": published_total,
        "logical_errors_remaining_after_truth_refinement": logical_total,
        "report_specific_errors_removed_by_truth_refinement": removed_total,
        "logical_share_of_published_errors": float(logical_total / published_total) if published_total else 0.0,
        "report_specific_share_of_published_errors": float(removed_total / published_total) if published_total else 0.0,
        "published_minimum_error_rate": float(published_total / len(outcomes)),
        "logical_error_rate": float(logical_total / len(outcomes)),
        "report_specific_error_rate": float(removed_total / len(outcomes)),
        "activity_strata": strata,
    }


def main() -> None:
    with np.load(BANK_PATH, allow_pickle=False) as saved:
        clauses = saved["clauses"]
        post_sat = saved["post_sat"]
        truth_tables = saved["truth_tables"].astype(np.uint8)
        published_units = saved["published_units"]
    with np.load(NATIVE_PATH, allow_pickle=False) as saved:
        if not np.array_equal(published_units, saved["units"]):
            raise AssertionError("published-unit order")
        bits = (saved["post"] >= 0.5).astype(np.uint8)

    published_ids = row_ids(bits)
    truth_ids = row_ids(truth_tables)
    activity = bits.sum(axis=1).astype(np.int16)
    ordered = sorted(
        map(int, ASSESSMENT),
        key=lambda origin: (
            hashlib.sha256(
                b"sat-report-bank-size-v1\0" + np.asarray(clauses[origin], dtype="<i2").tobytes()
            ).digest(),
            origin,
        ),
    )
    ordered = np.array(ordered, dtype=np.int64)

    curves = {}
    for size in SIZES:
        origins = ordered[:size]
        rows = edit_rows(clauses, post_sat, origins)
        curves[str(size)] = decomposition(
            published_ids, truth_ids, activity, *rows
        )

    full = curves[str(max(SIZES))]
    result = {
        "status": "PASS",
        "scope": "post hoc exact finite-bank sensitivity of published-report collisions to report activity and nested assessment-bank size",
        "limits": [
            "Bank-size prefixes use one deterministic post hoc formula-hash order and are not confidence intervals",
            "Rates are occupancy-dependent properties of the retained assessment bank, not population estimates",
            "Truth-refinement residuals are logical target ambiguity; only removed errors diagnose information absent from the published report",
        ],
        "inputs": {
            str(BANK_PATH.relative_to(ROOT)): sha256(BANK_PATH),
            str(NATIVE_PATH.relative_to(ROOT)): sha256(NATIVE_PATH),
        },
        "ordering_domain": "sat-report-bank-size-v1",
        "assessment_order_sha256": hashlib.sha256(ordered.astype("<i8").tobytes()).hexdigest(),
        "new_neural_forwards": 0,
        "fits": 0,
        "sizes": curves,
    }
    (HERE / "report_collision_sensitivity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    lines = [
        "# Published-report collision sensitivity",
        "",
        "Assessment origins are placed in one deterministic SHA-256 order before nested prefixes are formed. These are post hoc occupancy diagnostics, not confidence intervals or population estimates.",
        "",
        "## Nested bank size",
        "",
        "| Origins | Non-singleton groups | Conflicting groups | Published minimum errors | Logical errors after truth refinement | Report-specific errors removed | Report-specific rate over all rows |",
        "|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for size in SIZES:
        row = curves[str(size)]
        lines.append(
            f"| {size:,} | {row['non_singleton_groups']:,} | {row['conflicting_groups']:,} | "
            f"{row['published_minimum_errors']:,} | {row['logical_errors_remaining_after_truth_refinement']:,} | "
            f"{row['report_specific_errors_removed_by_truth_refinement']:,} | {100*row['report_specific_error_rate']:.4f}% |"
        )
    lines.extend([
        "",
        "## Full assessment bank by active published bits",
        "",
        "| Active bits | Origins | Non-singleton groups | Conflicting groups | Published errors | Logical errors | Report-specific errors | Report-specific rate |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for level, row in full["activity_strata"].items():
        lines.append(
            f"| {level} | {row['origins']:,} | {row['non_singleton_groups']:,} | {row['conflicting_groups']:,} | "
            f"{row['published_minimum_errors']:,} | {row['logical_errors_remaining_after_truth_refinement']:,} | "
            f"{row['report_specific_errors_removed_by_truth_refinement']:,} | {100*row['report_specific_error_rate_all_rows']:.4f}% |"
        )
    lines.extend([
        "",
        "Errors remaining after truth-table refinement are properties of the edit target under the declared context. Errors removed by refinement measure information absent from the published bits on this finite bank.",
        "",
    ])
    (HERE / "report_collision_sensitivity.md").write_text("\n".join(lines))
    print(json.dumps({
        "status": result["status"],
        "sizes": {
            size: {
                "published_errors": curves[str(size)]["published_minimum_errors"],
                "logical_errors": curves[str(size)]["logical_errors_remaining_after_truth_refinement"],
                "report_specific_errors": curves[str(size)]["report_specific_errors_removed_by_truth_refinement"],
                "report_specific_rate": curves[str(size)]["report_specific_error_rate"],
            }
            for size in SIZES
        },
        "activity_levels": list(full["activity_strata"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
