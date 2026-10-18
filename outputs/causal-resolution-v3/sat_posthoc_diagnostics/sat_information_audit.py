#!/usr/bin/env python3
"""Exact finite-bank information audit for the SAT edit-response target.

Groups every legal edit by (report, slot, old clause, new clause) and counts
groups containing both SAT and UNSAT outcomes.  This uses retained semantic and
native arrays only; it performs no fit and no neural forward.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE
INPUTS = HERE / "inputs"
BANK_PATH = INPUTS / "pilot_bank.npz"
NATIVE_PATH = INPUTS / "original_native.npz"
ARCHIVED_PATH = INPUTS / "predictive_metrics.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_ids(array: np.ndarray) -> tuple[np.ndarray, int]:
    contiguous = np.ascontiguousarray(array)
    packed = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()
    _, inverse = np.unique(packed, return_inverse=True)
    return inverse.astype(np.int64), int(inverse.max()) + 1


def edit_rows(clauses: np.ndarray, post_sat: np.ndarray, origin_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reports = []
    actions = []
    outcomes = []
    for origin in origin_ids:
        for slot in range(10):
            old = int(clauses[origin, slot])
            new = np.arange(40, dtype=np.int64)
            keep = new != old
            new = new[keep]
            reports.append(np.full(39, origin, dtype=np.int64))
            actions.append((slot * 1600 + old * 40 + new).astype(np.int64))
            outcomes.append((~post_sat[origin, slot, keep]).astype(np.uint8))
    return np.concatenate(reports), np.concatenate(actions), np.concatenate(outcomes)


def audit(report_id_by_origin: np.ndarray, report_classes: int, origins: np.ndarray,
          row_origins: np.ndarray, actions: np.ndarray, outcomes: np.ndarray) -> dict:
    row_report_ids = report_id_by_origin[row_origins]
    keys = row_report_ids * 16_000 + actions
    _, inverse, group_sizes = np.unique(keys, return_inverse=True, return_counts=True)
    positive = np.bincount(inverse, weights=outcomes, minlength=len(group_sizes)).astype(np.int64)
    negative = group_sizes - positive
    conflicting = (positive > 0) & (negative > 0)
    non_singleton = group_sizes > 1
    unavoidable = np.minimum(positive, negative)
    conflicting_rows = int(group_sizes[conflicting].sum())
    non_singleton_rows = int(group_sizes[non_singleton].sum())
    return {
        "origins": int(len(origins)),
        "rows": int(len(outcomes)),
        "origin_report_classes": int(report_classes),
        "total_report_action_groups": int(len(group_sizes)),
        "conflicting_report_action_groups": int(conflicting.sum()),
        "conflicting_group_fraction": float(conflicting.mean()),
        "non_singleton_report_action_groups": int(non_singleton.sum()),
        "rows_in_non_singleton_groups": non_singleton_rows,
        "conflicting_fraction_among_non_singleton_groups": float(
            conflicting.sum() / non_singleton.sum()
        ) if non_singleton.any() else 0.0,
        "conflicting_row_fraction_within_non_singleton_rows": float(
            conflicting_rows / non_singleton_rows
        ) if non_singleton_rows else 0.0,
        "rows_in_conflicting_groups": conflicting_rows,
        "conflicting_row_fraction": float(conflicting_rows / len(outcomes)),
        "unavoidable_zero_one_errors": int(unavoidable.sum()),
        "minimum_empirical_zero_one_error": float(unavoidable.sum() / len(outcomes)),
        "maximum_absolute_information_error": 0.5 if conflicting.any() else 0.0,
        "scope": "exact unrestricted lookup on this finite bank; not held-out generalisation or a low-complexity decoder",
    }


def decompose_published_conflicts(
    published_ids: np.ndarray,
    truth_ids: np.ndarray,
    row_origins: np.ndarray,
    actions: np.ndarray,
    outcomes: np.ndarray,
) -> dict:
    """Refine each published-report/action class by the current truth table.

    Conflicts that remain after this refinement are target-specification
    conflicts: even current satisfiability semantics plus the action do not fix
    the response.  The reduction in minimum lookup error after refinement is
    the ambiguity attributable to information absent from the published bits.
    """
    published_keys = published_ids[row_origins] * 16_000 + actions
    _, published_inverse, published_sizes = np.unique(
        published_keys, return_inverse=True, return_counts=True
    )
    published_positive = np.bincount(
        published_inverse, weights=outcomes, minlength=len(published_sizes)
    ).astype(np.int64)
    published_negative = published_sizes - published_positive
    published_conflict = (published_positive > 0) & (published_negative > 0)
    published_error = np.minimum(published_positive, published_negative)

    refined_keys = (
        (published_ids[row_origins].astype(np.int64) * 4_096
         + truth_ids[row_origins].astype(np.int64)) * 16_000
        + actions
    )
    _, first, refined_inverse, refined_sizes = np.unique(
        refined_keys, return_index=True, return_inverse=True, return_counts=True
    )
    refined_positive = np.bincount(
        refined_inverse, weights=outcomes, minlength=len(refined_sizes)
    ).astype(np.int64)
    refined_negative = refined_sizes - refined_positive
    refined_conflict = (refined_positive > 0) & (refined_negative > 0)
    refined_error = np.minimum(refined_positive, refined_negative)
    refined_parent = published_inverse[first]

    parent_has_within_truth_conflict = np.zeros(len(published_sizes), dtype=bool)
    parent_has_within_truth_conflict[refined_parent[refined_conflict]] = True
    conflict_within_truth = published_conflict & parent_has_within_truth_conflict
    conflict_only_across_truth = published_conflict & ~parent_has_within_truth_conflict

    target_specification_errors = int(refined_error.sum())
    published_errors = int(published_error.sum())
    report_specific_errors = published_errors - target_specification_errors
    if report_specific_errors < 0:
        raise RuntimeError("truth-table refinement increased lookup error")
    return {
        "published_conflicting_groups": int(published_conflict.sum()),
        "published_conflicting_groups_with_within_truth_table_conflict": int(
            conflict_within_truth.sum()
        ),
        "published_conflicting_groups_only_across_truth_tables": int(
            conflict_only_across_truth.sum()
        ),
        "rows_in_groups_with_within_truth_table_conflict": int(
            published_sizes[conflict_within_truth].sum()
        ),
        "rows_in_groups_conflicting_only_across_truth_tables": int(
            published_sizes[conflict_only_across_truth].sum()
        ),
        "published_minimum_errors": published_errors,
        "errors_remaining_after_truth_table_refinement": target_specification_errors,
        "errors_removed_by_truth_table_refinement": report_specific_errors,
        "target_specification_error_share": float(
            target_specification_errors / published_errors
        ) if published_errors else 0.0,
        "report_specific_error_share": float(
            report_specific_errors / published_errors
        ) if published_errors else 0.0,
        "refined_conflicting_groups": int(refined_conflict.sum()),
        "scope": (
            "exact finite-bank decomposition obtained by refining each published-bit/action "
            "class with the complete current truth table"
        ),
    }


def main() -> None:
    bank = np.load(BANK_PATH, allow_pickle=False)
    native = np.load(NATIVE_PATH, allow_pickle=False)
    if not np.array_equal(bank["published_units"], native["units"]):
        raise RuntimeError("published-unit order mismatch")

    reports = {
        "context_only": np.zeros((4096, 1), dtype=np.uint8),
        "published_bits34": (native["post"] >= 0.5).astype(np.uint8),
        "truth_table32": bank["truth_tables"].astype(np.uint8),
        "continuous_post34_exact_float32": native["post"].astype("<f4", copy=False),
    }
    report_ids = {name: row_ids(value) for name, value in reports.items()}
    splits = {
        "entire_development_bank": np.arange(4096, dtype=np.int64),
        "assessment_origins_only": np.arange(3072, 4096, dtype=np.int64),
    }
    result = {
        "status": "PASS",
        "inputs": {
            str(BANK_PATH.relative_to(ROOT)): sha256(BANK_PATH),
            str(NATIVE_PATH.relative_to(ROOT)): sha256(NATIVE_PATH),
            str(ARCHIVED_PATH.relative_to(ROOT)): sha256(ARCHIVED_PATH),
        },
        "new_neural_forwards": 0,
        "fits": 0,
        "report_identity": {},
        "splits": {},
    }
    for name, (ids, classes) in report_ids.items():
        result["report_identity"][name] = {
            "distinct_origin_reports": classes,
            "origin_reports_unique": bool(classes == 4096),
        }

    for split_name, origins in splits.items():
        row_origins, actions, outcomes = edit_rows(bank["clauses"], bank["post_sat"], origins)
        result["splits"][split_name] = {}
        for report_name, (ids, classes) in report_ids.items():
            result["splits"][split_name][report_name] = audit(
                ids, classes, origins, row_origins, actions, outcomes
            )
        result["splits"][split_name]["published_bits_truth_table_decomposition"] = (
            decompose_published_conflicts(
                report_ids["published_bits34"][0],
                report_ids["truth_table32"][0],
                row_origins,
                actions,
                outcomes,
            )
        )

    archived = json.loads(ARCHIVED_PATH.read_text())
    expected = {
        "context_only": archived["models"]["context_only"]["finite_bank_information_bound"],
        "published_bits34": archived["models"]["published_bits34"]["finite_bank_information_bound"],
        "truth_table32": archived["models"]["truth_table32"]["finite_bank_information_bound"],
    }
    observed = result["splits"]["entire_development_bank"]
    for name, old in expected.items():
        new = observed[name]
        checks = {
            "total_report_action_groups": old["total_report_action_groups"],
            "conflicting_report_action_groups": old["conflicting_report_action_groups"],
            "unavoidable_zero_one_errors": old["unavoidable_errors"],
        }
        for field, value in checks.items():
            if new[field] != value:
                raise RuntimeError(f"archived information bound mismatch: {name} {field}")
        if not np.isclose(new["minimum_empirical_zero_one_error"], old["min_empirical_error"]):
            raise RuntimeError(f"archived empirical error mismatch: {name}")

    output_json = HERE / "sat_information_audit.json"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Exact finite-bank SAT information audit",
        "",
        "Every legal edit is grouped by the complete declared lookup input: report, slot, old clause and new clause. A conflicting group contains both SAT and UNSAT responses. Its unrestricted worst-case absolute error is therefore 0.5. The empirical zero-one column instead counts the minority outcome in each group.",
        "",
    ]
    for split_name in splits:
        lines.extend([
            f"## {split_name.replace('_', ' ').title()}",
            "",
            "| Report | Groups | Non-singleton groups | Conflicting / non-singleton | Rows in conflicts / non-singleton rows | Minimum empirical error |",
            "|:--|--:|--:|--:|--:|--:|",
        ])
        for report_name, values in result["splits"][split_name].items():
            if report_name == "published_bits_truth_table_decomposition":
                continue
            lines.append(
                f"| {report_name} | {values['total_report_action_groups']:,} | "
                f"{values['non_singleton_report_action_groups']:,} | "
                f"{values['conflicting_report_action_groups']:,}/{values['non_singleton_report_action_groups']:,} "
                f"({100*values['conflicting_fraction_among_non_singleton_groups']:.3f}%) | "
                f"{values['rows_in_conflicting_groups']:,}/{values['rows_in_non_singleton_groups']:,} "
                f"({100*values['conflicting_row_fraction_within_non_singleton_rows']:.3f}%) | "
                f"{100*values['minimum_empirical_zero_one_error']:.3f}% |"
            )
        decomposition = result["splits"][split_name]["published_bits_truth_table_decomposition"]
        lines.extend([
            "",
            "Published-bit conflict decomposition:",
            "",
            "| Quantity | Count |",
            "|:--|--:|",
            f"| Conflicting published-bit/action groups with a within-truth-table conflict | {decomposition['published_conflicting_groups_with_within_truth_table_conflict']:,} |",
            f"| Conflicting published-bit/action groups only across different truth tables | {decomposition['published_conflicting_groups_only_across_truth_tables']:,} |",
            f"| Minimum errors remaining after truth-table refinement | {decomposition['errors_remaining_after_truth_table_refinement']:,} ({100*decomposition['target_specification_error_share']:.3f}% of published-bit minimum errors) |",
            f"| Minimum errors removed by truth-table refinement | {decomposition['errors_removed_by_truth_table_refinement']:,} ({100*decomposition['report_specific_error_share']:.3f}% of published-bit minimum errors) |",
            "",
        ])
    lines.extend([
        "Exact continuous postactivation is collision-free on this finite bank because all 4,096 stored origin reports are distinct. This permits lookup memorisation; it is not evidence for a compact decoder or transfer.",
        "",
        "The archived context, published-bit and truth-table counts are reproduced exactly before this report is accepted. No model is fitted and no transformer forward is made.",
        "",
    ])
    (HERE / "sat_information_audit.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
