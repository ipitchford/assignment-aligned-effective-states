#!/usr/bin/env python3
"""Zero-forward calibration audit for the protected SAT renaming result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT / "outputs" / "causal-resolution-v3" / "sat_renaming_case" / "work" / "sat_renaming_launch_recovery"
NATIVE_PATH = STUDY / "native_outputs.npz"
BANK_PATH = STUDY / "selected_bank.npz"
CLAMP_PATH = STUDY / "affine_clamps.npz"
UNITS = np.array([261, 291, 411, 492], dtype=np.int64)
SOURCE_PAIRS = ((261, 291), (411, 492))
STATE_PAIRS = ((0, 1), (2, 3))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparison = positive[:, None] - negative[None, :]
    return float(((comparison > 0).sum() + 0.5 * (comparison == 0).sum()) / comparison.size)


def cross_tab(left: np.ndarray, right: np.ndarray) -> dict:
    return {
        "left_false_right_false": int((~left & ~right).sum()),
        "left_false_right_true": int((~left & right).sum()),
        "left_true_right_false": int((left & ~right).sum()),
        "left_true_right_true": int((left & right).sum()),
    }


def main() -> None:
    native = np.load(NATIVE_PATH, allow_pickle=False)
    bank = np.load(BANK_PATH, allow_pickle=False)
    clamp = np.load(CLAMP_PATH, allow_pickle=False)
    if not np.array_equal(native["units"], UNITS):
        raise RuntimeError("unit order mismatch")

    directions = {}
    endpoint_rows = []
    for pair_index, source_pair in enumerate(SOURCE_PAIRS):
        blocks = np.flatnonzero(bank["pair_index"] == pair_index)
        for direction, (_, after) in enumerate(STATE_PAIRS):
            local_unit = 2 * pair_index + direction
            unit = int(source_pair[direction])
            positive = native["pre"][blocks, 0, after, local_unit].astype(np.float64)
            all_scores = native["pre"][blocks, :, :, local_unit].astype(np.float64)
            expected = bank["expected_unit_bits"][blocks, :, :, local_unit].astype(bool)
            if int(expected.sum()) != 32:
                raise RuntimeError(f"unexpected positive denominator for unit {unit}")
            negative = all_scores[~expected]
            if len(negative) != 224:
                raise RuntimeError(f"unexpected negative denominator for unit {unit}")

            min_positive = float(positive.min())
            max_negative = float(negative.max())
            full_sensitivity_threshold = min_positive
            false_positives_at_full_sensitivity = int((negative >= full_sensitivity_threshold).sum())
            zero_fp_threshold = float(np.nextafter(max_negative, np.inf))
            sensitivity_at_zero_fp = int((positive >= zero_fp_threshold).sum())
            separation_exists = bool(min_positive > max_negative)

            detected = positive >= 0.5
            endpoint_logits = native["logits"][blocks, 0, after]
            endpoint_labels = bank["labels"][blocks, 0, after]
            initially_correct = endpoint_logits.argmax(axis=-1) == endpoint_labels
            restored_class = clamp["nominal_class"][blocks, 0, direction, 0]
            correct_to_wrong = initially_correct & (restored_class != endpoint_labels)
            key = f"pair_{pair_index}_{'original' if direction == 0 else 'renamed'}_unit_{unit}"
            directions[key] = {
                "source_pair": list(source_pair),
                "direction": "original" if direction == 0 else "renamed",
                "unit": unit,
                "positive_state": "edited" if direction == 0 else "renamed_edited",
                "positive_preactivation": distribution(positive),
                "negative_preactivation": distribution(negative),
                "threshold_free_auc": pairwise_auc(positive, negative),
                "source_threshold": 0.5,
                "source_threshold_true_positives": int(detected.sum()),
                "source_threshold_false_positives": int((negative >= 0.5).sum()),
                "threshold_for_32_of_32_sensitivity": full_sensitivity_threshold,
                "false_positives_at_32_of_32_sensitivity": false_positives_at_full_sensitivity,
                "smallest_float64_threshold_above_maximum_negative": zero_fp_threshold,
                "true_positives_at_zero_false_positives": sensitivity_at_zero_fp,
                "perfect_threshold_separation_exists": separation_exists,
                "initially_correct_endpoints": int(initially_correct.sum()),
                "origin_restoration_correct_to_wrong": int(correct_to_wrong.sum()),
                "source_detection_by_origin_restoration_correct_to_wrong": cross_tab(detected, correct_to_wrong),
            }
            endpoint_rows.append((pair_index, direction, unit, detected, correct_to_wrong))

    p2_rows = [row for row in endpoint_rows if row[0] == 1]
    p2_detected = np.concatenate([row[3] for row in p2_rows])
    p2_c2w = np.concatenate([row[4] for row in p2_rows])

    origin_nonzero = {}
    activation_change_controls = {}
    for pair_index, source_pair in enumerate(SOURCE_PAIRS):
        blocks = np.flatnonzero(bank["pair_index"] == pair_index)
        primary = np.array([2 * pair_index, 2 * pair_index + 1])
        off = np.array([u for u in range(4) if u not in primary])
        primary_origin_nonzero = []
        negative_primary_change = []
        positive_off_change = []
        for direction, (before, after) in enumerate(STATE_PAIRS):
            primary_origin_nonzero.append(
                np.any(native["post"][blocks, 0, before][:, primary] != 0, axis=1)
            )
            negative_primary_change.append(
                np.any(
                    native["post"][blocks, 1, after][:, primary]
                    != native["post"][blocks, 1, before][:, primary],
                    axis=1,
                )
            )
            positive_off_change.append(
                np.any(
                    native["post"][blocks, 0, after][:, off]
                    != native["post"][blocks, 0, before][:, off],
                    axis=1,
                )
            )
        origin_nonzero[f"pair_{pair_index}"] = {
            "source_units": list(source_pair),
            "positive_endpoints": 64,
            "endpoints_with_any_nonzero_origin_pair_value": int(
                np.concatenate(primary_origin_nonzero).sum()
            ),
        }
        activation_change_controls[f"pair_{pair_index}"] = {
            "negative_endpoints_with_any_nominated_pair_activation_change": int(
                np.concatenate(negative_primary_change).sum()
            ),
            "negative_endpoints": 64,
            "positive_endpoints_with_any_off_pair_activation_change": int(
                np.concatenate(positive_off_change).sum()
            ),
            "positive_endpoints": 64,
        }

    result = {
        "status": "PASS",
        "scope": "post hoc zero-forward calibration audit of the protected renaming arrays",
        "new_neural_forwards": 0,
        "new_fits": 0,
        "inputs": {
            str(NATIVE_PATH.relative_to(ROOT)): sha256(NATIVE_PATH),
            str(BANK_PATH.relative_to(ROOT)): sha256(BANK_PATH),
            str(CLAMP_PATH.relative_to(ROOT)): sha256(CLAMP_PATH),
        },
        "directions": directions,
        "pair_411_492_combined_detection_by_origin_restoration_correct_to_wrong": cross_tab(
            p2_detected, p2_c2w
        ),
        "origin_values": origin_nonzero,
        "near_automatic_control_activity": activation_change_controls,
    }
    (HERE / "renaming_threshold_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    lines = [
        "# Renaming threshold and origin-restoration audit",
        "",
        "This post hoc audit uses the saved protected-study arrays and performs zero neural forwards and zero fits.",
        "",
        "| Unit and direction | Source-threshold TP/32 | FP/224 | AUROC | Minimum positive | Maximum negative | TP at zero FP | FP at 32/32 TP |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for item in directions.values():
        lines.append(
            f"| {item['unit']} ({item['direction']}) | {item['source_threshold_true_positives']}/32 | "
            f"{item['source_threshold_false_positives']}/224 | {item['threshold_free_auc']:.6f} | "
            f"{item['positive_preactivation']['minimum']:.6f} | {item['negative_preactivation']['maximum']:.6f} | "
            f"{item['true_positives_at_zero_false_positives']}/32 | "
            f"{item['false_positives_at_32_of_32_sensitivity']}/224 |"
        )
    lines.extend([
        "",
        "The 411/492 cross-tab compares source-threshold detection with whether origin restoration changes an initially correct positive endpoint to wrong:",
        "",
        "```json",
        json.dumps(result["pair_411_492_combined_detection_by_origin_restoration_correct_to_wrong"], indent=2, sort_keys=True),
        "```",
        "",
        "The intervention restores the nominated pair from each edited endpoint to its corresponding origin values. It is not pure zeroing unless those origin values are both zero. Counts of nonzero origin values and nontrivial control activation changes are retained in the JSON receipt.",
        "",
    ])
    (HERE / "renaming_threshold_audit.md").write_text("\n".join(lines))
    print(json.dumps({
        "status": "PASS",
        "directions": {key: {
            "tp": value["source_threshold_true_positives"],
            "auc": value["threshold_free_auc"],
            "tp_at_zero_fp": value["true_positives_at_zero_false_positives"],
            "fp_at_full_sensitivity": value["false_positives_at_32_of_32_sensitivity"],
        } for key, value in directions.items()},
        "pair_411_492_cross_tab": result["pair_411_492_combined_detection_by_origin_restoration_correct_to_wrong"],
        "origin_values": origin_nonzero,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
