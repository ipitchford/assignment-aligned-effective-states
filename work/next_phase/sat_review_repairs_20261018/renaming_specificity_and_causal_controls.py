#!/usr/bin/env python3
"""Post hoc specificity strata, activity-matched nulls, and patch-in controls.

All calculations use saved SAT-renaming tensors. The script makes no neural
forward and fits no learner. Activity-matched controls are selected from
activation-change magnitudes alone, before their output effects are evaluated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT / "outputs" / "causal-resolution-v3" / "sat_renaming_case" / "work" / "sat_renaming_launch_recovery"
MAPPING_PATH = ROOT / "work" / "sat_mechanism" / "source_mapping.json"
NATIVE_PATH = STUDY / "native_outputs.npz"
BANK_PATH = STUDY / "selected_bank.npz"
WEIGHTS_PATH = STUDY / "readout_weights.npz"
SOURCE_PAIRS = ((261, 291), (411, 492))
STATE_PAIRS = ((0, 1), (2, 3))
STATE_NAMES = ("original", "edited", "renamed_original", "renamed_edited")
MATCHED_CONTROLS = 32
MATCH_DISTANCE_THRESHOLD = 0.25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(solution_ids: np.ndarray, labelled_assignment: int) -> str:
    if len(solution_ids) == 0:
        return "unsat"
    if len(solution_ids) == 1 and int(solution_ids[0]) != labelled_assignment:
        return "sat_different_sole_solution"
    if labelled_assignment not in set(map(int, solution_ids)):
        return "sat_multi_solution_excluding_label"
    raise AssertionError("negative target bit on an input containing its labelled assignment")


def class_effects(delta: np.ndarray, logits_before: np.ndarray, logits_after: np.ndarray,
                  labels_before: np.ndarray, labels_after: np.ndarray,
                  coefficients: np.ndarray) -> dict[str, np.ndarray]:
    """Return restoration and patch-in outcomes for every unit."""
    native_before = logits_before.argmax(axis=1)
    native_after = logits_after.argmax(axis=1)
    restoration_c2w = np.empty(512, dtype=np.int16)
    restoration_changes = np.empty(512, dtype=np.int16)
    patchin_sat = np.empty(512, dtype=np.int16)
    patchin_changes = np.empty(512, dtype=np.int16)
    patchin_correct = np.empty(512, dtype=np.int16)
    for unit in range(512):
        change = delta[:, unit, None] * coefficients[unit][None, :]
        restored = (logits_after - change).argmax(axis=1)
        patched = (logits_before + change).argmax(axis=1)
        restoration_c2w[unit] = ((native_after == labels_after) & (restored != labels_after)).sum()
        restoration_changes[unit] = (restored != native_after).sum()
        patchin_sat[unit] = ((native_before == 14) & (patched == 13)).sum()
        patchin_changes[unit] = (patched != native_before).sum()
        patchin_correct[unit] = (patched == labels_after).sum()
    return {
        "restoration_correct_to_wrong": restoration_c2w,
        "restoration_class_changes": restoration_changes,
        "patchin_unsat_to_sat": patchin_sat,
        "patchin_class_changes": patchin_changes,
        "patchin_matches_edited_label": patchin_correct,
        "native_before_unsat": np.array(int((native_before == 14).sum())),
        "native_after_correct": np.array(int((native_after == labels_after).sum())),
    }


def activity_matched_controls(delta: np.ndarray, source_unit: int,
                              excluded: set[int]) -> tuple[np.ndarray, np.ndarray]:
    source_profile = np.sort(np.abs(delta[:, source_unit]))
    denominator = max(float(np.linalg.norm(source_profile)), 1e-12)
    profiles = np.sort(np.abs(delta), axis=0)
    distances = np.linalg.norm(profiles - source_profile[:, None], axis=0) / denominator
    distances[list(excluded)] = np.inf
    order = np.argsort(distances, kind="stable")
    selected = order[:MATCHED_CONTROLS].astype(np.int16)
    if not np.isfinite(distances[selected]).all():
        raise AssertionError("insufficient activity-matched controls")
    return selected, distances


def metric_summary(values: np.ndarray, source_unit: int, controls: np.ndarray,
                   qualified: np.ndarray) -> dict:
    control_values = values[controls]
    source_value = int(values[source_unit])
    return {
        "source": source_value,
        "nearest_32_control_min": int(control_values.min()),
        "nearest_32_control_median": float(np.median(control_values)),
        "nearest_32_control_max": int(control_values.max()),
        "source_rank_among_source_plus_32_nearest_best_is_one": int((control_values > source_value).sum()) + 1,
        "nearest_32_controls_tied_with_source": int((control_values == source_value).sum()),
        "qualified_control_count": int(len(qualified)),
        "qualified_control_values": values[qualified].astype(int).tolist(),
        "qualified_control_max": int(values[qualified].max()) if len(qualified) else None,
    }


def main() -> None:
    with np.load(NATIVE_PATH, allow_pickle=False) as saved:
        native = {key: saved[key] for key in saved.files}
    with np.load(BANK_PATH, allow_pickle=False) as saved:
        bank = {key: saved[key] for key in saved.files}
    with np.load(WEIGHTS_PATH, allow_pickle=False) as saved:
        coefficients = saved["W_out"].astype(np.float64).T @ saved["W_U"].astype(np.float64)
    mapping = json.loads(MAPPING_PATH.read_text())
    rule_by_unit = {int(row["unit"]): row for row in mapping["rules"]}
    assignment_by_unit = {
        int(row["unit"]): int(row["truth_table_index"])
        for row in mapping["single_assignment_general_units"]
    }
    bits = native["post"] >= 0.5

    specificity = {}
    for pair_index, source_pair in enumerate(SOURCE_PAIRS):
        blocks = np.flatnonzero(bank["pair_index"] == pair_index)
        primary_positions = (2 * pair_index, 2 * pair_index + 1)
        off_positions = tuple(index for index in range(4) if index not in primary_positions)
        for direction, states in enumerate(STATE_PAIRS):
            counts = {}
            for block in blocks:
                for cohort in range(2):
                    for state in states:
                        solution_ids = np.flatnonzero(bank["counts"][block, cohort, state] == 0)
                        for position, unit in zip(primary_positions, source_pair):
                            expected = int(bank["expected_unit_bits"][block, cohort, state, position])
                            if expected:
                                continue
                            name = category(solution_ids, assignment_by_unit[unit])
                            entry = counts.setdefault(name, {"true_negative": 0, "false_positive": 0})
                            entry["false_positive" if bits[block, cohort, state, position] else "true_negative"] += 1
            for name in ("unsat", "sat_different_sole_solution", "sat_multi_solution_excluding_label"):
                counts.setdefault(name, {"true_negative": 0, "false_positive": 0})
                counts[name]["denominator"] = counts[name]["true_negative"] + counts[name]["false_positive"]
            key = f"pair_{pair_index}_{'original' if direction == 0 else 'renamed'}"
            specificity[key] = {
                "source_pair": list(source_pair),
                "direction": "original" if direction == 0 else "renamed",
                "negative_target_bits": counts,
            }

        off_counts = {}
        for block in blocks:
            for cohort in range(2):
                for state in range(4):
                    solution_ids = np.flatnonzero(bank["counts"][block, cohort, state] == 0)
                    for position in off_positions:
                        unit = int(native["units"][position])
                        name = category(solution_ids, assignment_by_unit[unit])
                        entry = off_counts.setdefault(name, {"true_negative": 0, "false_positive": 0})
                        entry["false_positive" if bits[block, cohort, state, position] else "true_negative"] += 1
        for name in ("unsat", "sat_different_sole_solution", "sat_multi_solution_excluding_label"):
            off_counts.setdefault(name, {"true_negative": 0, "false_positive": 0})
            off_counts[name]["denominator"] = off_counts[name]["true_negative"] + off_counts[name]["false_positive"]
        specificity[f"pair_{pair_index}_off_pair"] = {
            "source_pair": list(source_pair),
            "negative_target_bits": off_counts,
        }

    causal = {}
    arrays = {}
    for pair_index, source_pair in enumerate(SOURCE_PAIRS):
        blocks = np.flatnonzero(bank["pair_index"] == pair_index)
        for direction, (before, after) in enumerate(STATE_PAIRS):
            source_unit = source_pair[direction]
            delta = (
                native["full_post"][blocks, 0, after].astype(np.float64)
                - native["full_post"][blocks, 0, before].astype(np.float64)
            )
            logits_before = native["logits"][blocks, 0, before].astype(np.float64)
            logits_after = native["logits"][blocks, 0, after].astype(np.float64)
            labels_before = bank["labels"][blocks, 0, before].astype(np.int64)
            labels_after = bank["labels"][blocks, 0, after].astype(np.int64)
            effects = class_effects(
                delta, logits_before, logits_after, labels_before, labels_after, coefficients
            )
            controls, distances = activity_matched_controls(delta, source_unit, set(source_pair))
            qualified = controls[distances[controls] <= MATCH_DISTANCE_THRESHOLD]
            name = f"pair_{pair_index}_{'original' if direction == 0 else 'renamed'}"
            causal[name] = {
                "source_pair": list(source_pair),
                "direction": "original" if direction == 0 else "renamed",
                "source_unit": source_unit,
                "endpoints": 32,
                "native_before_unsat": int(effects["native_before_unsat"]),
                "native_after_correct": int(effects["native_after_correct"]),
                "source_activation_change": {
                    "mean_absolute": float(np.mean(np.abs(delta[:, source_unit]))),
                    "rms": float(np.sqrt(np.mean(delta[:, source_unit] ** 2))),
                    "nonzero": int(np.count_nonzero(delta[:, source_unit])),
                },
                "matched_control_units": controls.tolist(),
                "qualified_match_threshold": MATCH_DISTANCE_THRESHOLD,
                "qualified_control_units": qualified.tolist(),
                "match_distance": {
                    "minimum": float(distances[controls].min()),
                    "median": float(np.median(distances[controls])),
                    "maximum": float(distances[controls].max()),
                },
                "restoration_correct_to_wrong": metric_summary(
                    effects["restoration_correct_to_wrong"], source_unit, controls, qualified
                ),
                "patchin_unsat_to_sat": metric_summary(
                    effects["patchin_unsat_to_sat"], source_unit, controls, qualified
                ),
                "patchin_matches_edited_label": metric_summary(
                    effects["patchin_matches_edited_label"], source_unit, controls, qualified
                ),
            }
            arrays[name + "__controls"] = controls
            arrays[name + "__match_distance"] = distances.astype(np.float32)
            arrays[name + "__delta"] = delta.astype(np.float32)
            for metric, values in effects.items():
                if values.ndim:
                    arrays[name + "__" + metric] = values

    # Unit-411 compensation audit on the 32 original-direction positive edits.
    blocks = np.flatnonzero(bank["pair_index"] == 1)
    before, after = STATE_PAIRS[0]
    delta = (
        native["full_post"][blocks, 0, after].astype(np.float64)
        - native["full_post"][blocks, 0, before].astype(np.float64)
    )
    logits_before = native["logits"][blocks, 0, before].astype(np.float64)
    logits_after = native["logits"][blocks, 0, after].astype(np.float64)
    effects = class_effects(
        delta,
        logits_before,
        logits_after,
        bank["labels"][blocks, 0, before].astype(np.int64),
        bank["labels"][blocks, 0, after].astype(np.int64),
        coefficients,
    )
    misses = native["full_post"][blocks, 0, after, 411] < 0.5
    published_units = np.array(sorted(rule_by_unit), dtype=np.int16)
    restored_wrong = np.empty((32, len(published_units)), dtype=bool)
    patchin_sat = np.empty_like(restored_wrong)
    native_after = logits_after.argmax(axis=1)
    native_before = logits_before.argmax(axis=1)
    for column, unit in enumerate(published_units):
        change = delta[:, unit, None] * coefficients[unit][None, :]
        restored_wrong[:, column] = (
            (native_after == bank["labels"][blocks, 0, after])
            & ((logits_after - change).argmax(axis=1) != bank["labels"][blocks, 0, after])
        )
        patchin_sat[:, column] = (native_before == 14) & ((logits_before + change).argmax(axis=1) == 13)
    unit_rows = []
    for column, unit_raw in enumerate(published_units):
        unit = int(unit_raw)
        unit_rows.append({
            "unit": unit,
            "source_rule": rule_by_unit[unit].get("general_rule_ast"),
            "active_on_411_misses": int((native["full_post"][blocks[misses], 0, after, unit] >= 0.5).sum()),
            "active_on_411_hits": int((native["full_post"][blocks[~misses], 0, after, unit] >= 0.5).sum()),
            "restoration_wrong_on_411_misses": int(restored_wrong[misses, column].sum()),
            "restoration_wrong_on_411_hits": int(restored_wrong[~misses, column].sum()),
            "patchin_sat_on_411_misses": int(patchin_sat[misses, column].sum()),
            "patchin_sat_on_411_hits": int(patchin_sat[~misses, column].sum()),
        })
    unit_rows.sort(
        key=lambda row: (
            -row["restoration_wrong_on_411_misses"],
            -row["active_on_411_misses"],
            row["unit"],
        )
    )
    unit_201 = next(row for row in unit_rows if row["unit"] == 201)
    compensation = {
        "direction": "pair 411/492 original, positive edited endpoints",
        "unit_411_misses": int(misses.sum()),
        "unit_411_hits": int((~misses).sum()),
        "unit_201": unit_201,
        "all_published_units_sorted_by_restoration_on_411_misses": unit_rows,
        "interpretation": (
            "Unit 201's published rule includes TTFFT and excludes TFTFF. Coactivation or a saved-tensor "
            "effect is descriptive evidence of distributed compensation, not proof that unit 201 implements "
            "the same feature or causally mediates unit 411's threshold misses."
        ),
    }
    arrays["compensation__published_units"] = published_units
    arrays["compensation__unit_411_miss"] = misses
    arrays["compensation__restored_wrong"] = restored_wrong
    arrays["compensation__patchin_sat"] = patchin_sat

    result = {
        "status": "PASS",
        "scope": "post hoc saved-tensor specificity strata, activation-change-matched direct-logit nulls, patch-in sufficiency diagnostic, and unit-411 compensation audit",
        "limits": [
            "No multi-solution negative inputs exist in the protected bank",
            "The 32-nearest reference sets are not called matched; a qualified control requires relative sorted-profile distance at most 0.25",
            "Only one qualified control exists in each original direction and none in either renamed direction, so a full activity-matched null is unavailable",
            "Restoration and patch-in operate at the exact final linear readout cut and may be off-manifold",
            "All analyses were chosen after inspecting the protected result",
        ],
        "new_neural_forwards": 0,
        "fits": 0,
        "inputs": {
            str(NATIVE_PATH.relative_to(ROOT)): sha256(NATIVE_PATH),
            str(BANK_PATH.relative_to(ROOT)): sha256(BANK_PATH),
            str(WEIGHTS_PATH.relative_to(ROOT)): sha256(WEIGHTS_PATH),
            str(MAPPING_PATH.relative_to(ROOT)): sha256(MAPPING_PATH),
        },
        "specificity": specificity,
        "causal_controls": causal,
        "unit_411_compensation": compensation,
    }
    (HERE / "renaming_specificity_and_causal_controls.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    np.savez_compressed(HERE / "renaming_specificity_and_causal_controls_arrays.npz", **arrays)

    lines = [
        "# Renaming specificity and direct-logit controls",
        "",
        "## Specificity by semantic input class",
        "",
        "| Pair / direction | UNSAT TN/denom | SAT different-sole TN/denom | Multi-solution excluding-label TN/denom |",
        "|:--|--:|--:|--:|",
    ]
    for name, row in specificity.items():
        if name.endswith("off_pair"):
            continue
        values = row["negative_target_bits"]
        fields = []
        for category_name in ("unsat", "sat_different_sole_solution", "sat_multi_solution_excluding_label"):
            entry = values[category_name]
            fields.append(f"{entry['true_negative']}/{entry['denominator']}")
        lines.append(f"| {row['source_pair']} {row['direction']} | " + " | ".join(fields) + " |")
    lines.extend([
        "",
        "## Activation-change-matched restoration and patch-in",
        "",
        "| Direction | Source | Native edited correct | Qualified controls | Restoration C→W, source / qualified values | Patch-in UNSAT→SAT, source / qualified values |",
        "|:--|--:|--:|:--|:--|:--|",
    ])
    for row in causal.values():
        restoration = row["restoration_correct_to_wrong"]
        patchin = row["patchin_unsat_to_sat"]
        qualified = row["qualified_control_units"]
        lines.append(
            f"| {row['source_pair']} {row['direction']} | {row['source_unit']} | {row['native_after_correct']}/32 | "
            f"{qualified if qualified else 'none'} | "
            f"{restoration['source']} / {restoration['qualified_control_values'] if qualified else 'n/a'} | "
            f"{patchin['source']} / {patchin['qualified_control_values'] if qualified else 'n/a'} |"
        )
    lines.extend([
        "",
        "Controls are selected using only the sorted 32-case absolute activation-change profile. A qualified match has relative Euclidean profile distance at most 0.25. Only unit 374 qualifies for unit 261 and only unit 201 qualifies for unit 411; neither renamed direction has a qualified control. The wider 32-nearest sets remain in the JSON as references and are not described as matched. Effects are then evaluated through the saved exact linear readout. These post hoc interventions may be off-manifold.",
        "",
        "## Unit 411 and unit 201",
        "",
        f"Unit 411 misses the 0.5 threshold on {compensation['unit_411_misses']}/32 positive edited endpoints. Unit 201 is active on {unit_201['active_on_411_misses']}/{compensation['unit_411_misses']} misses and {unit_201['active_on_411_hits']}/{compensation['unit_411_hits']} hits; restoring unit 201 changes {unit_201['restoration_wrong_on_411_misses']} missed-threshold decisions and {unit_201['restoration_wrong_on_411_hits']} hit decisions from correct to wrong. Its published rule contains TTFFT and excludes TFTFF. This is a compensation diagnostic, not a unique-mechanism result.",
        "",
    ])
    (HERE / "renaming_specificity_and_causal_controls.md").write_text("\n".join(lines))
    print(json.dumps({
        "status": result["status"],
        "specificity": specificity,
        "causal": {
            name: {
                "source": row["source_unit"],
                "restoration": row["restoration_correct_to_wrong"],
                "patchin": row["patchin_unsat_to_sat"],
            }
            for name, row in causal.items()
        },
        "unit_201": unit_201,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
