#!/usr/bin/env python3
"""Direction-specific single-unit and pair nulls for saved origin restoration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE
INPUTS = HERE / "inputs"
NATIVE_PATH = INPUTS / "renaming_native_outputs.npz"
BANK_PATH = INPUTS / "renaming_selected_bank.npz"
WEIGHTS_PATH = INPUTS / "readout_weights.npz"
SOURCE_PAIRS = ((261, 291), (411, 492))
STATE_PAIRS = ((0, 1), (2, 3))
CHUNK = 2048


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values: np.ndarray, source_index: int) -> dict:
    source = values[source_index]
    return {
        "value": float(source),
        "rank_best_is_one": int((values > source).sum()) + 1,
        "tied_at_value": int((values == source).sum()),
        "population": int(len(values)),
    }


def intervention_metrics(
    delta: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    coefficients: np.ndarray,
    members: tuple[np.ndarray, ...],
) -> dict[str, np.ndarray]:
    native_pred = logits.argmax(axis=1)
    initially_correct = native_pred == labels
    count = len(members[0])
    correct_to_wrong = np.empty(count, dtype=np.int16)
    class_changes = np.empty(count, dtype=np.int16)
    signed = np.empty(count, dtype=np.float64)
    for start in range(0, count, CHUNK):
        stop = min(start + CHUNK, count)
        change = np.zeros((len(delta), stop - start, logits.shape[1]), dtype=np.float64)
        for member in members:
            ids = member[start:stop]
            change += delta[:, ids, None] * coefficients[ids][None, :, :]
        restored = (logits[:, None, :] - change).argmax(axis=2)
        correct_to_wrong[start:stop] = (
            initially_correct[:, None] & (restored != labels[:, None])
        ).sum(axis=0)
        class_changes[start:stop] = (restored != native_pred[:, None]).sum(axis=0)
        signed[start:stop] = (change[:, :, 13] - change[:, :, 14]).mean(axis=0)
    return {
        "correct_to_wrong": correct_to_wrong,
        "class_changes": class_changes,
        "mean_signed_sat_minus_unsat_contribution": signed,
        "initially_correct": np.array(int(initially_correct.sum())),
    }


def best_non_source(values: np.ndarray, labels: list[str], excluded: set[int]) -> dict:
    order = np.argsort(-values, kind="stable")
    index = next(int(i) for i in order if int(i) not in excluded)
    return {"label": labels[index], "value": float(values[index]), "index": index}


def main() -> None:
    native = np.load(NATIVE_PATH, allow_pickle=False)
    bank = np.load(BANK_PATH, allow_pickle=False)
    weights = np.load(WEIGHTS_PATH, allow_pickle=False)
    coefficients = weights["W_out"].astype(np.float64).T @ weights["W_U"].astype(np.float64)
    units = np.arange(512, dtype=np.int16)
    pair_i, pair_j = np.triu_indices(512, k=1)
    pair_lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(pair_i, pair_j))}
    pair_labels = [f"{int(i)}+{int(j)}" for i, j in zip(pair_i, pair_j)]
    unit_labels = [str(i) for i in units]

    results = {}
    arrays = {"pair_i": pair_i.astype(np.int16), "pair_j": pair_j.astype(np.int16)}
    for pair_index, source_pair in enumerate(SOURCE_PAIRS):
        blocks = np.flatnonzero(bank["pair_index"] == pair_index)
        source_pair_index = pair_lookup[tuple(sorted(source_pair))]
        for direction, (before, after) in enumerate(STATE_PAIRS):
            delta = (
                native["full_post"][blocks, 0, after].astype(np.float64)
                - native["full_post"][blocks, 0, before].astype(np.float64)
            )
            logits = native["logits"][blocks, 0, after].astype(np.float64)
            labels = bank["labels"][blocks, 0, after].astype(np.int64)
            single = intervention_metrics(delta, logits, labels, coefficients, (units,))
            pairs = intervention_metrics(delta, logits, labels, coefficients, (pair_i, pair_j))
            active_source_unit = source_pair[direction]
            other_source_unit = source_pair[1 - direction]
            pairs_containing_source = set(
                np.flatnonzero(
                    np.isin(pair_i, source_pair) | np.isin(pair_j, source_pair)
                ).tolist()
            )
            name = f"pair_{pair_index}_{'original' if direction == 0 else 'renamed'}"
            results[name] = {
                "source_pair": list(source_pair),
                "direction": "original" if direction == 0 else "renamed",
                "endpoints": 32,
                "initially_correct": int(single["initially_correct"]),
                "active_source_unit": int(active_source_unit),
                "active_source_unit_ranks": {
                    metric: ranks(single[metric], active_source_unit)
                    for metric in ("correct_to_wrong", "class_changes", "mean_signed_sat_minus_unsat_contribution")
                },
                "other_source_unit_correct_to_wrong": int(single["correct_to_wrong"][other_source_unit]),
                "best_non_source_single_by_correct_to_wrong": best_non_source(
                    single["correct_to_wrong"], unit_labels, set(source_pair)
                ),
                "source_pair_ranks": {
                    metric: ranks(pairs[metric], source_pair_index)
                    for metric in ("correct_to_wrong", "class_changes", "mean_signed_sat_minus_unsat_contribution")
                },
                "best_pair_excluding_both_source_units_by_correct_to_wrong": best_non_source(
                    pairs["correct_to_wrong"], pair_labels, pairs_containing_source
                ),
            }
            for metric in ("correct_to_wrong", "class_changes", "mean_signed_sat_minus_unsat_contribution"):
                arrays[f"{name}__single__{metric}"] = single[metric]
                arrays[f"{name}__pair__{metric}"] = pairs[metric]

    receipt = {
        "status": "PASS",
        "scope": "post hoc direction-specific direct-logit origin-restoration null on saved arrays",
        "new_neural_forwards": 0,
        "new_fits": 0,
        "inputs": {
            str(NATIVE_PATH.relative_to(ROOT)): sha256(NATIVE_PATH),
            str(BANK_PATH.relative_to(ROOT)): sha256(BANK_PATH),
            str(WEIGHTS_PATH.relative_to(ROOT)): sha256(WEIGHTS_PATH),
        },
        "directions": results,
        "interpretation_boundary": (
            "The construction selected the source units before this post hoc null. Rankings are descriptive. "
            "Origin restoration is an off-manifold intervention at the final MLP cut and does not establish "
            "upstream mediation, sufficiency, or unique causal routing."
        ),
    }
    (HERE / "direction_specific_clamp_null.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    np.savez_compressed(HERE / "direction_specific_clamp_null_arrays.npz", **arrays)

    lines = [
        "# Direction-specific origin-restoration null",
        "",
        "Each construction direction is evaluated on its own 32 positive edited endpoints. The active source unit is ranked against all 512 units; the source pair is ranked against all 130,816 unordered pairs.",
        "",
        "| Direction | Initially correct | Active source unit | Correct→wrong | Single-unit rank | Best non-source unit | Source-pair correct→wrong | Pair rank | Best non-source pair |",
        "|:--|--:|--:|--:|--:|:--|--:|--:|:--|",
    ]
    for item in results.values():
        single_rank = item["active_source_unit_ranks"]["correct_to_wrong"]
        pair_rank = item["source_pair_ranks"]["correct_to_wrong"]
        best_single = item["best_non_source_single_by_correct_to_wrong"]
        best_pair = item["best_pair_excluding_both_source_units_by_correct_to_wrong"]
        lines.append(
            f"| pair {item['source_pair']} {item['direction']} | {item['initially_correct']}/32 | "
            f"{item['active_source_unit']} | {single_rank['value']:.0f} | "
            f"{single_rank['rank_best_is_one']}/{single_rank['population']} | "
            f"{best_single['label']} ({best_single['value']:.0f}) | "
            f"{pair_rank['value']:.0f} | {pair_rank['rank_best_is_one']}/{pair_rank['population']} | "
            f"{best_pair['label']} ({best_pair['value']:.0f}) |"
        )
    lines.extend([
        "",
        "These are saved-tensor direct-logit calculations under origin restoration. They are exploratory and do not establish sufficiency or an on-manifold mechanism.",
        "",
    ])
    (HERE / "direction_specific_clamp_null.md").write_text("\n".join(lines))
    print(json.dumps({
        key: {
            "unit": value["active_source_unit"],
            "unit_rank": value["active_source_unit_ranks"]["correct_to_wrong"]["rank_best_is_one"],
            "pair_rank": value["source_pair_ranks"]["correct_to_wrong"]["rank_best_is_one"],
            "correct_to_wrong": value["source_pair_ranks"]["correct_to_wrong"]["value"],
        }
        for key, value in results.items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
