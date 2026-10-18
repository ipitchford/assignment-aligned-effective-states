#!/usr/bin/env python3
"""Post-review all-pair null for the saved SAT direct-logit clamp.

This is a read-only calculation on saved final-block activations and readout
weights.  It performs no model forward pass and does not alter the frozen
renaming study.  The analysis is exploratory because the original contract
retained the 512-vector solely for affine parity, not for a unit search.
"""

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
AFFINE_PATH = INPUTS / "affine_accounting.json"
SOURCE_PAIRS = ((261, 291), (411, 492))
CHUNK = 2048


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_summary(values: np.ndarray, source_index: int, descending: bool) -> dict:
    source = values[source_index]
    better = int((values > source).sum()) if descending else int((values < source).sum())
    tied = int((values == source).sum())
    return {
        "value": float(source),
        "rank_best_is_one": better + 1,
        "tied_at_value": tied,
        "total_pairs": int(len(values)),
        "percentile_fraction_at_or_below": float((values <= source).mean()),
    }


def metrics_for_endpoints(
    delta: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    coefficients: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> dict[str, np.ndarray]:
    native_pred = logits.argmax(axis=1)
    initially_correct = native_pred == labels
    count = len(pair_i)
    class_changes = np.empty(count, dtype=np.int16)
    correct_to_wrong = np.empty(count, dtype=np.int16)
    signed_contribution = np.empty(count, dtype=np.float64)
    absolute_contribution = np.empty(count, dtype=np.float64)
    activity_rms = np.empty(count, dtype=np.float64)

    for start in range(0, count, CHUNK):
        stop = min(start + CHUNK, count)
        ii = pair_i[start:stop]
        jj = pair_j[start:stop]
        change = (
            delta[:, ii, None] * coefficients[ii][None, :, :]
            + delta[:, jj, None] * coefficients[jj][None, :, :]
        )
        clamped_pred = (logits[:, None, :] - change).argmax(axis=2)
        changed = clamped_pred != native_pred[:, None]
        wrong = clamped_pred != labels[:, None]
        class_changes[start:stop] = changed.sum(axis=0)
        correct_to_wrong[start:stop] = (initially_correct[:, None] & wrong).sum(axis=0)
        margin_contribution = change[:, :, 13] - change[:, :, 14]
        signed_contribution[start:stop] = margin_contribution.mean(axis=0)
        absolute_contribution[start:stop] = np.abs(margin_contribution).mean(axis=0)
        activity_rms[start:stop] = np.sqrt(np.mean(delta[:, ii] ** 2 + delta[:, jj] ** 2, axis=0))
    return {
        "class_changes": class_changes,
        "correct_to_wrong": correct_to_wrong,
        "mean_signed_sat_minus_unsat_contribution": signed_contribution,
        "mean_absolute_sat_minus_unsat_contribution": absolute_contribution,
        "activation_change_rms": activity_rms,
        "initially_correct_endpoints": np.array(int(initially_correct.sum())),
        "endpoints": np.array(len(labels)),
    }


def quantiles(values: np.ndarray) -> dict:
    return {
        "minimum": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    native = np.load(NATIVE_PATH, allow_pickle=False)
    bank = np.load(BANK_PATH, allow_pickle=False)
    weights = np.load(WEIGHTS_PATH, allow_pickle=False)
    coefficients = weights["W_out"].astype(np.float64).T @ weights["W_U"].astype(np.float64)
    if coefficients.shape != (512, 16):
        raise RuntimeError("unexpected readout shape")
    if not json.loads(AFFINE_PATH.read_text())["parity_pass"]:
        raise RuntimeError("saved affine parity did not pass")

    pair_i, pair_j = np.triu_indices(512, k=1)
    pair_lookup = {(int(i), int(j)): index for index, (i, j) in enumerate(zip(pair_i, pair_j))}
    report = {}
    arrays = {"pair_i": pair_i.astype(np.int16), "pair_j": pair_j.astype(np.int16)}

    for target, source_pair in enumerate(SOURCE_PAIRS):
        blocks = np.flatnonzero(bank["pair_index"] == target)
        if len(blocks) != 32:
            raise RuntimeError(f"expected 32 blocks for pair {target}, found {len(blocks)}")
        deltas = []
        logits = []
        labels = []
        for block in blocks:
            for before, after in ((0, 1), (2, 3)):
                deltas.append(
                    native["full_post"][block, 0, after].astype(np.float64)
                    - native["full_post"][block, 0, before].astype(np.float64)
                )
                logits.append(native["logits"][block, 0, after].astype(np.float64))
                labels.append(int(bank["labels"][block, 0, after]))
        delta = np.stack(deltas)
        endpoint_logits = np.stack(logits)
        endpoint_labels = np.asarray(labels, dtype=np.int64)
        result = metrics_for_endpoints(delta, endpoint_logits, endpoint_labels, coefficients, pair_i, pair_j)
        source_index = pair_lookup[tuple(sorted(source_pair))]

        activity = result["activation_change_rms"]
        source_activity = activity[source_index]
        match_factor = 1.5
        matched_indices = np.flatnonzero(
            (activity >= source_activity / match_factor)
            & (activity <= source_activity * match_factor)
        )
        matched_count = len(matched_indices)
        if source_index not in matched_indices:
            raise RuntimeError("source pair missing from nearest-activity set")
        source_within = int(np.flatnonzero(matched_indices == source_index)[0])

        metrics = {
            "correct_to_wrong": rank_summary(result["correct_to_wrong"], source_index, descending=True),
            "class_changes": rank_summary(result["class_changes"], source_index, descending=True),
            "mean_signed_sat_minus_unsat_contribution": rank_summary(
                result["mean_signed_sat_minus_unsat_contribution"], source_index, descending=True
            ),
            "mean_absolute_sat_minus_unsat_contribution": rank_summary(
                result["mean_absolute_sat_minus_unsat_contribution"], source_index, descending=True
            ),
            "activation_change_rms": rank_summary(activity, source_index, descending=True),
        }
        matched_metrics = {
            "correct_to_wrong": rank_summary(result["correct_to_wrong"][matched_indices], source_within, descending=True),
            "class_changes": rank_summary(result["class_changes"][matched_indices], source_within, descending=True),
            "mean_signed_sat_minus_unsat_contribution": rank_summary(
                result["mean_signed_sat_minus_unsat_contribution"][matched_indices], source_within, descending=True
            ),
            "mean_absolute_sat_minus_unsat_contribution": rank_summary(
                result["mean_absolute_sat_minus_unsat_contribution"][matched_indices], source_within, descending=True
            ),
        }
        report[f"pair_{target}"] = {
            "source_units": list(source_pair),
            "selected_blocks": blocks.tolist(),
            "positive_edited_endpoints": int(result["endpoints"]),
            "initially_correct_endpoints": int(result["initially_correct_endpoints"]),
            "full_pair_null": metrics,
            "activity_matched_within_factor_1_5": {
                "definition": "unit pairs whose activation-change RMS is between source_RMS/1.5 and source_RMS*1.5",
                "pair_count": int(matched_count),
                "source_activation_change_rms": float(source_activity),
                "metrics": matched_metrics,
            },
            "null_quantiles": {
                name: quantiles(result[name])
                for name in (
                    "correct_to_wrong",
                    "class_changes",
                    "mean_signed_sat_minus_unsat_contribution",
                    "mean_absolute_sat_minus_unsat_contribution",
                    "activation_change_rms",
                )
            },
        }
        for name, values in result.items():
            if np.ndim(values):
                arrays[f"pair_{target}__{name}"] = values
        arrays[f"pair_{target}__activity_matched_indices"] = matched_indices.astype(np.int32)

    combined_initial = sum(item["initially_correct_endpoints"] for item in report.values())
    combined_c2w = sum(item["full_pair_null"]["correct_to_wrong"]["value"] for item in report.values())
    receipt = {
        "status": "PASS",
        "scope": "post-review exploratory all-pair direct-logit calculation on saved arrays",
        "new_neural_forwards": 0,
        "new_fits": 0,
        "candidate_unit_pairs": int(len(pair_i)),
        "inputs": {
            str(NATIVE_PATH.relative_to(ROOT)): sha256(NATIVE_PATH),
            str(BANK_PATH.relative_to(ROOT)): sha256(BANK_PATH),
            str(WEIGHTS_PATH.relative_to(ROOT)): sha256(WEIGHTS_PATH),
            str(AFFINE_PATH.relative_to(ROOT)): sha256(AFFINE_PATH),
        },
        "pairs": report,
        "combined_nominated_correct_to_wrong": int(combined_c2w),
        "combined_nominated_initially_correct": int(combined_initial),
        "interpretation_boundary": (
            "Ranks are descriptive and post hoc. The source pair changes were selected by the source interpretation, "
            "whereas each null pair is constant on the same endpoints. This is direct-logit attribution under an "
            "off-manifold clamp, not upstream mediation, unique necessity, or a protected unit-selection test."
        ),
    }
    (HERE / "clamp_pair_null.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    np.savez_compressed(HERE / "clamp_pair_null_arrays.npz", **arrays)

    lines = [
        "# Saved-tensor all-pair clamp null",
        "",
        "This post-review calculation uses the saved 512-coordinate final-block vectors and exact affine readout. It performs zero neural forwards and evaluates all 130,816 unordered unit pairs.",
        "",
        "| Source pair | Positive endpoints | Initially correct | Correct→wrong | Full-null rank | Activity-matched rank | Mean signed SAT−UNSAT contribution | Full-null rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in range(2):
        item = report[f"pair_{target}"]
        c2w = item["full_pair_null"]["correct_to_wrong"]
        c2w_matched = item["activity_matched_within_factor_1_5"]["metrics"]["correct_to_wrong"]
        contribution = item["full_pair_null"]["mean_signed_sat_minus_unsat_contribution"]
        lines.append(
            f"| {item['source_units'][0]}+{item['source_units'][1]} | {item['positive_edited_endpoints']} | "
            f"{item['initially_correct_endpoints']} | {c2w['value']:.0f} | {c2w['rank_best_is_one']}/{c2w['total_pairs']} | "
            f"{c2w_matched['rank_best_is_one']}/{c2w_matched['total_pairs']} | {contribution['value']:.6f} | "
            f"{contribution['rank_best_is_one']}/{contribution['total_pairs']} |"
        )
    lines += [
        "",
        f"Across the two source-matched endpoint sets, the nominated pairs change {combined_c2w:.0f} decisions from correct to wrong among {combined_initial} initially correct positive endpoints.",
        "",
        "The activity-matched set contains every unit pair whose mean two-coordinate activation-change RMS is within a factor of 1.5 of the source pair. It is a descriptive post hoc comparator, not a prespecified test.",
        "",
        "The calculation is direct-logit attribution under the stated saved-tensor clamp. It does not show upstream mediation, unique causal routing or necessity under an on-manifold intervention.",
    ]
    (HERE / "clamp_pair_null.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "combined_correct_to_wrong": int(combined_c2w), "combined_initially_correct": int(combined_initial), "ranks": {key: value["full_pair_null"]["correct_to_wrong"]["rank_best_is_one"] for key, value in report.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
