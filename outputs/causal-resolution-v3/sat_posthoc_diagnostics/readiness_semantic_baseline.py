#!/usr/bin/env python3
"""Read-only post-review audit of the missing semantic-only readiness baseline.

This script consumes the completed SAT mechanism arrays and their archived
bootstrap multiplicities.  It performs no neural inference, fitting, threshold
selection, or resampling, and writes only to this exploratory repair directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE
SOURCE = HERE / "inputs"
STAGE_PATH = SOURCE / "stage_b_raw.npz"
BOOT_PATH = SOURCE / "bootstrap_raw.npz"
MAPPING_PATH = SOURCE / "source_mapping.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, int | float | None]:
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    positive = int(y.sum())
    negative = int((~y).sum())
    tp = int((y & pred).sum())
    tn = int(((~y) & (~pred)).sum())
    fp = negative - tn
    fn = positive - tp
    sensitivity = tp / positive if positive else None
    specificity = tn / negative if negative else None
    ba = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "n": len(y),
        "positive": positive,
        "negative": negative,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": ba,
    }


def assignment_macro(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    unit_index: np.ndarray,
    scope: np.ndarray,
    assignments: list[str],
) -> dict:
    per_unit: dict[str, dict] = {}
    for unit in range(len(assignments)):
        mask = scope & (unit_index == unit)
        per_unit[str(unit)] = {
            name: metric(y[mask].reshape(-1), pred[mask].reshape(-1))
            for name, pred in predictions.items()
        }

    names = sorted(set(assignments))
    per_assignment: dict[str, dict[str, float | None]] = {}
    for assignment in names:
        members = [i for i, value in enumerate(assignments) if value == assignment]
        per_assignment[assignment] = {}
        for name in predictions:
            values = [per_unit[str(i)][name]["balanced_accuracy"] for i in members]
            per_assignment[assignment][name] = (
                float(np.mean(values)) if all(value is not None for value in values) else None
            )

    macro: dict[str, float | None] = {}
    for name in predictions:
        values = [per_assignment[a][name] for a in names]
        macro[name] = float(np.mean(values)) if all(value is not None for value in values) else None
    macro["structured_minus_semantic_only"] = (
        macro["structured"] - macro["semantic_only"]
        if macro["structured"] is not None and macro["semantic_only"] is not None
        else None
    )
    return {
        "paired_unit_origins": int(scope.sum()),
        "action_rows": int(scope.sum() * y.shape[1]),
        "per_unit": per_unit,
        "per_assignment": per_assignment,
        "macro": macro,
    }


def percentile(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "draws": int(len(values)),
        "valid_draws": int(len(finite)),
        "interval_2_5_97_5": np.quantile(finite, [0.025, 0.975]).tolist(),
    }


def bootstrap(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    unit_index: np.ndarray,
    origin_index: np.ndarray,
    scope: np.ndarray,
    assignments: list[str],
    origin_ids: np.ndarray,
    multiplicities: np.ndarray,
) -> tuple[dict, np.ndarray]:
    names = list(predictions)
    lookup = {int(origin): i for i, origin in enumerate(origin_ids)}
    counts = np.zeros((len(origin_ids), len(assignments), len(names), 4), dtype=np.int64)
    observed_origins = set()
    for row in np.flatnonzero(scope):
        origin = int(origin_index[row])
        observed_origins.add(origin)
        oi = lookup[origin]
        ui = int(unit_index[row])
        truth = y[row]
        for model, name in enumerate(names):
            pred = predictions[name][row]
            counts[oi, ui, model] += np.array(
                [
                    int((truth & pred).sum()),
                    int(((~truth) & (~pred)).sum()),
                    int(((~truth) & pred).sum()),
                    int((truth & (~pred)).sum()),
                ],
                dtype=np.int64,
            )
    missing = observed_origins.difference(int(value) for value in origin_ids)
    if missing:
        raise RuntimeError(f"assessment origins missing from archived bootstrap: {sorted(missing)}")

    totals = (
        multiplicities.astype(np.float64)
        @ counts.reshape(len(origin_ids), -1).astype(np.float64)
    ).reshape(len(multiplicities), len(assignments), len(names), 4)
    positive = totals[..., 0] + totals[..., 3]
    negative = totals[..., 1] + totals[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_ba = (totals[..., 0] / positive + totals[..., 1] / negative) / 2

    assignment_names = sorted(set(assignments))
    assignment_ba = np.stack(
        [
            unit_ba[:, [i for i, value in enumerate(assignments) if value == assignment], :].mean(axis=1)
            for assignment in assignment_names
        ],
        axis=1,
    )
    macro = assignment_ba.mean(axis=1)
    delta = macro[:, names.index("structured")] - macro[:, names.index("semantic_only")]
    report = {
        "scope": "saved original-origin multiplicities; all 2000 draws reused; no resampling",
        "origin_count": int(len(origin_ids)),
        "models": {name: percentile(macro[:, i]) for i, name in enumerate(names)},
        "structured_minus_semantic_only": percentile(delta),
    }
    arrays = np.column_stack([macro, delta])
    return report, arrays


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    stage = np.load(STAGE_PATH, allow_pickle=False)
    boot = np.load(BOOT_PATH, allow_pickle=False)
    mapping = json.loads(MAPPING_PATH.read_text())["single_assignment_general_units"]
    assignments = [row["assignment"] for row in mapping]

    model_names = stage["model_names"].tolist()
    structured_index = model_names.index("structured")
    y = stage["native_high"].astype(bool)
    predictions = {
        "structured": stage["predictions"][:, :, structured_index].astype(bool),
        "semantic_only": (stage["action_effect"] == 1),
    }
    if not np.array_equal(predictions["semantic_only"], np.tile([True, False], (len(y), 1))):
        raise RuntimeError("semantic-only action-effect baseline is not the expected repair/neutral rule")

    split = stage["split"]
    clipped = stage["clipped"]
    scopes = {
        "primary_assessment": (split == "assessment") & clipped,
        "excluded_unclipped_assessment": (split == "assessment") & (~clipped),
        "all_selected_assessment": split == "assessment",
        "primary_fit": (split == "fit") & clipped,
        "excluded_unclipped_fit": (split == "fit") & (~clipped),
        "all_selected_fit": split == "fit",
    }
    results = {
        name: assignment_macro(
            y,
            predictions,
            stage["unit_index"],
            scope,
            assignments,
        )
        for name, scope in scopes.items()
    }

    bootstrap_report, bootstrap_arrays = bootstrap(
        y,
        predictions,
        stage["unit_index"],
        stage["origin_index"],
        scopes["primary_assessment"],
        assignments,
        boot["origin_ids"],
        boot["origin_multiplicities"],
    )
    existing_structured = boot["macro_balanced_accuracy"][:, model_names.index("structured")]
    if not np.array_equal(bootstrap_arrays[:, 0], existing_structured):
        max_difference = float(np.nanmax(np.abs(bootstrap_arrays[:, 0] - existing_structured)))
        raise RuntimeError(f"structured bootstrap reproduction mismatch: {max_difference}")

    receipt = {
        "status": "PASS",
        "scope": "post-review exploratory read-only analysis of saved development data",
        "new_neural_forwards": 0,
        "new_fits": 0,
        "new_resampling": 0,
        "inputs": {
            str(STAGE_PATH.relative_to(ROOT)): sha256(STAGE_PATH),
            str(BOOT_PATH.relative_to(ROOT)): sha256(BOOT_PATH),
            str(MAPPING_PATH.relative_to(ROOT)): sha256(MAPPING_PATH),
        },
        "baseline_definition": "predict native activation iff the declared action effect m_v(a) equals one",
        "results": results,
        "bootstrap": bootstrap_report,
        "interpretation": (
            "The structured-minus-semantic-only comparison isolates the incremental contribution "
            "of the fitted readiness bit within this saved cohort. It is not a protected test, "
            "a causal mediation estimate, or evidence for a global SAT policy."
        ),
    }
    result_path = HERE / "readiness_semantic_baseline.json"
    result_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    np.savez_compressed(
        HERE / "readiness_semantic_baseline_bootstrap.npz",
        model_names=np.array(["structured", "semantic_only"]),
        macro_balanced_accuracy=bootstrap_arrays[:, :2],
        structured_minus_semantic_only=bootstrap_arrays[:, 2],
        origin_ids=boot["origin_ids"],
        origin_multiplicities_sha256=np.array(sha256(BOOT_PATH)),
    )

    primary = results["primary_assessment"]
    excluded = results["excluded_unclipped_assessment"]
    all_selected = results["all_selected_assessment"]
    interval = bootstrap_report["structured_minus_semantic_only"]["interval_2_5_97_5"]
    lines = [
        "# Missing semantic-only readiness baseline",
        "",
        "This is a post-review, read-only analysis of the saved development archive. It performs zero neural forwards, zero fits and zero new resampling.",
        "",
        "| Scope | Paired unit-origins | Structured BA | Semantic-only BA | Structured minus semantic-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in [
        ("Primary clipped assessment", primary),
        ("Excluded unclipped assessment", excluded),
        ("All selected assessment", all_selected),
    ]:
        macro = row["macro"]
        lines.append(
            f"| {label} | {row['paired_unit_origins']} | {macro['structured']:.6f} | "
            f"{macro['semantic_only']:.6f} | {macro['structured_minus_semantic_only']:.6f} |"
        )
    lines += [
        "",
        f"For the primary assessment, the paired 95% original-origin bootstrap interval for structured minus semantic-only is [{interval[0]:.6f}, {interval[1]:.6f}]. All 2,000 archived multiplicity vectors were reused, and the structured draws reproduce the frozen archive exactly.",
        "",
        "The semantic-only predictor says that the named unit will activate after the declared repair and will not activate after the neutral edit. The structured predictor adds the fitted original-state readiness bit. Their difference therefore measures the incremental value of that bit under this exact action pair and opened cohort.",
        "",
        "This comparison does not establish causal mediation, a global SAT repair, or protected generalisation. The unclipped rows are shown separately because excluding them was part of the frozen primary estimand.",
    ]
    (HERE / "readiness_semantic_baseline.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "primary_macro": primary["macro"], "bootstrap_delta": bootstrap_report["structured_minus_semantic_only"]}, sort_keys=True))


if __name__ == "__main__":
    main()
