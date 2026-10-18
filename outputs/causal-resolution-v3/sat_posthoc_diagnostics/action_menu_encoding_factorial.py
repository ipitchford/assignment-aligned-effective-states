#!/usr/bin/env python3
"""Cached 2x2 diagnostic for SAT action menu sampling and action encoding.

The diagnostic crosses the original pilot's all-slot/sparse-fit menu regime with
the later screen's selected-slot/dense-fit regime and crosses one-hot with
natural-binary action features.  It compares published bits with continuous
postactivation using one fixed learner configuration.  No neural forward pass
or new formula is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


HERE = Path(__file__).resolve().parent
ROOT = HERE
INPUTS = HERE / "inputs"
BANK_PATH = INPUTS / "pilot_bank.npz"
NATIVE_PATH = INPUTS / "original_native.npz"
ARCHIVED_PILOT_PATH = INPUTS / "predictive_metrics.json"
ARCHIVED_SCREEN_PATH = INPUTS / "task_policy_result.json"
TAG = b"already-clean-task-policy-augmentation-v1\0"
SETTINGS = {
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "learning_rate": 0.08,
    "l2_regularization": 1.0,
    "max_bins": 63,
    "early_stopping": False,
    "random_state": 2026101514,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_columns(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return ((values[..., None] >> np.arange(width, dtype=np.int64)) & 1).astype(np.float32)


def selected_slots(tokens: np.ndarray) -> np.ndarray:
    slots = []
    for row in tokens:
        payload = np.asarray(row, dtype="<i8").tobytes(order="C")
        slots.append(int.from_bytes(hashlib.sha256(TAG + payload).digest()[:8], "big") % 10)
    return np.asarray(slots, dtype=np.int8)


def pilot_rows(clauses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026101513)
    fit = []
    assessment = []
    for origin in range(3072):
        for slot in range(10):
            old = int(clauses[origin, slot])
            choices = np.delete(np.arange(40), old)
            for new in rng.choice(choices, size=2, replace=False):
                fit.append((origin, slot, old, int(new)))
    for origin in range(3072, 4096):
        for slot in range(10):
            old = int(clauses[origin, slot])
            for new in range(40):
                if new != old:
                    assessment.append((origin, slot, old, new))
    fit_rows = np.asarray(fit, dtype=np.int32)
    assessment_rows = np.asarray(assessment, dtype=np.int32)
    if fit_rows.shape != (61_440, 4) or assessment_rows.shape != (399_360, 4):
        raise RuntimeError("pilot row construction changed")
    return fit_rows, assessment_rows


def selected_slot_rows(clauses: np.ndarray, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def build(start: int, stop: int) -> np.ndarray:
        rows = []
        for origin in range(start, stop):
            slot = int(slots[origin])
            old = int(clauses[origin, slot])
            for new in range(40):
                if new != old:
                    rows.append((origin, slot, old, new))
        return np.asarray(rows, dtype=np.int32)

    fit_rows = build(0, 3072)
    assessment_rows = build(3072, 4096)
    if fit_rows.shape != (119_808, 4) or assessment_rows.shape != (39_936, 4):
        raise RuntimeError("selected-slot row construction changed")
    return fit_rows, assessment_rows


def action_features(rows: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "one_hot":
        result = np.zeros((len(rows), 90), dtype=np.float32)
        index = np.arange(len(rows))
        result[index, rows[:, 1]] = 1
        result[index, 10 + rows[:, 2]] = 1
        result[index, 50 + rows[:, 3]] = 1
        return result
    if encoding == "natural_binary":
        return np.concatenate(
            (
                binary_columns(rows[:, 1], 4),
                binary_columns(rows[:, 2], 6),
                binary_columns(rows[:, 3], 6),
            ),
            axis=1,
        )
    raise ValueError(encoding)


def design(report: np.ndarray, rows: np.ndarray, encoding: str) -> np.ndarray:
    return np.concatenate((report[rows[:, 0]], action_features(rows, encoding)), axis=1)


def unsat_labels(post_sat: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return ~post_sat[rows[:, 0], rows[:, 1], rows[:, 3]]


def predict_chunks(model: HistGradientBoostingClassifier, report: np.ndarray, rows: np.ndarray, encoding: str) -> np.ndarray:
    true_column = list(model.classes_).index(True)
    output = []
    for start in range(0, len(rows), 8192):
        batch = design(report, rows[start : start + 8192], encoding)
        output.append(model.predict_proba(batch)[:, true_column])
    return np.concatenate(output)


def select(probability: np.ndarray, rows: np.ndarray, outcome: np.ndarray) -> dict[str, np.ndarray]:
    if len(rows) % 39:
        raise RuntimeError("assessment rows do not form 39-action menus")
    p = probability.reshape(-1, 39)
    r = rows.reshape(-1, 39, 4)
    y = outcome.reshape(-1, 39)
    chosen_index = np.argmax(p, axis=1)
    index = np.arange(len(chosen_index))
    chosen_rows = r[index, chosen_index]
    return {
        "origin": chosen_rows[:, 0],
        "slot": chosen_rows[:, 1],
        "clause": chosen_rows[:, 3],
        "confidence": p[index, chosen_index],
        "success": y[index, chosen_index],
        "reachable": y.any(axis=1),
    }


def summarize(choice: dict[str, np.ndarray]) -> dict:
    success = choice["success"]
    reachable = choice["reachable"]
    return {
        "menus": int(len(success)),
        "reachable_menus": int(reachable.sum()),
        "success_all_menus": float(success.mean()),
        "success_when_reachable": float(success[reachable].mean()),
        "mean_regret": float((reachable.astype(np.int8) - success.astype(np.int8)).mean()),
        "coverage_at_0_5": float((choice["confidence"] >= 0.5).mean()),
    }


def per_origin_success(choice: dict[str, np.ndarray]) -> np.ndarray:
    output = np.empty(1024, dtype=np.float64)
    for index, origin in enumerate(range(3072, 4096)):
        selected = choice["origin"] == origin
        if not selected.any():
            raise RuntimeError(f"missing assessment origin {origin}")
        output[index] = choice["success"][selected].mean()
    return output


def interval(draws: np.ndarray) -> dict:
    return {
        "draws": int(len(draws)),
        "interval_2_5_97_5": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    started = time.monotonic()
    HERE.mkdir(parents=True, exist_ok=True)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    expected = {"python": "3.11.14", "numpy": "2.4.6", "scipy": "1.17.1", "sklearn": "1.9.0"}
    if versions != expected:
        raise RuntimeError(f"pinned analysis runtime drift: {versions}")

    bank = np.load(BANK_PATH, allow_pickle=False)
    native = np.load(NATIVE_PATH, allow_pickle=False)
    if not np.array_equal(bank["published_units"], native["units"]):
        raise RuntimeError("published-unit order mismatch")
    reports = {
        "published_bits34": (native["post"] >= 0.5).astype(np.float32),
        "continuous_post34": native["post"].astype(np.float32),
    }
    menus = {
        "all_slot_sparse_fit": pilot_rows(bank["clauses"]),
        "selected_slot_dense_fit": selected_slot_rows(bank["clauses"], selected_slots(bank["tokens"])),
    }
    encodings = ("one_hot", "natural_binary")
    cells = {}
    saved = {}

    for menu_name, (fit_rows, assessment_rows) in menus.items():
        fit_y = unsat_labels(bank["post_sat"], fit_rows)
        assessment_y = unsat_labels(bank["post_sat"], assessment_rows)
        for encoding in encodings:
            cell_name = f"{menu_name}__{encoding}"
            cell = {
                "fit_rows": int(len(fit_rows)),
                "assessment_rows": int(len(assessment_rows)),
                "assessment_menus": int(len(assessment_rows) // 39),
                "models": {},
            }
            choices = {}
            for report_name, report in reports.items():
                model = HistGradientBoostingClassifier(**SETTINGS)
                model.fit(design(report, fit_rows, encoding), fit_y)
                probability = predict_chunks(model, report, assessment_rows, encoding)
                choice = select(probability, assessment_rows, assessment_y)
                choices[report_name] = choice
                cell["models"][report_name] = summarize(choice)
                saved[f"{cell_name}__{report_name}__origin_success"] = per_origin_success(choice)
                saved[f"{cell_name}__{report_name}__chosen_clause"] = choice["clause"]
            bits = cell["models"]["published_bits34"]["success_all_menus"]
            continuous = cell["models"]["continuous_post34"]["success_all_menus"]
            cell["continuous_minus_bits"] = continuous - bits
            cell["chosen_clause_agreement"] = float(
                np.mean(choices["published_bits34"]["clause"] == choices["continuous_post34"]["clause"])
            )
            cells[cell_name] = cell

    rng = np.random.default_rng(2026101804)
    multiplicities = np.array(
        [np.bincount(rng.integers(0, 1024, size=1024), minlength=1024) for _ in range(2000)],
        dtype=np.int16,
    )
    paired = {}
    cell_deltas = {}
    for cell_name in cells:
        bits = saved[f"{cell_name}__published_bits34__origin_success"]
        continuous = saved[f"{cell_name}__continuous_post34__origin_success"]
        origin_delta = continuous - bits
        draws = multiplicities.astype(np.float64) @ origin_delta / 1024
        cell_deltas[cell_name] = draws
        paired[cell_name] = {
            "point_continuous_minus_bits": float(origin_delta.mean()),
            "paired_origin_interval": interval(draws),
        }

    contrasts = {}
    for menu_name in menus:
        left = f"{menu_name}__natural_binary"
        right = f"{menu_name}__one_hot"
        draws = cell_deltas[left] - cell_deltas[right]
        contrasts[f"encoding_effect_on_continuous_minus_bits__{menu_name}"] = {
            "point": cells[left]["continuous_minus_bits"] - cells[right]["continuous_minus_bits"],
            "paired_origin_interval": interval(draws),
        }
    for encoding in encodings:
        left = f"selected_slot_dense_fit__{encoding}"
        right = f"all_slot_sparse_fit__{encoding}"
        draws = cell_deltas[left] - cell_deltas[right]
        contrasts[f"menu_effect_on_continuous_minus_bits__{encoding}"] = {
            "point": cells[left]["continuous_minus_bits"] - cells[right]["continuous_minus_bits"],
            "paired_origin_interval": interval(draws),
        }
    interaction_draws = (
        cell_deltas["selected_slot_dense_fit__natural_binary"]
        - cell_deltas["selected_slot_dense_fit__one_hot"]
        - cell_deltas["all_slot_sparse_fit__natural_binary"]
        + cell_deltas["all_slot_sparse_fit__one_hot"]
    )
    interaction_point = (
        cells["selected_slot_dense_fit__natural_binary"]["continuous_minus_bits"]
        - cells["selected_slot_dense_fit__one_hot"]["continuous_minus_bits"]
        - cells["all_slot_sparse_fit__natural_binary"]["continuous_minus_bits"]
        + cells["all_slot_sparse_fit__one_hot"]["continuous_minus_bits"]
    )
    contrasts["menu_by_encoding_interaction"] = {
        "point": interaction_point,
        "paired_origin_interval": interval(interaction_draws),
    }

    archived_pilot = json.loads(ARCHIVED_PILOT_PATH.read_text())
    reproduced = cells["all_slot_sparse_fit__one_hot"]
    archived_pilot_names = {
        "published_bits34": "published_bits34",
        "continuous_post34": "published_float34",
    }
    reproduction = {
        name: {
            "archived": archived_pilot["models"][archived_pilot_names[name]]["decision"]["success_all_menus"],
            "factorial": reproduced["models"][name]["success_all_menus"],
        }
        for name in reports
    }
    for values in reproduction.values():
        values["absolute_difference"] = abs(values["archived"] - values["factorial"])
        if values["absolute_difference"] > 1e-15:
            raise RuntimeError(f"original pilot cell failed exact reproduction: {reproduction}")

    archived_screen = json.loads(ARCHIVED_SCREEN_PATH.read_text())
    screen_cell = cells["selected_slot_dense_fit__natural_binary"]
    screen_reference = {
        name: {
            "archived_different_seed_and_padded_geometry": archived_screen["arms"][name]["success_all_menus"],
            "factorial_common_seed_and_34_report_columns": screen_cell["models"][name]["success_all_menus"],
        }
        for name in reports
    }
    for values in screen_reference.values():
        values["absolute_difference"] = abs(
            values["archived_different_seed_and_padded_geometry"] - values["factorial_common_seed_and_34_report_columns"]
        )

    receipt = {
        "status": "PASS",
        "scope": "post-review exploratory factorial on opened cached development data",
        "new_neural_forwards": 0,
        "new_formulas": 0,
        "fits": 8,
        "settings": SETTINGS,
        "versions": versions,
        "inputs": {
            str(BANK_PATH.relative_to(ROOT)): sha256(BANK_PATH),
            str(NATIVE_PATH.relative_to(ROOT)): sha256(NATIVE_PATH),
            str(ARCHIVED_PILOT_PATH.relative_to(ROOT)): sha256(ARCHIVED_PILOT_PATH),
            str(ARCHIVED_SCREEN_PATH.relative_to(ROOT)): sha256(ARCHIVED_SCREEN_PATH),
        },
        "cells": cells,
        "paired_uncertainty": paired,
        "factorial_contrasts": contrasts,
        "exact_original_pilot_reproduction": reproduction,
        "archived_screen_reference": screen_reference,
        "bootstrap": {
            "seed": 2026101804,
            "draws": 2000,
            "cluster": "assessment origin",
            "same_multiplicities_used_for_all_four cells": True,
        },
        "interpretation_boundary": (
            "The factorial diagnoses the earlier reversal under a common learner seed. It is post-review, "
            "opened-data analysis and does not turn either development cohort into protected confirmation."
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (HERE / "action_menu_encoding_factorial.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        HERE / "action_menu_encoding_factorial_arrays.npz",
        origin_ids=np.arange(3072, 4096, dtype=np.int32),
        bootstrap_multiplicities=multiplicities,
        **saved,
    )

    lines = [
        "# Cached action-menu × action-encoding factorial",
        "",
        "This post-review diagnostic uses zero new formulas and zero neural forwards. It fits the same two 100-tree report arms in four cells and pairs uncertainty by assessment origin.",
        "",
        "| Menu/training regime | Action encoding | Published-bit success | Continuous-post success | Continuous minus bits | Paired 95% interval |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for menu_name in menus:
        for encoding in encodings:
            cell_name = f"{menu_name}__{encoding}"
            cell = cells[cell_name]
            bounds = paired[cell_name]["paired_origin_interval"]["interval_2_5_97_5"]
            lines.append(
                f"| {menu_name} | {encoding} | {cell['models']['published_bits34']['success_all_menus']:.6f} | "
                f"{cell['models']['continuous_post34']['success_all_menus']:.6f} | {cell['continuous_minus_bits']:.6f} | "
                f"[{bounds[0]:.6f}, {bounds[1]:.6f}] |"
            )
    lines += ["", "## Factorial contrasts", ""]
    for name, value in contrasts.items():
        bounds = value["paired_origin_interval"]["interval_2_5_97_5"]
        lines.append(f"- `{name}`: {value['point']:.6f}, paired 95% interval [{bounds[0]:.6f}, {bounds[1]:.6f}].")
    lines += [
        "",
        "The all-slot/one-hot cell reproduces both original pilot selection rates exactly. The other cells use the same learner seed and report geometry to identify the menu and encoding contributions. This is an opened-data diagnostic, not a preregistered confirmation.",
    ]
    (HERE / "action_menu_encoding_factorial.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "cells": {name: value["continuous_minus_bits"] for name, value in cells.items()}, "interaction": contrasts["menu_by_encoding_interaction"]}, sort_keys=True))


if __name__ == "__main__":
    main()
