"""Independent replay of the frozen SAT-renaming proposal and selection stream.

This script deliberately imports no module from the original study.  It
reimplements the five-variable formula grammar, Boolean semantics, PCG64 draw,
domain-separated hash ordering, and exclusion rules from the frozen contract,
then compares every reconstructed array with the saved study artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
WORK = PACKAGE / "sat_renaming_case" / "work"
STUDY = WORK / "sat_renaming_launch_recovery"
CONTRACT = WORK / "next_phase" / "sat_renaming_execution_contract.md"

CONTRACT_SHA256 = "4b9dc2df3002dbbe56cbcc5d38c119ace801f16e2f3bd018d345deb4eda7af8c"
SAMPLE_SEED = 2026101604
PROPOSALS = 100_000
QUOTA = 32
PAIRS = ((27, 23), (25, 21))
ASSIGNMENT_IDS = np.array((27, 23, 25, 21), dtype=np.int64)
VARIABLE_PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
                  (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
ACTION_FIELDS = ("slot", "old_clause_id", "kept_literal", "new_variable", "new_clause_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def clause_from_id(clause_id: int) -> tuple[int, int]:
    pair_index, signs = divmod(int(clause_id), 4)
    left, right = VARIABLE_PAIRS[pair_index]
    return left + 5 * (signs // 2), right + 5 * (signs % 2)


CLAUSES = tuple(clause_from_id(index) for index in range(40))
CLAUSE_LOOKUP = {clause: index for index, clause in enumerate(CLAUSES)}
ALLOWED = np.array(
    [index for index, clause in enumerate(CLAUSES)
     if {clause[0] % 5, clause[1] % 5} != {1, 2}],
    dtype=np.int16,
)


def assignment_bits(index: int) -> tuple[bool, ...]:
    return tuple(bool(index & (1 << (4 - variable))) for variable in range(5))


ASSIGNMENTS = tuple(assignment_bits(index) for index in range(32))


def literal_value(literal: int, assignment: int) -> bool:
    value = ASSIGNMENTS[assignment][literal % 5]
    return value if literal < 5 else not value


def clause_value(clause_id: int, assignment: int) -> bool:
    left, right = CLAUSES[int(clause_id)]
    return literal_value(left, assignment) or literal_value(right, assignment)


CLAUSE_TRUTH = np.array(
    [[clause_value(clause_id, assignment) for assignment in range(32)]
     for clause_id in range(40)],
    dtype=bool,
)


def semantic_counts(formulas: np.ndarray) -> np.ndarray:
    counts = np.zeros((len(formulas), 32), dtype=np.int16)
    for slot in range(10):
        counts += (~CLAUSE_TRUTH[formulas[:, slot]]).astype(np.int16)
    return counts


def encode(formula: np.ndarray) -> bytes:
    tokens: list[int] = []
    for clause_id in formula:
        left, right = CLAUSES[int(clause_id)]
        tokens.extend((10, left, right, 11))
    tokens.append(12)
    return bytes(tokens)


def rename_literal(literal: int) -> int:
    variable = literal % 5
    renamed = 2 if variable == 1 else 1 if variable == 2 else variable
    return renamed + (5 if literal >= 5 else 0)


def rename_clause_id(clause_id: int) -> int:
    renamed = tuple(sorted((rename_literal(token) for token in CLAUSES[int(clause_id)]),
                           key=lambda token: token % 5))
    return CLAUSE_LOOKUP[renamed]


CLAUSE_RENAME = np.array([rename_clause_id(index) for index in range(40)], dtype=np.int16)


def orbit_bytes(formula: np.ndarray) -> bytes:
    return min(encode(formula), encode(CLAUSE_RENAME[formula]))


def domain_digest(domain: str, *parts: bytes) -> bytes:
    digest = hashlib.sha256(domain.encode("ascii") + b"\0")
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


def eligible_actions(formula: np.ndarray, counts: np.ndarray, pair_index: int,
                     positive: bool) -> list[tuple[int, int, int, int, int]]:
    anchor, partner = PAIRS[pair_index]
    choices = []
    allowed = set(map(int, ALLOWED))
    for slot, old_clause_id_raw in enumerate(formula):
        old_clause_id = int(old_clause_id_raw)
        if CLAUSE_TRUTH[old_clause_id, anchor]:
            continue
        old_clause = CLAUSES[old_clause_id]
        for kept in old_clause:
            if literal_value(kept, anchor) or not literal_value(kept, partner):
                continue
            old_variables = {old_clause[0] % 5, old_clause[1] % 5}
            for variable in range(5):
                if variable in old_variables:
                    continue
                added = variable if ASSIGNMENTS[anchor][variable] else variable + 5
                new_clause = tuple(sorted((kept, added), key=lambda token: token % 5))
                new_clause_id = CLAUSE_LOOKUP[new_clause]
                if new_clause_id not in allowed:
                    continue
                after = counts - (~CLAUSE_TRUTH[old_clause_id]).astype(np.int16)
                after = after + (~CLAUSE_TRUTH[new_clause_id]).astype(np.int16)
                solutions = np.flatnonzero(after == 0)
                if positive:
                    if not np.array_equal(solutions, np.array([anchor])):
                        continue
                elif len(solutions):
                    continue
                if after[partner] != counts[partner] or after[anchor] != counts[anchor] - 1:
                    raise AssertionError("edit-count algebra")
                choices.append((slot, old_clause_id, kept, variable, new_clause_id))
    return choices


def quartet(formula: np.ndarray, action: tuple[int, ...]) -> np.ndarray:
    edited = np.array(formula, dtype=np.int16, copy=True)
    edited[action[0]] = action[4]
    return np.stack((formula, edited, CLAUSE_RENAME[formula], CLAUSE_RENAME[edited]))


def compare_array(name: str, actual: np.ndarray, expected: np.ndarray,
                  checks: list[dict]) -> None:
    equal = actual.shape == expected.shape and actual.dtype == expected.dtype and np.array_equal(actual, expected)
    detail = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_sha256": array_sha256(actual),
        "expected_sha256": array_sha256(expected),
    }
    checks.append({"name": name, "pass": bool(equal), "detail": detail})
    if not equal:
        raise AssertionError(name)


def main() -> None:
    started = time.monotonic()
    checks: list[dict] = []
    inputs: dict[str, dict] = {}

    def record(path: Path) -> None:
        inputs[str(path.resolve())] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    record(CONTRACT)
    if inputs[str(CONTRACT.resolve())]["sha256"] != CONTRACT_SHA256:
        raise AssertionError("contract identity")

    prior_path = STUDY / "prior_ledger_tokens.npz"
    record(prior_path)
    with np.load(prior_path, allow_pickle=False) as saved_prior:
        prior_set = {bytes(map(int, row)) for row in saved_prior["tokens"]}

    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
    candidate_clauses = ALLOWED[
        rng.integers(0, len(ALLOWED), size=(PROPOSALS, 10), dtype=np.int16)
    ]
    candidate_counts = semantic_counts(candidate_clauses)

    diagnostics = Counter()
    pair_diagnostics = [Counter(), Counter()]
    eligibility = np.zeros((PROPOSALS, 2), dtype=np.int8)
    multiplicity = np.zeros((PROPOSALS, 2), dtype=np.int16)
    records = []
    seen_originals: set[bytes] = set()
    for trial, (formula, counts) in enumerate(zip(candidate_clauses, candidate_counts)):
        key = bytes(map(int, formula))
        if key in seen_originals:
            diagnostics["duplicate_original"] += 1
            eligibility[trial, :] = 1
            continue
        seen_originals.add(key)
        if np.any(counts == 0):
            diagnostics["initially_sat"] += 1
            eligibility[trial, :] = 2
            continue
        diagnostics["unique_initially_unsat"] += 1
        orbit = orbit_bytes(formula)
        for pair_index, (anchor, partner) in enumerate(PAIRS):
            pair_summary = pair_diagnostics[pair_index]
            if counts[partner] < 2:
                pair_summary["partner_count_below_two"] += 1
                eligibility[trial, pair_index] = 3
                continue
            positive = counts[anchor] == 1
            choices = eligible_actions(formula, counts, pair_index, positive)
            multiplicity[trial, pair_index] = len(choices)
            if not choices:
                pair_summary["no_eligible_action_positive" if positive else "no_eligible_action_negative"] += 1
                eligibility[trial, pair_index] = 4
                continue
            action = min(
                choices,
                key=lambda item: (
                    domain_digest("sat-renaming/action-v1", bytes([pair_index]), orbit, bytes(item)),
                    item,
                ),
            )
            cohort = 0 if positive else 1
            eligibility[trial, pair_index] = 5 + cohort
            pair_summary["eligible_positive" if positive else "eligible_negative"] += 1
            records.append({
                "trial": trial,
                "pair": pair_index,
                "cohort": cohort,
                "action": action,
                "multiplicity": len(choices),
                "orbit": orbit,
            })

    eligible_trial = np.array([row["trial"] for row in records], dtype=np.int64)
    eligible_pair = np.array([row["pair"] for row in records], dtype=np.int8)
    eligible_cohort = np.array([row["cohort"] for row in records], dtype=np.int8)
    eligible_action = np.array([row["action"] for row in records], dtype=np.int16).reshape(-1, 5)

    pools = [defaultdict(lambda: [[], []]), defaultdict(lambda: [[], []])]
    for row in records:
        pools[row["pair"]][row["action"]][row["cohort"]].append(row)
    descriptors = []
    for pair_index in range(2):
        for action, cohorts in pools[pair_index].items():
            for cohort in cohorts:
                cohort.sort(key=lambda row: (
                    domain_digest(
                        "sat-renaming/origin-v1",
                        bytes([pair_index]),
                        bytes(action),
                        bytes([row["cohort"]]),
                        row["orbit"],
                        encode(candidate_clauses[row["trial"]]),
                    ),
                    row["trial"],
                ))
        available = [action for action, cohorts in pools[pair_index].items() if cohorts[0] and cohorts[1]]
        pair_diagnostics[pair_index]["descriptors_any"] = len(pools[pair_index])
        pair_diagnostics[pair_index]["descriptors_both_cohorts"] = len(available)
        descriptors.append(iter(sorted(
            available,
            key=lambda action: (
                domain_digest("sat-renaming/descriptor-v1", bytes([pair_index]), bytes(action)),
                action,
            ),
        )))

    used_orbits: set[bytes] = set()
    used_endpoints: set[bytes] = set()
    selected = []

    def assess(row: dict, extra_endpoints=(), extra_orbits=()):
        formulas = quartet(candidate_clauses[row["trial"]], row["action"])
        tokens = [encode(formula) for formula in formulas]
        if len(set(tokens)) != 4:
            return None
        if row["orbit"] in used_orbits or row["orbit"] in extra_orbits:
            return None
        if any(token in prior_set for token in tokens):
            return None
        if any(token in used_endpoints or token in extra_endpoints for token in tokens):
            return None
        return formulas, tokens

    for _round in range(QUOTA):
        for pair_index in range(2):
            for action in descriptors[pair_index]:
                positive = pools[pair_index][action][0][0]
                positive_data = assess(positive)
                if positive_data is None:
                    continue
                negative = None
                negative_data = None
                for candidate in pools[pair_index][action][1]:
                    candidate_data = assess(candidate, positive_data[1], (positive["orbit"],))
                    if candidate_data is not None:
                        negative = candidate
                        negative_data = candidate_data
                        break
                if negative is None:
                    continue
                selected.append((pair_index, action, positive, negative, positive_data, negative_data))
                used_orbits.update((positive["orbit"], negative["orbit"]))
                used_endpoints.update(positive_data[1] + negative_data[1])
                break
            else:
                raise AssertionError("selection support shortfall")

    selected_clauses = np.stack([
        np.stack((entry[4][0], entry[5][0])) for entry in selected
    ]).astype(np.int16)
    selected_tokens = np.array([
        [[list(token) for token in entry[4][1]], [list(token) for token in entry[5][1]]]
        for entry in selected
    ], dtype=np.int64)
    selected_counts = semantic_counts(selected_clauses.reshape(-1, 10)).reshape(64, 2, 4, 32)
    selected_labels = np.where((selected_counts == 0).any(axis=-1), 13, 14).astype(np.int16)
    expected_bits = (selected_counts[..., ASSIGNMENT_IDS] == 0).astype(np.int8)
    selected_pairs = np.array([entry[0] for entry in selected], dtype=np.int8)
    selected_trials = np.array([[entry[2]["trial"], entry[3]["trial"]] for entry in selected], dtype=np.int64)

    candidate_path = STUDY / "candidate_bank.npz"
    selection_path = STUDY / "candidate_selection_data.npz"
    selected_path = STUDY / "selected_bank.npz"
    metadata_path = STUDY / "selected_bank.json"
    for path in (candidate_path, selection_path, selected_path, metadata_path):
        record(path)

    with np.load(candidate_path, allow_pickle=False) as saved:
        compare_array("candidate clauses", candidate_clauses, saved["clauses"], checks)
        compare_array("candidate semantic counts", candidate_counts, saved["counts"], checks)
    with np.load(selection_path, allow_pickle=False) as saved:
        compare_array("eligibility codes", eligibility, saved["eligibility_codes"], checks)
        compare_array("action multiplicities", multiplicity, saved["action_multiplicity"], checks)
        compare_array("eligible trial order", eligible_trial, saved["eligible_trial"], checks)
        compare_array("eligible pair order", eligible_pair, saved["eligible_pair"], checks)
        compare_array("eligible cohort order", eligible_cohort, saved["eligible_cohort"], checks)
        compare_array("eligible action order", eligible_action, saved["eligible_action"], checks)
    with np.load(selected_path, allow_pickle=False) as saved:
        for name, actual in (
            ("clauses", selected_clauses),
            ("tokens", selected_tokens),
            ("counts", selected_counts),
            ("labels", selected_labels),
            ("expected_unit_bits", expected_bits),
            ("pair_index", selected_pairs),
            ("trial_ids", selected_trials),
        ):
            compare_array("selected " + name, actual, saved[name], checks)

    metadata = json.loads(metadata_path.read_text())
    reconstructed_blocks = []
    for block_id, entry in enumerate(selected):
        pair_index, action, positive, negative, positive_data, negative_data = entry
        reconstructed_blocks.append({
            "block_id": block_id,
            "pair": "P" + str(pair_index + 1),
            "pair_index": pair_index,
            "trial_ids": [positive["trial"], negative["trial"]],
            "action": dict(zip(ACTION_FIELDS, map(int, action))),
            "orbit_ids": [hashlib.sha256(row["orbit"]).hexdigest() for row in (positive, negative)],
            "endpoint_sha256": [
                [hashlib.sha256(token).hexdigest() for token in data[1]]
                for data in (positive_data, negative_data)
            ],
            "eligible_action_multiplicities": [positive["multiplicity"], negative["multiplicity"]],
        })
    metadata_equal = reconstructed_blocks == metadata["blocks"]
    checks.append({"name": "selected metadata blocks", "pass": metadata_equal,
                   "detail": {"reconstructed": len(reconstructed_blocks), "saved": len(metadata["blocks"])}})
    if not metadata_equal:
        raise AssertionError("selected metadata blocks")

    replay_arrays = HERE / "replay_arrays.npz"
    with replay_arrays.open("xb") as handle:
        np.savez_compressed(
            handle,
            eligibility_codes=eligibility,
            action_multiplicity=multiplicity,
            eligible_trial=eligible_trial,
            eligible_pair=eligible_pair,
            eligible_cohort=eligible_cohort,
            eligible_action=eligible_action,
            selected_pair_index=selected_pairs,
            selected_trial_ids=selected_trials,
            selected_clauses=selected_clauses,
            selected_tokens=selected_tokens,
        )

    report = {
        "status": "PASS",
        "scope": "Independent regeneration of the frozen 100,000-proposal PCG64 stream, semantic eligibility, domain-separated hash ordering, prior-ledger exclusions, and final selected bank",
        "limits": [
            "Replays the declared NumPy PCG64 algorithm and therefore remains conditional on NumPy's implementation of that named generator",
            "Does not rerun the neural checkpoint or establish external preregistration",
            "Verifies deterministic construction identity, not representativeness of the selected finite cohort",
        ],
        "utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "imports_original_study_modules": False,
        "new_neural_forwards": 0,
        "learner_fits": 0,
        "proposals_regenerated": PROPOSALS,
        "selected_blocks": len(selected),
        "selected_origin_orbits": len(used_orbits),
        "selected_endpoints": len(used_endpoints),
        "diagnostics": dict(diagnostics),
        "pair_diagnostics": [dict(summary) for summary in pair_diagnostics],
        "checks": checks,
        "inputs": inputs,
        "output": {"path": replay_arrays.name, "sha256": sha256(replay_arrays), "bytes": replay_arrays.stat().st_size},
    }
    report_path = HERE / "report.json"
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": report["status"],
        "elapsed_seconds": report["elapsed_seconds"],
        "proposals_regenerated": report["proposals_regenerated"],
        "selected_blocks": report["selected_blocks"],
        "checks": len(checks),
        "report_sha256": sha256(report_path),
        "output_sha256": report["output"]["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
