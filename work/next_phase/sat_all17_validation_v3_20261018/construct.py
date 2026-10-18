"""Construct the frozen semantic banks without importing Torch or model outputs."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
import platform
import numpy as np

from protocol import *
from pilot_common import CLAUSES, CLAUSE_TRUTH, encode, semantic_counts


def token_key(row: np.ndarray) -> bytes:
    row = np.asarray(row)
    if row.shape != (41,) or np.any((row < 0) | (row > 15)):
        raise ValueError("invalid token row")
    return bytes(map(int, row))


def clause_permutation(pair: tuple[int, int]) -> np.ndarray:
    variable = list(range(5))
    i, j = pair
    variable[i], variable[j] = variable[j], variable[i]
    return clause_variable_permutation(tuple(variable))


def clause_variable_permutation(variable: tuple[int, ...]) -> np.ndarray:
    lookup = {tuple(map(int, c)): i for i, c in enumerate(CLAUSES)}
    mapping = []
    for clause in CLAUSES:
        literals = []
        for literal in map(int, clause):
            sign = literal >= 5
            v = variable[literal % 5]
            literals.append(v + (5 if sign else 0))
        mapping.append(lookup[tuple(sorted(literals, key=lambda x: x % 5))])
    return np.asarray(mapping, dtype=np.int16)


def load_prior_tokens() -> set[bytes]:
    if sha256(PRIOR_LEDGER) != PRIOR_LEDGER_SHA256 or sha256(PRIOR_RENAMING) != PRIOR_RENAMING_SHA256:
        raise RuntimeError("historical input hash mismatch")
    prior = set()
    with np.load(PRIOR_LEDGER, allow_pickle=False) as z:
        if tuple(z.files) != PRIOR_LEDGER_KEYS or z["tokens"].reshape(-1, 41).shape[0] != PRIOR_LEDGER_TOKEN_ROWS:
            raise RuntimeError("prior ledger inventory mismatch")
        prior.update(token_key(row) for row in z["tokens"].reshape(-1, 41))
    with np.load(PRIOR_RENAMING, allow_pickle=False) as z:
        if tuple(z.files) != PRIOR_RENAMING_KEYS or z["tokens"].reshape(-1, 41).shape[0] != PRIOR_RENAMING_TOKEN_ROWS:
            raise RuntimeError("prior renaming inventory mismatch")
        prior.update(token_key(row) for row in z["tokens"].reshape(-1, 41))
    return prior


def canonical_orbit_id(formula: np.ndarray, permutation_maps: list[np.ndarray]) -> bytes:
    canonical = min(token_key(encode(cmap[np.asarray(formula)][None, :])[0]) for cmap in permutation_maps)
    return hashlib.sha256(canonical).hexdigest().encode("ascii")


def save_npz_exclusive(path, **arrays):
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)


def save_json_exclusive(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main() -> None:
    frozen = verify_frozen_packet()
    if any((HERE / name).exists() for name in ("candidate_bank.npz", "selected_bank.npz", "construction.json")):
        raise RuntimeError("exclusive construction outputs already exist")
    if platform.python_version() != ANALYSIS_PYTHON or np.__version__ != ANALYSIS_NUMPY:
        raise RuntimeError("construction environment drift")
    prior = load_prior_tokens()
    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
    clauses = rng.integers(0, len(CLAUSES), size=(PROPOSALS, 10), dtype=np.int16)
    counts = semantic_counts(clauses)
    tokens = encode(clauses)
    keys = [token_key(row) for row in tokens]
    save_npz_exclusive(HERE / "candidate_bank.npz", clauses=clauses, counts=counts)

    selected = np.full((len(UNIQUE_ASSIGNMENTS), len(STRATA), QUOTA), -1, dtype=np.int32)
    diagnostics = Counter()
    for ai, assignment0 in enumerate(UNIQUE_ASSIGNMENTS):
        assignment = int(assignment0)
        pools = [[] for _ in STRATA]
        negative_pools = {3: {"near": [], "far": []}, 4: {"near": [], "far": []}}
        seen = set()
        for proposal, (formula, semantic, key) in enumerate(zip(clauses, counts, keys)):
            if key in prior:
                diagnostics["historical_overlap"] += 1
                continue
            if key in seen:
                continue
            stratum = classify_stratum(semantic, assignment)
            if stratum is None:
                continue
            seen.add(key)
            ranked = (domain_hash("sat-all17/base-v2", bytes([assignment, stratum]), bytes(formula), proposal.to_bytes(8, "big")), proposal)
            if stratum in negative_pools:
                distance = min_solution_hamming(semantic, assignment)
                negative_pools[stratum]["near" if distance == 1 else "far"].append(ranked)
            else:
                pools[stratum].append(ranked)
        for stratum, pool in enumerate(pools):
            if stratum in negative_pools:
                near = sorted(negative_pools[stratum]["near"])
                far = sorted(negative_pools[stratum]["far"])
                if len(near) < QUOTA // 2 or len(far) < QUOTA // 2:
                    raise RuntimeError(f"near/far quota shortfall assignment={assignment} stratum={STRATA[stratum]} near={len(near)} far={len(far)}")
                chosen = near[:QUOTA // 2] + far[:QUOTA // 2]
                chosen.sort()
                selected[ai, stratum] = [proposal for _, proposal in chosen]
                diagnostics[f"eligible_a{assignment}_{STRATA[stratum]}_near"] = len(near)
                diagnostics[f"eligible_a{assignment}_{STRATA[stratum]}_far"] = len(far)
            else:
                pool.sort()
                if len(pool) < QUOTA:
                    raise RuntimeError(f"quota shortfall assignment={assignment} stratum={STRATA[stratum]} count={len(pool)}")
                selected[ai, stratum] = [proposal for _, proposal in pool[:QUOTA]]
                diagnostics[f"eligible_a{assignment}_{STRATA[stratum]}"] = len(pool)

    base_formula_ids = np.unique(selected)
    base_clauses = clauses[base_formula_ids]
    base_tokens = tokens[base_formula_ids]
    if any(token_key(row) in prior for row in base_tokens):
        raise RuntimeError("selected historical overlap")

    assignment_set = set(map(int, UNIQUE_ASSIGNMENTS))
    permutations = [clause_permutation(pair) for pair in TRANSPOSITIONS]
    transform_rows = []
    transform_tokens = []
    for ti, (pair, cmap) in enumerate(zip(TRANSPOSITIONS, permutations)):
        for ai, assignment0 in enumerate(UNIQUE_ASSIGNMENTS):
            assignment = int(assignment0)
            target = swap_assignment(assignment, pair)
            if target not in assignment_set:
                continue
            for stratum in range(len(STRATA)):
                accepted = 0
                for proposal in selected[ai, stratum]:
                    transformed = cmap[clauses[int(proposal)]]
                    transformed_token = encode(transformed[None, :])[0]
                    transformed_key = token_key(transformed_token)
                    if transformed_key == keys[int(proposal)] or transformed_key in prior:
                        continue
                    transformed_counts = semantic_counts(transformed[None, :])[0]
                    if classify_stratum(transformed_counts, target) != stratum:
                        raise RuntimeError("transformation semantic mismatch")
                    transform_rows.append((ti, assignment, target, stratum, int(proposal)))
                    transform_tokens.append(transformed_token)
                    accepted += 1
                    if accepted == TRANSFORM_QUOTA:
                        break
                if accepted != TRANSFORM_QUOTA:
                    raise RuntimeError(f"transform quota shortfall pair={pair} assignment={assignment} stratum={stratum} count={accepted}")

    transform_tokens = np.asarray(transform_tokens, dtype=np.int16)
    all_tokens = np.concatenate([base_tokens, transform_tokens], axis=0)
    unique_tokens, inverse = np.unique(all_tokens, axis=0, return_inverse=True)
    base_lookup = inverse[:len(base_tokens)]
    transform_lookup = inverse[len(base_tokens):]
    base_id_lookup = {int(fid): int(base_lookup[i]) for i, fid in enumerate(base_formula_ids)}
    selection_unique_ids = np.vectorize(base_id_lookup.__getitem__, otypes=[np.int32])(selected)
    negative_distance_class = np.zeros_like(selected, dtype=np.int8)
    for ai, assignment0 in enumerate(UNIQUE_ASSIGNMENTS):
        assignment = int(assignment0)
        for stratum in (3, 4):
            for qi, proposal in enumerate(selected[ai, stratum]):
                distance = min_solution_hamming(counts[int(proposal)], assignment)
                negative_distance_class[ai, stratum, qi] = 1 if distance == 1 else 2
    transform_rows = np.asarray(transform_rows, dtype=np.int32)

    decoded_unique_clauses = np.asarray([[int(np.where(np.all(CLAUSES == row[None, :], axis=1))[0][0]) for row in formula]
                                         for formula in np.asarray([[[t[1 + 4*k], t[2 + 4*k]] for k in range(10)] for t in unique_tokens])], dtype=np.int16)
    all_permutation_maps = [clause_variable_permutation(p) for p in ALL_VARIABLE_PERMUTATIONS]
    orbit_ids = np.asarray([canonical_orbit_id(formula, all_permutation_maps) for formula in decoded_unique_clauses], dtype="S64")
    save_npz_exclusive(
        HERE / "selected_bank.npz",
        unique_tokens=unique_tokens.astype(np.int16),
        unique_clauses=decoded_unique_clauses,
        orbit_ids=orbit_ids,
        base_proposal_ids=selected,
        base_unique_ids=selection_unique_ids,
        negative_distance_class=negative_distance_class,
        transform_rows=transform_rows,
        transform_unique_ids=transform_lookup.astype(np.int32),
        unit_ids=UNITS,
        unit_assignment_ids=ASSIGNMENT_IDS,
        unique_assignment_ids=UNIQUE_ASSIGNMENTS,
    )
    replay_ids = np.array(sorted(range(len(unique_tokens)), key=lambda i: domain_hash("sat-all17/replay-v1", token_key(unique_tokens[i])))[:8], dtype=np.int32)
    summary = {
        "status": "PASS",
        "schema": "sat-all17-construction-v3",
        "proposals": PROPOSALS,
        "seed": SAMPLE_SEED,
        "quota_per_assignment_stratum": QUOTA,
        "transform_quota_per_edge_stratum": TRANSFORM_QUOTA,
        "base_assignment_stratum_rows": int(selected.size),
        "base_distinct_formulas": int(len(base_formula_ids)),
        "transform_rows": int(len(transform_rows)),
        "unique_native_inputs": int(len(unique_tokens)),
        "historical_ledger_size": int(len(prior)),
        "replay_unique_ids": replay_ids.tolist(),
        "diagnostics": dict(diagnostics),
        "input_sha256": {"prior_ledger": sha256(PRIOR_LEDGER), "prior_renaming": sha256(PRIOR_RENAMING)},
        "candidate_bank_sha256": sha256(HERE / "candidate_bank.npz"),
        "selected_bank_sha256": sha256(HERE / "selected_bank.npz"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        **frozen,
    }
    save_json_exclusive(HERE / "construction.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
