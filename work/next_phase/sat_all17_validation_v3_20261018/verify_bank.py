"""Independent replay of selection, decoding, truth tables and transpositions."""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import numpy as np

from protocol import *
from pilot_common import CLAUSES


def token_key(row: np.ndarray) -> bytes:
    row = np.asarray(row)
    if row.shape != (41,) or np.any((row < 0) | (row > 15)):
        raise ValueError("invalid token row")
    return bytes(map(int, row))


def encode_formula(formula: np.ndarray) -> np.ndarray:
    out = np.empty(41, dtype=np.int16)
    for slot, clause_id in enumerate(map(int, formula)):
        out[4 * slot:4 * slot + 4] = (10, int(CLAUSES[clause_id, 0]), int(CLAUSES[clause_id, 1]), 11)
    out[40] = 12
    return out


def decode(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row)
    if row.shape != (41,) or row[-1] != 12:
        raise ValueError("bad token shape or final token")
    lookup = {tuple(map(int, c)): i for i, c in enumerate(CLAUSES)}
    out = []
    for slot in range(10):
        start = 4 * slot
        if row[start] != 10 or row[start + 3] != 11:
            raise ValueError("bad clause delimiters")
        out.append(lookup[(int(row[start + 1]), int(row[start + 2]))])
    return np.asarray(out, dtype=np.int16)


def truth_counts(formula: np.ndarray) -> np.ndarray:
    counts = np.zeros(32, dtype=np.int16)
    for assignment in range(32):
        bits = [(assignment >> (4 - i)) & 1 for i in range(5)]
        for clause_id in map(int, formula):
            values = []
            for literal in map(int, CLAUSES[clause_id]):
                value = bool(bits[literal % 5])
                values.append(not value if literal >= 5 else value)
            counts[assignment] += int(not any(values))
    return counts


def independent_candidate_counts(formulas: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Second vectorised Boolean oracle, independent of pilot_common.CLAUSE_TRUTH."""
    assignments = ((np.arange(32, dtype=np.uint8)[:, None] >> np.arange(4, -1, -1)) & 1).astype(bool)
    literal_truth = np.concatenate([assignments.T, (~assignments).T], axis=0)
    out = np.empty((len(formulas), 32), dtype=np.int16)
    for start in range(0, len(formulas), batch_size):
        block = np.asarray(formulas[start:start + batch_size], dtype=np.int16)
        literals = CLAUSES[block]
        clause_satisfied = literal_truth[literals].any(axis=2)
        out[start:start + len(block)] = (~clause_satisfied).sum(axis=1, dtype=np.int16)
    return out


def local_stratum(counts: np.ndarray, assignment: int) -> int:
    solutions = np.flatnonzero(counts == 0)
    target = assignment in solutions
    if len(solutions) == 1 and target: return 0
    if len(solutions) >= 2 and target: return 1
    if len(solutions) == 0: return 2
    if len(solutions) == 1: return 3
    return 4


def local_min_hamming(counts: np.ndarray, assignment: int) -> int | None:
    solutions = np.flatnonzero(counts == 0)
    return None if not len(solutions) else min((int(x) ^ assignment).bit_count() for x in solutions)


def local_domain_hash(domain: str, *parts: bytes) -> bytes:
    h = hashlib.sha256(domain.encode("ascii") + b"\0")
    for part in parts:
        h.update(len(part).to_bytes(8, "big")); h.update(part)
    return h.digest()


def local_swap_assignment(assignment: int, pair: tuple[int, int]) -> int:
    bits = [(assignment >> (4 - i)) & 1 for i in range(5)]
    bits[pair[0]], bits[pair[1]] = bits[pair[1]], bits[pair[0]]
    return sum(bit << (4 - i) for i, bit in enumerate(bits))


def transform_formula(formula: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    variables = list(range(5)); variables[pair[0]], variables[pair[1]] = variables[pair[1]], variables[pair[0]]
    return permute_formula(formula, tuple(variables))


def permute_formula(formula: np.ndarray, variables: tuple[int, ...]) -> np.ndarray:
    lookup = {tuple(map(int, c)): i for i, c in enumerate(CLAUSES)}
    out = []
    for clause_id in map(int, formula):
        literals = []
        for literal in map(int, CLAUSES[clause_id]):
            literals.append(variables[literal % 5] + (5 if literal >= 5 else 0))
        out.append(lookup[tuple(sorted(literals, key=lambda x: x % 5))])
    return np.asarray(out, dtype=np.int16)


def load_prior() -> set[bytes]:
    if sha256(PRIOR_LEDGER) != PRIOR_LEDGER_SHA256 or sha256(PRIOR_RENAMING) != PRIOR_RENAMING_SHA256:
        raise RuntimeError("historical hash mismatch")
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


def main() -> None:
    frozen = verify_frozen_packet()
    if platform.python_version() != ANALYSIS_PYTHON or np.__version__ != ANALYSIS_NUMPY:
        raise RuntimeError("verification environment drift")
    construction = json.loads((HERE / "construction.json").read_text())
    if any(construction.get(key) != frozen[key] for key in ("freeze_sha256", "code_sha256", "selfcheck_sha256", "source_sha256")):
        raise RuntimeError("construction used a different frozen code packet")
    if construction.get("status") != "PASS" or construction["candidate_bank_sha256"] != sha256(HERE / "candidate_bank.npz"):
        raise RuntimeError("candidate bank binding mismatch")
    with np.load(HERE / "candidate_bank.npz", allow_pickle=False) as z:
        candidate_clauses, candidate_counts = z["clauses"].copy(), z["counts"].copy()
    regenerated = np.random.Generator(np.random.PCG64(SAMPLE_SEED)).integers(
        0, len(CLAUSES), size=(PROPOSALS, 10), dtype=np.int16,
    )
    if not np.array_equal(regenerated, candidate_clauses):
        raise RuntimeError("candidate proposal stream does not replay from the frozen seed")
    if not np.array_equal(independent_candidate_counts(candidate_clauses), candidate_counts):
        raise RuntimeError("candidate truth-count array fails independent full replay")
    with np.load(HERE / "selected_bank.npz", allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in z.files}
    if construction["selected_bank_sha256"] != sha256(HERE / "selected_bank.npz"):
        raise RuntimeError("selected bank binding mismatch")
    tokens = bank["unique_tokens"]
    clauses = np.stack([decode(row) for row in tokens])
    if not np.array_equal(clauses, bank["unique_clauses"]):
        raise RuntimeError("independent token decode mismatch")
    counts = np.stack([truth_counts(row) for row in clauses])
    expected_orbits = []
    for formula in clauses:
        canonical = min(token_key(encode_formula(permute_formula(formula, permutation)))
                        for permutation in itertools.permutations(range(5)))
        expected_orbits.append(hashlib.sha256(canonical).hexdigest().encode("ascii"))
    if not np.array_equal(np.asarray(expected_orbits, dtype="S64"), bank["orbit_ids"]):
        raise RuntimeError("independent full-permutation orbit-id mismatch")
    prior = load_prior()
    if any(token_key(row) in prior for row in tokens):
        raise RuntimeError("historical overlap")

    candidate_keys = [token_key(encode_formula(row)) for row in candidate_clauses]
    expected_selected = np.full_like(bank["base_proposal_ids"], -1)
    selection_checks = 0
    for ai, assignment0 in enumerate(bank["unique_assignment_ids"]):
        assignment = int(assignment0)
        pools = [[] for _ in STRATA]
        split = {3: {"near": [], "far": []}, 4: {"near": [], "far": []}}
        seen = set()
        for proposal, (formula, semantic, key) in enumerate(zip(candidate_clauses, candidate_counts, candidate_keys)):
            if key in prior or key in seen: continue
            stratum = local_stratum(semantic, assignment)
            seen.add(key)
            ranked = (local_domain_hash("sat-all17/base-v2", bytes([assignment, stratum]), bytes(formula), proposal.to_bytes(8, "big")), proposal)
            if stratum in split:
                distance = local_min_hamming(semantic, assignment)
                split[stratum]["near" if distance == 1 else "far"].append(ranked)
            else:
                pools[stratum].append(ranked)
        for stratum in range(len(STRATA)):
            if stratum in split:
                chosen = sorted(split[stratum]["near"])[:QUOTA // 2] + sorted(split[stratum]["far"])[:QUOTA // 2]
                chosen.sort()
            else:
                chosen = sorted(pools[stratum])[:QUOTA]
            expected_selected[ai, stratum] = [proposal for _, proposal in chosen]
            selection_checks += len(chosen)
    if not np.array_equal(expected_selected, bank["base_proposal_ids"]):
        raise RuntimeError("base hash-order selection mismatch")

    base_checks = 0
    expected_distance_class = np.zeros_like(bank["base_proposal_ids"], dtype=np.int8)
    for ai, assignment0 in enumerate(bank["unique_assignment_ids"]):
        assignment = int(assignment0); ids = bank["base_unique_ids"][ai]
        if len(np.unique(ids)) != len(STRATA) * QUOTA: raise RuntimeError("within-assignment duplicate")
        for stratum in range(len(STRATA)):
            for uid in ids[stratum]:
                if local_stratum(counts[int(uid)], assignment) != stratum: raise RuntimeError("base semantic mismatch")
                base_checks += 1
            if stratum in (3, 4):
                for qi, uid in enumerate(ids[stratum]):
                    distance = local_min_hamming(counts[int(uid)], assignment)
                    expected_distance_class[ai, stratum, qi] = 1 if distance == 1 else 2
                values, frequencies = np.unique(expected_distance_class[ai, stratum], return_counts=True)
                if not np.array_equal(values, np.array([1, 2], dtype=np.int8)) or not np.array_equal(frequencies, np.array([16, 16])):
                    raise RuntimeError("independent near/far balance mismatch")
    if not np.array_equal(expected_distance_class, bank["negative_distance_class"]):
        raise RuntimeError("stored near/far labels fail independent verification")

    expected_rows, expected_tokens = [], []
    assignment_set = set(map(int, bank["unique_assignment_ids"]))
    for ti, pair in enumerate(TRANSPOSITIONS):
        for ai, assignment0 in enumerate(bank["unique_assignment_ids"]):
            assignment = int(assignment0); target = local_swap_assignment(assignment, pair)
            if target not in assignment_set: continue
            for stratum in range(len(STRATA)):
                accepted = 0
                for proposal in bank["base_proposal_ids"][ai, stratum]:
                    transformed = transform_formula(candidate_clauses[int(proposal)], pair)
                    transformed_token = encode_formula(transformed)
                    if token_key(transformed_token) == candidate_keys[int(proposal)] or token_key(transformed_token) in prior: continue
                    if local_stratum(truth_counts(transformed), target) != stratum: raise RuntimeError("transform semantic mismatch")
                    expected_rows.append((ti, assignment, target, stratum, int(proposal))); expected_tokens.append(transformed_token)
                    accepted += 1
                    if accepted == TRANSFORM_QUOTA: break
                if accepted != TRANSFORM_QUOTA: raise RuntimeError("transform quota mismatch")
    if not np.array_equal(np.asarray(expected_rows, dtype=np.int32), bank["transform_rows"]):
        raise RuntimeError("first-eight transform selection mismatch")
    for expected, uid in zip(expected_tokens, bank["transform_unique_ids"]):
        if not np.array_equal(expected, tokens[int(uid)]): raise RuntimeError("transformed token mismatch")

    replay = sorted(range(len(tokens)), key=lambda i: local_domain_hash("sat-all17/replay-v1", token_key(tokens[i])))[:8]
    if replay != construction["replay_unique_ids"]: raise RuntimeError("replay-id mismatch")
    result = {"status": "PASS", "base_semantic_checks": base_checks,
              "selection_order_checks": selection_checks, "transform_semantic_checks": len(expected_rows),
              "unique_formula_checks": int(len(tokens)), "orbit_id_checks": int(len(tokens)), "historical_overlap": 0,
              "candidate_stream_checks": int(candidate_clauses.size),
              "candidate_truth_count_checks": int(candidate_counts.size),
              "near_far_cells_checked": int(len(UNIQUE_ASSIGNMENTS) * 2),
              "candidate_bank_sha256": sha256(HERE / "candidate_bank.npz"),
              "selected_bank_sha256": sha256(HERE / "selected_bank.npz"),
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              **frozen,
              "scope": "independent token/Boolean implementation and complete selection-order replay; no model import"}
    with (HERE / "bank_verification.json").open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
