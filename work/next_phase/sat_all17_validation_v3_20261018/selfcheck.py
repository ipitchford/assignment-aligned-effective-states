"""Code-only checks. No checkpoint, historical arrays or candidate sampling."""
from __future__ import annotations

import ast
import json
import numpy as np

from protocol import *
from pilot_common import CLAUSES, semantic_counts
from construct import clause_permutation


def main() -> None:
    checks = 0
    for assignment in range(32):
        for pair in TRANSPOSITIONS:
            assert swap_assignment(swap_assignment(assignment, pair), pair) == assignment
            checks += 1
    examples = [
        (np.array([0] + [1] * 31), 0, 0),
        (np.array([0, 0] + [1] * 30), 0, 1),
        (np.array([0] * 9 + [1] * 23), 0, 1),
        (np.ones(32), 0, 2),
        (np.array([1, 0] + [1] * 30), 0, 3),
        (np.array([1, 0, 0] + [1] * 29), 0, 4),
        (np.array([1] + [0] * 9 + [1] * 22), 0, 4),
    ]
    for counts, assignment, expected in examples:
        assert classify_stratum(counts, assignment) == expected
        checks += 1
    assert exact_auc(np.array([1, 2]), np.array([0, 1])) == 0.875
    checks += 1
    for name, assignment in zip(ASSIGNMENT_NAMES, ASSIGNMENT_IDS):
        assert int("".join("1" if value == "T" else "0" for value in name), 2) == int(assignment)
        checks += 1
    assert min_solution_hamming(np.array([1, 0] + [1] * 30), 0) == 1
    assert min_solution_hamming(np.array([1] * 3 + [0] + [1] * 28), 0) == 2
    assert min_solution_hamming(np.ones(32), 0) is None
    checks += 3
    rng = np.random.default_rng(1)
    formulas = rng.integers(0, len(CLAUSES), size=(64, 10), dtype=np.int16)
    original_counts = semantic_counts(formulas)
    for pair in TRANSPOSITIONS:
        cmap = clause_permutation(pair)
        transformed_counts = semantic_counts(cmap[formulas])
        for assignment in range(32):
            target = swap_assignment(assignment, pair)
            assert np.array_equal(original_counts[:, assignment], transformed_counts[:, target])
            checks += len(formulas)
    for filename in ("protocol.py", "construct.py", "verify_bank.py", "native.py", "analyze.py"):
        ast.parse((HERE / filename).read_text(), filename=filename)
        checks += 1
    result = {"status": "PASS", "checks": checks, "scope": "pure semantics, transposition algebra and Python syntax; zero data/model access"}
    with (HERE / "selfcheck.json").open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
