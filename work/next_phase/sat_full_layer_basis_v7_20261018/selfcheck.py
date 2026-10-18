"""Synthetic and protected-cache checks: real bank/prior arrays, zero model access."""
from __future__ import annotations

import ast
import itertools
import json
import platform
import tempfile
from pathlib import Path

import numpy as np

from protocol import (HERE, BANK, PRIOR_NATIVE, UNITS, COVERED_ASSIGNMENTS,
                      UNCOVERED_ASSIGNMENTS, compare_prior_cache,
                      exclusive_scope_from_labels, semantic_counts, sha256, strata,
                      validate_analysis_prerequisites, validate_hook_dependencies,
                      validate_replay_prerequisites)


CLAUSES = np.array([(i + 5*a, j + 5*b) for i in range(5) for j in range(i+1, 5)
                    for a in range(2) for b in range(2)], dtype=np.int16)


def brute_auc(score, label):
    positive = np.asarray(score)[np.asarray(label, dtype=bool)]
    negative = np.asarray(score)[~np.asarray(label, dtype=bool)]
    delta = positive[:, None] - negative[None, :]
    return (np.count_nonzero(delta > 0) + 0.5*np.count_nonzero(delta == 0)) / delta.size


def brute_counts(formula):
    out = []
    for assignment in range(32):
        bits = [(assignment >> (4-i)) & 1 for i in range(5)]
        violated = 0
        for clause_id in map(int, formula):
            values = []
            for literal in map(int, CLAUSES[clause_id]):
                value = bool(bits[literal % 5])
                values.append(not value if literal >= 5 else value)
            violated += int(not any(values))
        out.append(violated)
    return np.asarray(out, dtype=np.int16)


def brute_matching(weights):
    best = None
    for columns in itertools.permutations(range(weights.shape[1]), weights.shape[0]):
        value = weights[np.arange(weights.shape[0]), columns].sum()
        if best is None or value > best: best = value
    return best


def synthetic_execute_check():
    import analyze
    with tempfile.TemporaryDirectory(prefix="sat-v7-selfcheck-") as name:
        temporary = Path(name)
        with np.load(PRIOR_NATIVE, allow_pickle=False) as prior:
            prior_arrays = {key: prior[key].copy() for key in ("pre", "post", "logits", "unique_ids")}
        rows = len(prior_arrays["logits"])
        pre = np.zeros((rows, 512), dtype=np.float32)
        post = np.zeros((rows, 512), dtype=np.float32)
        pre[:, UNITS] = prior_arrays["pre"]
        post[:, UNITS] = prior_arrays["post"]
        arrays = {"pre": pre, "post": post,
                  "resid_mid": np.zeros((rows, 128), dtype=np.float32),
                  "resid_post": np.zeros((rows, 128), dtype=np.float32),
                  "logits": prior_arrays["logits"],
                  "unit_logit_coefficients": np.zeros((512, 16), dtype=np.float64),
                  "W_out": np.zeros((128, 512), dtype=np.float32),
                  "b_out": np.zeros(128, dtype=np.float32),
                  "W_U": np.zeros((128, 16), dtype=np.float32),
                  "unique_ids": prior_arrays["unique_ids"]}
        np.savez_compressed(temporary / "full_native_outputs.npz", **arrays)
        np.savez_compressed(temporary / "replay_outputs.npz", sentinel=np.array([1], dtype=np.int8))
        freeze_hash = "f" * 64
        native_hash = sha256(temporary / "full_native_outputs.npz")
        (temporary / "native_receipt.json").write_text(json.dumps(
            {"status": "PASS", "stage": "initial", "freeze_sha256": freeze_hash,
             "output_sha256": native_hash}) + "\n")
        (temporary / "replay_receipt.json").write_text(json.dumps(
            {"status": "PASS", "stage": "replay", "freeze_sha256": freeze_hash,
             "output_sha256": sha256(temporary / "replay_outputs.npz"),
             "initial_output_sha256": native_hash}) + "\n")
        saved = (analyze.HERE, analyze.ROTATIONS, analyze.RESAMPLES,
                 analyze.verify_live_preflight, analyze.np.linalg.matrix_rank)
        try:
            analyze.HERE = temporary
            analyze.ROTATIONS = 2
            analyze.RESAMPLES = 1
            analyze.verify_live_preflight = lambda: {"status": "PASS", "synthetic": True}
            analyze.np.linalg.matrix_rank = lambda value: 17
            analyze.execute({"freeze_sha256": freeze_hash}, {"synthetic": True})
        finally:
            (analyze.HERE, analyze.ROTATIONS, analyze.RESAMPLES,
             analyze.verify_live_preflight, analyze.np.linalg.matrix_rank) = saved
        result = json.loads((temporary / "analysis.json").read_text())
        assert result["status"] == "PASS" and len(result["rotations"]) == 2
        assert result["primary_inference"]["conclusion_key"] in {
            "both_primaries_and_exclusive", "both_primaries_only", "one_primary", "neither_primary"}
    return 3


def main():
    from analyze import (auc_matrix, basis_statistics, derangement,
                         exclusive_uncovered_auc, maximum_matching, monte_carlo,
                         primary_decision)
    checks = 0; rng = np.random.default_rng(7)
    for _ in range(32):
        score = rng.integers(-2, 3, size=(25, 4)).astype(float)
        label = np.column_stack([rng.permutation(np.r_[np.ones(9, dtype=bool), np.zeros(16, dtype=bool)]) for _ in range(3)])
        observed = auc_matrix(score, label)
        for i in range(3):
            for j in range(4):
                expected = brute_auc(score[:, j], label[:, i])
                assert observed[i, j] == expected
                assert np.isclose(auc_matrix(-score[:, [j]], label[:, [i]])[0, 0], 1 - expected,
                                  rtol=0, atol=1e-15)
                checks += 2
    for _ in range(25):
        weights = rng.integers(0, 5, size=(3, 6)).astype(float)
        assignment, selected, certificate = maximum_matching(weights, True)
        assert len(set(map(int, assignment))) == 3
        assert selected.sum() == brute_matching(weights)
        assert certificate["absolute_duality_gap"] <= 1e-9
        assert certificate["minimum_reduced_cost"] >= -1e-10
        checks += 4
    for shape in ((5, 8), (6, 9)):
        weights = rng.integers(-2, 8, size=shape).astype(float)
        assignment, selected, certificate = maximum_matching(weights, True)
        assert len(set(map(int, assignment))) == shape[0]
        assert selected.sum() == brute_matching(weights)
        assert certificate["absolute_duality_gap"] <= 1e-9
        checks += 3
    for n in (2, 3, 5, 17):
        for _ in range(32):
            proposal = derangement(n, rng)
            assert sorted(proposal.tolist()) == list(range(n))
            assert np.all(proposal != np.arange(n))
            checks += 2
    try:
        derangement(1, rng)
        raise AssertionError("singleton derangement did not fail")
    except RuntimeError:
        checks += 1
    mc = monte_carlo(2.0, [1.0, 2.0, 3.0])
    assert mc["rotation_exceedance_count"] == 2 and mc["directional_monte_carlo_p"] == 0.75
    checks += 2
    formulas = rng.integers(0, 40, size=(64, 10), dtype=np.int16)
    vectorised = semantic_counts(formulas)
    for row, observed in zip(formulas, vectorised):
        assert np.array_equal(observed, brute_counts(row)); checks += 32
    synthetic = np.array([[0] + [1]*31, [0, 0] + [1]*30, [1]*32,
                          [1, 0] + [1]*30, [1, 0, 0] + [1]*29], dtype=np.int16)
    assert np.array_equal(strata(synthetic, 0), np.arange(5, dtype=np.int8)); checks += 5
    decision_cases = [({"matching_uncovered16": 0.01, "matching_32": 0.02}, True, "both_primaries_and_exclusive"),
                      ({"matching_uncovered16": 0.01, "matching_32": 0.02}, False, "both_primaries_only"),
                      ({"matching_uncovered16": 0.01, "matching_32": 0.20}, True, "one_primary"),
                      ({"matching_uncovered16": 0.20, "matching_32": 0.30}, True, "neither_primary")]
    for pvalues, exclusive, expected_key in decision_cases:
        passed, key, text = primary_decision(pvalues, exclusive)
        assert key == expected_key and "generic coordinate privilege" in text and len(passed) == 2
        checks += 3
    with np.load(PRIOR_NATIVE, allow_pickle=False) as prior:
        prior_arrays = {key: prior[key].copy() for key in ("pre", "post", "logits")}
    full_arrays = {"pre": np.zeros((len(prior_arrays["pre"]), 512), dtype=np.float32),
                   "post": np.zeros((len(prior_arrays["post"]), 512), dtype=np.float32),
                   "logits": prior_arrays["logits"].copy()}
    full_arrays["pre"][:, UNITS] = prior_arrays["pre"]
    full_arrays["post"][:, UNITS] = prior_arrays["post"]
    comparison = compare_prior_cache(full_arrays, prior_arrays)
    assert comparison["bitwise_equal"] and comparison["same_threshold_events"]
    checks += 2
    for mutation in ("magnitude", "threshold", "argmax"):
        changed = {key: value.copy() for key, value in full_arrays.items()}
        if mutation == "magnitude":
            changed["pre"][0, int(UNITS[0])] += 1e-2
        elif mutation == "threshold":
            changed["post"][0, int(UNITS[0])] = 1.0 if prior_arrays["post"][0, 0] < 0.5 else 0.0
        else:
            winner = int(np.argmax(changed["logits"][0]))
            changed["logits"][0, (winner + 1) % changed["logits"].shape[1]] = changed["logits"][0, winner] + 10.0
        try:
            compare_prior_cache(changed, prior_arrays)
            raise AssertionError(f"prior comparator accepted {mutation} mutation")
        except RuntimeError:
            checks += 1
    scope_labels = np.zeros((4, 32), dtype=bool)
    scope_labels[1, int(COVERED_ASSIGNMENTS[0])] = True
    scope_labels[2, int(UNCOVERED_ASSIGNMENTS[0])] = True
    scope_labels[3, int(UNCOVERED_ASSIGNMENTS[1])] = True
    scope_labels[3, int(UNCOVERED_ASSIGNMENTS[2])] = True
    assert np.array_equal(exclusive_scope_from_labels(scope_labels), [False, False, True, True])
    checks += 4
    exclusive_labels = np.array([[1, 0], [0, 1], [0, 0], [1, 0]], dtype=bool)
    exclusive_scores = np.arange(12, dtype=float).reshape(4, 3)
    saved_uncovered = __import__("analyze").UNCOVERED_ASSIGNMENTS
    try:
        __import__("analyze").UNCOVERED_ASSIGNMENTS = np.array([0, 1], dtype=np.int8)
        observed = exclusive_uncovered_auc(exclusive_scores, exclusive_labels, np.ones(4, dtype=bool))
        assert observed.shape == (2, 3) and np.isfinite(observed).all()
    finally:
        __import__("analyze").UNCOVERED_ASSIGNMENTS = saved_uncovered
    checks += 2
    with tempfile.TemporaryDirectory(prefix="sat-v6-replay-") as name:
        root = Path(name)
        try:
            validate_replay_prerequisites(root, "f" * 64)
            raise AssertionError("missing replay prerequisites accepted")
        except RuntimeError:
            assert not (root / "attempt_replay.json").exists(); checks += 2
    with tempfile.TemporaryDirectory(prefix="sat-v7-analysis-") as name:
        root = Path(name); freeze_hash = "f" * 64
        try:
            validate_analysis_prerequisites(root, freeze_hash)
            raise AssertionError("missing analysis prerequisites accepted")
        except RuntimeError:
            checks += 1
        (root / "forward_started_initial.json").write_text("{}\n")
        (root / "forward_started_replay.json").write_text("{}\n")
        (root / "full_native_outputs.npz").write_bytes(b"initial")
        (root / "replay_outputs.npz").write_bytes(b"replay")
        initial_hash = sha256(root / "full_native_outputs.npz")
        replay_hash = sha256(root / "replay_outputs.npz")
        (root / "native_receipt.json").write_text(json.dumps(
            {"status": "PASS", "stage": "initial", "freeze_sha256": freeze_hash,
             "output_sha256": initial_hash}) + "\n")
        (root / "replay_receipt.json").write_text(json.dumps(
            {"status": "PASS", "stage": "replay", "freeze_sha256": freeze_hash,
             "output_sha256": replay_hash, "initial_output_sha256": initial_hash}) + "\n")
        assert set(validate_analysis_prerequisites(root, freeze_hash)) == {"initial", "replay"}
        checks += 2
        (root / "failure_replay.json").write_text("{}\n")
        try:
            validate_analysis_prerequisites(root, freeze_hash)
            raise AssertionError("forbidden analysis artifact accepted")
        except RuntimeError:
            checks += 1
        (root / "failure_replay.json").unlink()
        (root / "replay_outputs.npz").write_bytes(b"changed")
        try:
            validate_analysis_prerequisites(root, freeze_hash)
            raise AssertionError("analysis hash mismatch accepted")
        except RuntimeError:
            checks += 1
    class HookPoint:
        def add_hook(self, hook): return None
    class MLP:
        hook_pre = HookPoint(); hook_post = HookPoint()
    class Final:
        mlp = MLP(); hook_resid_mid = HookPoint(); hook_resid_post = HookPoint()
    class Model:
        def remove_all_hooks(self): return None
    assert set(validate_hook_dependencies(Model(), Final(), {"serializable": True})) == {
        "pre", "post", "resid_mid", "resid_post"}
    checks += 2
    class BrokenFinal:
        mlp = MLP(); hook_resid_post = HookPoint()
    try:
        validate_hook_dependencies(Model(), BrokenFinal(), {})
        raise AssertionError("missing hook dependency accepted")
    except AttributeError:
        checks += 1
    checks += synthetic_execute_check()
    for filename in ("protocol.py", "preflight.py", "native.py", "analyze.py", "selfcheck.py"):
        ast.parse((HERE / filename).read_text(), filename=filename); checks += 1
    tested = {name: sha256(HERE / name) for name in ("protocol.py", "native.py", "analyze.py", "selfcheck.py")}
    result = {"status": "PASS", "checks": checks, "python": platform.python_version(),
              "numpy": np.__version__, "tested_file_sha256": tested,
              "scope": "tie-heavy exact AUROC, sign symmetry, certified rectangular matching through 6x9, protected-cache comparison, exclusive-scope labels, Holm conclusion branches, synthetic end-to-end analysis, Boolean semantics, strata and syntax; zero model access"}
    with (HERE / "selfcheck.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
