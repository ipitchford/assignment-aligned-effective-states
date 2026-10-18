#!/usr/bin/env python3
"""Replay prospective certificates without LP, SymPy, or sampled outcomes.

Uses integer Bareiss determinants rather than the construction's modular
elimination. Fractions evaluate both input vectors and the complete head.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
APP = ROOT.parent / "application"
DEN = 65536 * 10 * 4096
EXPECTED_SEEDS = [17, 29, 43]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def determinant_bareiss(matrix):
    a = [list(map(int, row)) for row in matrix]
    if not a:
        return 1
    sign, previous = 1, 1
    for k in range(len(a) - 1):
        pivot_row = next((i for i in range(k, len(a)) if a[i][k]), None)
        if pivot_row is None:
            return 0
        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, len(a)):
            for j in range(k + 1, len(a)):
                numerator = a[i][j] * pivot - a[i][k] * a[k][j]
                assert numerator % previous == 0
                a[i][j] = numerator // previous
            a[i][k] = 0
        previous = pivot
    return sign * a[-1][-1]


def check_minor(matrix, certificate, expected):
    assert certificate["rank_lower"] == expected
    rows, columns = certificate["minor_rows"], certificate["minor_columns"]
    assert len(rows) == len(set(rows)) == expected
    assert len(columns) == len(set(columns)) == expected
    minor = [[int(matrix[r][c]) for c in columns] for r in rows]
    determinant = determinant_bareiss(minor)
    assert determinant != 0
    assert determinant % certificate["prime"] == certificate["minor_determinant_mod_prime"]
    return {"dimension": expected, "integer_minor_determinant": str(determinant)}


def matvec(matrix, vector):
    assert all(len(row) == len(vector) for row in matrix)
    return [sum((F(int(a)) * b for a, b in zip(row, vector)), F()) for row in matrix]


def validate_certificate_inventory(receipt):
    """Reject missing or duplicated records before any paired iteration."""
    for key in ("rank_certificates", "exact_realised_aliases"):
        records = receipt[key]
        assert len(records) == len(EXPECTED_SEEDS)
        assert [record["seed"] for record in records] == EXPECTED_SEEDS
    for rank in receipt["rank_certificates"]:
        assert len(rank["group_rank_witnesses"]) == 4
        assert len(rank["current_output_plus_first_m_actions_rank_witnesses"]) == 5
        assert len(rank["coarsened_two_group_rank_witnesses"]) == 2
    for alias in receipt["exact_realised_aliases"]:
        for key in ("input_plus_exact", "input_minus_exact"):
            assert len(alias[key]) == 64
        for key in ("hidden_plus_integer_scale_exact", "hidden_minus_integer_scale_exact"):
            assert len(alias[key]) == 32
        for key in ("common_current_centred_projection_exact", "centred_output_bias_exact",
                    "common_full_current_centred_logits_exact"):
            assert len(alias[key]) == 10
        assert 0 <= alias["group_index"] < 4
        assert 0 <= alias["output_coordinate"] < 10


def missing_record_controls(receipt):
    rejected = 0
    for key in ("rank_certificates", "exact_realised_aliases"):
        mutant = json.loads(json.dumps(receipt))
        mutant[key].pop()
        try:
            validate_certificate_inventory(mutant)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError(f"Accepted missing record in {key}")
    assert rejected == 2
    return rejected


def midpoint_code(value, lower, width, bits):
    t = (value - lower) * (1 << bits) / width
    return max(0, min((1 << bits) - 1, t.numerator // t.denominator))


def main():
    source = ROOT / "results.json"
    receipt = json.loads(source.read_text())
    validate_certificate_inventory(receipt)
    rejected_missing_records = missing_record_controls(receipt)
    assert receipt["script_sha256"] == sha(ROOT / "audit.py")
    contract_path = APP / "contract.json"
    assert receipt["contract_sha256"] == sha(contract_path)
    groups = json.loads(contract_path.read_text())["interventions"]["groups"]
    assert groups == receipt["groups"]
    assert len(groups) == 4 and all(len(group) == 8 for group in groups)
    assert sorted(j for g in groups for j in g) == list(range(32))
    projection_path = ROOT.parent / "theory/target_projection_control.json"
    projection_rows = json.loads(projection_path.read_text())["models"]
    assert sorted(r["seed"] for r in projection_rows) == EXPECTED_SEEDS
    projection = {r["seed"]: r for r in projection_rows}
    precision_path = ROOT.parent / "theory/results/precision_optimality_sensitivity.json"
    precision_rows = [r for r in json.loads(precision_path.read_text())["allocations"]
                      if r["epsilon_exact"] == "2"]
    assert sorted(r["seed"] for r in precision_rows) == EXPECTED_SEEDS
    precision = {r["seed"]: r for r in precision_rows}
    reports = []
    for rank, alias in zip(receipt["rank_certificates"], receipt["exact_realised_aliases"]):
        seed = rank["seed"]
        assert alias["seed"] == seed
        model_path = APP / f"seed-{seed}/model.npz"
        design_path = APP / f"seed-{seed}/precision-gain-1.npz"
        assert sha(model_path) == rank["model_sha256"]
        assert sha(design_path) == rank["precision_source_sha256"]
        with np.load(model_path, allow_pickle=False) as data:
            W, bias, head = [np.asarray(data[k], dtype=np.int64) for k in ("w1", "b1", "w2")]
            output_bias = np.asarray(data["b2"], dtype=np.int64)
        assert W.shape == (64, 32) and bias.shape == (32,)
        assert head.shape == (32, 10) and output_bias.shape == (10,)
        with np.load(design_path, allow_pickle=False) as data:
            C, M = [np.asarray(data[k], dtype=np.int64) for k in ("C", "M")]
        assert C.shape == (32, 10) and M.shape == (32,)
        assert np.array_equal(C, 10 * head - head.sum(axis=1, keepdims=True))
        assert np.array_equal(M, np.maximum(0, 16 * (np.maximum(W, 0).sum(axis=0) + bias)))
        assert np.all(C.sum(axis=1) == 0) and np.all(M > 0)
        minor_checks = [check_minor(C.T, rank["current_output_rank_witness"], 9)]
        bank = []
        for group, block in zip(groups, rank["group_rank_witnesses"]):
            minor_checks.append(check_minor(C[group].T, block, 8))
            z = np.zeros_like(C)
            z[group] = C[group]
            bank.extend(z.T.tolist())
        minor_checks.append(check_minor(bank, rank["prospective_rank_witness"], 32))
        assert rank["current_output_plus_first_m_actions_ranks"] == [9, 17, 25, 32, 32]
        for count, certificate in enumerate(rank["current_output_plus_first_m_actions_rank_witnesses"]):
            expected = min(32, 9 + 8 * count)
            minor_checks.append(check_minor(C.T.tolist() + bank[:10 * count], certificate, expected))
        for group, certificate in zip([groups[0] + groups[1], groups[2] + groups[3]],
                                      rank["coarsened_two_group_rank_witnesses"]):
            minor_checks.append(check_minor(C[group].T, certificate, 9))
        probes = np.vstack([np.zeros((1, 64), dtype=np.int64),
                            np.full((1, 64), 16, dtype=np.int64),
                            16 * np.eye(64, dtype=np.int64), 16 * (W.T > 0)])
        assert len(probes) == rank["constructed_probe_count"] == 98
        hidden_probes = np.maximum(0, probes @ W + 16 * bias)
        minor_checks.append(check_minor(hidden_probes, rank["constructed_span_rank_witness"], 32))
        assert rank["constructed_hidden_affine_span_rank"] == 32
        minor_checks.append(check_minor(hidden_probes[1:] - hidden_probes[0],
                                        rank["constructed_affine_span_rank_witness"], 32))

        x, y = [[F(v) for v in alias[key]] for key in ("input_plus_exact", "input_minus_exact")]
        assert all(0 <= v <= 16 for v in x + y)
        def hidden(v):
            return [max(F(), a + 16 * int(b)) for a, b in zip(matvec(W.T, v), bias)]
        hx, hy = hidden(x), hidden(y)
        assert hx == [F(v) for v in alias["hidden_plus_integer_scale_exact"]]
        assert hy == [F(v) for v in alias["hidden_minus_integer_scale_exact"]]
        assert all(0 <= a <= int(m) for a, m in zip(hx, M))
        assert all(0 <= a <= int(m) for a, m in zip(hy, M))
        px, py = [v / DEN for v in matvec(C.T, hx)], [v / DEN for v in matvec(C.T, hy)]
        assert px == py == [F(v) for v in alias["common_current_centred_projection_exact"]]
        centered_bias = [F(10 * int(v) - int(output_bias.sum()), 10 * 4096) for v in output_bias]
        assert centered_bias == [F(v) for v in alias["centred_output_bias_exact"]]
        full_x = [v + b for v, b in zip(px, centered_bias)]
        full_y = [v + b for v, b in zip(py, centered_bias)]
        assert full_x == full_y == [F(v) for v in alias["common_full_current_centred_logits_exact"]]
        pre = 8 * W.sum(axis=0) + 16 * bias
        assert np.all(pre != 0)
        active = (pre > 0).astype(np.int64)
        assert [j for j in range(32) if active[j]] == alias["active_units_at_centre"]
        E = (C * active[:, None]).T @ W.T
        minor_checks.append(check_minor(E, alias["exact_head_nullspace_rank_witness"], 9))
        g, k = alias["group_index"], alias["output_coordinate"]
        assert alias["group_units"] == groups[g]
        gap = sum((F(int(C[j, k])) * (hx[j] - hy[j]) for j in groups[g]), F()) / DEN
        assert gap == F(alias["full_ablation_effect_gap_exact"])
        assert gap / 2 == F(alias["full_ablation_decoder_error_lower_exact"])
        assert gap / 4 == F(alias["half_ablation_decoder_error_lower_exact"])
        assert gap / 4 > 1
        assert F(alias["full_ablation_strength_exact"]) == 1
        assert F(alias["half_ablation_strength_exact"]) == F(1, 2)
        project = projection[seed]
        assert project["source_sha256"] == sha(design_path)
        lo = [F(v, DEN) for v in project["lower_numerators"]]
        width = [F(v, DEN) for v in project["width_numerators"]]
        bits = project["bits_by_output"]
        assert len(lo) == len(width) == len(bits) == 10
        qx = [midpoint_code(v, l, w, b) for v, l, w, b in zip(px, lo, width, bits)]
        qy = [midpoint_code(v, l, w, b) for v, l, w, b in zip(py, lo, width, bits)]
        assert qx == qy and sum(bits) == 77

        prospective = precision[seed]
        precision_bits = prospective["bits_by_unit"]
        assert len(precision_bits) == 32
        prospective_bounds = [sum((F(int(M[j]) * abs(int(C[j, output])), DEN * (1 << (precision_bits[j] + 1)))
                                   for j in range(32)), F()) for output in range(10)]
        assert max(prospective_bounds) <= 1
        assert sum(precision_bits) == prospective["state_bits"]
        reports.append({"seed": seed, "status": "PASS", "minor_checks": minor_checks,
                        "identical_77_bit_report": qx,
                        "input_alias_full_ablation_gap_exact": str(gap),
                        "prospective_all_diagonal_contraction_bits": sum(precision_bits),
                        "prospective_all_diagonal_contraction_bound_exact": str(max(prospective_bounds))})
    output = {"status": "PASS", "test_outcomes_read": False,
              "expected_seeds": EXPECTED_SEEDS, "missing_record_controls_rejected": rejected_missing_records,
              "method": "integer Bareiss minors and Fraction evaluation, no LP or symbolic algebra",
              "results_sha256": sha(source), "verifier_sha256": sha(Path(__file__)),
              "target_projection_receipt_sha256": sha(projection_path),
              "precision_sensitivity_receipt_sha256": sha(precision_path), "models": reports}
    (ROOT / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "models": [{k: r[k] for k in
          ("seed", "status", "prospective_all_diagonal_contraction_bits")} for r in reports]}))


if __name__ == "__main__":
    main()
