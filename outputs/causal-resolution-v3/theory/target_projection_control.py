#!/usr/bin/env python3
"""Exact weight-only output-projection coding control; no test outcomes loaded."""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APPLICATION = HERE.parent / "application"
DEN = 65536 * 10 * 4096
EPSILON = F(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    native_path = HERE / "results" / "precision_optimality.json"
    native_receipt = json.loads(native_path.read_text())
    native = {r["seed"]:r["state_bits"] for r in native_receipt["allocations"]}
    rows=[]
    for seed in (17,29,43):
        path=APPLICATION/f"seed-{seed}/precision-gain-1.npz"
        with np.load(path,allow_pickle=False) as data:
            M=list(map(int,data["M"]))
            C=[[int(v) for v in row] for row in data["C"].tolist()]
        assert len(M)==32 and len(C)==32 and all(len(row)==10 for row in C)
        assert all(m>=0 for m in M)
        assert all(sum(row)==0 for row in C)
        low=[sum(M[j]*min(C[j][k],0) for j in range(32)) for k in range(10)]
        high=[sum(M[j]*max(C[j][k],0) for j in range(32)) for k in range(10)]
        width=[high[k]-low[k] for k in range(10)]
        assert width==[sum(M[j]*abs(C[j][k]) for j in range(32)) for k in range(10)]
        bits=[]
        for numerator in width:
            b=0
            while F(numerator,DEN*(1<<b))>EPSILON:
                b+=1
            assert F(numerator,DEN*(1<<b))<=EPSILON
            assert b==0 or F(numerator,DEN*(1<<(b-1)))>EPSILON
            bits.append(b)
        bounds=[F(width[k],DEN*(1<<bits[k])) for k in range(10)]
        decoder_scales=bounds.copy()
        # Since each output quantiser has its own inequality, every component
        # is already at its exact smallest admissible bit depth for this bound.
        rows.append({
            "seed":seed,"source_file":str(path.relative_to(APPLICATION)),"source_sha256":sha(path),
            "projection_denominator":DEN,"lower_numerators":low,"upper_numerators":high,
            "width_numerators":width,"widths_exact":[str(F(v,DEN)) for v in width],
            "widths_float":[float(F(v,DEN)) for v in width],
            "bits_by_output":bits,"state_bits":sum(bits),"paired_bits":2*sum(bits),
            "paired_error_bounds_exact":[str(v) for v in bounds],
            "maximum_paired_error_bound_exact":str(max(bounds)),
            "maximum_paired_error_bound_float":float(max(bounds)),
            "decoder_scales_exact":[str(v) for v in decoder_scales],
            "native_precision_bound_optimum_bits":native[seed],
            "projection_encoder_matrix_coefficient_slots":32*10,
            "projection_encoder_nonzero_matrix_coefficients":sum(v!=0 for row in C for v in row),
            "projection_encoder_quantiser_lower_limits":10,
            "projection_encoder_quantiser_widths":10,
            "projection_paired_decoder_nonzero_diagonal_coefficients":sum(v!=0 for v in decoder_scales),
            "native_encoder_cross_unit_matrix_coefficients":0,
            "native_encoder_interval_upper_limits":32,
            "native_paired_decoder_dense_coefficient_slots":32*10,
            "exact_width_and_minimum_bit_checks":"PASS",
        })
    receipt={
        "status":"PASS","analysis":"retrospective weight-only target-projection calibration",
        "test_outcomes_read":False,"new_test_evaluation":False,
        "epsilon_exact":str(EPSILON),"target":"ten centred-logit change coordinates",
        "report":"separate uniform midpoint quantisation of the ten known centred output projections",
        "decoder":"for output k, scale_k times (code_prime_k minus code_k)",
        "scope":"exact per-output minimum bit depths satisfying the output-projection box-width bound; not semantic explanation, a universal minimum code, or a change to the neuron-local optimum",
        "cost_convention":"matrix coefficients and scalar quantiser constants are reported separately; shared/derived constants must not be silently counted twice",
        "native_optimality_receipt_sha256":sha(native_path),
        "script_sha256":sha(Path(__file__)),"models":rows,
    }
    out=HERE/"target_projection_control.json"
    out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"status":"PASS","test_outcomes_read":False,
                      "models":[{"seed":r["seed"],"state_bits":r["state_bits"],
                                 "paired_bits":r["paired_bits"],
                                 "maximum_bound":r["maximum_paired_error_bound_float"]} for r in rows]},indent=2))


if __name__=="__main__":
    main()
