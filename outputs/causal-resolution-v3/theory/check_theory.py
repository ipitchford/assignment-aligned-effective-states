"""Small independent numerical and rational controls for report.md.

These checks calibrate implementations and examples, not mathematical novelty
or population prevalence. All local calculations are small and single-process.
"""
from __future__ import annotations

import itertools
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
import sympy as sp


def primal(A, y, bounds=None):
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = A.shape
    result = linprog(
        np.r_[np.zeros(d), 1.0],
        A_ub=np.vstack([np.c_[A, -np.ones(n)], np.c_[-A, -np.ones(n)]]),
        b_ub=np.r_[y, -y],
        bounds=(bounds or [(None, None)] * d) + [(0, None)],
        method="highs",
    )
    assert result.success, result.message
    return result


def independent_dual(A, y):
    n, d = A.shape
    result = linprog(
        np.r_[-y, y],
        A_ub=np.ones((1, 2*n)), b_ub=[1.0],
        A_eq=np.c_[A.T, -A.T] if d else None,
        b_eq=np.zeros(d) if d else None,
        bounds=(0, None), method="highs",
    )
    assert result.success, result.message
    return -result.fun


def exact_witness(A, y, result):
    n = len(y)
    u = result.ineqlin.marginals[:n] - result.ineqlin.marginals[n:]
    support = np.flatnonzero(np.abs(u) > 1e-8)
    assert len(support) <= np.linalg.matrix_rank(A) + 1
    signs = [1 if u[i] > 0 else -1 for i in support]
    M = sp.Matrix(A[support].astype(int).tolist()).T
    M = M.col_join(sp.Matrix([signs]))
    rhs = sp.zeros(M.rows, 1)
    rhs[-1] = 1
    solution, parameters = M.gauss_jordan_solve(rhs)
    assert not parameters.rows
    exact_u = [F(0) for _ in y]
    for i, value in zip(support, solution):
        exact_u[int(i)] = F(int(value.p), int(value.q))
    assert sum(map(abs, exact_u)) == 1
    for j in range(A.shape[1]):
        assert sum(exact_u[i] * int(A[i, j]) for i in range(n)) == 0
    value = sum(exact_u[i] * F(str(y[i])) for i in range(n))
    assert abs(float(value) - result.fun) < 1e-7
    return exact_u, value


def bounded_dual(A0, phi, y, capacities):
    n, p = phi.shape
    d = A0.shape[1]
    objective = np.r_[-y, y, capacities]
    inequalities = [np.r_[np.ones(2*n), np.zeros(p)]]
    rhs = [1.0]
    for j in range(p):
        row = np.r_[phi[:, j], -phi[:, j], np.zeros(p)]
        row[2*n+j] = -1
        inequalities.append(row)
        row2 = np.r_[-phi[:, j], phi[:, j], np.zeros(p)]
        row2[2*n+j] = -1
        inequalities.append(row2)
        rhs.extend([0.0, 0.0])
    eq = np.c_[A0.T, -A0.T, np.zeros((d, p))] if d else None
    result = linprog(objective, A_ub=inequalities, b_ub=rhs,
                     A_eq=eq, b_eq=np.zeros(d) if d else None,
                     bounds=(0, None), method="highs")
    assert result.success
    return -result.fun


def main():
    counts = {}
    # Exact XOR lower and matching upper certificates, including the bound edge.
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    yq = [F(0), F(1), F(1), F(0)]
    uq = [F(-1,4), F(1,4), F(1,4), F(-1,4)]
    for j in range(3):
        col = [1 if j == 0 else s[j-1] for s in states]
        assert sum(u*v for u,v in zip(uq,col)) == 0
    assert sum(u*v for u,v in zip(uq,yq)) == F(1,2)
    for b in [F(0), F(1,2), F(1), F(3,2), F(2), F(3)]:
        cap = min(b, F(2))
        t = (2-cap)/4
        pred = [t + cap/2*x1 + cap/2*x2 - cap*x1*x2 for x1,x2 in states]
        assert max(abs(yi-gi) for yi,gi in zip(yq,pred)) == t
        assert t == max(F(0), F(1,2)-b/4)
    counts["xor_exact_coefficient_budgets"] = 6

    # Small random rational designs include duplicate columns and rank deficiency.
    rng = np.random.default_rng(20261015)
    exact_certificates = 0
    largest_support = 0
    for trial in range(64):
        n = int(rng.integers(3, 9)); d = int(rng.integers(0, n+2))
        A = rng.integers(-1, 2, size=(n,d))
        if d > 1 and trial % 3 == 0:
            A[:, -1] = A[:, 0]
        y = rng.integers(-8, 9, size=n) / 4
        result = primal(A, y)
        assert abs(result.fun-independent_dual(A,y)) < 1e-8
        if result.fun > 1e-8:
            u, _ = exact_witness(A, y, result)
            exact_certificates += 1
            largest_support = max(largest_support, sum(bool(v) for v in u))
    counts["primal_independent_dual_comparisons"] = 64
    counts["rationally_reconstructed_exact_witnesses"] = exact_certificates
    counts["largest_retained_witness_support"] = largest_support

    # Box-constrained primal and independently formulated support-function dual.
    for trial in range(32):
        n, d, p = 6, 2, 3
        A0 = rng.integers(-1, 2, size=(n,d))
        phi = rng.integers(-1, 2, size=(n,p))
        y = rng.integers(-8, 9, size=n)/4
        capacities = rng.integers(0, 5, size=p)/4
        result = primal(np.c_[A0,phi], y,
                        [(None,None)]*d + [(-b,b) for b in capacities])
        assert abs(result.fun-bounded_dual(A0,phi,y,capacities)) < 1e-8
    counts["bounded_coefficient_primal_dual_comparisons"] = 32

    # Exact two-feature synergy, checked independently by LP.
    A0 = np.ones((3,1)); phi = np.array([[0,0],[1,0],[1,1]])
    y = np.array([0,1,0])
    expected = [0.5,0.5,0.5,0]
    for mask in range(4):
        cols = [j for j in range(2) if mask & (1<<j)]
        assert abs(primal(np.c_[A0,phi[:,cols]], y).fun-expected[mask])<1e-9
    counts["greedy_synergy_subsets"] = 4

    # Every subset on every nonempty graph of three vertices checks the reduction.
    edges = [(0,1),(0,2),(1,2)]
    graph_checks = 0
    for graph_mask in range(1,8):
        es = [e for j,e in enumerate(edges) if graph_mask & (1<<j)]
        phi = np.array([[int(v in e) for v in range(3)] for e in es])
        for feature_mask in range(8):
            cols = [j for j in range(3) if feature_mask & (1<<j)]
            covers = all(any(v in cols for v in e) for e in es)
            result = primal(phi[:,cols], np.ones(len(es)), [(-0.5,0.5)]*len(cols))
            assert (result.fun <= 0.5+1e-9) == covers
            graph_checks += 1
    counts["vertex_cover_reduction_subsets"] = graph_checks

    # Contrast-cover search against exhaustive subset LP enumeration.
    total_iterations = 0
    for trial in range(12):
        n, p = 6, 5
        A0 = np.ones((n,1), dtype=int)
        phi = rng.integers(-1,2,size=(n,p))
        y = rng.integers(-8,9,size=n)/4
        eps = F(1,4)
        costs = rng.integers(1,5,size=p)
        masks = sorted(range(1<<p), key=lambda m:(sum(costs[j] for j in range(p) if m&(1<<j)),m))
        feasible = {}
        for m in masks:
            cols = [j for j in range(p) if m&(1<<j)]
            feasible[m] = primal(np.c_[A0,phi[:,cols]], y).fun <= float(eps)+1e-9
        oracle = next((m for m in masks if feasible[m]), None)
        cuts = []
        found = None
        for _ in range(1<<p):
            m = next((m for m in masks if all(m&cut for cut in cuts)), None)
            if m is None:
                break
            cols = [j for j in range(p) if m&(1<<j)]
            A = np.c_[A0,phi[:,cols]]
            result = primal(A,y)
            if result.fun <= float(eps)+1e-9:
                found = m
                break
            u, value = exact_witness(A,y,result)
            assert value > eps
            cut = sum(1<<j for j in range(p)
                      if sum(u[i]*int(phi[i,j]) for i in range(n)) != 0)
            assert not m&cut
            cuts.append(cut)
            total_iterations += 1
        assert found == oracle
    counts["contrast_cover_vs_exhaustive_instances"] = 12
    counts["exact_violated_contrast_cuts"] = total_iterations

    receipt = {"status":"PASS", "seed":20261015,
               "versions":{"numpy":np.__version__,"scipy":scipy.__version__,"sympy":sp.__version__},
               "checks":counts,
               "scope":"Constructed implementation controls; not novelty, prevalence, or theorem proof."}
    (Path(__file__).parent/"checks.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
