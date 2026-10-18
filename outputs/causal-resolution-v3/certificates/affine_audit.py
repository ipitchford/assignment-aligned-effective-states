"""Finite affine-response audits with independently checkable rational witnesses.

LP duality and sparse feature selection are established methods. This module
applies them to a declared report/decoder class; it does not infer semantics.
The numerical LP proposes witnesses. Fraction arithmetic validates them.
"""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations
import json
import numpy as np
from scipy.optimize import linprog


def minimax_fit(design, response):
    a = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float)
    if a.ndim != 2 or y.shape != (len(a),) or not len(a):
        raise ValueError("nonempty two-dimensional design and one response per row required")
    if not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    n, d = a.shape
    c = np.r_[np.zeros(d), 1.]
    constraints = np.r_[np.c_[a, -np.ones(n)], np.c_[-a, -np.ones(n)]]
    fit = linprog(c, A_ub=constraints, b_ub=np.r_[y, -y],
                  bounds=[(None, None)]*d + [(0., None)], method="highs")
    if not fit.success:
        raise RuntimeError(fit.message)
    u = fit.ineqlin.marginals[:n] - fit.ineqlin.marginals[n:]
    return {"error": float(fit.fun), "coef": fit.x[:d], "dual": u,
            "max_residual": float(np.max(np.abs(y-a@fit.x[:d])))}


def nullspace_fraction(rows):
    """RREF nullspace over Q, with no reliance on numerical rank."""
    a = [[Fraction(x) for x in row] for row in rows]
    m, n = len(a), len(a[0]) if a else 0
    pivots = []
    r = 0
    for c in range(n):
        pivot = next((i for i in range(r, m) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        factor = a[r][c]
        a[r] = [v/factor for v in a[r]]
        for i in range(m):
            if i != r and a[i][c]:
                factor = a[i][c]
                a[i] = [v-factor*w for v,w in zip(a[i], a[r])]
        pivots.append(c)
        r += 1
        if r == m:
            break
    basis = []
    for c in range(n):
        if c in pivots:
            continue
        v = [Fraction(0)]*n
        v[c] = 1
        for i,p in enumerate(pivots):
            v[p] = -a[i][c]
        basis.append(v)
    return basis


def rational_witness(design, response_numerators, denominator, dual):
    """Extract and verify a small exact lower-bound certificate from LP support.

    Design entries and response numerators must be integers. If the proposed
    support has several dependencies, each rational basis vector is considered;
    this still gives a valid lower bound, without claiming optimality.
    """
    a = np.asarray(design)
    nums = np.asarray(response_numerators)
    if a.ndim != 2 or nums.shape != (len(a),) or int(denominator) != denominator or denominator <= 0:
        raise ValueError("invalid exact response representation")
    if not np.isfinite(a).all() or not np.equal(a, np.rint(a)).all():
        raise ValueError("exact checker requires an integer design")
    if not np.equal(nums, np.rint(nums)).all():
        raise ValueError("response numerators must be integers")
    ids = np.flatnonzero(np.abs(dual) > 1e-8).tolist()
    if not ids:
        return None
    rows = [[int(a[i,j]) for i in ids] for j in range(a.shape[1])]
    basis = nullspace_fraction(rows)
    if not basis:
        raise ValueError("numerical support has no exact annihilating contrast")
    best = None
    for v in basis:
        norm = sum(abs(t) for t in v)
        v = [t/norm for t in v]
        value = sum(t*Fraction(int(nums[i]), int(denominator)) for t,i in zip(v,ids))
        if value < 0:
            value = -value
            v = [-t for t in v]
        if best is None or value > best[0]:
            best = value, v
    value, v = best
    keep = [(i,t) for i,t in zip(ids,v) if t]
    cert = {"rows": [i for i,t in keep], "weights": [str(t) for i,t in keep],
            "lower_bound": str(value), "lower_bound_float": float(value),
            "response_denominator": int(denominator), "support_size": len(keep)}
    verify_witness(a, nums, denominator, cert)
    return cert


def verify_witness(design, nums, denominator, cert):
    """Trust boundary: exact rational dot products only, no optimiser needed."""
    ids, weights = cert["rows"], [Fraction(x) for x in cert["weights"]]
    if len(ids) != len(weights) or len(ids) != len(set(ids)):
        raise ValueError("invalid witness rows")
    if sum(abs(w) for w in weights) != 1:
        raise ValueError("witness is not l1-normalised")
    for j in range(len(design[0])):
        if sum(w*Fraction(int(design[i][j])) for i,w in zip(ids,weights)):
            raise ValueError("witness does not annihilate decoder design")
    value = sum(w*Fraction(int(nums[i]),int(denominator)) for i,w in zip(ids,weights))
    if value != Fraction(cert["lower_bound"]) or value < 0:
        raise ValueError("witness bound does not match exact responses")
    return value


def exhaustive_repair(base, candidates, response, epsilon, costs=None):
    """Small-bank exact selection by exhaustive subsets and numerical LP fits.

    The selected subset's optimality is numerical unless the caller supplies
    separate rational certificates excluding every cheaper subset.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    p = candidates.shape[1]
    costs = np.ones(p) if costs is None else np.asarray(costs)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("costs must be finite and nonnegative")
    options = [s for k in range(p+1) for s in combinations(range(p), k)]
    options.sort(key=lambda s: (float(sum(costs[list(s)])), len(s), s))
    for s in options:
        design = np.c_[base, candidates[:,s]]
        fit = minimax_fit(design, response)
        if fit["error"] <= epsilon+1e-9:
            return {"selected": list(s), "cost": float(sum(costs[list(s)])), "fit": fit}
    return None


def known_answer_controls():
    a = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]], dtype=int)
    y = np.array([0,1,1,0], dtype=int)
    fit = minimax_fit(a,y)
    cert = rational_witness(a,y,1,fit["dual"])
    if verify_witness(a,y,1,cert) != Fraction(1,2):
        raise AssertionError("XOR minimax lower bound must be 1/2")
    if np.max(np.abs(y-a@np.array([.5,0,0]))) != .5:
        raise AssertionError("matching upper witness failed")
    phi = (a[:,1]*a[:,2])[:,None]
    repair = minimax_fit(np.c_[a,phi],y)
    if repair["error"] > 1e-9:
        raise AssertionError("XOR interaction must repair exactly")
    # Every report is distinct, so a lookup decoder has zero empirical error.
    if len(set(map(tuple,a))) != len(a):
        raise AssertionError("control reports must be injective")
    # Negative controls exercise the scientific trust boundary.
    rejected = 0
    for kind in ("weights", "response", "bound"):
        bad = json.loads(json.dumps(cert))
        yn = y.copy()
        if kind == "weights": bad["weights"][0] = str(Fraction(bad["weights"][0])+Fraction(1,8))
        elif kind == "response": yn[cert["rows"][0]] += 1
        else: bad["lower_bound"] = "1/3"
        try:
            verify_witness(a,yn,1,bad)
        except ValueError:
            rejected += 1
    if rejected != 3:
        raise AssertionError("mutated certificate accepted")
    # Refining a report by a nonlinear product adds no information, but enlarges
    # the affine decoder class. Equal-information reports can have different cost.
    return {"xor": cert, "affine_error": fit["error"],
            "repair_error": repair["error"], "mutation_controls_rejected": rejected,
            "scope": "known-answer control; not trained-network evidence"}


if __name__ == "__main__":
    print(json.dumps(known_answer_controls(), indent=2))
