#!/usr/bin/env python3
"""Weight-only precision allocation with exact, replayable arithmetic receipts.

No test outcomes are loaded. Optimisation is supplementary and post-test.
Floating-point LP/MILP proposes candidates; all reported rigorous bounds are
recomputed with Fraction or integer arithmetic. --verify runs no optimiser.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

HERE = Path(__file__).resolve().parent
APPLICATION = HERE.parent / "application"
DEN = 65536 * 10 * 4096
MU_DEN = 10**9
SEEDS = (17, 29, 43)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ceil(x):
    return -((-x.numerator)//x.denominator)


def source(seed, epsilon):
    filename = f"seed-{seed}/precision-gain-{float(epsilon):g}.npz"
    path = APPLICATION / filename
    with np.load(path, allow_pickle=False) as data:
        M = list(map(int, data["M"]))
        C = data["C"].tolist()
        incumbent = list(map(int, data["bits"]))
    numerators = [[M[j]*abs(int(C[j][k])) for k in range(10)] for j in range(32)]
    assert len(M) == 32 and all(m >= 0 for m in M)
    return filename, path, numerators, incumbent


def exact_bounds(numerators, bits):
    return [sum((F(numerators[j][k], DEN*(1 << bits[j])) for j in range(32)), F())
            for k in range(10)]


def lagrange(numerators, epsilon, mu):
    weighted = [sum((mu[k]*F(row[k], DEN) for k in range(10)), F()) for row in numerators]
    minbits, minimums = [], []
    for s in weighted:
        b = 0
        while s > (1 << (b+1)):
            b += 1
        # Exact global minimum on all integers b >= 0, including ties.
        assert s <= (1 << (b+1)) and (b == 0 or s >= (1 << b))
        minbits.append(b)
        minimums.append(F(b)+s/(1 << b))
    bound = sum(minimums, F()) - epsilon*sum(mu, F())
    return {"weighted":weighted, "minbits":minbits, "minimums":minimums, "bound":bound}


def allowed_bits(dual, budget):
    L = dual["bound"]
    choices = []
    for s, m in zip(dual["weighted"], dual["minimums"]):
        cap = math.floor(F(budget)-L+m)
        choices.append([b for b in range(max(-1, cap)+1)
                        if L+F(b)+s/(1 << b)-m <= budget])
    return choices


def matrices(numerators, choices):
    indexes = [(j,b) for j in range(32) for b in choices[j]]
    eq = np.zeros((32, len(indexes)))
    response = np.empty((10,len(indexes)))
    costs = np.empty(len(indexes))
    for i,(j,b) in enumerate(indexes):
        eq[j,i] = 1
        response[:,i] = np.asarray(numerators[j],float)/(DEN*(1 << b))
        costs[i] = b
    return indexes, eq, response, costs


def exact_exclude(numerators, epsilon, dual, budget, node_limit=2_000_000, seconds=25):
    """Prove no feasible integer allocation has cost <= budget, or find one.

    The Lagrange bound restricts each coordinate to a finite safe set. An exact
    suffix knapsack DP gives a lower response bound for each output separately.
    DFS visits the remaining finite product using integer comparisons only.
    """
    options = allowed_bits(dual, budget)
    receipt = {"budget":budget, "allowed_bits":options,
               "method":"exact_integer_DFS_with_Lagrange_and_suffix_knapsack_bounds"}
    if any(not o for o in options):
        return receipt | {"status":"infeasible", "nodes":0, "reason":"empty_exact_allowed_set"}
    maximum_b = max(max(o) for o in options)
    factor = DEN*(1 << maximum_b)*epsilon.denominator
    threshold = epsilon.numerator*DEN*(1 << maximum_b)
    terms = {(j,b):[int(a)*epsilon.denominator*(1 << (maximum_b-b)) for a in numerators[j]]
             for j in range(32) for b in options[j]}
    gap = F(budget)-dual["bound"]
    penalties = {(j,b):F(b)+dual["weighted"][j]/(1 << b)-dual["minimums"][j]
                 for j in range(32) for b in options[j]}
    common = math.lcm(gap.denominator, *(v.denominator for v in penalties.values()))
    gap_integer = int(gap*common)
    penalty = {key:int(value*common) for key,value in penalties.items()}
    fixed = [j for j in range(32) if len(options[j]) == 1]
    order = [j for j in range(32) if len(options[j]) > 1]
    order.sort(key=lambda j:(-max(penalty[j,b] for b in options[j]),
                             -max(terms[j,min(options[j])]),j))
    initial_cost = sum(options[j][0] for j in fixed)
    initial_penalty = sum(penalty[j,options[j][0]] for j in fixed)
    initial_response = [sum(terms[j,options[j][0]][k] for j in fixed) for k in range(10)]
    available = budget-initial_cost
    if available < 0:
        return receipt | {"status":"infeasible", "nodes":0, "reason":"fixed_cost"}
    # For each suffix and output, minimum response among allocations using at
    # most c bits. Different outputs may use different minimisers: the result
    # is still a valid coordinatewise lower bound on every common allocation.
    suffix = [None]*(len(order)+1)
    suffix[-1] = [[0]*(available+1) for _ in range(10)]
    for pos in reversed(range(len(order))):
        j = order[pos]
        layer = [[None]*(available+1) for _ in range(10)]
        for k in range(10):
            for c in range(available+1):
                values = [terms[j,b][k]+suffix[pos+1][k][c-b]
                          for b in options[j] if b <= c and suffix[pos+1][k][c-b] is not None]
                if values:
                    layer[k][c] = min(values)
        suffix[pos] = layer
    assignment = [options[j][0] for j in range(32)]
    counters = {"nodes":0,"penalty_pruned":0,"response_or_cost_pruned":0,"leaves":0}
    start = time.monotonic()
    interrupted = False

    def visit(pos, cost, spent, response):
        nonlocal interrupted
        counters["nodes"] += 1
        if counters["nodes"] > node_limit or (counters["nodes"] % 1000 == 0 and time.monotonic()-start > seconds):
            interrupted = True
            return None
        if spent > gap_integer:
            counters["penalty_pruned"] += 1
            return None
        remaining = budget-cost
        if remaining < 0 or any(suffix[pos][k][remaining] is None or
                                response[k]+suffix[pos][k][remaining] > threshold for k in range(10)):
            counters["response_or_cost_pruned"] += 1
            return None
        if pos == len(order):
            counters["leaves"] += 1
            assert cost <= budget and all(v <= threshold for v in response)
            return assignment.copy()
        j = order[pos]
        for b in sorted(options[j], key=lambda b:(penalty[j,b],b)):
            assignment[j] = b
            found = visit(pos+1,cost+b,spent+penalty[j,b],
                          [response[k]+terms[j,b][k] for k in range(10)])
            if found is not None or interrupted:
                return found
        return None

    found = visit(0,initial_cost,initial_penalty,initial_response)
    return receipt | counters | {"status":"incomplete" if interrupted else ("feasible" if found else "infeasible"),
                                  "allocation":found,"search_order":order,
                                  "response_common_denominator":str(factor),
                                  "response_threshold_integer":str(threshold),
                                  "penalty_common_denominator":str(common),
                                  "gap_integer":str(gap_integer),
                                  "seconds":time.monotonic()-start}


def optimise(seed, epsilon):
    filename,path,numerators,incumbent = source(seed,epsilon)
    original_bounds = exact_bounds(numerators,incumbent)
    assert max(original_bounds) <= epsilon
    U = sum(incumbent)
    initial_max = max(16,max(incumbent)+2)
    _,eq,response,cost = matrices(numerators,[list(range(initial_max+1))]*32)
    lp = linprog(cost,A_ub=response,b_ub=np.full(10,float(epsilon)),
                 A_eq=eq,b_eq=np.ones(32),bounds=(0,None),method="highs")
    assert lp.success,lp.message
    mu_numerators = [max(0,int(round(-v*MU_DEN))) for v in lp.ineqlin.marginals]
    mu = [F(v,MU_DEN) for v in mu_numerators]
    dual = lagrange(numerators,epsilon,mu)
    choices = allowed_bits(dual,U)
    assert all(choices)
    indexes,eq,response,cost = matrices(numerators,choices)
    constraints = [LinearConstraint(eq,np.ones(32),np.ones(32)),
                   LinearConstraint(response,np.full(10,-np.inf),np.full(10,float(epsilon)))]
    result = milp(cost,integrality=np.ones(len(indexes)),bounds=Bounds(np.zeros(len(indexes)),np.ones(len(indexes))),
                  constraints=constraints,options={"time_limit":25,"mip_rel_gap":0})
    allocation = incumbent.copy()
    if result.x is not None:
        selected = [indexes[i] for i,x in enumerate(result.x) if x > 0.5]
        if len(selected)==32 and len({j for j,b in selected})==32:
            proposed = [dict(selected)[j] for j in range(32)]
            if max(exact_bounds(numerators,proposed)) <= epsilon and sum(proposed) <= U:
                allocation = proposed
    lower = max(0,ceil(dual["bound"]))
    bound = exact_bounds(numerators,allocation)
    assert max(bound) <= epsilon
    upper = sum(allocation)
    branch = None
    if lower < upper:
        branch = exact_exclude(numerators,epsilon,dual,upper-1)
        if branch["status"] == "infeasible":
            lower = upper
        elif branch["status"] == "feasible":
            allocation = branch["allocation"]
            upper = sum(allocation)
            bound = exact_bounds(numerators,allocation)
            assert max(bound) <= epsilon
    record = {"seed":seed,"epsilon_exact":str(epsilon),"source_file":filename,
              "source_sha256":sha(path),"gain_denominator":DEN,
              "original_greedy_bits":U,"optimality_interval_bits":[lower,upper],
              "state_bits":upper,"bits_by_unit":allocation,
              "bounds_exact":[str(v) for v in bound],"maximum_bound_exact":str(max(bound)),
              "maximum_bound_float":float(max(bound)),
              "mu_numerators":mu_numerators,"mu_denominator":MU_DEN,
              "lagrange_minimising_bits":dual["minbits"],
              "lagrange_minima_exact":[str(v) for v in dual["minimums"]],
              "lagrange_lower_exact":str(dual["bound"]),"lagrange_lower_float":float(dual["bound"]),
              "lagrange_integer_lower":max(0,ceil(dual["bound"])),
              "milp_variable_count":len(indexes),"milp_status":int(result.status),
              "milp_objective":None if result.fun is None else float(result.fun),
              "milp_gap":None if getattr(result,"mip_gap",None) is None else float(result.mip_gap),
              "exact_branch_certificate":branch,
              "certificate":"exact_rational_Lagrange" if lower==upper and branch is None else
                            ("exact_rational_Lagrange_plus_integer_exclusion" if lower==upper else "exact_lower_upper_gap")}
    print(json.dumps({k:record[k] for k in ["seed","epsilon_exact","state_bits","optimality_interval_bits","lagrange_lower_float","certificate"]}),flush=True)
    return record


def verify(receipt):
    verified=[]
    for record in receipt["allocations"]:
        epsilon=F(record["epsilon_exact"])
        filename,path,numerators,_=source(record["seed"],epsilon)
        assert filename==record["source_file"] and sha(path)==record["source_sha256"]
        allocation=record["bits_by_unit"]
        assert all(type(b) is int and b>=0 for b in allocation) and len(allocation)==32
        bounds=exact_bounds(numerators,allocation)
        assert [str(v) for v in bounds]==record["bounds_exact"] and max(bounds)<=epsilon
        assert sum(allocation)==record["state_bits"]
        mu=[F(v,record["mu_denominator"]) for v in record["mu_numerators"]]
        assert all(v>=0 for v in mu)
        dual=lagrange(numerators,epsilon,mu)
        assert str(dual["bound"])==record["lagrange_lower_exact"]
        assert dual["minbits"]==record["lagrange_minimising_bits"]
        lower=max(0,ceil(dual["bound"]))
        branch=record["exact_branch_certificate"]
        if branch and branch["status"]=="infeasible":
            replay=exact_exclude(numerators,epsilon,dual,branch["budget"],
                                 node_limit=max(2_000_000,branch["nodes"]+100),seconds=60)
            assert replay["status"]=="infeasible"
            assert replay["allowed_bits"]==branch["allowed_bits"]
            assert replay["nodes"]==branch["nodes"]
            lower=max(lower,branch["budget"]+1)
        assert record["optimality_interval_bits"]==[lower,sum(allocation)]
        verified.append({"seed":record["seed"],"epsilon":str(epsilon),"interval":[lower,sum(allocation)]})
    return {"status":"PASS","optimisers_called":False,"arithmetic":"Fraction and integers","verified":verified}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify",type=Path)
    parser.add_argument("--sensitivity",action="store_true")
    args=parser.parse_args()
    out=HERE/"results"
    out.mkdir(exist_ok=True)
    if args.verify:
        result=verify(json.loads(args.verify.read_text()))
        (out/"precision_verification.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
        return
    epsilons=[F(1)] if not args.sensitivity else [F(1,4),F(1,2),F(1),F(2),F(4)]
    records=[optimise(seed,eps) for eps in epsilons for seed in SEEDS]
    receipt={"status":"complete","analysis":"post-test supplementary weight-only optimisation",
             "test_outcomes_read":False,"scope":"minimum sum of integer per-unit bits satisfying the declared sufficient global box bound; not all possible encoders or actual error",
             "script_sha256":sha(Path(__file__)),"allocations":records}
    path=out/("precision_optimality_sensitivity.json" if args.sensitivity else "precision_optimality.json")
    path.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"receipt":str(path),"verified":verify(receipt)["status"]}),flush=True)


if __name__=="__main__":
    main()
