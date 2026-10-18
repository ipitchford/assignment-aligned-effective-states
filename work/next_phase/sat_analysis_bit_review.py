"""Independent deterministic bit-only review. No formulas, model, files of outcomes, or RNG.

AST-load only analysis functions and arithmetic statements; never import protocol or
call the scientific entrypoint. Retain source hashes in the printed receipt.
"""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "sat_renaming"
source = (ROOT / "analyze.py").read_text()
tree = ast.parse(source)
allowed = {"radius", "bit_bounds", "correctness", "enumerate_admissible", "triple", "endpoint_statistics", "change_statistics", "task_bounds", "minimum_condition", "maximum_condition", "classify_conditions"}
env = {"np": np, "ATOL": 1e-4, "RTOL": 1e-5,
       "PERMUTATIONS": {"correct": np.array([1, 0, 3, 2]), "identity": np.arange(4), "wrong": np.array([3, 2, 1, 0])},
       "COHORT_ORDER": ["positive", "negative"], "PAIRS": [("P1", 27, 23), ("P2", 25, 21)],
       "ANALYSIS_SECONDS": 300, "deadline": lambda *args: None}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in allowed]
assert {f.name for f in functions} == allowed
exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / "analyze.py"), "exec"), env)
checks = {}

# Independent scalar reference: all 3^8 endpoint interval boxes. Sum of all
# admissible assignments across these boxes is only 4^8 = 65,536.
perms = ((1, 0, 3, 2), (0, 1, 2, 3), (3, 2, 1, 0))
for box in itertools.product(((0, 0), (1, 1), (0, 1)), repeat=8):
    lo = np.array([v[0] for v in box], dtype=np.int8).reshape(2, 4)
    hi = np.array([v[1] for v in box], dtype=np.int8).reshape(2, 4)
    reference = []
    for bits in itertools.product(*(range(v[0], v[1]+1) for v in box)):
        a, b = bits[:4], bits[4:]
        err = [sum(b[u] != a[p[u]] for u in range(4))/4 for p in perms]
        reference.append(err + [err[1]-err[0], err[2]-err[0]] + [sum(b[u] != a[perms[0][u]] for u in (0, 1))])
    ref = np.asarray(reference)
    actual = env["endpoint_statistics"](lo, lo, hi, np.array([0, 1]))
    assembled = np.concatenate((actual[0], actual[1], actual[2][None, :]), axis=0)
    assert np.array_equal(assembled[:, 1], ref.min(axis=0))
    assert np.array_equal(assembled[:, 2], ref.max(axis=0))
checks["exhaustive_endpoint_interval_boxes"] = 3**8

# Signed changes can be negative; independent scalar calculation over every
# four-vector assignment for the fully ambiguous box.
lo = np.zeros((4, 4), dtype=np.int8); hi = np.ones_like(lo)
reference = []
for bits in itertools.product((0, 1), repeat=16):
    vectors = [bits[4*i:4*i+4] for i in range(4)]
    a = [vectors[1][i]-vectors[0][i] for i in range(4)]
    b = [vectors[3][i]-vectors[2][i] for i in range(4)]
    reference.append([sum(abs(b[u]-a[p[u]]) for u in range(4))/4 for p in perms]
                     + [sum(b[u] != a[perms[0][u]] for u in (2, 3))])
ref = np.asarray(reference)
actual = env["change_statistics"](lo, lo, hi, np.array([2, 3]))
assembled = np.concatenate((actual[0], actual[1][None, :]), axis=0)
assert np.array_equal(assembled[:, 1], ref.min(axis=0))
assert np.array_equal(assembled[:, 2], ref.max(axis=0))
checks["exhaustive_signed_change_assignments"] = 2**16

# Full-vocabulary winner and ties, including a non-answer winner.
logits = np.zeros((3, 16)); logits[0, 13] = 1; logits[1, [13, 14]] = 1; logits[2, 2] = 1
pred, possible, correct = env["task_bounds"](logits, np.array([13, 13, 14]))
assert pred.tolist() == [13, 13, 2]
assert [np.flatnonzero(x).tolist() for x in possible] == [[13], [13, 14], [2]]
assert correct.tolist() == [[1, 1, 1], [1, 0, 1], [0, 0, 0]]
checks["full_vocabulary_and_tie_cases"] = 3

# Extract only pre-RNG arithmetic and benchmark construction from main. No
# scientific entrypoint, RNG call, file IO, formula or native inference executes.
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
start_index = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and any(isinstance(t, ast.Name) and t.id == "bits" for t in n.targets[0].elts))
end_index = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "rng" for t in n.targets))
arithmetic = main.body[start_index:end_index]
benchmark_start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and any(isinstance(t, ast.Name) and t.id == "benchmark" for t in n.targets[0].elts))
benchmark_end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "status" for t in n.targets)) + 1
benchmark = main.body[benchmark_start:benchmark_end]
arithmetic_code = compile(ast.Module(body=arithmetic, type_ignores=[]), str(ROOT / "analyze.py"), "exec")
benchmark_code = compile(ast.Module(body=benchmark, type_ignores=[]), str(ROOT / "analyze.py"), "exec")

pair_ids = np.tile([0, 1], 32)
target = np.zeros((64, 2, 4, 4), dtype=np.int8)
labels = np.full((64, 2, 4), 14, dtype=np.int16); labels[:, 0, [1, 3]] = 13
for block, pair in enumerate(pair_ids):
    target[block, 0, 1, 2*pair] = 1
    target[block, 0, 3, 2*pair+1] = 1
bank = {"pair_index": pair_ids, "expected_unit_bits": target, "labels": labels}
logits = np.zeros((64, 2, 4, 16), dtype=np.float32)
np.put_along_axis(logits, labels[..., None], 1, axis=-1)
scenario_receipts = []
for scenario in ("exact_pattern", "all_zero", "all_one", "seven_uncertain_positive_bits"):
    post = target.astype(np.float32)
    if scenario == "all_zero": post.fill(0)
    if scenario == "all_one": post.fill(1)
    if scenario == "seven_uncertain_positive_bits":
        for block in np.flatnonzero(pair_ids == 0)[:7]: post[block, 0, 1, 0] = .5
    scope = dict(env, bank=bank, native={"post": post, "logits": logits}, start=0)
    exec(arithmetic_code, scope)
    # Constant quantile inputs exercise adjudication only; no resampling occurs.
    scope["intervals"] = np.stack([scope["point"], scope["point"]])
    exec(benchmark_code, scope)
    expected_status = {"exact_pattern": "PASS", "all_zero": "CRITERION_NOT_MET", "all_one": "CRITERION_NOT_MET", "seven_uncertain_positive_bits": "UNRESOLVED_NUMERICAL_SENSITIVITY"}[scenario]
    assert scope["status"] == expected_status
    assert scope["denominators"]["primary_sensitivity"]*32 == 64
    assert scope["denominators"]["primary_specificity"]*32 == 448
    assert scope["denominators"]["off_pair_false_positive_rate"]*32 == 512
    assert scope["denominators"]["original_sensitivity"]*32 == 32
    assert scope["denominators"]["original_specificity"]*32 == 224
    if scenario == "exact_pattern":
        for name in ("primary_sensitivity", "primary_specificity", "positive_exact_pattern", "negative_exact_pattern", "task_accuracy"):
            assert np.all(scope["point"][:, scope["names"].index(name)] == 1)
        for name in ("edited_advantage_identity", "edited_advantage_wrong"):
            assert np.all(scope["point"][:, scope["names"].index(name)] == .5)
    if scenario == "all_one":
        assert np.all(scope["point"][:, scope["names"].index("off_pair_false_positive_rate")] == 1)
    scenario_receipts.append({"scenario": scenario, "status": scope["status"]})
checks["metric_and_benchmark_scenarios"] = scenario_receipts

print(json.dumps({"status": "PASS", "scope": "Deterministic bit-only code check; no formulas, neural outputs, RNG calls or scientific bootstrap draws.",
                  "analyze_sha256": hashlib.sha256(source.encode()).hexdigest(), "numpy": np.__version__, "checks": checks}, indent=2))
