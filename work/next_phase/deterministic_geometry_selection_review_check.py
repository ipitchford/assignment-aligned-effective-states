"""Independent deterministic arithmetic checks; no image access or construction."""
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import isqrt, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/deterministic_geometry_feasibility/numeric_checker.py"
OUTPUT = ROOT / "work/next_phase/deterministic_geometry_selection_review_check.json"
source_bytes = SOURCE.read_bytes()
tree = ast.parse(source_bytes)
names = {"fraction_record", "fraction_value", "match_metrics", "exact_loss_equal",
         "sqrt_bounds", "compare_loss", "compare_candidates"}
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in functions} == names
namespace = {"Fraction": Fraction, "isqrt": isqrt, "sqrt": sqrt}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)

checks = 0
def require(condition):
    global checks
    assert condition
    checks += 1

fr = namespace["fraction_record"]
records = []
for a_num in range(7):
    a = Fraction(a_num, 3)
    for r_num in range(13):
        r = Fraction(r_num, 6)
        q = r*r
        loss = {"base": fr((a-1)**2 + q + 1), "radicand": fr(q)}
        records.append((loss, (a-1)**2 + (r-1)**2))
for one, expected_one in records:
    for two, expected_two in records:
        expected_sign = (expected_one > expected_two) - (expected_one < expected_two)
        require(namespace["compare_loss"](one, two) == expected_sign)
        require(namespace["exact_loss_equal"](one, two) == (expected_sign == 0))

# Equal algebraic losses at different rational square roots and exact amplitude ties.
one = {"base": fr(Fraction(3)), "radicand": fr(Fraction(1))}
two = {"base": fr(Fraction(5)), "radicand": fr(Fraction(4))}
require(namespace["compare_loss"](one, two) == 0)
for numerator_one in (-2, -1, 1, 2):
    for numerator_two in (-2, -1, 1, 2):
        c1 = {"matching": {"loss_exact": one}, "amplitude_numerator": numerator_one}
        c2 = {"matching": {"loss_exact": two}, "amplitude_numerator": numerator_two}
        k1, k2 = (abs(numerator_one), numerator_one > 0), (abs(numerator_two), numerator_two > 0)
        require(namespace["compare_candidates"](c1, c2) == ((k1 > k2)-(k1 < k2)))

target = {"changed_pixels224": 100, "rgb_squared_difference_sum224": 10000}
for a_count, sq_count, expected_a, expected_r in (
    (80, 6400, True, True), (120, 14400, True, True),
    (79, 10000, False, True), (121, 10000, False, True),
    (100, 6399, True, False), (100, 14401, True, False),
):
    result = namespace["match_metrics"]({"changed_pixels224": a_count,
        "rgb_squared_difference_sum224": sq_count}, target)
    require(result["A_ratio_gate"] is expected_a)
    require(result["R_ratio_gate"] is expected_r)

# Independent exact square-root interval enclosure, including irrational cases.
for q in (Fraction(0), Fraction(1, 4), Fraction(2), Fraction(8), Fraction(16, 25), Fraction(36, 25)):
    for bits in (1, 4, 32):
        lower, upper = namespace["sqrt_bounds"](q, bits)
        require(lower >= 0 and lower*lower <= q <= upper*upper)
        require(upper-lower <= Fraction(1, 1 << bits))

receipt = {
    "utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS", "checks": checks,
    "numeric_checker_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "AST-extracted exact scalar selection arithmetic tested against rational oracles; no production CLI imported",
    "image_reads": 0, "renders": 0, "native_forwards": 0, "random_draws": 0,
}
with OUTPUT.open("x") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
