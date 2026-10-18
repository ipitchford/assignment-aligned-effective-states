"""AST-only inverse-equation fixtures; no image reads or image construction."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/deterministic_geometry_feasibility/geometry.py"
OUTPUT = ROOT / "work/next_phase/deterministic_geometry_inverse_review_check.json"
source_bytes = SOURCE.read_bytes()
tree = ast.parse(source_bytes)
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve_inverse")
xx = np.array([[5., 6.], [7., 8.]])
yy = np.array([[9., 10.], [11., 12.]])
namespace = {"np": np, "INVERSE_ITERATIONS": 30, "NonfiniteCoordinates": RuntimeError,
             "grid": lambda local: (xx.copy(), yy.copy())}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
checks = []
for name, ax, ay, bx, by in (
    ("identity", 0., 0., 0., 0.),
    ("constant_translation", 0., 0., 2., -3.),
    ("anisotropic_affine", .1, -.2, 1., -2.),
):
    namespace["displacement"] = lambda x, y, kind, amplitude, ax=ax, ay=ay, bx=bx, by=by: (ax*x+bx, ay*y+by)
    qx, qy, residual = namespace["solve_inverse"]("fixture")
    expected_x, expected_y = (xx-bx)/(1+ax), (yy-by)/(1+ay)
    error = float(max(np.abs(qx-expected_x).max(), np.abs(qy-expected_y).max()))
    residual_max = float(residual.max())
    assert error < 1e-12 and residual_max < 1e-12
    checks.append({"fixture": name, "maximum_coordinate_error": error, "maximum_residual": residual_max})

receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "checks": checks,
    "geometry_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "Production inverse iteration AST with analytic affine displacement fixtures; actual construction fields not evaluated",
    "image_reads": 0, "renders": 0, "native_forwards": 0, "random_draws": 0}
with OUTPUT.open("x") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
