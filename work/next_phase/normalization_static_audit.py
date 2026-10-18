"""Source/metadata inspection only: no model import or image inference."""
from pathlib import Path
import ast
import hashlib
import json
import struct
import tomllib

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "work/inception_source_audit/sources"
model_path = SOURCES / "lucent_0_1_8/lucent/modelzoo/inceptionv1/InceptionV1.py"
graph_path = ROOT / "work/inception_feasibility/graph_metadata.json"
tree = ast.parse(model_path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
convs = {}
for statement in init.body:
    if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "Conv2d"):
        convs[statement.targets[0].attr] = {k.arg: ast.literal_eval(k.value) for k in statement.value.keywords}
graph = {n["name"]: n for n in json.loads(graph_path.read_text())}
needed = set()
def visit(name):
    name = name.split(":")[0]
    if name in needed:
        return
    needed.add(name)
    for parent in graph[name]["inputs"]:
        visit(parent)
visit("mixed4e")
checks = []
for name, kw in convs.items():
    base = name.removesuffix("_pre_relu_conv")
    weight_shape = graph[base + "_w"]["attrs"]["value"]["shape"]
    native = graph[base + "_pre_relu/conv"]
    checks.append({"source_module": name, "native_convolution": native["name"],
                   "in_target_prefix": native["name"] in needed,
                   "shape_matches": weight_shape == [*kw["kernel_size"], kw["in_channels"], kw["out_channels"]],
                   "stride_matches": list(kw["stride"]) == native["attrs"]["strides"][1:3],
                   "groups_one": kw["groups"] == 1, "bias_present": kw["bias"]})
lock_path = SOURCES / "narang/olt/uv.lock"
packages = tomllib.loads(lock_path.read_text())["package"]
alpha_torch = 9.99999974738e-05
alpha_tf32 = struct.unpack("f", struct.pack("f", alpha_torch / 9))[0]
identity_paths = [model_path, graph_path, lock_path, SOURCES / "lucent_0_1_8.whl",
                  SOURCES / "lucent_0_1_8/lucent/modelzoo/inceptionv1/helper_layers.py",
                  SOURCES / "narang/olt/src/olt/tfms.py"]
receipt = {
    "scope": "Static source/metadata audit; no weights equality, inference, gradients, or historical reproduction claimed",
    "input_files": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in identity_paths],
    "locked_versions": {p["name"]: p["version"] for p in packages if p["name"] in {"torch", "torch-lucent", "torchvision", "numpy", "pillow"}},
    "source_convolutions": len(checks), "target_prefix_convolutions": sum(c["in_target_prefix"] for c in checks),
    "all_checked_attributes_match": all(c[k] for c in checks for k in ["shape_matches", "stride_matches", "groups_one", "bias_present"]),
    "convolution_checks": checks,
    "lucent_lrn_alpha_literal": alpha_torch, "proposed_tf_alpha_float32": alpha_tf32,
    "proposed_tf_alpha_float32_hex_native_bytes": struct.pack("f", alpha_tf32).hex(),
    "no_inference": True,
}
out = Path(__file__).with_name("normalization_static_audit.json")
out.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: receipt[k] for k in ["source_convolutions", "target_prefix_convolutions", "all_checked_attributes_match", "proposed_tf_alpha_float32", "no_inference"]}, indent=2))
