"""Frozen constants and integrity checks for the revised full-layer SAT audit."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
SOURCE_STUDY = WORKSPACE / "work/next_phase/sat_all17_validation_v3_20261018"
MODEL_DIR = WORKSPACE / "work/sat_case"
BANK = SOURCE_STUDY / "selected_bank.npz"
PRIOR_NATIVE = SOURCE_STUDY / "native_outputs.npz"
PRIOR_CONSTRUCTION = SOURCE_STUDY / "construction.json"
PRIOR_ANALYSIS = SOURCE_STUDY / "analysis.json"
PRIOR_NATIVE_RECEIPT = SOURCE_STUDY / "native_receipt.json"
PRIOR_REPLAY_RECEIPT = SOURCE_STUDY / "replay_receipt.json"
PRIOR_BANK_VERIFICATION = SOURCE_STUDY / "bank_verification.json"
PRIOR_NATIVE_SCRIPT = SOURCE_STUDY / "native.py"
RUNTIME = MODEL_DIR / "runtime.py"
MODEL_SOURCE = MODEL_DIR / "upstream/model.py"
CHECKPOINT = MODEL_DIR / "upstream/models/layers_2_heads_1_4_1M/900.pth"
HELPERS_SOURCE = MODEL_DIR / "upstream/helpers.py"
INTERPRETATION_SOURCE = MODEL_DIR / "upstream/interpretation.py"

SOURCE_HASHES = {
    BANK: "38a8980436b67d7af8b897b813f68ca23f5d72b088fbfde7c6d64d4e4529d35e",
    PRIOR_NATIVE: "20c3a3dbb0c5a161d5a53e12542e3ae2da88f55ca5df78d15305fa2de3a39fc6",
    PRIOR_CONSTRUCTION: "2257309670bd6523f1d5abcaef0a270b9f6b104fbc11d7250a4db48bc4190c0e",
    PRIOR_ANALYSIS: "d81470a4b98a0b04e5dddf0bff7d5b445955ff86346639b917ce9514d99ac2bd",
    PRIOR_NATIVE_RECEIPT: "97ef289558183bf50e3127d7a733d99ea4f9bbd8b186ebacbca73579cf085915",
    PRIOR_REPLAY_RECEIPT: "3ed7caf368e6ade1d6fe9c0e59ef827e4372874e81a92ff1e8432939ced3c6ab",
    PRIOR_BANK_VERIFICATION: "32a680c6ab37d30fb191a13874c97b34758f29171de21fe1b45727134d0f7940",
    PRIOR_NATIVE_SCRIPT: "0b2358f9baf91f7e9e677d1509b5d7d26c24b9de7b45b643fc2b3601eaaebea9",
    RUNTIME: "bbc76d5f2d4a856944f54269a6da2d388f0ea6aee81ab03d9486e2e5c8bca230",
    MODEL_SOURCE: "7555e1f2f27546225397aa12de4fde1f6c903e9804766de477fe4c02770691ea",
    CHECKPOINT: "d587446c64b7a8d68de93808ccf84a84364b85dcaf2149b8983b77895e24046d",
    HELPERS_SOURCE: "ece43db7317fa8e3ad6cb6fb40bb1ba9f98ecd9191cbec6689b0bc67c0a268fe",
    INTERPRETATION_SOURCE: "b8f87ceaecb0c2449f3f626983ec028f2d3661c394d08f7d96641f4e06ba2177",
}
UNITS = np.array([10, 41, 55, 77, 81, 185, 224, 245, 261, 291, 317, 326,
                  380, 411, 416, 492, 495], dtype=np.int16)
ASSIGNMENTS = np.array([16, 0, 28, 7, 14, 3, 1, 9, 27, 23, 17, 30,
                        19, 25, 28, 21, 2], dtype=np.int8)
COVERED_ASSIGNMENTS = np.array(sorted(set(map(int, ASSIGNMENTS))), dtype=np.int8)
UNCOVERED_ASSIGNMENTS = np.setdiff1d(np.arange(32, dtype=np.int8), COVERED_ASSIGNMENTS)
SOURCE_THRESHOLD = 0.5
SAT_TOKEN = 13
UNSAT_TOKEN = 14
EXPECTED_BASELINE_CORRECT = 7215
EXPECTED_ROWS = 7252
ROTATION_SEED = 2026101811
ROTATIONS = 63
RESAMPLE_SEED = 2026101812
RESAMPLES = 64
ALIGNMENT_THRESHOLDS = (0.90, 0.95, 0.98, 0.99)
NATIVE_BATCH_SIZE = 256
NATIVE_THREADS = 2
ATOL = 1e-4
RTOL = 1e-5
NATIVE_ENV = {"python": "3.11.14", "numpy": "1.26.4", "torch": "2.1.2", "einops": "0.8.1"}
ANALYSIS_ENV = {"python": "3.14.7", "numpy": "2.4.2"}
FREEZE_SCHEMA = "sat-full-layer-basis-static-freeze-v7"
FREEZE_ENV = "SAT_FULL_LAYER_FREEZE_SHA256"
FROZEN_FILES = ("protocol.md", "runtime_bindings.md", "protocol.py", "preflight.py",
                "preflight.json", "native.py", "analyze.py", "selfcheck.py", "selfcheck.json")
EXPECTED_NATIVE_HOST = {"machine": "arm64", "node": "Admins-Mac-mini.local", "processor": "arm",
                        "torch_config_sha256": "34394c67c950ca9585d92f1aa7954ba28d61ed463002565e2d28363e24a949b4",
                        "torch_cpu_capability": "NO AVX"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def native_host_fingerprint(torch_module) -> dict:
    capability = getattr(getattr(torch_module.backends, "cpu", None), "get_cpu_capability", None)
    return {"node": platform.node(), "processor": platform.processor(), "machine": platform.machine(),
            "torch_cpu_capability": capability() if callable(capability) else "unavailable",
            "torch_config_sha256": hashlib.sha256(torch_module.__config__.show().encode()).hexdigest()}


def verify_sources() -> dict:
    observed = {}
    for path, expected in SOURCE_HASHES.items():
        value = sha256(path)
        if value != expected:
            raise RuntimeError(f"source hash mismatch: {path}")
        observed[str(path.relative_to(WORKSPACE))] = value
    return observed


def verify_frozen_packet() -> dict:
    expected = os.environ.get(FREEZE_ENV, "")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise RuntimeError(f"{FREEZE_ENV} must contain the reviewed freeze SHA-256")
    freeze_path = HERE / "freeze.json"
    if sha256(freeze_path) != expected:
        raise RuntimeError("externally anchored freeze hash mismatch")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise RuntimeError("freeze schema mismatch")
    rows = freeze.get("files", [])
    if tuple(row.get("path") for row in rows) != FROZEN_FILES:
        raise RuntimeError("freeze inventory mismatch")
    hashes = {}
    for row in rows:
        path = HERE / row["path"]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"frozen file mismatch: {row['path']}")
        hashes[row["path"]] = row["sha256"]
    sources = verify_sources()
    check = json.loads((HERE / "selfcheck.json").read_text())
    if check.get("status") != "PASS":
        raise RuntimeError("selfcheck did not pass")
    for name in ("protocol.py", "native.py", "analyze.py", "selfcheck.py"):
        if check.get("tested_file_sha256", {}).get(name) != hashes[name]:
            raise RuntimeError(f"selfcheck linkage mismatch: {name}")
    return {"freeze_sha256": expected, "code_sha256": hashes,
            "selfcheck_sha256": hashes["selfcheck.json"], "source_sha256": sources}


def semantic_counts(formulas: np.ndarray) -> np.ndarray:
    clauses = np.array([(i + 5*a, j + 5*b) for i in range(5) for j in range(i+1, 5)
                        for a in range(2) for b in range(2)], dtype=np.int16)
    assignments = ((np.arange(32)[:, None] >> np.arange(4, -1, -1)) & 1).astype(bool)
    literal_truth = np.concatenate((assignments, ~assignments), axis=1).T
    clause_truth = literal_truth[clauses].any(axis=1)
    return (~clause_truth[np.asarray(formulas, dtype=np.int16)]).sum(axis=1, dtype=np.int16)


def strata(counts: np.ndarray, assignment: int) -> np.ndarray:
    counts = np.asarray(counts)
    nsol = np.count_nonzero(counts == 0, axis=1)
    target = counts[:, assignment] == 0
    out = np.full(len(counts), -1, dtype=np.int8)
    out[(nsol == 1) & target] = 0
    out[(nsol >= 2) & target] = 1
    out[nsol == 0] = 2
    out[(nsol == 1) & ~target] = 3
    out[(nsol >= 2) & ~target] = 4
    if np.any(out < 0):
        raise RuntimeError("unclassified semantic row")
    return out


def compare_prior_cache(arrays: dict, prior_arrays: dict) -> dict:
    deltas = {"max_pre_difference": float(np.max(np.abs(arrays["pre"][:, UNITS].astype(np.float64) - prior_arrays["pre"].astype(np.float64)))),
              "max_post_difference": float(np.max(np.abs(arrays["post"][:, UNITS].astype(np.float64) - prior_arrays["post"].astype(np.float64)))),
              "max_logit_difference": float(np.max(np.abs(arrays["logits"].astype(np.float64) - prior_arrays["logits"].astype(np.float64))))}
    close = {key: bool(np.allclose(arrays[key][:, UNITS] if key in ("pre", "post") else arrays[key],
                                  prior_arrays[key], atol=1e-6, rtol=1e-6))
             for key in ("pre", "post", "logits")}
    same_threshold_events = bool(np.array_equal(arrays["post"][:, UNITS] >= SOURCE_THRESHOLD,
                                                 prior_arrays["post"] >= SOURCE_THRESHOLD))
    same_task_argmax = bool(np.array_equal(arrays["logits"].argmax(axis=1),
                                            prior_arrays["logits"].argmax(axis=1)))
    if not all(close.values()) or not same_threshold_events or not same_task_argmax:
        raise RuntimeError("full cache exceeds magnitude-aware prior tolerance or changes semantic decisions")
    return {**deltas, "allclose_atol": 1e-6, "allclose_rtol": 1e-6,
            "allclose_by_array": close, "same_threshold_events": same_threshold_events,
            "same_task_argmax": same_task_argmax,
            "bitwise_equal": all(value == 0.0 for value in deltas.values())}


def exclusive_scope_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != 32:
        raise RuntimeError("exclusive-scope label shape mismatch")
    return labels.any(axis=1) & ~labels[:, COVERED_ASSIGNMENTS].any(axis=1)


def validate_replay_prerequisites(root: Path, freeze_sha256: str) -> dict:
    root = Path(root)
    receipt_path, output_path = root / "native_receipt.json", root / "full_native_outputs.npz"
    if (not (root / "forward_started_initial.json").exists() or
            (root / "failure_initial.json").exists() or
            not receipt_path.exists() or not output_path.exists()):
        raise RuntimeError("replay prerequisites absent")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "PASS" or receipt.get("stage") != "initial" or
            receipt.get("freeze_sha256") != freeze_sha256 or
            sha256(output_path) != receipt.get("output_sha256")):
        raise RuntimeError("initial receipt gate failed")
    return receipt


def validate_analysis_prerequisites(root: Path, freeze_sha256: str) -> dict:
    root = Path(root)
    required = ("forward_started_initial.json", "forward_started_replay.json",
                "native_receipt.json", "replay_receipt.json",
                "full_native_outputs.npz", "replay_outputs.npz")
    forbidden = ("failure_initial.json", "failure_replay.json", "analysis.json",
                 "attempt_analysis.json", "failure_analysis.json")
    if any(not (root / name).exists() for name in required) or any((root / name).exists() for name in forbidden):
        raise RuntimeError("analysis prerequisites absent")
    initial = json.loads((root / "native_receipt.json").read_text())
    replay = json.loads((root / "replay_receipt.json").read_text())
    for receipt, stage in ((initial, "initial"), (replay, "replay")):
        if (receipt.get("status") != "PASS" or receipt.get("stage") != stage or
                receipt.get("freeze_sha256") != freeze_sha256):
            raise RuntimeError(f"{stage} receipt gate failed")
    if (sha256(root / "full_native_outputs.npz") != initial.get("output_sha256") or
            sha256(root / "replay_outputs.npz") != replay.get("output_sha256") or
            replay.get("initial_output_sha256") != initial.get("output_sha256")):
        raise RuntimeError("analysis receipt linkage failed")
    return {"initial": initial, "replay": replay}


def validate_hook_dependencies(model, final, model_receipt: dict) -> dict:
    hook_points = {"pre": final.mlp.hook_pre, "post": final.mlp.hook_post,
                   "resid_mid": final.hook_resid_mid, "resid_post": final.hook_resid_post}
    if (any(not callable(getattr(hook_point, "add_hook", None)) for hook_point in hook_points.values()) or
            not callable(getattr(model, "remove_all_hooks", None))):
        raise RuntimeError("hook inventory mismatch before forward")
    json.dumps(model_receipt, allow_nan=False)
    return hook_points


def check_bank_tokens(tokens: np.ndarray, formulas: np.ndarray) -> None:
    clauses = np.array([(i + 5*a, j + 5*b) for i in range(5) for j in range(i+1, 5)
                        for a in range(2) for b in range(2)], dtype=np.int16)
    lookup = {tuple(map(int, row)): index for index, row in enumerate(clauses)}
    if tokens.shape != (EXPECTED_ROWS, 41) or formulas.shape != (EXPECTED_ROWS, 10):
        raise RuntimeError("bank shape mismatch")
    if np.any(tokens[:, 0:40:4] != 10) or np.any(tokens[:, 3:40:4] != 11) or np.any(tokens[:, 40] != 12):
        raise RuntimeError("token framing mismatch")
    decoded = np.empty_like(formulas)
    for slot in range(10):
        pairs = tokens[:, 1 + 4*slot:3 + 4*slot]
        decoded[:, slot] = [lookup.get(tuple(map(int, row)), -1) for row in pairs]
    if not np.array_equal(decoded, formulas):
        raise RuntimeError("token/formula row alignment mismatch")


def bank_preflight() -> dict:
    """Check every deterministic bank condition before a new model is loaded."""
    with np.load(BANK, allow_pickle=False) as z:
        tokens = z["unique_tokens"].copy()
        formulas = z["unique_clauses"].copy()
        base_ids = z["base_unique_ids"].copy()
        unique_assignments = z["unique_assignment_ids"].copy()
        units = z["unit_ids"].copy()
        assignments = z["unit_assignment_ids"].copy()
    if tokens.shape != (EXPECTED_ROWS, 41) or formulas.shape != (EXPECTED_ROWS, 10):
        raise RuntimeError("bank row shape mismatch")
    check_bank_tokens(tokens, formulas)
    if base_ids.shape != (16, 5, 32):
        raise RuntimeError("base id shape mismatch")
    if not np.array_equal(unique_assignments, COVERED_ASSIGNMENTS):
        raise RuntimeError("covered assignment axis mismatch")
    if not np.array_equal(units, UNITS) or not np.array_equal(assignments, ASSIGNMENTS):
        raise RuntimeError("published mapping mismatch")
    counts = semantic_counts(formulas)
    labels = counts == 0
    sat = labels.any(axis=1)
    nsol = labels.sum(axis=1)
    clause_presence = np.column_stack([np.any(formulas == clause_id, axis=1) for clause_id in range(40)])
    control_labels = np.column_stack([clause_presence, sat, nsol == 1, nsol >= 2])
    if np.any(control_labels.sum(axis=0) == 0) or np.any(control_labels.sum(axis=0) == len(labels)):
        raise RuntimeError("control label missing a class")
    for scope, scope_labels in (("full", labels), ("sat_only", labels[sat])):
        positive = scope_labels.sum(axis=0)
        negative = len(scope_labels) - positive
        if np.any(positive == 0) or np.any(negative == 0):
            raise RuntimeError(f"missing assignment class in {scope}")
    stratum_sizes = {}
    for unit, assignment in zip(UNITS, ASSIGNMENTS):
        semantic = strata(counts, int(assignment))
        sizes = [int(np.count_nonzero(semantic == group)) for group in range(5)]
        if any(size == 1 for size in sizes):
            raise RuntimeError(f"singleton resample stratum for unit {int(unit)}")
        stratum_sizes[str(int(unit))] = sizes
    covered_any = labels[:, COVERED_ASSIGNMENTS].any(axis=1)
    exclusive_scope = exclusive_scope_from_labels(labels)
    overlap = {}
    for assignment in UNCOVERED_ASSIGNMENTS:
        assignment = int(assignment)
        positive = labels[:, assignment]
        exclusive = positive & exclusive_scope
        exclusive_negative = ~positive & exclusive_scope
        row = {"positives": int(positive.sum()),
               "also_covered_solution": int((positive & covered_any).sum()),
               "exclusive_positives": int(exclusive.sum()),
               "exclusive_sat_negatives": int(exclusive_negative.sum()),
               "sole_positive": int((positive & (nsol == 1)).sum())}
        if row["exclusive_positives"] < 20 or row["exclusive_sat_negatives"] < 20:
            raise RuntimeError("uncovered exclusive SAT scope has fewer than 20 examples in a class")
        overlap[str(assignment)] = row
    for ai, assignment in enumerate(unique_assignments):
        semantic = strata(counts, int(assignment))
        for si in range(5):
            ids = np.asarray(base_ids[ai, si]).ravel()
            if ids.size != 32 or ids.min() < 0 or ids.max() >= EXPECTED_ROWS or len(np.unique(ids)) != len(ids):
                raise RuntimeError("invalid base row ids")
            if not np.all(semantic[ids] == si):
                raise RuntimeError("assignment convention or stratum linkage mismatch")
    with np.load(PRIOR_NATIVE, allow_pickle=False) as prior:
        required_prior = {"pre": (EXPECTED_ROWS, len(UNITS)), "post": (EXPECTED_ROWS, len(UNITS)),
                          "logits": (EXPECTED_ROWS, 16), "unique_ids": (EXPECTED_ROWS,)}
        if set(required_prior) - set(prior.files):
            raise RuntimeError("prior cache inventory mismatch")
        prior_arrays = {key: prior[key].copy() for key in required_prior}
    for key, shape in required_prior.items():
        if prior_arrays[key].shape != shape or not np.isfinite(prior_arrays[key]).all():
            raise RuntimeError(f"prior cache shape or finiteness mismatch: {key}")
    for key in ("pre", "post", "logits"):
        if prior_arrays[key].dtype != np.float32:
            raise RuntimeError(f"prior cache dtype mismatch: {key}")
    if prior_arrays["unique_ids"].dtype != np.int32:
        raise RuntimeError("prior unique-id dtype mismatch")
    if not np.array_equal(prior_arrays["post"], np.maximum(prior_arrays["pre"], 0)):
        raise RuntimeError("prior cache exact ReLU identity mismatch")
    if not np.array_equal(prior_arrays["unique_ids"], np.arange(EXPECTED_ROWS, dtype=np.int32)):
        raise RuntimeError("prior cache row order mismatch")
    prior_receipt = json.loads(PRIOR_NATIVE_RECEIPT.read_text())
    if prior_receipt.get("batch_size_ceiling") != NATIVE_BATCH_SIZE:
        raise RuntimeError("prior batch size mismatch")
    if prior_receipt.get("model", {}).get("threads") != NATIVE_THREADS:
        raise RuntimeError("prior thread count mismatch")
    if prior_receipt.get("environment") != NATIVE_ENV:
        raise RuntimeError("prior native environment mismatch")
    prior_analysis = json.loads(PRIOR_ANALYSIS.read_text())
    names = ("sole_positive", "multi_positive", "unsat_negative", "sole_negative", "multi_negative")
    rows = prior_analysis.get("base", {}).get("per_unit", [])
    if len(rows) != 17:
        raise RuntimeError("prior per-unit inventory mismatch")
    for unit, row in zip(UNITS, rows):
        cells = row.get("threshold_cells", [])
        if row.get("unit") != int(unit) or len(cells) != 5:
            raise RuntimeError("prior threshold-cell inventory mismatch")
        assignment = int(ASSIGNMENTS[list(UNITS).index(unit)])
        ai = int(np.flatnonzero(unique_assignments == assignment)[0])
        for si, (name, cell) in enumerate(zip(names, cells)):
            detected = int(np.count_nonzero(prior_arrays["post"][base_ids[ai, si], list(UNITS).index(unit)] >= SOURCE_THRESHOLD))
            if (cell.get("stratum") != name or cell.get("denominator") != 32 or
                    "detected" not in cell or cell.get("detected") != detected):
                raise RuntimeError("prior threshold-cell schema mismatch")
    prior_task_label = np.where(sat, SAT_TOKEN, UNSAT_TOKEN)
    prior_correct = int(np.count_nonzero(prior_arrays["logits"].argmax(axis=1) == prior_task_label))
    if prior_correct != EXPECTED_BASELINE_CORRECT:
        raise RuntimeError("prior baseline task accuracy mismatch")
    construction = json.loads(PRIOR_CONSTRUCTION.read_text())
    replay_ids = np.asarray(construction.get("replay_unique_ids", []), dtype=np.int32)
    if replay_ids.shape != (8,) or len(np.unique(replay_ids)) != 8 or np.any(replay_ids < 0) or np.any(replay_ids >= EXPECTED_ROWS):
        raise RuntimeError("prior replay id inventory mismatch")
    return {"status": "PASS", "rows": EXPECTED_ROWS,
            "positive_counts_full": labels.sum(axis=0).astype(int).tolist(),
            "positive_counts_sat_only": labels[sat].sum(axis=0).astype(int).tolist(),
            "sat_rows": int(sat.sum()), "unsat_rows": int((~sat).sum()),
            "stratum_sizes_by_unit": stratum_sizes,
            "uncovered_overlap": overlap,
            "control_positive_counts": control_labels.sum(axis=0).astype(int).tolist(),
            "prior_cache": {"keys_checked": sorted(required_prior),
                            "column_identity": "hash-pinned prior native.py extracts post/pre columns in UNITS order",
                            "max_abs_pre": float(np.max(np.abs(prior_arrays["pre"]))),
                            "max_abs_post": float(np.max(np.abs(prior_arrays["post"]))),
                            "max_abs_logits": float(np.max(np.abs(prior_arrays["logits"]))),
                            "min_abs_post_minus_threshold": float(np.min(np.abs(prior_arrays["post"] - SOURCE_THRESHOLD))),
                            "min_top_two_logit_gap": float(np.min(np.sort(prior_arrays["logits"], axis=1)[:, -1] - np.sort(prior_arrays["logits"], axis=1)[:, -2])),
                            "baseline_correct": prior_correct,
                            "replay_unique_ids": replay_ids.astype(int).tolist()},
            "prior_batch_size": NATIVE_BATCH_SIZE, "prior_threads": NATIVE_THREADS,
            "prior_batch_partition": "hash-pinned native.py uses contiguous unique-id-ordered blocks range(0,N,256)",
            "scope": "bank semantics, assignment convention, resample feasibility and prior execution compatibility; zero new model access"}


def verify_live_preflight() -> dict:
    live = bank_preflight()
    frozen = json.loads((HERE / "preflight.json").read_text())
    core = {key: value for key, value in frozen.items()
            if key not in ("source_sha256", "protocol_sha256", "environment")}
    if live != core:
        raise RuntimeError("live bank preflight differs from frozen receipt")
    return live
