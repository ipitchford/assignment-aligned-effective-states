#!/usr/bin/env python3
"""Prospective cached task-policy screen. Freeze and execution are separate."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import pickle
import resource
import signal
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BANK = ROOT / "work/sat_case/pilot_bank.npz"
NATIVE = ROOT / "work/sat_mechanism/original_native.npz"
MAPPING = ROOT / "work/sat_mechanism/source_mapping.json"
ADMISSION = ROOT / "work/next_phase/task_policy_augmentation_admission.md"
ALIGNMENT = ROOT / "work/next_phase/task_policy_cache_alignment_audit.md"
THEORY = ROOT / "work/next_phase/set_valued_policy_sufficiency.md"
CLAIM_LADDER = ROOT / "work/next_phase/task_policy_claim_ladder.md"
CONTRIBUTION_BRIDGE = ROOT / "work/next_phase/decisive_contribution_bridge.md"
LITERATURE = ROOT / "work/next_phase/task_policy_literature_update.md"
STATIC_RECEIPT = HERE / "static_selfcheck_receipt_v8.json"
SMOKE_RECEIPT = HERE / "runtime_smoke_receipt_v2.json"
HANDOFF = HERE / "IMPLEMENTATION_HANDOFF.md"
ORIGINAL_CONTRACT = ROOT / "work/sat_case/contract.md"
ORIGINAL_PILOT_CODE = ROOT / "work/sat_case/fit_pilot.py"
GOVERNANCE = (ADMISSION, ALIGNMENT, THEORY, CLAIM_LADDER, CONTRIBUTION_BRIDGE, LITERATURE)
EVIDENCE = (STATIC_RECEIPT, SMOKE_RECEIPT, HANDOFF, ORIGINAL_CONTRACT, ORIGINAL_PILOT_CODE)

INPUT_HASHES = {
    str(BANK.relative_to(ROOT)): "610d02cc85ba825b7ac3100efd30ad1ec8f90f02e11a9fe8444f0c62a9e2971b",
    str(NATIVE.relative_to(ROOT)): "54f86e7c03c2366345662ce65a90e62201d71869a60f88e5544a267b49e941c2",
    str(MAPPING.relative_to(ROOT)): "5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d",
    str(ALIGNMENT.relative_to(ROOT)): "f7b2c611f6686a98ba923e3cfe1eddd1b6b7ec4454e38543e8c99e3d4e8e9fba",
}
TAG = b"already-clean-task-policy-augmentation-v1\0"
ARMS = ("context_only", "published_bits34", "negative_bin68", "pre_fourbin68",
        "post_fourbin68", "continuous_pre34", "continuous_post34")
SETTINGS = dict(max_iter=100, max_leaf_nodes=15, learning_rate=0.08,
                l2_regularization=1.0, max_bins=63, early_stopping=False,
                random_state=2026101606, class_weight=None)
WALL_TIME_LIMIT_SECONDS = 30 * 60
MEMORY_LIMIT_BYTES = 4 * 1024**3
TREE_LIMIT = 700
LEAF_LIMIT = 10500
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_exclusive(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic_exclusive(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    write_json_exclusive(pending, value)
    try:
        os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def selected_slots(tokens):
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or tokens.shape[1] != 41:
        raise ValueError("Expected canonical 41-token rows")
    slots = []
    for row in tokens:
        payload = np.asarray(row, dtype="<i8").tobytes(order="C")
        slots.append(int.from_bytes(hashlib.sha256(TAG + payload).digest()[:8], "big") % 10)
    return np.asarray(slots, dtype=np.int8)


def lower_median(values):
    values = np.sort(np.asarray(values, dtype=np.float32))
    if not len(values):
        raise ValueError("Empty median input")
    return values[(len(values) - 1) // 2]


def thresholds(pre, post, fit_stop=3072):
    pre, post = np.asarray(pre), np.asarray(post)
    tau = np.empty(34, dtype=np.float32)
    upper = np.empty(34, dtype=np.float32)
    no_negative = np.zeros(34, dtype=bool)
    no_upper = np.zeros(34, dtype=bool)
    for j in range(34):
        negative = pre[:fit_stop, j][pre[:fit_stop, j] < 0]
        if len(negative):
            tau[j] = lower_median(negative)
        else:
            tau[j] = -np.inf
            no_negative[j] = True
        high = post[:fit_stop, j][post[:fit_stop, j] >= 0.5]
        if len(high):
            upper[j] = lower_median(high)
        else:
            upper[j] = np.float32(0.5)
            no_upper[j] = True
    return tau, upper, no_negative, no_upper


def binary_columns(values, width):
    values = np.asarray(values)
    return ((values[..., None] >> np.arange(width, dtype=np.int64)) & 1).astype(np.float32)


def reports(pre, post, tau, upper, no_negative):
    result = {}
    zeros = np.zeros((len(pre), 68), dtype=np.float32)
    bits = (pre >= 0.5).astype(np.float32)
    result["context_only"] = zeros.copy()
    result["published_bits34"] = np.concatenate((bits, np.zeros_like(bits)), axis=1)
    negative = ((pre > tau) & (pre < 0) & (~no_negative)).astype(np.float32)
    result["negative_bin68"] = np.concatenate((bits, negative), axis=1)
    pre_bin = np.where(pre <= tau, 0, np.where(pre < 0, 1, np.where(pre < 0.5, 2, 3)))
    post_bin = np.where(post == 0, 0, np.where(post < 0.5, 1,
                       np.where(post <= upper, 2, 3)))
    result["pre_fourbin68"] = binary_columns(pre_bin, 2).reshape(len(pre), 68)
    result["post_fourbin68"] = binary_columns(post_bin, 2).reshape(len(post), 68)
    result["continuous_pre34"] = np.concatenate((pre.astype(np.float32), np.zeros_like(pre, dtype=np.float32)), axis=1)
    result["continuous_post34"] = np.concatenate((post.astype(np.float32), np.zeros_like(post, dtype=np.float32)), axis=1)
    return result


def menu_rows(clauses, slots, start, stop):
    rows = []
    for origin in range(start, stop):
        slot = int(slots[origin])
        old = int(clauses[origin, slot])
        for new in range(40):
            if new != old:
                rows.append((origin, slot, old, new))
    result = np.asarray(rows, dtype=np.int32)
    if result.shape != ((stop - start) * 39, 4):
        raise AssertionError("Unexpected menu row count")
    return result


def action_descriptor(rows):
    rows = np.asarray(rows)
    return np.concatenate((binary_columns(rows[:, 1], 4),
                           binary_columns(rows[:, 2], 6),
                           binary_columns(rows[:, 3], 6)), axis=1)


def design(report, rows):
    return np.concatenate((report[rows[:, 0]], action_descriptor(rows)), axis=1).astype(np.float32)


def labels(post_sat, rows):
    return ~post_sat[rows[:, 0], rows[:, 1], rows[:, 3]]


def choose(probabilities, rows, outcomes):
    p = np.asarray(probabilities).reshape(-1, 39)
    r = np.asarray(rows).reshape(-1, 39, 4)
    y = np.asarray(outcomes).reshape(-1, 39)
    # Rows are already ascending by clause ID. argmax therefore provides the declared tie break.
    index = np.argmax(p, axis=1)
    k = np.arange(len(index))
    return dict(chosen_clause=r[k, index, 3], confidence=p[k, index],
                success=y[k, index], reachable=y.any(axis=1))


def metrics(choice):
    success = choice["success"]
    reachable = choice["reachable"]
    confidence = choice["confidence"]
    result = {
        "menus": int(len(success)),
        "reachable_menus": int(reachable.sum()),
        "success_all_menus": float(success.mean()),
        "success_when_reachable": float(success[reachable].mean()) if reachable.any() else None,
        "mean_regret": float((reachable.astype(np.int8) - success.astype(np.int8)).mean()),
    }
    for threshold in (0.5, 0.8):
        accepted = confidence >= threshold
        result[f"coverage_at_{threshold}"] = float(accepted.mean())
        result[f"success_among_accepted_at_{threshold}"] = (float(success[accepted].mean())
                                                               if accepted.any() else None)
    accepted = confidence >= 0.8
    utility = np.where(~accepted, 0, np.where(success, 1, -4))
    result["mean_utility_at_0.8"] = float(utility.mean())
    return result


def exact_policy_ceiling(report, rows, outcomes):
    report = np.ascontiguousarray(report)
    menu_rows = np.asarray(rows).reshape(-1, 39, 4)
    menu_outcomes = np.asarray(outcomes, dtype=bool).reshape(-1, 39)
    groups = {}
    for menu_index, menu in enumerate(menu_rows):
        origin, slot, old = map(int, menu[0, :3])
        key = (report[origin].tobytes(order="C"), slot, old)
        if key not in groups:
            groups[key] = {"reachable": 0, "success_counts": np.zeros(40, dtype=np.int64),
                           "intersection": np.ones(40, dtype=bool), "members": 0,
                           "reachable_members": 0}
        group = groups[key]
        group["members"] += 1
        success_set = np.zeros(40, dtype=bool)
        success_set[menu[:, 3]] = menu_outcomes[menu_index]
        if success_set.any():
            group["reachable"] += 1
            group["reachable_members"] += 1
            group["success_counts"] += success_set
            group["intersection"] &= success_set
    maximum_success = 0
    zero_regret_classes = 0
    chosen = []
    for group in groups.values():
        if not group["reachable_members"]:
            continue
        best = int(np.flatnonzero(group["success_counts"] == group["success_counts"].max())[0])
        chosen.append(best)
        maximum_success += int(group["success_counts"][best])
        zero_regret_classes += int(group["intersection"].any())
    reachable = int(sum(group["reachable"] for group in groups.values()))
    minimum_regret = reachable - maximum_success
    class_sizes = np.asarray([group["members"] for group in groups.values()], dtype=np.int64)
    reachable_sizes = np.asarray([group["reachable_members"] for group in groups.values()
                                  if group["reachable_members"] > 0], dtype=np.int64)
    def histogram(values):
        sizes, counts = np.unique(values, return_counts=True)
        return {str(int(size)): int(count) for size, count in zip(sizes, counts)}
    return {"menus": len(menu_rows), "report_context_classes": len(groups),
            "class_size_histogram": histogram(class_sizes),
            "singleton_class_fraction": float((class_sizes == 1).mean()),
            "reachable_member_class_size_histogram": histogram(reachable_sizes),
            "reachable_menus": reachable, "maximum_successes": maximum_success,
            "maximum_success_all_menus": maximum_success / len(menu_rows),
            "maximum_success_when_reachable": maximum_success / reachable if reachable else None,
            "minimum_total_regret": minimum_regret,
            "minimum_mean_regret": minimum_regret / len(menu_rows),
            "reachable_classes_with_common_success": zero_regret_classes,
            "reachable_classes": sum(int(group["reachable_members"] > 0) for group in groups.values()),
            "tie_break": "lowest clause ID", "scope": "Exact finite assessment bank only"}


def development_gates(arm_metrics, ceilings):
    candidate = arm_metrics["negative_bin68"]
    candidate_ceiling = ceilings["negative_bin68"]
    return {
        "gain_over_published_at_least_005":
            candidate["success_all_menus"] - arm_metrics["published_bits34"]["success_all_menus"] >= 0.05,
        "gain_over_post_fourbin_at_least_005":
            candidate["success_all_menus"] - arm_metrics["post_fourbin68"]["success_all_menus"] >= 0.05,
        "within_002_of_pre_fourbin":
            candidate["success_all_menus"] - arm_metrics["pre_fourbin68"]["success_all_menus"] >= -0.02,
        "coverage_at_08_at_least_01": candidate["coverage_at_0.8"] >= 0.10,
        "utility_at_08_positive": candidate["mean_utility_at_0.8"] > 0,
        "ceiling_gain_over_published_at_least_005":
            candidate_ceiling["maximum_success_all_menus"] -
            ceilings["published_bits34"]["maximum_success_all_menus"] >= 0.05,
        "ceiling_gain_over_post_fourbin_at_least_005":
            candidate_ceiling["maximum_success_all_menus"] -
            ceilings["post_fourbin68"]["maximum_success_all_menus"] >= 0.05,
    }


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def fitted_model_size(model):
    predictors = [tree for iteration in model._predictors for tree in iteration]
    leaves = sum(int(np.asarray(tree.nodes)["is_leaf"].sum()) for tree in predictors)
    return {"iterations": int(model.n_iter_), "trees": len(predictors), "leaves": leaves}


def resource_snapshot(started, model_sizes):
    elapsed = time.monotonic() - started
    rss = peak_rss_bytes()
    trees = sum(item["trees"] for item in model_sizes.values())
    leaves = sum(item["leaves"] for item in model_sizes.values())
    return {"elapsed_seconds": elapsed, "peak_rss_bytes": rss,
            "trees": trees, "leaves": leaves}


def enforce_caps(started, model_sizes):
    observed = resource_snapshot(started, model_sizes)
    elapsed = observed["elapsed_seconds"]
    rss = observed["peak_rss_bytes"]
    trees = observed["trees"]
    leaves = observed["leaves"]
    if elapsed > WALL_TIME_LIMIT_SECONDS:
        raise RuntimeError("Development wall-time cap exceeded")
    if rss > MEMORY_LIMIT_BYTES:
        raise RuntimeError("Development resident-memory cap exceeded")
    if trees > TREE_LIMIT or leaves > LEAF_LIMIT:
        raise RuntimeError("Development tree/leaf cap exceeded")
    return observed


def failure_record(exc, stage, arm, attempted_fits, completed_fits,
                   arm_metrics, ceilings, model_sizes, versions,
                   freeze_sha256, observed_resources, governance_sha256,
                   evidence_sha256, thread_environment):
    return {"status": "INCOMPLETE_ERROR", "stage": stage, "arm": arm,
            "type": type(exc).__name__, "message": str(exc),
            "attempted_fits": attempted_fits, "completed_fits": completed_fits,
            "completed_arm_metrics": arm_metrics, "completed_ceilings": ceilings,
            "model_sizes": model_sizes, "resource_usage": observed_resources,
            "versions": versions, "freeze_sha256": freeze_sha256,
            "input_sha256": INPUT_HASHES,
            "governance_sha256": governance_sha256,
            "evidence_sha256": evidence_sha256,
            "thread_environment": thread_environment, "native_forwards": 0}


def verify_inputs():
    for relative, expected in INPUT_HASHES.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("Input identity changed: " + relative)


def verify_cache_contract(bank, native):
    expected = {
        "tokens": ((4096, 41), np.dtype("int64")),
        "clauses": ((4096, 10), None),
        "post_sat": ((4096, 10, 40), np.dtype("bool")),
    }
    for name, (shape, dtype) in expected.items():
        if name not in bank.files or bank[name].shape != shape:
            raise RuntimeError(f"Cache contract changed: {name} shape")
        if dtype is not None and bank[name].dtype != dtype:
            raise RuntimeError(f"Cache contract changed: {name} dtype")
    clauses = bank["clauses"]
    if not np.issubdtype(clauses.dtype, np.integer):
        raise RuntimeError("Cache contract changed: clauses dtype")
    if int(clauses.min()) < 0 or int(clauses.max()) >= 40:
        raise RuntimeError("Cache contract changed: clauses range")
    for name in ("pre", "post"):
        if name not in native.files or native[name].shape != (4096, 34):
            raise RuntimeError(f"Cache contract changed: native {name} shape")


def validate_admission(path, bound, governance):
    if HERE in path.parents:
        raise RuntimeError("Independent admission must be outside the implementation directory")
    text = path.read_text()
    decisions = [line.strip() for line in text.splitlines()
                 if line.strip().startswith("DECISION:")]
    if decisions != ["DECISION: ADMIT_STATIC"] or text.rstrip().splitlines()[-1].strip() != decisions[0]:
        raise RuntimeError("Independent admission lacks one final ADMIT_STATIC decision")
    required = {name: sha(HERE / name) for name in bound}
    required.update({str(item.relative_to(ROOT)): sha(item) for item in governance})
    missing = [f"{name} {digest}" for name, digest in required.items()
               if name not in text or digest not in text]
    if missing:
        raise RuntimeError("Independent admission does not bind exact inputs: " + ", ".join(missing))


def prepare_freeze(admission):
    verify_inputs()
    if not admission.exists():
        raise FileNotFoundError(admission)
    if (HERE / "freeze.json").exists() or (HERE / "run").exists():
        raise RuntimeError("Freeze or run already exists")
    bound = ("screen.py", "static_selfcheck.py", "runtime_smoke.py", "protocol.md",
             "requirements-analysis.txt")
    validate_admission(admission, bound, GOVERNANCE)
    record = {
        "status": "FROZEN_BEFORE_THRESHOLDS_OR_FITS",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "independent_static_admission": {"path": str(admission), "sha256": sha(admission)},
        "input_sha256": INPUT_HASHES,
        "governance_sha256": {str(path.relative_to(ROOT)): sha(path) for path in GOVERNANCE},
        "evidence_sha256": {str(path.relative_to(ROOT)): sha(path) for path in EVIDENCE},
        "code_sha256": {name: sha(HERE / name) for name in bound},
        "admission_card_sha256": sha(ADMISSION),
        "arms": list(ARMS),
        "settings": SETTINGS,
        "native_forwards": 0,
    }
    write_json_exclusive(HERE / "freeze.json", record)
    print(json.dumps({"status": record["status"], "native_forwards": 0}, indent=2))


def verify_freeze():
    freeze = json.loads((HERE / "freeze.json").read_text())
    verify_inputs()
    for name, expected in freeze["code_sha256"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError("Frozen code changed: " + name)
    if sha(ADMISSION) != freeze["admission_card_sha256"]:
        raise RuntimeError("Admission card changed")
    for relative, expected in freeze["governance_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("Frozen governance changed: " + relative)
    for relative, expected in freeze["evidence_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("Frozen evidence changed: " + relative)
    if sha(Path(freeze["independent_static_admission"]["path"])) != freeze["independent_static_admission"]["sha256"]:
        raise RuntimeError("Independent static admission changed")
    return freeze


def execute():
    freeze = verify_freeze()
    run = HERE / "run"
    if run.exists():
        raise RuntimeError("Refusing repeat execution")
    run.mkdir()
    started = time.monotonic()
    def timeout_handler(_signum, _frame):
        raise TimeoutError("Development wall-time cap reached")
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(WALL_TIME_LIMIT_SECONDS)
    stage = "runtime_imports"
    current_arm = None
    attempted_fits = 0
    completed_fits = 0
    versions = {}
    arm_metrics, choices, ceilings, model_sizes = {}, {}, {}, {}
    terminal = None
    try:
        import platform
        import scipy
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier
        versions = {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "sklearn": sklearn.__version__}
        expected_versions = {"python": "3.11.14", "numpy": "2.4.6",
                             "scipy": "1.17.1", "sklearn": "1.9.0"}
        if versions != expected_versions:
            raise RuntimeError(f"Analysis runtime changed: {versions!r}")
        stage = "load_inputs"
        bank = np.load(BANK, allow_pickle=False)
        native = np.load(NATIVE, allow_pickle=False)
        verify_cache_contract(bank, native)
        if not np.array_equal(bank["published_units"], native["units"]):
            raise AssertionError("Unit order changed")
        slots = selected_slots(bank["tokens"])
        fit_rows = menu_rows(bank["clauses"], slots, 0, 3072)
        assessment_rows = menu_rows(bank["clauses"], slots, 3072, 4096)
        fit_y = labels(bank["post_sat"], fit_rows)
        assessment_y = labels(bank["post_sat"], assessment_rows)
        tau, upper, no_negative, no_upper = thresholds(native["pre"], native["post"])
        report_bank = reports(native["pre"], native["post"], tau, upper, no_negative)
        models = {}
        for arm in ARMS:
            current_arm = arm
            model = HistGradientBoostingClassifier(**SETTINGS)
            stage = "pre_fit"
            fit_design = design(report_bank[arm], fit_rows)
            enforce_caps(started, model_sizes)
            attempted_fits += 1
            stage = "fit"
            model.fit(fit_design, fit_y)
            completed_fits += 1
            model_sizes[arm] = fitted_model_size(model)
            enforce_caps(started, model_sizes)
            del fit_design
            stage = "pre_prediction"
            assessment_design = design(report_bank[arm], assessment_rows)
            enforce_caps(started, model_sizes)
            stage = "prediction"
            probability = model.predict_proba(assessment_design)[:, list(model.classes_).index(True)]
            enforce_caps(started, model_sizes)
            del assessment_design
            stage = "analysis"
            choice = choose(probability, assessment_rows, assessment_y)
            arm_metrics[arm] = metrics(choice)
            choices[arm] = choice
            ceilings[arm] = exact_policy_ceiling(report_bank[arm], assessment_rows, assessment_y)
            models[arm] = model
            enforce_caps(started, model_sizes)
        current_arm = None
        stage = "final_analysis"
        gates = development_gates(arm_metrics, ceilings)
        exact = choose(assessment_y.astype(np.float64), assessment_rows, assessment_y)
        calculation_resources = enforce_caps(started, model_sizes)
        decision = "GO_PROSPECTIVE_DESIGN" if all(gates.values()) else "STOP_DEVELOPMENT_GATE_FAILED"
        result = {
            "status": "CALCULATION_COMPLETE_OUTPUTS_UNCOMMITTED",
            "decision": decision,
            "scope": "Opened cached development outcomes; zero new native forwards; not protected evaluation",
            "freeze_sha256": sha(HERE / "freeze.json"),
            "fit_rows": int(len(fit_rows)), "assessment_rows": int(len(assessment_rows)),
            "arms": arm_metrics, "exact_policy_ceilings": ceilings,
            "exact_semantic_control": metrics(exact), "gates": gates,
            "elapsed_seconds": time.monotonic() - started, "native_forwards": 0,
            "versions": versions,
            "thread_environment": {key: os.environ[key] for key in THREAD_KEYS},
            "model_sizes": model_sizes, "calculation_resources": calculation_resources,
            "attempted_fits": attempted_fits, "completed_fits": completed_fits,
            "resource_caps": {"wall_time_seconds": WALL_TIME_LIMIT_SECONDS,
                              "resident_memory_bytes": MEMORY_LIMIT_BYTES,
                              "trees": TREE_LIMIT, "leaves": LEAF_LIMIT},
        }
        stage = "serialize_arrays"
        with (run / "screen_arrays.npz").open("xb") as handle:
            np.savez_compressed(handle, slots=slots, fit_rows=fit_rows,
                                assessment_rows=assessment_rows, fit_unsat=fit_y,
                                assessment_unsat=assessment_y, tau=tau, upper=upper,
                                no_negative=no_negative, no_upper=no_upper,
                                **{f"{arm}_chosen": choices[arm]["chosen_clause"] for arm in ARMS},
                                **{f"{arm}_confidence": choices[arm]["confidence"] for arm in ARMS},
                                **{f"{arm}_success": choices[arm]["success"] for arm in ARMS})
        enforce_caps(started, model_sizes)
        stage = "serialize_models"
        with (run / "models.pkl").open("xb") as handle:
            pickle.dump(models, handle, protocol=5)
        enforce_caps(started, model_sizes)
        stage = "serialize_result"
        write_json_exclusive(run / "result.json", result)
        enforce_caps(started, model_sizes)
        stage = "hash_outputs"
        manifest = {p.name: sha(p) for p in sorted(run.iterdir())}
        write_json_exclusive(run / "manifest.json", manifest)
        final_resources = enforce_caps(started, model_sizes)
        stage = "commit_terminal"
        terminal = {"status": "COMPLETE", "decision": decision,
                    "manifest_sha256": sha(run / "manifest.json"),
                    "attempted_fits": attempted_fits, "completed_fits": completed_fits,
                    "model_sizes": model_sizes, "resource_usage": final_resources,
                    "native_forwards": 0}
        write_json_atomic_exclusive(run / "terminal.json", terminal)
    except Exception as exc:
        failure = failure_record(exc, stage, current_arm, attempted_fits, completed_fits,
                                 arm_metrics, ceilings, model_sizes, versions,
                                 sha(HERE / "freeze.json"),
                                 resource_snapshot(started, model_sizes),
                                 freeze["governance_sha256"], freeze["evidence_sha256"],
                                 {key: os.environ[key] for key in THREAD_KEYS})
        write_json_exclusive(run / "error.json", failure)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(json.dumps({"status": terminal["status"], "decision": terminal["decision"],
                      "gates": gates, "native_forwards": 0,
                      "elapsed_seconds": terminal["resource_usage"]["elapsed_seconds"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare-freeze")
    freeze.add_argument("--admission", required=True, type=Path)
    sub.add_parser("execute")
    args = parser.parse_args()
    if args.command == "prepare-freeze":
        prepare_freeze(args.admission.resolve())
    else:
        execute()


if __name__ == "__main__":
    main()
