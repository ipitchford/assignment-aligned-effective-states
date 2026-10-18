#!/usr/bin/env python3
"""Synthetic/code checks only: no research cache reads, fits, freezes or model loads."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import platform
import sys
import tempfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np
import scipy
import sklearn
import screen as s

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = []

    def check(name, value):
        checks.append({"name": name, "pass": bool(value)})
        if not value:
            raise AssertionError(name)

    for name in ("screen.py", "static_selfcheck.py", "runtime_smoke.py"):
        ast.parse((HERE / name).read_text(), filename=name)
        check("Python syntax: " + name, True)
    check("No freeze exists", not (HERE / "freeze.json").exists())
    check("No run exists", not (HERE / "run").exists())
    check("Exactly seven fixed arms", len(s.ARMS) == 7 and len(set(s.ARMS)) == 7)
    check("Freeze binds all governance documents",
          tuple(path.name for path in s.GOVERNANCE) ==
          ("task_policy_augmentation_admission.md", "task_policy_cache_alignment_audit.md",
           "set_valued_policy_sufficiency.md", "task_policy_claim_ladder.md",
           "decisive_contribution_bridge.md", "task_policy_literature_update.md"))
    check("Freeze binds static evidence",
          tuple(path.name for path in s.EVIDENCE) ==
          ("static_selfcheck_receipt_v8.json", "runtime_smoke_receipt_v2.json",
           "IMPLEMENTATION_HANDOFF.md", "contract.md", "fit_pilot.py"))
    check("Fixed learner budget", s.SETTINGS == dict(max_iter=100, max_leaf_nodes=15,
          learning_rate=0.08, l2_regularization=1.0, max_bins=63,
          early_stopping=False, random_state=2026101606, class_weight=None))
    check("Hard computation caps", s.WALL_TIME_LIMIT_SECONDS == 1800 and
          s.MEMORY_LIMIT_BYTES == 4 * 1024**3 and s.TREE_LIMIT == 700 and
          s.LEAF_LIMIT == 10500)

    class Archive(dict):
        @property
        def files(self):
            return list(self.keys())
    contract_bank = Archive(tokens=np.zeros((4096, 41), dtype=np.int64),
                            clauses=np.zeros((4096, 10), dtype=np.int16),
                            post_sat=np.zeros((4096, 10, 40), dtype=bool))
    contract_native = Archive(pre=np.zeros((4096, 34), dtype=np.float32),
                              post=np.zeros((4096, 34), dtype=np.float32))
    s.verify_cache_contract(contract_bank, contract_native)
    check("Cache shape dtype and range contract accepts canonical fixture", True)
    bad_contract = Archive(contract_bank)
    bad_contract["post_sat"] = np.zeros((4096, 40, 10), dtype=bool)
    try:
        s.verify_cache_contract(bad_contract, contract_native)
        rejected_bad_contract = False
    except RuntimeError:
        rejected_bad_contract = True
    check("Cache contract rejects transposed outcome axes", rejected_bad_contract)

    bound = ("screen.py", "static_selfcheck.py", "runtime_smoke.py", "protocol.md",
             "requirements-analysis.txt")
    admission_lines = [f"{name} {s.sha(HERE / name)}" for name in bound]
    admission_lines += [f"{path.relative_to(s.ROOT)} {s.sha(path)}" for path in s.GOVERNANCE]
    admission_lines += ["DECISION: ADMIT_STATIC"]
    with tempfile.TemporaryDirectory() as temporary:
        admitted = Path(temporary) / "admission.md"
        admitted.write_text("\n".join(admission_lines) + "\n")
        s.validate_admission(admitted, bound, s.GOVERNANCE)
        rejected_nominal = False
        nominal = Path(temporary) / "nominal.md"
        nominal.write_text("looks fine\n")
        try:
            s.validate_admission(nominal, bound, s.GOVERNANCE)
        except RuntimeError:
            rejected_nominal = True
        nonadmitting = Path(temporary) / "nonadmitting.md"
        nonadmitting.write_text("The packet says DECISION: ADMIT_STATIC\n" +
                                "\n".join(admission_lines[:-1]) +
                                "\nDECISION: REVISE_STATIC\n")
        try:
            s.validate_admission(nonadmitting, bound, s.GOVERNANCE)
            rejected_nonadmitting = False
        except RuntimeError:
            rejected_nonadmitting = True
    check("Admission validator binds decision and exact hashes", rejected_nominal)
    check("Admission validator rejects quoted or revising decisions", rejected_nonadmitting)

    tokens = np.zeros((3, 41), dtype=np.int64)
    tokens[1, 0] = 1
    tokens[2, -1] = 12
    first = s.selected_slots(tokens)
    second = s.selected_slots(tokens.copy())
    check("Slot hashing deterministic", np.array_equal(first, second))
    check("Slot hashing bounded", first.dtype == np.int8 and np.all((first >= 0) & (first < 10)))
    altered = tokens.copy(); altered[0, 20] = 1
    check("Slot hash consumes canonical token bytes", not np.array_equal(first, s.selected_slots(altered)))

    check("Lower median odd", s.lower_median(np.array([-4, -1, -3], dtype=np.float32)) == -3)
    check("Lower median even", s.lower_median(np.array([-4, -1, -3, -2], dtype=np.float32)) == -3)
    pre = np.tile(np.array([-4, -3, -2, -1], dtype=np.float32)[:, None], (1, 34))
    post = np.maximum(pre, 0)
    pre[:, 1] = 1
    post[:, 1] = 1
    tau, upper, no_negative, no_upper = s.thresholds(pre, post, fit_stop=4)
    check("Negative lower median exact", tau[0] == -3 and not no_negative[0])
    check("No-negative flags and minus infinity", no_negative[1] and np.isneginf(tau[1]))
    check("No-upper fallback", no_upper[0] and upper[0] == np.float32(0.5))
    check("Upper lower median", not no_upper[1] and upper[1] == np.float32(1))

    fixture_pre = np.zeros((5, 34), dtype=np.float32)
    fixture_pre[:, 0] = np.array([-4, -2, -0.5, 0.25, 0.75], dtype=np.float32)
    fixture_post = np.maximum(fixture_pre, 0)
    fixture_tau = np.full(34, -1, dtype=np.float32)
    fixture_upper = np.full(34, 0.6, dtype=np.float32)
    fixture_flags = np.zeros(34, dtype=bool)
    reports = s.reports(fixture_pre, fixture_post, fixture_tau, fixture_upper, fixture_flags)
    check("All reports are 68 float32 columns",
          all(value.shape == (5, 68) and value.dtype == np.float32 for value in reports.values()))
    check("Candidate original bit threshold", np.array_equal(reports["negative_bin68"][:, 0], [0, 0, 0, 0, 1]))
    check("Candidate negative interval strict", np.array_equal(reports["negative_bin68"][:, 34], [0, 0, 1, 0, 0]))
    pre_codes = reports["pre_fourbin68"][:, :2].astype(int)
    check("Pre bins natural binary", np.array_equal(pre_codes, [[0,0],[0,0],[1,0],[0,1],[1,1]]))
    post_codes = reports["post_fourbin68"][:, :2].astype(int)
    check("Post bins natural binary", np.array_equal(post_codes, [[0,0],[0,0],[0,0],[1,0],[1,1]]))
    check("Published bits zero padding", np.all(reports["published_bits34"][:, 34:] == 0))
    check("Continuous arms zero padding", np.all(reports["continuous_pre34"][:, 34:] == 0) and
          np.all(reports["continuous_post34"][:, 34:] == 0))

    rows = np.array([[7, 9, 17, 39], [8, 0, 0, 1]], dtype=np.int32)
    descriptor = s.action_descriptor(rows)
    check("Action descriptor has16 bits", descriptor.shape == (2, 16))
    reconstructed_slot = (descriptor[:, :4] * (1 << np.arange(4))).sum(axis=1)
    reconstructed_old = (descriptor[:, 4:10] * (1 << np.arange(6))).sum(axis=1)
    reconstructed_new = (descriptor[:, 10:] * (1 << np.arange(6))).sum(axis=1)
    check("Action descriptor round trip", np.array_equal(reconstructed_slot, rows[:, 1]) and
          np.array_equal(reconstructed_old, rows[:, 2]) and np.array_equal(reconstructed_new, rows[:, 3]))
    synthetic_report = np.arange(9 * 68, dtype=np.float32).reshape(9, 68)
    matrix = s.design(synthetic_report, rows)
    check("Design has common84 columns", matrix.shape == (2, 84))
    check("Design selects origin report before action", np.array_equal(matrix[:, :68], synthetic_report[rows[:, 0]]))

    clauses = np.zeros((2, 10), dtype=np.int16)
    slots = np.array([0, 9], dtype=np.int8)
    menu = s.menu_rows(clauses, slots, 0, 2)
    check("Exactly39 ordered alternatives per menu", menu.shape == (78, 4) and
          np.array_equal(menu[:39, 3], np.arange(1, 40)) and np.array_equal(menu[39:, 3], np.arange(1, 40)))
    probabilities = np.zeros(78)
    probabilities[3] = probabilities[4] = 0.9
    probabilities[39 + 10] = 0.8
    outcomes = np.zeros(78, dtype=bool)
    outcomes[3] = True
    chosen = s.choose(probabilities, menu, outcomes)
    check("Probability ties choose lower clause ID", chosen["chosen_clause"][0] == 4)
    check("Choice uses supplied outcomes", bool(chosen["success"][0]) and not bool(chosen["success"][1]))
    summary = s.metrics(chosen)
    check("Utility definition exact", summary["mean_utility_at_0.8"] == -1.5)
    check("Regret algebra exact", summary["mean_regret"] == 0.0)

    ceiling_report = np.zeros((4, 68), dtype=np.float32)
    ceiling_rows = []
    ceiling_y = []
    success_sets = ({1, 2}, {2, 3}, {4}, set())
    for origin, successes in enumerate(success_sets):
        for new in range(1, 40):
            ceiling_rows.append((origin, 0, 0, new))
            ceiling_y.append(new in successes)
    ceiling = s.exact_policy_ceiling(ceiling_report, np.asarray(ceiling_rows, dtype=np.int32),
                                     np.asarray(ceiling_y, dtype=bool))
    check("Set-valued ceiling groups identical reports", ceiling["report_context_classes"] == 1)
    check("Set-valued ceiling exact reachable regret", ceiling["reachable_menus"] == 3 and
          ceiling["maximum_successes"] == 2 and ceiling["minimum_total_regret"] == 1)
    check("Set-valued ceiling detects empty common intersection",
          ceiling["reachable_classes"] == 1 and ceiling["reachable_classes_with_common_success"] == 0)
    check("Ceiling records exact class occupancy",
          ceiling["class_size_histogram"] == {"4": 1} and
          ceiling["singleton_class_fraction"] == 0.0 and
          ceiling["reachable_member_class_size_histogram"] == {"3": 1})

    metric_fixture = {
        "published_bits34": {"success_all_menus": 0.60},
        "post_fourbin68": {"success_all_menus": 0.61},
        "pre_fourbin68": {"success_all_menus": 0.67},
        "negative_bin68": {"success_all_menus": 0.66, "coverage_at_0.8": 0.20,
                           "mean_utility_at_0.8": 0.05},
    }
    ceiling_fixture = {
        "published_bits34": {"maximum_success_all_menus": 0.70},
        "post_fourbin68": {"maximum_success_all_menus": 0.69},
        "negative_bin68": {"maximum_success_all_menus": 0.76},
    }
    gates = s.development_gates(metric_fixture, ceiling_fixture)
    check("Development gates require all learned criteria", all(gates.values()))
    ceiling_fixture["negative_bin68"]["maximum_success_all_menus"] = 0.70
    no_information_gain = s.development_gates(metric_fixture, ceiling_fixture)
    check("Development gate stops decoder-only gain",
          not no_information_gain["ceiling_gain_over_published_at_least_005"] and
          not no_information_gain["ceiling_gain_over_post_fourbin_at_least_005"])

    node_dtype = np.dtype([("is_leaf", np.uint8)])
    class Tree:
        def __init__(self, leaves):
            self.nodes = np.zeros(leaves + 1, dtype=node_dtype)
            self.nodes["is_leaf"][:leaves] = 1
    class Model:
        n_iter_ = 2
        _predictors = [[Tree(3)], [Tree(4)]]
    check("Realized tree/leaf accounting", s.fitted_model_size(Model()) ==
          {"iterations": 2, "trees": 2, "leaves": 7})

    failure = s.failure_record(RuntimeError("fixture"), "prediction", "negative_bin68",
                               3, 2, {"published_bits34": {"menus": 8}},
                               {"published_bits34": {"menus": 8}},
                               {"published_bits34": {"trees": 3, "leaves": 15}},
                               {"python": "fixture"}, "freeze-fixture",
                               {"elapsed_seconds": 2.5, "peak_rss_bytes": 1024,
                                "trees": 3, "leaves": 15},
                               {"governance": "fixture"}, {"evidence": "fixture"},
                               {key: "2" for key in s.THREAD_KEYS})
    check("Failure receipt retains stage and fit accounting",
          failure["status"] == "INCOMPLETE_ERROR" and
          failure["stage"] == "prediction" and failure["arm"] == "negative_bin68" and
          failure["attempted_fits"] == 3 and failure["completed_fits"] == 2)
    check("Failure receipt retains partial metrics and resources",
          failure["completed_arm_metrics"]["published_bits34"]["menus"] == 8 and
          failure["completed_ceilings"]["published_bits34"]["menus"] == 8 and
          failure["resource_usage"]["peak_rss_bytes"] == 1024 and
          failure["governance_sha256"]["governance"] == "fixture" and
          failure["evidence_sha256"]["evidence"] == "fixture" and
          set(failure["thread_environment"].values()) == {"2"} and
          failure["native_forwards"] == 0)
    source = (HERE / "screen.py").read_text()
    check("NPZ output uses exclusive binary handle",
          'with (run / "screen_arrays.npz").open("xb") as handle:' in source and
          "np.savez_compressed(handle," in source)
    check("Terminal receipt is authoritative and exclusive",
          '"status": "CALCULATION_COMPLETE_OUTPUTS_UNCOMMITTED"' in source and
          'terminal = {"status": "COMPLETE"' in source and
          'write_json_atomic_exclusive(run / "terminal.json", terminal)' in source)
    check("Final cap check follows manifest hashing",
          source.index('write_json_exclusive(run / "manifest.json", manifest)') <
          source.index('final_resources = enforce_caps(started, model_sizes)') <
          source.index('write_json_atomic_exclusive(run / "terminal.json", terminal)'))
    check("JSON writes flush and sync before close", "os.fsync(handle.fileno())" in source)

    result = {
        "status": "PASS",
        "scope": "Synthetic and source checks only",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "thread_environment": {key: os.environ[key] for key in s.THREAD_KEYS},
        "bytecode_writes_disabled": bool(sys.dont_write_bytecode),
        "checks": checks,
        "check_count": len(checks),
        "research_caches_opened": 0,
        "fits": 0,
        "formulas_generated": 0,
        "native_forwards": 0,
        "freeze_created": False,
        "run_created": False,
        "code_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                         for name in ("screen.py", "static_selfcheck.py", "runtime_smoke.py", "protocol.md",
                                      "requirements-analysis.txt")},
    }
    if args.output:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "static_checks": len(checks), "fits": 0,
                      "native_forwards": 0, "freeze_created": False}, indent=2))


if __name__ == "__main__":
    main()
