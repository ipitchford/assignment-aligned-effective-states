#!/usr/bin/env python3
"""Read-only audit of the committed cached task-policy screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / "work/task_policy_augmentation"
RUN = IMPLEMENTATION / "run"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def recompute_gates(result: dict) -> dict[str, bool]:
    metrics = result["arms"]
    ceilings = result["exact_policy_ceilings"]
    candidate = metrics["negative_bin68"]
    candidate_ceiling = ceilings["negative_bin68"]
    return {
        "gain_over_published_at_least_005":
            candidate["success_all_menus"] - metrics["published_bits34"]["success_all_menus"] >= 0.05,
        "gain_over_post_fourbin_at_least_005":
            candidate["success_all_menus"] - metrics["post_fourbin68"]["success_all_menus"] >= 0.05,
        "within_002_of_pre_fourbin":
            candidate["success_all_menus"] - metrics["pre_fourbin68"]["success_all_menus"] >= -0.02,
        "coverage_at_08_at_least_01": candidate["coverage_at_0.8"] >= 0.10,
        "utility_at_08_positive": candidate["mean_utility_at_0.8"] > 0,
        "ceiling_gain_over_published_at_least_005":
            candidate_ceiling["maximum_success_all_menus"] -
            ceilings["published_bits34"]["maximum_success_all_menus"] >= 0.05,
        "ceiling_gain_over_post_fourbin_at_least_005":
            candidate_ceiling["maximum_success_all_menus"] -
            ceilings["post_fourbin68"]["maximum_success_all_menus"] >= 0.05,
    }


def audit() -> dict:
    freeze_path = IMPLEMENTATION / "freeze.json"
    terminal_path = RUN / "terminal.json"
    manifest_path = RUN / "manifest.json"
    result_path = RUN / "result.json"
    require(freeze_path.is_file(), "freeze.json missing")
    require(terminal_path.is_file(), "terminal.json missing")
    require(manifest_path.is_file(), "manifest.json missing")
    require(result_path.is_file(), "result.json missing")
    require(not (RUN / "error.json").exists(), "error.json conflicts with committed success")

    freeze = json.loads(freeze_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())

    require(freeze["status"] == "FROZEN_BEFORE_THRESHOLDS_OR_FITS", "freeze status changed")
    require(terminal["status"] == "COMPLETE", "terminal is not complete")
    require(result["status"] == "CALCULATION_COMPLETE_OUTPUTS_UNCOMMITTED", "result status changed")
    require(terminal["manifest_sha256"] == sha256(manifest_path), "terminal does not bind manifest")
    require(set(manifest) == {"models.pkl", "result.json", "screen_arrays.npz"}, "manifest members changed")
    for name, expected in manifest.items():
        if name == "models.pkl":
            continue  # Historical opaque bytes unavailable; numerical screen arrays remain checked.
        require(sha256(RUN / name) == expected, f"manifest mismatch: {name}")
    require(result["freeze_sha256"] == sha256(freeze_path), "result does not bind freeze")
    require(terminal["decision"] == result["decision"] == "STOP_DEVELOPMENT_GATE_FAILED",
            "decision changed")
    require(terminal["attempted_fits"] == terminal["completed_fits"] == 7, "terminal fit count changed")
    require(result["attempted_fits"] == result["completed_fits"] == 7, "result fit count changed")
    require(terminal["native_forwards"] == result["native_forwards"] == 0, "native forward occurred")
    require(result["fit_rows"] == 119_808 and result["assessment_rows"] == 39_936,
            "row count changed")

    observed_gates = recompute_gates(result)
    require(observed_gates == result["gates"], "stored gates do not recompute")
    require(not any(observed_gates.values()), "a development gate unexpectedly passed")

    expected_arms = {
        "context_only", "published_bits34", "negative_bin68", "pre_fourbin68",
        "post_fourbin68", "continuous_pre34", "continuous_post34",
    }
    require(set(result["arms"]) == expected_arms, "arm set changed")
    require(set(result["exact_policy_ceilings"]) == expected_arms, "ceiling arm set changed")
    for arm in expected_arms:
        metric = result["arms"][arm]
        ceiling = result["exact_policy_ceilings"][arm]
        require(metric["menus"] == ceiling["menus"] == 1024, f"menu count changed: {arm}")
        require(metric["reachable_menus"] == ceiling["reachable_menus"] == 850,
                f"reachable count changed: {arm}")
        require(ceiling["scope"] == "Exact finite assessment bank only", f"ceiling scope changed: {arm}")
        require(sum(int(size) * count for size, count in ceiling["class_size_histogram"].items()) == 1024,
                f"class histogram does not cover all menus: {arm}")
        require(sum(int(size) * count for size, count in
                    ceiling["reachable_member_class_size_histogram"].items()) == 850,
                f"reachable histogram does not cover all reachable menus: {arm}")

    candidate = result["arms"]["negative_bin68"]
    ceilings = result["exact_policy_ceilings"]
    metrics = result["arms"]
    deltas = {
        "candidate_minus_published_success":
            candidate["success_all_menus"] - metrics["published_bits34"]["success_all_menus"],
        "candidate_minus_post_fourbin_success":
            candidate["success_all_menus"] - metrics["post_fourbin68"]["success_all_menus"],
        "candidate_minus_pre_fourbin_success":
            candidate["success_all_menus"] - metrics["pre_fourbin68"]["success_all_menus"],
        "candidate_minus_published_ceiling":
            ceilings["negative_bin68"]["maximum_success_all_menus"] -
            ceilings["published_bits34"]["maximum_success_all_menus"],
        "candidate_minus_post_fourbin_ceiling":
            ceilings["negative_bin68"]["maximum_success_all_menus"] -
            ceilings["post_fourbin68"]["maximum_success_all_menus"],
    }
    require(deltas == {
        "candidate_minus_published_success": -18 / 1024,
        "candidate_minus_post_fourbin_success": 4 / 1024,
        "candidate_minus_pre_fourbin_success": -27 / 1024,
        "candidate_minus_published_ceiling": 0.0,
        "candidate_minus_post_fourbin_ceiling": 0.0,
    }, "candidate deltas changed")

    sizes = terminal["model_sizes"]
    require(set(sizes) == expected_arms, "model-size arm set changed")
    require(all(size == {"iterations": 100, "trees": 100, "leaves": 1500}
                for size in sizes.values()), "realized model size changed")
    resources = terminal["resource_usage"]
    require(resources["trees"] == 700 and resources["leaves"] == 10_500,
            "terminal totals changed")
    require(resources["elapsed_seconds"] <= result["resource_caps"]["wall_time_seconds"],
            "wall-time cap exceeded")
    require(resources["peak_rss_bytes"] <= result["resource_caps"]["resident_memory_bytes"],
            "memory cap exceeded")

    return {
        "status": "PASS",
        "scope": "Committed-output audit; no model or research-input decoding",
        "decision": terminal["decision"],
        "freeze_sha256": sha256(freeze_path),
        "terminal_sha256": sha256(terminal_path),
        "manifest_sha256": sha256(manifest_path),
        "result_sha256": sha256(result_path),
        "manifest_members_verified": len(manifest) - 1,
        "unavailable_original_byte_identity": "models.pkl",
        "gates_recomputed": observed_gates,
        "candidate_deltas": deltas,
        "candidate_success_all_menus": candidate["success_all_menus"],
        "best_learned_arm": max(metrics, key=lambda arm: metrics[arm]["success_all_menus"]),
        "best_learned_success_all_menus": max(item["success_all_menus"] for item in metrics.values()),
        "published_singleton_class_fraction": ceilings["published_bits34"]["singleton_class_fraction"],
        "candidate_singleton_class_fraction": ceilings["negative_bin68"]["singleton_class_fraction"],
        "exact_semantic_success_all_menus": result["exact_semantic_control"]["success_all_menus"],
        "attempted_fits": terminal["attempted_fits"],
        "completed_fits": terminal["completed_fits"],
        "native_forwards": terminal["native_forwards"],
        "resource_usage": resources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    receipt = audit()
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
