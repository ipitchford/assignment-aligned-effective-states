#!/usr/bin/env python3
"""Stage and replay the frozen 5,407-check cached SAT comparator audit.

Usage: python src/replay_sat_margin_audit.py /path/to/new-replay-directory
Requires NumPy. No sklearn, pickle loading, fitting, or model inference is used.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import posixpath
import shutil
import subprocess
import sys

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
BASELINE_MANIFEST = "3a2c115056d9512d72095f96e8a6766691663122a2c10997496e70e21a31edc5"
REVIEW_MANIFEST = "94aab5659bd7a45764aabfa14217c60c0e41db7f8754fbe6f5332f6e214aa60e"
CHECKER = "a4beacc97383da74dc2e5ec696cde8c0e5ba4f5f2774a212c3d72195f2d869c9"
GENERATED = ("checks.json", "unit_metrics_recomputed.csv", "bootstrap_recomputed.npz")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


PRESENTATION_PATH = BUNDLE / 'PUBLIC_PRESENTATION_DERIVATIVES.json'
require(sha(PRESENTATION_PATH) == '324de49b83ab2bc5d63606852d56f837ab6c91ccca031defc1e83a2798eef618', 'Presentation map identity mismatch')
PRESENTATION = read(PRESENTATION_PATH)

def historical_identity(path):
    """Return recorded identity only for an explicitly pinned presentation derivative."""
    path=Path(path)
    actual=sha(path)
    for row in PRESENTATION['files']:
        if actual == row['public_sha256']:
            if path.stat().st_size != row['public_bytes']:
                raise RuntimeError('Presentation derivative size mismatch')
            return row['original_sha256'], row['original_bytes']
    return actual, path.stat().st_size

def dependency_plan():
    """Resolve every consumed/hash-checked file before creating the destination."""
    baseline = BUNDLE / "sat_margin_baseline"
    review = BUNDLE / "review/sat_margin_baseline"
    require(sha(baseline / "MANIFEST.json") == BASELINE_MANIFEST, "Unexpected baseline manifest")
    require(sha(review / "MANIFEST.json") == REVIEW_MANIFEST, "Unexpected review manifest")
    bm, rm = read(baseline / "MANIFEST.json"), read(review / "MANIFEST.json")
    plan = {}

    def add(target, source, expected):
        target = posixpath.normpath(target)
        require(not target.startswith(("/", "../")) and target != "..", "Invalid staging path")
        source = source.resolve()
        require(source.is_relative_to(BUNDLE) and source.is_file(), f"Missing packaged input: {source}")
        require(historical_identity(source)[0] == expected, f"Packaged input hash mismatch: {source}")
        item = dict(source=str(source.relative_to(BUNDLE)), sha256=sha(source), historical_sha256=expected,
                    bytes=source.stat().st_size)
        require(target not in plan or plan[target] == item, f"Conflicting staging target: {target}")
        plan[target] = item

    add("sat_margin_baseline/MANIFEST.json", baseline / "MANIFEST.json", BASELINE_MANIFEST)
    for name, entry in bm["files"].items():
        if name.endswith(".pkl"):
            continue  # Original hash/size retained in public omission catalogue.
        add("sat_margin_baseline/" + name, baseline / name, entry["sha256"])
        require(historical_identity(baseline / name)[1] == entry["bytes"], f"Size mismatch: {name}")
    freeze = read(baseline / "freeze.json")
    for name, expected in freeze["input_sha256"].items():
        target = posixpath.normpath("sat_margin_baseline/" + name)
        # The redistributed contract is already present with identical bytes.
        source = baseline / "contract.md" if target == "next_phase/sat_margin_baseline_contract.md" else BUNDLE / target
        add(target, source, expected)
    mechanism = read(BUNDLE / "sat_mechanism/MANIFEST.json")
    add("sat_mechanism/query_tokens.npz", BUNDLE / "sat_mechanism/query_tokens.npz",
        mechanism["files"]["query_tokens.npz"]["sha256"])

    # Preserve the historical reviewer files separately from newly generated outputs.
    add("reference_review/MANIFEST.json", review / "MANIFEST.json", REVIEW_MANIFEST)
    for name, entry in rm["files"].items():
        if name == "recompute.py":
            add("reference_review/" + name, review / name, CHECKER)
            continue
        add("reference_review/" + name, review / name, entry["sha256"])
        require(historical_identity(review / name)[1] == entry["bytes"], f"Review size mismatch: {name}")
    add("sat_margin_baseline_review/recompute.py", review / "recompute.py", CHECKER)
    add("PUBLIC_PRESENTATION_DERIVATIVES.json", PRESENTATION_PATH, "324de49b83ab2bc5d63606852d56f837ab6c91ccca031defc1e83a2798eef618")
    return plan


def compare_outputs(destination):
    reference = destination / "reference_review"
    produced = destination / "sat_margin_baseline_review"
    expected, actual = read(reference / "checks.json"), read(produced / "checks.json")
    require(actual["status"] == "PASS" and actual["checks"] == 5390, "The public audit did not pass its 5,390 available checks")
    require(all(actual[k] == 0 for k in ["classifier_fits", "neural_forwards", "tree_predictions"]), "Unexpected computation scope")
    # Only wall-clock timestamp changes. Assertion names, numbers and conclusions agree.
    require(actual["checker_sha256"] == CHECKER, "Public checker identity mismatch")
    expected["assertions"] = [a for a in expected["assertions"] if not (a.startswith("manifest:") and a.endswith(".pkl"))]
    expected["checks"] = len(expected["assertions"])
    expected.pop("checker_sha256")
    actual.pop("checker_sha256")
    expected.pop("audit_utc")
    actual.pop("audit_utc")
    require(actual == expected, "Scientific checks differ from the retained review")
    require((reference / GENERATED[1]).read_bytes() == (produced / GENERATED[1]).read_bytes(),
            "Recomputed unit metrics differ")
    with np.load(reference / GENERATED[2], allow_pickle=False) as left, np.load(produced / GENERATED[2], allow_pickle=False) as right:
        require(set(left.files) == set(right.files), "Bootstrap array names differ")
        for name in left.files:
            require(left[name].dtype == right[name].dtype and left[name].shape == right[name].shape
                    and np.array_equal(left[name], right[name], equal_nan=True),
                    f"Bootstrap array differs: {name}")
    return dict(assertions=5390, checks_equal_except_declared_presentation_and_omissions=True,
                unavailable_original_pickle_hash_checks=17,
                pickle_byte_counts="historical manifest metadata, not revalidated bytes",
                unit_metrics_bytes_equal=True, bootstrap_arrays_equal=True,
                primary=actual["aggregate"]["primary_assessment"], bootstrap=actual["bootstrap"],
                classifier_fits=0, neural_forwards=0, tree_predictions=0, new_bootstrap_draws=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="Fresh directory outside the source bundle; must not exist")
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    require(not destination.exists(), f"Destination already exists: {destination}")
    require(not destination.is_relative_to(BUNDLE), "Destination must be outside the source bundle")
    plan = dependency_plan()
    destination.mkdir(parents=True, exist_ok=False)
    receipt_path = destination / "portable_replay_receipt.json"
    receipt = dict(status="STAGING", start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   helper_sha256=sha(__file__), python=sys.version, numpy=np.__version__,
                   baseline_manifest_sha256=BASELINE_MANIFEST, review_manifest_sha256=REVIEW_MANIFEST,
                   staged_files=plan)

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for target, item in plan.items():
            out = destination / target
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BUNDLE / item["source"], out)
            require(sha(out) == item["sha256"], f"Staged input mismatch: {target}")
        checker = destination / "sat_margin_baseline_review/recompute.py"
        receipt.update(status="RUNNING", command=[sys.executable, "-I", "-B", str(checker)])
        save()
        env = os.environ.copy()
        for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env[name] = "1"
        with (destination / "replay.log").open("w") as log:
            result = subprocess.run(receipt["command"], cwd=checker.parent, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        receipt["returncode"] = result.returncode
        require(result.returncode == 0, "Retained checker failed; inspect replay.log")
        receipt["comparison"] = compare_outputs(destination)
        for target, item in plan.items():
            require(sha(destination / target) == item["sha256"], f"A staged input changed: {target}")
            require(sha(BUNDLE / item["source"]) == item["sha256"], f"A packaged input changed: {item['source']}")
        receipt["outputs"] = {name: dict(sha256=sha(checker.parent / name),
                                         bytes=(checker.parent / name).stat().st_size) for name in GENERATED}
        receipt.update(status="PASS", source_inputs_unchanged=True, staged_inputs_unchanged=True,
                       staged_file_count=len(plan), staged_bytes=sum(v["bytes"] for v in plan.values()))
    except Exception as error:
        receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    print(json.dumps(dict(status=receipt["status"], receipt=str(receipt_path),
                          **receipt["comparison"]), indent=2))


if __name__ == "__main__":
    main()
