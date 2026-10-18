#!/usr/bin/env python3
"""Verify the package and replay only saved-data semantics/arithmetic."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A new directory outside the supplement; never overwrites receipts")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(ROOT):
        parser.error("Output directory must be new and outside the supplement")
    spec = importlib.util.spec_from_file_location("package_identity", ROOT / "checks/verify_package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package_identity = module.verify(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    work = ROOT / "work"
    study = work / "sat_renaming_launch_recovery"
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    jobs = [
        ("semantic", [sys.executable, "-B", str(study / "verify_bank.py"),
                      "--bank", str(study / "selected_bank.npz"), "--metadata", str(study / "selected_bank.json"),
                      "--ledger", str(study / "prior_ledger_tokens.npz"), "--ledger-metadata", str(study / "prior_ledger.json"),
                      "--work-root", str(work), "--output", str(output / "bank_replay.json")]),
        ("arithmetic", [sys.executable, "-B", str(ROOT / "checks/recompute_saved.py"),
                        "--work-root", str(work), "--output-dir", str(output)])]
    for name, command in jobs:
        completed = subprocess.run(command, env=environment, cwd=output, text=True, capture_output=True)
        (output / (name + ".stdout.txt")).write_text(completed.stdout)
        (output / (name + ".stderr.txt")).write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(name + " failed; inspect " + str(output))
    identity_after = module.verify(ROOT)
    assert identity_after == package_identity
    receipt = json.loads((output / "receipt.json").read_text())
    semantic = json.loads((output / "bank_replay.json").read_text())
    assert receipt["status"] == semantic["status"] == "PASS"
    summary = {"status": "PASS", "scope": "Saved-data portable replay only", "package": package_identity,
               "arithmetic_checks": receipt["checks_passed"], "semantic_check_counts": semantic["passed_check_counts"],
               "failed_study_criteria": receipt["bootstrap_replay"]["failed_criteria"],
               "new_neural_forwards": 0, "new_formulas": 0, "fits": 0,
               "new_random_values": 0, "saved_bootstrap_draws_reused": 2000,
               "original_source_files_modified": 0}
    with (output / "portable_replay_summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "semantic_check_counts"}, indent=2))


if __name__ == "__main__":
    main()
