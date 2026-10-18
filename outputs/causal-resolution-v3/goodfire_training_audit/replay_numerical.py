#!/usr/bin/env python3
"""Verify local files and run the independent numerical-only saved-data checker."""
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
    before = module.verify(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    command = [sys.executable, "-B", str(ROOT / "checks/recompute.py"),
               "--audit-dir", str(ROOT), "--output", str(output / "numerical"), "--numerical-only"]
    result = subprocess.run(command, env=env, cwd=output, text=True, capture_output=True)
    (output / "checker.stdout.txt").write_text(result.stdout)
    (output / "checker.stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError("Numerical check failed; inspect " + str(output))
    after = module.verify(ROOT)
    assert after == before
    receipt = json.loads((output / "numerical/receipt.json").read_text())
    if not receipt["passed"] or receipt["mode"] != "numerical_only":
        raise RuntimeError("Numerical-only checker did not pass")
    if sorted(receipt["omitted_original_bound_files"]) != ["inputs/PROTOCOL.md", "inputs/RESULTS.md"]:
        raise RuntimeError("Unexpected original bound-file omissions")
    summary = {"status": "PASS", "package": before, "numerical_checks": receipt["checks"],
               "scope": "Retained numerical inputs/arithmetic; excludes original full-text and external historical-corpus identity",
               "omitted_source_prose": receipt["omitted_original_bound_files"],
               "new_network_requests": 0, "model_forwards": 0, "fits": 0, "bootstrap_draws": 0,
               "raw_response_auroc_computations": 0}
    with (output / "portable_replay_summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
