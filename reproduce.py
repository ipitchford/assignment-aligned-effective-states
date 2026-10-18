"""Replay a released archive in a new workspace, without changing archived evidence.

Run from anywhere with --output NEW_DIRECTORY and the three pinned Python runtimes.
Scientific inputs and scripts come from this archive except three explicitly
hash-pinned external upstream source files. Interpreters and
installed third-party packages are external dependencies, recorded in receipts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ALL17 = Path("work/next_phase/sat_all17_validation_v3_20261018")
FULL = Path("work/next_phase/sat_full_layer_basis_v7_20261018")
RESEARCH = Path("outputs/causal-resolution-v3")
EXPECTED_FREEZES = {
    'SAT_ALL17_FREEZE_SHA256': (ALL17, '77ac6f5f3c41abce5c90e538ad9a6856e7869d872c590d713f3b684eae177b80'),
    'SAT_FULL_LAYER_FREEZE_SHA256': (FULL, '4662155b6d3e4d73a6a68a312bb1d1c402fec2a17aa691fee5882af642d9b8fb'),
}
REPLAY_WORKSPACE = None
RELOCATIONS = []
UNAVAILABLE_STAGES = {
    "inception_saved": "Photographic inputs and native image assets are not redistributed.",
    "construct_validity": "Photographic source and edited-image inputs are not redistributed.",
    "geometry_failure": "Image-derived geometric fixtures are not redistributed.",
    "geometry_bisection": "Image-derived bisection fixtures are not redistributed.",
    "fixed_seam_saved": "Blinded photographic assessor packets are not redistributed.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inventory(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    expected = {row["path"] for row in manifest["files"]} | {"MANIFEST.json", "SHA256SUMS.txt"}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if expected != actual:
        raise RuntimeError("release file inventory differs: " + repr(sorted(expected ^ actual)))
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != row["size"] or sha(path) != row["sha256"]:
            raise RuntimeError("release identity mismatch: " + row["path"])
    return len(manifest["files"])


def compare_json(expected, actual, label="", differences=None):
    if differences is None:
        differences = []
    # Absolute provenance paths relocate; all hashes and scientific values still compare.
    if isinstance(expected, str) and isinstance(actual, str) and expected.startswith("/") and actual.startswith("/"):
        actual_path = Path(actual).resolve()
        if REPLAY_WORKSPACE is None or not actual_path.is_relative_to(REPLAY_WORKSPACE) or not actual_path.is_file():
            differences.append(label + ': regenerated provenance path escapes replay workspace or is absent')
        else:
            relative = actual_path.relative_to(REPLAY_WORKSPACE)
            if not expected.endswith('/' + relative.as_posix()):
                differences.append(label + ': relocated relative path differs')
            else:
                RELOCATIONS.append({'field': label, 'archived': expected, 'regenerated': actual,
                                    'workspace_relative': str(relative), 'referenced_sha256': sha(actual_path)})
    elif isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            differences.append(label + ": key inventory differs")
        for key in sorted(set(expected) & set(actual)):
            if key in {"started_utc", "completed_utc", "elapsed_seconds", "elapsed_s", "runtime_seconds"}:
                continue
            compare_json(expected[key], actual[key], label + "/" + key, differences)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(label + ": list length differs")
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare_json(left, right, label + "/" + str(i), differences)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool) and isinstance(actual, (float, int)):
        if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-11):
            differences.append(f"{label}: {expected!r} != {actual!r}")
    elif expected != actual:
        differences.append(f"{label}: {expected!r} != {actual!r}")
    return differences


def main():
    global REPLAY_WORKSPACE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-python", required=True, help="Python 3.14.7, NumPy 2.4.2")
    parser.add_argument("--scientific-python", required=True, help="Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0")
    parser.add_argument("--native-python", required=True, help="Python 3.11.14, torch 2.1.2, NumPy 1.26.4, einops 0.8.1")
    parser.add_argument("--upstream-directory", type=Path, required=True, help="Externally retrieved exact sources; see setup_external.py. No implicit network access.")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or destination.is_relative_to(ROOT):
        parser.error("output must be a new directory outside the extracted release")
    count = verify_inventory(ROOT)
    destination.mkdir(parents=True)
    sandbox = destination / "workspace"
    REPLAY_WORKSPACE = sandbox
    shutil.copytree(ROOT, sandbox, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    external_inputs = []
    for row in json.loads((ROOT / 'EXTERNAL_DEPENDENCIES.json').read_text())['files']:
        source = args.upstream_directory.resolve() / row['filename']
        if not source.is_file() or source.is_symlink() or source.stat().st_size != row['bytes'] or sha(source) != row['sha256']:
            raise RuntimeError('Missing or mismatched external source: ' + row['filename'])
        target = sandbox / row['target']
        if not target.resolve().is_relative_to(sandbox) or target.exists():
            raise RuntimeError('Unsafe external dependency target')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
            raise RuntimeError('Copied external source identity mismatch')
        external_inputs.append(row)
    # ZIP transports bytes but cannot represent the prescribed Unix-epoch-zero
    # blinding timestamps in its DOS date fields. Reconstruct that declared
    # metadata in the disposable copy, preserving every byte and frozen checker.
    metadata_reconstruction = []
    packet = sandbox / "work/next_phase/fixed_seam_sham/run/assessor_inputs"
    if packet.exists():
        for path in sorted(packet.iterdir()):
            if path.is_file():
                identity = sha(path)
                before = path.stat().st_mtime_ns
                os.utime(path, ns=(0, 0))
                if sha(path) != identity or path.stat().st_mtime_ns != 0:
                    raise RuntimeError("blinding metadata reconstruction failed")
                metadata_reconstruction.append({"path": str(path.relative_to(sandbox)),
                    "sha256": identity, "extracted_mtime_ns": before, "restored_mtime_ns": 0})
    logs = destination / "logs"
    logs.mkdir()
    retained = destination / "regenerated"
    retained.mkdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONOPTIMIZE", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Do not read stale or cloud-evicted bytecode beside installed package sources.
    environment["PYTHONPYCACHEPREFIX"] = str(destination / "python-bytecode")
    environment["MPLCONFIGDIR"] = str(destination / "matplotlib")
    archived_path = json.loads((sandbox / FULL / 'effective_state_followup_v6.json').read_text())['supersedes']['path']
    original_root = str(Path(archived_path).parents[3])
    if not original_root or Path(original_root) in (sandbox, ROOT, Path('/')):
        raise RuntimeError('cannot identify an original workspace for accidental-read guard')
    environment['REPLAY_DENIED_ROOTS'] = json.dumps([original_root])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    for key, (directory, expected) in EXPECTED_FREEZES.items():
        if sha(sandbox / directory / 'freeze.json') != expected:
            raise RuntimeError('original frozen identity changed: ' + key)
        environment[key] = expected
    steps = []

    def run(name, interpreter, script, arguments=(), outputs=(), compare=(), cwd=None):
        if name in UNAVAILABLE_STAGES:
            print(json.dumps({"stage": name, "state": "UNAVAILABLE_NOT_DISTRIBUTED",
                              "reason": UNAVAILABLE_STAGES[name]}), flush=True)
            return
        start = time.monotonic()
        originals = {}
        regenerated = retained / name
        regenerated.mkdir()
        for relative in outputs:
            target = sandbox / relative
            if target.exists():
                originals[relative] = target.read_bytes()
                target.unlink()
        # Preserve a virtual environment's lexical interpreter path, not its symlink target.
        command = [os.path.abspath(interpreter), "-B", str(sandbox / 'guarded_run.py'), str(sandbox / script), *map(str, arguments)]
        print(json.dumps({"stage": name, "state": "running"}), flush=True)
        completed = subprocess.run(command, cwd=cwd or sandbox, env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (logs / (name + ".log")).write_text(completed.stdout)
        comparisons = []
        failure = None
        try:
            if completed.returncode:
                raise RuntimeError(f"{name} exited {completed.returncode}; see logs/{name}.log")
            if '"replay_guard": "ACTIVE"' not in completed.stdout:
                raise RuntimeError('missing child guard activation: ' + name)
            for relative in compare:
                target = sandbox / relative
                if relative not in originals or not target.exists():
                    raise RuntimeError("missing compared output: " + str(relative))
                expected = json.loads(originals[relative])
                actual = json.loads(target.read_bytes())
                start_relocations = len(RELOCATIONS)
                differences = compare_json(expected, actual)
                if differences:
                    (regenerated / "differences.json").write_text(json.dumps(differences, indent=2))
                    raise RuntimeError(f"{name}: {len(differences)} scientific/provenance differences")
                comparisons.append({"file": str(relative), "status": "PASS",
                                    "archived_sha256": hashlib.sha256(originals[relative]).hexdigest(),
                                    "regenerated_sha256": sha(target),
                                    "relocated_fields": RELOCATIONS[start_relocations:]})
                (regenerated / 'comparison_details.json').write_text(json.dumps(comparisons, indent=2) + '\n')
        except Exception as error:
            failure = error
        finally:
            for relative in outputs:
                target = sandbox / relative
                if target.exists():
                    saved = regenerated / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(target), str(saved))
                if relative in originals:
                    target.write_bytes(originals[relative])
        record = {"name": name, "status": "FAIL" if failure else "PASS", "returncode": completed.returncode,
                  "seconds": time.monotonic() - start, "script": str(script), "script_sha256": sha(sandbox / script),
                  "interpreter": os.path.abspath(interpreter), "comparisons": comparisons}
        steps.append(record)
        (destination / "progress.json").write_text(json.dumps(steps, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if failure:
            raise failure

    run("all17_bank", args.analysis_python, ALL17 / "verify_bank.py",
        outputs=[ALL17 / "bank_verification.json"], compare=[ALL17 / "bank_verification.json"])
    run("all7252_native", args.native_python, Path("replay_native.py"),
        arguments=["--root", sandbox, "--output", destination / "native"])
    run("all17_analysis", args.analysis_python, ALL17 / "analyze.py",
        outputs=[ALL17 / x for x in ("analysis.json", "REPORT.md", "bootstrap_macro_auroc.npz")],
        compare=[ALL17 / "analysis.json"])
    run("full_layer_analysis", args.analysis_python, FULL / "analyze.py",
        outputs=[FULL / x for x in ("analysis.json", "attempt_analysis.json")], compare=[FULL / "analysis.json"])
    for version in (2, 3):
        run("all17_posthoc_v" + str(version), args.analysis_python,
            Path("work/next_phase") / f"sat_all17_posthoc_diagnostics_v{version}.py",
            outputs=[ALL17 / f"posthoc_diagnostics_v{version}.json", ALL17 / f"POSTHOC_DIAGNOSTICS_V{version}.md"],
            compare=[ALL17 / f"posthoc_diagnostics_v{version}.json"])
    run("full_layer_posthoc", args.analysis_python, Path("work/next_phase/sat_full_layer_posthoc_v1.py"),
        outputs=[FULL / "posthoc_diagnostics_v2.json"], compare=[FULL / "posthoc_diagnostics_v2.json"])
    for name, script, output in (
        ("live_state", FULL / "effective_live_state_analysis.py", FULL / "effective_live_state_analysis_v2.json"),
        ("readout_accounting", FULL / "effective_state_followup_v3.py", FULL / "effective_state_followup_v4.json"),
        ("corrected_diagnostics", FULL / "effective_state_followup_v6.py", FULL / "effective_state_followup_v6.json"),
    ):
        interpreter = (args.scientific_python if name == "corrected_diagnostics" else
                       args.native_python if name == "readout_accounting" else args.analysis_python)
        run(name, interpreter, script, outputs=[output], compare=[output])
    run("edit_information", args.scientific_python,
        Path("work/next_phase/sat_review_repairs_20261018/sat_information_audit.py"),
        outputs=[Path("work/next_phase/sat_review_repairs_20261018") / n for n in ("sat_information_audit.json", "sat_information_audit.md")],
        compare=[Path("work/next_phase/sat_review_repairs_20261018/sat_information_audit.json")])
    run('collision_sensitivity', args.analysis_python,
        Path('work/next_phase/sat_review_repairs_20261018/report_collision_sensitivity.py'),
        outputs=[Path('work/next_phase/sat_review_repairs_20261018') / n for n in ('report_collision_sensitivity.json', 'report_collision_sensitivity.md')],
        compare=[Path('work/next_phase/sat_review_repairs_20261018/report_collision_sensitivity.json')])
    run('nmf_attribution', args.scientific_python, Path('nmf_revision_diagnostic.py'),
        arguments=['--root', sandbox, '--output', sandbox / 'nmf_revision_diagnostic.json'],
        outputs=[Path('nmf_revision_diagnostic.json')], compare=[Path('nmf_revision_diagnostic.json')])
    run("calibrations", args.scientific_python, RESEARCH / "src/verify_bundle.py",
        arguments=[destination / "calibrations"])
    run("renaming_saved", args.analysis_python, RESEARCH / "sat_renaming_case/replay_saved.py",
        arguments=["--output-dir", destination / "renaming"])
    run("margin_audit", args.analysis_python, RESEARCH / "src/replay_sat_margin_audit.py",
        arguments=[destination / "margin"])
    run("goodfire_exports", args.analysis_python, RESEARCH / "goodfire_training_audit/replay_numerical.py",
        arguments=["--output-dir", destination / "goodfire"])
    run("inception_saved", args.scientific_python, RESEARCH / "inception_case/check_saved_data.py")
    run("construct_validity", args.scientific_python, RESEARCH / "construct_validity/check_supplement.py")
    run("geometry_failure", args.scientific_python, RESEARCH / "deterministic_geometry_failure/replay_saved.py",
        arguments=["--output-dir", destination / "geometry_failure"])
    run("geometry_bisection", args.scientific_python, RESEARCH / "deterministic_geometry_bisection/replay_saved.py",
        arguments=["--output-dir", destination / "geometry_bisection"])
    run("fixed_seam_saved", args.scientific_python, Path("work/next_phase/fixed_seam_sham/check_saved.py"),
        arguments=["--run-dir", sandbox / "work/next_phase/fixed_seam_sham/run", "--output", destination / "fixed_seam_receipt.json"])
    run("negative_score_screen", args.analysis_python, Path("work/next_phase/task_policy_postrun_audit.py"),
        arguments=["--out", destination / "negative_score_screen_receipt.json"])
    after = verify_inventory(ROOT)
    if len(steps) != 18:
        raise RuntimeError("Expected exactly 18 redistributed replay stages")
    record = {"status": "PASS_REDISTRIBUTED_SCOPE", "unavailable_stages": UNAVAILABLE_STAGES,
              "external_source_dependencies": external_inputs,
              "release_manifest_sha256": sha(ROOT / "MANIFEST.json"),
              "files_verified_before": count, "files_verified_after": after, "steps": steps,
              "scope": "clean-extraction replay, full protected-bank checkpoint execution, protected statistics, corrected diagnostics and packaged exact/saved-data calibrations; no retraining or external scientific replication",
              "comparison_policy": "12 JSON-result stages compare recursively, numeric rtol=1e-9 atol=1e-11; regenerated absolute provenance paths must exist inside replay workspace and match the full archived relative suffix; all relocated fields logged; run timestamps excluded. Native stage checks arrays separately at atol=rtol=1e-6 and threshold events/argmax identity, recording bitwise identity too. Five supporting stages are accepted by zero exit plus their internal exact/saved-data checks, not uniform JSON comparison.",
              "isolation_scope": "Copied scientific inputs, top-level stage Python audit hook denying the original workspace with a per-child rejection self-check, explicit path validation, anchored freeze hashes and cleared PYTHONPATH/PYTHONOPTIMIZE; native-library I/O can bypass Python audit events, so this is not an OS filesystem sandbox or a second-host replication. Statistics use archived arrays after fresh complete-bank equivalence has been checked.",
              "not_reexecuted": ["training", "legacy Torch decoder optimizer fits", "early SAT witness and development-bank native forwards", "readiness and matched-edit pilot native forwards", "Inception TensorFlow forwards", "Anthropic assessor calls"],
              "expected_original_freezes": {key: value[1] for key, value in EXPECTED_FREEZES.items()},
              "transport_metadata_reconstruction": metadata_reconstruction,
              "original_extracted_files_modified": 0}
    (destination / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "stages": len(steps), "receipt": str(destination / "receipt.json")}))


if __name__ == "__main__":
    main()
