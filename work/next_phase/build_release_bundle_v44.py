#!/usr/bin/env python3
"""Build the curated v4.4 paper and evidence release without mutating source evidence."""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
SOURCE = OUTPUTS / "causal-resolution-v3"
STAGE = OUTPUTS / "already-clean-effective-state-evidence-v4.4-20261018"
ZIP = OUTPUTS / "already-clean-effective-state-evidence-v4.4-20261018.zip"
PAPER = OUTPUTS / "already-clean-effective-state-paper-v4.4.pdf"

EXCLUDED_PARTS = {
    "__pycache__", ".DS_Store", "review", "review-round1", "internal_review",
    "archived_independent_review", "archived_independent_audit",
}
EXCLUDED_PREFIXES = {
    "next_phase",
    "provenance/pdf-build-v32",
    "provenance/pdf-build-v33-reviewed",
}
EXCLUDED_FILES = {"manuscript 2.pdf"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def included(relative: Path) -> bool:
    if relative.name in EXCLUDED_FILES or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    value = relative.as_posix()
    return not any(value == prefix or value.startswith(prefix + "/") for prefix in EXCLUDED_PREFIXES)


def copy_tree_filtered(source: Path, destination: Path) -> list[str]:
    omissions = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not included(relative):
            if path.is_file():
                omissions.append(relative.as_posix())
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return omissions


def add_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination)


def manifest(stage: Path) -> list[dict]:
    rows = []
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        rows.append({
            "path": path.relative_to(stage).as_posix(),
            "size": path.stat().st_size,
            "sha256": digest(path),
        })
    return rows


def main() -> int:
    for target in (STAGE, ZIP, PAPER, ZIP.with_suffix(ZIP.suffix + ".sha256")):
        if target.exists():
            raise RuntimeError(f"exclusive release target already exists: {target}")

    STAGE.mkdir(parents=True)
    omissions = copy_tree_filtered(SOURCE, STAGE / "research")

    central = STAGE / "central_studies"
    central.mkdir()
    add_tree(ROOT / "work/next_phase/sat_all17_validation_v3_20261018", central / "sat_all17_validation_v3_20261018")
    add_tree(ROOT / "work/next_phase/sat_full_layer_basis_v7_20261018", central / "sat_full_layer_basis_v7_20261018")
    for script in (
        "sat_all17_posthoc_diagnostics_v2.py",
        "sat_all17_posthoc_diagnostics_v3.py",
        "sat_full_layer_posthoc_v1.py",
        "make_sat_all17_figure_v3.py",
        "make_sat_full_layer_figure.py",
        "make_sat_sparse_report_figure.py",
        "revise_manuscript_v44.py",
    ):
        shutil.copy2(ROOT / "work/next_phase" / script, central / script)

    reviews = STAGE / "model_generated_reviews"
    reviews.mkdir()
    for name in (
        "sat_full_layer_basis_v7_static_review_opus55_20261018",
        "manuscript_v42_review_opus55_20261018_v2",
        "manuscript_v43_review_opus55_20261018",
    ):
        source = ROOT / "work/next_phase" / name
        if source.exists():
            add_tree(source, reviews / name)

    shutil.copy2(SOURCE / "paper/manuscript.pdf", PAPER)
    shutil.copy2(PAPER, STAGE / "paper.pdf")
    shutil.copy2(SOURCE / "paper/manuscript.md", STAGE / "paper.md")
    shutil.copy2(SOURCE / "paper/abstracts.md", STAGE / "abstracts.md")

    readme = """# Already Clean causal-resolution evidence release v4.4

This archive accompanies **A Sparse Native Assignment Report Spans the Effective Final-MLP State of a 2-SAT Transformer** (18 October 2026, working draft).

The central result concerns one epoch-900 five-variable 2-SAT checkpoint and one engineered 7,252-formula bank. The source-nominated 34 final-MLP units contain effectively all post-ReLU energy and span the observed rank-34 state. Native assignment axes exceed dimension-matched rotations and nonnegative mixtures, while state-level probes recover three assignments that lack equally clean single native axes. The exact final readout is additive; grouped ablations cross redundant argmax margins, and the source's Boolean thresholds discard useful magnitude.

## Start here

- `paper.pdf` and `paper.md`: the reader-facing manuscript.
- `central_studies/sat_all17_validation_v3_20261018/`: protected label study.
- `central_studies/sat_full_layer_basis_v7_20261018/`: protected full-layer study and retrospective effective-state analysis.
- `research/sat_case/`: source checkpoint, notebook and upstream code.
- `research/provenance/pdf-build/receipt.json`: exact paper-build receipt.
- `MANIFEST.json` and `SHA256SUMS.txt`: hashes for every included file.

## Evidence boundaries

Hash and replay records establish artifact identity and stated computational consistency. They are not independent scientific replication, formal verification, or human peer review. The Anthropic records in `model_generated_reviews/` are model-generated criticism and provenance. The archive retains negative studies and failed construction routes because they delimit the claims.

The paper does not estimate neuron cleanliness prevalence, identify developmental origin, demonstrate capacity-driven superposition, or recover the upstream causal route. The decisive next study is an independently nominated seed-by-width sweep with natural-distribution evaluation and upstream interventions.
"""
    (STAGE / "README.md").write_text(readme)
    omission_text = """# Omission ledger

The release excludes historical manuscript copies, caches, prior PDF-build directories, internal editorial material, archived assessment folders, and the workspace planning directory. These exclusions reduce duplication and keep private process records out of the reader-facing scientific archive. Source evidence is unchanged in the workspace.

Excluded source-relative files:
""" + "".join(f"- `{path}`\n" for path in omissions)
    (STAGE / "OMISSIONS.md").write_text(omission_text)

    rows = manifest(STAGE)
    manifest_doc = {
        "release": "v4.4-20261018",
        "paper_sha256": digest(PAPER),
        "file_count_before_manifest": len(rows),
        "total_bytes_before_manifest": sum(row["size"] for row in rows),
        "files": rows,
    }
    (STAGE / "MANIFEST.json").write_text(json.dumps(manifest_doc, indent=2, sort_keys=True) + "\n")
    final_rows = manifest(STAGE)
    (STAGE / "SHA256SUMS.txt").write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in final_rows))

    with zipfile.ZipFile(ZIP, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(p for p in STAGE.rglob("*") if p.is_file()):
            archive.write(path, Path(STAGE.name) / path.relative_to(STAGE))
    ZIP.with_suffix(ZIP.suffix + ".sha256").write_text(f"{digest(ZIP)}  {ZIP.name}\n")
    print(json.dumps({
        "status": "PASS",
        "paper": str(PAPER),
        "paper_sha256": digest(PAPER),
        "stage": str(STAGE),
        "zip": str(ZIP),
        "zip_sha256": digest(ZIP),
        "zip_bytes": ZIP.stat().st_size,
        "included_files": len(final_rows),
        "omitted_files": len(omissions),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
