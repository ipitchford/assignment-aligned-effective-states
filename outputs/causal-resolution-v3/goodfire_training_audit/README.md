# Goodfire public training exports: portable arithmetic audit

This is a retrospective audit of the three source-declared fresh runs. It
recalculates onset rules and signed candidate-versus-control comparisons from
the published numerical exports. It performs no training, model inference,
image analysis, new statistical fit, bootstrap or re-estimation of AUROC.

Read `findings.md` for the analysis and `audit_scope.md` for its frozen field
specification. The source's reported onset and final numerical table values were
reproduced. Across all three runs, the final response-change and AUROC-decrease
values were inside the respective 16-control ranges. Earlier exceptions and
other-field exceptions are retained in the tables. These are descriptive ranks
within source-selected controls, not randomization significance or proof of
causal inheritance. Source declarations about training/checkpoints are not
independently reproduced here; complete checkpoints and per-image probe arrays
were not established as public inputs.

## Safe numerical replay

With standard-library Python, run from any working directory:

```sh
python3 -B /path/to/goodfire_training_audit/replay_numerical.py --output-dir /path/to/NEW-replay-output
```

The helper requires a new output directory outside this supplement. It verifies
the local package manifest and invokes the independent Fraction-based checker
in **numerical-only** mode. It does not execute the archival acquisition or
analysis scripts and makes no network requests. Every retained numerical input
is checked against the unchanged original manifests before arithmetic is
recomputed. Existing source decimals, signs, ties, ordering, absent/undefined
cells and fixed denominators remain authoritative.

## Included and omitted material

The package contains the nine measurement JSON exports (`metrics`, `lineage`
and `causal` for each run), the publisher's numerical manifest, all three CSVs,
`audit.json`, our code, findings, provenance records and original receipts.
`SOURCE_COPY_MAP.json` records exact copied paths, byte lengths and SHA-256
identities. `PACKAGE_MANIFEST.json` binds the redistributed file inventory.

The complete publisher-authored `inputs/RESULTS.md` and `inputs/PROTOCOL.md` are
deliberately **not redistributed**. Their exact URLs, byte lengths and hashes are
in `SOURCE_OMISSIONS.json`; `SOURCE_RECOVERY.md` explains optional recovery into a
separate working copy. The original `manifest.json` and `input_freeze.json` remain
unchanged and still list these missing files. They document the historical full
audit, not a complete portable inventory. Until exact prose bytes are recovered,
the portable replay excludes their identity and direct full-text verification.
The source factual table entries preserved in `audit.json` can be compared with
the numerical exports but do not replace independent reading of the source text.
It also excludes full-mode checks against the original workspace's earlier
seven-file corpus and acquisition-copy paths. Their historical identities and
the full review's receipt are retained; those external source files are not
silently inferred or fetched. See `SOURCE_RECOVERY.md` for the precise boundary.

`acquire_inputs.py` and `audit_exports.py` are archival. The first performs
retrieval; the second expects the omitted prose. Neither is invoked by the safe
command. `verify_arithmetic.py` is the original same-agent rational replay; the
default uses the separately authored independent checker in `checks/`.

Primary links: [training manifest](https://already-clean.goodfire.com/assets/training-map/manifest.json),
[source results](https://already-clean.goodfire.com/assets/early-training/RESULTS.md),
[source protocol](https://already-clean.goodfire.com/assets/early-training/PROTOCOL.md).
Individual exact JSON URLs are recorded in `input_provenance.json`. This is a
numerical/provenance supplement, not a neural reproduction or permission to
redistribute the omitted full text.
