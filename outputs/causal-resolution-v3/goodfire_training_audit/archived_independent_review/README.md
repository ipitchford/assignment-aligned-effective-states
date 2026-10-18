# Independent Goodfire exported-value replay

This directory contains a separate standard-library checker and two completed read-only runs. It does not import or execute the reviewed audit's programs. `recompute.py` performs rational arithmetic on the decimal values written in the source JSONs and checks source-to-summary/CSV agreement.

Run from the workspace root with a **fresh output directory**:

```sh
python3 -B work/goodfire_training_independent_review/recompute.py --audit-dir work/goodfire_training_audit --output work/goodfire_training_independent_review/new_full_replay
```

The default mode needs all original bound files and the original local paths named in provenance/corpus receipts. It reads no network and does no model execution, fitting, resampling or raw-response AUROC computation.

For a redistributed component retaining numerical inputs and the original manifest/input freeze:

```sh
python3 -B checks/recompute.py --audit-dir /path/to/goodfire_training_audit --output /path/to/new_output_directory --numerical-only
```

`--output` names a directory, not a JSON file. The checker refuses an existing output directory. It writes `receipt.json` (actual assertion counts, exact identities, failures, omissions and execution boundary) and `recomputed.json` (nine onsets, 120 comparisons, all range exceptions and control identities). Missing required numerical inputs or mismatched original hashes cannot pass.

Completed runs:

- `full_replay/`: 29,543 assertions passed against the original audit. All 27 distinct files bound by the original manifest/input freeze matched, plus the root manifest identity itself. No original files omitted.
- `portable_replay/`: 29,491 assertions passed against the staged redistribution, which omits the original report and protocol prose. This mode also deliberately skips original external acquisition/corpus provenance checks. The scientific output is byte-identical to `full_replay/recomputed.json`.

Counts are software assertions, not independent observations or evidence strength. The full critical review is `work/next_phase/goodfire_training_independent_review.md`. No original study/audit files were changed.
