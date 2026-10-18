# Response-blind source-grammar audit

This public derivative retains the exact new bank and native arrays. It is separate from the original protected 7,252-formula experiment. The 4,096 formulas are balanced, history-excluded distinct-clause draws, not literal IID draws at natural prevalence or proven unseen training inputs. The unchanged checkpoint achieved 4,087 correct decisions. Assignment-macro AUROC is 0.9974772794362474; threshold deficiencies remain. No new checkpoint or fitted decoder is supplied.

The original pre-inference evaluation freeze SHA-256 was `9ee46c4c3392cc9d2eb4520b0c23bae43ffbe838c9ce0f858555a066225bf434`. Portable adapters below are post-observation derivatives: they do not pretend to be the original frozen evaluator. Scientific labels, threshold, bank, arrays and comparisons are unchanged. The original freeze is retained privately; its digest is a provenance commitment, not public full-history regeneration evidence.

## Recount saved arrays

Use Python 3.11 with NumPy 2.4.6 and SciPy 1.17.1. Each output must be a new filename.

```
python -B code/check_saved.py --out recount-normal.json
python -B -O code/check_saved.py --out recount-optimized.json
```

Both were executed successfully on this derivative. They independently evaluate Boolean semantics, rank-sum AUROC and all 85 threshold cells; they are not new transformer forwards.

## Fresh native replay

Use Python 3.11.14, NumPy 1.26.4, Torch 2.1.2 and einops 0.8.1. Set BLAS thread counts to one; the loader uses two native threads. `RELEASE` is the containing public research package with its pinned Palumbo runtime and checkpoint. Output directory must not exist.

```
python -B code/evaluate_native.py --release-root RELEASE --out NEW_OUTPUT
python -B code/check_saved.py --arrays NEW_OUTPUT/native-outputs.npz --out new-recount.json
```

The adapter loads `work/sat_case/runtime.py`; the containing release manifest must authenticate this dependency. The portable adapter was executed against the resident public-package dependency tree and passed all 4,096 rows plus the 256-row repeated batch. Its compressed native array file is byte-identical to the original observed file (SHA-256 `ed534712e0451cc76b0e9f0bdf24ae4ec39949495fc1cbf00fdee7501f3645dd`). `observed/adapter-native-receipt.json` and `observed/adapter-replay-recount.json` record that run and its independent recount. This is a separate extension replay, not a claim that the final combined public archive has already passed its final package gate.

## Original-bank decoder agreement

Use the existing public full replay's regenerated v6 diagnostic and native arrays:

```
python -B code/check_agreement.py --release-root RELEASE --diagnostic V6_JSON --native-arrays ORIGINAL_BANK_NATIVE_NPZ --out agreement.json
```

The complete binary truth-error sets reconstruct predictions without fitting. Standardised bits/continuous predictions agree with native argmax on 7,205/7,252 and 7,221/7,252 cases; unscaled continuous agreement is 7,224/7,252. This is retrospective agreement, not new-bank deployment or logit reconstruction. Coefficients are not recovered.

The source-grammar bank excluded 365,753 clause multisets in 51 declared historical arrays. An initial v1 bank was rejected before any forward pass after additional historical overlaps were found. The retained v2 bank used the unchanged seed 2026101801 and balanced quotas. Full historical regeneration depends on historical inputs not all distributed here; the exact retained bank is sufficient for the stated native replay and semantic recount.
