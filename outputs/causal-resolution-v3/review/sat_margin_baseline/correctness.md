# Focused correctness review: cached continuous-margin comparator

16 October 2026. **Pass within the requested scope: no material arithmetic, row-alignment, feature-provenance or contract-interpretation defect found.** The reported comparison is reproducible from its cached outputs. It remains an opened-development comparison with unequal supervision, not evidence that a semantic bit contains information absent from the continuous score.

I reviewed the original SAT mechanism study earlier but did not implement this comparator. This review is independent of its implementation, not blind to its reported outcomes. I inspected the frozen contract, implementation, exported arrays and receipts. I did not repeat the earlier 19,922-check study audit, import the study code, load a pickle, fit any classifier, evaluate a tree predictor or run a neural checkpoint. The new checker performs cached arithmetic and artifact inspection only.

## Exact inputs and replay

| Item | SHA-256 |
|---|---|
| Baseline `MANIFEST.json` | `3a2c115056d9512d72095f96e8a6766691663122a2c10997496e70e21a31edc5` |
| Baseline pre-fit `freeze.json` | `67ec0b6d5c1ee18d214378d86d9247f1f886df5a1874e72a79fa00648a8bdb75` |
| Baseline `contract.md` | `bc330905a32edccb90d2ab511e5c2bc5714b7e6156d661085a7188ba131efdde` |
| Independent `recompute.py` | `a59fbc0367c6096d735a856044c52190826c13527c3b12ad31d3604c82b7b900` |

`recompute.py` passes **5,407 assertions**. `checks.json`, `unit_metrics_recomputed.csv` and `bootstrap_recomputed.npz` retain the results. All files in the new study manifest, all new frozen local/input hashes and the original cached native-input mapping used here match their bound identities. The recorded freeze precedes recorded execution; these are consistent local receipts, not an external timestamp service or proof of unrecorded history.

## Verified result and aggregation

| Fixed scope | Continuous BA | Readiness BA | Continuous minus readiness |
|---|---:|---:|---:|
| Clipped assessment, primary | 0.8926837071 | 0.9095426567 | −0.0168589496 |
| All selected assessment | 0.8976256723 | 0.9108561166 | −0.0132304444 |
| Clipped fit | 0.9907414641 | 0.9115401119 | +0.0792013522 |
| All selected fit | 0.9689494748 | 0.9118665024 | +0.0570829724 |

The primary cohort is unchanged: **955 clipped unit-origin pairs out of 1,088 selected assessment pairs**, each with repair and neutral actions. All 17 units are retained; units 55 and 416 are averaged within their shared TTTFF assignment before the 16-assignment macro-average. I reproduced all unit confusion counts, secondary probability AUROCs with independent pairwise comparisons, CSV summaries and 272 q/action disagreement strata. Primary unit differences are heterogeneous: five positive, two zero and ten negative. No unit was discarded.

I recomputed the bootstrap directly by weighting each selected action row with its origin multiplicity, rather than using the submitted per-origin confusion arrays. Both actions and all units from an origin receive the same weight. Every per-unit, assignment and macro draw, and all **2,000 paired differences**, agree exactly. The unchanged structured draws agree with the original cache. All 2,000 draws are valid, over the same 685 origin IDs; no draws were replaced. The percentile interval is **[−0.0316021762, −0.0035432272]**.

This interval favors readiness directionally on the opened cohort. Under the predeclared two-point rule it neither establishes continuous-predictor noninferiority (`L < −.02`) nor establishes a loss exceeding two points (`U > −.02`). **Unresolved at that resolution** is correct. Readiness separately meets its development noninferiority condition (`U <= .02`); descriptive two-point parity fails. No significance, population or protected-confirmation claim follows from these conditional draws.

## Features, chronology and leakage

The 2,176 cached pair rows match the original ledger and remain in query order with repair before neutral. Original fit/assessment indices and origin-content hashes are disjoint. The stored deduplicated edited-input mapping also has no overlap across those splits. Duplicate native inputs within the permitted cohort remain repeated query rows, as required.

The frozen code builds only the allowed 92 columns: float32 original score promoted exactly to float64; same-assignment action effect; and 10/40/40 one-hot slot/old/new-clause columns. I independently recovered the action effect from the fixed assignment and old/new clauses. It does not require the rest of the formula or the original unsatisfied-clause count. q is retained for cohort/strata, not passed into the learner. Native post-edit outcomes are fit labels, not predictors. The unchanged semantic forecast is read from its original cache.

The fit mask uses only selected clipped **fit** rows. All recorded class weights equal the specified fit-label formula and are absent from assessment rows. All 17 exported models have 92 features, 200 trees, the frozen parameters and the correct root sample counts. Every stored bin boundary across the 92 features equals the midpoint calculation on the fit features alone; no assessment-derived scaling/binning appears. Model exports use valid feature indices. The recorded single `predict_proba` result per unit is assembled correctly into the global cache, with the positive class identified correctly and strict probability `> .5`; there are no .5 ties.

These checks support the recorded data flow. They do **not** independently reproduce the optimizer or establish that exported trees generate the cached probabilities, because fitting and prediction replay were outside this bounded audit. No source or artifact inconsistency indicates such a mismatch.

## Cost, supervision and scientific scope

The new source and receipts record exactly 17 fits, one assessment and zero neural forwards/new formulas/protected cases. Recorded fitting totals **0.851101 s**, probability prediction **0.028409 s**, and process peak memory **198,705,152 bytes**. I checked the sums, not elapsed time by rerunning the fits. Actual exported ensembles contain **3,400 trees, 16,778 leaves and 13,378 internal split nodes**. Pickles total 2,182,704 bytes; prediction-array ZIP files total 3,773,760 bytes, versus 1,711,070 raw array bytes. The larger ZIP total is not a missing-data error: many small archived arrays incur container overhead. Manifested files total 10,601,374 bytes, excluding the manifest and live transcript as declared.

The reported **17 versus 544 state bits** and **67 versus 594 bits including the retained report and fixed action** are correct for these nominated states. Shared native extraction, decoder arrays, thresholds, assignment/clause lookup and action representation are charged separately. This is neither a capacity-matched comparison nor global minimality for semantic SAT.

Readiness used **45,693 unit-origin q labels**, versus **1,922 unit-action native labels** for this comparator, only 98–128 edited fit rows per unit. These are different supervision types and repeated unit/observation counts, not independent formula counts. The much higher continuous fit accuracy and lower assessment accuracy are consistent with finite-sample overfitting, without identifying its exact cause.

The strongest alternative remains straightforward: the readiness rule is a deterministic threshold of the original continuous score, followed by the supplied action gate. A sufficiently informed function of the continuous score and that action can therefore reproduce it exactly. This fitted learner's shortfall cannot refute unrestricted continuous-margin sufficiency or establish additional semantic information. At most, the result supports a useful accuracy/compression tradeoff for this frozen rule and cohort, with its static prior and supervision costs disclosed. The source count/threshold mechanism remains compatible with the findings; this comparison adds no context-invariance or unique-mechanism test.

## Material issues and editorial correction

**No material correction required for the reported numerical result or contract category.** Preserve the comparator even though its point result favors readiness, and stop under the stated outcome-neutral rule.

**Minor wording:** the report's generic sentence “unresolved at the declared resolution or lacks required support” leaves an unnecessary ambiguity. Here support is available and every draw is valid. Replace it in any reader-facing account with: “The comparison is unresolved at the two-point development resolution; all required outcome and bootstrap support was available.” Keep the frozen report intact and record any wording correction separately.

No broader generality is assigned. This audit verifies the bounded new comparison, not its field-level significance or a protected replication.
