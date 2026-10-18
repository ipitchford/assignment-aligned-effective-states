# SAT native transport: one cached-development margin challenge

15 October 2026. **Design contract; no baseline has been fitted or evaluated by this assessment.** This is a new, bounded analysis of already opened development data. Its author has seen the original aggregate outcomes and participated in design/governance; it is neither blind review nor protected confirmation. The original experiment, its negative magnitude pilot, frozen implementation and receipts remain unchanged. Save this contract and the complete new implementation with hashes before fitting. Use a new output directory, `work/sat_margin_baseline/`.

**Admission: one generic continuous-score predictor per source unit, fitted once to cached native edited-fit outcomes, followed by one fixed assessment. No native inference, new formulas, protected cases, model retraining, new units or context permutations are admitted by this step.** Stop after the report regardless of outcome.

## 1. Scientific question and prior expectation

The completed development assay reports assignment-weighted native balanced accuracy 0.909543 for `readiness AND repair`, with a 0.809763 difference between the q=1 and q>=2 repair-minus-neutral effects. These are useful finite findings. They do not eliminate ordinary margin persistence: units near their activation boundary can cross after a favourable edit, while more negative units do not. The source's approximate weighted-clause-count/threshold rationale already anticipates much of this behaviour. The post-score, wrong-unit, wrong-assignment and action-only comparisons do not provide the missing own-continuous-score comparator.

The narrow question is: **on the exact cached cohort, does the semantic readiness bit retain prospective native-response accuracy relative to a flexible predictor using the original continuous score and the identical action information?** An accurate generic predictor would establish a simpler predictive account of the existing finding. Competitive one-bit performance could establish a useful compression tradeoff on this cohort. Neither outcome identifies a unique latent mechanism.

The semantic score and the generic margin are not mutually exclusive. A tree on `s_i` and `m_a` can represent the existing threshold-and-AND rule. Giving the baseline only raw clause IDs while withholding `m_a` would manufacture a semantic-information advantage, so **the true same-assignment `m_a` is explicitly supplied to both methods**. This comparison concerns how the original score is used, not whether action semantics are available.

Palumbo et al.'s Appendix D.2 provides the closest count/threshold motif; the native singleton mapping is an approximate source hypothesis. Broader negative-preactivation and future-activation ideas have substantial prior art. This step cannot support priority for either idea. See the retained source and prior-art audits, [Palumbo et al.](https://proceedings.mlr.press/v267/palumbo25a.html) and [versioned appendices](https://arxiv.org/html/2407.13594v2).

## 2. Exact data, cohort and features

Read only the pinned `work/sat_mechanism/` receipts and the existing source/grammar metadata needed to verify their identities. `stage_b_raw.npz` already contains original scores, native outcomes, split, source unit, q, clipping, true action effect, existing predictions and action contexts. `selected_queries.json` supplies the unchanged origin/action ledger. No model loading or neural forward is necessary.

Use the existing 17 units in their frozen order, representing 16 assignments; retain both TTTFF units 55 and 416. The source bank is conditional on formulas initially being SAT, and selection balances q classes before clipping; the resulting estimand is this assay, not the source paper's balanced test distribution or all formulas. Fit each unit only on its selected **fit** origin/action rows satisfying the original primary condition `q>=1 and original_pre<0`. Every origin has its existing repair and neutral row. Retain duplicated query rows even if their native input was deduplicated for inference. Preserve row order: ascending saved query ID, repair before neutral. Never select a cohort using native outcomes, prediction errors or model confidence.

The 92 predictors, in this exact column order, are:

1. the target unit's original preactivation `s_i`, read from stored float32 and converted exactly to float64;
2. the same target assignment's declared action effect `m_a` (1 for the saved repair, 0 for its neutral partner);
3. the original 90-dimensional one-hot context: slot columns 0–9, old-clause columns 0–39, new-clause columns 0–39, in the same order as the original context probe.

Use **identity scaling**: no centring, normalization, clipping, score reversal, per-assessment transform or rank transform. Trees split numerical thresholds and do not require scale normalization. Assert feature finiteness and all one-hot/range identities. A malformed row is a data-integrity failure, not a row to impute or discard.

The target is the saved same-unit native event `y=1[post_after>=0.5]`, including all native deviations from the symbolic oracle. Do not provide q, q class, other-unit scores, other clauses, original logits, post-edit states or semantic SAT labels as predictors. q is used only for the already fixed cohort and declared reporting strata.

## 3. One fixed learner and fit rule

Use the already pinned analysis environment: NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0. Archive Python and package versions, implementation source, thread settings and `get_params()` before fitting. Use one CPU thread through `threadpoolctl.threadpool_limits(1)`, without changing the runtime of other research jobs. No package substitution or upgrade is part of this contract.

For each unit, use exactly:

```python
HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=7,
    max_depth=3,
    min_samples_leaf=5,
    l2_regularization=1.0,
    max_bins=255,
    categorical_features=None,
    early_stopping=False,
    random_state=2026101522,
)
```

All other parameters are the pinned version's defaults, saved verbatim in the freeze. Do not introduce monotonic or interaction constraints. There is no validation set, hyperparameter search, early stopping or assessment-dependent fit. This capacity can represent a score threshold conditional on action effect and permits additional nonlinear action-context effects; it is deliberately more flexible than the one-bit rule.

With n clipped fit action rows and n_c examples of outcome c, pass sample weight `n/(2*n_c)` to each row of class c. These weights depend only on this unit's fit outcomes. Do not additionally use a class-weight option. If there are no fit rows, retain an unestimable predictor with NA predictions. If only one fit class occurs, retain its constant prediction and an explicit one-class warning. Do not borrow rows, pool units or change the rule.

Save each fitted estimator, its complete prediction parameters, warnings and fit duration. Evaluate the positive-class probability with `predict_proba`; predict 1 iff that float64 probability is **strictly greater than 0.5**. An exact 0.5 tie predicts 0. Do not calibrate a second threshold on fit or assessment data. Library tree-split ties use the frozen implementation, feature/row order and seed; no manual tie-breaking retry is permitted. Store both probabilities and binary predictions in saved row order. A fit exception or nonfinite output remains a failure for that unit; no changed settings or second fit follows it.

This is a predictive comparator trained on edited-fit outcomes. The readiness rule instead learned its q=1 threshold from all eligible original fit formulas before seeing edited outcomes. They have different supervision and sample counts. Print those counts and label types side by side. A structured advantage could reflect this prior/supervision advantage or finite-sample fitting; it would not by itself refute continuous-margin sufficiency.

## 4. One assessment and its interpretation

The primary assessment uses exactly the original selected clipped assessment rows. Also report unchanged all-selected assessment rows and the corresponding fit summaries. For every source unit retain raw confusion counts, sensitivity, specificity, balanced accuracy and outcome support. Report probability AUROC as secondary, plus the descriptive prediction disagreement/error table against the **unchanged** structured forecast, split by q=1/q>=2 and repair/neutral. Do not tune to those tables or mine individual errors into a new rule.

Average the two TTTFF units first, then macro-average the 16 assignments. The primary difference is `Delta = BA_continuous - BA_structured`. Reuse the **same 2,000 saved origin multiplicity vectors** and origin-ID mapping in `bootstrap_raw.npz`; carry each origin's units and both actions together. Do not draw a new bootstrap or refit within draws. Archive all per-draw differences. Apply the original NA propagation: a missing required native class gives NA, one NA constituent makes its assignment/macro NA, and an interval requires at least 1,900 of the fixed 2,000 valid draws. Retain every invalid draw; never replenish. Report the original support and clipping denominators, not a new denominator of correctly predicted cases. These intervals are conditional on the fitted predictors and opened cohort; they omit fit-sample, selection and hypothesis-development uncertainty.

Use a **two-percentage-point development resolution**, chosen before this baseline fit, to describe outcomes. It is an operational tolerance, not a significance level or publication threshold. With [L,U] the descriptive 2.5–97.5 percentile interval for Delta:

| Observed comparison | Permissible conclusion and falsification boundary |
|---|---|
| L > 0.02 | The one-bit forecast loses material native predictive accuracy relative to this fitted comparator on this cohort. This contradicts a claim of two-point predictive sufficiency here; it does not refute the approximate assignment interpretation. |
| L >= −0.02, without the preceding condition | This comparator meets the declared development noninferiority criterion. Generic continuous-margin prediction remains a viable account. This condition alone does not establish that the readiness bit is equally accurate. |
| U < −0.02 | This particular trained comparator misses the declared parity target. Its limited edited-fit sample, capacity/optimization and asymmetric supervision remain alternatives. This does not falsify the unrestricted generic-margin account. |
| All other cases, or unavailable interval/point estimate | The comparison is unresolved at the declared resolution or lacks support. Report the point estimate and exact limitation; do not add cases or tune. |

The readiness rule separately meets its two-point development noninferiority criterion only when U <= 0.02; an interval wholly inside [−0.02,0.02] supports descriptive parity at this resolution. These are development descriptions after previous aggregate outcomes were known. A good fit score plus poor assessment score is consistent with overfitting; poor fit and assessment flag inadequate fit/capacity. Neither permits trying another decoder under this contract. Probability AUROC, single-unit successes and post-hoc strata cannot replace the primary comparison. The baseline's frozen result must be reported even if it removes the apparent need for a more elaborate account.

## 5. Full cost and information ledger

Record prediction-state, static decoder, supervision and computation costs separately. Both forecasts require the same original native checkpoint computation; a readiness bit is not free to obtain. `m_a` is computed from the same supplied action and fixed assignment lookup, not obtained from a hidden full-formula oracle.

| Cost | Structured readiness | Continuous comparator |
|---|---|---|
| Fixed known-unit state report | 1 readiness bit | 32 bits for the stored preactivation |
| Action descriptor | Same 16 fixed bits, or 14 packed syntactic bits | Same descriptor; its one-hot expansion adds computation, not independent information |
| All 17 nominated states | 17 bits | 544 bits |
| With the original 34-bit report retained | 51 state bits; 67 with fixed action | 578 state bits; 594 with fixed action |
| Fixed decoder information | Up to 17 float32 thresholds (544 numerical bits), rule tags, source unit/assignment table | Actual serialized prediction parameters, initial scores, trees, thresholds, leaf values, binning metadata and any class constants |
| Per known-unit prediction | Stored comparison bit, clause lookup and AND; extraction includes the original score comparison | At most 200 trees, each depth at most 3: up to 600 split comparisons plus aggregation, sigmoid and final comparison; also action encoding/lookup |

The known target unit costs no per-query index; variable-target queries would add the same five-bit index to both. The shared 17-unit/assignment table costs 238 fixed bits (9 unit-ID and 5 assignment bits per pair); the retained implementation's full clause-truth lookup costs 1,280 static bits. Report actual artifacts as well as these logical sizes. The 17 tree ensembles contain at most 3,400 trees, 23,800 leaves and 20,400 internal split nodes; print actual counts and serialized bytes, including arrays needed for prediction. Do not call this capacity matched or hide static parameters by charging only state bits.

Charge original-fit q labels used to learn readiness, native edited-fit labels used to learn the comparator, row counts, elapsed fitting/prediction time, peak memory and archive bytes. This step uses **zero neural forwards** and at most 17 classifier fits, one per source unit; missing/constant fits do not release a budget for another model. There is no whole-layer feature bank.

For perspective, the full ordered formula costs 60 fixed or 54 packed bits, and its semantic multiset costs 33 packed bits plus unranking/SAT computation. The multiset can answer semantic edit questions with valid old-clause context but does not preserve sequence-sensitive native responses. Consequently neither report is globally minimal for semantic SAT, and this local comparison does not repair the old collision, decode all 34 published units or certify global UNSAT. Using 1,088 bits for an unnecessary 34-float comparator would exaggerate the actual 17-score cost.

## 6. Outcome-neutral stop and separately admitted future question

After the single fit/assessment, save the frozen inputs/code, all predictions, warnings, unit/macro tables, paired draws, cost ledger and a report; verify the original manifests are unchanged, then **stop**. A favourable, unfavourable, mixed or unestimable result does not authorize another learner or protected data. Preserve the original 0.909543 finding as development accuracy on its stated cohort, never as established context invariance.

A possible later question is whether the same readiness rule transports across semantic-preserving clause-context changes: for example, permuting the other nine clause slots while keeping the edited slot and its old/new clauses fixed. That would preserve the exact assignment counts and repair effect while changing native sequential context. No such cases are generated, inferred, selected or admitted here.

Before that question can merit protected data, a separate contract must specify credible **different frozen predictions** of a semantic-invariance account and a context-sensitive margin account, native-grammar support, score-blind selection, practical error bounds, costs and null/hybrid interpretations. Both accounts may predict invariance, and a weighted-count implementation can itself be order invariant. Merely calling a permutation test “semantic” does not create an identifying contrast. An exact invariant claim is falsified by reproducible deviations beyond its declared tolerance; an approximate source interpretation is not falsified by any one exception. The later study would assess a prospective abstraction, not demonstrate that an old negative value currently causes the model's output through ReLU.

If this cached comparison leaves no consequential forecast disagreement beyond a familiar weighted-margin mechanism, retain the SAT study as a rigorous bounded case and stop this avenue. Repeating its accuracy on more IID formulas would improve confidence, but would not create the missing field-level contribution. No outcome of this contract or a later confirmation guarantees a broader generality.

## Input identities and assessment limits

| Retained input | SHA-256 |
|---|---|
| `work/sat_mechanism/MANIFEST.json` | `6e5a46e1466339ccc08f5ce8b91a5cf5381bf3c99f789e3aa40d145b7f42b181` |
| `work/sat_mechanism/contract.md` | `9f25cb332a06c10dacb1315c8b5f7b7f7bbb9ef8a082c3cd2802f219902d3176` |
| `work/sat_mechanism/REPORT.md` | `57a119b1272de6854acd7cf2bd6f4edf66b4096e829df4a98e161542a660a59d` |
| `work/sat_mechanism/stage_b_raw.npz` | `2167186a9824c51141ed63810a9753dcb563f968de6d0e433b36af876034c7c2` |
| `work/sat_mechanism/bootstrap_raw.npz` | `9a209a05ffe419b2cde0edf982a94b54e825b69fab61db69050d80f82254a88c` |
| `work/sat_mechanism/selected_queries.json` | `afd5e553254698bfb89d79f97bc40c21b56692c92b5f683252cb6a1974bea255` |
| `work/next_phase/sat_native_mechanism_sources.md` | `5a0e36cc0c26dfc3da45218050f7ceedf59e3ff777aed1fc8f2676f2a98ba72f` |
| `work/next_phase/negative_preactivation_prior_art.md` | `c63f0f495dce85f637b7161edd88c1038193f1fbedfaedc7afec2676ddb4aeed` |

I read the reported outcomes, implementation notes, analysis/protocol source, source audits and the completed `work/sat_mechanism_review/correctness.md`; hashes above were read directly. I did not load outcome arrays for new analysis, fit a predictor or run a model. The independent review reports 19,922 passing assertions and the remaining margin-persistence objection; I did not repeat its audit. This document makes no new empirical result or manuscript quality assessment.
