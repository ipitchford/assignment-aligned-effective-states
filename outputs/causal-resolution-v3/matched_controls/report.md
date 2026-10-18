# Post-test exploratory matched minimax controls

**Conclusion:** Contrast-guided selection does not consistently outperform matched native12 or direct greedy minimax selection. Its value here remains diagnostic certification rather than an empirically established superior repair rule. Every evaluated model still fails the primary worst-case tolerance of one.

This analysis was commissioned after the original test outcomes and independent empirical review were known. Selection in this implementation uses discovery responses only, and its final decoders were frozen before this implementation evaluated the existing test banks. Those procedural steps do not restore pristine validation: these are exploratory comparisons on previously opened data. Original contracts, freezes, model files and review manuscript were unchanged.

## Matched methods

All four methods use an intercept, ten independent affine minimax output fits, the original uncompressed LP implementation for their final fit, and nearest-integer coefficient export over 2^32. Each 12-predicate method transmits 12 bits per state/24 per pair, uses 12 native threshold predicates and 130 dyadic decoder coefficients including the intercept. Native12 retains the originally selected median predicates. Contrast12 retains the original contrast-selected quartile predicates. Direct greedy minimax starts from native8 and, for each of four additions, evaluates **every remaining quartile candidate on all ten outputs**, selecting the lowest maximum objective with the declared numeric tie rule. No shortlist, candidate pruning or test-driven choice was used.

For candidate search only, identical feature rows are collapsed to per-output minimum/maximum responses. The inequalities are exactly equivalent: a shared prediction must lie within t of both extrema and therefore every response in the group. Full-row and compressed objectives were checked at the initial design for each seed. The final selected sets were fitted and certified with the original full-row implementation. Greedy choices are numerical, not exact proofs of stepwise or global feature optimality.

## Discovery worst-case error

The table displays the exact primal upper bound rounded to four decimals. Every output has a retained rational lower certificate and dyadic primal upper certificate.

| Seed | Native 8 | Native 12 | Contrast 12 | Greedy minimax 12 |
|---:|---:|---:|---:|---:|
| 17 | 8.2135 | 6.0309 | 6.3390 | 5.7873 |
| 29 | 6.7024 | 6.6413 | 5.9212 | 5.9212 |
| 43 | 9.9300 | 7.5511 | 6.6040 | 6.3408 |

Direct greedy minimax has lower discovery maximum than contrast selection for seeds 17 and 43 and ties seed 29. This does not imply that its four-step result is globally optimal over all 12-predicate sets.

## Existing held-out banks: maximum error

| Bank | Seed | Native 8 | Native 12 | Contrast 12 | Greedy minimax 12 |
|:--|---:|---:|---:|---:|---:|
| seen_strength | 17 | 8.9245 | 7.3666 | 7.5635 | 7.4612 |
| seen_strength | 29 | 7.4114 | 7.6636 | 7.0023 | 7.1797 |
| seen_strength | 43 | 9.9820 | 9.2965 | 9.2628 | 8.2437 |
| unseen_strength | 17 | 10.4075 | 8.0728 | 8.3834 | 7.6478 |
| unseen_strength | 29 | 8.2440 | 8.0044 | 7.8349 | 7.8349 |
| unseen_strength | 43 | 8.3717 | 7.7431 | 8.5218 | 8.6144 |
| natural_pairs | 17 | 30.4376 | 25.9464 | 29.6122 | 28.8126 |
| natural_pairs | 29 | 34.8572 | 28.8115 | 29.1693 | 29.1693 |
| natural_pairs | 43 | 32.4168 | 30.6161 | 27.0557 | 28.1214 |

On unseen strengths, contrast selection beats native12 on maximum error in one seed and loses in two. Against direct greedy minimax it wins once, loses once and ties once. It does not dominate on either same-strength or natural-pair transfer. All methods remain far above tolerance one.

## Existing held-out banks: mean squared error

Values are average squared errors summed across all ten outputs; these are descriptive finite-bank quantities, not independent-replication estimates.

| Bank | Seed | Native 8 | Native 12 | Contrast 12 | Greedy minimax 12 |
|:--|---:|---:|---:|---:|---:|
| seen_strength | 17 | 44.208 | 36.699 | 37.857 | 36.747 |
| seen_strength | 29 | 43.281 | 42.464 | 40.754 | 41.576 |
| seen_strength | 43 | 53.837 | 43.907 | 41.030 | 42.659 |
| unseen_strength | 17 | 30.025 | 26.196 | 28.285 | 29.046 |
| unseen_strength | 29 | 29.449 | 30.573 | 31.261 | 31.941 |
| unseen_strength | 43 | 38.261 | 33.178 | 30.006 | 33.217 |
| natural_pairs | 17 | 624.409 | 503.104 | 533.781 | 485.982 |
| natural_pairs | 29 | 606.948 | 485.994 | 603.450 | 599.434 |
| natural_pairs | 43 | 652.435 | 538.119 | 552.934 | 534.894 |

Contrast selection has lower unseen-strength MSE than direct greedy minimax in all three seeds, despite its mixed maximum-error comparison; maximum and average objectives differ. Compared with native12, contrast MSE is lower only in seed 43 on unseen strengths. In seed 29, every minimax method has unseen-strength MSE exceeding a zero-response predictor, illustrating that an optimum for discovery maximum error need not optimise held-out average error. The complete CSV includes the normalised MSE.

## Execution, exact certificates and limits

| Seed | Candidate coordinate LPs | Four-step selection seconds | Search, final fits and certificates seconds |
|---:|---:|---:|---:|
| 17 | 3180 | 2.685 | 3.097 |
| 29 | 3380 | 2.737 | 3.129 |
| 43 | 3100 | 2.471 | 2.866 |

The complete run used 9,660 candidate coordinate LPs and 9.112 seconds for search, final fits and certificates, in one process with single-thread BLAS/HiGHS configuration. This is a local observed runtime, not a general complexity claim. All three newly refitted contrast12 coefficient arrays exactly match their original frozen dyadic arrays, so its displayed predictions reproduce the original contrast results.

`verify.py` checks input/freeze integrity, exhaustive candidate identities, the declared selection tie rule, 120 exact discovery output brackets and 36 exact test maxima without invoking an optimiser. The precise brackets, selected predicates, all candidate objectives, coefficients, exact responses and runtime are retained in `discovery_results.json`, `seed-*`, and `greedy-*.json`. `verification.json` records the replay outcome.

These controls close the missing matched-comparator gap descriptively. They do not turn the same small linear-readout study into evidence about semantic explanation quality, foundation models, prevalence or independent generalisation. The exact sparse witnesses still serve their rejection purpose. The present measurements do not justify marketing witness-contrast selection as the best four-predicate repair.

Reproduce in a new empty directory with `run.py run`, followed by `run.py evaluate`, keeping its relative `../review-round1` input location or adjusting that constant. These stages refuse to overwrite their contract/freeze/results. Run `verify.py` for solver-free replay and `summarise.py` to regenerate this report and the CSV. The interpreter/environment and input hashes are retained in the exploratory contract.
