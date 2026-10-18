# Cached continuous-score comparison: development result

Continuous predictor balanced accuracy: **0.892683707059619**; unchanged readiness rule: **0.9095426566623142**. Delta (continuous minus readiness): **-0.016858949602695093**, with descriptive 95% paired origin-cluster interval **[-0.03160217617687335, -0.0035432272098065393]**.

Contract interpretation: **UNRESOLVED_OR_UNSUPPORTED**. Readiness meets its separate two-point development noninferiority criterion: **True**. Descriptive parity at two points: **False**.

The comparison is unresolved at the declared resolution or lacks required support. No additional fits, cases or tuning follow.

| Scope | Continuous BA | Structured BA | Delta | Continuous probability AUROC |
|---|---:|---:|---:|---:|
| primary_assessment | 0.892683707059619 | 0.9095426566623142 | -0.016858949602695093 | 0.9495144495718761 |
| all_selected_assessment | 0.8976256722616471 | 0.9108561166267333 | -0.01323044436508624 | 0.9486535282551278 |
| primary_fit | 0.9907414641226026 | 0.9115401118944446 | 0.07920135222815795 | 0.9993030608539508 |
| all_selected_fit | 0.9689494748034639 | 0.9118665023535455 | 0.057082972449918404 | 0.9857397282727198 |

Primary assessment retains 955 clipped unit-origin pairs out of 1088 selected pairs. Both actions and repeated native input rows remain present. The 17 source units represent 16 assignments, with the two TTTFF units averaged first.

All 2000 saved bootstrap vectors were reused over 685 origin IDs. Valid delta draws: 2000; NA draws: 0. The unchanged structured bootstrap matches exactly. These intervals condition on the fitted predictors and opened cohort; they omit fitting, selection and hypothesis-development uncertainty.

| Unit | Original-fit q labels for readiness | Edited-fit native labels for continuous | Fit status |
|---|---:|---:|---|
| 10 | 2653 | 104 | FITTED_AND_PREDICTED |
| 41 | 2713 | 108 | FITTED_AND_PREDICTED |
| 55 | 2637 | 112 | FITTED_AND_PREDICTED |
| 77 | 2636 | 108 | FITTED_AND_PREDICTED |
| 81 | 2637 | 110 | FITTED_AND_PREDICTED |
| 185 | 2849 | 124 | FITTED_AND_PREDICTED |
| 224 | 2823 | 128 | FITTED_AND_PREDICTED |
| 245 | 2688 | 98 | FITTED_AND_PREDICTED |
| 261 | 2656 | 116 | FITTED_AND_PREDICTED |
| 291 | 2644 | 120 | FITTED_AND_PREDICTED |
| 317 | 2816 | 122 | FITTED_AND_PREDICTED |
| 326 | 2663 | 106 | FITTED_AND_PREDICTED |
| 380 | 2803 | 128 | FITTED_AND_PREDICTED |
| 411 | 2635 | 112 | FITTED_AND_PREDICTED |
| 416 | 2576 | 110 | FITTED_AND_PREDICTED |
| 492 | 2581 | 100 | FITTED_AND_PREDICTED |
| 495 | 2683 | 116 | FITTED_AND_PREDICTED |

Readiness uses 45693 unit-origin semantic q labels; continuous uses 1922 selected unit-action native labels. These are different labels and sample counts, not independent formula counts. A readiness advantage could reflect this asymmetry or finite-sample fitting.

Exactly 17 classifier fits; zero neural forwards and zero new formulas. Fitting took 0.851101 seconds; probability prediction took 0.028409 seconds. Process peak memory: 198705152 bytes. Actual ensembles: 3400 trees, 16778 leaves, 13378 internal split nodes. Full estimator pickles total 2182704 bytes; prediction-array exports total 3773760 compressed bytes (1711070 raw array bytes).

Readiness charges 17 state bits versus 544 float32 score bits across the nominated units. Keeping the original report gives 51 versus 578 state bits, or 67 versus 594 with the same fixed action descriptor. Static thresholds, assignment lookup, clause lookup, all ensemble parameters, label supervision and common native extraction are charged separately. This is not a capacity-matched comparison or a global information minimum.

Fit/prediction warning count: 0; nonstandard fit statuses: []. Raw probabilities, strict >0.5 decisions, fit weights, complete model files, per-unit confusions, q/action disagreement tables and all paired draws are retained.

This post-aggregate development comparison is not blind or protected confirmation. It establishes neither a unique latent mechanism nor global SAT repair, and does not address context invariance, all 34 published units or the earlier collision. The source's weighted-clause-count/threshold account remains the relevant prior art. The earlier post-ReLU magnitude pilot stays unchanged. The original pilot and mechanism manifests were verified before and after this run. Stop: no further learner, neural inference, context permutation or protected data is authorized by this result.
