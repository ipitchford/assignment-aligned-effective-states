# Post hoc diagnostics for the all-17 SAT study

These diagnostics use only the retained bank and native outputs. They do not alter the protected decisions.

## Competing graded-score summaries

| Descriptive OLS model | Rows | R² |
|:--|--:|--:|
| all rows unit fixed effects plus violated clauses | 2720 | 0.831529 |
| sat rows unit fixed effects plus violated clauses | 2176 | 0.824792 |
| sat rows unit fixed effects plus nearest solution hamming | 2176 | 0.584691 |
| sat rows unit fixed effects plus both | 2176 | 0.827810 |

The regressions include unit fixed effects. They are descriptive fits on an engineered, dependent bank; they do not identify the represented variable.

## Ranking-loss locality

Near negatives account for 935/970 (96.4%) strict positive-negative inversions.

## Threshold-label error versus task error

| Event cell | Count |
|:--|--:|
| Both correct | 2580 |
| Label error, task correct | 130 |
| Label correct, task error | 9 |
| Both error | 1 |

Counts are unit-formula events and therefore inherit shared-formula dependence.

## Near-negative false positives by flipped variable

| Variable | FP / nearest-membership rows | FP / exclusive-nearest rows |
|:--|--:|--:|
| x_0 | 2/112 | 0/100 |
| x_1 | 16/113 | 11/101 |
| x_2 | 10/104 | 8/99 |
| x_3 | 2/116 | 0/108 |
| x_4 | 15/123 | 8/112 |

Membership rows can appear under more than one variable when a formula has several Hamming-one solutions; exclusive rows cannot.

## Task errors by solution class

| Class | Errors / rows |
|:--|--:|
| unsat | 2/1453 |
| sole solution | 35/2900 |
| multi solution | 0/2899 |

## Transformed composition

The JSON records fixed/switch and near/far/UNSAT composition for both edge-weighted and deduplicated unit-input events. The transformed bank is a dependent consistency check under a permutation-symmetric generator.
