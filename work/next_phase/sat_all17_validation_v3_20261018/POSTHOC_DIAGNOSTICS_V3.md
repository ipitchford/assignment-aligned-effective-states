# Version-3 zero-forward diagnostics

These analyses are post hoc and cannot alter the protected decisions.

## Decision taxonomy

the protected broad-discrimination and source-threshold fidelity decisions share the same threshold-specificity failure and are not independent

## Boundary ranking and separability

Mean base near-only AUROC: 0.973145.

| Unit | Near AUROC | TP at source | Near FP | TP at zero near FP | TP at <=1 near FP | Runner-up | Hamming |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 10 | 0.943848 | 63/64 | 7/32 | 22/64 | 25/64 | TFTFF | 1 |
| 41 | 0.989258 | 62/64 | 1/32 | 61/64 | 62/64 | FTFFF | 1 |
| 55 | 0.936523 | 49/64 | 2/32 | 14/64 | 14/64 | TTTFT | 1 |
| 77 | 0.976074 | 64/64 | 5/32 | 35/64 | 54/64 | FTTTT | 1 |
| 81 | 0.889648 | 39/64 | 4/32 | 9/64 | 10/64 | FTTTT | 1 |
| 185 | 1.000000 | 64/64 | 0/32 | 64/64 | 64/64 | FFTTT | 1 |
| 224 | 1.000000 | 63/64 | 0/32 | 64/64 | 64/64 | FTFFT | 1 |
| 245 | 0.999023 | 63/64 | 1/32 | 62/64 | 64/64 | FTFFF | 1 |
| 261 | 0.981934 | 50/64 | 0/32 | 58/64 | 59/64 | TTTTT | 1 |
| 291 | 0.999023 | 63/64 | 1/32 | 63/64 | 63/64 | TFTTF | 1 |
| 317 | 1.000000 | 64/64 | 0/32 | 64/64 | 64/64 | TFTFT | 1 |
| 326 | 1.000000 | 64/64 | 1/32 | 64/64 | 64/64 | TTTTT | 1 |
| 380 | 1.000000 | 63/64 | 0/32 | 64/64 | 64/64 | TFTTT | 1 |
| 411 | 0.924805 | 51/64 | 5/32 | 23/64 | 30/64 | TTTFT | 1 |
| 416 | 0.943359 | 49/64 | 4/32 | 28/64 | 38/64 | TFTFF | 1 |
| 492 | 0.970215 | 61/64 | 4/32 | 29/64 | 48/64 | TFTFF | 1 |
| 495 | 0.989746 | 61/64 | 1/32 | 61/64 | 61/64 | FTFTF | 1 |

## Within-q=1 negative comparison

| Regime | Rows | FP | Mean score | Strict inversions |
|:--|--:|--:|--:|--:|
| near two neighbours | 22 | 9 | 0.221273 | 184 |
| near one neighbour | 237 | 25 | -0.212630 | 657 |
| unsat | 122 | 0 | -0.398634 | 22 |
| far | 44 | 0 | -0.768142 | 5 |

## Task-error geometry

{"multi_solution_misses": 48, "multi_solution_misses_task_wrong": 0, "satisfiable_negative_false_positives": 36, "satisfiable_negative_false_positives_task_wrong": 0, "sole_solution_misses": 47, "sole_solution_misses_task_wrong": 1}

## Gate operating characteristics

{"assumptions": "independent homogeneous cells; illustrative only and false for the dependent engineered bank", "near_multi_rate_used": 0.08455882352941177, "probability_all_34_positive_cells_pass_if_each_true_sensitivity_0_8": 0.00146726288785962, "probability_all_34_positive_cells_pass_if_each_true_sensitivity_0_9": 0.8938404678186129, "probability_any_of_17_near_halves_has_at_least_4_fp": 0.5071272891548693, "probability_one_16_case_near_half_has_at_least_4_fp": 0.04076376704324911}

