# SAT renaming: bounded source-fidelity result

Status: **CRITERION_NOT_MET**.

Equal-weight mixture of the 32 selected descriptors per pair under the frozen selection algorithm; one fixed checkpoint.
The benchmark is an author-chosen finite-cohort criterion; it is not a guarantee from the source paper. The two fixed pairs are not a sample of a unit population.

| Pair | Primary sensitivity | Primary specificity | Exact positive pattern | Off-pair false-positive rate | Identity advantage | Wrong-map advantage |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 0.984375 | 1.000000 | 0.968750 | 0.000000 | 0.484375 | 0.484375 |
| P2 | 0.875000 | 1.000000 | 0.750000 | 0.000000 | 0.375000 | 0.375000 |

All 512 task rows and 2,048 four-unit event rows remain in the analysis. Task performance and numerical sensitivity are reported separately in analysis.json and input_unit_outcomes.csv.

Pooled bootstrap intervals are descriptive conditional intervals, not two separately supported interval claims. Primary bit patterns, transport errors and mapping advantages reuse the same events.

A failed criterion limits this selected extension; it does not refute the source's distribution-dependent interpretation. A positive result remains compatible with assignment-selective scalar margins or confidence-plus-syntax models.

Native cost: 512 initial and 16 replay input forwards in 66 eight-row batch calls; zero fits. Bootstrap: exactly 2,000 saved block draws, with the same indices used for nominal and numerical-bound statistics.

Secondary affine accounting: PASS_SECONDARY_ACCOUNTING_ONLY. It is a float64 downstream clamp calculation from saved native arrays, not a patched-native experiment or an upstream mechanism discovery.
