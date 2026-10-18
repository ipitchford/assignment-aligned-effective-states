# All-unit SAT label validation

Protected broad-discrimination criterion met: **False**

Separate source-threshold calibration criterion met: **False**

Assignment-macro base AUROC: 0.991394 (solution-count-stratified permutation-orbit interval [0.9882583485097707, 0.9941471425679168])

| Unit | Assignment | Base AUROC | Transformed AUROC | Threshold balanced accuracy |
|---:|:---:|---:|---:|---:|
| 10 | TFFFF | 0.981283 | 0.989950 | 0.955729 |
| 41 | FFFFF | 0.996094 | 0.999349 | 0.979167 |
| 55 | TTTFF | 0.978516 | 0.995589 | 0.872396 |
| 77 | FFTTT | 0.992025 | 0.995483 | 0.973958 |
| 81 | FTTTF | 0.961100 | 0.963469 | 0.783854 |
| 185 | FFFTT | 1.000000 | 1.000000 | 1.000000 |
| 224 | FFFFT | 1.000000 | 1.000000 | 0.992188 |
| 245 | FTFFT | 0.999674 | 0.997975 | 0.986979 |
| 261 | TTFTT | 0.993490 | 0.999512 | 0.890625 |
| 291 | TFTTT | 0.999674 | 0.999471 | 0.986979 |
| 317 | TFFFT | 1.000000 | 1.000000 | 1.000000 |
| 326 | TTTTF | 1.000000 | 0.999919 | 0.994792 |
| 380 | TFFTT | 1.000000 | 1.000000 | 0.992188 |
| 411 | TTFFT | 0.973633 | 0.988467 | 0.872396 |
| 416 | TTTFF | 0.981120 | 0.990062 | 0.861979 |
| 492 | TFTFT | 0.990072 | 0.995646 | 0.955729 |
| 495 | FFFTF | 0.995443 | 0.998372 | 0.971354 |

All threshold numerators, near/far specificity cells, semantic strata, task cells, gates, transformed records, hashes and environment evidence are retained in `analysis.json`.

Even if every discrimination gate passes, the conclusion is restricted to one checkpoint and this balanced conditional mixture. It does not validate the full source threshold unless the separate calibration criterion passes; it does not establish equivariance, specificity outside the declared near/far-balanced cohort, prevalence, causal role, or training-set exclusion.
