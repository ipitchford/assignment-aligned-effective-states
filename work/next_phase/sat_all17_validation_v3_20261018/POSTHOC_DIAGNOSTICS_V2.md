# Extended post hoc diagnostics for the all-17 SAT study

These analyses fit no neural model and make no new checkpoint forward. They cannot alter the protected decisions.

## Candidate-label competition

The published assignment ranks first among all 32 assignment labels for 17/17 units on the 7,252-input selected bank.

| Unit | Published rank / 32 | Published AUROC | Best alternative margin | Rank among 17 saved units for that label |
|---:|---:|---:|---:|---:|
| 10 | 1 | 0.996321 | 0.105977 | 1 |
| 41 | 1 | 0.997153 | 0.100937 | 1 |
| 55 | 1 | 0.989441 | 0.051785 | 2 |
| 77 | 1 | 0.997283 | 0.123025 | 1 |
| 81 | 1 | 0.983424 | 0.027367 | 1 |
| 185 | 1 | 1.000000 | 0.195250 | 1 |
| 224 | 1 | 0.999989 | 0.215355 | 1 |
| 245 | 1 | 0.997252 | 0.114931 | 1 |
| 261 | 1 | 0.996526 | 0.112149 | 1 |
| 291 | 1 | 0.998162 | 0.158496 | 1 |
| 317 | 1 | 0.999999 | 0.165269 | 1 |
| 326 | 1 | 0.998152 | 0.129270 | 1 |
| 380 | 1 | 0.999964 | 0.189638 | 1 |
| 411 | 1 | 0.988546 | 0.031502 | 1 |
| 416 | 1 | 0.990173 | 0.035171 | 1 |
| 492 | 1 | 0.996489 | 0.093949 | 1 |
| 495 | 1 | 0.997789 | 0.161538 | 1 |

## Neighbour multiplicity

Near negatives with at least two Hamming-one solutions: 9/24 false positives; every row has one distinct violated clause type.
Near negatives with exactly one Hamming-one solution: 27/520 false positives.

## Orbit-grouped five-fold regression

| Model | CV R² | CV RMSE |
|:--|--:|--:|
| unit means | 0.037884 | 1.319436 |
| unit means plus q categorical | 0.654772 | 0.790364 |
| unit means plus distance categorical | 0.347574 | 1.086526 |
| unit means plus distinct type categorical | 0.593303 | 0.857848 |
| unit means plus q and distance categorical | 0.688212 | 0.751112 |
| unit specific linear q | 0.721025 | 0.710488 |

## Gate uncertainty

Exact 95% interval for 4/32: 0.0351--0.2899.
At a true rate of 0.10, the probability of failing the <=3/32 guard is 0.400; at 0.05 it is 0.074.

## Transformed selection

For every labelled assignment-transposition edge and each of five base strata, retain the first eight frozen hash-ordered base cases whose transformed token differs and is absent from the historical ledger.
