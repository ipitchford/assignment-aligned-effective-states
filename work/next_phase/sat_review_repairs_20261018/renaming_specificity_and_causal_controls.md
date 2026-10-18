# Renaming specificity and direct-logit controls

## Specificity by semantic input class

| Pair / direction | UNSAT TN/denom | SAT different-sole TN/denom | Multi-solution excluding-label TN/denom |
|:--|--:|--:|--:|
| [261, 291] original | 192/192 | 32/32 | 0/0 |
| [261, 291] renamed | 192/192 | 32/32 | 0/0 |
| [411, 492] original | 192/192 | 32/32 | 0/0 |
| [411, 492] renamed | 192/192 | 32/32 | 0/0 |

## Activation-change-matched restoration and patch-in

| Direction | Source | Native edited correct | Qualified controls | Restoration C→W, source / qualified values | Patch-in UNSAT→SAT, source / qualified values |
|:--|--:|--:|:--|:--|:--|
| [261, 291] original | 261 | 32/32 | [374] | 29 / [1] | 29 / [4] |
| [261, 291] renamed | 291 | 32/32 | none | 32 / n/a | 32 / n/a |
| [411, 492] original | 411 | 30/32 | [201] | 22 / [10] | 18 / [6] |
| [411, 492] renamed | 492 | 32/32 | none | 32 / n/a | 32 / n/a |

Controls are selected using only the sorted 32-case absolute activation-change profile. A qualified match has relative Euclidean profile distance at most 0.25. Only unit 374 qualifies for unit 261 and only unit 201 qualifies for unit 411; neither renamed direction has a qualified control. The wider 32-nearest sets remain in the JSON as references and are not described as matched. Effects are then evaluated through the saved exact linear readout. These post hoc interventions may be off-manifold.

## Unit 411 and unit 201

Unit 411 misses the 0.5 threshold on 8/32 positive edited endpoints. Unit 201 is active on 4/8 misses and 17/24 hits; restoring unit 201 changes 6 missed-threshold decisions and 4 hit decisions from correct to wrong. Its published rule contains TTFFT and excludes TFTFF. This is a compensation diagnostic, not a unique-mechanism result.
