# Published-report collision sensitivity

Assessment origins are placed in one deterministic SHA-256 order before nested prefixes are formed. These are post hoc occupancy diagnostics, not confidence intervals or population estimates.

## Nested bank size

| Origins | Non-singleton groups | Conflicting groups | Published minimum errors | Logical errors after truth refinement | Report-specific errors removed | Report-specific rate over all rows |
|--:|--:|--:|--:|--:|--:|--:|
| 128 | 468 | 35 | 35 | 32 | 3 | 0.0060% |
| 256 | 2,340 | 224 | 224 | 169 | 55 | 0.0551% |
| 512 | 6,435 | 591 | 591 | 419 | 172 | 0.0861% |
| 1,024 | 23,244 | 2,474 | 2,484 | 1,484 | 1,000 | 0.2504% |

## Full assessment bank by active published bits

| Active bits | Origins | Non-singleton groups | Conflicting groups | Published errors | Logical errors | Report-specific errors | Report-specific rate |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 18 | 1,053 | 291 | 292 | 20 | 272 | 3.8746% |
| 1 | 263 | 12,909 | 1,478 | 1,485 | 1,041 | 444 | 0.4329% |
| 2 | 285 | 7,293 | 598 | 600 | 373 | 227 | 0.2042% |
| 3 | 195 | 1,482 | 92 | 92 | 36 | 56 | 0.0736% |
| 4 | 122 | 351 | 11 | 11 | 10 | 1 | 0.0021% |
| 5 | 70 | 156 | 4 | 4 | 4 | 0 | 0.0000% |
| 6 | 45 | 0 | 0 | 0 | 0 | 0 | 0.0000% |
| 7 | 15 | 0 | 0 | 0 | 0 | 0 | 0.0000% |
| 8 | 9 | 0 | 0 | 0 | 0 | 0 | 0.0000% |
| 9 | 1 | 0 | 0 | 0 | 0 | 0 | 0.0000% |
| 10 | 1 | 0 | 0 | 0 | 0 | 0 | 0.0000% |

Errors remaining after truth-table refinement are properties of the edit target under the declared context. Errors removed by refinement measure information absent from the published bits on this finite bank.
