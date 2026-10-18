# Exact finite-bank SAT information audit

Every legal edit is grouped by the complete declared lookup input: report, slot, old clause and new clause. A conflicting group contains both SAT and UNSAT responses. Its unrestricted worst-case absolute error is therefore 0.5. The empirical zero-one column instead counts the minority outcome in each group.

## Entire Development Bank

| Report | Groups | Non-singleton groups | Conflicting / non-singleton | Rows in conflicts / non-singleton rows | Minimum empirical error |
|:--|--:|--:|--:|--:|--:|
| context_only | 15,600 | 15,600 | 15,598/15,600 (99.987%) | 1,597,233/1,597,440 (99.987%) | 10.446% |
| published_bits34 | 1,291,758 | 219,180 | 23,831/219,180 (10.873%) | 63,114/524,862 (12.025%) | 1.577% |
| truth_table32 | 1,257,867 | 238,797 | 21,170/238,797 (8.865%) | 56,879/578,370 (9.834%) | 1.409% |
| continuous_post34_exact_float32 | 1,597,440 | 0 | 0/0 (0.000%) | 0/0 (0.000%) | 0.000% |

Published-bit conflict decomposition:

| Quantity | Count |
|:--|--:|
| Conflicting published-bit/action groups with a within-truth-table conflict | 15,603 |
| Conflicting published-bit/action groups only across different truth tables | 8,228 |
| Minimum errors remaining after truth-table refinement | 16,328 (64.825% of published-bit minimum errors) |
| Minimum errors removed by truth-table refinement | 8,860 (35.175% of published-bit minimum errors) |

## Assessment Origins Only

| Report | Groups | Non-singleton groups | Conflicting / non-singleton | Rows in conflicts / non-singleton rows | Minimum empirical error |
|:--|--:|--:|--:|--:|--:|
| context_only | 15,600 | 15,600 | 14,375/15,600 (92.147%) | 370,803/399,360 (92.849%) | 10.553% |
| published_bits34 | 373,893 | 23,244 | 2,474/23,244 (10.644%) | 5,375/48,711 (11.034%) | 0.622% |
| truth_table32 | 371,280 | 25,116 | 2,077/25,116 (8.270%) | 4,543/53,196 (8.540%) | 0.526% |
| continuous_post34_exact_float32 | 399,360 | 0 | 0/0 (0.000%) | 0/0 (0.000%) | 0.000% |

Published-bit conflict decomposition:

| Quantity | Count |
|:--|--:|
| Conflicting published-bit/action groups with a within-truth-table conflict | 1,482 |
| Conflicting published-bit/action groups only across different truth tables | 992 |
| Minimum errors remaining after truth-table refinement | 1,484 (59.742% of published-bit minimum errors) |
| Minimum errors removed by truth-table refinement | 1,000 (40.258% of published-bit minimum errors) |

Exact continuous postactivation is collision-free on this finite bank because all 4,096 stored origin reports are distinct. This permits lookup memorisation; it is not evidence for a compact decoder or transfer.

The archived context, published-bit and truth-table counts are reproduced exactly before this report is accepted. No model is fitted and no transformer forward is made.
