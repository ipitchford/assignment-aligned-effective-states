# Renaming threshold and origin-restoration audit

This post hoc audit uses the saved protected-study arrays and performs zero neural forwards and zero fits.

| Unit and direction | Source-threshold TP/32 | FP/224 | AUROC | Minimum positive | Maximum negative | TP at zero FP | FP at 32/32 TP |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 261 (original) | 31/32 | 0/224 | 1.000000 | 0.435197 | 0.101130 | 32/32 | 0/224 |
| 291 (renamed) | 32/32 | 0/224 | 1.000000 | 0.688817 | 0.208991 | 32/32 | 0/224 |
| 411 (original) | 24/32 | 0/224 | 0.999302 | -0.004926 | 0.278990 | 31/32 | 5/224 |
| 492 (renamed) | 32/32 | 0/224 | 1.000000 | 0.757909 | 0.443528 | 32/32 | 0/224 |

The 411/492 cross-tab compares source-threshold detection with whether origin restoration changes an initially correct positive endpoint to wrong:

```json
{
  "left_false_right_false": 4,
  "left_false_right_true": 4,
  "left_true_right_false": 6,
  "left_true_right_true": 50
}
```

The intervention restores the nominated pair from each edited endpoint to its corresponding origin values. It is not pure zeroing unless those origin values are both zero. Counts of nonzero origin values and nontrivial control activation changes are retained in the JSON receipt.
